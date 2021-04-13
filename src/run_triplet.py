import argparse
import gc
import json
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from tqdm.auto import tqdm
from transformers import AdamW, AutoTokenizer, get_linear_schedule_with_warmup

from src.triplet.datasets import TripletDataset
from src.triplet.engine import train_fn, validation_fn
from src.triplet.models import EfficientModel
from src.utils import constants
from src.utils.params import optimizer_params
from src.utils.signature import print_signature
from src.utils.train_utils import (
    EarlyStopping,
    get_train_transforms,
    get_valid_transforms,
    official_evaluate,
    seed_everything,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("Using Torch version:", torch.__version__)
print("Using Transformers version:", transformers.__version__)


parser = argparse.ArgumentParser(description="CVPR 2021 Challenge Track 5: Natural Language-Based Vehicle Retrieval")

parser.add_argument(
    "--train_json",
    type=str,
    help="path to the train data tracks",
)

parser.add_argument(
    "--validation_json",
    type=str,
    help="path to the validation data tracks",
)

parser.add_argument(
    "--test_tracks",
    type=str,
    help="path to the test tracks",
)

parser.add_argument(
    "--test_queries",
    type=str,
    help="path to the test queries",
)

parser.add_argument(
    "--sample_submission",
    type=str,
    help="path to the sample submission",
)

parser.add_argument(
    "--submission_path",
    type=str,
    help="path to the submission save path",
)

parser.add_argument(
    "--output_dir",
    type=str,
    help="path to directory for models saving",
)

parser.add_argument(
    "--image_dir",
    type=str,
    help="path to the image directory",
)

parser.add_argument(
    "--image_model",
    default="efficientnet-b0",
    type=str,
    help="pretrained efficient net model name ",
)

parser.add_argument(
    "--bert_model",
    default="roberta-base",
    type=str,
    help="path to pretrained bert model path or directory (e.g: https://huggingface.co/models)",
)

parser.add_argument(
    "--image_size",
    default=224,
    type=int,
    help="size of image",
)

parser.add_argument(
    "--num_warmup_steps",
    default=0,
    type=int,
    help="number of warm up step for learning rate scheduler",
)

parser.add_argument(
    "--max_len",
    default=64,
    type=int,
    help="max sequence length for padding and truncation (Bert word tokenizer)",
)

parser.add_argument(
    "--n_hiddens",
    default=4,
    type=int,
    help="concatenate n_hiddens final layer to get sequence's bert embedding",
)

parser.add_argument(
    "--out_features",
    default=256,
    type=int,
    help="embedding dim for both image and text",
)

parser.add_argument(
    "--patience",
    default=256,
    type=int,
    help="patience for early stopping",
)

parser.add_argument(
    "--lr",
    default=3e-5,
    type=float,
    help="learning rate",
)

parser.add_argument(
    "--batch_size",
    default=32,
    type=int,
    help="num examples per batch",
)

parser.add_argument(
    "--n_epochs",
    default=5,
    type=int,
    help="num epochs required for training",
)

parser.add_argument(
    "--num_workers",
    default=2,
    type=int,
    help="num workers for data loader",
)

parser.add_argument(
    "--seed",
    default=1710,
    type=int,
    help="seed for reproceduce",
)

parser.add_argument(
    "--do_train",
    action="store_true",
    default=False,
    help="whether train the pretrained model with provided train data",
)

parser.add_argument(
    "--do_infer",
    action="store_true",
    default=False,
    help="whether predict the provided test data with the trained models from checkpoint directory",
)

parser.add_argument(
    "--ckpt_path",
    type=str,
    help="path to the checkpoint (.bin) model",
)

args = parser.parse_args()

if __name__ == "__main__":

    seed_everything(seed=args.seed)
    print_signature()

    with open(args.train_json, "r") as f:
        my_train = json.load(f)
    with open(args.validation_json, "r") as f:
        my_validation = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = EfficientModel(
        image_model=args.image_model,
        bert_model=args.bert_model,
        n_hiddens=args.n_hiddens,
        out_features=args.out_features,
    )

    model.to(device)

    if args.do_train:

        train_dataset = TripletDataset(
            image_dir=args.image_dir,
            tracks=my_train,
            tokenizer=tokenizer,
            max_len=args.max_len,
            image_size=args.image_size,
            augmentation=get_train_transforms(),
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=args.num_workers,
        )

        validation_dataset = TripletDataset(
            image_dir=args.image_dir,
            tracks=my_validation,
            tokenizer=tokenizer,
            max_len=args.max_len,
            image_size=args.image_size,
            augmentation=get_valid_transforms(),
        )

        validation_loader = torch.utils.data.DataLoader(
            validation_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        optimizer = AdamW(optimizer_params(model), lr=args.lr, correct_bias=False)
        total_steps = len(train_loader) * args.n_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        # optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
        # scheduler = None

        os.makedirs(args.output_dir, exist_ok=True)
        es = EarlyStopping(patience=10, mode="max")

        criterion = nn.TripletMarginWithDistanceLoss(margin=0.5, distance_function=nn.PairwiseDistance())
        criterion.to(device)

        for epoch in range(args.n_epochs):
            gc.collect()
            print("Training on epoch", epoch + 1)

            train_loss = train_fn(
                dataloader=train_loader,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                scheduler=scheduler,
            )

            validation_loss = validation_fn(validation_loader, model, criterion, device)

            mrr, recall_5, recall_10 = official_evaluate(
                model,
                args.image_dir,
                my_validation,
                tokenizer=tokenizer,
                max_len=args.max_len,
                image_size=args.image_size,
                augmentation=get_valid_transforms(),
                device=device,
            )

            model_path = os.path.join(args.output_dir, f"epoch_{epoch}_mrr_{mrr:.4f}.bin")
            es(mrr, model, model_path=model_path)
            if es.early_stop:
                print("Early stopping")
                break
    if args.do_infer:

        model.load_state_dict(torch.load(args.ckpt_path))

        with open(args.test_tracks) as f_r:
            tracks = json.load(f_r)

        with open(args.test_queries) as f_r:
            queries = json.load(f_r)

        augmentation = get_valid_transforms()

        model.eval()
        embedding_queries = {}
        embedding_tracks = {}
        track_ids = list(tracks.keys())
        with torch.no_grad():
            for query_id, track_id in tqdm(
                zip(queries.keys(), tracks.keys()), desc="Getting embeddings", total=len(queries)
            ):
                sample = tracks[track_id]

                image_fp = os.path.join(args.image_dir, sample["frames"][-1])
                cropped_path = image_fp.replace(".jpg", "_croped.jpg")

                background_image = cv2.imread(image_fp)
                cropped_image = cv2.imread(cropped_path)

                background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)
                cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

                center_boxes = [(box[0], box[1]) for box in sample["boxes"]]
                background_image = cv2.polylines(
                    np.array(background_image),
                    [np.array(center_boxes)],
                    constants.isClosed,
                    constants.color,
                    constants.thickness,
                )

                if args.image_size is not None:
                    background_image = cv2.resize(background_image, (args.image_size, args.image_size))
                    cropped_image = cv2.resize(cropped_image, (args.image_size, args.image_size))

                if augmentation is not None:
                    background_image = augmentation(image=background_image)["image"]
                    cropped_image = augmentation(image=cropped_image)["image"]

                background_image = background_image.astype(np.float32)
                cropped_image = cropped_image.astype(np.float32)
                background_image = background_image.transpose(2, 0, 1)
                cropped_image = cropped_image.transpose(2, 0, 1)
                background_image = torch.tensor(background_image, dtype=torch.float32).unsqueeze(0).to(device)
                cropped_image = torch.tensor(cropped_image, dtype=torch.float32).unsqueeze(0).to(device)
                output_image = model.forward_image(background_image=background_image, cropped_image=cropped_image)

                embedding_tracks[track_id] = output_image.detach().cpu().numpy()

                text_embeds = []

                for text in queries[query_id]:
                    inputs = tokenizer.encode_plus(
                        text,
                        max_length=args.max_len,
                        padding="max_length",
                        return_token_type_ids=True,
                        truncation=True,
                    )
                    input_ids = inputs["input_ids"]
                    attention_mask = inputs["attention_mask"]
                    token_type_ids = inputs["token_type_ids"]

                    input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)

                    attention_mask = torch.tensor(attention_mask).unsqueeze(0).to(device)
                    token_type_ids = torch.tensor(token_type_ids).unsqueeze(0).to(device)

                    (output_text,) = model.forward_text(
                        input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
                    )
                    text_embeds.append(output_text.detach().cpu().numpy())

                embedding_queries[query_id] = text_embeds

        results = {}
        for query_id in tqdm(queries.keys(), desc="Computing pairwise distances"):
            text_embeds = embedding_queries[query_id]
            result = []
            for track_id in list(tracks.keys()):
                image_embed = embedding_tracks[track_id]
                d1 = F.pairwise_distance(torch.tensor(image_embed), torch.tensor(text_embeds[0]))
                d2 = F.pairwise_distance(torch.tensor(image_embed), torch.tensor(text_embeds[1]))
                d3 = F.pairwise_distance(torch.tensor(image_embed), torch.tensor(text_embeds[2]))

                d = (d1 + d2 + d3) / 3
                result.append([track_id, d.item()])
            results[query_id] = [k[0] for k in sorted(result, key=lambda tup: tup[1], reverse=False)]

        with open(args.sample_submission) as f_r:
            queries = json.load(f_r)

        submission = {}
        for query in queries.keys():
            submission[query] = results[query]
        with open(args.submission_path, "w") as f:
            json.dump(submission, f)
