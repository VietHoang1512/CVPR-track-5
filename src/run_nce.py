import argparse
import json
import os

import torch
import transformers
from torch.utils.data import DataLoader
from transformers import AdamW

from src.nce.datasets import CityFlowNLDataset, CityFlowNLDatasetInference
from src.nce.learner import Learner
from src.nce.models import NCEModel
from src.utils.signature import print_signature
from src.utils.train_utils import seed_everything

os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("Using Torch version:", torch.__version__)
print("Using Transformers version:", transformers.__version__)

parser = argparse.ArgumentParser(description="CVPR 2021 Challenge Track 5: Natural Language-Based Vehicle Retrieval")

parser.add_argument(
    "--train_json",
    default="data/data/my_train.json",
    type=str,
    help="path to the train data tracks",
)

parser.add_argument(
    "--validation_json",
    default="data/data/my_validation.json",
    type=str,
    help="path to the validation data tracks",
)

parser.add_argument(
    "--test_tracks",
    default="data/data/test-tracks_2.json",
    type=str,
    help="path to the test tracks",
)

parser.add_argument(
    "--test_queries",
    default="data/data/test-queries.json",
    type=str,
    help="path to the test queries",
)


parser.add_argument(
    "--submission_path",
    default="outputs/clip/results.json",
    type=str,
    help="path to the submission save path",
)

parser.add_argument(
    "--output_dir",
    default="outputs/clip/",
    type=str,
    help="path to directory for models saving",
)

parser.add_argument(
    "--image_dir",
    default="data/data/",
    type=str,
    help="path to the image directory",
)

parser.add_argument(
    "--clip_model",
    default="ViT-B/32",
    type=str,
    help="pretrained clip model name ",
)

parser.add_argument(
    "--out_features",
    default=256,
    type=int,
    help="embedding dim for both image and text",
)

parser.add_argument(
    "--batch_size",
    default=16,
    type=int,
    help="num examples per batch",
)

parser.add_argument(
    "--accu_grad_step",
    default=2,
    type=int,
    help="num accumulative gradient step",
)


parser.add_argument(
    "--lr",
    default=1e-5,
    type=float,
    help="learning rate",
)

parser.add_argument(
    "--n_epochs",
    default=10,
    type=int,
    help="num epochs required for training",
)

parser.add_argument(
    "--seed",
    default=1710,
    type=int,
    help="seed for reproceduce",
)

args = parser.parse_args()

if __name__ == "__main__":

    seed_everything(seed=args.seed)
    print_signature()

    with open(args.train_json, "r") as f:
        my_train = json.load(f)
    with open(args.validation_json, "r") as f:
        my_validation = json.load(f)
    with open(args.test_tracks) as f:
        test_tracks = json.load(f)
    with open(args.test_queries) as f:
        queries = json.load(f)

    train_data, val_data = [], []

    for k, v in my_train.items():
        v["id"] = k
        train_data.append(v)
    for k, v in my_validation.items():
        v["id"] = k
        val_data.append(v)
    for k, v in test_tracks.items():
        v["id"] = k

    test_data = list(test_tracks.values())

    train_ds = CityFlowNLDataset(cityflow_path=args.image_dir, data=train_data, image_size=args.image_size, k_neg=5)
    val_ds = CityFlowNLDatasetInference(
        cityflow_path=args.image_dir,
        data=val_data,
        image_size=args.image_size,
    )
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False)
    test_ds = CityFlowNLDatasetInference(
        cityflow_path=args.image_dir,
        data=test_data,
        image_size=args.image_size,
    )
    test_dl = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False)

    model = NCEModel(args.clip_model, args.out_features)
    model.cuda()
    learner = Learner(
        epoch=args.n_epochs,
        lr=args.lr,
        accu_grad_step=args.accu_grad_step,
        train_dl=train_dl,
        val_dl=val_dl,
        model=model,
        optim=AdamW,
    )
    learner.train(output_dir=args.output_dir)
    learner.model.load_state_dict(torch.load(os.path.join(args.output_dir, "ckpt.pth")))
    learner.model.eval()
    learner.eval_one_epoch(val_dl)

    with torch.no_grad():
        submits = learner.match_track2querry(queries, test_dl)

    with open(args.submission_path, "w") as f:
        json.dump(submits, f)
