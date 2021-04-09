import os
import random

import albumentations
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


def seed_everything(seed: int) -> None:
    """
    Seed for reproceducing

    Args:
        seed (int): seed number
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    def __init__(self, patience=7, mode="max", delta=0.001):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch_score, model, model_path):

        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print("EarlyStopping counter: {} out of {}".format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            print(
                "Validation score improved ({} --> {}). Saving model to {}!".format(
                    self.val_score, epoch_score, model_path
                )
            )
            torch.save(model.state_dict(), model_path)
        self.val_score = epoch_score


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad is True)


def get_train_transforms():
    return albumentations.Compose(
        [
            # albumentations.HorizontalFlip(p=0.5),
            # albumentations.VerticalFlip(p=0.5),
            # albumentations.Rotate(limit=120, p=0.8),
            # albumentations.RandomBrightness(limit=(0.09, 0.6), p=0.5),
            # albumentations.Cutout(num_holes=8, max_h_size=8, max_w_size=8, fill_value=0, always_apply=False, p=0.5),
            # albumentations.ShiftScaleRotate(
            #   shift_limit=0.25, scale_limit=0.1, rotate_limit=0
            # ),
            albumentations.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            # ToTensorV2(p=1.0),
        ]
    )


def get_valid_transforms():

    return albumentations.Compose(
        [
            albumentations.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            # ToTensorV2(p=1.0)
        ]
    )


def official_evaluate(model, image_dir, tracks, tokenizer, max_len, image_size, augmentation, device):
    model.eval()
    embedding_queries = {}
    embedding_tracks = {}
    track_ids = list(tracks.keys())
    with torch.no_grad():
        for query_id in tqdm(track_ids, desc="Getting embeddings"):
            sample = tracks[query_id]

            image_fp = os.path.join(image_dir, sample["frames"][-1])
            cropped_path = image_fp.replace(".jpg", "_croped.jpg")

            background_image = cv2.imread(image_fp)
            cropped_image = cv2.imread(cropped_path)

            background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

            center_boxes = [(box[0], box[1]) for box in sample["boxes"]]
            isClosed = False
            color = (0, 255, 0)
            thickness = 2
            background_image = cv2.polylines(
                np.array(background_image), [np.array(center_boxes)], isClosed, color, thickness
            )

            if image_size is not None:
                background_image = cv2.resize(background_image, (image_size, image_size))
                cropped_image = cv2.resize(cropped_image, (image_size, image_size))

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

            embedding_tracks[query_id] = output_image.detach().cpu().numpy()

            text_embeds = []

            for text in sample["nl"]:
                inputs = tokenizer.encode_plus(
                    text,
                    max_length=max_len,
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
    for query_id in tqdm(track_ids, desc="Computing pairwise distances"):
        text_embeds = embedding_queries[query_id]
        result = []
        for track_id in track_ids:
            image_embed = embedding_tracks[track_id]
            d1 = F.pairwise_distance(torch.tensor(image_embed), torch.tensor(text_embeds[0]))
            d2 = F.pairwise_distance(torch.tensor(image_embed), torch.tensor(text_embeds[1]))
            d3 = F.pairwise_distance(torch.tensor(image_embed), torch.tensor(text_embeds[2]))

            d = (d1 + d2 + d3) / 3
            result.append([track_id, d.item()])
        results[query_id] = [k[0] for k in sorted(result, key=lambda tup: tup[1], reverse=False)]

    recall_5 = 0
    recall_10 = 0
    mrr = 0
    for query_id in track_ids:
        result = results[query_id]
        target = query_id
        try:
            rank = result.index(target)
        except ValueError:
            rank = 100
        if rank < 10:
            recall_10 += 1
        if rank < 5:
            recall_5 += 1
        mrr += 1.0 / (rank + 1)
    recall_5 /= len(track_ids)
    recall_10 /= len(track_ids)
    mrr /= len(track_ids)

    print("Recall@5 is %.4f" % recall_5)
    print("Recall@10 is %.4f" % recall_10)
    print("MRR is %.4f" % mrr)

    return mrr, recall_5, recall_10
