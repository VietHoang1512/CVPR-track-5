import random

import albumentations
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def get_train_transforms(cfg):
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
            albumentations.Normalize(cfg.MEAN, cfg.STD, max_pixel_value=255.0, always_apply=True),
            # ToTensorV2(p=1.0),
        ]
    )


def get_valid_transforms(cfg):

    return albumentations.Compose(
        [
            albumentations.Normalize(cfg.MEAN, cfg.STD, max_pixel_value=255.0, always_apply=True),
            # ToTensorV2(p=1.0)
        ]
    )


class CVPRDataset(Dataset):
    def __init__(self, tracks, tokenizer, max_len, dim=(224, 224), augmentation=None):
        self.tracks = tracks
        self.track_ids = list(tracks.keys())
        self.tokenizer = tokenizer
        self.dim = dim
        self.max_len = max_len
        self.dim = dim
        self.augmentation = augmentation

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, index):
        sample = self.tracks[self.track_ids[index]]
        # IMAGE

        # TODO: add data path to argument parser
        image_fp = "data/" + sample["frames"][-1]
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

        if self.dim:
            background_image = cv2.resize(background_image, self.dim)
            cropped_image = cv2.resize(cropped_image, self.dim)

        if self.augmentation:
            background_image = self.augmentation(image=background_image)["image"]
            cropped_image = self.augmentation(image=cropped_image)["image"]

        background_image = background_image.astype(np.float32)
        cropped_image = cropped_image.astype(np.float32)
        background_image = background_image.transpose(2, 0, 1)
        cropped_image = cropped_image.transpose(2, 0, 1)

        # TEXT
        pos_text = random.choice(sample["nl"])
        rand = 0
        while index == rand:
            rand = random.randint(0, len(self.tracks) - 1)

        neg_text = random.choice(self.tracks[self.track_ids[rand]]["nl"])

        pos_inputs = self.tokenizer.encode_plus(
            pos_text,
            # add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True,
        )
        pos_input_ids = pos_inputs["input_ids"]
        pos_attention_mask = pos_inputs["attention_mask"]
        pos_token_type_ids = pos_inputs["token_type_ids"]

        neg_inputs = self.tokenizer.encode_plus(
            neg_text,
            # add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True,
        )
        neg_input_ids = neg_inputs["input_ids"]
        neg_attention_mask = neg_inputs["attention_mask"]
        neg_token_type_ids = neg_inputs["token_type_ids"]

        item = {}
        item["pos_input_ids"] = torch.tensor(pos_input_ids, dtype=torch.long)
        item["pos_attention_mask"] = torch.tensor(pos_attention_mask, dtype=torch.long)
        item["pos_token_type_ids"] = torch.tensor(pos_token_type_ids, dtype=torch.long)

        item["neg_input_ids"] = torch.tensor(neg_input_ids, dtype=torch.long)
        item["neg_attention_mask"] = torch.tensor(neg_attention_mask, dtype=torch.long)
        item["neg_token_type_ids"] = torch.tensor(neg_token_type_ids, dtype=torch.long)

        item["background_image"] = torch.tensor(background_image, dtype=torch.float)
        item["cropped_image"] = torch.tensor(cropped_image, dtype=torch.float)

        return item
