import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import constants


class ContrastiveDataset(Dataset):
    def __init__(self, image_dir, tracks, tokenizer, max_len, dim=(224, 224), augmentation=None):
        self.image_dir = image_dir
        self.tracks = tracks
        self.track_ids = list(tracks.keys())
        self.tokenizer = tokenizer
        self.dim = dim
        self.max_len = max_len
        self.dim = dim
        self.augmentation = augmentation

    def __len__(self):
        return 3 * len(self.tracks)

    def __getitem__(self, index):
        sample = self.tracks[self.track_ids[index // 3]]
        # TODO: add data path to argument parser
        image_fp = os.path.join(self.image_dir, sample["frames"][-1])
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

        if random.uniform(0, 1) > 0.5:
            label = 1
            text = sample["nl"][index % 3]
        else:
            label = 0
            text = random.choice(sample["neg_nl"])

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
        inputs = self.tokenizer.encode_plus(
            text,
            # add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True,
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        item = {}
        item["input_ids"] = torch.tensor(input_ids, dtype=torch.long)
        item["attention_mask"] = torch.tensor(attention_mask, dtype=torch.long)
        item["token_type_ids"] = torch.tensor(token_type_ids, dtype=torch.long)
        item["background_image"] = torch.tensor(background_image, dtype=torch.float)
        item["cropped_image"] = torch.tensor(cropped_image, dtype=torch.float)
        item["label"] = torch.tensor(label, dtype=torch.float)
        return item
