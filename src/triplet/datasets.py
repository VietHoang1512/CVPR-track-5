import os
import random

import clip
import cv2
import numpy as np
import PIL
import torch
from torch.utils.data import Dataset
from utils import constants


class TripletDataset(Dataset):
    def __init__(self, image_dir, tracks, tokenizer, max_len, image_size, augmentation=None):
        self.image_dir = image_dir
        self.tracks = tracks
        self.track_ids = list(tracks.keys())
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.max_len = max_len
        self.augmentation = augmentation

    def __len__(self):
        return len(self.tracks) * 3

    def __getitem__(self, index):
        sample = self.tracks[self.track_ids[index // 3]]
        # IMAGE

        # TODO: add data path to argument parser
        image_fp = os.path.join(self.image_dir, sample["frames"][-1])
        assert os.path.isfile(image_fp), f"File {image_fp} not found"

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

        if self.image_size:
            background_image = cv2.resize(background_image, (self.image_size, self.image_size))
            cropped_image = cv2.resize(cropped_image, (self.image_size, self.image_size))

        if self.augmentation:
            background_image = self.augmentation(image=background_image)["image"]
            cropped_image = self.augmentation(image=cropped_image)["image"]

        background_image = background_image.astype(np.float32)
        cropped_image = cropped_image.astype(np.float32)
        background_image = background_image.transpose(2, 0, 1)
        cropped_image = cropped_image.transpose(2, 0, 1)

        # TEXT
        pos_text = sample["nl"][index % 3]
        neg_text = random.choice(sample["neg_nl"])

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


class CLIPDataset(Dataset):
    def __init__(self, image_dir, tracks, max_len, augmentation=None):
        self.image_dir = image_dir
        self.tracks = tracks
        self.track_ids = list(tracks.keys())
        self.max_len = max_len
        self.augmentation = augmentation

    def __len__(self):
        return len(self.tracks) * 3

    def __getitem__(self, index):
        sample = self.tracks[self.track_ids[index // 3]]
        # IMAGE

        # TODO: add data path to argument parser
        image_fp = os.path.join(self.image_dir, sample["frames"][-1])
        assert os.path.isfile(image_fp), f"File {image_fp} not found"

        cropped_path = image_fp.replace(".jpg", "_croped.jpg")

        background_image = PIL.Image.open(image_fp)
        cropped_image = PIL.Image.open(cropped_path)

        center_boxes = [(box[0], box[1]) for box in sample["boxes"]]
        background_image = cv2.polylines(
            np.array(background_image),
            [np.array(center_boxes)],
            constants.isClosed,
            constants.color,
            constants.thickness,
        )
        background_image = PIL.Image.fromarray(background_image.astype(np.uint8))

        if self.augmentation:
            background_image = self.augmentation(background_image)
            cropped_image = self.augmentation(cropped_image)

        # TEXT
        pos_text = sample["nl"][index % 3]
        neg_text = random.choice(sample["neg_nl"])
        pos_inputs, neg_inputs = clip.tokenize([pos_text, neg_text], context_length=self.max_len)

        #         print(type(pos_inputs), type(neg_inputs), type(background_image), type(cropped_image))
        #         print(pos_inputs, neg_inputs, background_image, cropped_image)

        item = {}
        item["pos_inputs"] = pos_inputs
        item["neg_inputs"] = neg_inputs

        item["background_image"] = background_image.float()
        item["cropped_image"] = cropped_image.float()

        return item
