import copy
import os
import random

import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm


class CityFlowNLDataset(Dataset):
    def __init__(self, cityflow_path, data, image_size, k_neg=5, max_rnn_length=256):
        self.max_rnn_length = max_rnn_length
        self.list_of_crops = list()
        self.data = data
        self.k_neg = k_neg
        self.cityflow_path = cityflow_path

        self.transforms = [
            transforms.Resize((image_size, image_size)),
            #  transforms.RandomRotation(degrees= 10),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]

        self.new_data = []
        for track in tqdm(data):
            frame_crop = track["frames"][-1].split(".")[0] + "_croped.jpg"
            frame_crop_path = os.path.join(self.cityflow_path, frame_crop)
            frame_crop = Image.open(frame_crop_path)
            for t in self.transforms:
                frame_crop = t(frame_crop)
            frame_crop_img = frame_crop
            for nl in track["nl"]:
                new = copy.deepcopy(track)
                new["nl"] = nl
                new["frame_crop_img"] = frame_crop_img
                self.new_data.append(new)

    def __len__(self):
        return len(self.new_data)

    def get_frame(self, dp):

        frame_idx = int(random.uniform(0, 1) * len(dp["frames"]))
        frame = dp["frames"][frame_idx]
        frame_path = os.path.join(self.cityflow_path, frame)

        # print(frame_path)
        frame = Image.open(frame_path)
        # ### TODO: adjust boxes

        boxes_points = [(box[0], box[1]) for box in dp["boxes"]]
        image_draw = ImageDraw.Draw(frame)

        for i in range(len(boxes_points) - 1):
            image_draw.line([boxes_points[i], boxes_points[i + 1]], fill=(0, 255, 0), width=5)

        for t in self.transforms:
            frame = t(frame)
        frame_img = frame
        boxes_points = torch.tensor(boxes_points, dtype=torch.float) / 256.0
        if boxes_points.shape[0] >= self.max_rnn_length:
            boxes_points = boxes_points[: self.max_rnn_length]
        else:
            boxes_points = torch.cat(
                [boxes_points, torch.zeros(self.max_rnn_length - boxes_points.shape[0], 2, dtype=torch.float)], dim=0
            )

        return frame_img, boxes_points

    def __getitem__(self, index):
        dp = copy.deepcopy(self.new_data[index])

        frame_img, boxes_points = self.get_frame(dp)
        dp["frame_img"] = frame_img
        dp["boxes_points"] = boxes_points

        neg_tracks = []
        i = 0
        while i < self.k_neg:
            neg_track = copy.deepcopy(random.sample(self.new_data, 1)[0])
            if neg_track["id"] != dp["id"]:
                i += 1
                neg_tracks.append(neg_track)

        dp["neg_frames"] = []
        dp["neg_crops"] = []
        dp["neg_boxes_points"] = []
        dp["neg_nl"] = []

        for neg in neg_tracks:
            neg_frames, neg_boxes_points = self.get_frame(neg)
            dp["neg_frames"].append(neg_frames)
            dp["neg_crops"].append(neg["frame_crop_img"])
            dp["neg_boxes_points"].append(neg_boxes_points)
            dp["neg_nl"].append(neg["nl"])
        dp["neg_frames"] = torch.stack(dp["neg_frames"], dim=0)
        dp["neg_crops"] = torch.stack(dp["neg_crops"], dim=0)
        dp["neg_boxes_points"] = torch.stack(dp["neg_boxes_points"], dim=0)

        del dp["boxes"]
        del dp["frames"]
        return dp


class CityFlowNLDatasetInference(Dataset):
    def __init__(self, cityflow_path, data, image_size, k_neg=5, max_rnn_length=256):
        self.max_rnn_length = max_rnn_length
        self.list_of_crops = list()
        self.data = data
        self.k_neg = k_neg
        self.cityflow_path = cityflow_path
        self.transforms = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]
        self.new_data = []
        for track in tqdm(data):
            frame_crop = track["frames"][-1].split(".")[0] + "_croped.jpg"
            frame_crop_path = os.path.join(self.cityflow_path, frame_crop)
            frame_crop = Image.open(frame_crop_path)
            for t in self.transforms:
                frame_crop = t(frame_crop)
            frame_crop_img = frame_crop
            new = copy.deepcopy(track)
            new["frame_crop_img"] = frame_crop_img
            self.new_data.append(new)

    def __len__(self):
        return len(self.new_data)

    def get_frame(self, dp):

        frame = dp["frames"][-1]
        frame_path = os.path.join(self.cityflow_path, frame)

        frame = Image.open(frame_path)

        boxes_points = [(box[0], box[1]) for box in dp["boxes"]]
        image_draw = ImageDraw.Draw(frame)

        for i in range(len(boxes_points) - 1):
            image_draw.line([boxes_points[i], boxes_points[i + 1]], fill=(0, 255, 0), width=5)

        for t in self.transforms:
            frame = t(frame)
        frame_img = frame

        boxes_points = torch.tensor(boxes_points, dtype=torch.float) / 256.0

        if boxes_points.shape[0] >= self.max_rnn_length:
            boxes_points = boxes_points[: self.max_rnn_length]
        else:
            boxes_points = torch.cat(
                [boxes_points, torch.zeros(self.max_rnn_length - boxes_points.shape[0], 2, dtype=torch.float)], dim=0
            )

        return frame_img, boxes_points

    def __getitem__(self, index):

        dp = copy.deepcopy(self.new_data[index])

        frame_img, boxes_points = self.get_frame(dp)
        dp["frame_img"] = frame_img
        dp["boxes_points"] = boxes_points

        del dp["boxes"]
        del dp["frames"]
        return dp
