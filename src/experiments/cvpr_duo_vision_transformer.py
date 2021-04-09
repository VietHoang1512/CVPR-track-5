"""Main configuration."""

import torch


class CFG:
    OUTPUT_DIR = "/content/drive/MyDrive/AI_VN/CVPR/outputs"

    IMAGE_SIZE = 224
    NUM_WORKERS = 2
    BATCH_SIZE = 8
    EPOCHS = 20
    SEED = 1710
    LR = 1e-5
    OUT_FEATURES = 256
    N_HIDDENS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    MAX_LEN = 64
    MODEL_NAME = "efficientnet-b4"
    BERT_MODEL = "roberta-base"
    IMAGE_DIR = "data"


import json


def evaluate_retrieval_results(gt_tracks, results):
    recall_5 = 0
    recall_10 = 0
    mrr = 0
    for query in gt_tracks:
        result = results[query]
        target = gt_tracks[query]
        try:
            rank = result.index(target)
        except ValueError:
            rank = 100
        if rank < 10:
            recall_10 += 1
        if rank < 5:
            recall_5 += 1
        mrr += 1.0 / (rank + 1)
    recall_5 /= len(gt_tracks)
    recall_10 /= len(gt_tracks)
    mrr /= len(gt_tracks)
    print("Recall@5 is %.4f" % recall_5)
    print("Recall@10 is %.4f" % recall_10)
    print("MRR is %.4f" % mrr)
    return recall_5, recall_10, mrr


"""
    Custom logger class (handle both console and file output)
"""

import logging
import logging.handlers
import os
import re


class ColorCodes:
    grey = "\x1b[38;21m"
    green = "\x1b[1;32m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    blue = "\x1b[1;34m"
    light_blue = "\x1b[1;36m"
    purple = "\x1b[1;35m"
    reset = "\x1b[0m"


class ColorizedArgsFormatter(logging.Formatter):
    arg_colors = [ColorCodes.purple, ColorCodes.light_blue]
    level_fields = ["levelname", "levelno"]
    level_to_color = {
        logging.DEBUG: ColorCodes.grey,
        logging.INFO: ColorCodes.green,
        logging.WARNING: ColorCodes.yellow,
        logging.ERROR: ColorCodes.red,
        logging.CRITICAL: ColorCodes.bold_red,
    }

    def __init__(self, fmt: str):
        super().__init__()
        self.level_to_formatter = {}

        def add_color_format(level: int):
            color = ColorizedArgsFormatter.level_to_color[level]
            _format = fmt
            for fld in ColorizedArgsFormatter.level_fields:
                search = "(%\(" + fld + "\).*?s)"
                _format = re.sub(search, f"{color}\\1{ColorCodes.reset}", _format)
            formatter = logging.Formatter(_format)
            self.level_to_formatter[level] = formatter

        add_color_format(logging.DEBUG)
        add_color_format(logging.INFO)
        add_color_format(logging.WARNING)
        add_color_format(logging.ERROR)
        add_color_format(logging.CRITICAL)

    @staticmethod
    def rewrite_record(record: logging.LogRecord):
        if not BraceFormatStyleFormatter.is_brace_format_style(record):
            return

        msg = record.msg
        msg = msg.replace("{", "_{{")
        msg = msg.replace("}", "_}}")
        placeholder_count = 0
        # add ANSI escape code for next alternating color before each formatting parameter
        # and reset color after it.
        while True:
            if "_{{" not in msg:
                break
            color_index = placeholder_count % len(ColorizedArgsFormatter.arg_colors)
            color = ColorizedArgsFormatter.arg_colors[color_index]
            msg = msg.replace("_{{", color + "{", 1)
            msg = msg.replace("_}}", "}" + ColorCodes.reset, 1)
            placeholder_count += 1

        record.msg = msg.format(*record.args)
        record.args = []

    def format(self, record):
        orig_msg = record.msg
        orig_args = record.args
        formatter = self.level_to_formatter.get(record.levelno)
        self.rewrite_record(record)
        formatted = formatter.format(record)

        # restore log record to original state for other handlers
        record.msg = orig_msg
        record.args = orig_args
        return formatted


class BraceFormatStyleFormatter(logging.Formatter):
    def __init__(self, fmt: str):
        super().__init__()
        self.formatter = logging.Formatter(fmt)

    @staticmethod
    def is_brace_format_style(record: logging.LogRecord):
        if len(record.args) == 0:
            return False

        msg = record.msg
        if "%" in msg:
            return False

        count_of_start_param = msg.count("{")
        count_of_end_param = msg.count("}")

        if count_of_start_param != count_of_end_param:
            return False

        if count_of_start_param != len(record.args):
            return False

        return True

    @staticmethod
    def rewrite_record(record: logging.LogRecord):
        if not BraceFormatStyleFormatter.is_brace_format_style(record):
            return

        record.msg = record.msg.format(*record.args)
        record.args = []

    def format(self, record):
        orig_msg = record.msg
        orig_args = record.args
        self.rewrite_record(record)
        formatted = self.formatter.format(record)

        # restore log record to original state for other handlers
        record.msg = orig_msg
        record.args = orig_args
        return formatted


def custom_logger(logging_dir, name="CVPR 2021"):
    path = os.path.join(logging_dir, "log.txt")

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    colored_formatter = ColorizedArgsFormatter(log_format)
    console_handler.setFormatter(colored_formatter)

    file_handler = logging.FileHandler(path, mode="a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format))

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


"""
    My signature
"""

import os


def print_signature():
    os.system("")
    print(
        "\033[36m"
        + r'''
                                        .
                         |              |
                         |             ,|
                ,_       |\            F'   ,.-""`.
               /  `-._   |`           // ,-"_,..  |
              |   ___ `. | \ ,"""`-. /.-' ,'    ' |_....._
              |  /   `-.`.  L......|j j_.'      ` |       `._
              | |      _,'| |______|' | '-._     ||  ,.-.    `.
               L|    ,'   | |      | j      `-.  || '    `.    \
___            | \_,'     | |`"----| |         `.||       |\    \
 ""=+...__     `,'   ,.-.   |....._|   _....     Y \      j_),..+=--
     `"-._"._  .   ,' |  `   \    /  ,' |   \     \ j,..-"_..+-"  L
          `-._-+. j   !   \   `--'  .   !    \  ,.-" _..<._  |    |
              |-. |   |    L        |   !     |  .-/'      `.|-.,-|
              |__ '   '    |        '   |    /    /|   `, -. |   j
        _..--'"__  `-.___,'          `.___,.'  __/_|_  /   / '   |
   _.-_..---""_.-\                            .,...__""--./L/_   |
 -'""'     ,""  ,-`-.    .___.,...___,    _,.+"      """"`-+-==++-
          / /  `.   )"-.._`v \|    V/  /-'    \._,._.'"-. /    /
          ` `.  )---.     `""\\__  / .'        /    \    Y"._.'
           `"'`"     `-.     /|._""_/         |  ,..   _ |  |
                        `"""' |  "'           `. `-'  (_|,-.'
                               \               |`       .`-
                                `.           . j`._    /
                                 |`.._     _.'|    `""/
                                 |    /""'"   |  .". j
                                .`.__j         \ `.' |
                                j    |          `._.'
                               /     |
                              /,  ,  \
                              \|  |   L
                               `..|_..' vh
        ********************************************************
              ********************************************
                    😀  Credit: Hoang Phan Viet  😀
                          ******************
                                ******
    '''
    )


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


"""Baseline Siamese model."""
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel


class CVPRModel(nn.Module):
    def __init__(self, image_model, bert_model, n_hiddens, out_features):
        super().__init__()
        # 3 image models
        self.pooling = nn.AdaptiveAvgPool2d(1)

        self.background_image_model = timm.create_model("vit_base_patch16_224", pretrained=True)
        self.cropped_image_model = timm.create_model("vit_base_patch16_224", pretrained=True)

        n_features = (
            self.background_image_model.head.in_features
            + self.cropped_image_model.head.in_features
            # + self.flow_image_model._fc.in_features
        )

        self.background_image_model.head = nn.Identity()
        self.cropped_image_model.head = nn.Identity()

        self.image_classifier = nn.Linear(n_features, out_features)

        # bert model
        config = AutoConfig.from_pretrained(
            bert_model,
            from_tf=False,
            output_hidden_states=True,
        )
        self.bert_model = AutoModel.from_pretrained(bert_model, config=config)
        self.n_hiddens = n_hiddens
        self.text_classifier = nn.Linear(config.hidden_size * self.n_hiddens, out_features)

    def forward_image(self, background_image, cropped_image):
        background_image.size(0)
        background_output = self.background_image_model(background_image)
        cropped_output = self.cropped_image_model(cropped_image)
        # flow_output = self.flow_image_model(flow_image)
        output = torch.cat(
            [
                background_output,
                cropped_output,
                # flow_output
            ],
            axis=1,
        )
        #         output = self.pooling(output).view(bs, -1)
        output = self.image_classifier(output)
        return output

    def forward_text(self, input_ids, attention_mask, token_type_ids):
        output = self.bert_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        output = torch.mean(torch.cat([output[2][-i] for i in range(self.n_hiddens)], axis=-1), axis=1)
        output = self.text_classifier(output)
        return output

    # TODO: consider a image model for flow only
    def forward(
        self,
        background_image,
        cropped_image,
        pos_input_ids,
        pos_attention_mask,
        pos_token_type_ids,
        neg_input_ids,
        neg_attention_mask,
        neg_token_type_ids,
    ):
        pos_output_text = self.forward_text(pos_input_ids, pos_attention_mask, pos_token_type_ids)
        neg_output_text = self.forward_text(neg_input_ids, neg_attention_mask, neg_token_type_ids)
        output_image = self.forward_image(background_image, cropped_image)

        pos_output_text = F.normalize(pos_output_text, p=2, dim=1)
        neg_output_text = F.normalize(neg_output_text, p=2, dim=1)
        output_image = F.normalize(output_image, p=2, dim=1)

        return output_image, pos_output_text, neg_output_text


import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class CVPRDataset(Dataset):
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
        isClosed = False
        color = (0, 255, 0)
        thickness = 2
        background_image = cv2.polylines(
            np.array(background_image), [np.array(center_boxes)], isClosed, color, thickness
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


import torch
from tqdm.auto import tqdm

# from src.utils.train_utils import AverageMeter


def train_fn(dataloader, model, criterion, optimizer, device, scheduler):
    model.train()

    loss_score = AverageMeter()

    pbar = tqdm(dataloader, total=len(dataloader), desc="Training")
    for data in pbar:
        gc.collect()
        optimizer.zero_grad()

        background_image = data["background_image"].to(device)
        cropped_image = data["cropped_image"].to(device)

        pos_input_ids = data["pos_input_ids"].to(device)
        pos_attention_mask = data["pos_attention_mask"].to(device)
        pos_token_type_ids = data["pos_token_type_ids"].to(device)

        neg_input_ids = data["neg_input_ids"].to(device)
        neg_attention_mask = data["neg_attention_mask"].to(device)
        neg_token_type_ids = data["neg_token_type_ids"].to(device)

        batch_size = background_image.shape[0]

        output_image, pos_output_text, neg_output_text = model(
            background_image=background_image,
            cropped_image=cropped_image,
            pos_input_ids=pos_input_ids,
            pos_attention_mask=pos_attention_mask,
            pos_token_type_ids=pos_token_type_ids,
            neg_input_ids=neg_input_ids,
            neg_attention_mask=neg_attention_mask,
            neg_token_type_ids=neg_token_type_ids,
        )

        loss = criterion(output_image, pos_output_text, neg_output_text)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        # model.zero_grad()

        loss_score.update(loss.detach().item(), batch_size)
        pbar.set_postfix(train_loss=loss_score.avg, lr=optimizer.param_groups[0]["lr"])

    if scheduler is not None:
        scheduler.step()

    return loss_score.avg


def validation_fn(data_loader, model, criterion, device):
    model.eval()
    loss_score = AverageMeter()
    with torch.no_grad():
        pbar = tqdm(data_loader, total=len(data_loader), desc="Evaluating")
        for data in pbar:
            background_image = data["background_image"].to(device)
            cropped_image = data["cropped_image"].to(device)

            pos_input_ids = data["pos_input_ids"].to(device)
            pos_attention_mask = data["pos_attention_mask"].to(device)
            pos_token_type_ids = data["pos_token_type_ids"].to(device)

            neg_input_ids = data["neg_input_ids"].to(device)
            neg_attention_mask = data["neg_attention_mask"].to(device)
            neg_token_type_ids = data["neg_token_type_ids"].to(device)

            batch_size = background_image.shape[0]

            output_image, pos_output_text, neg_output_text = model(
                background_image=background_image,
                cropped_image=cropped_image,
                pos_input_ids=pos_input_ids,
                pos_attention_mask=pos_attention_mask,
                pos_token_type_ids=pos_token_type_ids,
                neg_input_ids=neg_input_ids,
                neg_attention_mask=neg_attention_mask,
                neg_token_type_ids=neg_token_type_ids,
            )

            loss = criterion(output_image, pos_output_text, neg_output_text)

            loss_score.update(loss.detach().item(), batch_size)

            pbar.set_postfix(validation_loss=loss_score.avg)
    return loss_score.avg


import copy
import gc
import json
import os

import torch
import torch.nn as nn
from transformers import AdamW, AutoTokenizer, get_linear_schedule_with_warmup

# from src.configs.configuration import CFG
# from src.triplet.datasets import CVPRDataset
# from src.triplet.engine import train_fn, validation_fn
# from src.triplet.models import CVPRModel
# from src.utils.train_utils import (
#     EarlyStopping,
#     get_train_transforms,
#     get_valid_transforms,
#     official_evaluate,
#     seed_everything,
# )

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

seed_everything(CFG.SEED)

with open("data/data/my_train.json", "r") as f:
    my_train = json.load(f)
with open("data/data/my_validation.json", "r") as f:
    my_validation = json.load(f)

my_train.update(copy.deepcopy(my_validation))
print(len(my_train), len(my_validation))

tokenizer = AutoTokenizer.from_pretrained(CFG.BERT_MODEL)
train_dataset = CVPRDataset(
    image_dir=CFG.IMAGE_DIR,
    tracks=my_train,
    tokenizer=tokenizer,
    max_len=CFG.MAX_LEN,
    image_size=CFG.IMAGE_SIZE,
    augmentation=get_train_transforms(),
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=CFG.BATCH_SIZE,
    # drop_last=True,
    #     shuffle=True,
    pin_memory=True,
    num_workers=CFG.NUM_WORKERS,
)

validation_dataset = CVPRDataset(
    image_dir=CFG.IMAGE_DIR,
    tracks=my_validation,
    tokenizer=tokenizer,
    max_len=CFG.MAX_LEN,
    image_size=CFG.IMAGE_SIZE,
    augmentation=get_valid_transforms(),
)

validation_loader = torch.utils.data.DataLoader(
    validation_dataset,
    batch_size=CFG.BATCH_SIZE,
    num_workers=CFG.NUM_WORKERS,
    #     shuffle=False
)

# validation_dataset[0]

device = torch.device(CFG.DEVICE)

model = CVPRModel(
    image_model=CFG.MODEL_NAME, bert_model=CFG.BERT_MODEL, n_hiddens=CFG.N_HIDDENS, out_features=CFG.OUT_FEATURES
)
model.to(device)

criterion = nn.TripletMarginWithDistanceLoss(margin=0.5, distance_function=nn.PairwiseDistance())
criterion.to(device)

EPOCHS = 10

# optimizer = torch.optim.Adam(model.parameters(), lr=CFG.LR)
# scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=CFG.T_0)

param_optimizer = list(model.named_parameters())
no_decay = ["LayerNorm.bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {"params": [p for n, p in param_optimizer if (not any(nd in n for nd in no_decay))], "weight_decay": 0.01},
    {"params": [p for n, p in param_optimizer if (any(nd in n for nd in no_decay))], "weight_decay": 0.0},
]
optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5, correct_bias=False)

total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
# scheduler = None

output_dir = os.path.join(CFG.OUTPUT_DIR, "triplet")
os.makedirs(output_dir, exist_ok=True)
es = EarlyStopping(patience=10, mode="max")

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
    CFG.IMAGE_DIR,
    my_validation,
    tokenizer=tokenizer,
    max_len=CFG.MAX_LEN,
    image_size=CFG.IMAGE_SIZE,
    augmentation=get_valid_transforms(),
    device=device,
)

for epoch in range(EPOCHS):
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
        CFG.IMAGE_DIR,
        my_validation,
        tokenizer=tokenizer,
        max_len=CFG.MAX_LEN,
        image_size=CFG.IMAGE_SIZE,
        augmentation=get_valid_transforms(),
        device=device,
    )

    #     model_path = os.path.join(output_dir, f"epoch_{epoch}_mrr_{mrr:.4f}.bin")
    #     es(mrr, model, model_path=model_path)
    #     if es.early_stop:
    #         print("Early stopping")
    #         break
    torch.save(model.state_dict(), os.path.join(output_dir, f"epoch_{epoch}.bin"))

with open("data/data/test-tracks_2.json") as f_r:
    tracks = json.load(f_r)
with open("data/data/test-queries.json") as f_r:
    queries = json.load(f_r)

# def official_evaluate(model, image_dir, tracks, tokenizer, max_len, image_size, augmentation, device):
image_dir = CFG.IMAGE_DIR
max_len = CFG.MAX_LEN
image_size = CFG.IMAGE_SIZE
augmentation = get_valid_transforms()

model.eval()
embedding_queries = {}
embedding_tracks = {}
track_ids = list(tracks.keys())
with torch.no_grad():
    for query_id, track_id in tqdm(zip(queries.keys(), tracks.keys()), desc="Getting embeddings"):
        sample = tracks[track_id]

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

        embedding_tracks[track_id] = output_image.detach().cpu().numpy()

        text_embeds = []

        for text in queries[query_id]:
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
