import json
import os

import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import AdamW, AutoTokenizer, get_linear_schedule_with_warmup

from src.configs.configuration import CFG
from src.constrastive.datasets import CVPRDataset
from src.constrastive.engine import train_fn
from src.constrastive.loss import ContrastiveLoss
from src.constrastive.models import CVPRModel
from src.utils.train_utils import (
    EarlyStopping,
    get_train_transforms,
    get_valid_transforms,
    official_evaluate,
    seed_everything,
)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

seed_everything(CFG.SEED)

with open("data/data/my_train.json", "r") as f:
    my_train = json.load(f)
with open("data/data/my_validation.json", "r") as f:
    my_validation = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(CFG.BERT_MODEL)
train_dataset = CVPRDataset(
    tracks=my_train,
    tokenizer=tokenizer,
    max_len=CFG.MAX_LEN,
    dim=CFG.DIM,
    augmentation=get_train_transforms(CFG),
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=CFG.BATCH_SIZE,
    # drop_last=True,
    shuffle=True,
    pin_memory=True,
    num_workers=CFG.NUM_WORKERS,
)

validation_dataset = CVPRDataset(
    tracks=my_validation,
    tokenizer=tokenizer,
    max_len=CFG.MAX_LEN,
    dim=CFG.DIM,
    augmentation=get_valid_transforms(CFG),
)

validation_loader = torch.utils.data.DataLoader(
    validation_dataset, batch_size=CFG.BATCH_SIZE, num_workers=CFG.NUM_WORKERS
)

device = torch.device(CFG.DEVICE)
model = CVPRModel(
    image_model=CFG.MODEL_NAME, bert_model=CFG.BERT_MODEL, n_hiddens=CFG.N_HIDDENS, out_features=CFG.OUT_FEATURES
)
model.to(device)

criterion = ContrastiveLoss()
criterion.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=CFG.LR)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=CFG.T_0)


param_optimizer = list(model.named_parameters())
no_decay = ["LayerNorm.bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {"params": [p for n, p in param_optimizer if (not any(nd in n for nd in no_decay))], "weight_decay": 0.01},
    {"params": [p for n, p in param_optimizer if (any(nd in n for nd in no_decay))], "weight_decay": 0.0},
]

optimizer = AdamW(optimizer_grouped_parameters, lr=CFG.LR, correct_bias=False)

total_steps = len(train_loader) * CFG.EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=200, num_training_steps=total_steps)


output_dir = os.path.join(CFG.OUTPUT_DIR, "constrastive")
os.makedirs(output_dir, exist_ok=True)
es = EarlyStopping(patience=2, mode="max")
# model.load_state_dict(torch.load('outputs/constrastive/epoch_5_loss_1.2122.bin'))

for epoch in range(CFG.EPOCHS):
    print("Training on epoch", epoch + 1)
    train_loss = train_fn(train_loader, model, criterion, optimizer, device, scheduler=scheduler)
    # validation_loss = validation_fn(validation_loader, model, criterion, device)
    mrr, recall_5, recall_10 = official_evaluate(
        model,
        my_validation,
        tokenizer=tokenizer,
        max_len=CFG.MAX_LEN,
        dim=CFG.DIM,
        augmentation=get_valid_transforms(CFG),
        device=CFG.DEVICE,
    )
    model_path = os.path.join(output_dir, f"epoch_{epoch}_loss_{mrr:.4f}.bin")
    es(mrr, model, model_path=model_path)
    if es.early_stop:
        print("Early stopping")
        break