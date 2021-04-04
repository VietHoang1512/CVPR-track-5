import json
import os

import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import AutoTokenizer

from src.configs.configuration import CFG
from src.losses.constrastive import ContrastiveLoss
from src.trainer.datasets import CVPRDataset, get_train_transforms
from src.trainer.models import CVPRModel
from src.trainer.train import train_fn

# from transformers import AdamW
# from transformers import get_linear_schedule_with_warmup, get_constant_schedule

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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
    train_dataset, batch_size=CFG.BATCH_SIZE, drop_last=True, shuffle=True, pin_memory=True, num_workers=2
)

device = torch.device(CFG.DEVICE)
model = CVPRModel(image_model="efficientnet-b2", bert_model="roberta-base", n_hiddens=2, out_features=10)
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
"""
optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5, correct_bias=False)

total_steps = len(train_loader) * CFG.EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer,
          num_warmup_steps = 200,
          num_training_steps = total_steps)
"""

best_loss = 10000
for epoch in range(CFG.EPOCHS):
    train_loss = train_fn(train_loader, model, criterion, optimizer, device, scheduler=None, epoch=epoch)

    if train_loss.avg < best_loss:
        best_loss = train_loss.avg
        torch.save(model.state_dict(), "model_best_loss.bin")
