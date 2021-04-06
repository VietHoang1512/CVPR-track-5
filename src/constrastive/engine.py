import torch
from tqdm.auto import tqdm

from src.utils.train_utils import AverageMeter


def train_fn(dataloader, model, criterion, optimizer, device, scheduler):
    model.train()

    loss_score = AverageMeter()

    pbar = tqdm(dataloader, total=len(dataloader))
    for data in pbar:

        background_image = data["background_image"].to(device)
        cropped_image = data["cropped_image"].to(device)

        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        token_type_ids = data["token_type_ids"].to(device)
        label = data["label"].to(device)
        batch_size = background_image.shape[0]

        optimizer.zero_grad()
        output_image, output_text = model(
            background_image=background_image,
            cropped_image=cropped_image,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        loss = criterion(output_image, output_text, label)
        loss.backward()
        optimizer.step()
        model.zero_grad()

        loss_score.update(loss.detach().item(), batch_size)

        pbar.set_postfix(train_loss=loss_score.avg, lr=optimizer.param_groups[0]["lr"])

    if scheduler is not None:
        scheduler.step()

    return loss_score.avg


def validation_fn(data_loader, model, criterion, device):
    model.eval()
    loss_score = AverageMeter()
    with torch.no_grad():
        pbar = tqdm(data_loader, total=len(data_loader))
        for data in pbar:

            background_image = data["background_image"].to(device)
            cropped_image = data["cropped_image"].to(device)

            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            token_type_ids = data["token_type_ids"].to(device)
            label = data["label"].to(device)
            batch_size = background_image.shape[0]

            batch_size = background_image.shape[0]
            output_image, output_text = model(
                background_image=background_image,
                cropped_image=cropped_image,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            loss = criterion(output_image, output_text, label)

            loss_score.update(loss.detach().item(), batch_size)

            pbar.set_postfix(validation_loss=loss_score.avg)

    return loss_score.avg
