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

        pos_input_ids = data["pos_input_ids"].to(device)
        pos_attention_mask = data["pos_attention_mask"].to(device)
        pos_token_type_ids = data["pos_token_type_ids"].to(device)

        neg_input_ids = data["neg_input_ids"].to(device)
        neg_attention_mask = data["neg_attention_mask"].to(device)
        neg_token_type_ids = data["neg_token_type_ids"].to(device)

        batch_size = background_image.shape[0]

        optimizer.zero_grad()
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
