import torch
from tqdm.auto import tqdm
from utils.train_utils import AverageMeter


def train_fn(dataloader, model, criterion, optimizer, device, scheduler, epoch):
    model.train()

    loss_score = AverageMeter()

    pbar = tqdm(dataloader, total=len(dataloader))
    for data in pbar:

        # TODO: Fuck these lists
        background_image = data["background_image"][0].to(device)
        cropped_image = data["cropped_image"][0].to(device)

        input_ids = data["input_ids"][0].to(device)
        attention_mask = data["attention_mask"][0].to(device)
        token_type_ids = data["token_type_ids"][0].to(device)
        label = data["label"][0].to(device)

        batch_size = background_image.shape[0]

        optimizer.zero_grad()
        model.forward_image(background_image, cropped_image)
        output_1, output_2 = model(background_image, cropped_image, input_ids, attention_mask, token_type_ids)

        loss = criterion(output_1, output_2, label)
        loss.backward()
        optimizer.step()
        model.zero_grad()

        loss_score.update(loss.detach().item(), batch_size)

        pbar.set_postfix(Train_Loss=loss_score.avg, Epoch=epoch, LR=optimizer.param_groups[0]["lr"])

    if scheduler is not None:
        scheduler.step()

    return loss_score


def validation_fn(data_loader, model, criterion, device):
    model.eval()
    AverageMeter()
    with torch.no_grad():
        pbar = tqdm(data_loader, total=len(data_loader))
        for data in pbar:
            # TODO: Fuck these lists
            background_image = data["background_image"][0].to(device)
            cropped_image = data["cropped_image"][0].to(device)

            input_ids = data["input_ids"][0].to(device)
            attention_mask = data["attention_mask"][0].to(device)
            token_type_ids = data["token_type_ids"][0].to(device)
            label = data["label"][0].to(device)

            background_image.shape[0]
            model.forward_image(background_image, cropped_image)
            output_1, output_2 = model(background_image, cropped_image, input_ids, attention_mask, token_type_ids)
            criterion(output_1, output_2, label)
