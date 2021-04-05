import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from src.utils.train_utils import AverageMeter


def train_fn(dataloader, model, criterion, optimizer, device, scheduler):
    model.train()

    loss_score = AverageMeter()

    pbar = tqdm(dataloader, total=len(dataloader))
    for data in pbar:

        # TODO: Fuck these lists
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

        pbar.set_postfix(train_loss=loss_score.avg, lr=optimizer.param_groups["lr"])

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


def official_evaluate(model, tracks, tokenizer, max_len, dim, augmentation, device):
    embedding_queries = {}
    embedding_tracks = {}
    track_ids = list(tracks.keys())
    for query_id in tqdm(track_ids, desc="Getting embeddings"):
        sample = tracks[query_id]
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

            if dim is not None:
                background_image = cv2.resize(background_image, dim)
                cropped_image = cv2.resize(cropped_image, dim)

            if augmentation is not None:
                background_image = augmentation(image=background_image)["image"]
                cropped_image = augmentation(image=cropped_image)["image"]

            background_image = background_image.astype(np.float32)
            cropped_image = cropped_image.astype(np.float32)
            background_image = background_image.transpose(2, 0, 1)
            cropped_image = cropped_image.transpose(2, 0, 1)

            input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
            attention_mask = torch.tensor(attention_mask).unsqueeze(0).to(device)
            token_type_ids = torch.tensor(token_type_ids).unsqueeze(0).to(device)

            background_image = torch.tensor(background_image, dtype=torch.float32).unsqueeze(0).to(device)
            cropped_image = torch.tensor(cropped_image, dtype=torch.float32).unsqueeze(0).to(device)

            output_image, output_text = model(
                background_image=background_image,
                cropped_image=cropped_image,
                pos_input_ids=input_ids,
                pos_attention_mask=attention_mask,
                pos_token_type_ids=token_type_ids,
                neg_input_ids=input_ids,
                neg_attention_mask=attention_mask,
                neg_token_type_ids=token_type_ids,
            )
            text_embeds.append(output_text.detach().cpu().numpy())
        embedding_queries[query_id] = text_embeds
        embedding_tracks[query_id] = output_image.detach().cpu().numpy()

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
