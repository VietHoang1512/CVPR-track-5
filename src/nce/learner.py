import os

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from utils.train_utils import AverageMeter


class Learner:
    def __init__(self, epoch, lr, train_dl, val_dl, model, optim, accu_grad_step=1):
        self.epoch = epoch
        self.lr = lr
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.model = model

        self.optim = optim
        self.accu_grad_step = accu_grad_step

    def embeddings(self, dl):
        vi = []
        for batch in dl:
            vi.append(self.model.compute_vi_embed(batch))
            ids = [item["id"] for item in dl.dataset.data]
        return vi, ids

    def freeze(self):
        for param in self.model.clip_model.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def match_track2querry(self, queries, dl):
        results = dict()
        visual_embeds, ids = self.embeddings(dl)
        visual_embeds = torch.cat(visual_embeds, dim=0).cuda()
        q_ids, lang_embeds = [], []
        for q_id, qs in tqdm(queries.items()):
            q_ids.append(q_id)
            lang_embeds.append(self.model.compute_lang_embed(qs))
        lang_embeds = torch.cat(lang_embeds, dim=0).cuda()

        num_vi = len(dl.dataset.data)
        num_qu = len((queries.keys()))
        similarity = F.cosine_similarity(
            visual_embeds.unsqueeze(1).repeat(1, num_qu * 3, 1).view(-1, 256),
            lang_embeds.unsqueeze(0).repeat(num_vi, 1, 1).view(-1, 256),
        )
        similarity = similarity.view(num_vi, num_qu, 3).mean(-1).permute(1, 0)

        for q_id, simi in zip(q_ids, similarity):

            track_score = {id: si for id, si in zip(ids, simi.detach().cpu().numpy().tolist())}
            top_tracks = sorted(track_score, key=track_score.get, reverse=True)
            results[q_id] = top_tracks
        return results

    def eval_one_epoch(self, dl):

        queries = {item["id"]: item["nl"] for item in dl.dataset.data}
        gt_tracks = list(queries.keys())
        self.model.eval()
        with torch.no_grad():
            results = self.match_track2querry(queries, dl)

        recall_5 = 0
        recall_10 = 0
        mrr = 0

        for query in gt_tracks:
            result = results[query]
            target = query
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

        return recall_5, recall_10, mrr

    def train_one_epoch(self):
        epoch_loss = 0.0
        self.model.train()
        loss_score = AverageMeter()
        pbar = tqdm(self.train_dl)
        for i, (batch) in enumerate(pbar):
            loss = self.model.compute_loss(batch)
            loss_score.update(loss.detach().item())
            pbar.set_postfix(train_loss=loss_score.avg)
            epoch_loss += loss
            loss = loss / self.accu_grad_step
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            if (i + 1) % self.accu_grad_step == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
        print(f"loss: {epoch_loss.item()/(i+1)}")

    def train(self, output_dir, lr=None, epoch=None, best_mrr=None):
        if lr is not None:
            self.lr = lr
        if epoch is not None:
            self.epoch = epoch
        best_mrr = 0 if best_mrr is None else best_mrr
        self.optimizer = self.optim(self.model.parameters(), lr=self.lr)
        os.makedirs(output_dir, exist_ok=True)
        ckpt_fp = os.path.join(output_dir, "ckpt.pth")
        for i in range(self.epoch):
            self.train_one_epoch()
            recall_5, recall_10, mrr = self.eval_one_epoch(self.val_dl)
            if mrr > best_mrr:
                best_mrr = mrr
                print("Save model ....")
                torch.save(self.model.state_dict(), ckpt_fp)

            print(f"epoch {i} : recall_5: {recall_5}, recall_10 : {recall_10}, mrr: {mrr}")
