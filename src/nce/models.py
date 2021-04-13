import copy

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F


class NCEModel(torch.nn.Module):
    def __init__(self, clip_model, out_features, rnn_units=64, weight_lang_loss=0.3, t=0.5):

        super().__init__()
        self.clip_model, _ = clip.load(clip_model, device="cuda")
        output_dim = self.clip_model.text_projection.shape[1]

        self.linear_project = nn.Linear((output_dim + rnn_units) * 2, out_features).half().cuda()
        self.lang_fc = torch.nn.Linear(512, 256).cuda().half()
        self.weight_lang_loss = weight_lang_loss
        self.t = t
        self.rnn = nn.LSTM(
            input_size=2,
            hidden_size=rnn_units,
            num_layers=2,
            bias=True,
            batch_first=True,
            dropout=0.1,
            bidirectional=True,
        )
        self.rnn.half().cuda()

    def forward(self, track):
        bs = track["frame_img"].shape[0]
        nl = copy.errordeepcopy(track["nl"])
        for i in range(bs):
            nl += [item[i] for item in track["neg_nl"]]

        tokens = clip.tokenize(nl).cuda()

        outputs = self.clip_model.encode_text(tokens)

        lang_embeds = self.lang_fc(outputs)

        neg_lang_embeds = lang_embeds[bs:].view(bs, -1, 256)
        query_lang_embeds = lang_embeds[:bs]

        frame_imgs = torch.cat(
            [track["frame_img"], track["neg_frames"].view(-1, *track["neg_frames"].shape[-3:])], dim=0
        )
        frame_embeds = self.clip_model.encode_image(frame_imgs.cuda())

        crops = torch.cat([track["frame_crop_img"], track["neg_crops"].view(-1, *track["neg_crops"].shape[-3:])], dim=0)
        crops_embeds = self.clip_model.encode_image(crops.cuda())

        boxes = torch.cat(
            [track["boxes_points"], track["neg_boxes_points"].view(-1, *track["neg_boxes_points"].shape[-2:])], dim=0
        )
        boxes_embeds = self.rnn(boxes.half().cuda())[0][:, 0, :]

        visual_embeds = self.linear_project(F.relu(torch.cat([frame_embeds, crops_embeds, boxes_embeds], dim=-1)))

        neg_visual_embeds = visual_embeds[bs:].view(bs, -1, 256)
        query_visual_embeds = visual_embeds[:bs]

        return {"ql": query_lang_embeds, "qv": query_visual_embeds, "nl": neg_lang_embeds, "nv": neg_visual_embeds}

    def nce_loss(self, q, po, ne):
        N = q.shape[0]
        C = q.shape[1]
        M = ne.shape[1]

        q_norm = torch.norm(q, dim=1)
        pos_pair_norm = q_norm * torch.norm(po, dim=1).view(
            N,
        )
        neg_pair_norm = q_norm.view(N, 1) * torch.norm(ne, dim=-1).view(N, M)

        pos = torch.exp(
            torch.div(
                torch.bmm(q.view(N, 1, C), po.view(N, C, 1)).view(
                    N,
                )
                / pos_pair_norm,
                self.t,
            )
        )
        neg = torch.sum(
            torch.exp(torch.div(torch.bmm(q.view(N, 1, C), ne.permute(0, 2, 1)).view(N, M) / neg_pair_norm, self.t)),
            dim=1,
        )

        return torch.mean(-torch.log(pos / (pos + neg) + 1e-5))

    def compute_vi_embed(self, track):
        with torch.no_grad():
            frame_embeds = self.clip_model.encode_image(track["frame_img"].cuda())
            crops_embeds = self.clip_model.encode_image(track["frame_crop_img"].cuda())
            box_embeds = self.rnn(track["boxes_points"].half().cuda())[0][:, 0, :]
            visual_embeds = self.linear_project(F.relu(torch.cat([frame_embeds, crops_embeds, box_embeds], dim=-1)))

            return visual_embeds

    def compute_loss(self, track):
        embeds = self.forward(track)
        nce_visual_loss = self.nce_loss(embeds["qv"], embeds["ql"], embeds["nl"])
        nce_lang_loss = self.nce_loss(embeds["ql"], embeds["qv"], embeds["nv"])
        return (1 - self.weight_lang_loss) * nce_visual_loss + self.weight_lang_loss * nce_lang_loss

    def compute_lang_embed(self, nls):
        with torch.no_grad():
            tokens = clip.tokenize(nls).cuda()
            outputs = self.clip_model.encode_text(tokens)
            lang_embeds = self.lang_fc(outputs)
        return lang_embeds
