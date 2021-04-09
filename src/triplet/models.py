"""Baseline Siamese model."""
import clip
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from transformers import AutoConfig, AutoModel


class EfficientModel(nn.Module):
    def __init__(self, image_model, bert_model, n_hiddens, out_features):
        super().__init__()
        # 3 image models
        self.background_image_model = EfficientNet.from_pretrained(image_model, include_top=False)
        self.cropped_image_model = EfficientNet.from_pretrained("efficientnet-b0", include_top=False)
        # self.flow_image_model = EfficientNet.from_pretrained("efficientnet-b0", include_top=False)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        n_features = (
            self.background_image_model._fc.in_features
            + self.cropped_image_model._fc.in_features
            # + self.flow_image_model._fc.in_features
        )
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
        bs = background_image.size(0)
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
        output = self.pooling(output).view(bs, -1)
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


class VisionTransFormerModel(nn.Module):
    def __init__(self, image_model, bert_model, n_hiddens, out_features):
        super().__init__()
        # 3 image models

        self.background_image_model = timm.create_model(image_model, pretrained=True)
        self.cropped_image_model = timm.create_model(image_model, pretrained=True)

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


class CLIPModel(nn.Module):
    def __init__(self, clip_model, device):
        super().__init__()
        # 3 image models
        self.background_image_model, background_preprocess = clip.load(clip_model, device=device, jit=False)
        self.cropped_image_model, cropped_preprocess = clip.load(clip_model, device=device, jit=False)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        n_features = (
            self.background_image_model.text_projection.shape[1] + self.cropped_image_model.text_projection.shape[1]
        )
        self.image_classifier = nn.Linear(n_features, self.background_image_model.text_projection.shape[1])
        self.preprocess = background_preprocess

    def forward_image(self, background_image, cropped_image):
        background_output = self.background_image_model.encode_image(background_image)
        cropped_output = self.background_image_model.encode_image(cropped_image)
        output = torch.cat(
            [
                background_output,
                cropped_output,
                # flow_output
            ],
            axis=1,
        )
        output = output.float()
        output = self.image_classifier(output)
        return output

    def forward_text(self, input_text):
        output = self.background_image_model.encode_text(input_text)
        return output

    # TODO: consider a image model for flow only
    def forward(self, background_image, cropped_image, pos_inputs, neg_inputs):
        pos_output_text = self.forward_text(pos_inputs)
        neg_output_text = self.forward_text(neg_inputs)
        output_image = self.forward_image(background_image, cropped_image)

        pos_output_text = F.normalize(pos_output_text, p=2, dim=1)
        neg_output_text = F.normalize(neg_output_text, p=2, dim=1)
        output_image = F.normalize(output_image, p=2, dim=1)

        return output_image, pos_output_text, neg_output_text
