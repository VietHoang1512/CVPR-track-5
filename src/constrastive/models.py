"""Baseline Siamese model."""
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from transformers import AutoConfig, AutoModel


class ContrastiveModel(nn.Module):
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
    def forward(self, background_image, cropped_image, input_ids, attention_mask, token_type_ids):
        output_text = self.forward_text(input_ids, attention_mask, token_type_ids)
        output_image = self.forward_image(background_image, cropped_image)
        # output_text = F.normalize(output_text, p=2, dim=1)
        # output_image = F.normalize(output_image, p=2, dim=1)

        return output_image, output_text


if __name__ == "__main__":
    import os

    from src.utils.train_utils import count_parameters

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    model = ContrastiveModel(image_model="efficientnet-b2", bert_model="roberta-base", n_hiddens=2, out_features=10)
    # print(model.eval())
    print("*" * 50)
    print("Total params:", count_parameters(model))
    print("*" * 50)
    image = torch.zeros((5, 3, 512, 512))
    ids = torch.tensor([0] * 30, dtype=torch.long).view(5, -1)
    mask = torch.tensor([0] * 30, dtype=torch.long).view(5, -1)
    token_type_ids = torch.tensor([0] * 30, dtype=torch.long).view(5, -1)

    output_image, output_text = model(image, image, ids, mask, token_type_ids)
    print(output_image.shape, output_text.shape)
    model.forward_image(image, image)
