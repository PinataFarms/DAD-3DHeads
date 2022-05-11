from typing import Dict, Any

import torch
import torch.nn as nn
from model_training.data.config import OUTPUT_LANDMARKS_HEATMAP, OUTPUT_3DMM_PARAMS, OUTPUT_2D_LANDMARKS
from model_training.model.encoders import get_encoder
from model_training.model.bifpn import BiFPN
from model_training.model.layers import IdentityLayer
from torch.nn import functional as F

__all__ = ["FlameRegression"]


class FlameHead(IdentityLayer):
    def __init__(self, model_config, network_metadata):
        super().__init__(model_config=model_config, network_metadata=network_metadata)
        self.heatmap = nn.Conv2d(
            model_config["num_filters"], out_channels=model_config["num_classes"], kernel_size=3, padding=1
        )
        self.heatmap.bias.data.fill_(0.0)

    def forward(self, decoder_output):
        x = decoder_output[0]
        heatmap = self.heatmap(x)
        return heatmap


class FusionLayer(nn.Module):
    def __init__(self, num_filters, num_heatmaps, output_filters):
        super().__init__()
        self.conv1x1 = nn.Conv2d(num_filters + num_heatmaps + output_filters, output_filters, kernel_size=1)

    def forward(self, x, heatmap, bifpn_map):
        _, _, h, w = x.size()
        original_h = h if isinstance(h, int) else h.item()
        original_w = w if isinstance(w, int) else w.item()
        heatmap = nn.functional.interpolate(
            heatmap, size=(original_h, original_w), mode="bilinear", align_corners=True
        ).sigmoid()
        fmap = torch.cat([x, heatmap, bifpn_map], dim=1)
        fmap = self.conv1x1(fmap)
        return fmap * x


class ClassificationHead(nn.Module):
    def __init__(self, num_filters, num_classes, dropout=0.3, linear_size=512):
        super().__init__()

        self.logit_image = nn.Sequential(
            nn.Linear(num_filters, linear_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(linear_size, num_classes),
        )

    def forward(self, x):
        batch_size, C, H, W = x.shape
        f = F.adaptive_avg_pool2d(x, output_size=1)
        return self.logit_image(f.view(batch_size, -1))


class FlameRegression(nn.Module):
    def __init__(self, model_config: Dict[str, Any], consts_config: Dict[str, Any], num_classes: int = 68):
        super().__init__()
        self.encoder = get_encoder(model_config["backbone"], model_config.get("pretrained", False))
        self.bifpn = BiFPN(
            [
                self.encoder.encoder_channels["layer3"],
                self.encoder.encoder_channels["layer2"],
                self.encoder.encoder_channels["layer1"],
            ],
            model_config["num_filters"],
        )
        self.head = FlameHead(
            {"num_filters": model_config["num_filters"], "num_classes": model_config["num_classes"]}, {}
        )
        self.max_layer = 4
        self.limit_value = model_config["limit_value"]
        self.fusion_layer = FusionLayer(
            model_config["num_filters"], model_config["num_classes"], self.encoder.encoder_channels["layer1"]
        )

        self.shape = ClassificationHead(self.encoder.encoder_channels["layer0"], 403)
        self.pose = ClassificationHead(self.encoder.encoder_channels["layer0"], 10)
        self.landmarks = ClassificationHead(self.encoder.encoder_channels["layer0"], num_classes * 2)

    def forward(self, x):
        encoder_output = []
        for stage in self.encoder.stages[: self.max_layer]:
            x = stage(x)
            encoder_output.append(x)
        decoder_output = self.bifpn(encoder_output[1:])
        heatmap = self.head(decoder_output)
        fmap = self.fusion_layer(x, heatmap, decoder_output[2])
        fmap = self.encoder.stages[-1](fmap)
        shape = self.shape(fmap).tanh() * self.limit_value
        pose = self.pose(fmap)
        landmarks = self.landmarks(fmap)
        B, N = landmarks.size()
        landmarks = F.relu(landmarks.reshape((B, N // 2, 2)), inplace=True)

        return {
            OUTPUT_LANDMARKS_HEATMAP: heatmap,
            OUTPUT_3DMM_PARAMS: torch.cat([shape, pose], dim=1),
            OUTPUT_2D_LANDMARKS: landmarks,
        }