import os
from typing import Any, List

import torch.nn as nn
from pytorchcv.model_provider import get_model
from utils import load_yaml


class Encoder(nn.Module):
    def __init__(self, model_name: str, pretrained: bool = True, config_name: str = "backbone.yaml") -> None:
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.model = self._load_model()
        self.stages = self._get_stages()
        config = load_yaml(os.path.join(os.path.dirname(__file__), config_name))
        self.encoder_channels = config[model_name].get("block_size", None)
        self.final_block_channels = config[model_name].get("final_block_channels", None)

    def _load_model(self) -> Any:
        model = get_model(self.model_name, pretrained=self.pretrained).features
        return model

    def _get_stages(self) -> List[Any]:
        stages = [
            nn.Sequential(self.model.init_block, self.model.stage1),
            self.model.stage2,
            self.model.stage3,
            self.model.stage4,
            self.model.stage5,
        ]
        return stages

    def forward(self, x: Any) -> List[Any]:
        encoder_maps = []
        for stage in self.stages:
            x = stage(x)
            encoder_maps.append(x)
        return encoder_maps


class StagedEncoder(Encoder):
    def __init__(self, model_name: str, pretrained: bool = True, config_name: str = "backbone.yaml") -> None:
        super().__init__(model_name=model_name, pretrained=pretrained, config_name=config_name)

    def _get_stages(self) -> List[Any]:
        stages = [self.model.init_block, self.model.stage1, self.model.stage2, self.model.stage3, self.model.stage4]
        return stages


encoder_mapping = {
    "resnet50": StagedEncoder,
    "mobilenet_w1": Encoder
}


def get_encoder(encoder_name: str, pretrained: bool = True, config_name: str = "backbone.yaml") -> Encoder:
    encoder = encoder_mapping[encoder_name](encoder_name, pretrained, config_name)
    return encoder
