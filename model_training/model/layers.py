from typing import cast, Any, Optional, Tuple, Union, Dict

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F

import pytorch_toolbelt.modules as pt_modules


def get_conv_block(conv_type: str) -> nn.Module:
    conv_blocks = {"regular": nn.Conv2d, "sep_conv": SepConv, "res_sep_conv": MixSepConv}
    conv_block = conv_blocks[conv_type]
    return cast(nn.Module, conv_block)


class BasePredictionHead(nn.Module):
    def __init__(self, model_config: Dict[str, Any], network_metadata: Dict[str, Any]):
        super().__init__()
        self.model_config = model_config
        self.network_metadata = network_metadata
        self.conv_block = get_conv_block(model_config.get("conv_block", "regular"))
        self.final_activation = pt_modules.instantiate_activation_block(model_config.get("final_activation", "none"))

    def forward(self, x, decoder_output):
        return self.final_activation(decoder_output)


class PixelShuffle(nn.Module):
    def __init__(self, upscale_factor: int = 2):
        super().__init__()
        self.shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x: Tensor) -> Tensor:
        return self.shuffle(x)


class IdentityLayer(nn.Module):
    def __init__(self, model_config: Dict[str, Any], network_metadata: Dict[str, Any]):
        super().__init__()
        self.model_config = model_config
        self.network_metadata = network_metadata

    def forward(self, decoder_output):
        x = decoder_output[0]
        return x


class PixelShuffleUpsample(IdentityLayer):
    def __init__(self, model_config: Dict[str, Any], network_metadata: Dict[str, Any]):
        super().__init__(model_config=model_config, network_metadata=network_metadata)
        is_coreml = model_config.get("is_coreml", False)
        self.shuffle = PixelShuffle(upscale_factor=4)

    def forward(self, decoder_output):
        x = decoder_output[0]
        x = self.shuffle(x)
        return x


class FlameHead(IdentityLayer):
    def __init__(self, model_config: Dict[str, Any], network_metadata: Dict[str, Any]):
        super().__init__(model_config=model_config, network_metadata=network_metadata)
        self.heatmap = nn.Conv2d(
            model_config["num_filters"], out_channels=model_config["num_classes"], kernel_size=3, padding=1
        )
        self.heatmap.bias.data.fill_(0.0)

    def forward(self, decoder_output):
        x = decoder_output[0]
        heatmap = self.heatmap(x)
        return heatmap


class ClassificationHead(nn.Module):
    def __init__(self, num_filters, num_classes, dropout=0.3, linear_size=512):
        super().__init__()

        self.logit_image = nn.Sequential(
            nn.Linear(num_filters, linear_size),
            nn.ReLU(inplace=False),
            nn.Dropout(p=dropout),
            nn.Linear(linear_size, num_classes),
        )

    def forward(self, x):
        batch_size, C, H, W = x.shape
        f = F.adaptive_avg_pool2d(x, output_size=1)
        return self.logit_image(f.view(batch_size, -1))


def conv3x3(conv_block: nn.Module, in_: int, out: int) -> nn.Module:
    return conv_block(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_: int, out: int, conv_block: nn.Module = nn.Conv2d) -> None:
        super(ConvRelu, self).__init__()
        self.conv = conv3x3(conv_block, in_, out)
        self.activation = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.activation(x)
        return x


class SepConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            groups=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


_mask_prediction_layers = {
    "identity": IdentityLayer,
    "pixel_shuffle": PixelShuffleUpsample,
}


def get_mask_prediction_layer(
    model_config: Dict[str, Any], network_metadata: Dict[str, Any], default_layer: str = "identity"
) -> IdentityLayer:
    mask_prediction_layer_name = model_config.get("upsample", default_layer)
    layer = _mask_prediction_layers[mask_prediction_layer_name](model_config, network_metadata)
    return layer
