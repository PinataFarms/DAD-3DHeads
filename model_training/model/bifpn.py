from typing import List

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["BiFPN", "BiFPNBlock", "BiFPNConvBlock", "BiFPNDepthwiseConvBlock"]


class BiFPNDepthwiseConvBlock(nn.Module):
    """
    Depthwise seperable convolution.


    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        act: nn.Module = nn.ReLU,
    ):
        super(BiFPNDepthwiseConvBlock, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False
        )

        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9997, eps=4e-5)
        self.act = act(inplace=False)

    def forward(self, inputs: Tensor) -> Tensor:
        x = self.depthwise(inputs)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class BiFPNConvBlock(nn.Module):
    """
    Convolution block with Batch Normalization and ReLU activation.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        act: nn.Module = nn.ReLU,
        dilation: int = 1,
        freeze_bn: bool = False,
    ):
        super(BiFPNConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9997, eps=4e-5)
        self.act = act(inplace=False)

    def forward(self, inputs: Tensor) -> Tensor:
        x = self.conv(inputs)
        x = self.bn(x)
        return self.act(x)


class BiFPNBlock(nn.Module):
    """
    Bi-directional Feature Pyramid Network
    """

    def __init__(self, feature_size: int = 64, epsilon: float = 0.0001, act: nn.Module = nn.ReLU):
        super(BiFPNBlock, self).__init__()
        self.epsilon = epsilon

        self.p3_td = BiFPNDepthwiseConvBlock(feature_size, feature_size, act=act)
        self.p4_td = BiFPNDepthwiseConvBlock(feature_size, feature_size, act=act)
        self.p5_td = BiFPNDepthwiseConvBlock(feature_size, feature_size, act=act)
        self.p6_td = BiFPNDepthwiseConvBlock(feature_size, feature_size, act=act)

        self.p4_out = BiFPNDepthwiseConvBlock(feature_size, feature_size, act=act)
        self.p5_out = BiFPNDepthwiseConvBlock(feature_size, feature_size, act=act)
        self.p6_out = BiFPNDepthwiseConvBlock(feature_size, feature_size, act=act)
        self.p7_out = BiFPNDepthwiseConvBlock(feature_size, feature_size, act=act)

        self.w1 = nn.Parameter(torch.Tensor(2, 4))
        self.w1_relu = nn.ReLU(inplace=False)
        self.w2 = nn.Parameter(torch.Tensor(3, 4))
        self.w2_relu = nn.ReLU(inplace=False)

        torch.nn.init.constant_(self.w1, 1)
        torch.nn.init.constant_(self.w2, 1)

    def forward(self, inputs) -> List[Tensor]:
        p3_x, p4_x, p5_x, p6_x, p7_x = inputs

        # Calculate Top-Down Pathway
        w1 = self.w1_relu(self.w1)
        w11 = w1 / torch.sum(w1, dim=0) + self.epsilon
        w2 = self.w2_relu(self.w2)
        w22 = w2 / torch.sum(w2, dim=0) + self.epsilon

        p7_td = p7_x
        p6_td = self.p6_td(w11[0, 0] * p6_x + w11[1, 0] * F.interpolate(p7_td, size=p6_x.size()[2:]))
        p5_td = self.p5_td(w11[0, 1] * p5_x + w11[1, 1] * F.interpolate(p6_td, size=p5_x.size()[2:]))
        p4_td = self.p4_td(w11[0, 2] * p4_x + w11[1, 2] * F.interpolate(p5_td, size=p4_x.size()[2:]))
        p3_td = self.p3_td(w11[0, 3] * p3_x + w11[1, 3] * F.interpolate(p4_td, size=p3_x.size()[2:]))

        # Calculate Bottom-Up Pathway
        p3_out = p3_td
        p4_out = self.p4_out(
            w22[0, 0] * p4_x + w22[1, 0] * p4_td + w22[2, 0] * F.interpolate(p3_out, size=p4_x.size()[2:])
        )
        p5_out = self.p5_out(
            w22[0, 1] * p5_x + w22[1, 1] * p5_td + w22[2, 1] * F.interpolate(p4_out, size=p5_x.size()[2:])
        )
        p6_out = self.p6_out(
            w22[0, 2] * p6_x + w22[1, 2] * p6_td + w22[2, 2] * F.interpolate(p5_out, size=p6_x.size()[2:])
        )
        p7_out = self.p7_out(
            w22[0, 3] * p7_x + w22[1, 3] * p7_td + w22[2, 3] * F.interpolate(p6_out, size=p7_x.size()[2:])
        )

        return [p3_out, p4_out, p5_out, p6_out, p7_out]


class BiFPN(nn.Module):
    def __init__(self, size: List[int], feature_size: int = 128, num_layers: int = 2, act: nn.Module = nn.ReLU):
        super(BiFPN, self).__init__()
        self.p3 = nn.Conv2d(size[0], feature_size, kernel_size=1, stride=1, padding=0)
        self.p4 = nn.Conv2d(size[1], feature_size, kernel_size=1, stride=1, padding=0)
        self.p5 = nn.Conv2d(size[2], feature_size, kernel_size=1, stride=1, padding=0)

        # p6 is obtained via a 3x3 stride-2 conv on C5
        self.p6 = nn.Conv2d(size[2], feature_size, kernel_size=3, stride=2, padding=1)

        # p7 is computed by applying ReLU followed by a 3x3 stride-2 conv on p6
        self.p7 = BiFPNConvBlock(feature_size, feature_size, kernel_size=3, stride=2, padding=1, act=act)

        bifpns = []
        for _ in range(num_layers):
            bifpns.append(BiFPNBlock(feature_size, act=act))
        self.bifpn = nn.Sequential(*bifpns)

    def forward(self, inputs: List[Tensor]) -> List[Tensor]:
        c2, c3, c4 = inputs

        # Calculate the input column of BiFPN
        p3_x = self.p3(c2)
        p4_x = self.p4(c3)
        p5_x = self.p5(c4)
        p6_x = self.p6(c4)
        p7_x = self.p7(p6_x)

        features = [p3_x, p4_x, p5_x, p6_x, p7_x]
        return self.bifpn.forward(features)
