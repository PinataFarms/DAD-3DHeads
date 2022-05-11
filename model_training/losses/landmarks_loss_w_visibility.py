from torch import nn, Tensor
from typing import List


__all__ = ["LandmarksLossWVisibility"]

losses = {"l1": nn.L1Loss, "l2": nn.MSELoss, "smooth_l1": nn.SmoothL1Loss}


class LandmarksLossWVisibility(nn.Module):
    def __init__(self, criterion):
        super().__init__()
        if criterion not in losses.keys():
            raise ValueError(f"Unsupported discrepancy loss type {criterion}")
        self.criterion = losses[criterion]()

    def forward(self, predicted: List[Tensor], target: List[Tensor]) -> Tensor:
        """
        Args:
            predicted: [Tensor[B,N,2] of landmarks, Tensor[B, N] of their presence]
            target: [Tensor[B,N,2] of landmarks, Tensor[B, N] of their presence]
        """
        pred_vertices, pred_presence = predicted[0], predicted[1]
        target_vertices, target_presence = target[0], target[1]

        return self.criterion(pred_vertices * pred_presence[..., None], target_vertices * target_presence[..., None])
