from typing import List, Union

import torch
from torch import nn, Tensor

from model_training.utils import indices_reweighing
from model_training.head_mesh import HeadMesh

__all__ = ["ReprojectionLoss"]
losses = {"l1": nn.L1Loss, "l2": nn.MSELoss, "smooth_l1": nn.SmoothL1Loss}


class ReprojectionLoss(nn.Module):
    def __init__(self, criterion, batch_size, consts, img_size, weights_and_indices):
        super().__init__()
        if criterion not in losses.keys():
            raise ValueError(f"Unsupported discrepancy loss type {criterion}")
        self.criterion = losses[criterion]()
        self.weights, self.indices = indices_reweighing(weights_and_indices)
        self.head_mesh = HeadMesh(flame_config=consts, batch_size=batch_size, image_size=img_size)

    @torch.cuda.amp.autocast(False)
    def forward(self, predicted: Tensor, target: Union[Tensor, List[Tensor]]) -> Tensor:
        """

        Args:
            predicted: [B,3,H,W]
            target: [B,3,H,W]

        Returns:

        """
        projected_vertices = self.head_mesh.reprojected_vertices(params_3dmm=predicted, to_2d=True)

        c_losses = []

        if isinstance(target, list):
            full_target = target[0]
        else:
            full_target = target

        for w, i in zip(self.weights, self.indices):
            loss = self.criterion(projected_vertices[:, i], full_target[:, i]) * w
            c_losses.append(loss)

        return torch.stack(c_losses).sum()
