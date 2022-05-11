import torch
from torch import nn, Tensor
from model_training.head_mesh import HeadMesh
from model_training.model.utils import normalize_to_cube
from math import pi
from ..utils import indices_reweighing


__all__ = ["Vertices3DLoss"]

losses = {"l1": nn.L1Loss, "l2": nn.MSELoss, "smooth_l1": nn.SmoothL1Loss}


class Vertices3DLoss(nn.Module):
    def __init__(
        self,
        criterion,
        batch_size,
        consts,
        weights_and_indices,
    ):
        super().__init__()
        if criterion not in losses.keys():
            raise ValueError(f"Unsupported discrepancy loss type {criterion}")
        self.criterion = losses[criterion]()
        self.head_mesh = HeadMesh(flame_config=consts, batch_size=batch_size)

        self.weights, self.indices = indices_reweighing(weights_and_indices)

    @torch.cuda.amp.autocast(False)
    def forward(self, predicted: Tensor, target: Tensor) -> Tensor:
        """

        Args:
            predicted: [B,N,3]
            target: [B,N,3]

        Returns:

        """
        pred_vertices = self.head_mesh.vertices_3d(params_3dmm=predicted, zero_rotation=True)

        v_losses = []
        for w, i in zip(self.weights, self.indices):
            loss = self.criterion(*tuple(map(normalize_to_cube, (pred_vertices[:, i], target[:, i])))) * w
            v_losses.append(loss)

        return torch.stack(v_losses).sum()