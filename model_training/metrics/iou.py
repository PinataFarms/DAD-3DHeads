import torch
import torch.nn

from torch import Tensor
from typing import List, Optional, Callable, Any, Union

from torchmetrics import Metric

__all__ = [
    "soft_iou",
    "SoftIoUMetric",
]


def soft_iou(output: Tensor, target: Tensor, eps: float = 1e-6) -> Union[Tensor, List[Tensor]]:
    """
    Compute the Soft IoU score between predicted and target binary mask.

        Args:
            output (Tensor): Predicted class probabilities of shape [B,C,H,W]
            target (Tensor): Target class probabilities of shape [B,C,H,W]
            eps (float): Optional eps for numerical stability
    """

    def op_sum(x: Tensor) -> Tensor:
        return x.sum(dim=(2, 3))

    loss = (op_sum(target * output) + eps) / (op_sum(target ** 2) + op_sum(output ** 2) - op_sum(target * output) + eps)

    loss = torch.mean(loss)
    return loss


class SoftIoUMetric(Metric):
    """Compute the Soft IoU metric with averaging across individual examples."""

    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.add_state("ious", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """

        iou: Tensor = soft_iou(preds, target)

        self.ious += torch.sum(iou)
        self.total += 1

    def compute(self) -> Tensor:
        """
        Computes accuracy over state.
        """
        return torch.mean(self.ious / self.total)
