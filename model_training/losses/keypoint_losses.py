import torch
import torch.nn as nn


class IoULoss(nn.Module):
    """IoU loss"""

    def __init__(self) -> None:
        super(IoULoss, self).__init__()

    @staticmethod
    def iou_metric(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        _EPSILON = 1e-6

        def op_sum(x: torch.Tensor) -> torch.Tensor:
            return x.view(x.shape[0], x.shape[1], -1).sum(2)

        loss = (op_sum(y_true * y_pred) + _EPSILON) / (
            op_sum(y_true ** 2) + op_sum(y_pred ** 2) - op_sum(y_true * y_pred) + _EPSILON
        )

        loss = torch.mean(loss)
        return loss

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute IoU loss
        Args:
            y_pred (torch.Tensor): predicted values
            y_true (torch.Tensor): target values
        """
        return 1 - self.iou_metric(torch.sigmoid(y_pred), y_true)
