from typing import Optional, Callable, Any, Dict, Union

import torch
import torch.nn
from torch import Tensor
from torchmetrics import Metric

__all__ = ["FailureRate", "KeypointsNME"]


def metrics_w_bbox_wrapper(
    outputs: Tensor, gts: Union[Tensor, Dict[str, Tensor]], function: Callable, *args: Any, **kwargs: Any
) -> Tensor:
    gt_bboxes = gts["bboxes"] if "bboxes" in gts.keys() else None
    gt_keypoints = gts["keypoints"]
    return function(outputs, gt_keypoints, gt_bboxes, *args, **kwargs)


def keypoints_nme(
    output_kp: Tensor,
    target_kp: Tensor,
    bbox: Tensor = None,
    reduce: str = "mean",
) -> Tensor:
    """
    https://arxiv.org/pdf/1708.07517v2.pdf
    """
    err = (output_kp - target_kp).norm(2, -1).mean(-1)
    # norm_distance = 2.0 for the 3D case, where the keypoints are in the normalized cube [-1; 1] ^ 3.
    norm_distance = torch.sqrt(bbox[:, 2] * bbox[:, 3]) if bbox is not None else 2.0
    nme = torch.div(err, norm_distance)
    if reduce == "mean":
        nme = torch.mean(nme)
    return nme


def percentage_of_errors_below_IOD(
    output_kp: Tensor,
    target_kp: Tensor,
    bbox: Tensor = None,
    threshold: float = 0.05,
    below: bool = True,
) -> Tensor:
    """
    https://arxiv.org/pdf/1708.07517v2.pdf
    """
    bs = output_kp.shape[0]
    err = (output_kp - target_kp).norm(2, -1).mean(-1)
    # norm_distance = 2.0 for the 3D case, where the keypoints are in the normalized cube [-1; 1] ^ 3.
    norm_distance = torch.sqrt(bbox[:, 2] * bbox[:, 3]) if bbox is not None else 2.0
    number_of_images = (err < threshold * norm_distance).sum() if below else (err > threshold * norm_distance).sum()
    return number_of_images / bs  # percentage of such examples in a batch


class FailureRate(Metric):
    """Compute the Failure Rate metric for [2/3]D keypoints with averaging across individual examples."""

    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
        threshold: float = 0.05,
        below: bool = True,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.threshold = threshold
        self.below = below
        self.add_state("failure_rate", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, pred_keypoints: Tensor, gts: Dict[str, Tensor]) -> None:
        """
        Update state with predictions and targets.

        Args:
            pred_keypoints: (Tensor[B, C, dim]): predicted keypoints tensor
            gts: Dict of tensors:
                {'keypoints' : Tensor[B, C, dim], 'bboxes': Tensor[B, 4]}.
                The key 'bboxes' expected for dim=2.

        dim reflects 2D-3D mode.
        """

        self.failure_rate += metrics_w_bbox_wrapper(
            function=percentage_of_errors_below_IOD,
            outputs=pred_keypoints,
            gts=gts,
            threshold=self.threshold,
            below=self.below,
        )

        self.total += 1

    def compute(self) -> torch.Tensor:
        """
        Computes accuracy over state.
        """
        return self.failure_rate / self.total


class KeypointsNME(Metric):
    """Compute the NME Metric for [2/3]D keypoints with averaging across individual examples.
    https://arxiv.org/pdf/1708.07517v2.pdf
    """

    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
        weight: int = 100,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.weight = weight
        self.add_state("nme", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, pred_keypoints: Tensor, gts: Dict[str, Tensor]) -> None:
        """
        Update state with predictions and targets.

        Args:
            pred_keypoints: (Tensor[B, C, dim]): predicted keypoints tensor
            gts: Dict of tensors:
                {'keypoints' : Tensor[B, C, dim], 'bboxes': Tensor[B, 4]}.
                The key 'bboxes' expected for dim=2.

        dim reflects 2D-3D mode.
        """
        self.nme += metrics_w_bbox_wrapper(function=keypoints_nme, outputs=pred_keypoints, gts=gts)
        self.total += 1

    def compute(self) -> torch.Tensor:
        """
        Computes accuracy over state.
        """
        return self.weight * (self.nme / self.total)
