from typing import List, Union, Dict, Tuple, Any

import torch
from hydra.utils import instantiate
from torch import nn, Tensor


class LossModule(nn.Module):
    def __init__(
        self,
        names: List[str],
        output_keys: List[str],
        target_keys: List[str],
        criterions: List[nn.Module],
        weights: List[float],
        schedule: List[int],
        reduction: str = "sum",
    ):
        super().__init__()
        self.criterions = nn.ModuleList(criterions)
        self.names = names
        self.weights = weights
        self.output_keys = output_keys
        self.target_keys = target_keys
        self.reduction = reduction
        self.schedule = schedule

    def _get_values(
        self, values: Union[Tensor, Dict[str, Tensor]], key: Union[None, str, List[str]]
    ) -> Union[Tensor, List[Tensor]]:
        if torch.is_tensor(values) and key is None:
            return values
        if isinstance(values, Dict) and isinstance(key, str):
            return values[key]
        if isinstance(values, Dict) and isinstance(key, list):
            lst = []
            for k in key:
                lst.append(values[k])
            return lst
        raise ValueError(f"Unsupported combination of values {type(values)} and key {key}")

    def forward(
        self, predictions: Union[Tensor, Dict[str, Tensor]], targets: Union[Tensor, Dict[str, Tensor]], epoch: int
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        losses_dict = {}
        losses: List[Tensor] = []

        for criterion_name, criterion, weight, predicted_key, target_key, epoch_start in zip(
            self.names, self.criterions, self.weights, self.output_keys, self.target_keys, self.schedule
        ):
            if epoch >= epoch_start:
                joint_dict = {**predictions, **targets}
                input = self._get_values(joint_dict, predicted_key)
                target = self._get_values(joint_dict, target_key)
                loss = criterion(input, target) * weight
                losses_dict[criterion_name] = loss
                losses.append(loss)

        losses_stack = torch.stack(losses)
        if self.reduction == "sum":
            total_loss = losses_stack.sum()
        elif self.reduction == "mean":
            total_loss = losses_stack.mean()
        elif self.reduction == "none":
            total_loss = losses_stack
        else:
            raise ValueError(f"Unsupported reduction value {self.reduction}")

        return total_loss, losses_dict

    @staticmethod
    def from_config(config: Dict[str, Any]) -> "LossModule":
        reduction = config.get("reduction", "sum")

        names = []
        output_keys = []
        target_keys = []
        criterions = []
        weights = []
        schedule = []

        for criterion in config["criterions"]:
            name = criterion["name"]

            output_key = criterion.get("output_key", None)
            target_key = criterion["target_key"]
            loss = instantiate(criterion["loss"])
            weight = criterion.get("weight", float(1.0))
            epoch_start = criterion.get("epoch_start", 0)

            names.append(name)
            criterions.append(loss)
            weights.append(weight)
            output_keys.append(output_key)
            target_keys.append(target_key)
            schedule.append(epoch_start)

        return LossModule(
            names=names,
            output_keys=output_keys,
            target_keys=target_keys,
            weights=weights,
            criterions=criterions,
            reduction=reduction,
            schedule=schedule,
        )
