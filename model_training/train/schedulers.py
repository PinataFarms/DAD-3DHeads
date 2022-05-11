import copy
import math
from typing import Any, Dict, List

import torch.optim as optim
from omegaconf import OmegaConf
from torch.optim import lr_scheduler
import torch.optim as pytorch_optim
from hydra.utils import instantiate

from model_training.utils import create_logger

logger = create_logger(__name__)


class FlatCosineAnnealingLR(lr_scheduler._LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:
    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)
        T_{cur} \neq (2k+1)T_{max};\\
        \eta_{t+1} = \eta_{t} + (\eta_{max} - \eta_{min})\frac{1 -
        \cos(\frac{1}{T_{max}}\pi)}{2},
        T_{cur} = (2k+1)T_{max}.\\
    When last_epoch=-1, sets initial lr as lr. Notice that because the schedule
    is defined recursively, the learning rate can be simultaneously modified
    outside this scheduler by other operators. If the learning rate is set
    solely by this scheduler, the learning rate at each step becomes:
    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)
    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
    implements the cosine annealing part of SGDR, and not the restarts.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(
        self, optimizer: pytorch_optim.Optimizer, T_max: int, T_flat: int, eta_min: float = 0, last_epoch: int = -1
    ) -> None:
        self.T_max = T_max
        self.T_flat = T_flat
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if not self._get_lr_called_within_step:
            logger.warning(
                "To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.",
                DeprecationWarning,
            )

        if self.last_epoch <= self.T_flat:
            return self.base_lrs
        elif (max(0, self.last_epoch - self.T_flat) - 1 - max(0, self.T_max - self.T_flat)) % (
            2 * max(0, self.T_max - self.T_flat)
        ) == 0:
            return [
                group["lr"] + (base_lr - self.eta_min) * (1 - math.cos(math.pi / max(0, self.T_max - self.T_flat))) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        return [
            (1 + math.cos(math.pi * max(0, self.last_epoch - self.T_flat) / max(0, self.T_max - self.T_flat)))
            / (1 + math.cos(math.pi * (max(0, self.last_epoch - self.T_flat) - 1) / max(0, self.T_max - self.T_flat)))
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self) -> List[float]:
        return [
            self.eta_min
            + (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * max(0, self.last_epoch - self.T_flat) / max(0, self.T_max - self.T_flat)))
            / 2
            for base_lr in self.base_lrs
        ]


_schedulers = {
    "plateau": optim.lr_scheduler.ReduceLROnPlateau,
    "multi_step": optim.lr_scheduler.MultiStepLR,
    "exponential": optim.lr_scheduler.ExponentialLR,
    "cosine": optim.lr_scheduler.CosineAnnealingLR,
    "cyclic": optim.lr_scheduler.CyclicLR,
    "flat_cosine": FlatCosineAnnealingLR,
}


def get_scheduler(optimizer: pytorch_optim.Optimizer, scheduler_config: Dict[str, Any]) -> lr_scheduler._LRScheduler:
    """Creates scheduler from config
    Args:
        optimizer (pytorch_optim.Optimizer): PyTorch Optimizer:
        scheduler_config (dict): scheduler config:
        - name (str): scheduler name
        - warmup_steps (int): Number of warmup steps when LR will gradualy increase from 0 to target learning rate
        and other parameters for specified scheduler
    """
    if isinstance(scheduler_config, OmegaConf):
        scheduler_config = OmegaConf.to_container(scheduler_config, resolve=True)
    scheduler_config = copy.deepcopy(scheduler_config)

    if "warmup_steps" in scheduler_config:
        scheduler_config.pop("warmup_steps")  # Remove warmup_steps when instantiating scheduler instance

    scheduler: lr_scheduler._LRScheduler
    if "_target_" in scheduler_config.keys():
        scheduler = instantiate(scheduler_config, optimizer=optimizer)
    else:
        scheduler_name = scheduler_config.pop("name")
        scheduler = _schedulers[scheduler_name](optimizer, **scheduler_config)
    return scheduler
