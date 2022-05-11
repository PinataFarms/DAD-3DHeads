import copy
from typing import Dict, Any, Callable, List, Tuple

import torch.optim as pytorch_optim
import torch_optimizer as optim
from model_training.utils import create_logger
from torch import nn

_torch_optimizers = {
    "adam": pytorch_optim.Adam,
    "adamw": pytorch_optim.AdamW,
    "sgd": pytorch_optim.SGD,
}

logger = create_logger(__file__)

__all__ = ["get_optimizer"]


def get_optimizer(model: nn.Module, optimizer_config: Dict[str, Any]) -> pytorch_optim.Optimizer:
    """Creates optimizer from config
    Args:
        params (dict): neural network parameters:
        optimizer_config (dict): optimizer config:
        - name (str): optimizer name
        and other parameters for specified optimizer
    """
    optimizer_cls: Callable
    config = copy.deepcopy(optimizer_config)
    optimizer_name = config.pop("name")
    if optimizer_name in _torch_optimizers:
        optimizer_cls = _torch_optimizers[optimizer_name]
    else:
        optimizer_cls = optim.get(optimizer_name)

    parameters: List[nn.Parameter] = list(filter(lambda x: x[1].requires_grad, model.parameters()))
    optimizer: pytorch_optim.Optimizer = optimizer_cls(parameters)
    return optimizer
