import os
from typing import Dict, Any, List

import numpy as np
import torch
from torch.types import Device
import pytorch_lightning as pl
from model_training.utils import create_logger

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, LightningLoggerBase

logger = create_logger(__file__)


def _init_logger(config: Dict[str, Any]) -> List[LightningLoggerBase]:
    if "experiment" not in config.keys():
        return []
    experiment_dir = os.path.join(config["experiment"]["folder"], config["experiment"]["name"])
    version_tag = config["experiment"]["version"] if "version" in config["experiment"].keys() else 0
    tt_logger = TensorBoardLogger(
        save_dir=os.path.join(experiment_dir, "logs"),
        version=version_tag,
        name="logs",
    )
    return [tt_logger]


def get_callbacks(config: Dict[str, Any]) -> List[pl.Callback]:
    """
    Return callbacks, order matters
    """
    from model_training.train.callbacks import (
        ModelCheckpointCallback,
        EarlyStoppingCallback,
    )

    callback_classes = [
        ModelCheckpointCallback,
        EarlyStoppingCallback,
    ]
    callbacks = list()
    for callback_class in callback_classes:
        callback = callback_class.from_config(config)
        if callback is not None:
            callbacks.append(callback)
    return callbacks


def create_trainer(config: dict) -> Trainer:
    if "backend" in config and "accelerator" not in config:
        logger.warning(
            "Your config has 'backend' key, which is deprecated in favor of 'accelerator' to "
            "match the namings in pytorch-lightning. We will use 'backend' as for now, but "
            "consider updating your configs to reflect this change."
        )
        config["accelerator"] = config["backend"]

    trainer = Trainer(
        logger=_init_logger(config),
        gpus=config["gpus"],
        accelerator=config.get("accelerator", None),
        sync_batchnorm=config.get("sync_bn", False),
        auto_scale_batch_size=config.get("auto_bs", False),
        benchmark=config.get("cuda_benchmark", True),
        precision=config.get("precision", 32),
        callbacks=get_callbacks(config=config),
        min_epochs=config["min_epochs"],
        max_epochs=config["max_epochs"],
        gradient_clip_val=config.get("gradient_clip_val", 0),
        val_check_interval=config.get("val_check_interval", 1.0),
        limit_train_batches=config.get("train_percent", 1.0),
        limit_val_batches=config.get("val_percent", 1.0),
        progress_bar_refresh_rate=config.get("progress_bar_refresh_rate", 10),
        num_sanity_val_steps=config.get("sanity_steps", 5),
        log_every_n_steps=1,
        auto_lr_find=config.get("auto_lr", False),
        replace_sampler_ddp=config.get("replace_sampler_ddp", True),
        check_val_every_n_epoch=config.get("check_val_every_n_epoch", 1),
    )
    return trainer


def any2device(value: Any, device: Device) -> Any:
    """
    Move tensor, list of tensors, list of list of tensors,
    dict of tensors, tuple of tensors to target device.

    Args:
        value: Object to be moved
        device: target device ids

    Returns:
        Same structure as value, but all tensors moved to specified device
    """
    if isinstance(value, dict):
        return {k: any2device(v, device) for k, v in value.items()}
    elif isinstance(value, (tuple, list)):
        return [any2device(v, device) for v in value]
    elif torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    elif isinstance(value, (np.ndarray, np.void)) and value.dtype.fields is not None:
        return {k: any2device(value[k], device) for k in value.dtype.fields.keys()}
    return value

