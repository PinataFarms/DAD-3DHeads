import os
from typing import Dict, Any, Optional

from pytorch_lightning.callbacks import ModelCheckpoint

from .base import BaseCallback

__all__ = ["ModelCheckpointCallback"]


class ModelCheckpointCallback(ModelCheckpoint, BaseCallback):
    """
    Drop-in replacement for pl.ModelCheckpoint with the support of keys `metric/metric_name'
    """

    def format_checkpoint_name(self, metrics: Dict[str, Any], ver: Optional[int] = None) -> str:
        filename = self._format_checkpoint_name(
            self.filename, metrics, auto_insert_metric_name=self.auto_insert_metric_name
        )
        filename = self.sanitize_metric_name(filename)

        if ver is not None:
            filename = self.CHECKPOINT_JOIN_CHAR.join((filename, f"v{ver}"))

        ckpt_name = f"{filename}{self.FILE_EXTENSION}"
        return os.path.join(self.dirpath, ckpt_name) if self.dirpath else ckpt_name

    @staticmethod
    def sanitize_metric_name(metric_name: str) -> str:
        """
        Replace characters in string that are not path-friendly with underscore
        """
        for s in ["?", "/", "\\", ":", "<", ">", "|", "'", '"', "#", "="]:
            metric_name = metric_name.replace(s, "_")
        return metric_name

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> Optional["ModelCheckpointCallback"]:
        if "experiment" not in config.keys():
            return None
        monitor_key = config.get("metric_to_monitor", "valid/loss")
        return cls(
            dirpath=os.path.join(config["experiment"]["folder"], config["experiment"]["name"], "checkpoints"),
            verbose=True,
            save_top_k=config.get("save_top_k", 1),
            monitor=monitor_key,
            mode=config.get("metric_mode", "min"),
            save_last=config.get("save_last", True),
            save_weights_only=config.get("checkpoints_save_weights_only", True),
            filename=f"{{epoch:04d}}-{{{monitor_key}:.4f}}",
        )
