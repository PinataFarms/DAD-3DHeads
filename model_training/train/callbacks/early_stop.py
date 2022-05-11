from typing import Dict, Any, Optional

from pytorch_lightning.callbacks import EarlyStopping
from .base import BaseCallback


class EarlyStoppingCallback(EarlyStopping, BaseCallback):
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> Optional[BaseCallback]:
        if "early_stopping" not in config.keys():
            return None
        return cls(
            monitor=config.get("metric_to_monitor", "valid/loss"),
            min_delta=0.00,
            patience=config["early_stopping"],
            verbose=False,
            mode=config.get("metric_mode", "min"),
        )
