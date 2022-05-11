import copy
from typing import Dict, Any, Optional, Callable

import torch
from pytorch_lightning import Trainer
from model_training.train.flame_lightning_model import FlameLightningModel
from model_training.train.utils import create_trainer
from model_training.utils import create_logger
from pytorch_toolbelt.utils import fs


class DAD3DTrainer:
    def __init__(self, dad3d_net: FlameLightningModel, config: Dict[str, Any]) -> None:
        self.dad3d_net = dad3d_net
        self.config = config
        self.logger = create_logger(__name__)
        self.trainer = create_trainer(self.config)
        self.hparams: Dict[str, Any] = {}

    def local_train(
        self,
        test_fn: Optional[Callable] = None,
    ) -> None:
        self.trainer.fit(self.dad3d_net)

        if test_fn is not None:
            metrics = self.evaluate(self.trainer, test_fn)
            for metric_name, score in metrics.items():
                self.logger.info(f"{metric_name} - {score}")

        try:
            self.export_and_save_jit_model(self.trainer.checkpoint_callback.best_model_path)
        except (NotImplementedError, KeyError):
            self.logger.info("Skipping model tracing step since export_jit_model is not implemented")

    def evaluate(self, trainer: Trainer, test_fn: Optional[Callable] = None) -> Dict[str, Any]:
        if test_fn and self.dad3d_net.is_master:
            test_config = copy.deepcopy(self.config)
            test_config["model_path"] = trainer.checkpoint_callback.best_model_path
            test_config["weights_path"] = trainer.checkpoint_callback.best_model_path
            test_config["img_size"] = self.config["test"]["img_size"]

            metrics = test_fn(self.config["test"]["ann_path"], test_config)
            return metrics
        return {}

    def export_and_save_jit_model(self, checkpoint_filename: str) -> str:
        # Forward export call to underlying model and save model to trcd file
        output_path = fs.change_extension(checkpoint_filename, ".trcd")
        traced_model = self.dad3d_net.export_jit_model(checkpoint_filename)
        torch.jit.save(traced_model, output_path)
        return output_path

    def fit(
        self,
        test_fn: Optional[Callable] = None,
    ) -> None:
        self.local_train(test_fn=test_fn)
