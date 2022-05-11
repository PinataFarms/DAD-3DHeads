import os
from joblib import cpu_count
import typing
from typing import Dict, Any, Optional, Tuple, List, Union, Callable
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, ConcatDataset
from torchmetrics import MetricCollection
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.base import DummyLogger

from model_training.data.config import (
    TARGET_2D_LANDMARKS,
    OUTPUT_LANDMARKS_HEATMAP,
    TARGET_LANDMARKS_HEATMAP,
    OUTPUT_3DMM_PARAMS,
    TARGET_3D_MODEL_VERTICES,
    OUTPUT_2D_LANDMARKS,
    TARGET_2D_FULL_LANDMARKS,
    TARGET_2D_LANDMARKS_PRESENCE,
    INPUT_BBOX_KEY,
)
from model_training.model.utils import unravel_index, normalize_to_cube, load_from_lighting
from model_training.head_mesh import HeadMesh
from model_training.metrics.iou import SoftIoUMetric
from model_training.metrics.keypoints import FailureRate, KeypointsNME
from model_training.train.loss_module import LossModule
from model_training.train.mixins import KeypointsDataMixin, KeypointsVisualizationMixin
from model_training.train.optimizers import get_optimizer
from model_training.train.schedulers import get_scheduler
from model_training.train.utils import any2device
from model_training.utils import create_logger


logger = create_logger(__name__)


class FlameLightningModel(pl.LightningModule, KeypointsDataMixin, KeypointsVisualizationMixin):

    _initial_learning_rates: List[List[float]]

    def __init__(self, model: torch.nn.Module, config: Dict[str, Any], train: Dataset, val: Dataset) -> None:
        super().__init__()
        self.model = model
        self.config = config
        self.train_dataset = train
        self.val_dataset = val

        self._load_model(self.model)
        self.criterion = self._build_loss(config.get("loss", None))

        self.use_ddp = self.config.get("accelerator", None) == "ddp"
        self.log_step = config.get("log_step", 1000)
        self.current_step = 0
        self.epoch_num = 0
        self.tensorboard_logger = None
        self.learning_rate: Optional[float] = None

        self.stride = config["train"].get("stride", 2)
        self.images_log_freq = config.get("images_log_freq", 100)

        self.flame_indices = {}
        for key, value in config["train"]["flame_indices"]["files"].items():
            self.flame_indices[key] = np.load(os.path.join(config["train"]["flame_indices"]["folder"], value))

        self._img_size = self.config["model"]["model_config"]["img_size"]
        self.head_mesh = HeadMesh(flame_config=config["constants"], batch_size=config["batch_size"],
                                  image_size=self._img_size)

        # region metrics initialization
        self.iou_metric = SoftIoUMetric(compute_on_step=True)
        self.metrics_2d = MetricCollection(
            {
                "fr_2d_005": FailureRate(compute_on_step=True, threshold=0.05, below=True),
                "fr_2d_01": FailureRate(compute_on_step=True, threshold=0.1, below=True),
                "nme_2d": KeypointsNME(compute_on_step=True),
            }
        )

        self.metrics_reprojection = MetricCollection(
            {
                "reproject_fr_2d_005": FailureRate(compute_on_step=True, threshold=0.05, below=True),
                "reproject_fr_2d_01": FailureRate(compute_on_step=True, threshold=0.1, below=True),
                "reproject_nme_2d": KeypointsNME(compute_on_step=True),
            }
        )

        self.metrics_3d = MetricCollection(
            {
                "fr_3d_005": FailureRate(compute_on_step=True, threshold=0.05, below=True),
                "fr_3d_01": FailureRate(compute_on_step=True, threshold=0.1, below=True),
                "nme_3d": KeypointsNME(compute_on_step=True),
            }
        )
        # endregion

    @property
    def is_master(self) -> bool:
        """
        Returns True if the caller is the master node (Either code is running on 1 GPU or current rank is 0)
        """
        return (self.use_ddp is False) or (torch.distributed.get_rank() == 0)

    def _load_model(self, model: torch.nn.Module) -> torch.nn.Module:
        if self.config.get("load_weights", False):
            weights_path = self.config["weights_path"]
            if "h5" in weights_path:
                model = torch.load(weights_path)
            else:
                model = load_from_lighting(self.model, weights_path)
        return model

    def forward(self, x: Any) -> Any:
        return self.model(x)

    def compute_loss(
            self, loss: Union[torch.Tensor, Dict[str, torch.Tensor], List[torch.Tensor]]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Return tuple of loss tensor and dictionary of named losses as second argument (if possible)
        """
        if torch.is_tensor(loss):
            return loss, {}

        elif isinstance(loss, (tuple, list)):
            total_loss = sum(loss)
            return total_loss, dict((f"loss_{i}", l) for i, l in enumerate(loss))

        elif isinstance(loss, dict):
            total_loss = 0
            for k, v in loss.items():
                total_loss = total_loss + v

            return total_loss, loss
        else:
            raise ValueError("Incompatible Loss type")

    def _get_batch_size(self, mode: str = "train") -> int:
        if isinstance(self.config["batch_size"], dict):
            return self.config["batch_size"][mode]
        return self.config["batch_size"]

    def _get_num_workers(self, loader_name: str) -> int:
        if "num_workers" not in self.config:
            return cpu_count()
        if isinstance(self.config["num_workers"], float):
            return int(cpu_count() * self.config["num_workers"])
        if isinstance(self.config["num_workers"], dict):
            return self.config["num_workers"][loader_name]
        return self.config["num_workers"]

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.train_dataset, self.config, "train")

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.val_dataset, self.config, "val")

    def _get_dataloader(self, dataset: Dataset, config: Dict[str, Any], loader_name: str) -> DataLoader:
        """
        Instantiate DataLoader for given dataset w.r.t to config and mode.
        It supports creating a custom sampler.
        Note: For DDP mode, we support custom samplers, but trainer must be called with:
            >>> replace_sampler_ddp=False

        Args:
           dataset: Dataset instance
            config: Dataset config
            loader_name: Loader name (train or val)

        Returns:

        """
        collate_fn = get_collate_for_dataset(dataset)

        dataset_config = config[loader_name]
        if "sampler" not in dataset_config or dataset_config["sampler"] == "none":
            sampler = None

        drop_last = loader_name == "train"

        if self.use_ddp:
            world_size = torch.distributed.get_world_size()
            local_rank = torch.distributed.get_rank()
            if sampler is None:
                sampler = DistributedSampler(dataset, world_size, local_rank)
            # else:
            #     sampler = DistributedSamplerWrapper(sampler, world_size, local_rank)

        should_shuffle = (sampler is None) and (loader_name == "train")
        batch_size = self._get_batch_size(loader_name)
        # Number of workers must not exceed batch size
        num_workers = min(batch_size, self._get_num_workers(loader_name))
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=should_shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=drop_last,
            collate_fn=collate_fn,
        )
        return loader

    def training_step(self, batch: Dict[str, Any], batch_nb: int) -> Dict[str, Any]:
        batch = any2device(batch, self.device)
        return self._step_fn(batch, batch_nb, loader_name="train")

    def validation_step(self, batch: Dict[str, Any], batch_nb: int) -> Dict[str, Any]:
        batch = any2device(batch, self.device)
        return self._step_fn(batch, batch_nb, loader_name="valid")

    def _get_optim(self, model: torch.nn.Module, optimizer_config: Dict[str, Any]) -> torch.optim.Optimizer:
        """Creates model optimizer from Trainer config
        Args:
            params (list): list of named model parameters to be trained
        Returns:
            torch.optim.Optimizer: model optimizer
        """
        if self.learning_rate:
            optimizer_config["lr"] = self.learning_rate
        optimizer = get_optimizer(model, optimizer_config=optimizer_config)
        return optimizer

    def _get_scheduler(self, optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
        """Creates scheduler for a given optimizer from Trainer config
        Args:
            optimizer (torch.optim.Optimizer): optimizer to be updated
        Returns:
            torch.optim.lr_scheduler._LRScheduler: optimizer scheduler
        """
        scheduler_config = self.config["scheduler"]
        scheduler = get_scheduler(optimizer, scheduler_config)
        return {"scheduler": scheduler, "monitor": self.config.get("metric_to_monitor", "valid/loss")}

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler._LRScheduler]]:
        self.optimizer = self._get_optim(self.model, self.config["optimizer"])
        scheduler = self._get_scheduler(self.optimizer)
        return [self.optimizer], [scheduler]

    def on_epoch_end(self) -> None:
        self.epoch_num += 1

    def on_pretrain_routine_start(self) -> None:
        if not isinstance(self.logger, DummyLogger):
            for logger in self.logger:
                if isinstance(logger, TensorBoardLogger):
                    self.tensorboard_logger = logger

    def on_pretrain_routine_end(self) -> None:
        optimizers = self.optimizers()
        if isinstance(optimizers, torch.optim.Optimizer):
            optimizers = [optimizers]
        if optimizers is None or len(optimizers) == 0:
            raise RuntimeError("List of optimizers is not available on the start of the training")
        self._initial_learning_rates = [[float(pg["lr"]) for pg in opt.param_groups] for opt in optimizers]

    def _learning_rate(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def optimizer_step(
        self,
        epoch: int = None,
        batch_idx: int = None,
        optimizer: torch.optim.Optimizer = None,
        optimizer_idx: int = None,
        optimizer_closure: Optional[Callable] = None,
        on_tpu: bool = None,
        using_native_amp: bool = None,
        using_lbfgs: bool = None,
    ) -> None:
        # Learning rate warmup
        num_warmup_steps = int(self.config.get("scheduler", {}).get("warmup_steps", 0))

        if self.trainer.global_step < num_warmup_steps:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / num_warmup_steps)
            optimizer = typing.cast(torch.optim.Optimizer, optimizer)
            optimizer_idx = optimizer_idx if optimizer_idx is not None else 0
            for pg_index, pg in enumerate(optimizer.param_groups):
                pg["lr"] = lr_scale * self._initial_learning_rates[optimizer_idx][pg_index]

        return super().optimizer_step(
            epoch=epoch,
            batch_idx=batch_idx,
            optimizer=optimizer,
            optimizer_idx=optimizer_idx,
            optimizer_closure=optimizer_closure,
            on_tpu=on_tpu,
            using_native_amp=using_native_amp,
            using_lbfgs=using_lbfgs,
        )

    def _get_keypoints_2d(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        if OUTPUT_2D_LANDMARKS in outputs.keys():
            return outputs[OUTPUT_2D_LANDMARKS] * self._img_size
        return float(self.stride) * unravel_index(outputs[OUTPUT_LANDMARKS_HEATMAP]).flip(-1)

    def _step_fn(self, batch: Dict[str, Any], batch_nb: int, loader_name: str):
        images, targets = self.get_input(batch)
        outputs = self.forward(images)
        total_loss, loss_dict = self.criterion(outputs, targets, self.epoch_num)

        process_2d_branch = OUTPUT_2D_LANDMARKS in outputs.keys() or OUTPUT_LANDMARKS_HEATMAP in outputs.keys()

        if process_2d_branch:
            self.log(
                f"{loader_name}/metrics/heatmap_iou",
                self.iou_metric(
                    outputs[OUTPUT_LANDMARKS_HEATMAP].sigmoid(),
                    targets[TARGET_LANDMARKS_HEATMAP],
                ),
                on_epoch=True,
            )

            outputs_2d = self._get_keypoints_2d(outputs=outputs) * targets[TARGET_2D_LANDMARKS_PRESENCE][..., None]
            targets_2d = (
                targets[TARGET_2D_LANDMARKS] * targets[TARGET_2D_LANDMARKS_PRESENCE][..., None] * self._img_size
            )
            metrics_2d = self.metrics_2d(outputs_2d, {"keypoints": targets_2d, "bboxes": targets[INPUT_BBOX_KEY]})
            for metric_name, metric_value in metrics_2d.items():
                self.log(
                    f"{loader_name}/metrics/{metric_name}",
                    metric_value,
                    on_epoch=True,
                )

        params_3dmm = outputs[OUTPUT_3DMM_PARAMS]
        projected_vertices = self.head_mesh.reprojected_vertices(params_3dmm=params_3dmm, to_2d=True)
        reprojected_pred = projected_vertices[:, self.flame_indices["face"]]
        reprojected_gt = targets[TARGET_2D_FULL_LANDMARKS][:, self.flame_indices["face"]]
        reprojected_metrics = self.metrics_reprojection(
            reprojected_pred, {"keypoints": reprojected_gt, "bboxes": targets[INPUT_BBOX_KEY]}
        )

        for metric_name, metric_value in reprojected_metrics.items():
            self.log(
                f"{loader_name}/metrics/{metric_name}",
                metric_value,
                on_epoch=True,
            )

        pred_3d_vertices = self.head_mesh.vertices_3d(params_3dmm=params_3dmm, zero_rotation=True)
        metrics_3d = self.metrics_3d(
            normalize_to_cube(pred_3d_vertices[:, self.flame_indices["face"]]),
            {
                "keypoints": normalize_to_cube(
                    targets[TARGET_3D_MODEL_VERTICES][:, self.flame_indices["face"]]
                )
            },
        )

        for metric_name, metric_value in metrics_3d.items():
            self.log(
                f"{loader_name}/metrics/{metric_name}",
                metric_value,
                on_epoch=True,
            )

        # Logging
        self.log(f"{loader_name}/loss", total_loss, prog_bar=True, sync_dist=self.use_ddp)
        if len(loss_dict):
            self.log_dict(
                dictionary=dict((f"{loader_name}/" + k, v) for k, v in loss_dict.items()),
                prog_bar=True,
                sync_dist=self.use_ddp,
            )
        return {"loss": total_loss}

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()

    def on_train_epoch_end(self, unused: Optional[Any] = None) -> None:
        super().on_train_epoch_end()
        learning_rate = self._learning_rate()
        self.log("train/learning_rate", learning_rate)

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()

    def export_jit_model(self, checkpoint_filename: str) -> torch.jit.ScriptModule:
        """
        Loads weighs of the model from given checkpoint into self.model and
        exports it via torch tracing.

        Note: If you don't want to export model, override this method and throw NotImplementedError

        Args:
            checkpoint_filename: Best checkpoint

        Returns:
            Instance of ScriptModule
        """

        load_from_lighting(self.model, checkpoint_filename)
        model_config = self.config["model"]["model_config"]
        example_input = torch.randn(1, model_config["num_channels"], model_config["img_size"], model_config["img_size"])
        return torch.jit.trace(self.model.eval().cuda(), example_input.cuda(), strict=False)

    def _build_loss(self, config: Dict) -> torch.nn.Module:
        return LossModule.from_config(config)


def get_collate_for_dataset(dataset: Union[Dataset, ConcatDataset]) -> Callable:
    """
    Returns collate_fn function for dataset. By default, default_collate returned.
    If the dataset has method get_collate_fn() we will use it's return value instead.
    If the dataset is ConcatDataset, we will check whether all get_collate_fn() returns
    the same function.

    Args:
       dataset: Input dataset

    Returns:
        Collate function to put into DataLoader
    """
    collate_fn = dataset.get_collate_fn()

    if isinstance(dataset, ConcatDataset):
        collates = [get_collate_for_dataset(ds) for ds in dataset.datasets]
        if len(set(collates)) != 1:
            raise ValueError("Datasets have different collate functions")
        collate_fn = collates[0]
    return collate_fn
