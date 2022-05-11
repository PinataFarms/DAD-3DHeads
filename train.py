import os
from typing import Dict, Any
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from model_training.utils import load_hydra_config, create_logger
from model_training.train.trainer import DAD3DTrainer
from model_training.model import load_model
from model_training.train.flame_lightning_model import FlameLightningModel
from model_training.data import FlameDataset

logger = create_logger(__name__)

torch.autograd.set_detect_anomaly(True)


def train(config):
    train_dataset = FlameDataset.from_config(config=config["train"])
    val_dataset = FlameDataset.from_config(config=config["val"])
    model = load_model(config["model"], config["constants"])
    dad3d_net = FlameLightningModel(model=model, config=config, train=train_dataset, val=val_dataset)
    dad3d_trainer = DAD3DTrainer(dad3d_net, config)
    dad3d_trainer.fit()


def prepare_experiment(hydra_config: DictConfig) -> Dict[str, Any]:
    experiment_dir = os.getcwd()
    save_path = os.path.join(experiment_dir, "experiment_config.yaml")
    OmegaConf.set_struct(hydra_config, False)
    hydra_config["yaml_path"] = save_path
    hydra_config["experiment"]["folder"] = experiment_dir
    logger.info(OmegaConf.to_yaml(hydra_config, resolve=True))
    config = load_hydra_config(hydra_config)
    with open(save_path, "w") as f:
        OmegaConf.save(config=config, f=f.name)
    return config


@hydra.main(config_name="train", config_path="model_training/config")
def run_experiment(hydra_config: DictConfig) -> None:
    config = prepare_experiment(hydra_config)
    logger.info("Experiment dir %s" % config["experiment"]["folder"])
    train(config)


if __name__ == "__main__":
    run_experiment()
