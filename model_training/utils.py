from typing import Dict, Any, List, Tuple
import os
import yaml
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf, DictConfig

import numpy as np

import logging, coloredlogs


COLOREDLOGS_FIELD_STYLES = coloredlogs.DEFAULT_FIELD_STYLES
COLOREDLOGS_FIELD_STYLES.update(
    {
        "asctime": {"color": "green"},
        "filename": {"color": "green"},
        "fileno": {"color": "green"},
    }
)


def create_logger(
    name: str,
    msg_format: str = "",
) -> logging.Logger:
    msg_format = msg_format or "%(asctime)s %(hostname)s %(name)s %(levelname)s - %(message)s - %(filename)s:%(lineno)d"
    logger = logging.Logger(name)
    console_handler = logging.StreamHandler()
    level = logging.DEBUG if os.environ.get("DEBUG") else logging.INFO
    console_handler.setLevel(level)
    logger.addHandler(console_handler)
    coloredlogs.install(
        level=level,
        logger=logger,
        field_styles=COLOREDLOGS_FIELD_STYLES,
        fmt=msg_format,
    )

    return logger


logger = create_logger(__name__)


def load_hydra_config(hydra_config: DictConfig) -> dict:
    """
    Load hydra config and returns ready-to-use dict.

    Notes:
        This function also restores current working directory (Hydra change it internally)

    Args:
        hydra_config:

    Returns:

    """
    os.chdir(get_original_cwd())
    return yaml.load(OmegaConf.to_yaml(hydra_config, resolve=True), Loader=yaml.FullLoader)


def load_2d_indices(config: Dict[str, Any]) -> List[int]:
    """
    Supports both folders with .npy files with keypoints, and .npz files.

    Config is expected to have "2d_subset_path" key with either folder path or a file path.
    """

    if config["2d_subset_name"] == "multipie_keypoints":
        return None
    indices = []
    subset = get_list_of_npy_files(config)
    for filename in sorted(subset):
        if os.path.exists(filename):
            indices += load_indices_from_npy(filename)
        else:
            raise ValueError(f"[{filename.split('.')[0].split('/')[-1]}] class of keypoints doesn't exist")
    return indices


def get_list_of_npy_files(config: Dict[str, Any]) -> List[str]:
    subset_path = str(config.get("2d_subset_path"))
    subset = config.get("2d_keys", "all")
    exclude = config.get("2d_keys_exclude", "cheeks")

    files = os.listdir(subset_path)
    if isinstance(subset, str) and subset == "all":
        subset = [x.split(".")[0] for x in files]
        if exclude is not None:
            if isinstance(exclude, str):
                exclude = [exclude]
            for feature in exclude:
                if feature in subset:
                    subset.remove(feature)
        subset = [os.path.join(subset_path, x + ".npy") for x in subset]
    return subset


def load_indices_from_npy(filepath: str) -> List[int]:
    # Expected to have Ordered Dict here.
    data = np.load(filepath, allow_pickle=True)[()]
    lst = []
    for value in data.values():
        lst += list(value)
    return lst


def indices_reweighing(weights_and_indices: Dict[str, Any]) -> Tuple[List, List]:
    weights_dict = weights_and_indices["weights"]
    weights = []
    indices = []

    for key, value in weights_and_indices["flame_indices"]["files"].items():
        if key in weights_dict.keys():
            indices.append(np.load(os.path.join(weights_and_indices["flame_indices"]["folder"], value)))
            weights.append(weights_dict[key])
    return weights, indices