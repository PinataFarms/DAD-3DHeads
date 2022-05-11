from typing import Dict, Any

import numpy as np
from model_training.data.utils import draw_gaussian


class HeatmapCoder:
    def __init__(self, data_config: Dict[str, Any], num_classes: int):
        super().__init__()
        self.num_classes = num_classes

        self._img_size = data_config["img_size"]
        self._radius = data_config.get("radius", 5)
        self._stride = data_config.get("stride", 2)

    def __call__(self, keypoints: np.ndarray, presence: np.ndarray) -> np.ndarray:
        size = self._img_size // self._stride
        heatmap = np.zeros((self.num_classes, size, size), dtype=np.float32)
        for index, point in enumerate(keypoints):
            if presence[index]:
                point = point // self._stride
                heatmap[index] = draw_gaussian(heatmap[index], point, self._radius)
        return heatmap
