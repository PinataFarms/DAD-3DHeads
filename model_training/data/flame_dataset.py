import os
import json
from typing import Dict, Any, List, Union, Tuple, Optional
from collections import namedtuple
import torch
from torch.utils.data import Dataset
import numpy as np
from hydra.utils import instantiate
import albumentations as A
import pytorch_toolbelt.utils as pt_utils

from model_training.data.config import (
    IMAGE_FILENAME_KEY,
    SAMPLE_INDEX_KEY,
    INPUT_IMAGE_KEY,
    INPUT_BBOX_KEY,
    INPUT_SIZE_KEY,
    TARGET_PROJECTION_MATRIX,
    TARGET_3D_MODEL_VERTICES,
    TARGET_3D_WORLD_VERTICES,
    TARGET_2D_LANDMARKS,
    TARGET_LANDMARKS_HEATMAP,
    TARGET_2D_FULL_LANDMARKS,
    TARGET_2D_LANDMARKS_PRESENCE,
)
from model_training.data.transforms import get_resize_fn, get_normalize_fn
from model_training.data.utils import ensure_bbox_boundaries, extend_bbox, read_as_rgb, get_68_landmarks
from model_training.utils import load_2d_indices, create_logger

MeshArrays = namedtuple(
    "MeshArrays",
    ["vertices3d", "vertices3d_world_homo", "projection_matrix"],
)

logger = create_logger(__name__)


def collate_skip_none(batch: Any) -> Any:
    len_batch = len(batch)
    batch = list(filter(lambda x: x is not None, batch))
    if len_batch > len(batch):
        diff = len_batch - len(batch)
        batch = batch + batch[:diff]
    return torch.utils.data.dataloader.default_collate(batch)


class FlameDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]], config: Dict[str, Any]) -> None:
        self.data = data
        self.config = config

        self.img_size = config["img_size"]
        self.filename_key = "img_path"
        self.aug_pipeline = self._get_aug_pipeline(config["transform"])

        self.num_classes = config.get("num_classes")
        self.keypoints_indices = load_2d_indices(config["keypoints"])
        self.tensor_keys = [INPUT_IMAGE_KEY]
        self.coder = instantiate(config["coder"], config, self.num_classes)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        item_anno = self._get_item_anno(idx=idx)
        item_data = self._parse_anno(item_anno)
        item_data = self._transform(item_data)
        item_dict = self._form_anno_dict(item_data)
        item_dict = self._add_index(idx, item_anno, item_dict)
        item_dict = self._convert_images_to_tensors(item_dict)
        return item_dict

    def _add_index(self, idx: int, annotation: Any, item_dict: Dict[str, Any]) -> Dict[str, Any]:
        if item_dict is not None:
            item_dict.update({SAMPLE_INDEX_KEY: idx, IMAGE_FILENAME_KEY: annotation[self.filename_key]})
        return item_dict

    def _get_item_anno(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        with open(config["ann_path"]) as json_file:
            anno = json.load(json_file)
        return cls(data=anno, config=config)

    def _convert_images_to_tensors(self, item_data: Dict[str, Any]) -> Dict[str, Any]:
        if item_data is not None:
            for key, item in item_data.items():
                if isinstance(item, np.ndarray) and key in self.tensor_keys:
                    item_data[key] = pt_utils.image_to_tensor(item.astype("float32"))
        return item_data

    def _parse_anno(self, item_anno: Dict[str, Any]) -> Dict[str, Any]:
        img = read_as_rgb(os.path.join(self.config["dataset_root"], item_anno["img_path"]))
        bbox = item_anno["bbox"]
        offset = tuple(0.1 * np.random.uniform(size=4) + 0.05)
        x, y, w, h = ensure_bbox_boundaries(extend_bbox(np.array(bbox), offset), img.shape[:2])
        cropped_img = img[y : y + h, x : x + w]
        (
            flame_vertices3d,
            flame_vertices3d_world_homo,
            projection_matrix,
        ) = self._load_mesh(os.path.join(self.config["dataset_root"], item_anno["annotation_path"]))
        return {
            INPUT_IMAGE_KEY: cropped_img,
            INPUT_BBOX_KEY: (x, y, w, h),
            INPUT_SIZE_KEY: img.shape,
            TARGET_3D_MODEL_VERTICES: flame_vertices3d,
            TARGET_3D_WORLD_VERTICES: flame_vertices3d_world_homo,
            TARGET_PROJECTION_MATRIX: projection_matrix
        }

    @staticmethod
    def _load_mesh(mesh_path: str) -> MeshArrays:
        with open(mesh_path) as json_data:
            data = json.load(json_data)
        flame_vertices3d = np.array(data["vertices"], dtype=np.float32)
        model_view_matrix = np.array(data["model_view_matrix"], dtype=np.float32)
        flame_vertices3d_homo = np.concatenate((flame_vertices3d, np.ones_like(flame_vertices3d[:, [0]])), -1)
        # rotated and translated (to world coordinates)
        flame_vertices3d_world_homo = np.transpose(np.matmul(model_view_matrix, np.transpose(flame_vertices3d_homo)))
        return MeshArrays(
            vertices3d=flame_vertices3d,
            vertices3d_world_homo=flame_vertices3d_world_homo,  # with pose and translation
            projection_matrix=np.array(data["projection_matrix"], dtype=np.float32),
        )

    @staticmethod
    def _project_vertices_onto_image(
            vertices3d_world_homo: np.ndarray,
            projection_matrix: np.ndarray,
            height: int,
            crop_point_x: int,
            crop_point_y: int
    ):
        vertices2d_homo = np.transpose(np.matmul(projection_matrix, np.transpose(vertices3d_world_homo)))
        vertices2d = vertices2d_homo[:, :2] / vertices2d_homo[:, [3]]
        vertices2d = np.stack((vertices2d[:, 0], (height - vertices2d[:, 1])), -1)
        vertices2d -= (crop_point_x, crop_point_y)
        return vertices2d

    def _get_2d_landmarks_w_presence(
        self,
        vertices3d_world_homo: np.ndarray,
        projection_matrix: np.ndarray,
        img_shape: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        if self.num_classes == 68:
            landmarks_3d_world_subset = get_68_landmarks(
                torch.from_numpy(vertices3d_world_homo[..., :3]).view(-1, 3)
            ).numpy()
            landmarks_3d_world_subset = np.concatenate(
                (landmarks_3d_world_subset, np.ones_like(landmarks_3d_world_subset[:, [0]])), -1
            )
        else:
            landmarks_3d_world_subset = vertices3d_world_homo[self.keypoints_indices]
        x, y, w, h = bbox

        landmarks_2d_subset = self._project_vertices_onto_image(
            landmarks_3d_world_subset, projection_matrix, img_shape[0], x, y
        )
        keypoints_2d = self._project_vertices_onto_image(vertices3d_world_homo, projection_matrix, img_shape[0], x, y)

        presence_subset = np.array([False] * len(landmarks_2d_subset))
        for i in range(len(landmarks_2d_subset)):
            if 0 < landmarks_2d_subset[i, 0] < w and 0 < landmarks_2d_subset[i, 1] < h:
                presence_subset[i] = True
        return landmarks_2d_subset, presence_subset, keypoints_2d

    def _transform(self, item_data: Dict[str, Any]) -> Dict[str, Any]:
        vertices_2d_subset, presence_subset, vertices_2d = self._get_2d_landmarks_w_presence(
            item_data[TARGET_3D_WORLD_VERTICES],
            item_data[TARGET_PROJECTION_MATRIX],
            item_data[INPUT_SIZE_KEY],
            item_data[INPUT_BBOX_KEY],
        )

        result = self.aug_pipeline(
            image=item_data[INPUT_IMAGE_KEY], keypoints=np.concatenate((vertices_2d_subset, vertices_2d), 0)
        )

        return {
            INPUT_IMAGE_KEY: result["image"],
            INPUT_BBOX_KEY: item_data[INPUT_BBOX_KEY],
            TARGET_3D_MODEL_VERTICES: item_data[TARGET_3D_MODEL_VERTICES],
            TARGET_2D_LANDMARKS: np.array(result["keypoints"][: self.num_classes], dtype=np.float32),
            TARGET_2D_FULL_LANDMARKS: np.array(result["keypoints"][self.num_classes :], dtype=np.float32),
            TARGET_2D_LANDMARKS_PRESENCE: presence_subset
        }

    def _form_anno_dict(self, item_data: Dict[str, np.ndarray]) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        landmarks = item_data[TARGET_2D_LANDMARKS]
        presence = item_data[TARGET_2D_LANDMARKS_PRESENCE]
        heatmap = self.coder(landmarks, presence)
        item_data[TARGET_2D_LANDMARKS] = landmarks / self.img_size
        item_data[TARGET_LANDMARKS_HEATMAP] = np.uint8(255.0 * heatmap)
        return item_data

    def _get_aug_pipeline(self, aug_config: Dict[str, Any]) -> A.Compose:
        normalize = get_normalize_fn(aug_config.get("normalize", "imagenet"))
        resize = get_resize_fn(self.img_size, mode=aug_config.get("resize_mode", "longest_max_size"))
        return A.Compose(
            [resize, normalize],
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False)
        )

    def get_collate_fn(self) -> Any:
        return collate_skip_none
