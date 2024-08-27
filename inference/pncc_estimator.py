from collections import namedtuple

import Sim3DR
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict
from typing import Tuple, Union
from utils import get_relative_path

from model_training.head_mesh import HeadMesh
from model_training.model.flame import FlameParams


def pncc(
    img: np.ndarray, vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray, with_bg_flag: bool = True
) -> np.ndarray:
    """
    Render a colored 3D face mesh

    Args:
        img: Image where to render 3D face, RGB image of [H,W,3] size
        vertices: List of 3D vertices [N,3]
        faces: List of faces [N,3]
        colors: List of RGB colors for each vertex, [N,3]
        with_bg_flag: If True, paint on top of the image, otherwise - on black background.

    Returns:
        Image of shape [H,W,3] shape
    """

    def _to_ctype(arr: np.ndarray) -> np.ndarray:
        if not arr.flags.c_contiguous:
            return arr.copy(order="C")
        return arr

    if with_bg_flag:
        overlap = img.copy()
    else:
        overlap = np.zeros_like(img)
    overlap = Sim3DR.rasterize(_to_ctype(vertices), _to_ctype(faces), _to_ctype(colors), bg=overlap)
    return overlap

def compute_ncc_color_codes(template_face: np.ndarray, subset_indexes: Optional[np.ndarray] = None) -> np.ndarray:
    if not isinstance(template_face, np.ndarray):
        raise ValueError(f"Argument template_face must be a numpy array, got type {type(template_face)}")
    if len(template_face.shape) != 2 or template_face.shape[1] != 3:
        raise ValueError(f"Argument template_face must have shape [N,3], got shape {template_face.shape}")
    if subset_indexes is not None and not isinstance(subset_indexes, np.ndarray):
        raise ValueError(f"Argument subset_indexes must be a numpy array, got type {type(subset_indexes)}")

    sub_vertices = template_face[subset_indexes] if subset_indexes is not None else template_face
    u_min = sub_vertices.min(axis=0, keepdims=True, initial=0)
    u_max = sub_vertices.max(axis=0, keepdims=True, initial=0)

    def normalize_to_unit(u: np.ndarray, min: np.ndarray, max: np.ndarray) -> np.ndarray:
        return (u - min) / (max - min)

    return normalize_to_unit(template_face, u_min, u_max)


PNCCResult = namedtuple("PNCCResult", ["tight_crop", "extended_crop", "mmparams"])


class PNCCEstimator:
    def __init__(self, img_size: int = 512):
        self.img_size = img_size
        self.head_mesh = HeadMesh()

        self.faces_wo_back_remapped = np.load(get_relative_path("../model_training/model/static/flame_indices/faces_wo_ears_remapped.npy", __file__))
        self.colors = compute_ncc_color_codes(self.head_mesh.flame.flame_model.v_template, np.unique(self.faces_wo_back_remapped))

    def _transform_3dmm_to_3d_face_polygons(self, mm_params: Union[torch.Tensor, np.ndarray], flame: nn.Module) -> Tuple[np.ndarray, np.ndarray]:
        """

        Args:
            mm_params:
            constants:
            flame:
            img_size:

        Returns:
            Tuple of vertices [N,3], and polygon indexes [K,3]
        """
        with torch.no_grad():
            vertices = self.head_mesh.reprojected_vertices(mm_params, to_2d=False)
            vertices[:, :, 2] *= -1  # Invert Z direction, w/o this like NCC rasterization produces incorrect results

        triangles = flame.faces.astype(int)
        return vertices[0].cpu().numpy(), triangles

    def __call__(self, image: np.ndarray, predictions: Dict[str, Tensor], with_background: bool = False) -> np.ndarray:
        v1, t1 = self._transform_3dmm_to_3d_face_polygons(predictions["3dmm_params"],self.head_mesh.flame)
        return pncc(
            image,
            v1,
            self.faces_wo_back_remapped,
            self.colors,
            with_background,
        )
