from typing import Tuple, Any, Union
import torch
import numpy as np
import cv2
from skimage.io import imread as sk_imread
import pickle
from smplx.lbs import find_dynamic_lmk_idx_and_bcoords
from smplx.utils import Struct

from model_training.model.flame import ROT_COEFFS, JAW_COEFFS, EYE_COEFFS, NECK_COEFFS
from model_training.utils import create_logger
from utils import get_relative_path


logger = create_logger(__name__)


def read_as_rgb(x: str) -> np.ndarray:
    """
    Read image from the disk and returns 24bpp RGB image (Channel order is R-G-B)

    Args:
        x:  Image Filepath

    Returns:
        Numpy array of [H,W,3] shape
    """
    img = cv2.imread(x, cv2.IMREAD_COLOR)
    if img is None:
        logger.warning(f"Can not read image {x} with OpenCV, switching to scikit-image")
        img = sk_imread(x)[:, :, 0:3]
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def pointwise_gaussian_2d() -> np.ndarray:
    pos_kernel = np.float32([[0.5, 0.75, 0.5], [0.75, 1.0, 0.75], [0.5, 0.75, 0.5]])
    return pos_kernel


def gaussian_2d(shape: Tuple[int, int], sigma: float = 1.0) -> np.ndarray:
    m, n = [int((ss - 1.0) / 2.0) for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap: np.ndarray, center: np.array, radius: Any, k: int = 1) -> np.ndarray:
    if radius == "pointwise":
        gaussian = pointwise_gaussian_2d()
        radius = 1
    else:
        diameter = 2 * radius + 1
        gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[radius - top : radius + bottom, radius - left : radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def extend_bbox(bbox: np.array, offset: Union[Tuple[float, ...], float] = 0.1) -> np.array:
    """
    Increases bbox dimensions by offset*100 percent on each side.

    IMPORTANT: Should be used with ensure_bbox_boundaries, as might return negative coordinates for x_new, y_new,
    as well as w_new, h_new that are greater than the image size the bbox is extracted from.

    :param bbox: [x, y, w, h]
    :param offset: (left, right, top, bottom), or (width_offset, height_offset), or just single offset that specifies
    fraction of spatial dimensions of bbox it is increased by.

    For example, if bbox is a square 100x100 pixels, and offset is 0.1, it means that the bbox will be increased by
    0.1*100 = 10 pixels on each side, yielding 120x120 bbox.

    :return: extended bbox, [x_new, y_new, w_new, h_new]
    """
    x, y, w, h = bbox

    if isinstance(offset, tuple):
        if len(offset) == 4:
            left, right, top, bottom = offset
        elif len(offset) == 2:
            w_offset, h_offset = offset
            left = right = w_offset
            top = bottom = h_offset
    else:
        left = right = top = bottom = offset

    return np.array([x - w * left, y - h * top, w * (1.0 + right + left), h * (1.0 + top + bottom)]).astype("int32")


def ensure_bbox_boundaries(bbox: np.array, img_shape: Tuple[int, int]) -> np.array:
    """
    Trims the bbox not the exceed the image.
    :param bbox: [x, y, w, h]
    :param img_shape: (h, w)
    :return: trimmed to the image shape bbox
    """
    x1, y1, w, h = bbox
    x1, y1 = min(max(0, x1), img_shape[1]), min(max(0, y1), img_shape[0])
    x2, y2 = min(max(0, x1 + w), img_shape[1]), min(max(0, y1 + h), img_shape[0])
    w, h = x2 - x1, y2 - y1
    return np.array([x1, y1, w, h]).astype("int32")


# region 68 landmarks

def mesh_points_by_barycentric_coordinates(
    mesh_verts: torch.Tensor, mesh_faces: torch.Tensor, lmk_face_idx: torch.Tensor, lmk_b_coords: torch.Tensor
) -> torch.Tensor:
    # function: evaluation 3d points given mesh and landmark embedding
    # modified from https://github.com/Rubikplayer/flame-fitting/blob/master/fitting/landmarks.py
    dif1 = torch.vstack(
        [
            (mesh_verts[mesh_faces[lmk_face_idx], 0] * lmk_b_coords).sum(axis=1),
            (mesh_verts[mesh_faces[lmk_face_idx], 1] * lmk_b_coords).sum(axis=1),
            (mesh_verts[mesh_faces[lmk_face_idx], 2] * lmk_b_coords).sum(axis=1),
        ]
    ).T
    return dif1


def get_static_lmks(
    mesh_vertices: torch.Tensor,
    mesh_faces: torch.Tensor,
    flame_static_embedding_path: str = get_relative_path("../model/static/flame_static_embedding.pkl", __file__)
) -> torch.Tensor:
    """
    mesh_vertices: torch.Tensor [N, 3]
    mesh_faces: torch.Tensor [M, 3]
    """
    with open(flame_static_embedding_path, "rb") as f:
        static_embeddings = Struct(**pickle.load(f, encoding="latin1"))

    lmk_faces_idx = torch.LongTensor(static_embeddings.lmk_face_idx.astype(np.int64))
    lmk_bary_coords = torch.Tensor(static_embeddings.lmk_b_coords)

    return mesh_points_by_barycentric_coordinates(mesh_vertices, mesh_faces, lmk_faces_idx, lmk_bary_coords)


def get_dynamic_lmks(
    mesh_vertices: torch.Tensor,
    mesh_faces: torch.Tensor,
    contour_embeddings_path: str = get_relative_path("../model/static/flame_dynamic_embedding.npy", __file__)
) -> torch.Tensor:
    """
    mesh_vertices: torch.Tensor [N, 3]
    mesh_faces: torch.Tensor [M, 3]
    """
    # Source: https://github.com/soubhiksanyal/FLAME_PyTorch/blob/master/FLAME.py
    conture_embeddings = np.load(contour_embeddings_path, allow_pickle=True, encoding="latin1")[()]

    dynamic_lmk_faces_idx = torch.LongTensor(np.array(conture_embeddings["lmk_face_idx"]).astype(np.int64))
    dynamic_lmk_bary_coords = torch.Tensor(np.array(conture_embeddings["lmk_b_coords"]))

    parents = torch.LongTensor([-1, 0, 1, 1, 1])

    neck_kin_chain_list = []
    curr_idx = torch.tensor(1, dtype=torch.long)
    while curr_idx != -1:
        neck_kin_chain_list.append(curr_idx)
        curr_idx = parents[curr_idx]
    neck_kin_chain = torch.stack(neck_kin_chain_list)

    # Zero pose: torch.zeros(1, 15, device=mesh_vertices.device)
    dyn_lmk_faces_idx, dyn_lmk_bary_coords = find_dynamic_lmk_idx_and_bcoords(
        mesh_vertices.view(1, -1, 3),
        torch.zeros(1, ROT_COEFFS + JAW_COEFFS + NECK_COEFFS + EYE_COEFFS, device=mesh_vertices.device),
        dynamic_lmk_faces_idx,
        dynamic_lmk_bary_coords,
        neck_kin_chain,
    )
    return mesh_points_by_barycentric_coordinates(
        mesh_vertices, mesh_faces, dyn_lmk_faces_idx[0], dyn_lmk_bary_coords[0]
    )


def get_68_landmarks(
    mesh_vertices: torch.Tensor,
    mesh_faces_path: str = get_relative_path("../model/static/flame_mesh_faces.pt", __file__)
) -> torch.Tensor:
    """
    mesh_vertices: torch.Tensor [N, 3]

    Returns [68, 3].
    """

    assert mesh_vertices.ndim == 2
    assert mesh_vertices.shape == (5023, 3)

    mesh_faces = torch.load(mesh_faces_path)
    static_lmks = get_static_lmks(mesh_vertices, mesh_faces)
    dynamic_lmks = get_dynamic_lmks(mesh_vertices, mesh_faces)
    return torch.cat((dynamic_lmks, static_lmks), 0)

#endregion