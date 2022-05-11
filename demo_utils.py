import os
from typing import Dict, Any, Tuple, List
import numpy as np
import torch
import cv2
import json

from model_training.model.flame import calculate_rpy, FlameParams, FLAME_CONSTS
from model_training.utils import load_indices_from_npy
from utils import get_relative_path

# region visualization
POINT_COLOR = (255, 0, 0)
EDGE_COLOR = (39, 48, 218)
OPACITY = .6
KEYPOINTS_INDICES_DIR = "model_training/model/static/face_keypoints"
FLAME_IDICES_DIR = "model_training/model/static/flame_indices/"


def draw_points(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Points are expected to have integer coordinates.
    """
    radius = max(1, int(min(image.shape[:2]) * 0.005))
    for pt in points:
        cv2.circle(image, (pt[0], pt[1]), radius, POINT_COLOR, -1)
    return image


def draw_landmarks(predictions: Dict[str, torch.Tensor], image: np.ndarray) -> np.ndarray:
    image = draw_points(image, predictions["points"])
    return image


def draw_3d_landmarks(predictions: Dict[str, torch.Tensor], image: np.ndarray, subset: str = "191") -> np.ndarray:
    if subset != "191" and subset != "445":
        ValueError("Invalid keypoints subset provided.\n"
                   "Available options are: 191, 445")
    subset_dir = get_relative_path(os.path.join(KEYPOINTS_INDICES_DIR, f"keypoints_{subset}"), __file__)
    projected_vertices = predictions["projected_vertices"].squeeze().numpy().astype(int)
    points = []
    for subs_file in os.listdir(subset_dir):
        subs_file_path = os.path.join(subset_dir, subs_file)
        points.extend(np.take(projected_vertices, load_indices_from_npy(subs_file_path), axis=0))
    return draw_points(image, points)


def draw_mesh(predictions: Dict[str, torch.Tensor], image: np.ndarray, subset: str = "head") -> np.ndarray:
    if subset != "face" and subset != "head":
        ValueError("Invalid FLAME mesh vertices subset provided.\n"
                   "Available options are: face, head")

    mesh_vis = image.copy()
    output = image.copy()
    projected_vertices = predictions["projected_vertices"].squeeze().numpy().astype(int)
    edges = np.load(get_relative_path(os.path.join(FLAME_IDICES_DIR, f"{subset}_edges.npy"), __file__))

    for edge in edges:
        pt1, pt2 = edge[0], edge[1]
        cv2.line(mesh_vis, projected_vertices[pt1], projected_vertices[pt2], EDGE_COLOR, 1, cv2.LINE_AA)

    cv2.addWeighted(mesh_vis, OPACITY, output, 1 - OPACITY, 0, output)
    return mesh_vis


def draw_pose(predictions: Dict[str, torch.Tensor], image: np.ndarray) -> np.ndarray:
    params_3dmm = predictions["3dmm_params"].float()
    flame_params = FlameParams.from_3dmm(params_3dmm, FLAME_CONSTS)
    rpy = calculate_rpy(flame_params)

    tdx, tdy = image.shape[1] // 2, image.shape[0] // 2

    roll = rpy.roll * np.pi / 180
    pitch = rpy.pitch * np.pi / 180
    yaw = -(rpy.yaw * np.pi / 180)

    size = image.shape[0] // 10

    x1 = size * (np.cos(yaw) * np.cos(roll)) + tdx
    y1 = size * (np.cos(pitch) * np.sin(roll) + np.cos(roll) * np.sin(pitch) * np.sin(yaw)) + tdy

    x2 = size * (-np.cos(yaw) * np.sin(roll)) + tdx
    y2 = size * (np.cos(pitch) * np.cos(roll) - np.sin(pitch) * np.sin(yaw) * np.sin(roll)) + tdy

    x3 = size * (np.sin(yaw)) + tdx
    y3 = size * (-np.cos(yaw) * np.sin(pitch)) + tdy

    cv2.arrowedLine(image, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), int(image.shape[0] * 0.005))
    cv2.arrowedLine(image, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), int(image.shape[0] * 0.005))
    cv2.arrowedLine(image, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), int(image.shape[0] * 0.005))

    return image


# endregion

def get_mesh(predictions: Dict[str, torch.Tensor], *args: Any) -> Tuple[np.ndarray, np.ndarray]:
    vertices = predictions['3d_vertices'].numpy()
    faces = torch.load('model_training/model/static/flame_mesh_faces.pt').numpy() + 1.
    return vertices, faces


def get_flame_params(predictions: Dict[str, torch.Tensor], *args: Any) -> Dict[str, List[float]]:
    params_3dmm = predictions['3dmm_params']
    flame_params = FlameParams.from_3dmm(params_3dmm, FLAME_CONSTS)
    result = {k: v[0].tolist() for k, v in vars(flame_params).items()}
    return result


# region saving
class ImageSaver:
    def __init__(self) -> None:
        self.extension = '.png'

    def __call__(self, image: np.ndarray, output_path: str) -> None:
        cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


class MeshSaver:
    def __init__(self) -> None:
        self.extension = '.obj'

    def __call__(self, mesh: Tuple[np.ndarray, np.ndarray], output_path: str) -> None:
        """
        mesh: tuple (vertices, faces)
        Vertices: [N, 3]. Faces: [N, 3], 1st vertex has index '1', not '0'.
        """
        vertices, faces = mesh
        with open(output_path, 'w') as f:
            for vertex in vertices:
                f.write(f'v %.8f %.8f %.8f\n' % tuple(vertex))
            for face in faces:
                f.write('f %d %d %d\n' % tuple(face))


class JsonSaver:
    def __init__(self) -> None:
        self.extension = '.json'

    def __call__(self, flame_params: Dict[str, List[float]], output_path: str) -> None:
        with open(output_path, "w") as out:
            json.dump(flame_params, out)


def get_output_path(input_image_path: str, outputs_folder: str, type_of_output: str, extension: str) -> str:
    """
    Returns the output_path in outputs_folder that matches input_image_filename(tail), extended with the 'type of
    output' substring, with corresponding extension.
    """
    input_filename = os.path.split(input_image_path)[1]
    output_path_wo_ext = os.path.join(
        outputs_folder,
        f'{os.path.splitext(input_filename)[0]}_{type_of_output}'
    )
    return output_path_wo_ext + extension
# endregion
