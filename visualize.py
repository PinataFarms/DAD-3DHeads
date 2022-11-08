import json
from typing import Dict, List
import numpy as np
from fire import Fire
import cv2
import os
from demo_utils import draw_points, get_output_path


def get_2d_keypoints(data: Dict[str, List], img_height: int) -> np.ndarray:
    flame_vertices3d = np.array(data["vertices"], dtype=np.float32)
    model_view_matrix = np.array(data["model_view_matrix"], dtype=np.float32)
    projection_matrix = np.array(data["projection_matrix"], dtype=np.float32)

    flame_vertices3d_homo = np.concatenate((flame_vertices3d, np.ones_like(flame_vertices3d[:, [0]])), -1)
    flame_vertices3d_world_homo = np.transpose(np.matmul(model_view_matrix, np.transpose(flame_vertices3d_homo)))

    flame_vertices2d_homo = np.transpose(
        np.matmul(projection_matrix, np.transpose(flame_vertices3d_world_homo))
    )
    flame_vertices2d = flame_vertices2d_homo[:, :2] / flame_vertices2d_homo[:, [3]]
    return np.stack((flame_vertices2d[:, 0], (img_height - flame_vertices2d[:, 1])), -1).astype(int)


def visualize(subset: str, id: str, base_path: str = 'dataset', outputs_folder: str = "outputs") -> None:
    """
    Visualizes the 3D vertices (GT annotations) over the 2D image from the dataset.

    Args:
        subset: 'train', 'val', or 'test'
        id: unique id (filename) of the data point
        base_path: path to the 'DAD-3DHeadsDataset' folder
        outputs_folder: folder to write the outputs to

    """
    os.makedirs(outputs_folder, exist_ok=True)
    json_path = os.path.join(base_path, 'DAD-3DHeadsDataset', subset, 'annotations', id + '.json')
    img_path = json_path.replace('annotations', 'images').replace('json', 'png')

    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    with open(json_path) as json_data:
        mesh_data = json.load(json_data)

    keypoints_2d = get_2d_keypoints(mesh_data, img.shape[0])
    img = draw_points(img, keypoints_2d)

    output_filename = get_output_path(img_path, outputs_folder, 'GT_landmarks', '.png')
    cv2.imwrite(output_filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    Fire(visualize)
