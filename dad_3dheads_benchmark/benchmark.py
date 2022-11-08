from typing import List, Dict, Any
import os
from collections import namedtuple
import torch
import numpy as np
from utils import get_68_landmarks, read_img, read_json, calc_ch_dist, get_flame_indices
from fire import Fire
import tqdm

MeshArrays = namedtuple(
    "MeshArrays",
    ["vertices3d", "vertices3d_world_homo", "projection_matrix", "model_view_matrix"],
)


class HeadAnnotation:
    """Class for storing a head annotation."""

    def __init__(self, id: str, mesh: MeshArrays, bbox: List[int], image_height: int):
        self.id = id
        self.mesh = mesh
        self.bbox = bbox
        self.image_height = image_height

    def landmarks_68_2d(self, height: int) -> np.ndarray:
        landmarks = get_68_landmarks(torch.from_numpy(self.mesh.vertices3d).view(-1, 3)).numpy()
        landmarks = np.concatenate((landmarks, np.ones_like(landmarks[:, [0]])), -1)
        # rotated and translated (to world coordinates)
        landmarks = np.transpose(np.matmul(self.mesh.model_view_matrix, np.transpose(landmarks)))
        landmarks = np.transpose(np.matmul(self.mesh.projection_matrix, np.transpose(landmarks)))
        landmarks = landmarks[:, :2] / landmarks[:, [3]]
        landmarks = np.stack((landmarks[:, 0], (height - landmarks[:, 1])), -1)
        return landmarks

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "HeadAnnotation":
        flame_vertices3d = np.array(config["vertices"], dtype=np.float32)
        model_view_matrix = np.array(config["model_view_matrix"], dtype=np.float32)
        flame_vertices3d_homo = np.concatenate((flame_vertices3d, np.ones_like(flame_vertices3d[:, [0]])), -1)
        # rotated and translated (to world coordinates)
        flame_vertices3d_world_homo = np.transpose(np.matmul(model_view_matrix, np.transpose(flame_vertices3d_homo)))
        mesh = MeshArrays(
            vertices3d=flame_vertices3d,
            vertices3d_world_homo=flame_vertices3d_world_homo,  # with pose and translation
            projection_matrix=np.array(config["projection_matrix"], dtype=np.float32),
            model_view_matrix=np.array(config["model_view_matrix"], dtype=np.float32),
        )

        return cls(
            id=config["id"],
            mesh=mesh,
            bbox=config["bbox"],
            image_height=config["image_height"]
        )


class DADEvaluator:
    def __init__(self, ground_truth_path, submission_path):
        self.target_file_path = ground_truth_path
        self.prediction_file_path = submission_path
        self.head_indices = get_flame_indices()

    def _get_data(self, anno_file: str) -> List[HeadAnnotation]:
        data = read_json(anno_file)
        return [HeadAnnotation.from_config(config=anno) for anno in data]

    @staticmethod
    def get_gt_rot_mat(annotation):
        rot_180 = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        mv = rot_180 @ annotation.mesh.model_view_matrix
        R_gt = mv[:3, :3]
        return R_gt

    def pose_error(self, annotation: HeadAnnotation, prediction: Dict[str, np.ndarray]):
        R_predicted = np.array(prediction['rotation_matrix'], dtype=np.float32)
        R_gt = self.get_gt_rot_mat(annotation)
        R_dist = R_predicted @ R_gt.T
        rot_error = np.linalg.norm(np.eye(3) - R_dist, "fro")
        return rot_error

    def nme(self, annotation: HeadAnnotation, prediction: Dict[str, np.ndarray]):
        predicted_68_2d_landmarks = np.array(prediction['68_landmarks_2d'], dtype=np.float32)
        gt_68_2d_landmarks = annotation.landmarks_68_2d(height=annotation.image_height)

        nme_2d = (
                float(
                    np.mean(
                        np.linalg.norm(gt_68_2d_landmarks - predicted_68_2d_landmarks, 2, -1)
                        / (np.sqrt(annotation.bbox[2] * annotation.bbox[3]))
                    )
                )
                * 100.0
        )
        return nme_2d

    def chamfer_distance(self, annotation: HeadAnnotation, prediction: Dict[str, np.ndarray]):
        gt_vertices_3d = torch.from_numpy(annotation.mesh.vertices3d_world_homo[:, :3]).view(-1, 3)
        pred_vertices_3d = torch.Tensor(prediction['N_landmarks_3d']).view(-1, 3)
        seven_vertices_3d = np.array(prediction['7_landmarks_3d'], dtype=np.float32).reshape(-1, 3)
        chamfer = calc_ch_dist(
            gt_vertices_3d, pred_vertices_3d, svn_pred_lmks=seven_vertices_3d
        )
        return chamfer

    @staticmethod
    def calc_zn(pred_landmarks: torch.Tensor, gt_landmarks: torch.Tensor, top_k: int = 5) -> float:
        """
        Pred_landmarks and gt_landmarks must have same number of points,
        and the meshes they correspond to are assumed to have the same topology.

        pred_landmarks: [B, N, 3]
        gt_landmarks: [B, N, 3]

        In our setup, we expect here subsample upon head_indices to arrive.
        """
        result = 0
        iterations = 0
        for sl in range(gt_landmarks.shape[0]):
            distances = torch.cdist(gt_landmarks[sl, ...], gt_landmarks[sl, ...])
            sorted_distances = torch.argsort(distances, dim=0)

            index_to_compare = sorted_distances[:, 1: top_k + 1]

            result_tmp = torch.zeros(sorted_distances.shape[0], top_k)
            for i in range(sorted_distances.shape[0]):
                for j in range(top_k):
                    result_tmp[i, j] = (gt_landmarks[sl, i, 2] >= gt_landmarks[sl, index_to_compare[i, j], 2]) == (
                            pred_landmarks[sl, i, 2] >= pred_landmarks[sl, index_to_compare[i, j], 2]
                    )

            result += torch.mean(result_tmp).data.cpu().numpy()
            iterations += 1
        return result / iterations

    def zn(self, annotation: HeadAnnotation, prediction: Dict[str, np.ndarray], n=5):
        gt_vertices_3d = torch.from_numpy(annotation.mesh.vertices3d_world_homo[:, :3]).view(-1, 3)
        pred_vertices_3d = torch.Tensor(prediction['N_landmarks_3d']).view(-1, 3)
        gt_vertices_3d_head = gt_vertices_3d[self.head_indices] * -1
        pred_vertices_head = pred_vertices_3d[self.head_indices]

        z_n = self.calc_zn(
            pred_vertices_head.view(1, -1, 3),
            gt_vertices_3d_head.view(1, -1, 3),
            top_k=n
        )
        return z_n

    def __call__(self):
        submission_dict = read_json(self.prediction_file_path)
        ground_truth = self._get_data(self.target_file_path)

        pose_error_list = []
        nme_list = []
        z5_list = []
        chamfer_list = []
        for annotation in tqdm.tqdm(ground_truth):
            try:
                prediction = submission_dict[annotation.id]
                pose_error_el = self.pose_error(annotation, prediction)
                nme_el = self.nme(annotation, prediction)
                z5_el = self.zn(annotation, prediction, n=5)
                chamfer_el = self.chamfer_distance(annotation, prediction)
                pose_error_list.append(pose_error_el)
                nme_list.append(nme_el)
                z5_list.append(z5_el)
                chamfer_list.append(chamfer_el)
            except:
                print(f'No prediction with ID: {annotation.id}.')

        result = {
            "nme_reprojection": np.array(nme_list).mean(),
            "z5_accuracy": np.array(z5_list).mean(),
            "chamfer": np.array(chamfer_list).mean(),
            "pose_error": np.array(pose_error_list).mean(),
        }

        return result


def evaluate(
        submission_path: str = 'data/sample_submission.json',
        gt_path: str = 'data/ground_truth_val.json',
):
    evaluator = DADEvaluator(gt_path, submission_path)
    result = evaluator()
    print(result)


if __name__ == "__main__":
    Fire(evaluate)
