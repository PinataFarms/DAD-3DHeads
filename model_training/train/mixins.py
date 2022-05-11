from typing import Dict, Any, Tuple, Union, Optional
import cv2
import torch
import numpy as np

from model_training.data.config import (
    TARGET_3D_MODEL_VERTICES,
    TARGET_2D_LANDMARKS,
    TARGET_LANDMARKS_HEATMAP,
    OUTPUT_LANDMARKS_HEATMAP,
    OUTPUT_3DMM_PARAMS,
    OUTPUT_2D_LANDMARKS,
    TARGET_2D_FULL_LANDMARKS,
    TARGET_2D_LANDMARKS_PRESENCE,
    INPUT_BBOX_KEY,
)
from model_training.model.flame import uint8_to_float32
from model_training.data import INPUT_IMAGE_KEY
from model_training.train.utils import any2device
from pytorch_toolbelt.utils.torch_utils import rgb_image_from_tensor
from pytorch_toolbelt.utils.visualization import vstack_autopad, vstack_header


class KeypointsDataMixin:
    input_key = INPUT_IMAGE_KEY
    input_type = torch.float32

    target_type = torch.float32

    def get_input(self, data: Dict[str, Any]) -> Tuple[torch.Tensor, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        inputs = any2device(data[self.input_key], self.device)
        targets = {
            key: value
            for key, value in data.items()
            if key
            in [
                TARGET_2D_LANDMARKS,
                TARGET_LANDMARKS_HEATMAP,
                TARGET_3D_MODEL_VERTICES,
                TARGET_2D_FULL_LANDMARKS,
                TARGET_2D_LANDMARKS_PRESENCE,
                INPUT_BBOX_KEY,
            ]
        }
        for key, gt_map in targets.items():
            if isinstance(gt_map, np.ndarray):
                torch.from_numpy(gt_map)
            gt_map = any2device(gt_map, self.device)
            targets[key] = gt_map
        targets[TARGET_LANDMARKS_HEATMAP] = uint8_to_float32(targets[TARGET_LANDMARKS_HEATMAP])
        targets[INPUT_BBOX_KEY] = torch.stack(targets[INPUT_BBOX_KEY]).T
        return inputs, targets


class KeypointsVisualizationMixin:
    def get_visuals(
        self,
        inputs: Dict[str, torch.Tensor],
        outputs: Dict[str, Any],
        score: float,
        max_images: Optional[int] = None,
    ) -> np.ndarray:
        process_2d_branch = OUTPUT_2D_LANDMARKS in outputs.keys() or OUTPUT_LANDMARKS_HEATMAP in outputs.keys()
        flame_params = outputs[OUTPUT_3DMM_PARAMS].detach()
        if process_2d_branch:
            heatmaps = outputs[OUTPUT_LANDMARKS_HEATMAP].detach()
        images = inputs[self.input_key]
        projected_vertices = self.head_mesh.reprojected_vertices(
            params_3dmm=flame_params, to_2d=True
        )[0][:, self.flame_indices["face"]]
        target_landmarks = inputs[TARGET_2D_FULL_LANDMARKS].detach()[:, self.flame_indices["face"]]
        num_images = len(images)
        if max_images is not None:
            num_images = min(num_images, max_images)

        batch_images = []
        for idx in range(num_images):
            input_img = rgb_image_from_tensor(images[idx][:3]).copy()
            visual_2d = None
            if process_2d_branch:
                if OUTPUT_2D_LANDMARKS in outputs.keys():
                    visual_2d = self.get_regressed_visuals(
                        input_img, outputs[OUTPUT_2D_LANDMARKS].detach()[idx], inputs[TARGET_2D_LANDMARKS].detach()[idx]
                    )
                else:
                    visual_2d = self.get_2d_visuals(
                        input_img, heatmaps[idx], inputs[TARGET_LANDMARKS_HEATMAP].detach()[idx]
                    )
            visual_3d = self.get_3dmm_visuals(
                input_img,
                projected_vertices[idx],
                target_landmarks[idx],
            )
            if visual_2d is not None:
                batch_images.append(vstack_autopad((visual_2d, visual_3d)))
            else:
                batch_images.append(visual_3d)
        res_img = vstack_autopad(batch_images)
        res_img = vstack_header(res_img, f"Batch Score {score:.4f}")
        return res_img

    def get_regressed_visuals(self, input_img: np.ndarray, output: torch.Tensor, target: torch.Tensor) -> np.ndarray:
        pred_landmarks = output.cpu().numpy() * self._img_size
        target_landmarks = target.cpu().numpy() * self._img_size

        pred_img = input_img.copy()
        target_img = input_img.copy()
        for landmark in pred_landmarks:
            cv2.circle(
                img=pred_img,
                center=(int(landmark[0]), int(landmark[1])),
                radius=1,
                color=(0, 0, 255),
                thickness=-1,
            )
        for landmark in target_landmarks:
            cv2.circle(
                img=target_img,
                center=(int(landmark[0]), int(landmark[1])),
                radius=1,
                color=(0, 0, 255),
                thickness=-1,
            )
        res_img = np.hstack((input_img, pred_img, target_img))
        return res_img

    def get_2d_visuals(self, input_img: np.ndarray, output: torch.Tensor, target: torch.Tensor) -> np.ndarray:
        pred_heatmap = output.cpu().numpy()
        target_heatmap = target.cpu().numpy()
        pred_landmarks = [np.unravel_index(np.argmax(heatmap), heatmap.shape) for heatmap in pred_heatmap]
        target_landmarks = [np.unravel_index(np.argmax(heatmap), heatmap.shape) for heatmap in target_heatmap]

        pred_img = input_img.copy()
        target_img = input_img.copy()
        for landmark in pred_landmarks:
            cv2.circle(
                img=pred_img,
                center=(self.stride * landmark[1], self.stride * landmark[0]),
                radius=1,
                color=(0, 0, 255),
                thickness=-1,
            )
        for landmark in target_landmarks:
            cv2.circle(
                img=target_img,
                center=(self.stride * landmark[1], self.stride * landmark[0]),
                radius=1,
                color=(0, 0, 255),
                thickness=-1,
            )
        res_img = np.hstack((input_img, pred_img, target_img))
        return res_img

    def get_3dmm_visuals(
        self,
        input_img: np.ndarray,
        pred_landmarks: torch.Tensor,
        target_landmarks: torch.Tensor,
    ) -> np.ndarray:
        pred_img = input_img.copy()
        target_img = input_img.copy()
        for landmark in pred_landmarks:
            cv2.circle(
                img=pred_img, center=(int(landmark[0]), int(landmark[1])), radius=1, color=(0, 0, 255), thickness=-1
            )
        for landmark in target_landmarks:
            cv2.circle(
                img=target_img, center=(int(landmark[0]), int(landmark[1])), radius=1, color=(0, 0, 255), thickness=-1
            )
        res_img = np.hstack((input_img, pred_img, target_img))
        return res_img
