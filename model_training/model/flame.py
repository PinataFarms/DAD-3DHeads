from typing import Any, Optional, Union
from collections import namedtuple
import numpy as np
import torch.nn as nn
from smplx.lbs import lbs
from smplx.utils import to_tensor, to_np
from scipy.spatial.transform import Rotation

from model_training.model.utils import rot_mat_from_6dof, get_flame_model, get_flame_indices

from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor

FLAME_CONSTS = {
    "shape": 300,
    "expression": 100,
    "rotation": 6,
    "jaw": 3,
    "eyeballs": 0,
    "neck": 0,
    "translation": 3,
    "scale": 1,
}


@dataclass
class FlameParams:
    shape: Tensor
    expression: Tensor
    rotation: Tensor
    translation: Tensor
    scale: Tensor
    jaw: Tensor
    eyeballs: Tensor
    neck: Tensor

    @classmethod
    def from_3dmm(cls, tensor_3dmm: Tensor, constants: Dict[str, int], zero_expr: bool = False) -> "FlameParams":
        """
        tensor_3dmm: [B, num_params]
        """

        assert tensor_3dmm.ndim == 2

        cur_index: int = 0
        shape = tensor_3dmm[:, : constants["shape"]]
        cur_index += constants["shape"]

        expression = tensor_3dmm[:, cur_index : cur_index + constants["expression"]]
        if zero_expr:
            expression = torch.zeros_like(expression)
        cur_index += constants["expression"]

        jaw = tensor_3dmm[:, cur_index: cur_index + constants["jaw"]]
        cur_index += constants["jaw"]

        rotation = tensor_3dmm[:, cur_index : cur_index + constants["rotation"]]
        cur_index += constants["rotation"]

        eyeballs = tensor_3dmm[:, cur_index : cur_index + constants["eyeballs"]]
        cur_index += constants["eyeballs"]

        neck = tensor_3dmm[:, cur_index : cur_index + constants["neck"]]
        cur_index += constants["neck"]

        translation = tensor_3dmm[:, cur_index : cur_index + constants["translation"]]
        cur_index += constants["translation"]

        scale = tensor_3dmm[:, cur_index : cur_index + constants["scale"]]
        cur_index += constants["scale"]

        return FlameParams(
            shape=shape,
            expression=expression,
            rotation=rotation,
            jaw=jaw,
            eyeballs=eyeballs,
            neck=neck,
            translation=translation,
            scale=scale,
        )

    def to_3dmm_tensor(self) -> Tensor:
        params_3dmm = torch.cat(
            [
                self.shape,
                self.expression,
                self.rotation,
                self.jaw,
                self.eyeballs,
                self.neck,
                self.translation,
                self.scale,
            ],
            -1,
        )

        return params_3dmm


RPY = namedtuple("RPY", ["roll", "pitch", "yaw"])


MAX_SHAPE = 300
MAX_EXPRESSION = 100

ROT_COEFFS = 3
JAW_COEFFS = 3
EYE_COEFFS = 6
NECK_COEFFS = 3
MESH_OFFSET_Z = 0.05



class FLAMELayer(nn.Module):
    """
    Based on https://github.com/soubhiksanyal/FLAME_PyTorch/blob/master/FLAME.py
    Given flame parameters this class generates a differentiable FLAME function which outputs vertices of the FLAME mesh, modified w.r.t. these parameters.
    """

    def __init__(self, consts: Dict[str, Any], batch_size: int = 1, flame_path: Optional[str] = None) -> None:
        super().__init__()
        self.flame_model = get_flame_model(flame_path)
        self.flame_constants = consts
        self.batch_size = batch_size
        self.dtype = torch.float32
        self.faces = self.flame_model.f
        self.register_buffer("faces_tensor", to_tensor(to_np(self.faces, dtype=np.int64), dtype=torch.long))
        indices_2d = get_flame_indices("indices_2d")  # 191 keypoint indices
        self.register_buffer("indices_2d", to_tensor(indices_2d, dtype=torch.long))
        # Fixing remaining Shape betas
        default_shape = torch.zeros(
            [self.batch_size, MAX_SHAPE - consts["shape"]], dtype=self.dtype, requires_grad=False
        )
        self.register_parameter("shape_betas", nn.Parameter(default_shape, requires_grad=False))

        # Fixing remaining expression betas
        default_exp = torch.zeros(
            [self.batch_size, MAX_EXPRESSION - consts["expression"]], dtype=self.dtype, requires_grad=False
        )
        self.register_parameter("expression_betas", nn.Parameter(default_exp, requires_grad=False))

        default_rot = torch.zeros([self.batch_size, ROT_COEFFS], dtype=self.dtype, requires_grad=False)
        self.register_parameter("rot", nn.Parameter(default_rot, requires_grad=False))

        default_jaw = torch.zeros([self.batch_size, JAW_COEFFS], dtype=self.dtype, requires_grad=False)
        self.register_parameter("jaw", nn.Parameter(default_jaw, requires_grad=False))

        # Eyeball and neck rotation
        default_eyeball_pose = torch.zeros([self.batch_size, EYE_COEFFS], dtype=self.dtype, requires_grad=False)
        self.register_parameter("eyeballs", nn.Parameter(default_eyeball_pose, requires_grad=False))

        default_neck_pose = torch.zeros([self.batch_size, NECK_COEFFS], dtype=self.dtype, requires_grad=False)
        self.register_parameter("neck_pose", nn.Parameter(default_neck_pose, requires_grad=False))

        # The vertices of the template model
        self.register_buffer("v_template", to_tensor(to_np(self.flame_model.v_template), dtype=self.dtype))

        # The shape components
        shapedirs = self.flame_model.shapedirs
        # The shape components
        self.register_buffer("shapedirs", to_tensor(to_np(shapedirs), dtype=self.dtype))

        j_regressor = to_tensor(to_np(self.flame_model.J_regressor), dtype=self.dtype)
        self.register_buffer("J_regressor", j_regressor)

        # Pose blend shape basis
        num_pose_basis = self.flame_model.posedirs.shape[-1]
        posedirs = np.reshape(self.flame_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer("posedirs", to_tensor(to_np(posedirs), dtype=self.dtype))

        # indices of parents for each joints
        parents = to_tensor(to_np(self.flame_model.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer("parents", parents)

        self.register_buffer("lbs_weights", to_tensor(to_np(self.flame_model.weights), dtype=self.dtype))

    def forward(self, flame_params: FlameParams, zero_rot: bool = False, zero_jaw: bool = False) -> torch.Tensor:
        """
        Input:
            shape_params: B X number of shape parameters
            expression_params: B X number of expression parameters
            pose_params: B X number of pose parameters
        return:
            vertices: B X V X 3
        """
        bs = flame_params.shape.shape[0]
        betas = torch.cat(
            [
                flame_params.shape,
                self.shape_betas[[0]].expand(bs, -1),
                flame_params.expression,
                self.expression_betas[[0]].expand(bs, -1),
            ],
            dim=1,
        )
        neck_pose = flame_params.neck if not (0 in flame_params.neck.shape) else self.neck_pose[[0]].expand(bs, -1)
        eyeballs = flame_params.eyeballs if not (0 in flame_params.eyeballs.shape) else self.eyeballs[[0]].expand(bs, -1)
        jaw = flame_params.jaw if not (0 in flame_params.jaw.shape) else self.jaw[[0]].expand(bs, -1)

        rotation = torch.zeros([bs, ROT_COEFFS], device=flame_params.rotation.device)
        if zero_jaw:
            jaw = torch.zeros_like(jaw)
        full_pose = torch.cat([rotation, neck_pose, jaw, eyeballs], dim=1)

        template_vertices = self.v_template.unsqueeze(0).repeat(bs, 1, 1)

        vertices, _ = lbs(
            betas,
            full_pose,
            template_vertices,
            self.shapedirs,
            self.posedirs,
            self.J_regressor,
            self.parents,
            self.lbs_weights,
        )

        # translate to skull center and rotate
        vertices[:, :, 2] += MESH_OFFSET_Z
        if not zero_rot:
            rotation_mat = rot_mat_from_6dof(flame_params.rotation).type(vertices.dtype)
            vertices = torch.matmul(rotation_mat.unsqueeze(1), vertices.unsqueeze(-1))
            vertices = vertices[..., 0]
        return vertices


def uint8_to_float32(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.uint8:
        return x.div(255.0).to(dtype=torch.float32)
    else:
        return x


def limit_angle(angle: Union[int, float], pi: Union[int, float] = 180.0) -> Union[int, float]:
    """
    Angle should be in degrees, not in radians.
    If you have an angle in radians - use the function radians_to_degrees.
    """
    if angle < -pi:
        k = -2 * (int(angle / pi) // 2)
        angle = angle + k * pi
    if angle > pi:
        k = 2 * ((int(angle / pi) + 1) // 2)
        angle = angle - k * pi

    return angle


def calculate_rpy(flame_params) -> RPY:
    rot_mat = rotation_mat_from_flame_params(flame_params)
    rot_mat_2 = np.transpose(rot_mat)
    angle = Rotation.from_matrix(rot_mat_2).as_euler("xyz", degrees=True)
    roll, pitch, yaw = list(map(limit_angle, [angle[2], angle[0] - 180, angle[1]]))
    return RPY(roll=roll, pitch=pitch, yaw=yaw)


def rotation_mat_from_flame_params(flame_params):
    rot_mat = rot_mat_from_6dof(flame_params.rotation).numpy()[0]
    return rot_mat
