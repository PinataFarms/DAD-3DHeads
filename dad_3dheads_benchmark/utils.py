import torch
import os
import numpy as np
import pickle
import json
import cv2
from smplx.lbs import find_dynamic_lmk_idx_and_bcoords
from smplx.utils import Struct
from typing import Dict, Any
import yaml


ROT_COEFFS = 3
JAW_COEFFS = 3
EYE_COEFFS = 6
NECK_COEFFS = 3


def read_json(anno_file):
    with open(anno_file, "r") as f:
        return json.load(f)


def read_img(path: str) -> np.ndarray:
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def load_yaml(x: str) -> Dict[str, Any]:
    with open(x) as fd:
        config = yaml.load(fd, yaml.FullLoader)
        return config

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
        flame_static_embedding_path: str = "data/static/flame_static_embedding.pkl"
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
        contour_embeddings_path: str = "data/static/flame_dynamic_embedding.npy"
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
        mesh_faces_path: str = "data/static/flame_mesh_faces.pt"
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

# region Chamfer distance

def calc_ch_dist(gt_vertices: torch.Tensor, pred_vertices: torch.Tensor, svn_pred_lmks: np.ndarray) -> float:
    """
    gt_vertices: torch.Tensor [5023, 3]
    pred_vertices: torch.Tensor [M, 3]
    pred_lmks: torch.Tensor [7, 3]
    """
    from kaolin.metrics.pointcloud import chamfer_distance as chamfer

    # scale to standard size:
    gt_vertices = scale_gt_to_standard(gt_vertices)
    # get gt seven lmk points
    full_gt_lmks = get_68_landmarks(gt_vertices)
    svn_gt_lmks = get_7_landmarks_from_68(full_gt_lmks)

    aligned_pred_vertices = align_pred_to_gt(pred_vertices, svn_pred_lmks, svn_gt_lmks)

    gt_face_vertices = get_face_vertices_from_flame(gt_vertices)
    ch = chamfer(gt_face_vertices.view(1, -1, 3).cuda(), aligned_pred_vertices.view(1, -1, 3).cuda(), 1.0, 0.0)
    return ch.cpu().numpy()[0]


def get_7_landmarks_from_68(
        full_face_lmks: torch.Tensor,
        svn_lmks_indices: np.array = np.array([36, 39, 42, 45, 33, 48, 54]),
) -> np.array:
    """
    full_face_lmks: torch.Tensor [68, 3]
    """
    return np.take(full_face_lmks.cpu().detach().numpy(), svn_lmks_indices, axis=0)


def get_face_vertices_from_flame(
        mesh_vertices: torch.Tensor,
        face_indices_path: str = "../model_training/model/static/flame_indices/face.npy",
) -> torch.Tensor:
    """
    mesh_vertices: torch.Tensor [N, 3]
    """
    face_indices = np.load(face_indices_path)
    face_vertices = np.take(mesh_vertices.cpu().detach().numpy(), face_indices, axis=0)
    return torch.Tensor(face_vertices)


def scale_gt_to_standard(
        mesh_vertices: torch.Tensor, const_inter_eye_dist: int = 20  # mean inter eye distance
) -> torch.Tensor:
    """
    mesh_vertices: torch.Tensor [N, 3]
    """
    full_gt_lmks = get_68_landmarks(mesh_vertices)
    svn_lmks = get_7_landmarks_from_68(full_gt_lmks)
    inter_eye_dist = np.linalg.norm(svn_lmks[1] - svn_lmks[2])
    scale = const_inter_eye_dist / inter_eye_dist
    return scale * mesh_vertices


def align_pred_to_gt(pred_vertices: torch.Tensor, pred_lmks: np.array, gt_lmks: np.array) -> torch.Tensor:
    """
    pred_vertices: torch.Tensor [N, 3]
    pred_lmks: np.array [7, 3]
    gt_lmks: np.array [7, 3]
    """
    # do procrustes based on the 7 points:
    d, Z, tform = procrustes(gt_lmks, pred_lmks)
    # use tform to transform all pred vertices to the gt reference space:
    tformed_pred_vertices = []

    for vertex in pred_vertices:
        scale = tform["scale"]
        rotation = tform["rotation"]
        translation = tform["translation"]

        transformed_vertex = scale * np.dot(vertex, rotation) + translation
        tformed_pred_vertices.append(transformed_vertex)

    return torch.from_numpy(np.array(tformed_pred_vertices))


def procrustes(X, Y, scaling=True, reflection="best"):
    """
    A port of MATLAB's `procrustes` function to Numpy.
    Code from: https://stackoverflow.com/a/18927641.
    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.
        d, Z, [tform] = procrustes(X, Y)
    Inputs:
    ------------
    X, Y
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.
    scaling
        if False, the scaling component of the transformation is forced
        to 1
    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.
    Outputs
    ------------
    d
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()
    Z
        the matrix of transformed Y-values
    tform
        a dict specifying the rotation, translation and scaling that
        maps X --> Y
    """

    n, m = X.shape
    ny, my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0 ** 2.0).sum()
    ssY = (Y0 ** 2.0).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m - my)), 0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection != "best":

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:, -1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA ** 2

        # transformed coords
        Z = normX * traceTA * np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY / ssX - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my, :]
    c = muX - b * np.dot(muY, T)

    # transformation values
    tform = {"rotation": T, "scale": b, "translation": c}

    return d, Z, tform


# endregion

def get_relative_path(x: str, rel_to: str) -> str:
    return os.path.join(os.path.dirname(rel_to), x)

def get_flame_indices(name: str = "head_indices") -> np.ndarray:
    return np.load(get_relative_path(f"../model_training/model/static/{name}.npy", __file__))