import os
import pickle

import numpy as np
from psbody.mesh import Mesh

from model_training.head_mesh import HeadMesh


def get_relative_path(x: str, rel_to: str) -> str:
    return os.path.join(os.path.dirname(rel_to), x)


class UVTextureCreator:
    def __init__(self):
        self.head_mesh = HeadMesh()
        self.texture_data = np.load(get_relative_path("texture_data.npy", __file__), allow_pickle=True, encoding='latin1').item()
        with open(get_relative_path("generic_model.pkl", __file__), 'rb') as f:
            self.flame_model = pickle.load(f, encoding="latin1")

    def _compute_texture_map(self, source_img, target_mesh):
        x_coords = self.texture_data.get('x_coords')
        y_coords = self.texture_data.get('y_coords')
        valid_pixel_ids = self.texture_data.get('valid_pixel_ids')
        valid_pixel_3d_faces = self.texture_data.get('valid_pixel_3d_faces')
        valid_pixel_b_coords = self.texture_data.get('valid_pixel_b_coords')
        img_size = self.texture_data.get('img_size')

        pixel_3d_points = target_mesh.v[valid_pixel_3d_faces[:, 0], :] * valid_pixel_b_coords[:, 0][:, np.newaxis] + \
                          target_mesh.v[valid_pixel_3d_faces[:, 1], :] * valid_pixel_b_coords[:, 1][:, np.newaxis] + \
                          target_mesh.v[valid_pixel_3d_faces[:, 2], :] * valid_pixel_b_coords[:, 2][:, np.newaxis]
        vertex_normals = target_mesh.estimate_vertex_normals()
        pixel_3d_normals = vertex_normals[valid_pixel_3d_faces[:, 0], :] * valid_pixel_b_coords[:, 0][:, np.newaxis] + \
                           vertex_normals[valid_pixel_3d_faces[:, 1], :] * valid_pixel_b_coords[:, 1][:, np.newaxis] + \
                           vertex_normals[valid_pixel_3d_faces[:, 2], :] * valid_pixel_b_coords[:, 2][:, np.newaxis]
        n_dot_view = -pixel_3d_normals[:, 2]

        proj_2d_points = np.round(pixel_3d_points[:, :2], 0).astype(int)

        texture = np.zeros((img_size, img_size, 3))
        for i, (x, y) in enumerate(proj_2d_points):
            if n_dot_view[i] < 0.0:
                continue
            if x > 0 and x < source_img.shape[1] and y > 0 and y < source_img.shape[0]:
                texture[y_coords[valid_pixel_ids[i]].astype(int), x_coords[valid_pixel_ids[i]].astype(int), :3] = \
                source_img[y, x]
        return texture.astype(np.uint8)

    def get_mesh(self, predicted_mesh):
        projected_vertices = self.head_mesh.reprojected_vertices(params_3dmm=predicted_mesh["3dmm_params"], to_2d=False)[0]
        return Mesh(projected_vertices, self.flame_model['f'])

    def __call__(self, image, mesh, *args, **kwargs):
        texture_map = self._compute_texture_map(image, self.get_mesh(mesh))
        return texture_map
