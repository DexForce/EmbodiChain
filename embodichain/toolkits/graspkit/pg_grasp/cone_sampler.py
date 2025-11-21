# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R


def rotate_to_ref(direc: np.ndarray, rotate_ref: np.ndarray):
    assert direc.shape == (3,)
    direc_len = np.linalg.norm(direc)
    assert direc_len > 1e-5
    direc_unit = direc / direc_len

    assert rotate_ref.shape == (3,)
    rotate_ref_len = np.linalg.norm(rotate_ref)
    assert rotate_ref_len > 1e-5
    rotate_ref_unit = rotate_ref / rotate_ref_len

    rotate_axis = np.cross(rotate_ref_unit, direc_unit)
    rotate_axis_len = np.linalg.norm(rotate_axis)
    if rotate_axis_len < 1e-5:
        # co axis, no need to do rotation
        dot_res = direc_unit.dot(rotate_ref_unit)
        if dot_res > 0:
            # identity rotation
            return np.eye(3, dtype=float)
        else:
            # negative, rotate 180 degree
            # rotate with a perpendicular axis
            random_axis = np.random.random(size=(3,))
            perpendicular_axis = np.cross(random_axis, rotate_ref_unit)
            perpendicular_axis = perpendicular_axis / np.linalg.norm(perpendicular_axis)
            ref_rotation = R.from_rotvec(perpendicular_axis * np.pi).as_matrix()
            return ref_rotation
    else:
        rotate_axis = rotate_axis / rotate_axis_len
    angle = np.arccos(direc_unit.dot(rotate_ref_unit))
    ref_rotation = R.from_rotvec(angle * rotate_axis, degrees=False).as_matrix()
    return ref_rotation


class ConeSampler:
    def __init__(
        self, max_angle: float, layer_num: int = 2, sample_each_layer: int = 4
    ) -> None:
        """cone ray sampler

        Args:
            max_angle (float): maximum ray angle to surface normal
            layer_num (int, optional): circle layer. Defaults to 2.
            sample_each_layer (int, optional): ray samples in each circle layer. Defaults to 4.
        """
        self._max_angle = max_angle
        self._layer_num = layer_num
        self._ray_num = layer_num * sample_each_layer + 1
        alpha_list = np.linspace(max_angle / layer_num, max_angle, layer_num)
        beta_list = np.linspace(
            2 * np.pi / sample_each_layer, 2 * np.pi, sample_each_layer
        )
        self._direc_ref = np.array([0, 0, 1])

        rotation_list = np.empty(shape=(self._ray_num, 3, 3), dtype=float)

        for i, alpha in enumerate(alpha_list):
            for j, beta in enumerate(beta_list):
                x_rotation = R.from_euler(
                    seq="XYZ", angles=np.array([alpha, 0, 0]), degrees=False
                ).as_matrix()
                z_rotation = R.from_euler(
                    seq="XYZ", angles=np.array([0, 0, beta]), degrees=False
                ).as_matrix()
                rotation_list[i * sample_each_layer + j + 1] = z_rotation @ x_rotation
        # original direction
        rotation_list[0] = np.eye(3)
        self._sample_direc = rotation_list[:, :3, 2]  # z-axis

    def cone_sample_direc(self, direc: np.ndarray, is_visual: bool = False):
        """sample cone directly

        Args:
            direc (np.ndarray): direction to sample a cone
            is_visual (bool, optional): use visualization or not. Defaults to False.

        Returns:
            np.ndarray: [_ray_num, 3] of float, cone direction list
        """
        ref_rotation = rotate_to_ref(direc, self._direc_ref)
        cone_direc_list = self._sample_direc @ ref_rotation.T
        if is_visual:
            self._visual(cone_direc_list)
        return cone_direc_list

    def _visual(self, cone_direc_list: np.ndarray):
        drawer = o3d.geometry.TriangleMesh.create_coordinate_frame(0.5)
        for cone_direc in cone_direc_list:
            arrow = o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=0.02,
                cone_radius=0.03,
                cylinder_height=0.9,
                cone_height=0.1,
            )
            arrow.compute_vertex_normals()
            arrow_rotation = rotate_to_ref(cone_direc, self._direc_ref)
            arrow.rotate(arrow_rotation, center=(0, 0, 0))
            arrow.paint_uniform_color(np.array([0.5, 0.5, 0.5]))
            drawer += arrow
        o3d.visualization.draw_geometries([drawer])
