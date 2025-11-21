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
import os
from embodichain.toolkits.graspkit.pg_grasp import (
    AntipodalGenerator,
    GraspSelectMethod,
)
from embodichain.data import get_data_path


def test_antipodal_score_selector(is_visual: bool = False):
    mesh_path = get_data_path("ChainRainSec/mesh.ply")
    mesh_o3dt = o3d.t.io.read_triangle_mesh(mesh_path)
    generator = AntipodalGenerator(
        mesh_o3dt=mesh_o3dt,
        open_length=0.1,
        max_angle=np.pi / 6,
        surface_sample_num=5000,
        cache_dir=None,
    )
    grasp_list = generator.select_grasp(
        approach_direction=np.array([0, 0, -1]),
        select_num=5,
        select_method=GraspSelectMethod.NORMAL_SCORE,
    )
    assert len(grasp_list) == 5
    if is_visual:
        visual_mesh_list = generator.grasp_pose_visual(grasp_list)
        visual_mesh_list = [visual_mesh.to_legacy() for visual_mesh in visual_mesh_list]
        o3d.visualization.draw_geometries(visual_mesh_list)


def test_antipodal_position_selector(is_visual: bool = False):
    mesh_path = get_data_path("ChainRainSec/mesh.ply")
    mesh_o3dt = o3d.t.io.read_triangle_mesh(mesh_path)
    generator = AntipodalGenerator(
        mesh_o3dt=mesh_o3dt,
        open_length=0.1,
        max_angle=np.pi / 6,
        surface_sample_num=5000,
        cache_dir=None,
    )
    grasp_list = generator.select_grasp(
        approach_direction=np.array([0, 0, -1]),
        select_num=5,
        select_method=GraspSelectMethod.NEAR_APPROACH,
    )
    assert len(grasp_list) == 5
    if is_visual:
        visual_mesh_list = generator.grasp_pose_visual(grasp_list)
        visual_mesh_list = [visual_mesh.to_legacy() for visual_mesh in visual_mesh_list]
        o3d.visualization.draw_geometries(visual_mesh_list)


def test_antipodal_center_selector(is_visual: bool = False):
    mesh_path = get_data_path("ChainRainSec/mesh.ply")
    mesh_o3dt = o3d.t.io.read_triangle_mesh(mesh_path)
    generator = AntipodalGenerator(
        mesh_o3dt=mesh_o3dt,
        open_length=0.1,
        max_angle=np.pi / 6,
        surface_sample_num=5000,
        cache_dir=None,
    )
    grasp_list = generator.select_grasp(
        approach_direction=np.array([0, 0, -1]),
        select_num=5,
        select_method=GraspSelectMethod.CENTER,
    )
    assert len(grasp_list) == 5
    if is_visual:
        visual_mesh_list = generator.grasp_pose_visual(grasp_list)
        visual_mesh_list = [visual_mesh.to_legacy() for visual_mesh in visual_mesh_list]
        o3d.visualization.draw_geometries(visual_mesh_list)


if __name__ == "__main__":
    test_antipodal_score_selector(True)
    test_antipodal_position_selector(True)
    test_antipodal_center_selector(True)
