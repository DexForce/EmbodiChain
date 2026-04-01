# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
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
import torch
from embodichain.data import get_data_path
import trimesh
from embodichain.toolkits.graspkit.pg_grasp.batch_collision_checker import (
    BatchConvexCollisionChecker,
    BatchConvexCollisionCheckerCfg,
)
from embodichain.utils.math import transform_points_mat


def batch_convex_collision_query(device=torch.device("cuda")):
    mug_path = get_data_path("ScannedBottle/moliwulong_processed.ply")
    mug_mesh = trimesh.load(mug_path, force="mesh", process=False)
    verts = torch.tensor(mug_mesh.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(mug_mesh.faces, dtype=torch.int32, device=device)
    collision_checker = BatchConvexCollisionChecker(
        verts, faces, max_decomposition_hulls=16
    )

    poses = torch.tensor(
        [
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 1.05],
                [0, 0, 0, 1],
            ],
            [
                [1, 0, 0, 0.05],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1],
            ],
        ],
        device=device,
    )
    from scipy.spatial.transform import Rotation

    rot = Rotation.from_euler("xyz", [12, 3, 32], degrees=True).as_matrix()
    poses[0, :3, :3] = torch.tensor(rot, dtype=torch.float32, device=device)
    poses[1, :3, :3] = torch.tensor(rot, dtype=torch.float32, device=device)

    obj_path = get_data_path("ScannedBottle/yibao_processed.ply")
    obj_mesh = trimesh.load(obj_path, force="mesh", process=False)
    obj_verts = torch.tensor(obj_mesh.vertices, dtype=torch.float32, device=device)
    obj_faces = torch.tensor(obj_mesh.faces, dtype=torch.int32, device=device)
    test_pc = transform_points_mat(obj_verts, poses)

    is_pose_collide, pose_surface_distance = collision_checker.query_batch_points(
        test_pc, collision_threshold=0.003, is_visual=False
    )
    assert is_pose_collide.sum().item() == 1
    assert abs(pose_surface_distance.max().item() - 0.8492) < 1e-2


def test_batch_convex_collision_cpu():
    wp.init()
    batch_convex_collision_query(torch.device("cpu"))


def test_batch_convex_collision_gpu():
    wp.init()
    batch_convex_collision_query(torch.device("cuda"))
