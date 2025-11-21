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

import torch
import numpy as np
import open3d as o3d

from typing import Union

from dexsim.models import MeshObject
from embodichain.utils import logger
from embodichain.lab.sim.objects import RigidObject
from embodichain.lab.gym.envs.managers.cfg import SceneEntityCfg
from embodichain.utils.utility import inv_transform


def get_pc_svd_frame(pc: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Computes the pose of a point cloud using Singular Value Decomposition (SVD).

    This function centers the point cloud, performs SVD to obtain the rotation,
    and constructs a 4x4 transformation matrix representing the pose of the point cloud.

    Args:
        pc (np.ndarray): A 2D numpy array of shape (N, 3) representing the point cloud,
                         where N is the number of points.

    Returns:
        np.ndarray: A 4x4 transformation matrix that includes the rotation and translation
                    of the point cloud.
    """
    if pc.ndim != 2:
        logger.log_error(
            f"get_pc_svd_frame only support the pc of 1 object, which means that pc.ndim==2, but got {pc.ndim}"
        )
    pc_center = pc.mean(axis=0)
    pc_centered = pc - pc_center
    u, s, vt = torch.linalg.svd(pc_centered)
    rotation = vt.T
    pc_pose = torch.eye(4, dtype=torch.float32)
    pc_pose[:3, :3] = rotation
    pc_pose[:3, 3] = pc_center
    return pc_pose


def apply_svd_transfer_pc(
    geometry: Union[
        np.ndarray,
        torch.Tensor,
        o3d.cuda.pybind.geometry.TriangleMesh,
        MeshObject,
        RigidObject,
    ],
    sample_points: int = 1000,
) -> np.ndarray:
    """
    Applies Singular Value Decomposition (SVD) transfer to a point cloud represented by geometry.

    Parameters:
    geometry (Union[np.ndarray, MeshObject]): The input geometry, which can be a NumPy array of vertices
                                               or a MeshObject containing vertex data.
    sample_points (int): The number of sample points to consider (default is 1000).

    Returns:
    np.ndarray: The transformed vertices in standard position after applying SVD.
    """
    if isinstance(geometry, (RigidObject, MeshObject)):
        verts = torch.as_tensor(geometry.get_vertices())
    elif isinstance(geometry, (np.ndarray, torch.Tensor)):
        verts = torch.as_tensor(geometry)
    elif isinstance(geometry, o3d.cuda.pybind.geometry.TriangleMesh):
        verts = torch.as_tensor(geometry.vertices)
    else:
        logger.log_error(
            f"Unsupported geometry type: {type(geometry)}. Expected np.ndarray, torch.Tensor, MeshObject, or RigidObject."
        )

    if verts.ndim < 3:
        verts = verts[None]

    sample_ids = (
        np.random.choice(verts.shape[1], sample_points)
        if isinstance(verts, np.ndarray)
        else torch.randint(0, verts.shape[1], (sample_points,))
    )
    verts = verts[:, sample_ids, :]

    standard_verts = []
    for object_verts in verts:
        pc_svd_frame = get_pc_svd_frame(object_verts)
        inv_svd_frame = inv_transform(pc_svd_frame)
        standard_object_verts = (
            object_verts @ inv_svd_frame[:3, :3].T + inv_svd_frame[:3, 3]
        )
        standard_verts.append(standard_object_verts)

    return torch.stack(standard_verts)


def compute_object_length(
    env,
    env_ids: Union[torch.Tensor, None],
    entity_cfg: SceneEntityCfg,
    sample_points: int,
    is_svd_frame: bool = True,
):
    rigid_object: RigidObject = env.sim.get_rigid_object(entity_cfg.uid)
    object_lengths = {}
    for axis in ["x", "y", "z"]:
        object_lengths.update(
            {axis: torch.zeros((env.num_envs,), dtype=torch.float32, device=env.device)}
        )
    pcs = rigid_object.get_vertices(env_ids)
    body_scale = rigid_object.get_body_scale(env_ids)
    scaled_pcs = pcs * body_scale

    if is_svd_frame:
        scaled_pcs = apply_svd_transfer_pc(scaled_pcs, sample_points)

    for axis, idx in zip(["x", "y", "z"], [0, 1, 2]):
        scaled_pos = scaled_pcs[..., idx]  # (num_envs, sample_points)
        length = scaled_pos.max(dim=1)[0] - scaled_pos.min(dim=1)[0]
        object_lengths.update({axis: length})

    return object_lengths
