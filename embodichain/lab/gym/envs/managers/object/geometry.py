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

from typing import Union

from embodichain.utils import logger
from embodichain.lab.sim.objects import RigidObject
from embodichain.lab.gym.envs.managers.cfg import SceneEntityCfg
from embodichain.utils.utility import inv_transform


def get_pcd_svd_frame(pc: torch.Tensor) -> torch.Tensor:
    """Computes the pose of a point cloud using Singular Value Decomposition (SVD).

    This function centers the point cloud, performs SVD to obtain the rotation,
    and constructs a 4x4 transformation matrix representing the pose of the point cloud.

    Args:
        pc (torch.Tensor): A 2D numpy array of shape (N, 3) representing the point cloud,
                         where N is the number of points.

    Returns:
        torch.Tensor: A 4x4 transformation matrix that includes the rotation and translation
                    of the point cloud.
    """
    if pc.ndim != 2:
        logger.log_error(
            f"get_pcd_svd_frame only support the pc of 1 object, which means that pc.ndim==2, but got {pc.ndim}"
        )
    pc_center = pc.mean(axis=0)
    pc_centered = pc - pc_center
    u, s, vt = torch.linalg.svd(pc_centered)
    rotation = vt.T
    pc_pose = torch.eye(4, dtype=torch.float32, device=pc.device)
    pc_pose[:3, :3] = rotation
    pc_pose[:3, 3] = pc_center
    return pc_pose


def apply_svd_transfer_pcd(
    geometry: Union[
        np.ndarray,
        torch.Tensor,
        RigidObject,
    ],
    sample_points: int = 1000,
) -> np.ndarray:
    """Applies Singular Value Decomposition (SVD) transfer to a point cloud represented by geometry.

    Args:
        geometry (Union[np.ndarray, torch.Tensor, RigidObject]): The input geometry, which can be a numpy array,
            a torch tensor, or a RigidObject instance.
        sample_points (int): The number of sample points to consider (default is 1000).

    Returns:
        np.ndarray: The transformed vertices in standard position after applying SVD.
    """
    if isinstance(geometry, RigidObject):
        verts = torch.as_tensor(geometry.get_vertices())
    elif isinstance(geometry, (np.ndarray, torch.Tensor)):
        verts = torch.as_tensor(geometry)
    else:
        logger.log_error(
            f"Unsupported geometry type: {type(geometry)}. Expected np.ndarray, torch.Tensor, or RigidObject."
        )

    if verts.ndim < 3:
        verts = verts[None]

    sample_ids = torch.randint(0, verts.shape[1], (sample_points,), device=verts.device)
    verts = verts[:, sample_ids, :]

    # TODO: Can be optimized with fullly batch operations
    standard_verts = []
    for object_verts in verts:
        pc_svd_frame = get_pcd_svd_frame(object_verts)
        inv_svd_frame = torch.linalg.inv(pc_svd_frame)
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
    """Compute per-environment object extents (lengths) along principal axes.

    Computes the size of a rigid object's point cloud along the x, y and z axes
    for each selected environment. The point cloud is first scaled by the
    object's per-environment body scale. If requested, points are transformed
    into an SVD-aligned coordinate frame prior to extent computation.

    Args:
        env: Environment handle that must provide the following attributes:
            - sim.get_rigid_object(uid) -> RigidObject-like object
            - num_envs (int): total number of parallel environments
            - device (torch.device): device for returned tensors
        env_ids (torch.Tensor or None): Optional 1-D tensor of environment indices
            (shape (k,)) to select a subset of environments. If None, all
            environments are used. The number of selected environments
            (num_envs_selected) is env.num_envs when env_ids is None or
            env_ids.shape[0] otherwise.
        entity_cfg (SceneEntityCfg): Scene entity configuration. Must contain
            attribute `uid` used to fetch the RigidObject via env.sim.
        sample_points (int): Number of points to sample per object when applying
            SVD alignment, must be a positive integer. When not applying SVD,
            this value is the expected number of sample points per object but
            is not strictly required to match the input point count.
        is_svd_frame (bool): If True, transform the scaled point cloud into an
            SVD-aligned frame using apply_svd_transfer_pcd before computing
            extents. If False, lengths are computed in the object's current
            frame.

    Returns:
        dict: Mapping with keys "x", "y", "z". Each value is a torch.Tensor of
        shape (num_envs_selected,) with dtype torch.float32 located on
        env.device. Each tensor contains the per-environment extent along that
        axis computed as max_coordinate - min_coordinate over the sampled points.

    Raises:
        ValueError: If sample_points <= 0.
        TypeError: If env or entity_cfg do not provide the expected attributes.

    Notes:
        - The RigidObject methods used are expected to return:
            pcs = rigid_object.get_vertices(env_ids)
                -> Tensor of shape (num_envs_selected, N, 3) (N is number of
                   points available per object; SVD sampling will draw with
                   replacement if sample_points > N).
            body_scale = rigid_object.get_body_scale(env_ids)
                -> Tensor broadcastable to pcs for per-environment scaling.
        - After scaling, scaled_pcs has shape (num_envs_selected, N, 3).
          If is_svd_frame is True, apply_svd_transfer_pcd(scaled_pcs,
          sample_points) is expected to return a tensor of shape
          (num_envs_selected, sample_points, 3).
        - The returned per-axis lengths correspond to the first dimension of
          the vertex tensor returned by get_vertices (num_envs_selected).
        - Degenerate point clouds (all points identical along an axis) yield
          zero length for that axis.

    Example:
        # Use all environments and compute SVD-aligned lengths from 1024 samples
        lengths = compute_object_length(env, None, entity_cfg, sample_points=1024, is_svd_frame=True)
        # lengths['x'].shape -> torch.Size([num_envs_selected])
    """

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
        scaled_pcs = apply_svd_transfer_pcd(scaled_pcs, sample_points)

    for axis, idx in zip(["x", "y", "z"], [0, 1, 2]):
        scaled_pos = scaled_pcs[..., idx]  # (num_envs, sample_points)
        length = scaled_pos.max(dim=1)[0] - scaled_pos.min(dim=1)[0]
        object_lengths.update({axis: length})

    return object_lengths
