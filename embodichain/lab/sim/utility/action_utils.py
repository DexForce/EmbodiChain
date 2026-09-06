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

"""Simulation trajectory adaptation and compatibility exports.

Pure trajectory computations are owned by ``embodichain.compute.trajectory``."""

from __future__ import annotations

import numpy as np
import torch
import warp as wp

from embodichain.compute.trajectory import (
    interpolate_with_distance,
    interpolate_with_nums,
    resample_with_distance,
    sort_and_padding_key_frame,
    warp_trajectory_qpos,
)
from embodichain.compute.trajectory._warp.warping import compute_offset_key_poses_kernel
from embodichain.lab.sim.solvers.base_solver import BaseSolver
from embodichain.utils.device_utils import standardize_device_string
from embodichain.utils.utility import inv_transform

__all__ = [
    "compute_pose_offset_related_to_first",
    "get_trajectory_object_offset_qpos",
    "interpolate_with_distance",
    "interpolate_with_nums",
    "resample_with_distance",
    "sort_and_padding_key_frame",
    "warp_trajectory_qpos",
]


def compute_pose_offset_related_to_first(full_pose: torch.Tensor) -> torch.Tensor:
    """Compute pose offset relative to the first pose.

    Args:
        full_pose (torch.Tensor): The full pose tensor of shape (N, 4, 4).

    Returns:
        torch.Tensor: The pose offset tensor of shape (N, 4, 4).
    """
    inv_pose0_np = inv_transform(full_pose[0].to("cpu").numpy())
    inv_pose0 = torch.tensor(inv_pose0_np, device=full_pose.device)
    inv_pose0_repeat = inv_pose0[None, :, :].repeat(full_pose.shape[0], 1, 1)
    return torch.bmm(inv_pose0_repeat, full_pose)


def get_trajectory_object_offset_qpos(
    trajectory: torch.Tensor,
    key_indices: torch.Tensor,
    key_obj_indices: torch.Tensor,
    obj_offset: torch.Tensor,
    solver: BaseSolver,
    base_xpos: torch.Tensor,
    device=torch.device("cuda"),
):
    """warp trajectory according to object pose offset

    Args:
        trajectory (torch.Tensor): raw trajectory. [n_waypoint, dof] of float, joint positions.
        key_indices (torch.Tensor): key frame waypoint indices. [n_keyframe,] of int.
        key_obj_indices (torch.Tensor): key frame belong to which object index. [n_keyframe,] of int.
        obj_offset (torch.Tensor): each object pose offset. [obj_num, n_batch, 4, 4] of float.
        solver (BaseSolver): robot kinematic solver.
        base_xpos (torch.Tensor): solver root link pose in world coordinate. [4, 4] of float.
        device (str, optional): torch tensor device. Defaults to "cuda".

    Returns:
        torch.Tensor: warped trajectory. [n_batch, n_waypoint, dof] of float.
    """
    assert key_indices.shape[0] == key_obj_indices.shape[0]
    dof = trajectory.shape[1]
    key_qpos = trajectory[key_indices]  # [n_keyframe, DOF]
    n_batch = obj_offset.shape[1]  # batch num, aws arena num
    n_keyframe = key_qpos.shape[0]
    key_xpos = solver.get_fk(key_qpos)  # [n_keyframe, 4, 4]

    base_xpos_repeat = base_xpos[None, :, :].repeat(n_keyframe, 1, 1)
    key_xpos = torch.bmm(base_xpos_repeat, key_xpos)

    base_xpos_inv_np = inv_transform(base_xpos.to("cpu").numpy())
    base_xpos_inv_wp = wp.mat44f(base_xpos_inv_np)
    key_obj_indices_wp = wp.from_torch(key_obj_indices.reshape(-1))
    obj_offset_wp = wp.from_torch(obj_offset.reshape(-1))
    key_xpos_wp = wp.from_torch(key_xpos.reshape(-1))
    key_obj_offset_wp = wp.zeros(
        n_batch * n_keyframe * 16, dtype=float, device=standardize_device_string(device)
    )

    wp.launch(
        kernel=compute_offset_key_poses_kernel,
        dim=(n_batch, n_keyframe),
        inputs=[
            key_obj_indices_wp,
            obj_offset_wp,
            key_xpos_wp,
            base_xpos_inv_wp,
            n_batch,
            n_keyframe,
        ],
        outputs=[
            key_obj_offset_wp,
        ],
        device=standardize_device_string(device),
    )
    key_xpos_offset = wp.to_torch(key_obj_offset_wp).reshape(n_batch * n_keyframe, 4, 4)
    key_qpos_batch = key_qpos[None, :, :].repeat(n_batch, 1, 1).reshape(-1, dof)
    # for pytorch solver, ik use qpos seed but not joint seed
    is_success, key_qpos_offset = solver.get_ik(
        target_xpos=key_xpos_offset,
        qpos_seed=key_qpos_batch,
    )
    key_qpos_offset = key_qpos_offset.reshape(n_batch, n_keyframe, -1)
    return is_success, key_qpos_offset
