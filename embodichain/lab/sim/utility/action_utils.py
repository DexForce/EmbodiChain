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

import numpy as np
import torch
import warp as wp

from typing import Tuple

from embodichain.utils.utility import inv_transform
from embodichain.utils.warp import (
    trajectory_get_diff_kernel,
    trajectory_interpolate_kernel,
    trajectory_add_origin_kernel,
    get_offset_qpos_kernel,
    pairwise_distances,
    cumsum_distances,
    repeat_first_point,
    interpolate_along_distance,
)
from embodichain.lab.sim.solvers.base_solver import BaseSolver
from embodichain.utils.device_utils import standardize_device_string


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


def sort_and_padding_key_frame(
    trajectory: np.ndarray, key_indices: np.ndarray, key_frames_batch: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """sort and padding key frames for warping trajectory

    Args:
        trajectory (torch.Tensor): raw trajectory. [n_waypoint, dof] of float.
        key_indices (torch.Tensor): key frame waypoint indices. [n_keyframe,] of int.
        key_frames_batch (torch.Tensor): batch key frames. [n_batch, n_keyframe, dof] of float.

    Returns:
        key_indices_ascending (np.ndarray): padded and sorted key frame indices. [n_keyframe_new,] of int.
        key_frames_ascending (np.ndarray): padded and sorted batch key frames. [n_batch, n_keyframe_new, dof] of float.
    """
    sort_ids = np.argsort(key_indices)
    key_indices_ascending = key_indices[sort_ids]
    key_frames_ascending = key_frames_batch[:, sort_ids, :]
    n_batch = key_frames_batch.shape[0]
    if key_indices_ascending[0] != 0:
        key_indices_ascending = np.hstack([0, key_indices_ascending])
        padding_frame = trajectory[0][None, None, :].repeat(n_batch, axis=0)
        key_frames_ascending = np.concatenate(
            [padding_frame, key_frames_ascending], axis=1
        )
    if key_indices_ascending[-1] != trajectory.shape[0] - 1:
        key_indices_ascending = np.hstack(
            [key_indices_ascending, trajectory.shape[0] - 1]
        )
        padding_frame = trajectory[trajectory.shape[0] - 1][None, None, :].repeat(
            n_batch, axis=0
        )
        key_frames_ascending = np.concatenate(
            [key_frames_ascending, padding_frame], axis=1
        )
    return key_indices_ascending, key_frames_ascending


def warp_trajectory_qpos(
    trajectory: torch.Tensor,
    key_indices: torch.Tensor,
    key_frames_batch: torch.Tensor,
    device: str = "cuda",
) -> torch.Tensor:
    """warp trajectory

    Args:
        trajectory (torch.Tensor): raw trajectory. [n_waypoint, dof] of float.
        key_indices (torch.Tensor): key frame waypoint indices. [n_keyframe,] of int.
        key_frames_batch (torch.Tensor): batch key frames. [n_batch, n_keyframe, dof] of float.
        device (str, optional): torch tensor device. Defaults to "cuda".

    Returns:
        torch.Tensor: warped trajectory. [n_batch, n_waypoint, dof] of float.
    """
    # sort and pad key frames
    trajectory_np = trajectory.to("cpu").numpy().astype(np.float32)
    key_indices_np = key_indices.to("cpu").numpy().astype(np.int32)
    key_frames_batch_np = key_frames_batch.to("cpu").numpy().astype(np.float32)

    key_indices_padded, key_frames_padded = sort_and_padding_key_frame(
        trajectory_np, key_indices_np, key_frames_batch_np
    )

    # allocate cuda memory
    n_batch = key_frames_padded.shape[0]
    n_keyframe = key_indices_padded.shape[0]
    n_waypoint, dof = trajectory_np.shape
    wp_in_trajectory = wp.array(
        trajectory_np.flatten(), dtype=float, device=standardize_device_string(device)
    )
    out_trajectory = np.zeros((n_batch, n_waypoint, dof), dtype=np.float32)
    wp_out_trajectory = wp.array(
        out_trajectory.flatten(), dtype=float, device=standardize_device_string(device)
    )
    wp_key_indices = wp.array(
        key_indices_padded, dtype=int, device=standardize_device_string(device)
    )
    wp_key_frames = wp.array(
        key_frames_padded.flatten(),
        dtype=float,
        device=standardize_device_string(device),
    )

    # calcuate
    wp.launch(
        kernel=trajectory_get_diff_kernel,
        dim=(n_batch, dof),
        inputs=[
            wp_in_trajectory,
            wp_key_indices,
            wp_key_frames,
            n_waypoint,
            dof,
            n_keyframe,
        ],
        outputs=[
            wp_out_trajectory,
        ],
        device=standardize_device_string(device),
    )
    wp.launch(
        kernel=trajectory_interpolate_kernel,
        dim=(n_batch, n_waypoint, dof),
        inputs=[wp_key_indices, n_waypoint, dof, n_keyframe],
        outputs=[
            wp_out_trajectory,
        ],
        device=standardize_device_string(device),
    )
    wp.launch(
        kernel=trajectory_add_origin_kernel,
        dim=(n_batch, n_waypoint, dof),
        inputs=[wp_in_trajectory, n_waypoint, dof],
        outputs=[
            wp_out_trajectory,
        ],
        device=standardize_device_string(device),
    )
    warp_traj = (
        wp.to_torch(wp_out_trajectory)
        .reshape(n_batch, n_waypoint, dof)
        .to(torch.device(device))
    )
    return warp_traj


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
        kernel=get_offset_qpos_kernel,
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


def interpolate_with_distance_warp(
    trajectory: torch.Tensor,  # expected shape [B, N, M], float or convertible to float
    interp_num: int,  # T
    device=torch.device("cuda"),
) -> torch.Tensor:
    """
    Resample a batch of trajectories of shape [B, N, M] into [B, T, M] by
    piecewise-linear interpolation over cumulative Euclidean distance
    along the N dimension, handling each batch independently.

    Args:
        trajectory: Torch.Tensor of shape [B, N, M].
        interp_num: Target number of samples T.
        device: Warp device string ('cpu', 'cuda', 'cuda:0', ...).
        dtype: Working dtype (wp.float32 or wp.float64). Defaults to wp.float32.

    Returns:
        Torch.Tensor of shape [B, T, M] with interpolated trajectories.
    """
    # Flatten input trajectory for warp kernels (avoid multi-dimensional wp.array bugs)
    trajectory_flat = trajectory.contiguous().to(device).view(-1)
    points = wp.from_torch(trajectory_flat)

    B, N, M = trajectory.shape  # original shape components
    T = int(interp_num)

    if T < 0:
        raise ValueError("`interp_num` must be non-negative.")

    # Handle degenerate T
    out = (
        wp.empty(
            (B * T * M,), dtype=wp.float32, device=standardize_device_string(device)
        )
        if T > 0
        else wp.empty((0,), dtype=wp.float32, device=standardize_device_string(device))
    )

    # Handle N < 2
    if N < 2:
        if N == 1 and T > 0:
            # Repeat the single point across T (kernel expects flattened arrays)
            wp.launch(
                kernel=repeat_first_point,
                dim=B * T,
                inputs=[points, out, B, T, M, N],
                device=standardize_device_string(device),
            )
        # N == 0 -> return empty (out already allocated)
        interp_trajectory = (
            wp.to_torch(out).view(B, T, M) if T > 0 else wp.to_torch(out).view(B, 0, M)
        )
        return interp_trajectory

    if T == 0:
        return out  # nothing to do

    # 1) pairwise distances along N
    dists = wp.empty(
        (B * (N - 1),), dtype=wp.float32, device=standardize_device_string(device)
    )
    wp.launch(
        kernel=pairwise_distances,
        dim=B * (N - 1),
        inputs=[points, dists, B, N, M],
        device=standardize_device_string(device),
    )

    # 2) cumulative distances per batch
    cumulative = wp.empty(
        (B * N,), dtype=wp.float32, device=standardize_device_string(device)
    )
    wp.launch(
        kernel=cumsum_distances,
        dim=B,
        inputs=[dists, cumulative, B, N],
        device=standardize_device_string(device),
    )

    # 3) interpolation per (b, t)
    wp.launch(
        kernel=interpolate_along_distance,
        dim=B * T,
        inputs=[points, cumulative, out, B, N, M, T],
        device=standardize_device_string(device),
    )

    # wp.synchronize_device(device)
    interp_trajectory = wp.to_torch(out).view(B, T, M)
    return interp_trajectory
