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

"""Adjust joint trajectories using batches of keyframe offsets."""

from __future__ import annotations

import numpy as np
import torch
import warp as wp

from embodichain.utils.device_utils import standardize_device_string
from ._warp.warping import (
    trajectory_get_diff_kernel,
    trajectory_interpolate_kernel,
    trajectory_add_origin_kernel,
)

__all__ = ["sort_and_padding_key_frame", "warp_trajectory_qpos"]


def sort_and_padding_key_frame(
    trajectory: np.ndarray, key_indices: np.ndarray, key_frames_batch: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Sort keyframes and pad missing trajectory endpoints.

    Args:
        trajectory (np.ndarray): Original trajectory, shaped ``(N, DOF)``.
        key_indices (np.ndarray): Non-empty array of keyframe waypoint indices.
        key_frames_batch (np.ndarray): Target keyframes, shaped ``(B, K, DOF)``.

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
    """Apply interpolated keyframe offsets to batches of a joint trajectory.

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
