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
import warp as wp


@wp.kernel
def trajectory_get_diff_kernel(
    in_trajectory: wp.array(dtype=float),
    key_indices: wp.array(dtype=int),
    key_frames: wp.array(dtype=float),
    waypoint_num: int,
    dof: int,
    key_frame_num: int,
    warp_trajectory: wp.array(dtype=float),
):
    """warp trajectory get diff kernel

    Args:
        in_trajectory (wp.array, optional): (waypoint_num * dof) of float. Input trajectory.
        key_indices (wp.array, optional): (key_frame_num,) of int. Key frame indices.
        key_frames (wp.array, optional): (bn * key_frame_num * dof) of float. Batch key frames.
        waypoint_num (int): number of waypoints.
        dof (int): number of degrees of freedom.
        key_frame_num (int): number of key frames.
        warp_trajectory (wp.array, optional): (bn * waypoint_num * dof) of float. Output warp trajectory.
    """
    arena_id, dim = wp.tid()

    # write_diff
    for i in range(key_frame_num):
        key_id = key_indices[i]
        warp_id = arena_id * waypoint_num * dof + key_id * dof + dim
        key_frame_id = arena_id * key_frame_num * dof + i * dof + dim
        in_trajectory_id = key_id * dof + dim
        warp_trajectory[warp_id] = (
            key_frames[key_frame_id] - in_trajectory[in_trajectory_id]
        )


@wp.kernel
def trajectory_interpolate_kernel(
    key_indices: wp.array(dtype=int),
    waypoint_num: int,
    dof: int,
    key_frame_num: int,
    warp_trajectory: wp.array(dtype=float),
):
    """warp trajectory interpolate kernel

    Args:
        key_indices (wp.array, optional): (key_frame_num,) of int. Key frame indices.
        waypoint_num (int): number of waypoints.
        dof (int): number of degrees of freedom.
        key_frame_num (int): number of key frames.
        warp_trajectory (wp.array, optional): (bn * waypoint_num * dof) of float. Output warp trajectory.
    """
    arena_id, waypoint_id, dim = wp.tid()
    inter_warp_id = arena_id * waypoint_num * dof + waypoint_id * dof + dim

    start_id = int(-1)
    end_id = int(-1)
    # find start id and end id
    # assume key_indices is sorted, start from 0, end at waypoint_num - 1
    for i in range(key_frame_num):
        key_id = key_indices[i]
        # to the final one
        if waypoint_id >= key_id:
            start_id = key_id
            end_id = key_indices[i + 1]

    if waypoint_id == end_id or waypoint_id == start_id:
        # start | final key frame, only add to interp id
        return

    if start_id == -1 or end_id == -1:
        # invalid, do nothing
        return

    alpha = float(waypoint_id - start_id) / float(end_id - start_id)
    start_warp_id = arena_id * waypoint_num * dof + start_id * dof + dim
    end_warp_id = arena_id * waypoint_num * dof + end_id * dof + dim

    warp_trajectory[inter_warp_id] = (1.0 - alpha) * warp_trajectory[
        start_warp_id
    ] + alpha * warp_trajectory[end_warp_id]


@wp.kernel
def trajectory_add_origin_kernel(
    in_trajectory: wp.array(dtype=float),
    waypoint_num: int,
    dof: int,
    warp_trajectory: wp.array(dtype=float),
):
    arena_id, waypoint_id, dim = wp.tid()
    inter_warp_id = arena_id * waypoint_num * dof + waypoint_id * dof + dim
    in_trajectory_id = waypoint_id * dof + dim
    warp_trajectory[inter_warp_id] += in_trajectory[in_trajectory_id]


@wp.kernel
def get_offset_qpos_kernel(
    key_obj_indices: wp.array(dtype=int),
    obj_offset: wp.array(dtype=float),
    key_xpos: wp.array(dtype=float),
    base_xpos_inv: wp.mat44f,
    n_batch: int,
    n_keyframe: int,
    key_xpos_offset: wp.array(dtype=float),
):
    batch_id, key_id = wp.tid()
    obj_idx = key_obj_indices[key_id]
    obj_offset_idx = n_batch * obj_idx + batch_id
    obj_offset_pose = wp.mat44f(
        obj_offset[obj_offset_idx * 16 + 0],
        obj_offset[obj_offset_idx * 16 + 1],
        obj_offset[obj_offset_idx * 16 + 2],
        obj_offset[obj_offset_idx * 16 + 3],
        obj_offset[obj_offset_idx * 16 + 4],
        obj_offset[obj_offset_idx * 16 + 5],
        obj_offset[obj_offset_idx * 16 + 6],
        obj_offset[obj_offset_idx * 16 + 7],
        obj_offset[obj_offset_idx * 16 + 8],
        obj_offset[obj_offset_idx * 16 + 9],
        obj_offset[obj_offset_idx * 16 + 10],
        obj_offset[obj_offset_idx * 16 + 11],
        obj_offset[obj_offset_idx * 16 + 12],
        obj_offset[obj_offset_idx * 16 + 13],
        obj_offset[obj_offset_idx * 16 + 14],
        obj_offset[obj_offset_idx * 16 + 15],
    )
    key_xpos_single = wp.mat44f(
        key_xpos[key_id * 16 + 0],
        key_xpos[key_id * 16 + 1],
        key_xpos[key_id * 16 + 2],
        key_xpos[key_id * 16 + 3],
        key_xpos[key_id * 16 + 4],
        key_xpos[key_id * 16 + 5],
        key_xpos[key_id * 16 + 6],
        key_xpos[key_id * 16 + 7],
        key_xpos[key_id * 16 + 8],
        key_xpos[key_id * 16 + 9],
        key_xpos[key_id * 16 + 10],
        key_xpos[key_id * 16 + 11],
        key_xpos[key_id * 16 + 12],
        key_xpos[key_id * 16 + 13],
        key_xpos[key_id * 16 + 14],
        key_xpos[key_id * 16 + 15],
    )
    key_xpos_offset_i = base_xpos_inv * key_xpos_single * obj_offset_pose
    key_xpos_offset_idx = batch_id * n_keyframe + key_id
    key_xpos_offset[key_xpos_offset_idx * 16 + 0] = key_xpos_offset_i[0][0]
    key_xpos_offset[key_xpos_offset_idx * 16 + 1] = key_xpos_offset_i[0][1]
    key_xpos_offset[key_xpos_offset_idx * 16 + 2] = key_xpos_offset_i[0][2]
    key_xpos_offset[key_xpos_offset_idx * 16 + 3] = key_xpos_offset_i[0][3]
    key_xpos_offset[key_xpos_offset_idx * 16 + 4] = key_xpos_offset_i[1][0]
    key_xpos_offset[key_xpos_offset_idx * 16 + 5] = key_xpos_offset_i[1][1]
    key_xpos_offset[key_xpos_offset_idx * 16 + 6] = key_xpos_offset_i[1][2]
    key_xpos_offset[key_xpos_offset_idx * 16 + 7] = key_xpos_offset_i[1][3]
    key_xpos_offset[key_xpos_offset_idx * 16 + 8] = key_xpos_offset_i[2][0]
    key_xpos_offset[key_xpos_offset_idx * 16 + 9] = key_xpos_offset_i[2][1]
    key_xpos_offset[key_xpos_offset_idx * 16 + 10] = key_xpos_offset_i[2][2]
    key_xpos_offset[key_xpos_offset_idx * 16 + 11] = key_xpos_offset_i[2][3]
    key_xpos_offset[key_xpos_offset_idx * 16 + 12] = key_xpos_offset_i[3][0]
    key_xpos_offset[key_xpos_offset_idx * 16 + 13] = key_xpos_offset_i[3][1]
    key_xpos_offset[key_xpos_offset_idx * 16 + 14] = key_xpos_offset_i[3][2]
    key_xpos_offset[key_xpos_offset_idx * 16 + 15] = key_xpos_offset_i[3][3]
