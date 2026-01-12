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

import re
import os
import ast
import cv2
import h5py
import torch
import inspect
import open3d as o3d

from copy import deepcopy
from functools import partial, wraps, lru_cache
from collections import OrderedDict
from importlib import import_module
from scipy.spatial.transform import Rotation as R
from typing import Any, Dict, List, Tuple, Union, Sequence, Callable, Mapping

import numpy as np

from embodichain.lab.sim.objects import Robot
from embodichain.utils.utility import inv_transform
from embodichain.utils.logger import log_info, log_warning, log_error


def no_validation(*args, **kwargs):
    return True


def mul_linear_expand(
    arr: np.ndarray, expand_times: Union[int, List[int]], is_interp: bool = True
) -> np.ndarray:
    """
    Linearly interpolate or repeat between points in an array.

    Args:
        arr (np.ndarray): Input array of shape (N, D).
        expand_times (int or List[int]): Number of samples between each pair.
        is_interp (bool): If True, interpolate; else, repeat.

    Returns:
        np.ndarray: Expanded/interpolated array.
    """
    arr = np.asarray(arr)
    arr_len, dim = arr.shape
    if isinstance(expand_times, int):
        interp_path = np.zeros(shape=(arr_len * expand_times, dim), dtype=float)
    else:
        assert len(expand_times) == arr_len - 1, "Invalid expand_times size."
        interp_path = np.zeros(shape=(sum(expand_times), dim), dtype=float)

    idx = 0
    for i in range(arr_len - 1):
        sample_times = (
            expand_times if isinstance(expand_times, int) else expand_times[i]
        )
        for k in range(sample_times):
            if is_interp:
                alpha = k / sample_times
                v = (1 - alpha) * arr[i] + alpha * arr[i + 1]
            else:
                v = arr[i]
            interp_path[idx] = v
            idx += 1
    interp_path = interp_path[:idx]
    return interp_path


def axis_idx(k: str) -> int:
    return {"x": 0, "y": 1, "z": 2}.get(k, None)


def axis_str_to_list(s: str):
    if any(c not in "xyz" for c in s):
        return None
    return ["xyz".index(c) for c in s]


def is_pose_axis_align(
    pose: List,
    vector: List,
    axis_str: str,
    mode: str,
    cos_threshold: float = None,
    degree_threshold: float = None,
):
    """Check if the given `axis` of a `pose` is aligned with given vector, under given mode and cos_threshold.
        i.e. the cosine value of the angle between the pose's `axis` and `vector` is respecting (leq or geq) the `cos_threhold` or not.

    Args:
        pose (List): The pose to be checked.
        vector (List): The vector to be aligning to.
        axis_str (str): The string of the axis.
        mode (str): leq or geq.
        cos_threshold (float): The threshold of the cosine value between the pose's axis and vector.
        degree_threshold (float): The threshold of the degree value between the pose's axis and vector, only functions when cos_threshold is not given.
    """
    pose = np.asarray(pose)
    vector = np.asarray(vector)

    if cos_threshold is None:
        if degree_threshold is None:
            log_error(
                'cos_threshold & angle_threshold are both None, illegal for "is_pose_axis_align".'
            )
        else:
            cos_threshold = np.cos(np.deg2rad(degree_threshold))

    axis_id = axis_idx(axis_str)
    axis = pose[:3, axis_id]
    cos_value = np.dot(axis, vector) / np.linalg.norm(vector)

    if "abs" in mode:
        cos_value = abs(cos_value)

    if "leq" in mode:
        return cos_value <= cos_threshold
    elif "geq" in mode:
        return cos_value >= cos_threshold


def is_pose_flip(
    pose: list, ref_pose: list, axis_str: str = "y", return_inverse: bool = False
):
    pose = np.asarray(pose)
    ref_pose = np.asarray(ref_pose)
    axis_idx = axis_idx(axis_str)
    if axis_idx is None:
        log_error(f'Axis {axis_str} is not among ["x", "y", "z"]')
    relative_angle = np.abs(np.arccos(pose[:3, axis_idx].dot(ref_pose[:3, axis_idx])))
    valid_ret = relative_angle > np.pi / 2

    if return_inverse:
        valid_ret = not valid_ret

    return valid_ret


def is_qpos_exceed(
    qpos: Union[np.ndarray, torch.Tensor], robot: Robot, control_part: str
) -> bool:
    """
    Check if the given qpos exceeds the joint limits of the specified control part.
    Supports both numpy and torch tensor inputs.

    Args:
        qpos (Union[np.ndarray, torch.Tensor]): The joint positions to check.
        robot (Robot): The robot object containing joint limits.
        control_part (str): The name of the control part to check.

    Returns:
        bool: True if qpos exceeds joint limits, False otherwise.
    """
    joint_limits = robot.body_data.qpos_limits[0][
        robot.get_joint_ids(name=control_part)
    ]
    # Convert joint_limits to tensor if qpos is tensor, else to numpy
    if isinstance(qpos, torch.Tensor):
        joint_limits = torch.as_tensor(
            joint_limits, dtype=qpos.dtype, device=qpos.device
        )
        exceed = torch.any(qpos < joint_limits[:, 0]) or torch.any(
            qpos > joint_limits[:, 1]
        )
        return not exceed
    else:
        qpos = np.asarray(qpos)
        # 保证 joint_limits 是 numpy 类型
        if isinstance(joint_limits, torch.Tensor):
            joint_limits = joint_limits.cpu().numpy()
        exceed = np.any(qpos < joint_limits[:, 0]) or np.any(qpos > joint_limits[:, 1])
        return not exceed


def is_qpos_flip(
    qpos: Union[np.ndarray, torch.Tensor],
    qpos_ref: Union[np.ndarray, torch.Tensor],
    qpos_ids: Union[List, np.ndarray],
    threshold: float = 1.1 * np.pi,
    mode: str = "delta",
    return_inverse: bool = False,
):
    """
    Check whether the joint positions (qpos) are flipped compared to a reference (qpos_ref).
    Supports both numpy and torch tensor inputs.

    Args:
        qpos (Union[np.ndarray, torch.Tensor]): The joint positions to check.
        qpos_ref (Union[np.ndarray, torch.Tensor]): The reference joint positions.
        qpos_ids (Union[List, np.ndarray]): Indices of joints to compare.
        threshold (float, optional): Threshold for delta mode. Defaults to 1.1 * np.pi.
        mode (str, optional): "delta" for norm difference, "sign" for sign difference. Defaults to "delta".
        return_inverse (bool, optional): If True, returns the inverse result. Defaults to False.

    Returns:
        bool: True if flipped, False otherwise.
    """
    # Ensure qpos_ids is numpy array for indexing
    if isinstance(qpos_ids, torch.Tensor):
        qpos_ids = qpos_ids.cpu().numpy()
    # If either input is torch.Tensor, convert both to tensor for comparison
    if isinstance(qpos, torch.Tensor) or isinstance(qpos_ref, torch.Tensor):
        if not isinstance(qpos, torch.Tensor):
            qpos = torch.from_numpy(qpos)
        if not isinstance(qpos_ref, torch.Tensor):
            qpos_ref = torch.from_numpy(qpos_ref)
        qpos_ids_tensor = torch.as_tensor(qpos_ids, dtype=torch.long)
        if mode == "delta":
            # Compute norm difference for selected joints
            qpos_diff = torch.norm(qpos[qpos_ids_tensor] - qpos_ref[qpos_ids_tensor])
            valid_ret = qpos_diff > threshold
        elif mode == "sign":
            # Check sign difference for selected joints
            valid_ret = (qpos[qpos_ids_tensor] * qpos_ref[qpos_ids_tensor]) < 0
        else:
            log_error(f"The qpos flip mode {mode} has not been implemented yet.")
        # Convert torch scalar to Python bool
        if isinstance(valid_ret, torch.Tensor):
            valid_ret = valid_ret.item() if valid_ret.numel() == 1 else bool(valid_ret)
    else:
        qpos_ids = np.asarray(qpos_ids)
        if mode == "delta":
            qpos_diff = np.linalg.norm(qpos[qpos_ids] - qpos_ref[qpos_ids])
            valid_ret = qpos_diff > threshold
        elif mode == "sign":
            valid_ret = (qpos[qpos_ids] * qpos_ref[qpos_ids]) < 0
        else:
            log_error(f"The qpos flip mode {mode} has not been implemented yet.")

    if return_inverse:
        valid_ret = not valid_ret

    return valid_ret


def get_replaced_pose(
    pose_to_change: np.ndarray,
    pose_replace_value: Union[float, List],
    axis_str_replace: str,
) -> np.ndarray:
    """
    Replace specific axes of a pose with new values.

    Args:
        pose_to_change (np.ndarray): The pose to be modified (4x4 matrix).
        pose_replace_value (Union[float, List]): The values to replace the specified axes.
        axis_str_replace (str): A string specifying the axes to replace (e.g., "xy").

    Returns:
        np.ndarray: The modified pose.

    Raises:
        ValueError: If the lengths of `pose_replace_value` and `axis_str_replace` do not match.
    """
    axis_list_replace = axis_str_to_list(axis_str_replace)
    if axis_list_replace is None:
        raise ValueError(f"Invalid axis string: {axis_str_replace}")

    if isinstance(pose_replace_value, (Sequence, np.ndarray)):
        pose_replace_value_length = len(pose_replace_value)
    else:
        pose_replace_value_length = 1
        pose_replace_value = [pose_replace_value]

    if pose_replace_value_length != len(axis_list_replace):
        log_error(
            f'The axis asked to be raplaced is "{axis_str_replace}", but got {pose_replace_value_length} changes quantity.'
        )
    for axis, replace_quantity in zip(axis_list_replace, pose_replace_value):
        pose_to_change[axis, 3] = replace_quantity
    return pose_to_change


def get_offset_pose(
    pose_to_change: np.ndarray,
    offset_value: Union[float, List[float]],
    direction: Union[str, List] = "z",
    mode: str = "extrinsic",
) -> np.ndarray:
    """Offset the `pose_to_change` given the `offset_value`, `direction` & `mode`. Returns the offset pose.

    Args:
        pose_to_change (np.ndarray): The pose to be offset.
        offset_value (Union[float, List[float]]): The offset.
        direction (Union[str, List], optional): String as "x", "y" or, "z" and 3-dim np.ndarray indicating the offset directions. Defaults to "z".
        mode (str, optional): String "extrinsic" or "intrinsic", indicating which system frame should each offset shall be done. Defaults to "extrinsic".

    Returns:
        np.ndarray: The resulting 4x4 offset pose.

    Raises:
        ValueError: If inputs are invalid or incompatible.
    """
    if isinstance(direction, str):
        minus = "-" in direction
        direction = direction.removeprefix("-")
        direction = np.isin(np.arange(3), axis_str_to_list(direction)).astype(int) * (
            -1 if minus else 1
        )

    direction = np.asarray(direction)
    direction = direction / np.linalg.norm(direction)
    offset_matrix = np.eye(4)
    offset_matrix[:3, 3] = offset_value * direction
    if mode == "extrinsic":
        offset_pose = offset_matrix @ pose_to_change
    elif mode == "intrinsic":
        offset_pose = pose_to_change @ offset_matrix
    else:
        log_error(f"Mode {mode} illegal.")
    return offset_pose


# TODO: This one is not for work
def get_offset_pose_list(
    pose_to_change: np.ndarray,
    offsets: Union[float, List[float]],
    directions: Union[str, np.ndarray, List[str], List[np.ndarray]] = [],
    modes: Union[str, List[str]] = [],
):
    """Offset the `pose_to_change` given the `offsets`, `directions` & `modes`. Returns the offset poses.

    Args:
        pose_to_change (np.ndarray): The pose to be offset.
        offsets (Union[float, List[float]]): The offset or the offset list.
        directions (Union[str, np.ndarray, List[str], List[np.ndarray]], optional): String as "x", "y" or, "z" and 3-dim np.ndarray indicating the offset directions, together with their list that have same size of `offsets` . Defaults to [].
        modes (Union[str, List[str]], optional): String "extrinsic" or "intrinsic", and its list, indicating which system frame should each offset shall be done. Defaults to [].
        return_single_pose: Whether return the single pose or not
    """
    num_offset_pose = len(offsets) if isinstance(offsets, list) else 1
    num_offset_direction = len(directions) if isinstance(directions, list) else 1
    num_offset_mode = len(modes) if isinstance(modes, list) else 1
    if num_offset_direction == 0:
        directions = ["z"] * num_offset_pose
        num_offset_direction = num_offset_pose
    if num_offset_mode == 0:
        modes = ["extrinsic"] * num_offset_direction
        num_offset_mode = num_offset_direction
    if num_offset_direction != num_offset_pose:
        log_error(
            f"The offsets {offsets} have a different length {num_offset_pose} other than directions {directions}'s {num_offset_direction}."
        )
    if num_offset_mode != num_offset_direction:
        log_warning(
            f"The directions {directions} have a different length {num_offset_direction} other than modes {modes}'s {num_offset_mode}."
        )
    if num_offset_direction == 1 and not isinstance(directions, list):
        directions = [directions]
    if num_offset_mode == 1 and not isinstance(modes, list):
        modes = [modes]

    offset_poses = []
    for idx, (offset, (direction, mode)) in enumerate(
        zip(offsets, zip(directions, modes))
    ):
        offset_pose = get_offset_pose(pose_to_change, offset, direction, mode)
        offset_poses.append(offset_pose)

    return offset_poses


def get_rotated_pose(
    pose_to_change: np.ndarray,
    rot_angle: float,
    rot_axis: Union[str, List] = "z",
    mode: str = "extrinsic",
    degrees: Union[bool, str] = None,
):
    """Rotate the `pose_to_change` given the `rot_angel`, `rot_axis` & `mode`. Returns the rotate pose.

    Args:
        pose_to_change (np.ndarray): The pose to be rotated.
        rot_angle (float): The rotation angle.
        rot_axis (Union[str, List], optional): String as "x", "y" or, "z" and 3-dim np.ndarray indicating the rotation axis. Defaults to "z".
        mode (str, optional): String "extrinsic" or "intrinsic", and its list, indicating which system frame should each rotation shall be done. Defaults to "extrinsic".
        degrees (str): If it's "degrees" then the input rotation angle is degree, then it's not degrees but radians.
    """
    if isinstance(rot_axis, str):
        rot_axis = np.isin(np.arange(3), axis_str_to_list(rot_axis)).astype(int)
    rot_axis = np.asarray(rot_axis)
    rot_axis = rot_axis / np.linalg.norm(rot_axis)

    if degrees == "degrees" or degrees == True:
        rot_angle = np.deg2rad(rot_angle)

    rotation_matrix = np.eye(4)
    rotation_matrix[:3, :3] = R.from_rotvec(rot_axis * rot_angle).as_matrix()
    if mode == "extrinsic":
        rotated_pose = rotation_matrix @ pose_to_change
    elif mode == "intrinsic":
        rotated_pose = pose_to_change @ rotation_matrix
    else:
        log_error(f"Mode {mode} illegal.")
    return rotated_pose


def get_rotation_replaced_pose(
    pose_to_change: np.ndarray,
    rotation_value: Union[float, List],
    rot_axis: Union[str, List] = "z",
    mode: str = "extrinsic",
    degrees: Union[bool, str] = None,
):
    if isinstance(rotation_value, (float, int, np.number)):
        replaced_rotation_matrix = get_rotated_pose(
            np.eye(4), rotation_value, rot_axis, mode, degrees
        )[:3, :3]
    elif isinstance(rotation_value, list):
        rotation_value = np.asarray(rotation_value)
        if rotation_value.shape == (3, 3):
            replaced_rotation_matrix == rotation_value
        elif rotation_value.shape == (3,):
            log_warning(
                f'Getting shape (3,) rotation_value {rotation_value} for "rotreplace", make sure it\'s rpy.'
            )
            replaced_rotation_matrix = R.from_euler("xyz", rotation_value).as_matrix()
        elif rotation_value.shape == (4,):
            log_warning(
                f'Getting shape (4,) rotation_value {rotation_value} for "rotreplace", make sure it\'s xyzw quaternion.'
            )
            replaced_rotation_matrix = R.from_quat(rotation_value).as_matrix()
        else:
            log_error(
                f'rotation_value has shape {rotation_value.shape}, not suppoorted by "rotreplace".'
            )
    else:
        log_error(
            f'rotation_value has type {type(rotation_value)}, not suppoorted by "rotreplace".'
        )
    rotation_replaced_pose = deepcopy(pose_to_change)
    rotation_replaced_pose[:3, :3] = replaced_rotation_matrix
    return rotation_replaced_pose


def get_frame_changed_pose(
    pose_to_change: np.ndarray,
    frame_change_matrix: Union[List, np.ndarray],
    mode: bool = "extrinsic",
    inverse: Union[bool, str] = False,
):
    if isinstance(frame_change_matrix, list):
        frame_change_matrix = np.asarray(frame_change_matrix)
    if not isinstance(frame_change_matrix, np.ndarray):
        log_error(
            f'frame_change_matrix has type{type(frame_change_matrix)} other than np.ndarray, not suppoorted by "get_frame_changed_pose".'
        )
    else:
        if frame_change_matrix.shape != (4, 4):
            log_error(
                f'frame_change_matrix has shape {frame_change_matrix.shape} other than (4,4), not suppoorted by "get_frame_changed_pose".'
            )

    if inverse == "inverse" or inverse == True:
        frame_change_matrix = inv_transform(frame_change_matrix)

    if mode == "extrinsic":
        pose_to_change = frame_change_matrix @ pose_to_change
    elif mode == "intrinsic":
        pose_to_change = pose_to_change @ frame_change_matrix
    else:
        log_error(f"Mode {mode} illegal.")

    return pose_to_change


def get_aligned_pose(
    pose_to_change: np.ndarray,
    align_vector: List,
    pose_axis: str = "z",
):
    align_vector = np.asarray(align_vector)
    pose_axis = axis_idx(pose_axis)
    rotation_axis = np.cross(pose_to_change[:3, pose_axis], align_vector)
    rotation_axis_norm = np.linalg.norm(rotation_axis)
    if rotation_axis_norm >= 1e-5:
        rotation_axis = rotation_axis / rotation_axis_norm
        rotation_angle = np.arccos(pose_to_change[:3, 2].dot(align_vector))
        pose_to_change[:3, :3] = (
            R.from_rotvec(rotation_axis * rotation_angle).as_matrix()
            @ pose_to_change[:3, :3]
        )
    return pose_to_change


# TODO: automatically routing,given kwargs automatically find the mode.
def get_changed_pose(
    pose_to_change: np.ndarray, pose_changes: List[Tuple[str, Any]] = []
):
    """Change the single pose given the `pose_changes` that indicates how to change the pose.

    Args:
        pose_to_change (np.ndarray): The pose to be changed.
        pose_changes (List[Tuple[str, Any]], optional): The list contains tuples that [0] refer to pose change name that indicates the change mode and parameters, split by "_", e.g. "offset_${np.array([0.05, -0.10, 0.125])}". And [1] be the change value, e.g. "${env.affordance_datas[\"cup_move_pose\"][:2,3]}". Defaults to [].

    Returns:
        pose_to_change (np.ndarray): The changed pose.
    """
    for pose_change_name, pose_change_value in pose_changes:
        change_partition = pose_change_name.split("_")
        change_mode = change_partition[0]
        if change_mode == "replace":
            pose_to_change = get_replaced_pose(
                pose_to_change,
                pose_replace_value=pose_change_value,
                axis_str_replace=change_partition[1],
            )
        elif change_mode == "offset":
            pose_to_change = get_offset_pose(
                pose_to_change, [pose_change_value], *change_partition[1:]
            )
        elif change_mode == "rotation":
            pose_to_change = get_rotated_pose(
                pose_to_change,
                pose_change_value,
                *change_partition[1:],
            )
        elif change_mode == "rotreplace":
            pose_to_change = get_rotation_replaced_pose(
                pose_to_change, pose_change_value, *change_partition[1:]
            )
        elif change_mode == "framechange":
            pose_to_change = get_frame_changed_pose(
                pose_to_change, pose_change_value, *change_partition[1:]
            )
        elif change_mode == "align":
            get_aligned_pose(pose_to_change, pose_change_value, change_partition[1])
        else:
            # TODO
            log_error(f"The {change_mode} change mode haven't realized yet!")
    return pose_to_change


def get_replaced_qpos(
    qpos_to_change: Union[np.ndarray, torch.Tensor],
    replace_value: Union[float, List[float]],
    joint_list_replace: List,
):
    if not isinstance(replace_value, Sequence):
        replace_value = [replace_value]
    for joint, replace_quantity in zip(joint_list_replace, replace_value):
        qpos_to_change[joint] = float(replace_quantity)
    return qpos_to_change


def get_offset_qpos(
    qpos_to_change: np.ndarray,
    offset_value: Union[float, List[float]],
    joint_list_offset: List,
    degrees: Union[str, bool, List[int]] = None,
) -> np.ndarray:
    if not isinstance(offset_value, Sequence):
        offset_value = [offset_value]

    degrees_joint_list = []
    if degrees is not None:
        if isinstance(degrees, str):
            degrees_all = degrees == "degrees"

            if degrees_all:
                degrees_joint_list = joint_list_offset
            else:
                degrees_joint_str = degrees.split("degrees")[1]
                for degerees_joint_idx_str in degrees_joint_str:
                    degrees_joint_list.append(int(degerees_joint_idx_str))
        elif isinstance(degrees, bool):
            degrees_all = degrees == True
            if degrees_all:
                degrees_joint_list = joint_list_offset
        elif isinstance(degrees, list):
            degrees_joint_list = degrees

    if not set(degrees_joint_list).issubset(set(joint_list_offset)):
        log_error(
            f"degrees_joint_list {degrees_joint_list}, not subset to joint_list_offset {joint_list_offset}."
        )

    for joint, offset_quantity in zip(joint_list_offset, offset_value):
        if joint in degrees_joint_list:
            offset_quantity = np.deg2rad(offset_quantity)
        qpos_to_change[joint] += offset_quantity

    return qpos_to_change


def get_changed_qpos(
    qpos_to_change: np.ndarray, qpos_changes: List[Tuple[str, Any]] = [], frame=None
):
    """Change the single qpos given the `qpos_changes` that indicates how to change the qpos.

    Args:
        qpos_to_change (np.ndarray): The qpos to be changed.
        qpos_changes (Dict[str, Any], optional): The list contains tuples that [0] refer to pose change name that indicates the change mode and parameters, split by "_", e.g. "offset_123". And [1] be the change value, e.g. "[0.5, 0.6, 0.7]". Defaults to []

    Returns:
        qpos_to_change (np.ndarray): The changed qpos.
    """
    if isinstance(qpos_to_change, torch.tensor):
        qpos_to_change = np.asarray(qpos_to_change)

    for qpos_change_name, qpos_change_value in qpos_changes:
        change_partition = qpos_change_name.split("_")
        change_mode = change_partition[0]
        if not isinstance(qpos_change_value, Sequence):
            qpos_change_value = [qpos_change_value]

        joint_str_change = change_partition[1]
        joint_list_change = []
        for joint_idx_str in joint_str_change:
            joint_list_change.append(int(joint_idx_str))
        if len(qpos_change_value) != len(joint_list_change):
            log_error(
                f'The joints asked to be raplaced is "{joint_str_change}", but got {len(qpos_change_value)} changes quantity.'
            )

        if change_mode == "replace":
            qpos_to_change = get_offset_qpos(
                qpos_to_change,
                replace_value=qpos_change_value,
                replace_joint_list=joint_list_change,
            )
        elif change_mode == "offset":
            qpos_to_change = get_offset_qpos(
                qpos_to_change,
                offset_value=qpos_change_value,
                offset_joint_list=joint_list_change,
                degrees=change_partition[2],
            )
        else:
            log_error(f"The {change_mode} change mode haven't realized yet!")
    return qpos_to_change


def camel_to_snake(name):
    # Insert underscores before each uppercase letter and convert to lowercase
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    snake_case = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
    return snake_case


def print_keys_recursively(h5file: h5py.Group, path="/"):
    """
    Recursively prints the keys in the HDF5 file.

    :param h5file: An open h5py File object.
    :param path: The current path in the HDF5 file.
    """
    for key in h5file[path].keys():
        print(f"{path}{key}")
        if isinstance(h5file[path + key], h5py.Group):
            print_keys_recursively(h5file, path + key + "/")


def hdf5_to_dict(h5file: h5py.Group):
    def recursive_dict(group):
        result = {}
        for key, item in group.items():
            if isinstance(item, h5py.Dataset):
                result[key] = item[()]
            elif isinstance(item, h5py.Group):
                result[key] = recursive_dict(item)
        return result

    return recursive_dict(h5file)


def extract_keys_hierarchically(d, assign_data_type: str = "list"):
    result = {}

    for key, value in d.items():
        # If the value is a dictionary, recursively extract its keys
        if isinstance(value, dict):
            result[key] = extract_keys_hierarchically(value)
        else:
            if assign_data_type == "null":
                result[key] = None
            elif assign_data_type == "list":
                result[key] = []
            else:
                raise ValueError(f"Invalid assign_data_type: {assign_data_type}")

    return result


def get_file_list(path: str, ext: str):
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(ext):
                file_list.append(os.path.join(root, file))

    return file_list


# Pattern to recognize an attribute or indexer sequence, e.g., foo, ["bar"], [0]
_TOKEN_RE = re.compile(
    r"""(
    (?P<attr>[A-Za-z_]\w*)   # attribute name
  | \[\s*(?P<index>"[^"]*"|'[^']*'|\d+)\s*\]  # bracket indexer with quoted key or integer
)""",
    re.VERBOSE,
)


def resolve_env_attr(obj: Any, env: Any) -> Any:
    """
    Recursively replace any string of the form 'env:...' by evaluating it as a Python expression on the given `env` object.
    Other containers (mappings, sequences) will be traversed and resolved element-wise.

    Supports:
      - Arbitrary attribute access (e.g. env.x.y.z)
      - Arbitrary indexing and slicing (e.g. env.x["key"][1:4])
      - Any valid Python expression after the 'env:' prefix.

    Args:
        obj: The object to resolve. If it's:
             - A dict-like Mapping: each value is passed back into resolve_env_attr.
             - A Sequence (list/tuple/etc.) but not str: each element is resolved.
             - A str starting with 'env:': the suffix is treated as a Python
               expression relative to `env` and eval'ed.
             - Anything else: returned unchanged.
        env: An object whose attributes, methods, indices, etc. may be
             referenced in the 'env:' expressions.

    Returns:
        The resolved object, with 'env:' strings replaced by their eval results.
    """
    # 1) If it's a mapping, recurse into its values
    if isinstance(obj, Mapping):
        return {k: resolve_env_attr(v, env) for k, v in obj.items()}

    # 2) If it's a non-str sequence, recurse into its elements
    if isinstance(obj, Sequence) and not isinstance(obj, str):
        return type(obj)(resolve_env_attr(item, env) for item in obj)

    # 3) If it's a string starting with "env.", eval it directly
    if isinstance(obj, str) and obj.startswith("env."):
        return eval(obj, {}, {"env": env})

    # 4) Everything else passes through unchanged
    return obj


_EXPR = re.compile(r"\$\{([^}]+)\}")  # For searching ${...} marker


def resolve_formatted_string(obj, local_vars=None, global_vars=None):
    """Given a dict carrys "${...}"-like strings , `eval` the "${...}$" values while keep the dict structure.

    Args:
        obj (Union[Dict, Sequence]): The original "Grand" dict or the iterables in it.
        local_vars (_type_, optional): _description_. Defaults to None.
        global_vars (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    # Gut the caller's locals & globals
    if local_vars is None or global_vars is None:
        frame = inspect.currentframe().f_back  # caller frame
        local_vars = local_vars or frame.f_locals
        global_vars = global_vars or frame.f_globals

    # 1) dict
    if isinstance(obj, Mapping):
        return {
            k: resolve_formatted_string(v, local_vars, global_vars)
            for k, v in obj.items()
        }

    # 2) list/tuple
    if isinstance(obj, Sequence) and not isinstance(obj, str):
        return type(obj)(
            resolve_formatted_string(v, local_vars, global_vars) for v in obj
        )

    # 3) str
    if isinstance(obj, str):
        full = _EXPR.fullmatch(obj.strip())
        if full:
            # the whole string is ${expr} -> return eval(expr)
            return eval(
                full.group(1),
                {"__builtins__": None},  # eval with given locals & globals
                {**global_vars, **local_vars},
            )

        # par tof the string is ${expr}：replace ...${expr}.. -> str(...eval(expr)..)
        def _sub(m):
            return str(
                eval(m.group(1), {"__builtins__": None}, {**global_vars, **local_vars})
            )

        return _EXPR.sub(_sub, obj)

    # 4) other type just return
    return obj


def resolve_params(resolve_func):
    """
    Decorator factory that applies `resolve_func` to each argument of the
    decorated function, with optional per-decorator `exclude` names.

    If `resolve_func`'s signature is: resolve_func(obj)
    then we call: resolve_func(val)

    If its signature is: resolve_func(obj, x, y, ...)
    then for each argument `val` of the decorated function we call: resolve_func(val, x=bound['x'], y=bound['y'], ...) pulling `x`, `y`, etc. by name from the decorated function's bound args.

    Usage patterns:

        # 1) create a decorator
        resolve_formatted_params = resolve_params(resolve_formatted_string) and use without exclude:
        @resolve_formatted_params
        def generate_func(a, b, c): ...

        # 2) use the same decorator with an exclude list:
        @resolve_formatted_params(exclude=['c'])
        def generate_func(a, b, c): ...

        # 3) or inline:
        @resolve_params(resolve_env_attr, exclude=['env'])
        def generate_func(env, path, mode): ...

    Args:
        resolve_func: function whose first parameter is the value to transform. Any additional parameters will be looked up by name in the decorated function's arguments.

    Returns:
        A decorator which can be used either as:
            @decorator
        or:
            @decorator(exclude=[...])
    """
    resolve_sig = inspect.signature(resolve_func)
    resolve_param_names = list(resolve_sig.parameters.keys())

    def decorator_factory(*, exclude=()):
        exclude = set(exclude)

        def decorator(func):
            func_sig = inspect.signature(func)

            @wraps(func)
            def wrapper(*args, **kwargs):
                bound = func_sig.bind_partial(*args, **kwargs)
                bound.apply_defaults()

                # Resolve each argument except those in exclude
                resolved = {}
                for name, val in bound.arguments.items():
                    if name in exclude:
                        resolved[name] = val
                        continue

                    try:
                        if len(resolve_param_names) == 1:
                            # single-arg resolver
                            resolved_val = resolve_func(val)
                        else:
                            # multi-arg resolver: gather extra args by name
                            extra_kwargs = {
                                pname: bound.arguments[pname]
                                for pname in resolve_param_names[1:]
                            }
                            resolved_val = resolve_func(val, **extra_kwargs)
                        resolved[name] = resolved_val
                    except Exception as e:
                        log_error(f"{e}")
                        resolved[name] = val

                # Rebuild positional and keyword args in original order
                args_to_pass = []
                kwargs_to_pass = {}
                for param in func_sig.parameters.values():
                    if param.kind in (
                        param.POSITIONAL_ONLY,
                        param.POSITIONAL_OR_KEYWORD,
                    ):
                        if param.name in resolved:
                            args_to_pass.append(resolved.pop(param.name))
                    elif param.kind is param.VAR_POSITIONAL:
                        args_to_pass.extend(resolved.pop(param.name, ()))
                    elif param.kind is param.KEYWORD_ONLY:
                        if param.name in resolved:
                            kwargs_to_pass[param.name] = resolved.pop(param.name)
                    elif param.kind is param.VAR_KEYWORD:
                        kwargs_to_pass.update(resolved.pop(param.name, {}))

                return func(*args_to_pass, **kwargs_to_pass)

            return wrapper

        return decorator

    def decorator_or_factory(func=None, *, exclude=()):
        # @decorator
        if func is not None and callable(func):
            return decorator_factory(exclude=())(func)
        # @decorator(exclude=[...])
        return decorator_factory(exclude=exclude)

    return decorator_or_factory


resolve_formatted_params = resolve_params(resolve_formatted_string)
resolve_env_params = resolve_params(resolve_env_attr)


def transfer_str_to_lambda(
    lambda_string: str, locals_dict: Dict = {}, globals_dict: Dict = {}
):
    """Transfer the string represented lambda function into a real lambda function.

    Args:
        lambda_string (str): The lambda string to be transfer
        locals_dict (dict): Read-only dict that carrys local variables for lambda function to use. Defaults to be {}.
        globals_dict (dict): Read-only dict that carrys global variables for lambda function to use. Defaults to be {}.

    Returns:
        lambda_function: The lambda function
    """
    # AST analyze
    node = ast.parse(lambda_string, mode="eval")
    # Assure the top to be a lambda
    if not isinstance(node.body, ast.Lambda):
        log_error(f'The lambda string "{lambda_string}" is not illegal.')
    # Compile to be a function object
    code = compile(node, filename="<ast>", mode="eval")
    # eval to be a real lambda function
    return eval(code, locals_dict, globals_dict)


def find_function(
    func_name: Union[str, Callable[..., Any]],
    instances: List = [],
    module_names: List[str] = [],
):
    """
    Finds and returns a function by its name from a list of instances or module names.

    Args:
        func_name (Union[str, Callable[..., Any]]): The name of the function to find,
            or the function itself.
        instances (List, optional): A list of instances to search for the function.
            Defaults to an empty list.
        module_names (List[str], optional): A list of module names to search for the function.
            Defaults to an empty list.

    Returns:
        Callable[..., Any] or bool: The found function if it exists, otherwise False.
    """
    if isinstance(func_name, str):
        if any(hasattr(instance := inst, func_name) for inst in instances):
            func = getattr(instance, func_name)
        elif any(
            hasattr((module := import_module(module_name)), func_name)
            for module_name in module_names
        ):
            func = getattr(module, func_name)
        else:
            return False
    else:
        func = func_name
    return func


def find_funcs_with_kwargs(
    funcs_name_kwargs_proc: List[Dict[str, Any]],
    instances: List,
    module_names: List,
):
    for func_name_kwargs_proc in funcs_name_kwargs_proc:
        func_name = func_name_kwargs_proc["name"]
        func = find_function(
            func_name,
            instances=instances,
            module_names=module_names,
        )
        if func != False:
            func_name_kwargs_proc.update({"func": func})
        else:
            log_warning(f"Function {func_name} not found, skipping...")

    return funcs_name_kwargs_proc


def validate_with_process(
    env,
    input: Any,
    valid_funcs_kwargs_proc: List[Dict[str, Any]],
):
    for valid_func_kwargs_proc in valid_funcs_kwargs_proc:
        validation_func = valid_func_kwargs_proc["func"]
        kwargs = valid_func_kwargs_proc["kwargs"]
        rejected_processes = valid_func_kwargs_proc.get("rejected_processes", None)
        pass_processes = valid_func_kwargs_proc.get("pass_processes", None)

        ret = validation_func(input, **kwargs)
        if not ret:
            log_warning(
                f"Validation function {validation_func.__name__} returns False."
            )
            if rejected_processes is not None:
                log_warning("Processing with rejected_processes..")
                for rejected_process in rejected_processes:
                    rejected_process_func_name = rejected_process["name"]
                    rejected_process_kwargs = rejected_process.get("kwargs", {})

                    rejected_process_func = find_function(
                        rejected_process_func_name,
                        instances=[env],
                        module_names=[
                            __name__,
                        ],
                    )
                    if rejected_process_func != False:
                        input = rejected_process_func(input, **rejected_process_kwargs)
                    else:
                        log_error(
                            f"rejected_process_func {rejected_process_func_name} after validation_func {validation_func.__name__} not found."
                        )
            else:
                log_warning("Skipping..")
                return None

        if pass_processes is not None:
            for pass_process in pass_processes:
                pass_process_func_name = pass_process["name"]
                pass_process_kwargs = pass_process.get("kwargs", {})

                pass_process_func = find_function(
                    pass_process_func_name,
                    instances=[env],
                    module_names=[
                        __name__,
                    ],
                )
                if pass_process_func != False:
                    input = pass_process_func(input, **pass_process_kwargs)
                else:
                    log_error(
                        f"pass_process_func {pass_process_func_name} after validation_func {validation_func.__name__} not found."
                    )

    return input


def validation_with_process_from_name(
    env,
    input: List[np.ndarray],
    valid_funcs_name_kwargs_proc: List[Dict[str, Any]],
    module_names: List[str] | None = None,
):
    """Apply a sequence of validation and processing functions (by name) to the input data.

    Args:
        env: The environment object, used for method lookup.
        input_data: The data to be validated and processed.
        valid_funcs_name_kwargs_proc: List of dicts, each specifying a function name and kwargs.
        module_names: List of module names to search for functions. Defaults to [__name__].

    Returns:
        The processed data if all validations pass, otherwise None.
    """
    if valid_funcs_name_kwargs_proc is None:
        valid_funcs_name_kwargs_proc = []
    if module_names is None:
        module_names = [__name__]

    valid_funcs_kwargs_proc = find_funcs_with_kwargs(
        valid_funcs_name_kwargs_proc,
        instances=[env],
        module_names=[__name__],
    )
    valid_output = validate_with_process(env, input, valid_funcs_kwargs_proc)
    return valid_output


def _get_valid_grasp(
    env,
    grasp_list: List[np.ndarray],
    valid_funcs_name_kwargs_proc: List[Union[str, Dict[str, Any]]],
) -> np.ndarray:
    """
    Validate a list of grasp poses using a sequence of validation and processing functions.

    This function iterates through each grasp in `grasp_list`, applies a series of validation
    and processing functions (specified in `valid_funcs_name_kwargs_proc`), and returns the first
    grasp pose that passes all validations. If no valid grasp is found, returns None.

    Args:
        env: The environment object, used for method lookup and as context for validation functions.
        grasp_list (List[np.ndarray]): List of grasp objects or poses to be validated.
        valid_funcs_name_kwargs_proc (List[Union[str, Dict[str, Any]]]): List of validation function
            specifications. Each item can be a function name (str) or a dict specifying the function
            name, kwargs, and optional processing steps.

    Returns:
        np.ndarray or None: The first valid grasp pose, or None if none are valid.
    """
    valid_func_kwargs_proc = find_funcs_with_kwargs(
        valid_funcs_name_kwargs_proc, instances=[env], module_names=[__name__]
    )

    for grasp in grasp_list:
        grasp_pose = grasp.pose  # TODO: Should this be a method?
        grasp_pose = validate_with_process(env, grasp_pose, valid_func_kwargs_proc)
        # Skip if any validation fails
        if grasp_pose is None:
            continue
        # Return the first valid grasp pose
        else:
            return grasp_pose
    return None


def lru_cache_n(maxsize: int = 10, max_count: int = 2) -> Callable:
    """
    Decorator to provide an LRU cache with a maximum call count per key.
    After a key is accessed `max_count` times, the result will be recomputed.

    Args:
        maxsize: Maximum number of cache entries.
        max_count: Number of times a cached result can be returned before recomputation.

    Returns:
        Decorator for caching function results.
    """

    def decorator(func):
        cache = OrderedDict()

        def _make_hashable(x):
            try:
                hash(x)
                return x
            except TypeError:
                if isinstance(x, np.ndarray):
                    return (x.shape, str(x.dtype), x.tobytes())
                if isinstance(x, dict):
                    return tuple(sorted((k, _make_hashable(v)) for k, v in x.items()))
                if isinstance(x, set):
                    return tuple(sorted(_make_hashable(i) for i in x))
                if isinstance(x, (list, tuple)):
                    return tuple(_make_hashable(i) for i in x)
                raise TypeError(f"Unhashable type in cache key: {type(x)}")

        @wraps(func)
        def wrapper(*args, **kwargs):
            key = (
                tuple(_make_hashable(a) for a in args),
                tuple(sorted((k, _make_hashable(v)) for k, v in kwargs.items())),
            )
            res, cnt = cache.pop(key, (None, max_count))
            if cnt == max_count:
                res, cnt = func(*args, **kwargs), 0
            cache[key] = (res, cnt + 1)
            if len(cache) > maxsize:
                cache.popitem(last=False)
            return res

        return wrapper

    return decorator


def multi_output_factory_function(
    func_name: Union[str, Callable],
    instances: List | None = None,
    module_names: List[str] | None = None,
    output_num: int = 1,
) -> Callable:
    """
    Factory to create a cached version of a function that may have multiple outputs.

    Args:
        func_name: Function name (str) or function object.
        instances: The instances that may carrys the method that match the func_name.
                   Usually be environment that carrys the methods wherein the function may be found.
        module_names: The list of modules that the function may be found. Defaults to [].
        output_num: Number of outputs expected from the function.

    Returns:
        Cached function callable.
    """
    if instances is None:
        instances = []
    if module_names is None:
        module_names = []
    func = find_function(func_name, instances, module_names)
    if not callable(func):
        raise ValueError(f"Function {func_name} not found or not callable.")

    max_count = max(1, output_num - 1)

    @lru_cache_n(max_count=max_count)
    def cached_func(*args, **kwargs):
        return func(*args, **kwargs)

    return cached_func


def cached_ik(
    target_xpos: np.ndarray,
    ik_func: Union[str, Callable],
    control_part: str,
    is_left: bool,
    qpos_seed: np.ndarray,
    instances: list = [],
    module_names: list = [],
) -> tuple:
    """
    Call the inverse kinematics (IK) function with caching for efficiency.

    Args:
        target_xpos: The target end-effector position (usually a numpy array).
        ik_func: The IK function or function name to be called.
        control_part: String of the cotrol part for IK computing.
        is_left: Whether the control part is on the left side.
        qpos_seed: The initial guess for the joint positions.
        instances: The instances that may carrys the method that match the func_name.
                   Usually be environment that carrys the methods wherein the function may be found.
        module_names: The list of modules that the function may be found. Defaults to [].

    Returns:
        Tuple: (ik_result, qpos_result), where ik_result is the IK status and qpos_result is the joint solution.
    """
    # cached_ik_func = multi_output_factory_function("_get_arm_ik", instances=[env], module_names=[__name__], output_num=2)
    cached_ik_func = multi_output_factory_function(
        ik_func, instances=instances, module_names=module_names, output_num=2
    )
    if control_part == "none":
        return cached_ik_func(target_xpos, is_left, qpos_seed)

    ret, qpos = cached_ik_func(torch.as_tensor(target_xpos), qpos_seed, control_part)
    if isinstance(ret, torch.Tensor):
        ret = ret.all().item()
    return ret, qpos.squeeze(0).cpu().numpy()


def get_ik_ret(
    target_xpos: np.ndarray,
    ik_func: Union[str, Callable],
    qpos_seed: np.ndarray,
    control_part: str = "none",
    is_left: bool = True,
    instances: list = [],
    module_names: list = [],
) -> bool:
    """
    Get the first return value from the cached IK function, typically the IK status or result flag.

    Args:
        target_xpos: The target end-effector position.
        ik_func: The IK function or function name to be called.
        control_part: String of the cotrol part for IK computing.
        qpos_seed: The initial guess for the joint positions.
        instances: The instances that may carrys the method that match the func_name.
                   Usually be environment that carrys the methods wherein the function may be found.
        module_names: The list of modules that the function may be found. Defaults to [].

    Returns:
        The first output of the IK function (e.g., success flag or status).
    """
    ret = cached_ik(
        target_xpos,
        ik_func,
        control_part,
        is_left,
        qpos_seed,
        instances=instances,
        module_names=module_names,
    )[0]
    return ret


def get_ik_qpos(
    target_xpos: np.ndarray,
    ik_func: Union[str, Callable],
    qpos_seed: np.ndarray,
    control_part: str = "none",
    is_left: bool = True,
    instances: list = [],
    module_names: list = [],
) -> np.ndarray:
    """
    Get the second return value from the cached IK function, typically the joint positions.

    Args:
        target_xpos: The target end-effector position.
        ik_func: The IK function or function name to be called.
        control_part: String of the control part for IK computing.
        is_left: Whether the control part is on the left side. Defaults to True.
        qpos_seed: The initial guess for the joint positions.
        instances: The instances that may carrys the method that match the func_name.
                   Usually be environment that carrys the methods wherein the function may be found.
        module_names: The list of modules that the function may be found. Defaults to [].

    Returns:
        The second output of the IK function (e.g., the joint position solution).
    """
    qpos = cached_ik(
        target_xpos,
        ik_func,
        control_part,
        is_left,
        qpos_seed,
        instances=instances,
        module_names=module_names,
    )[1]
    return qpos


def get_fk_xpos(
    target_qpos: np.ndarray,
    control_part: str,
    fk_func: Union[str, Callable],
) -> np.ndarray:
    xpos = fk_func(name=control_part, qpos=torch.as_tensor(target_qpos), to_matrix=True)

    # the xpos computed from robot is in the local arena frame, which is equivalent to world frame of the
    # old version.
    return xpos.squeeze(0).cpu().numpy()


# FIXME: remove
def data_key_to_control_part(robot, control_parts, data_key: str) -> str | None:
    # TODO: Temporary workaround, should be removed after refactoring data dict extractor.
    # @lru_cache(max_size=None) # NOTE: no way to pass a hashable parameter
    def is_eef_hand(robot, control_parts) -> bool:
        # TODO: This is a temporary workaround, should be used a more general method to check
        # whether the end-effector is a hand.
        for part in control_parts:
            if "eef" in part:
                joint_ids = robot.get_joint_ids(part, remove_mimic=True)
                return len(joint_ids) >= 2
        return False

    if "left_arm" in data_key:
        if "qpos" in data_key:
            return "left_arm"
        if "hand" in data_key and is_eef_hand(robot, control_parts):
            return "left_eef"
        if "gripper" in data_key and is_eef_hand(robot, control_parts) is False:
            return "left_eef"
        return None

    if "right_arm" in data_key:
        if "qpos" in data_key:
            return "right_arm"
        if "hand" in data_key and is_eef_hand(robot, control_parts):
            return "right_eef"
        if "gripper" in data_key and is_eef_hand(robot, control_parts) is False:
            return "right_eef"
        return None


def is_stereocam(sensor) -> bool:
    """
    Check if a sensor is a StereoCamera (binocular camera).

    Args:
        sensor: The sensor instance to check.

    Returns:
        bool: True if the sensor is a StereoCamera, False otherwise.
    """
    from embodichain.lab.sim.sensors import StereoCamera

    return isinstance(sensor, StereoCamera)
