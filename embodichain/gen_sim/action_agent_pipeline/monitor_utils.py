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

from __future__ import annotations

import numpy as np
import torch

from embodichain.gen_sim.action_agent_pipeline.atom_action_utils import get_arm_states
from embodichain.utils.logger import log_error
from embodichain.utils.math import matrix_from_quat


def _to_tensor(
    value: torch.Tensor | np.ndarray | list | tuple | float,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convert input to a tensor."""
    if isinstance(value, torch.Tensor):
        return value.to(device=device or value.device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _as_pose_matrix(
    pose: torch.Tensor | np.ndarray | list | tuple | dict,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Convert a pose-like input into a 4x4 pose matrix."""
    if isinstance(pose, dict):
        if "pose" not in pose:
            log_error("Pose dict must contain key 'pose'.")
        pose = pose["pose"]

    pose_tensor = _to_tensor(pose, device=device)

    if pose_tensor.dim() == 3 and pose_tensor.shape[0] == 1:
        pose_tensor = pose_tensor.squeeze(0)

    if pose_tensor.shape == (4, 4):
        return pose_tensor

    if pose_tensor.dim() == 1 and pose_tensor.shape[0] == 7:
        pose_matrix = torch.eye(4, dtype=torch.float32, device=pose_tensor.device)
        pose_matrix[:3, 3] = pose_tensor[:3]
        pose_matrix[:3, :3] = matrix_from_quat(pose_tensor[3:].unsqueeze(0)).squeeze(0)
        return pose_matrix

    log_error(
        f"Unsupported pose format {tuple(pose_tensor.shape)}. Expected (4, 4) or (7,)."
    )


def _get_rigid_object(env, obj_name: str):
    """Fetch a rigid object by name."""
    obj_uids = env.sim.get_rigid_object_uid_list()
    if obj_name not in obj_uids:
        log_error(
            f"Rigid object '{obj_name}' not found. Available objects: {obj_uids}."
        )
    return env.sim.get_rigid_object(obj_name)


def _get_object_pose(env, obj_name: str) -> torch.Tensor:
    """Get the current 4x4 local pose of a rigid object."""
    return _get_rigid_object(env, obj_name).get_local_pose(to_matrix=True).squeeze(0)


def _get_actual_arm_pose(env, robot_name: str) -> torch.Tensor:
    """Get the current end-effector pose of the selected arm."""
    is_left, control_part, _, _, _ = get_arm_states(env, robot_name)
    arm_joints = env.left_arm_joints if is_left else env.right_arm_joints
    arm_qpos = env.robot.get_qpos().squeeze(0)[arm_joints]
    arm_pose = env.robot.compute_fk(
        arm_qpos, name=control_part, to_matrix=True
    ).squeeze(0)
    return arm_pose


def capture_object_state(env, obj_name: str) -> dict[str, torch.Tensor]:
    """Capture the current object pose for frame-to-frame monitoring."""
    pose = _get_object_pose(env, obj_name)
    return {
        "pose": pose.clone(),
        "position": pose[:3, 3].clone(),
    }


def get_gripper_distance(env, robot_name: str) -> float:
    """Estimate the current gripper opening distance."""
    is_left, _, _, _, _ = get_arm_states(env, robot_name)
    side = "left" if is_left else "right"
    if hasattr(env, "get_agent_eef_control_part"):
        eef_control_part = env.get_agent_eef_control_part(is_left)
    else:
        eef_control_part = f"{side}_eef"
    if eef_control_part is None:
        return 0.0
    eef_qpos = env.robot.get_qpos(name=eef_control_part).squeeze(0)
    return float(torch.mean(torch.abs(eef_qpos)).item())


def get_arm_object_distance(env, robot_name: str, obj_name: str) -> float:
    """Compute the distance between the current arm end-effector and object."""
    arm_pose = _get_actual_arm_pose(env, robot_name)
    obj_pose = _get_object_pose(env, obj_name)
    return float(torch.norm(arm_pose[:3, 3] - obj_pose[:3, 3]).item())
