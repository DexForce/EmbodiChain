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

from embodichain.utils.logger import log_error
from embodichain.lab.sim.agent.monitor_utils import (
    _as_pose_matrix,
    _get_object_pose,
    get_arm_object_distance,
    get_gripper_distance,
)
import numpy as np
import torch


def monitor_object_moved(
    env,
    obj_name: str,
    last_frame_pose: torch.Tensor | np.ndarray | list | tuple | dict = None,
    threshold: float = 0.01,
    **kwargs,
) -> bool:
    """Trigger when an object moved from the last frame beyond a threshold.

    Args:
        env: The current agent environment.
        obj_name: Target rigid object name.
        last_frame_pose: Previous-frame pose or a state dict returned by
            :func:`capture_object_state`.
        threshold: Maximum allowed translation change in meters.

    Returns:
        ``True`` if the monitored failure occurs, i.e. the object moved more than
        the threshold.
    """
    if last_frame_pose is None:
        last_frame_pose = env.obj_info.get(obj_name).get("pose")
    current_pose = _get_object_pose(env, obj_name)
    previous_pose = _as_pose_matrix(last_frame_pose, device=current_pose.device)
    movement = torch.norm(current_pose[:3, 3] - previous_pose[:3, 3]).item()
    return movement > threshold


def monitor_object_held(
    env,
    robot_name: str,
    obj_name: str,
    threshold: float = 0.05,
    **kwargs,
) -> bool:
    """Trigger when an object is no longer being held by the corresponding arm.

    The function name is historical. To keep all monitor semantics consistent,
    this function returns ``True`` when failure occurs, namely when the object is
    too far from the corresponding arm end-effector and is treated as no longer held.

    Args:
        env: The current agent environment.
        robot_name: Robot-side selector containing ``left`` or ``right``.
        obj_name: Target rigid object name.
        threshold: Maximum allowed arm-object distance before hold failure is triggered.

    Returns:
        ``True`` if the monitored failure occurs, i.e. the object is no longer
        close enough to the corresponding arm.
    """
    arm_object_distance = get_arm_object_distance(env, robot_name, obj_name)
    return arm_object_distance > threshold


# TODO: not used currently
def monitor_gripper_distance(
    env,
    robot_name: str,
    threshold: float = 0.01,
    comparison: str = "less",
    **kwargs,
) -> bool:
    """Trigger when gripper distance violates the expected threshold condition.

    Args:
        env: The current agent environment.
        robot_name: Robot-side selector containing ``left`` or ``right``.
        threshold: Threshold for the gripper distance.
        comparison: ``"less"`` checks ``distance < threshold`` and ``"greater"``
            checks ``distance > threshold``.

    Returns:
        ``True`` if the monitored failure occurs and the comparison condition is unsatisfied.
    """
    distance = get_gripper_distance(env, robot_name)

    if comparison == "less":
        return distance < threshold
    if comparison == "greater":
        return distance > threshold

    log_error(f"Unsupported comparison '{comparison}'. Expected 'less' or 'greater'.")
