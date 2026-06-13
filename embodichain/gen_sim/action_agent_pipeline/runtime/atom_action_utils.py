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

import ast
from typing import List

from embodichain.utils.logger import log_error


def _available_arm_sides(env) -> list[str]:
    sides = []
    for side in ("left", "right"):
        if len(getattr(env, f"{side}_arm_joints", []) or []) > 0:
            sides.append(side)
    return sides


def resolve_arm_side(env, robot_name: str) -> str:
    """Resolve robot_name to an available left/right graph slot."""
    name = robot_name or ""
    if "right" in name:
        side = "right"
    elif "left" in name:
        side = "left"
    else:
        sides = _available_arm_sides(env)
        side = "right" if sides == ["right"] else "left"

    if side not in _available_arm_sides(env):
        log_error(
            f"Requested {side}_arm for robot_name='{robot_name}', but available "
            f"control parts are {getattr(env.robot, 'control_parts', None)}.",
            error_type=ValueError,
        )
    return side


def get_arm_states(env, robot_name):
    """Get the current state of the specified robot arm.

    Args:
        env: The simulation environment.
        robot_name: Name of the robot arm (should contain "left" or "right").

    Returns:
        Tuple of (is_left, select_arm, current_qpos, current_pose, current_gripper_state):
            - is_left: bool, whether this is the left arm
            - select_arm: str, arm identifier ("left_arm" or "right_arm")
            - current_qpos: Current joint positions
            - current_pose: Current end-effector pose (4x4 matrix)
            - current_gripper_state: Current gripper state
    """
    left_arm_current_qpos, right_arm_current_qpos = env.get_current_qpos_agent()
    left_arm_current_pose, right_arm_current_pose = env.get_current_xpos_agent()
    left_arm_current_gripper_state, right_arm_current_gripper_state = (
        env.get_current_gripper_state_agent()
    )

    side = resolve_arm_side(env, robot_name)
    is_left = True if side == "left" else False
    if hasattr(env, "get_agent_arm_control_part"):
        select_arm = env.get_agent_arm_control_part(is_left)
    else:
        select_arm = "left_arm" if is_left else "right_arm"

    arms = {
        "left": (
            left_arm_current_qpos,
            left_arm_current_pose,
            left_arm_current_gripper_state,
        ),
        "right": (
            right_arm_current_qpos,
            right_arm_current_pose,
            right_arm_current_gripper_state,
        ),
    }
    (
        select_arm_current_qpos,
        select_arm_current_pose,
        select_arm_current_gripper_state,
    ) = arms[side]

    return (
        is_left,
        select_arm,
        select_arm_current_qpos,
        select_arm_current_pose,
        select_arm_current_gripper_state,
    )


def extract_drive_calls(code_str: str) -> List[str]:
    """Extract all drive() function calls from a code string.

    Args:
        code_str: Python code string to parse.

    Returns:
        List of code blocks containing drive() calls.
    """
    tree = ast.parse(code_str)
    lines = code_str.splitlines()

    drive_blocks = []

    for node in tree.body:
        # Match: drive(...)
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "drive"
        ):
            # AST line numbers are 1-based
            start = node.lineno - 1
            end = node.end_lineno
            block = "\n".join(lines[start:end])
            drive_blocks.append(block)

    return drive_blocks


def apply_offset_to_pose(pose, offset: list):
    pose[0, 3] += offset[0]
    pose[1, 3] += offset[1]
    pose[2, 3] += offset[2]
    return pose


def resolve_action(action, env, kwargs):
    if callable(action):
        return action(env=env, **kwargs)
    return action


def sync_agent_state_from_robot(env) -> None:
    """Synchronize cached agent arm states from the physical robot state."""
    action = env.robot.get_qpos().squeeze(0)
    for side in ("left", "right"):
        is_left = side == "left"
        arm_joints = getattr(env, f"{side}_arm_joints", [])
        eef_joints = getattr(env, f"{side}_eef_joints", [])
        if arm_joints:
            arm_qpos = action[arm_joints]
            env.set_current_qpos_agent(arm_qpos, is_left=is_left)
            env.set_current_xpos_agent(
                env.get_arm_fk(qpos=arm_qpos, is_left=is_left),
                is_left=is_left,
            )
        if eef_joints:
            env.set_current_gripper_state_agent(
                action[eef_joints][0].unsqueeze(0),
                is_left=is_left,
            )
