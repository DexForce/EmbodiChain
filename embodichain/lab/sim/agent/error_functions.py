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

import random
import re

import numpy as np
import torch

from embodichain.lab.gym.utils.misc import apply_rotation
from embodichain.lab.sim.agent.atom_action_utils import (
    apply_offset_to_pose,
    finalize_actions,
    get_qpos,
    plan_trajectory,
)
from embodichain.utils.logger import log_warning

__all__ = [
    "judge_active_arms",
    "object_error_types",
    "action_error_types",
    "update_scene",
    "misplaced_object",
    "fallen_object",
    "setup_interactive_error_input",
    "restore_interactive_error_input",
    "interactive_error_requested",
    "inject_interactive_error",
    "wrong_affordance",
]


def judge_active_arms(action_str: str) -> list[str]:
    right_active = re.search(r"\bright_arm_action\s*=", action_str) is not None
    left_active = re.search(r"\bleft_arm_action\s*=", action_str) is not None
    right_none = re.search(r"\bright_arm_action\s*=\s*None\b", action_str) is not None
    left_none = re.search(r"\bleft_arm_action\s*=\s*None\b", action_str) is not None

    active_arms = []
    if left_active and not left_none:
        active_arms.append("left_arm")
    if right_active and not right_none:
        active_arms.append("right_arm")

    if len(active_arms) == 0:
        raise ValueError(
            "No active arm detected: both right_arm_action and left_arm_action are None."
        )

    return active_arms


object_error_types = [
    "misplaced_object",
    "fallen_object",
]
action_error_types = [
    "wrong_affordance",
]


def update_scene(env) -> None:
    env.sim.update(step=100)


def misplaced_object(env, error_obj, error_pose, relative_error_xyz) -> None:
    if error_pose is not None:
        obj_pose = torch.as_tensor(error_pose).clone()
        if obj_pose.ndim == 3:
            obj_pose = obj_pose.squeeze(0)
    else:
        obj_pose = (
            env.sim.get_rigid_object(error_obj)
            .get_local_pose(to_matrix=True)
            .squeeze(0)
            .clone()
        )
        obj_pose[0, 3] += relative_error_xyz[0]
        obj_pose[1, 3] += relative_error_xyz[1]
        obj_pose[2, 3] += relative_error_xyz[2]

    env.sim.get_rigid_object(error_obj).set_local_pose(obj_pose.unsqueeze(0))
    update_scene(env)


def fallen_object(env, error_obj, error_pose, relative_error_xyz) -> None:
    if error_pose is not None:
        obj_pose = torch.as_tensor(error_pose).clone()
        if obj_pose.ndim == 3:
            obj_pose = obj_pose.squeeze(0)
    else:
        obj_pose = (
            env.sim.get_rigid_object(error_obj)
            .get_local_pose(to_matrix=True)
            .squeeze(0)
            .clone()
        )
        obj_pose[0, 3] += relative_error_xyz[0]
        obj_pose[1, 3] += relative_error_xyz[1]
        obj_pose[2, 3] += relative_error_xyz[2]
        obj_pose = apply_rotation(obj_pose, "x", 90)
        obj_pose = apply_rotation(obj_pose, "y", 90)

    env.sim.get_rigid_object(error_obj).set_local_pose(obj_pose.unsqueeze(0))
    update_scene(env)


def setup_interactive_error_input(enabled: bool = False):
    if not enabled:
        return None

    import sys
    import termios
    import tty

    stdin = sys.stdin
    if not hasattr(stdin, "fileno") or not stdin.isatty():
        return None

    fd = stdin.fileno()
    try:
        old_settings = termios.tcgetattr(fd)
    except termios.error:
        return None

    tty.setcbreak(fd)
    return stdin, old_settings


def restore_interactive_error_input(interactive_input) -> None:
    if interactive_input is None:
        return

    import termios

    stdin, old_settings = interactive_input
    termios.tcsetattr(stdin.fileno(), termios.TCSADRAIN, old_settings)


def interactive_error_requested(interactive_input) -> bool:
    if interactive_input is None:
        return False

    import select

    stdin, _ = interactive_input
    readable, _, _ = select.select([stdin], [], [], 0)
    if not readable:
        return False

    key = stdin.read(1)
    return key.lower() == "f"


def _parse_relative_error_xyz(value: str) -> list[float]:
    value = value.strip().translate(
        str.maketrans({"[": " ", "]": " ", "(": " ", ")": " "})
    )
    parts = [part for part in re.split(r"[,\s]+", value) if part]
    if len(parts) != 3:
        raise ValueError(
            "relative_error_xyz must contain exactly three numbers, "
            "for example: 0.02, 0.02, 0 or [0.02, 0.02, 0]"
        )
    try:
        return [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError(
            "relative_error_xyz must contain exactly three numbers, "
            "for example: 0.02, 0.02, 0 or [0.02, 0.02, 0]"
        ) from exc


def _get_interactive_error_objects(env) -> list[str]:
    if hasattr(env, "sim") and hasattr(env.sim, "get_rigid_object_uid_list"):
        return [obj for obj in env.sim.get_rigid_object_uid_list() if obj != "table"]
    return [obj for obj in getattr(env, "obj_info", {}).keys() if obj != "table"]


def _select_interactive_error_object(env) -> str:
    object_names = _get_interactive_error_objects(env)
    if len(object_names) == 0:
        raise ValueError("No object is available for interactive error injection.")

    print("Objects:")
    for index, obj_name in enumerate(object_names, start=1):
        print(f"{index}: {obj_name}")

    selection = ""
    while selection == "":
        selection = input("Select error object: ").strip()
    if not selection.isdigit():
        raise ValueError(f"Unsupported object selection: {selection}.")

    index = int(selection)
    if index < 1 or index > len(object_names):
        raise ValueError(f"Unsupported object selection: {selection}.")

    return object_names[index - 1]


def inject_interactive_error(env) -> None:
    print("\nInteractive error injection requested.")
    print("1: misplaced_object")
    print("2: fallen_object")
    print("E: exit without injecting failure")
    error_choice = ""
    while error_choice == "":
        error_choice = input("Select error type: ").strip()
    if error_choice.lower() == "e":
        log_warning("Interactive error injection canceled.")
        return
    if error_choice not in {"1", "2"}:
        raise ValueError(f"Unsupported interactive error selection: {error_choice}.")

    error_obj = _select_interactive_error_object(env)
    print("relative_error_xyz example: 0.02, 0.02, 0 or [0.02, 0.02, 0]")
    relative_error_xyz = _parse_relative_error_xyz(
        input("relative_error_xyz (x, y, z): ")
    )

    if error_choice == "1":
        misplaced_object(
            env,
            error_obj=error_obj,
            error_pose=None,
            relative_error_xyz=relative_error_xyz,
        )
    elif error_choice == "2":
        fallen_object(
            env,
            error_obj=error_obj,
            error_pose=None,
            relative_error_xyz=relative_error_xyz,
        )

    log_warning(
        f"Injected interactive error {error_choice} on {error_obj} with relative_error_xyz={relative_error_xyz}."
    )


def wrong_affordance(env, action, error_arm, error_pose, relative_error_xyz):
    if action is None:
        raise ValueError("wrong_affordance requires a compiled arm action ndarray.")

    action = np.array(action, copy=True)
    if action.ndim != 2 or action.shape[1] < 3:
        raise ValueError(
            f"wrong_affordance expects action with shape [T, arm_dof+2], got {action.shape}."
        )

    is_left = "left" in error_arm
    select_arm = "left_arm" if is_left else "right_arm"
    start_qpos = torch.as_tensor(action[0, :-2], dtype=torch.float32)
    last_qpos = torch.as_tensor(action[-1, :-2], dtype=torch.float32)
    gripper_state_traj = action[:, -1:].copy()
    last_gripper_state = gripper_state_traj[-1]
    last_pose = env.get_arm_fk(qpos=last_qpos, is_left=is_left).clone()

    if error_pose is not None:
        target_pose = torch.as_tensor(error_pose).clone()
        if target_pose.ndim == 3:
            target_pose = target_pose.squeeze(0)
    else:
        if relative_error_xyz is None:
            raise ValueError(
                "wrong_affordance requires either error_pose or relative_error_xyz."
            )
        target_pose = apply_offset_to_pose(last_pose, relative_error_xyz)

    _, disturbed_qpos = get_qpos(
        env,
        is_left,
        select_arm,
        target_pose,
        last_qpos,
        force_valid=False,
        name="wrong affordance",
    )

    select_qpos_traj = []
    plan_trajectory(
        env,
        select_arm,
        [start_qpos, disturbed_qpos],
        len(action),
        last_gripper_state,
        select_qpos_traj,
        [],
    )

    return finalize_actions(select_qpos_traj, gripper_state_traj)
