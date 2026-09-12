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
"""Backend-specific joint-drive accessors for articulations."""

from __future__ import annotations

import numpy as np

__all__ = ["apply_joint_drive", "read_drive_properties"]


def read_drive_properties(entity: object, *, is_newton: bool) -> tuple[object, ...]:
    """Read backend-native joint drive values."""
    if is_newton:
        return tuple(entity.get_newton_drive())
    return tuple(entity.get_drive())


def apply_joint_drive(
    entity: object,
    *,
    is_newton: bool,
    joint_ids: object,
    row_index: int,
    resolved_drive_type: str | None,
    target_mode_value: int | None,
    stiffness: object | None = None,
    damping: object | None = None,
    max_effort: object | None = None,
    max_velocity: object | None = None,
    friction: object | None = None,
    armature: object | None = None,
) -> None:
    """Translate and apply one public joint-drive row for a backend."""

    def _drive_arg(value: object) -> float | np.ndarray:
        result = value[row_index].detach().cpu().numpy()
        return result.item() if result.size == 1 else result

    if is_newton:
        if resolved_drive_type == "acceleration" and target_mode_value in {1, 2, 3}:
            raise NotImplementedError(
                "Newton Spawn does not have an exact equivalent of the Default "
                "acceleration drive. Use drive_type='force' or disable the drive."
            )
        drive_args: dict[str, object] = {"joint_ids": joint_ids}
        if target_mode_value is not None:
            drive_args["target_mode"] = target_mode_value
        if stiffness is not None:
            drive_args["target_ke"] = _drive_arg(stiffness)
        if damping is not None:
            drive_args["target_kd"] = _drive_arg(damping)
        if max_effort is not None:
            drive_args["effort_limit"] = _drive_arg(max_effort)
        if max_velocity is not None:
            drive_args["velocity_limit"] = _drive_arg(max_velocity)
        if friction is not None:
            drive_args["friction"] = _drive_arg(friction)
        if armature is not None:
            drive_args["armature"] = _drive_arg(armature)
        if target_mode_value in {0, 4}:
            drive_args["target_ke"] = 0.0
            drive_args["target_kd"] = 0.0
        elif target_mode_value == 2:
            drive_args["target_ke"] = 0.0
        entity.set_newton_drive(**drive_args)
        return

    drive_args = {"joint_ids": joint_ids}
    default_drive_type = resolved_drive_type
    if target_mode_value in {0, 4}:
        default_drive_type = "none"
    elif target_mode_value in {1, 2, 3} and default_drive_type is None:
        default_drive_type = "force"
    if default_drive_type is not None:
        # Import lazily to keep this backend helper independent of the object
        # facade and avoid importing simulation utility modules at package load.
        from embodichain.lab.sim.utility.sim_utils import get_dexsim_drive_type

        drive_args["drive_type"] = get_dexsim_drive_type(default_drive_type)
    if stiffness is not None:
        drive_args["stiffness"] = _drive_arg(stiffness)
    if damping is not None:
        drive_args["damping"] = _drive_arg(damping)
    if max_effort is not None:
        drive_args["max_force"] = _drive_arg(max_effort)
    if max_velocity is not None:
        drive_args["max_velocity"] = _drive_arg(max_velocity)
    if friction is not None:
        drive_args["joint_friction"] = _drive_arg(friction)
    if armature is not None:
        drive_args["armature"] = _drive_arg(armature)
    if target_mode_value in {0, 4}:
        drive_args["stiffness"] = 0.0
        drive_args["damping"] = 0.0
    elif target_mode_value == 2:
        drive_args["stiffness"] = 0.0
    entity.set_drive(**drive_args)
