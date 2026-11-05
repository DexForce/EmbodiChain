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

"""Public model-owned hand/deployment descriptions for configured Task Programs."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from embodichain.lab.task_program.integrations.hand_calibration import (
        TaskHandCalibration,
    )

__all__ = [
    "declare_source_task_embodiment",
    "resolve_task_hand_calibrations",
    "declare_task_robot_mount",
    "declare_task_contact_pads",
    "task_robot_lateral_positions",
    "require_task_robot_assets",
]


def declare_source_task_embodiment(embodiment: Mapping[str, Any]) -> dict[str, Any]:
    """Expand supported model services without changing shared components.

    Args:
        embodiment: Explicit qualified component declaration.

    Returns:
        Owned source integration additions from its real model owner.

    Raises:
        ValueError: No implemented owner supports the selected component.
    """
    if embodiment.get("embodiment_id") == "tianji_marvin":
        from .tianji.marvin.task_program import declare_tianji_task_program_embodiment

        return declare_tianji_task_program_embodiment(embodiment)
    if embodiment.get("embodiment_id") not in {
        "dual_franka_robotiq_arg2f_140",
        "dual_ur10_robotiq_arg2f_140",
    }:
        raise ValueError("No configured task-hand owner for the selected embodiment.")
    return deepcopy(dict(embodiment))


def resolve_task_hand_calibrations(
    embodiment: Mapping[str, Any],
) -> tuple[TaskHandCalibration, ...]:
    """Resolve an explicitly selected model's public hand description."""
    if embodiment.get("embodiment_id") == "tianji_marvin":
        from .tianji.marvin.task_program import tianji_marvin_task_hand_calibrations

        return tianji_marvin_task_hand_calibrations(embodiment)
    from .robotiq.task_program import robotiq140_task_hand_calibrations

    return robotiq140_task_hand_calibrations(embodiment)


def declare_task_robot_mount(
    embodiment: Mapping[str, Any], *, table_top_z: float, table_bounds: Any = None
) -> dict[str, Any]:
    """Delegate declared scene placement to the selected physical model owner."""
    if embodiment.get("embodiment_id") == "tianji_marvin":
        from .tianji.marvin.deployment import source_tianji_mount_declaration

        return source_tianji_mount_declaration(
            embodiment, table_top_z=table_top_z, table_bounds=table_bounds
        )
    from .robotiq.deployment import source_robot_mount_declaration

    return source_robot_mount_declaration(
        embodiment, table_top_z=table_top_z, table_bounds=table_bounds
    )


def declare_task_contact_pads(embodiment: Mapping[str, Any]) -> dict[str, Any]:
    """Apply an explicit experiment to a supported model's pad declarations.

    Args:
        embodiment: Explicit qualified component with its real hand bindings.

    Returns:
        Owned material overlay from the selected model owner.

    Raises:
        ValueError: No implemented owner supports the selected component.
    """
    if embodiment.get("embodiment_id") == "tianji_marvin":
        from .tianji.marvin.deployment import source_tianji_contact_pad_declaration

        return source_tianji_contact_pad_declaration(embodiment)
    if embodiment.get("embodiment_id") not in {
        "dual_franka_robotiq_arg2f_140",
        "dual_ur10_robotiq_arg2f_140",
    }:
        raise ValueError("No configured task-hand owner for the selected embodiment.")
    from .robotiq.deployment import source_contact_pad_declaration

    return source_contact_pad_declaration(embodiment)


def task_robot_lateral_positions(
    identity: str, embodiment: Mapping[str, Any] | None = None
) -> dict[str, float] | None:
    """Return model-owned semantic hints without initializing a robot runtime."""
    if identity == "tianji_marvin":
        from .tianji.marvin.deployment import source_tianji_lateral_positions

        return source_tianji_lateral_positions(embodiment)
    from .robotiq.deployment import source_robot_lateral_positions

    return source_robot_lateral_positions(identity, embodiment)


def require_task_robot_assets(identity: str) -> None:
    """Preflight selected model assets without loading another hand model."""
    if identity == "tianji_marvin":
        from .tianji.marvin.deployment import require_tianji_task_assets

        require_tianji_task_assets()
        return
    from .robotiq.deployment import require_source_robot_assets

    require_source_robot_assets(identity)
