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

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from embodichain.lab.task_program.integrations.hand_calibration import (
        TaskHandCalibration,
    )

__all__ = [
    "TaskGripEffortCandidate",
    "declare_task_source_execution_calibration",
    "declare_source_task_embodiment",
    "resolve_task_hand_calibrations",
    "declare_task_robot_mount",
    "declare_task_contact_pads",
    "task_robot_lateral_positions",
    "require_task_robot_assets",
]


@dataclass(frozen=True, slots=True)
class TaskGripEffortCandidate:
    """Semantic resource eligibility and declared payload masses, without a recipe.

    The frontend supplies exact resource identity, aggregate eligibility and
    whether the acquisition carries a pouring payload. Masses are physical
    declaration values in kilograms; missing or invalid values cannot qualify
    a reduced drive limit. No task-family ID or object name is interpreted here.

    Attributes:
        resource_id: Exact physical resource declaration identity.
        eligible: Whether semantic acquisition/carrier relationships qualify.
        pouring: Whether the acquisition supports a pouring payload.
        payload_masses: Declared payload masses in kilograms; None is missing.
    """

    resource_id: str
    eligible: bool
    pouring: bool
    payload_masses: tuple[float | int | None, ...]

    def __post_init__(self) -> None:
        if (
            type(self.resource_id) is not str
            or not self.resource_id
            or self.resource_id != self.resource_id.strip()
        ):
            raise ValueError(
                "Source grip candidates require exact resource identities."
            )
        if type(self.eligible) is not bool or type(self.pouring) is not bool:
            raise TypeError("Source grip eligibility and pouring must be booleans.")
        object.__setattr__(self, "payload_masses", tuple(self.payload_masses))


def declare_task_source_execution_calibration(
    embodiment: Mapping[str, Any],
    candidates: Sequence[TaskGripEffortCandidate],
    *,
    runtime_profile: str,
    requires_prismatic_stiffness: bool,
    initial_prismatic_clearance: bool,
    active_resource_id: str | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Compose selected model calibration from neutral semantic requirements.

    Args:
        embodiment: Existing qualified physical component declaration.
        candidates: Resource eligibility and declared payload masses.
        runtime_profile: Explicit default, contact or composition profile.
        requires_prismatic_stiffness: Enable the original source-prismatic gain.
        initial_prismatic_clearance: Whether the first declared move needs the
            original prismatic clearance posture, if owned by this model.
        active_resource_id: Explicit first-move resource, or None when absent.

    Returns:
        Independent component and optional model-owned initial-posture receipt.
        Standard Marvin retains its existing component values; Robotiq model
        owners keep the original mass/effort/gain/park calibration. No task ID,
        object name, cached analysis, Robot or simulator is consumed.

    Raises:
        TypeError: Calibration selection flags are not explicit booleans.
        ValueError: The selected model or runtime profile is unsupported.
    """
    if (
        type(requires_prismatic_stiffness) is not bool
        or type(initial_prismatic_clearance) is not bool
    ):
        raise TypeError("Source calibration requirements must be explicit booleans.")
    if runtime_profile not in {
        "default",
        "contact_experiment",
        "composition_experiment",
    }:
        raise ValueError("Source calibration requires an explicit supported profile.")
    identity = embodiment.get("embodiment_id")
    if identity == "tianji_marvin":
        return deepcopy(dict(embodiment)), None
    if identity not in {
        "dual_franka_robotiq_arg2f_140",
        "dual_ur10_robotiq_arg2f_140",
    }:
        raise ValueError("No configured source calibration owner for the component.")
    from .robotiq.deployment import (
        source_arm_stiffness_declaration,
        source_grip_effort_caps,
        source_grip_effort_declaration,
        source_ur10_idle_posture_declaration,
    )

    owned = source_grip_effort_declaration(
        embodiment,
        source_grip_effort_caps(
            embodiment, candidates, runtime_profile=runtime_profile
        ),
    )
    posture = None
    if (
        identity == "dual_ur10_robotiq_arg2f_140"
        and initial_prismatic_clearance
        and active_resource_id in {"left", "right"}
    ):
        owned, posture = source_ur10_idle_posture_declaration(
            owned, active_resource_id=active_resource_id
        )
    if requires_prismatic_stiffness:
        owned = source_arm_stiffness_declaration(owned)
    return owned, posture


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
