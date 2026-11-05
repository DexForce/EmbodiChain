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

"""Resolve selected embodiment-owned runtime calibration without changing physics."""

from __future__ import annotations
from collections.abc import Mapping
from copy import deepcopy
import hashlib
from pathlib import Path
from typing import Any
from embodichain.utils.config_paths import resolve_config_path
from embodichain.utils.utility import load_config
from embodichain.lab.task_program.integrations.hand_calibration import (
    TaskHandCalibration,
)
from embodichain_robots.task_program import (
    declare_source_task_embodiment,
    resolve_task_hand_calibrations,
)

__all__ = [
    "declare_source_opening_bindings",
    "resolve_source_hand_services",
    "declare_source_grip_evidence",
]

_COMPONENTS = {
    identity: f"extensions/embodichain_tasks/configs/components/embodiments/{identity}.yaml"
    for identity in (
        "dual_franka_robotiq_arg2f_140",
        "dual_ur10_robotiq_arg2f_140",
        "tianji_marvin",
    )
}


def resolve_source_hand_services(
    embodiment: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[TaskHandCalibration, ...], dict[str, str]]:
    """Supply omitted historical geometry services from their canonical component owner.

    Args:
        embodiment: Explicit frozen source robot/sensor/skill-profile declaration.

    Returns:
        Independent component declaration, model-owned hand role descriptors and
        any additional canonical component file digests. Effective native physics
        and sensors retain their existing owner; standard presets can be expanded
        to complete declarations. Already-present geometry declarations
        retain their values; no code-default calibration silently replaces them.
    """
    owned = declare_source_task_embodiment(embodiment)
    services = owned["skill_profile"]["runtime_services"]
    inputs = {}
    if "hand_geometries" not in services:
        identity = owned.get("embodiment_id")
        if identity not in _COMPONENTS:
            raise ValueError(
                "Source hand services require an explicitly qualified embodiment."
            )
        path = Path(resolve_config_path(_COMPONENTS[identity])).resolve()
        original = path.read_bytes()
        canonical = load_config(path)
        if canonical.get("embodiment_id") != identity:
            raise ValueError("Canonical hand-service component identity changed.")
        services["hand_geometries"] = deepcopy(
            canonical["skill_profile"]["runtime_services"]["hand_geometries"]
        )
        if path.read_bytes() != original:
            raise ValueError("Canonical hand-service component changed during reading.")
        inputs[str(path)] = hashlib.sha256(original).hexdigest()
    hands = resolve_task_hand_calibrations(owned)
    return owned, hands, inputs


def declare_source_grip_evidence(
    embodiment: dict[str, Any],
    hands: tuple[TaskHandCalibration, ...],
    scene_binding: Mapping[str, Any],
    *,
    experimental_open_release: bool = False,
) -> dict[str, Any]:
    """Move source evidence selection to an explicit native-contact declaration.

    Args:
        embodiment: Owned mutable declaration; its former evidence selection is
            removed so composition retains exactly one owner.
        hands: Explicit model-owned pad and independent-driver identities.
        scene_binding: Complete canonical rigid/affordance declarations.
        experimental_open_release: Preserve the source opt-in driver-open release.

    Returns:
        Closed native evidence declaration with source object order, capacities
        and pad scope. No sensor is constructed or queried.
    """
    if type(experimental_open_release) is not bool:
        raise TypeError("experimental_open_release must be an explicit boolean.")
    services = embodiment["skill_profile"]["runtime_services"]
    source = services.get("control_part_evidence")
    if (
        not isinstance(source, Mapping)
        or source.get("kind") != "joint_position_constraint"
    ):
        raise ValueError(
            "Source native grip requires its declared joint evidence profile."
        )
    by_part = {hand.hand_part: hand for hand in hands}
    parts = tuple(source["control_parts"])
    if any(part not in by_part for part in parts):
        raise ValueError("Source evidence control parts lack qualified hand roles.")
    objects = scene_binding["rigid_objects"]
    by_id = {item["entity_id"]: item for item in objects}
    ids = source.get("object_ids") or [
        item["entity_id"]
        for item in objects
        if any(
            affordance["kind"] == "antipodal_grasp"
            for affordance in item["affordances"]
        )
    ]
    result = dict(
        kind="native_grip",
        sensor_uid="gen_sim_grip_contacts",
        algorithm_revision=2 if experimental_open_release else 1,
        object_bindings=[
            [identity, by_id[identity]["simulation_uid"]] for identity in ids
        ],
        pad_bindings=[[part, list(by_part[part].pad_link_names)] for part in parts],
        max_contacts_per_env=1024,
        trace_capacity=4096,
    )
    if experimental_open_release:
        result["opening_bindings"] = [
            dict(
                control_part=part,
                joint_names=list(embodiment["simulation"]["control_parts"][part]),
                driver_indices=list(
                    by_part[part].opening_driver_indices(
                        tuple(embodiment["simulation"]["control_parts"][part])
                    )
                ),
                open_qpos=list(source["open_qpos"]),
                tolerance=source["minimum_displacement"],
            )
            for part in parts
        ]
    del services["control_part_evidence"]
    return result


def declare_source_opening_bindings(
    embodiment: Mapping[str, Any],
    hands: tuple[TaskHandCalibration, ...],
) -> list[dict[str, Any]]:
    """Declare the source native driver-opening observation without sampling.

    Args:
        embodiment: Original component with its source joint-evidence profile.
        hands: Model-owned pad/independent-driver calibration descriptors.

    Returns:
        Closed public DriverOpeningBinding declarations in source part order.
        The model owner selects driver indices; no hand name infers geometry.
    """
    source = embodiment["skill_profile"]["runtime_services"]["control_part_evidence"]
    if source["kind"] != "joint_position_constraint":
        raise ValueError(
            "Source opening declarations require the original explicit joint profile."
        )
    by_part = {hand.hand_part: hand for hand in hands}
    return [
        dict(
            control_part=part,
            joint_names=list(embodiment["simulation"]["control_parts"][part]),
            driver_indices=list(
                by_part[part].opening_driver_indices(
                    tuple(embodiment["simulation"]["control_parts"][part])
                )
            ),
            open_qpos=list(source["open_qpos"]),
            tolerance=source["minimum_displacement"],
        )
        for part in source["control_parts"]
    ]
