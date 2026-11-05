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

"""Read-only model-owned hand descriptions for declarative Task Program assembly."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Protocol

__all__ = ["TaskHandCalibration", "physical_hand_resources"]


class TaskHandCalibration(Protocol):
    """Describe one physical hand without selecting a task, motion or validator.

    Concrete model/embodiment owners supply topology, commands and nominal
    reserves. Integrations bind these declarations to existing public services;
    physical acquisition and normative acceptance remain separate owners.
    """

    resource_id: str
    side: str
    grasp_target_id: str
    motion_part: str
    hand_part: str
    command_preset: str
    hand_root_link: str
    pad_link_names: tuple[str, str]
    finger_link_names: tuple[str, ...]
    geometry: Mapping[str, Any]

    def opening_driver_indices(self, joint_names: tuple[str, ...]) -> tuple[int, ...]:
        """Select independent drivers from the complete authored hand joint order."""
        ...

    def geometry_declaration(self) -> dict[str, Any]:
        """Return explicit shared hand-sampling inputs without capturing geometry."""
        ...

    def nominal_envelope_declaration(self) -> dict[str, Any]:
        """Return model-owned nominal reserves, independently of native geometry."""
        ...

    def nominal_pour_radius(self, max_opening_width: float) -> float:
        """Return the declared nominal swept-hand reserve for an explicit opening."""
        ...

    def revolute_grip_calibration(self) -> Mapping[str, Any] | None:
        """Return an explicit model qualification, or the generic conservative mode."""
        ...


def physical_hand_resources(profile: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Resolve explicit one-hand aliases to unique physical endpoint bindings.

    Args:
        profile: Existing resource/endpoints/member declaration from a skill profile.

    Returns:
        Owned leaf resource declarations in their original order. A member alias
        is not another physical hand. No control-part name implies a capability.

    Raises:
        ValueError: Resources are duplicated, cyclic or bind conflicting aliases.
    """
    rows = tuple(profile["resources"])
    by_id = {row["resource_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Hand resources require unique declared identities.")

    def leaf(identity: str, seen: frozenset[str] = frozenset()) -> str:
        if identity in seen or identity not in by_id:
            raise ValueError("Hand resource aliases are cyclic or undeclared.")
        members = tuple(by_id[identity].get("members", ()))
        if not members:
            return identity
        if len(members) != 1:
            raise ValueError("One-hand aliases require one explicit resource member.")
        return leaf(members[0], seen | {identity})

    result = []
    parts = set()
    for row in rows:
        endpoints = {item["endpoint_id"]: item for item in row.get("endpoints", ())}
        if "grasp" not in endpoints:
            continue
        if "motion" not in endpoints or len(endpoints) != len(row["endpoints"]):
            raise ValueError(
                "A physical hand requires unambiguous motion/grasp endpoints."
            )
        identity = row["resource_id"]
        selected = leaf(identity)
        if selected != identity:
            actual = {
                item["endpoint_id"]: item for item in by_id[selected]["endpoints"]
            }
            if any(
                endpoints.get(name) != actual.get(name) for name in ("motion", "grasp")
            ):
                raise ValueError(
                    "Hand alias endpoints differ from their explicit physical member."
                )
            continue
        key = endpoints["grasp"]["control_part"]
        if key in parts:
            raise ValueError(
                "Duplicate physical hand bindings require an explicit member alias."
            )
        parts.add(key)
        result.append(deepcopy(dict(row)))
    return tuple(result)
