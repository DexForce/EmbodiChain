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

"""Scene-independent semantic ontology for the canonical E1-E9 tasks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

__all__ = [
    "RELATIONS",
    "TASK_CONTRACTS",
    "TERMINAL_BEHAVIORS",
    "TRANSPORT_DIRECTIONS",
    "TaskContract",
    "task_contract",
    "task_success_type",
]


# These are protocol values consumed by executable planners. They are not a
# vocabulary for matching words in user instructions.
RELATIONS = frozenset(
    {
        "none",
        "on",
        "inside",
        "above",
        "left_of",
        "right_of",
        "front_of",
        "behind",
        "front_left_of",
        "front_right_of",
        "back_left_of",
        "back_right_of",
    }
)
TRANSPORT_DIRECTIONS = frozenset(
    {
        "none",
        "world_x",
        "world_y",
        "front",
        "back",
        "left",
        "right",
        "front_left",
        "front_right",
        "back_left",
        "back_right",
        "up",
        "down",
    }
)
TERMINAL_BEHAVIORS = frozenset({"none", "hold", "place"})
_RESOURCE_MODES = frozenset({"single_arm", "handover", "coordinated"})


@dataclass(frozen=True, slots=True)
class TaskContract:
    """One scene-independent semantic E-task contract."""

    task_type: str
    semantics: str
    applicable_intent_fields: frozenset[str]
    source_structure: str
    required_affordances: frozenset[str]
    success_type: str
    scene_affordances: frozenset[str]
    primary_role_field: str
    resource_mode: str
    moves_primary_object: bool
    accepts_direct_payloads: bool
    direct_payload_relations: frozenset[str]
    accepts_incoming_hold: bool
    terminal_success_types: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not self.primary_role_field.endswith("_role"):
            raise ValueError("primary_role_field must name one role parameter.")
        if self.resource_mode not in _RESOURCE_MODES:
            raise ValueError(f"Unknown task resource_mode {self.resource_mode!r}.")
        if self.direct_payload_relations - RELATIONS:
            raise ValueError("direct_payload_relations contain unknown relations.")
        terminal_behaviors = [item[0] for item in self.terminal_success_types]
        if len(terminal_behaviors) != len(set(terminal_behaviors)):
            raise ValueError("terminal_success_types must use unique behaviors.")
        if set(terminal_behaviors) - TERMINAL_BEHAVIORS:
            raise ValueError("terminal_success_types contain unknown behaviors.")


def _contract(
    task_type: str,
    semantics: str,
    applicable_intent_fields: frozenset[str],
    source_structure: str,
    required_affordances: frozenset[str],
    success_type: str,
    *,
    scene_affordances: frozenset[str] | None = None,
    primary_role_field: str = "object_role",
    resource_mode: str = "single_arm",
    moves_primary_object: bool = False,
    accepts_direct_payloads: bool = False,
    direct_payload_relations: frozenset[str] = frozenset(),
    accepts_incoming_hold: bool = False,
    terminal_success_types: tuple[tuple[str, str], ...] = (),
) -> TaskContract:
    return TaskContract(
        task_type=task_type,
        semantics=semantics,
        applicable_intent_fields=applicable_intent_fields,
        source_structure=source_structure,
        required_affordances=required_affordances,
        success_type=success_type,
        scene_affordances=scene_affordances or required_affordances,
        primary_role_field=primary_role_field,
        resource_mode=resource_mode,
        moves_primary_object=moves_primary_object,
        accepts_direct_payloads=accepts_direct_payloads,
        direct_payload_relations=direct_payload_relations,
        accepts_incoming_hold=accepts_incoming_hold,
        terminal_success_types=terminal_success_types,
    )


TASK_CONTRACTS: Mapping[str, TaskContract] = MappingProxyType(
    {
        "E1": _contract(
            "E1",
            "Pick, move, and place one object at a symbolic relation.",
            frozenset(
                {
                    "target",
                    "relation",
                    "required_arm",
                    "orientation_goal",
                    "layout",
                    "axis",
                }
            ),
            "rigid_object",
            frozenset({"graspable", "placeable"}),
            "semantic_goal",
            moves_primary_object=True,
            accepts_direct_payloads=True,
            direct_payload_relations=frozenset({"on", "inside"}),
            accepts_incoming_hold=True,
        ),
        "E2": _contract(
            "E2",
            "Make one fallen object upright and place it stably.",
            frozenset({"required_arm", "orientation_goal"}),
            "rigid_object",
            frozenset({"graspable", "orientable"}),
            "object_upright",
            moves_primary_object=True,
            accepts_incoming_hold=True,
        ),
        "E3": _contract(
            "E3",
            "Pick up a source container, execute a tilt-and-restore pour over "
            "a fixed target container, then place and home.",
            frozenset({"target", "relation", "required_arm"}),
            "rigid_object",
            frozenset({"graspable", "pourable"}),
            "poured",
            primary_role_field="source_role",
            moves_primary_object=True,
            accepts_incoming_hold=True,
        ),
        "E4": _contract(
            "E4",
            "Transfer one held object from one arm to the other.",
            frozenset({"transfer_arm", "receive_arm", "orientation_goal"}),
            "rigid_object",
            frozenset({"graspable", "handover"}),
            "handover_complete",
            resource_mode="handover",
            moves_primary_object=True,
            accepts_incoming_hold=True,
        ),
        "E5": _contract(
            "E5",
            "Use both arms to pick, move, and optionally release one shared rigid object.",
            frozenset({"target", "relation", "direction", "terminal_behavior"}),
            "rigid_object",
            frozenset({"dual_graspable"}),
            "held_by_both_grippers",
            scene_affordances=frozenset({"dual_graspable", "rigid"}),
            resource_mode="coordinated",
            moves_primary_object=True,
            accepts_direct_payloads=True,
            terminal_success_types=(
                ("hold", "held_by_both_grippers"),
                ("place", "semantic_goal"),
            ),
        ),
        "E6": _contract(
            "E6",
            "Pull an articulated part to its requested state.",
            frozenset({"required_arm", "target_state"}),
            "articulation",
            frozenset({"pullable"}),
            "articulation_joint_near",
            scene_affordances=frozenset({"articulated", "pullable"}),
        ),
        "E7": _contract(
            "E7",
            "Push an articulated part to its requested state.",
            frozenset({"required_arm", "target_state"}),
            "articulation",
            frozenset({"pushable"}),
            "articulation_joint_near",
            scene_affordances=frozenset({"articulated", "pushable"}),
        ),
        "E8": _contract(
            "E8",
            "Turn one knob to a requested setting.",
            frozenset({"required_arm", "target_setting"}),
            "articulation",
            frozenset({"turnable"}),
            "articulation_joint_near",
        ),
        "E9": _contract(
            "E9",
            "Press one button until its requested terminal state.",
            frozenset({"required_arm", "target_state"}),
            "articulation",
            frozenset({"pressable"}),
            "pressed",
        ),
    }
)


def task_contract(task_type: str) -> TaskContract:
    """Return the canonical contract or reject an unknown E-task type."""
    try:
        return TASK_CONTRACTS[str(task_type)]
    except KeyError as exc:
        raise ValueError(f"Unsupported task type {task_type!r}.") from exc


def task_success_type(
    task_type: str,
    params: Mapping[str, Any] | None = None,
) -> str:
    """Resolve a TaskSpec success type, including E5's terminal behavior."""
    contract = task_contract(task_type)
    if not contract.terminal_success_types:
        return contract.success_type
    terminal_behavior = str((params or {}).get("terminal_behavior", "hold"))
    success_by_behavior = dict(contract.terminal_success_types)
    try:
        return success_by_behavior[terminal_behavior]
    except KeyError as exc:
        raise ValueError(
            f"{contract.task_type} terminal_behavior must be one of "
            f"{sorted(success_by_behavior)}."
        ) from exc
