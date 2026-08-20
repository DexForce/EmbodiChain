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

"""Action Engine recipes layered over the Task Engine semantic ontology."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from embodichain.gen_sim.task_engine.ontology import (
    RELATIONS,
    TERMINAL_BEHAVIORS,
    TRANSPORT_DIRECTIONS,
    TaskContract as SemanticTaskContract,
    task_success_type,
)
from embodichain.gen_sim.task_engine.ontology import (
    TASK_CONTRACTS as SEMANTIC_TASK_CONTRACTS,
)

__all__ = [
    "PLACEMENT_RELATIONS",
    "RELATIONS",
    "TASK_CONTRACTS",
    "TERMINAL_BEHAVIORS",
    "TRANSPORT_DIRECTIONS",
    "TaskContract",
    "task_contract",
    "task_success_type",
    "normalize_placement_relation",
]

PLACEMENT_RELATIONS = RELATIONS - {"none"}
_SUPPORTED_PLACEMENT_ALIASES = frozenset({"above", "on_top", "on_top_of"})


def normalize_placement_relation(value: Any) -> str:
    """Lower task-language relations to physically executable release goals.

    A released object cannot remain freely hovering. Task-language ``above``
    therefore lowers to the supported ``on`` relation for placement operators;
    non-placement operators such as pouring retain their distinct ``above``
    semantics.
    """
    relation = str(value)
    if (
        relation not in PLACEMENT_RELATIONS
        and relation not in _SUPPORTED_PLACEMENT_ALIASES
    ):
        raise ValueError(f"Unsupported placement relation {relation!r}.")
    return "on" if relation in _SUPPORTED_PLACEMENT_ALIASES else relation


_CORE_ACTIONS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "E1": ("PickUp", "MoveHeldObject", "Place"),
        "E2": ("PickUp", "MoveHeldObject", "Place"),
        "E3": ("Pour",),
        "E4": ("PickUp", "MoveHeldObject", "HandOver"),
        "E5": ("CoordinatedPickment",),
        "E6": ("PullArticulatedPart",),
        "E7": ("PushArticulatedPart",),
        "E8": ("TurnKnob",),
        "E9": ("Press",),
    }
)


@dataclass(frozen=True, slots=True)
class TaskContract:
    """Action-facing view of one Task Engine semantic contract."""

    task_type: str
    semantics: str
    core_actions: tuple[str, ...]
    applicable_intent_fields: frozenset[str]
    source_structure: str
    required_affordances: frozenset[str]
    success_type: str
    scene_affordances: frozenset[str]


def _action_contract(value: SemanticTaskContract) -> TaskContract:
    return TaskContract(
        task_type=value.task_type,
        semantics=value.semantics,
        core_actions=_CORE_ACTIONS[value.task_type],
        applicable_intent_fields=value.applicable_intent_fields,
        source_structure=value.source_structure,
        required_affordances=value.required_affordances,
        success_type=value.success_type,
        scene_affordances=value.scene_affordances,
    )


TASK_CONTRACTS: Mapping[str, TaskContract] = MappingProxyType(
    {
        task_type: _action_contract(contract)
        for task_type, contract in SEMANTIC_TASK_CONTRACTS.items()
    }
)


def task_contract(task_type: str) -> TaskContract:
    """Return the Action Engine view of one canonical task contract."""
    try:
        return TASK_CONTRACTS[str(task_type)]
    except KeyError as exc:
        raise ValueError(f"Unsupported task type {task_type!r}.") from exc
