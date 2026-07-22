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

"""Deterministic nominal task-graph construction for generated agent configs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import re
from typing import Any

__all__ = [
    "NominalGraphStep",
    "build_nominal_task_graph",
]

_SAFE_ID_CHARS_RE = re.compile(r"[^0-9A-Za-z_]+")
_MULTI_UNDERSCORE_RE = re.compile(r"_+")


@dataclass(frozen=True)
class NominalGraphStep:
    """One deterministic edge in a generated nominal graph."""

    semantic: str
    left_arm_action: Mapping[str, Any] | None = None
    right_arm_action: Mapping[str, Any] | None = None


def build_nominal_task_graph(
    *,
    task_name: str,
    steps: Sequence[NominalGraphStep],
) -> dict[str, Any]:
    """Build a validated single-chain graph from deterministic action steps."""
    if not steps:
        raise ValueError("Nominal task graph requires at least one step.")

    nodes: list[dict[str, str]] = [
        {
            "id": "v0_start",
            "semantic": "Initial state before executing the nominal action graph",
        }
    ]
    edges: list[dict[str, Any]] = []
    previous_node_id = "v0_start"
    final_index = len(steps)
    goal_node_id = f"v{final_index}_done"

    for index, step in enumerate(steps, start=1):
        if step.left_arm_action is None and step.right_arm_action is None:
            raise ValueError(f"Nominal graph step {index} has no arm action.")

        slug = _slugify(step.semantic)
        target_node_id = goal_node_id if index == final_index else f"v{index}_{slug}"
        nodes.append({"id": target_node_id, "semantic": step.semantic})
        edges.append(
            {
                "id": f"e{index:02d}_{slug}",
                "source": previous_node_id,
                "target": target_node_id,
                "left_arm_action": _copy_action(step.left_arm_action),
                "right_arm_action": _copy_action(step.right_arm_action),
            }
        )
        previous_node_id = target_node_id

    return {
        "task": task_name,
        "start": "v0_start",
        "goal": goal_node_id,
        "nodes": nodes,
        "edges": edges,
    }


def _copy_action(action: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if action is None:
        return None
    return deepcopy(dict(action))


def _slugify(text: str) -> str:
    slug = _SAFE_ID_CHARS_RE.sub("_", text.lower())
    slug = _MULTI_UNDERSCORE_RE.sub("_", slug).strip("_")
    if not slug:
        return "step"
    if not slug[0].isalpha():
        slug = f"step_{slug}"
    return slug[:64].rstrip("_") or "step"
