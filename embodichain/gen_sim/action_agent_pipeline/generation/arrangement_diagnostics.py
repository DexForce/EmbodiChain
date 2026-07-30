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

"""Render coordinate-free arrangement diagnostics from the executable Seed."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from embodichain.gen_sim.action_agent_pipeline.generation.diagnostic_common import (
    _format_runtime_object_registry,
    _robot_context,
)
from embodichain.gen_sim.action_agent_pipeline.generation.robot_profiles import (
    DEFAULT_ROBOT_PROFILE_ID,
    RobotProfile,
    resolve_robot_profile,
)

__all__ = [
    "make_arrangement_task_prompt",
    "make_arrangement_basic_background",
    "make_arrangement_atom_actions_prompt",
]


def make_arrangement_task_prompt(
    task_name: str,
    project_name: str,
    seed_graph: Mapping[str, Any],
    *,
    robot_profile: RobotProfile | str = DEFAULT_ROBOT_PROFILE_ID,
    task_description: str = "",
) -> str:
    profile = resolve_robot_profile(robot_profile)
    steps = _arrangement_steps(seed_graph)
    first_goal = steps[0]["goal"]
    step_lines = [
        (
            f"- {step['id']}: place `{step['object']}` in nominal slot "
            f"{step['goal']['nominal_slot_index']} "
            f"({step['goal']['slot_constraint']}); execute edges "
            f"{', '.join(step['edge_ids'])}."
        )
        for step in steps
    ]
    return "\n".join(
        [
            f"# Task `{task_name}`",
            "",
            task_description or "Arrange the selected objects into one line.",
            "",
            f"- Project: `{project_name}`",
            f"- Robot: {profile.display_name}",
            f"- Axis: `{first_goal['axis']}`",
            f"- Anchor: `{first_goal['anchor']}`",
            f"- Order constraint: `{first_goal['order_constraint']}`",
            "- Runtime contract: resolve table bounds, spacing, target poses, and "
            "active arm independently for every environment.",
            "",
            "## Seed Semantic Steps",
            *step_lines,
        ]
    )


def make_arrangement_basic_background(
    project_name: str,
    seed_graph: Mapping[str, Any],
    *,
    robot_profile: RobotProfile | str = DEFAULT_ROBOT_PROFILE_ID,
    object_registry: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    profile = resolve_robot_profile(robot_profile)
    object_lines = "\n".join(
        (
            f"- `{step['object']}`: nominal slot "
            f"{step['goal']['nominal_slot_index']}, "
            f"constraint={step['goal']['slot_constraint']}, actor=auto."
        )
        for step in _arrangement_steps(seed_graph)
    )
    registry = _format_runtime_object_registry(object_registry)
    return "\n".join(
        [
            f"# Arrangement Background for `{project_name}`",
            "",
            f"Robot: {profile.display_name}",
            _robot_context(profile),
            "",
            "The Seed is coordinate-free. Object poses, table geometry, motion "
            "policy values, and semantic-to-physical arm mappings are runtime data.",
            "",
            "## Nominal Bindings",
            object_lines,
            "",
            "## Runtime Object Registry",
            registry,
        ]
    )


def make_arrangement_atom_actions_prompt(
    seed_graph: Mapping[str, Any],
    *,
    robot_profile: RobotProfile | str = DEFAULT_ROBOT_PROFILE_ID,
) -> str:
    profile = resolve_robot_profile(robot_profile)
    edge_by_id = {str(edge["id"]): edge for edge in seed_graph["edges"]}
    blocks: list[str] = []
    for step in _arrangement_steps(seed_graph):
        action_lines: list[str] = []
        for edge_id in step["edge_ids"]:
            action = edge_by_id[str(edge_id)]["actions"][0]
            binding = action["target_binding"]
            phase = f", phase={binding['phase']}" if "phase" in binding else ""
            action_lines.append(
                f"- `{edge_id}`: {action['atomic_action_class']}; "
                f"actor={action['actor']['mode']}; binding={binding['kind']}"
                f"{phase}; policy={action['motion_policy']}."
            )
        blocks.extend([f"## {step['id']} ({step['object']})", *action_lines, ""])
    return "\n".join(
        [
            f"# Symbolic Atomic Actions for {profile.display_name}",
            "",
            "All geometric and arm-selection fields are resolved at runtime.",
            "",
            *blocks,
        ]
    )


def _arrangement_steps(
    seed_graph: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    if seed_graph.get("route") != "arrangement_line":
        raise ValueError("Arrangement diagnostics require an arrangement Seed.")
    steps = seed_graph.get("semantic_steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("Arrangement diagnostics require semantic steps.")
    return list(steps)
