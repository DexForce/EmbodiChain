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

"""Human-readable diagnostics derived directly from an executable Seed."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["make_seed_diagnostic_records"]


def make_seed_diagnostic_records(
    seed_graph: Mapping[str, Any],
    *,
    task_description: str,
    project_name: str,
    robot_display_name: str,
) -> dict[str, str]:
    """Return review records that cannot disagree with the authoritative Seed."""
    steps = list(seed_graph["semantic_steps"])
    edges = list(seed_graph["edges"])
    group_by_step = {
        str(step_id): group
        for group in seed_graph.get("allocation_groups", ())
        for step_id in group["semantic_step_ids"]
    }
    step_lines = []
    for step in steps:
        actor = _actor_text(step["actor"])
        group = group_by_step.get(str(step["id"]))
        if group is not None:
            actor += (
                f"; group={group['id']} ({group['arm_constraint']}, "
                f"{group['execution_policy']})"
            )
        step_lines.append(
            f"- {step['id']}: {step['operator']} `{step['object']}`; "
            f"actor={actor}; depends_on={step['depends_on'] or 'none'}; "
            f"goal={_goal_text(step['goal'])}."
        )

    edge_lines = []
    for edge in edges:
        action_parts = []
        for action in edge["actions"]:
            binding = action["target_binding"]
            binding_text = str(binding["kind"])
            if binding.get("object"):
                binding_text += f":{binding['object']}"
            if binding.get("semantic_step"):
                binding_text += f":{binding['semantic_step']}"
            if binding.get("phase"):
                binding_text += f":{binding['phase']}"
            action_parts.append(
                f"{action['atomic_action_class']} "
                f"[{_actor_text(action['actor'])}; target={binding_text}; "
                f"policy={action['motion_policy']}]"
            )
        edge_lines.append(
            f"- {edge['id']}: {edge['source']} -> {edge['target']}; "
            f"depends_on={edge['depends_on'] or 'none'}; " + " | ".join(action_parts)
        )

    task_prompt = "\n".join(
        (
            f"Task: {seed_graph['task']}",
            "",
            f"Original instruction: {task_description}",
            f"Project: {project_name}",
            "",
            "Authoritative Seed semantic program:",
            *step_lines,
            "",
            "Runtime resolves every auto actor, live object pose, target pose, IK, "
            "trajectory, and motion-policy value per environment. Only declared "
            "PickUp allocation groups may execute in parallel; shared-target "
            "transport and placement remain serial.",
        )
    )
    basic_background = "\n".join(
        (
            f"Robot: {robot_display_name}",
            f"Seed schema: {seed_graph['schema_version']}",
            f"Motion policy: {seed_graph['motion_policy_version']}",
            "",
            "This file is generated from seed_task_graph.json. It is diagnostic "
            "only and does not define a second plan.",
            "",
            "Semantic steps:",
            *step_lines,
        )
    )
    atom_actions = "\n".join(
        (
            "Authoritative symbolic Seed edges:",
            "",
            *edge_lines,
            "",
            "No config-stage coordinates, resolved arm assignment, IK result, or "
            "trajectory is part of these action records.",
        )
    )
    return {
        "task_prompt": task_prompt,
        "basic_background": basic_background,
        "atom_actions": atom_actions,
    }


def _actor_text(actor: Mapping[str, Any]) -> str:
    mode = str(actor["mode"])
    if mode == "required":
        return f"required:{actor['arm']}"
    if mode == "coordinated":
        return "coordinated:left_arm+right_arm"
    return "auto(runtime)"


def _goal_text(goal: Mapping[str, Any]) -> str:
    fields = []
    for key in (
        "relation",
        "reference_object",
        "reference_state",
        "orientation_goal",
        "orientation_axis",
    ):
        if key in goal:
            fields.append(f"{key}={goal[key]}")
    return ", ".join(fields) or "symbolic"
