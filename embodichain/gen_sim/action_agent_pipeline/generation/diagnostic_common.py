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

"""Shared formatting helpers for human-readable generation diagnostics."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from embodichain.gen_sim.action_agent_pipeline.generation.builder_protocols import (
    RelativePlacementLike,
    RelativeSpecLike,
)
from embodichain.gen_sim.action_agent_pipeline.generation.robot_profiles import (
    RobotProfile,
    resolve_robot_profile,
)
from embodichain.gen_sim.action_agent_pipeline.semantics import (
    relative_relation_phrase as _canonical_relative_relation_phrase,
)

__all__ = [
    "_format_runtime_object_registry",
    "_one_line_registry_text",
    "_format_indexed_edge_blocks",
    "_format_numbered_edge_blocks",
    "_robot_context",
    "_format_action_sketch",
    "_relative_reference_line",
    "_relative_final_planning_rule",
    "_dual_relative_final_planning_rule",
    "_relative_relation_phrase",
    "_display_noun",
    "_plural",
]


def _format_runtime_object_registry(
    object_registry: Sequence[Mapping[str, Any]] | None,
) -> str:
    if not object_registry:
        return ""

    lines = []
    for item in object_registry:
        runtime_uid = str(item.get("runtime_uid", "")).strip()
        source_uid = str(item.get("source_uid", "")).strip()
        if not runtime_uid or not source_uid:
            continue
        role = str(item.get("source_role", item.get("role", ""))).strip()
        description = _one_line_registry_text(item.get("description", ""))
        role_text = f", role `{role}`" if role else ""
        description_text = (
            json.dumps(description, ensure_ascii=False)
            if description
            else '"No source description."'
        )
        lines.append(
            f"- runtime_uid `{runtime_uid}` maps to source_uid `{source_uid}`"
            f"{role_text}; description: {description_text}"
        )
    if not lines:
        return ""

    return (
        "\nRuntime object registry:\n" + "\n".join(lines) + "\n\nRegistry rules:\n"
        "- Descriptions are read-only semantic hints for identifying objects.\n"
        "- In every generated graph action, use only `runtime_uid` values as "
        "`obj_name`, `align_to`, `support`, `support_uid`, and object pose "
        "reference ids.\n"
        "- Do not copy `source_uid`, `description`, or registry metadata into "
        "the action JSON.\n"
    )


def _one_line_registry_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _format_indexed_edge_blocks(
    edge_blocks: Sequence[tuple[str, Mapping[str, str | None]]],
    *,
    start_index: int,
) -> str:
    formatted_blocks = []
    for index, (title, actions) in enumerate(edge_blocks, start=start_index):
        action_lines = "\n".join(
            f"   - {slot}: {action if action is not None else 'null'}"
            for slot, action in actions.items()
        )
        formatted_blocks.append(f"{index}. {title}:\n{action_lines}")
    return "\n\n".join(formatted_blocks)


def _format_numbered_edge_blocks(
    edge_blocks: Sequence[tuple[str, Mapping[str, str | None]]],
) -> str:
    formatted_blocks = []
    for index, (title, actions) in enumerate(edge_blocks, start=1):
        action_lines = "\n".join(
            f"   - {slot}: {action if action is not None else 'null'}"
            for slot, action in actions.items()
        )
        formatted_blocks.append(f"{index}. {title}:\n{action_lines}")
    return "\n\n".join(formatted_blocks)


def _robot_context(robot_profile: RobotProfile | str | None) -> str:
    return resolve_robot_profile(robot_profile).prompt_robot_context()


def _format_action_sketch(action_sketch: list[str]) -> str:
    return "\n".join(f"- {item}" for item in action_sketch)


def _relative_reference_line(spec: RelativePlacementLike) -> str:
    if getattr(spec, "upright_in_place", False):
        return (
            f"Use `{spec.reference_runtime_uid}` as the support surface while "
            f"anchoring XY at the initial position of `{spec.moved_runtime_uid}`."
        )
    if getattr(spec, "reference_is_initial_pose", False):
        return (
            f"Use the initial position of `{spec.moved_runtime_uid}` as the fixed "
            f"spatial anchor. Source object: `{spec.moved_source_uid}`."
        )
    return (
        f"Use `{spec.reference_runtime_uid}` as the spatial reference. Source "
        f"object: `{spec.reference_source_uid}`."
    )


def _relative_final_planning_rule(
    project_name: str,
    spec: RelativePlacementLike,
) -> str:
    if getattr(spec, "reference_is_initial_pose", False) or getattr(
        spec,
        "upright_in_place",
        False,
    ):
        return (
            "Use the exact absolute target_pose JSON specs shown above. Do not "
            "rewrite this placement as a table-centered object-referenced pose; "
            "its XY anchor is the moved object's initial position."
        )
    return (
        f"Always plan to the current object poses from the exported {project_name} "
        "environment config. Do not hard-code absolute object coordinates in the "
        "generated graph."
    )


def _dual_relative_final_planning_rule(
    project_name: str,
    spec: RelativeSpecLike,
) -> str:
    if any(
        getattr(placement, "reference_is_initial_pose", False)
        for placement in spec.placements
    ):
        return (
            "Use the exact absolute target_pose JSON specs shown above for any "
            "initial-position placement. Do not rewrite those self-relative "
            "steps as object-referenced poses."
        )
    return (
        f"Always plan to the current object poses from the exported {project_name} "
        "environment config. Do not hard-code absolute object coordinates in the "
        "generated graph."
    )


def _relative_relation_phrase(relation: str) -> str:
    # Keep this private wrapper for compatibility with existing imports while
    # delegating the vocabulary to the shared generation/runtime contract.
    return _canonical_relative_relation_phrase(relation)


def _display_noun(uid: str) -> str:
    return uid.replace("_", " ")


def _plural(noun: str) -> str:
    if noun.endswith("s"):
        return noun
    if noun.endswith(("ch", "sh", "x")):
        return f"{noun}es"
    return f"{noun}s"
