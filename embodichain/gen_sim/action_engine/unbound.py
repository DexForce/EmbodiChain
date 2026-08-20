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

"""Scene-independent Action Engine draft produced before final UID binding."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
from typing import Any, Final, TypeAlias

from embodichain.gen_sim.action_engine.domain.task_contracts import TASK_CONTRACTS

__all__ = [
    "UNBOUND_ACTION_PLAN_SCHEMA",
    "UnboundActionPlan",
    "build_unbound_action_plan",
    "validate_unbound_action_plan",
]

UNBOUND_ACTION_PLAN_SCHEMA: Final = "embodichain.unbound-action-plan/v1"
UnboundActionPlan: TypeAlias = dict[str, Any]

_PLAN_KEYS = frozenset(
    {
        "schema_version",
        "task_id",
        "candidate_id",
        "instruction",
        "steps",
        "required_actions",
    }
)
_STEP_KEYS = frozenset(
    {"step_id", "task_type", "object", "target", "depends_on", "actions"}
)


def build_unbound_action_plan(candidate: Mapping[str, Any]) -> UnboundActionPlan:
    """Lower a TaskCandidate into an Action-owned plan without scene UIDs.

    Args:
        candidate: Validated Task Engine candidate or an equivalent mapping.

    Returns:
        A strict JSON plan whose selectors remain logical references.

    Raises:
        TypeError: If the candidate or draft is not a mapping.
        ValueError: If the draft references an unsupported task type.
    """
    value = _mapping(candidate, "candidate")
    draft = _mapping(value.get("draft"), "candidate.draft")
    task_id = _nonempty(draft.get("task_id"), "candidate.draft.task_id")
    instruction = _nonempty(draft.get("instruction"), "candidate.draft.instruction")
    candidate_id = _nonempty(value.get("candidate_id"), "candidate.candidate_id")
    steps = []
    required_actions: set[str] = set()
    for index, raw in enumerate(_sequence(draft.get("steps"), "candidate.draft.steps")):
        step = _mapping(raw, f"candidate.draft.steps[{index}]")
        task_type = _nonempty(
            step.get("task_type"), f"candidate.draft.steps[{index}].task_type"
        )
        contract = TASK_CONTRACTS.get(task_type)
        if contract is None:
            raise ValueError(f"Action Engine does not support task type {task_type!r}.")
        actions = [str(name) for name in contract.core_actions]
        required_actions.update(actions)
        steps.append(
            {
                "step_id": _nonempty(
                    step.get("id"), f"candidate.draft.steps[{index}].id"
                ),
                "task_type": task_type,
                "object": deepcopy(step.get("object")),
                "target": deepcopy(step.get("target")),
                "depends_on": deepcopy(step.get("depends_on", [])),
                "actions": actions,
            }
        )
    return validate_unbound_action_plan(
        {
            "schema_version": UNBOUND_ACTION_PLAN_SCHEMA,
            "task_id": task_id,
            "candidate_id": candidate_id,
            "instruction": instruction,
            "steps": steps,
            "required_actions": sorted(required_actions),
        }
    )


def validate_unbound_action_plan(
    value: Mapping[str, Any],
) -> UnboundActionPlan:
    """Validate and detach one scene-independent Action plan.

    Args:
        value: Candidate plan mapping.

    Returns:
        A strict JSON-safe detached plan.

    Raises:
        TypeError: If a mapping or sequence field has the wrong type.
        ValueError: If the schema, dependency graph, or actions are invalid.
    """
    result = _mapping(value, "UnboundActionPlan")
    if set(result) != _PLAN_KEYS:
        raise ValueError("UnboundActionPlan fields are invalid.")
    if result.get("schema_version") != UNBOUND_ACTION_PLAN_SCHEMA:
        raise ValueError("UnboundActionPlan.schema_version is invalid.")
    for key in ("task_id", "candidate_id", "instruction"):
        result[key] = _nonempty(result.get(key), f"UnboundActionPlan.{key}")

    steps = []
    seen: set[str] = set()
    actions_used: set[str] = set()
    for index, raw in enumerate(_sequence(result.get("steps"), "steps")):
        context = f"UnboundActionPlan.steps[{index}]"
        step = _mapping(raw, context)
        if set(step) != _STEP_KEYS:
            raise ValueError(f"{context} fields are invalid.")
        step_id = _nonempty(step.get("step_id"), f"{context}.step_id")
        if step_id in seen:
            raise ValueError("UnboundActionPlan step IDs must be unique.")
        task_type = _nonempty(step.get("task_type"), f"{context}.task_type")
        contract = TASK_CONTRACTS.get(task_type)
        if contract is None:
            raise ValueError(f"{context}.task_type is unsupported.")
        dependencies = _strings(step.get("depends_on"), f"{context}.depends_on")
        if any(dependency not in seen for dependency in dependencies):
            raise ValueError(
                f"{context}.depends_on must reference preceding unbound steps."
            )
        actions = _strings(step.get("actions"), f"{context}.actions")
        if actions != [str(name) for name in contract.core_actions]:
            raise ValueError(f"{context}.actions do not match the task contract.")
        for selector_name in ("object", "target"):
            if not isinstance(step.get(selector_name), Mapping):
                raise TypeError(f"{context}.{selector_name} must be a mapping.")
            step[selector_name] = deepcopy(dict(step[selector_name]))
        step["step_id"] = step_id
        step["task_type"] = task_type
        step["depends_on"] = dependencies
        step["actions"] = actions
        steps.append(step)
        seen.add(step_id)
        actions_used.update(actions)
    if not steps:
        raise ValueError("UnboundActionPlan.steps must not be empty.")
    required = _strings(result.get("required_actions"), "required_actions")
    if required != sorted(actions_used):
        raise ValueError("UnboundActionPlan.required_actions is not canonical.")
    result["steps"] = steps
    result["required_actions"] = required
    json.dumps(result, ensure_ascii=False, allow_nan=False)
    return result


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping.")
    return deepcopy(dict(value))


def _sequence(value: Any, context: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{context} must be a sequence.")
    return list(value)


def _strings(value: Any, context: str) -> list[str]:
    result = _sequence(value, context)
    if any(not isinstance(item, str) or not item for item in result):
        raise ValueError(f"{context} must contain non-empty strings.")
    if len(set(result)) != len(result):
        raise ValueError(f"{context} must not contain duplicates.")
    return list(result)


def _nonempty(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string.")
    return value.strip()
