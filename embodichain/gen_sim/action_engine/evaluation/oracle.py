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

"""Path-independent private-oracle evaluation for generated L4 tasks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from embodichain.gen_sim.action_engine.domain import validate_task_spec
from embodichain.gen_sim.action_engine.runtime.predicates import evaluate_predicate

__all__ = ["evaluate_task_oracle"]


def evaluate_task_oracle(
    task_spec: Mapping[str, Any],
    env: Any,
    role_bindings: Mapping[str, str],
    *,
    visual_facts: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
) -> torch.Tensor:
    """Evaluate an L4 goal from final state, without inspecting the action path."""
    task = validate_task_spec(task_spec)
    if task["level"] != "L4":
        raise ValueError("Private oracle evaluation is defined only for L4 tasks.")
    bindings = _bindings(task, role_bindings)
    custom = getattr(env, "evaluate_action_engine_oracle", None)
    if callable(custom):
        return _mask(
            custom(task=task, role_bindings=bindings, visual_facts=visual_facts),
            env,
        )

    success_type = str(task["success"].get("type", ""))
    if success_type == "original_order_restored":
        order = task["oracle"].get("order_bottom_to_top")
        if not isinstance(order, Sequence) or isinstance(order, (str, bytes)):
            raise ValueError("Memory oracle requires order_bottom_to_top.")
        result = _constant(env, True)
        for support_role, object_role in zip(order, order[1:]):
            result &= evaluate_predicate(
                env,
                {
                    "type": "object_on_object",
                    "object": _uid(bindings, object_role),
                    "support": _uid(bindings, support_role),
                },
            )
        return result
    if success_type == "sum_equals":
        return _sum_selection(task, env, bindings)
    if success_type == "functional_place_setting":
        return _functional_layout(task, env, bindings)
    if success_type == "stable_unobstructed":
        stable = _constant(env, True)
        for instance in task["task_instances"]:
            role = instance["params"].get("object_role")
            if isinstance(role, str):
                stable &= evaluate_predicate(
                    env,
                    {"type": "object_not_fallen", "object": _uid(bindings, role)},
                )
        return stable & _visual_result(
            visual_facts,
            env,
            relation=None,
            required_visible_uid=_uid(
                bindings, str(task["success"].get("reference_role", ""))
            ),
        )
    if success_type == "visual_relation":
        return _visual_result(
            visual_facts,
            env,
            relation=str(task["success"].get("relation", "")),
            required_visible_uid=None,
        )
    raise ValueError(f"Unsupported L4 oracle success type {success_type!r}.")


def _sum_selection(
    task: Mapping[str, Any], env: Any, bindings: Mapping[str, str]
) -> torch.Tensor:
    selections = task["oracle"].get("valid_selections")
    if not isinstance(selections, Sequence) or isinstance(selections, (str, bytes)):
        raise ValueError("Logic oracle requires valid_selections.")
    candidate_roles = sorted(
        {
            str(role)
            for selection in selections
            if isinstance(selection, Sequence)
            and not isinstance(selection, (str, bytes))
            for role in selection
        }
    )
    targets = {
        str(instance["params"].get("target_role"))
        for instance in task["task_instances"]
        if instance["params"].get("target_role") is not None
    }
    if len(targets) != 1:
        raise ValueError("Logic oracle requires one selection target role.")
    target_uid = _uid(bindings, targets.pop())
    selected = {
        role: evaluate_predicate(
            env,
            {
                "type": "object_in_container",
                "object": _uid(bindings, role),
                "container": target_uid,
            },
        )
        for role in candidate_roles
    }
    result = _constant(env, False)
    for selection in selections:
        expected = {str(role) for role in selection}
        match = _constant(env, True)
        for role, value in selected.items():
            match &= value if role in expected else ~value
        result |= match
    return result


def _functional_layout(
    task: Mapping[str, Any], env: Any, bindings: Mapping[str, str]
) -> torch.Tensor:
    required = task["oracle"].get("required_roles")
    if not isinstance(required, Sequence) or isinstance(required, (str, bytes)):
        raise ValueError("Common-sense oracle requires required_roles.")
    targets = {
        str(instance["params"].get("target_role"))
        for instance in task["task_instances"]
        if instance["params"].get("target_role") is not None
    }
    if len(targets) != 1:
        raise ValueError("Common-sense oracle requires one layout target role.")
    target_uid = _uid(bindings, targets.pop())
    result = _constant(env, True)
    for role in required:
        result &= evaluate_predicate(
            env,
            {
                "type": "object_in_container",
                "object": _uid(bindings, role),
                "container": target_uid,
            },
        )
    return result


def _visual_result(
    facts: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
    env: Any,
    *,
    relation: str | None,
    required_visible_uid: str | None,
) -> torch.Tensor:
    rows = _fact_rows(facts, int(env.num_envs))
    values = []
    for row in rows:
        entities = row.get("entities", ())
        relations = row.get("relations", ())
        visible = True
        if required_visible_uid is not None:
            visible = any(
                isinstance(entity, Mapping)
                and entity.get("uid") == required_visible_uid
                and entity.get("visible", True) is True
                for entity in entities
            )
            visible &= not any(
                isinstance(item, Mapping)
                and str(item.get("type", "")).lower() in {"occludes", "obstructs"}
                and required_visible_uid in item.get("uids", ())
                for item in relations
            )
        relation_met = relation is None or any(
            isinstance(item, Mapping)
            and item.get("type") == relation
            and float(item.get("confidence", 0.0)) >= 0.5
            for item in relations
        )
        values.append(bool(visible and relation_met))
    return torch.tensor(values, dtype=torch.bool, device=env.device)


def _fact_rows(
    facts: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
    num_envs: int,
) -> list[Mapping[str, Any]]:
    if facts is None:
        raise ValueError("This L4 oracle requires post-execution visual facts.")
    if isinstance(facts, Mapping):
        return [facts] * num_envs
    if not isinstance(facts, Sequence) or isinstance(facts, (str, bytes)):
        raise ValueError("visual_facts must be a mapping or one mapping per env.")
    rows = list(facts)
    if len(rows) != num_envs or any(not isinstance(row, Mapping) for row in rows):
        raise ValueError("visual_facts must contain exactly one mapping per env.")
    return rows


def _bindings(
    task: Mapping[str, Any], role_bindings: Mapping[str, str]
) -> dict[str, str]:
    bindings = {str(role): str(uid) for role, uid in role_bindings.items()}
    referenced = {
        str(value)
        for instance in task["task_instances"]
        for key, value in instance["params"].items()
        if key.endswith("_role") and isinstance(value, str) and value != "table"
    }
    missing = sorted(referenced - set(bindings))
    if missing:
        raise ValueError(f"L4 oracle role bindings are missing {missing}.")
    if len(bindings.values()) != len(set(bindings.values())):
        raise ValueError("L4 oracle role bindings must resolve to unique UIDs.")
    return bindings


def _uid(bindings: Mapping[str, str], role: Any) -> str:
    role = str(role)
    if role == "table":
        return role
    try:
        return bindings[role]
    except KeyError as exc:
        raise ValueError(f"L4 oracle references unbound role {role!r}.") from exc


def _constant(env: Any, value: bool) -> torch.Tensor:
    return torch.full((int(env.num_envs),), value, dtype=torch.bool, device=env.device)


def _mask(value: Any, env: Any) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=torch.bool, device=env.device).reshape(-1)
    if result.numel() == 1:
        result = result.repeat(int(env.num_envs))
    if result.numel() != int(env.num_envs):
        raise ValueError("Oracle callback returned the wrong number of env rows.")
    return result
