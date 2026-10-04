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

"""Task Engine adapter that emits reusable normative TaskSpecs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from embodichain.gen_sim.task_spec import (
    KNOWN_PREDICATES,
    TaskRequirement,
    Predicate,
    RoleSpec,
    MilestoneSpec,
    TaskSpec,
    TaskSpecCache,
    validate_task_spec,
)

from .contracts import TaskCandidate, validate_task_candidate

__all__ = [
    "TaskSpecGenerator",
    "generate_task_spec",
    "task_spec_from_candidate",
    "task_template_from_candidate",
]


def task_spec_from_candidate(
    candidate: Mapping[str, Any] | TaskCandidate,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> TaskSpec:
    """Convert one validated TaskCandidate into a normative TaskSpec.

    The adapter consumes scene references and success semantics, while leaving
    concrete calls, robot poses, trajectories, and planner choices in the
    candidate/bundle layer.  Candidate IDs and the legacy step hash are kept
    only in metadata and therefore do not define the reusable semantic hash.
    """
    selected = validate_task_candidate(candidate)
    draft = selected["draft"]
    references = selected["scene_request"]["references"]
    steps = draft["steps"]
    role_specs, reference_roles = _roles(references)
    step_roles = _step_roles(steps, references, reference_roles)
    steps_by_id = {str(step["id"]): step for step in steps}

    init: list[Predicate] = []
    for role in role_specs:
        for key, expected in sorted(role.initial_state.items()):
            init.append(
                Predicate(
                    name="state",
                    arguments={"subject": role.name, "key": key},
                    expected=expected,
                )
            )

    step_goals: dict[str, list[Predicate]] = defaultdict(list)
    for term in selected["success_spec"]["terms"]:
        step_id = str(term["step_id"])
        name = str(term["type"])
        if name not in KNOWN_PREDICATES:
            raise ValueError(f"SuccessSpec predicate {name!r} is not in TaskSpec v0.1.")
        roles = step_roles.get(step_id, {})
        arguments: dict[str, Any] = {}
        if roles.get("object") is not None:
            arguments["subject"] = roles["object"]
        if roles.get("target") is not None:
            arguments["reference"] = roles["target"]
        arguments.update(_semantic_step_arguments(steps_by_id[step_id]))
        # When grounding has no target role, retain the semantic step type but
        # do not invent a physical pose or coordinate.
        step_goals[step_id].append(
            Predicate(
                name=name,
                arguments=arguments,
                expected=True,
            )
        )

    # Candidate success terms describe every intermediate achievement.  The
    # normative ``goal`` is the final authored step; earlier achievements stay
    # available as ordered milestones instead of being treated as simultaneous
    # terminal state.
    terminal_steps = [str(steps[-1]["id"])] if steps else []
    goals = [
        predicate
        for step_id in terminal_steps
        for predicate in step_goals.get(step_id, ())
    ]
    if not goals:
        goals = [predicate for values in step_goals.values() for predicate in values]
    requirements = _requirements(role_specs, steps, goals)
    milestones = tuple(
        MilestoneSpec(
            milestone_id=str(step["id"]),
            achieve=tuple(step_goals.get(str(step["id"]), ())),
            after=tuple(str(value) for value in step.get("depends_on", ())),
        )
        for step in steps
        if step_goals.get(str(step["id"]))
    )
    source_metadata: dict[str, Any] = {
        "generator": "embodichain.gen_sim.task_engine",
        "candidate_id": selected["candidate_id"],
        "legacy_semantic_hash": selected["semantic_hash"],
        "task_candidate_schema": draft["schema_version"],
    }
    if metadata is not None:
        source_metadata["caller"] = dict(metadata)
    template = TaskSpec(
        task_id=draft["task_id"],
        instruction=draft["instruction"],
        roles=role_specs,
        init=tuple(init),
        goal=tuple(goals),
        invariants=(),
        temporal=(),
        milestones=milestones,
        requirements=requirements,
        metadata=source_metadata,
    )
    validate_task_spec(template)
    return template


def task_template_from_candidate(
    candidate: Mapping[str, Any] | TaskCandidate,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> TaskSpec:
    """Compatibility alias for :func:`task_spec_from_candidate`."""
    return task_spec_from_candidate(candidate, metadata=metadata)


def generate_task_spec(
    candidate_or_set: Mapping[str, Any] | TaskCandidate,
    *,
    candidate_id: str | None = None,
    cache: TaskSpecCache | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> TaskSpec:
    """Generate or reuse a TaskSpec from a candidate or candidate set.

    Args:
        candidate_or_set: A validated ``TaskCandidate`` or a candidate set
            returned by :class:`TaskAgent`.
        candidate_id: Optional candidate ID when a set contains several votes.
        cache: Optional content-addressed cache.  A matching entry is loaded
            after generation and newly generated templates are stored.
        metadata: Non-semantic generation metadata retained on the template.

    Returns:
        A validated, reusable :class:`~embodichain.gen_sim.task_spec.TaskSpec`.
    """
    candidate = _select_candidate(candidate_or_set, candidate_id=candidate_id)
    template = task_spec_from_candidate(candidate, metadata=metadata)
    if cache is None:
        return template
    cached = cache.get(template.semantic_hash)
    if cached is not None:
        return cached
    cache.put(template)
    return template


@dataclass(frozen=True, slots=True)
class TaskSpecGenerator:
    """Small Task Engine façade for generation and optional caching."""

    cache: TaskSpecCache | None = None

    def generate(
        self,
        candidate_or_set: Mapping[str, Any] | TaskCandidate,
        *,
        candidate_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> TaskSpec:
        """Generate one template using this generator's cache."""
        return generate_task_spec(
            candidate_or_set,
            candidate_id=candidate_id,
            cache=self.cache,
            metadata=metadata,
        )


def _select_candidate(
    value: Mapping[str, Any] | TaskCandidate,
    *,
    candidate_id: str | None,
) -> dict[str, Any]:
    if "candidates" not in value:
        return validate_task_candidate(value)
    raw_candidates = value.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("TaskCandidateSet.candidates must be a non-empty list.")
    candidates = [validate_task_candidate(item) for item in raw_candidates]
    if candidate_id is not None:
        for candidate in candidates:
            if candidate["candidate_id"] == candidate_id:
                return candidate
        raise KeyError(f"Unknown TaskCandidate ID {candidate_id!r}.")
    return max(
        candidates,
        key=lambda item: (int(item["vote_count"]), str(item["candidate_id"])),
    )


def _roles(
    references: list[Mapping[str, Any]],
) -> tuple[tuple[RoleSpec, ...], dict[tuple[str, str, str], str]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for reference in references:
        key = (
            str(reference["role"]),
            str(reference["reference"]),
            str(reference["source_structure"]),
        )
        grouped[key].append(reference)
    roles: list[RoleSpec] = []
    role_by_key: dict[tuple[str, str, str], str] = {}
    used: set[str] = set()
    for index, key in enumerate(sorted(grouped), start=1):
        role, reference, source_structure = key
        # Role IDs are canonical protocol references.  The grounded natural
        # language reference remains an implementation detail of SceneAdapter
        # and must not leak into the TaskSpec semantic identity.
        name = f"role_{index:03d}"
        used.add(name)
        items = grouped[key]
        affordances = tuple(
            sorted({item for raw in items for item in raw["affordances"]})
        )
        initial_state: dict[str, Any] = {}
        attributes: dict[str, Any] = {}
        for item in items:
            initial_state.update(item["initial_state"])
            attributes.update(item["attributes"])
        kind = _role_kind(role, source_structure)
        quantifier = str(items[0]["quantifier"])
        count = int(items[0]["count"])
        roles.append(
            RoleSpec(
                name=name,
                kind=kind,
                required=True,
                quantifier=quantifier,
                count=count,
                source_structure=source_structure,
                affordances=affordances,
                initial_state=initial_state,
                attributes=attributes,
            )
        )
        role_by_key[key] = name
    return tuple(roles), role_by_key


def _step_roles(
    steps: list[Mapping[str, Any]],
    references: list[Mapping[str, Any]],
    role_by_key: Mapping[tuple[str, str, str], str],
) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for step in steps:
        step_id = str(step["id"])
        result[step_id] = {}
        for role in ("object", "target"):
            selector = step[role]
            if selector["kind"] != "scene_ref":
                continue
            reference = next(
                item
                for item in references
                if item["step_id"] == step_id and item["role"] == role
            )
            key = (
                str(reference["role"]),
                str(reference["reference"]),
                str(reference["source_structure"]),
            )
            result[step_id][role] = role_by_key[key]
    return result


def _role_kind(role: str, source_structure: str) -> str:
    if role == "target":
        if source_structure in {"container", "rigid_object"}:
            return "target"
        return "surface"
    if source_structure == "articulation":
        return "articulation"
    if source_structure == "container":
        return "container"
    if source_structure == "button":
        return "button"
    return "object"


def _semantic_step_arguments(step: Mapping[str, Any]) -> dict[str, Any]:
    """Retain normative task parameters while excluding action sequencing."""
    result: dict[str, Any] = {"task_type": str(step["task_type"])}
    defaults: dict[str, Any] = {
        "relation": "none",
        "orientation_goal": "none",
        "target_state": "none",
        "target_setting": 0,
        "layout": "none",
        "axis": "none",
        "direction": "none",
        "terminal_behavior": "none",
    }
    for name, default in defaults.items():
        value = step.get(name, default)
        if value != default:
            result[name] = value
    required_arm = str(step.get("required_arm", "none"))
    if required_arm not in {"none", "auto"}:
        # An explicitly requested arm is a task requirement; automatic arm
        # selection remains a witness/planner choice.
        result["required_arm"] = required_arm
    return result


def _requirements(
    roles: tuple[RoleSpec, ...],
    steps: list[Mapping[str, Any]],
    goals: list[Predicate],
) -> tuple[TaskRequirement, ...]:
    requirements: list[TaskRequirement] = []
    for role in roles:
        requirements.append(
            TaskRequirement(
                consumer="asset",
                key=role.name,
                value={
                    "source_structure": role.source_structure,
                    "affordances": list(role.affordances),
                    "attributes": dict(role.attributes),
                },
            )
        )
        requirements.append(
            TaskRequirement(
                consumer="scene",
                key=role.name,
                value={"kind": role.kind, "initial_state": dict(role.initial_state)},
            )
        )
    required_arms = sorted(
        {
            str(step["required_arm"])
            for step in steps
            if str(step.get("required_arm", "none")) not in {"none", "auto"}
        }
    )
    if required_arms:
        requirements.append(
            TaskRequirement(
                consumer="embodiment", key="required_arms", value=required_arms
            )
        )
    requirements.append(
        TaskRequirement(
            consumer="task_program",
            key="goal_predicates",
            value=sorted({item.name for item in goals}),
        )
    )
    requirements.append(
        TaskRequirement(
            consumer="expansion",
            key="lineage_key",
            value="task_spec.semantic_hash",
        )
    )
    return tuple(requirements)
