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

from .contracts import TaskCandidate, canonical_hash, validate_task_candidate

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
        orientation = (
            arguments.pop("orientation_goal", None)
            if name in {"semantic_goal", "handover_complete", "held_by_both_grippers"}
            else None
        )
        step_goals[step_id].append(
            Predicate(
                name=name,
                arguments=arguments,
                expected=True,
            )
        )
        if orientation is not None:
            step_goals[step_id].append(
                Predicate(
                    "object_upright" if orientation == "upright" else "orientation",
                    {
                        "subject": roles["object"],
                        "task_type": str(steps_by_id[step_id]["task_type"]),
                        "orientation_goal": orientation,
                    },
                )
            )
        if steps_by_id[step_id]["task_type"] == "E4" and name != "handover_complete":
            step_goals[step_id].append(Predicate("handover_complete", arguments))
        if (
            steps_by_id[step_id]["task_type"] == "E5"
            and name != "held_by_both_grippers"
        ):
            step_goals[step_id].append(Predicate("held_by_both_grippers", arguments))

    goals, goal_sources = _terminal_goals(steps, step_roles, step_goals)
    requirements = _requirements(role_specs, steps, goals)
    milestones = []
    milestone_sources = {}
    for step in steps:
        step_id = str(step["id"])
        achievements = list(step_goals.get(step_id, ()))
        after = tuple(str(value) for value in step.get("depends_on", ()))
        # Acquisition and release are successive states, never one conjunction.
        if step["task_type"] in {"E4", "E5"} and step["terminal_behavior"] == "place":
            acquisition = (
                "handover_complete"
                if step["task_type"] == "E4"
                else "held_by_both_grippers"
            )
            acquire_id = f"{step_id}.acquired"
            milestones.append(
                MilestoneSpec(
                    milestone_id=acquire_id,
                    achieve=tuple(
                        p.with_id(f"milestone:{acquire_id}:{i}")
                        for i, p in enumerate(achievements)
                        if p.name == acquisition
                    ),
                    after=after,
                )
            )
            milestone_sources[acquire_id] = step_id
            achievements = [p for p in achievements if p.name != acquisition]
            after = (acquire_id,)
        milestones.append(
            MilestoneSpec(
                milestone_id=step_id,
                achieve=tuple(
                    p.with_id(f"milestone:{step_id}:{i}")
                    for i, p in enumerate(achievements)
                ),
                after=after,
            )
        )
        milestone_sources[step_id] = step_id
    source_metadata: dict[str, Any] = {
        "generator": "embodichain.gen_sim.task_engine",
        "candidate_id": selected["candidate_id"],
        "legacy_semantic_hash": selected["semantic_hash"],
        "task_candidate_schema": draft["schema_version"],
        "terminal_goal_policy": "explicit_state_slots/v1",
        "instruction_coverage": "validated_draft_only",
        "step_roles": step_roles,
        "role_references": {
            role.name: [
                str(ref["reference_id"])
                for ref in references
                if reference_roles[_reference_key(ref)] == role.name
            ]
            for role in role_specs
        },
        "goal_sources": goal_sources,
        "milestone_sources": milestone_sources,
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
        milestones=tuple(milestones),
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
        # Cache normative identity, never another run's role/evidence metadata.
        return template
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
) -> tuple[tuple[RoleSpec, ...], dict[tuple[str, str, str, str, int], str]]:
    grouped: dict[tuple[str, str, str, str, int], list[Mapping[str, Any]]] = (
        defaultdict(list)
    )
    for reference in references:
        key = _reference_key(reference)
        grouped[key].append(reference)
    roles: list[RoleSpec] = []
    role_by_key: dict[tuple[str, str, str, str, int], str] = {}
    used: set[str] = set()
    for index, key in enumerate(sorted(grouped), start=1):
        role, reference, source_structure, quantifier, count = key
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
            for target, incoming in (
                (initial_state, item["initial_state"]),
                (attributes, item["attributes"]),
            ):
                for field, value in incoming.items():
                    if field in target and target[field] != value:
                        raise ValueError(
                            f"Conflicting TaskSpec role declaration: {reference!r}.{field}."
                        )
                    target[field] = value
        kind = _role_kind(role, source_structure)
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
    role_by_key: Mapping[tuple[str, str, str, str, int], str],
) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for step in steps:
        step_id = str(step["id"])
        result[step_id] = {}
        for role in ("object", "target"):
            selector = step[role]
            if selector["kind"] == "step_result":
                source = next(
                    item for item in steps if item["id"] == selector["step_id"]
                )
                if (
                    source["object"]["quantifier"] == "all"
                    or source["object"]["count"] > 1
                ):
                    raise ValueError(
                        "A step_result cannot silently select from multiple subjects."
                    )
                result[step_id][role] = result[str(selector["step_id"])]["object"]
                continue
            if selector["kind"] != "scene_ref":
                continue
            reference = next(
                item
                for item in references
                if item["step_id"] == step_id and item["role"] == role
            )
            key = _reference_key(reference)
            result[step_id][role] = role_by_key[key]
    return result


def _reference_key(reference: Mapping[str, Any]) -> tuple[str, str, str, str, int]:
    return (
        str(reference["role"]),
        str(reference["reference"]),
        str(reference["source_structure"]),
        str(reference["quantifier"]),
        int(reference["count"]),
    )


def _terminal_goals(
    steps: list[Mapping[str, Any]],
    roles: Mapping[str, Mapping[str, str]],
    achievements: Mapping[str, list[Predicate]],
) -> tuple[list[Predicate], dict[str, list[str]]]:
    # State slots can be superseded; historical achievements remain milestones.
    slots: dict[tuple[str, str], tuple[Predicate, str]] = {}
    events: dict[str, tuple[Predicate, str]] = {}
    for step in steps:
        step_id, kind = str(step["id"]), str(step["task_type"])
        subject = roles[step_id]["object"]
        predicates = achievements[step_id]
        if kind in {"E1", "E3", "E4", "E5"}:
            slots.pop((subject, "placement"), None)
            slots.pop((subject, "attachment"), None)
            if step["orientation_goal"] != "preserve":
                slots.pop((subject, "orientation"), None)
                slots.pop((subject, "orientation_preserve"), None)
        if kind == "E2":
            slots.pop((subject, "attachment"), None)
            slots.pop((subject, "orientation_preserve"), None)
            previous = slots.get((subject, "placement"))
            if previous is not None and previous[0].arguments.get("relation") not in {
                "on",
                "above",
            }:
                slots.pop((subject, "placement"), None)
        for predicate in predicates:
            if predicate.name in {"poured", "pour_motion", "pressed"} or (
                predicate.name == "semantic_goal"
                and predicate.arguments.get("task_type") == "E8"
            ):
                events[canonical_hash(predicate.to_dict())] = (predicate, step_id)
            elif predicate.name in {"object_upright", "orientation"}:
                slot = (
                    "orientation_preserve"
                    if predicate.name == "orientation"
                    and (subject, "orientation") in slots
                    else "orientation"
                )
                slots[(subject, slot)] = (predicate, step_id)
            elif predicate.name == "articulation_joint_near":
                slots[(subject, "articulation")] = (predicate, step_id)
            elif predicate.name == "semantic_goal":
                slots[(subject, "placement")] = (predicate, step_id)
            elif (
                predicate.name == "handover_complete"
                and step["terminal_behavior"] == "hold"
            ):
                arguments = dict(predicate.arguments)
                arguments.pop("orientation_goal", None)
                slots[(subject, "attachment")] = (Predicate("held", arguments), step_id)
            elif (
                predicate.name == "held_by_both_grippers"
                and step["terminal_behavior"] == "hold"
            ):
                slots[(subject, "attachment")] = (predicate, step_id)
    goals, sources = [], {}
    for index, (predicate, step_id) in enumerate(
        (*slots.values(), *events.values()), 1
    ):
        predicate = predicate.with_id(f"goal_{index:03d}")
        goals.append(predicate)
        sources[predicate.id] = [step_id]
    return goals, sources


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
    if step.get("twist_goal_mode") == "coarse":
        result["twist_goal_mode"] = "coarse"
    for field in ("required_arm", "transfer_arm", "receive_arm"):
        required_arm = str(step.get(field, "none"))
        if required_arm in {"none", "auto"}:
            continue
        # An explicitly requested arm is a task requirement; automatic arm
        # selection remains a witness/planner choice.
        result[field] = required_arm
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
            str(step[field])
            for step in steps
            for field in ("required_arm", "transfer_arm", "receive_arm")
            if str(step.get(field, "none")) not in {"none", "auto"}
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
