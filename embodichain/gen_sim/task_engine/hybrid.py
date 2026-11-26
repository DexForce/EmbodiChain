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
"""Hybrid authoring with native witnesses and graph-only semantic intent."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator
import hashlib
from itertools import combinations
import json
from pathlib import Path
import re
from typing import Any

from .authoring_intent import (
    _AXIS_INVERSES,
    _GROUP_RELATIONS,
    _INDEPENDENT_STATE_MEMBERS,
    _PROMPT as _INTENT_FORMAT,
    AuthoringEvidenceError,
    _advance_arms,
    _ArmState,
    _expand_fact_shorthand,
    _intent_phases,
    _intent_step_arms,
    _is_control_transition,
    _relation_pairs,
    _role_ancestry,
    compile_intent,
    source_spans,
    validate_intent,
    verify_spatial_claim,
    verify_state_claim,
)
from .goal_planning import author_goal_plan, _deployment_check
from .ontology import task_contract
from .scene_context import (
    _SCENE_ROLE_ID,
    ArticulationDeclarationError,
    require_articulation,
    scene_objects_requirement,
)
from .semantic_graph import validate_semantic_task_graph
from .orchestration.workflow_artifacts import WorkflowRunArtifacts, write_run_json
from ..task_spec import TaskSpec, canonical_hash
from ..task_spec.publication import freeze_task_spec

__all__: list[str] = []

_OPERATION_FIELDS = {
    "E1": {"reference", "relation"},
    "E2": {"orientation"},
    "E3": {"reference", "mode"},
    "E4": {"source", "destination"},
    "E5": {"reference", "relation"},
    "E6": {"state"},
    "E7": {"state"},
    "E8": {"mode"},
    "E9": set(),
}
_TERMINAL_FAMILIES = {"E2", "E3", "E4"}


def _load_scene_context(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != "embodichain.task-scene-context/v1"
        or value.keys() - {"schema_version", "description", "objects"}
    ):
        raise ValueError("Expected embodichain.task-scene-context/v1 JSON")
    description = value.get("description", "")
    objects = value.get("objects", [])
    if not isinstance(objects, list) or not isinstance(description, str):
        raise ValueError("scene-context requires a description string or objects array")
    if not (description.strip() or objects):
        raise ValueError("scene-context must describe the scene or at least one object")
    seen = set()
    for entry in objects:
        if not isinstance(entry, dict) or entry.keys() - {
            "id",
            "description",
            "asset_kind",
        }:
            raise ValueError("Expected a scene-context object declaration")
        role = entry.get("id")
        if not isinstance(role, str) or _SCENE_ROLE_ID.fullmatch(role) is None:
            raise ValueError("Invalid scene-context object ID")
        if role in seen:
            raise ValueError(f"Duplicate scene-context object ID {role!r}")
        seen.add(role)
        text = entry.get("description")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"scene-context object {role!r} requires a description")
        if "asset_kind" in entry and entry["asset_kind"] not in (
            "rigid",
            "articulated",
        ):
            raise ValueError(f"Invalid scene-context asset_kind for {role!r}")
    return value


def _source_quote(value: Any, instruction: str) -> bool:
    return isinstance(value, str) and bool(value.strip()) and value in instruction


_TIMING_CUE = re.compile(
    r"\b(?:simultaneously|concurrently|synchronously|at (?:the )?same time|at once|in (?:parallel|unison))\b|同时|同步|并行",
    re.IGNORECASE,
)
_PICK_CUE = re.compile(
    r"\b(?:(?:pick|grasp)(?:s|ed|ing)?|grab(?:s|bed|bing)?|acquir(?:e[sd]?|ing))\b|抓|拿起",
    re.IGNORECASE,
)
_PICK_TIMING_BOUNDARY = re.compile(
    r"[.!?;。！？；]|\b(?:then|afterwards|before)\b"
    r"|(?:,\s*(?:(?:and|or|but|while)\s+)?|\b(?:and|or|but|while)\s+)"
    r"(?=(?:(?:you|they|we|it|both|(?:the|a|an|both)\s+(?:[\w-]+\s+){0,2}[\w-]+)\s+)?"
    rf"(?:(?:also|together|[a-z]+ly|will|should|must|can|are|is|be|{_TIMING_CUE.pattern}|"
    r"with\s+(?:(?:the|both|left|right|your)\s+)*arms?)[\s,]+)*"
    r"(?:(?:plac|releas|rais|mov|clos|rotat|actuat)(?:e[sd]?|ing)|"
    r"(?:deposit|lift|lower|transfer|pour|open|push|pull|turn|hold|press|return)(?:s|ed|ing)?|"
    r"(?:put|set)(?:s|ting)?|drop(?:s|ped|ping)?|carr(?:y|ies|ied|ying))\b)"
    r"|(?:，|然后|并|再)\s*(?=(?:同时|同步|并行)?\s*(?:放|置|松开|抬|移|转|倒))",
    re.IGNORECASE,
)
_NEGATED_CUE = re.compile(r"\b(?:not|never|avoid)\b|不要|不得|别", re.IGNORECASE)
_FUTURE_SIDES = re.compile(
    r"\b(?:one on (?:each|either) side|opposite sides) of (?:(?:the|a) )?"
    r"(?:future|eventual|later[- ]placed)\b",
    re.IGNORECASE,
)
_STABILITY_CUE = re.compile(
    r"\b(?:stabl\w*|steady|balanc\w*|toppl\w*|tipp?ing|falling)\b|稳定|稳固|平衡|倾倒",
    re.IGNORECASE,
)


def _explicit_pick_timing(quote: str, instruction: str) -> bool:
    if not _TIMING_CUE.search(quote):
        return False
    # A cue must occur in the quoted range of an acquisition clause itself.
    quotes = [match.span() for match in re.finditer(re.escape(quote), instruction)]
    boundaries = [match.span() for match in _PICK_TIMING_BOUNDARY.finditer(instruction)]
    for start, stop in zip(
        [0, *(end for _, end in boundaries)],
        [*(begin for begin, _ in boundaries), len(instruction)],
    ):
        clause = instruction[start:stop]
        if not _PICK_CUE.search(clause):
            continue
        for cue in _TIMING_CUE.finditer(clause):
            if not _NEGATED_CUE.search(clause[: cue.start()]) and any(
                left <= start + cue.start() and start + cue.end() <= right
                for left, right in quotes
            ):
                return True
    return False


def _unrequested_stability_issue(instruction: str, audit: Any) -> bool:
    issues = audit.get("issues") if isinstance(audit, dict) else None
    return (
        isinstance(issues, list)
        and bool(issues)
        and not _STABILITY_CUE.search(instruction)
        and all(
            isinstance(issue, dict)
            and _source_quote(issue.get("quote"), instruction)
            and isinstance(issue.get("reason"), str)
            and _STABILITY_CUE.search(issue["reason"])
            for issue in issues
        )
    )


def _operation_rows(response: Any, instruction: str) -> list[dict[str, Any]]:
    """Validate source-bearing operation declarations without matching vocabulary."""
    rows = (
        response["operations"]
        if isinstance(response, dict) and set(response) == {"operations"}
        else response
    )
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a nonempty operations array")
    result = []
    for index, op in enumerate(rows):
        label = f"operations[{index}]"
        if not isinstance(op, dict) or not all(
            isinstance(op.get(field), str) and op[field].strip()
            for field in ("template", "object", "quote")
        ):
            raise ValueError(f"{label} requires template, object and quote strings")
        if not _source_quote(op["quote"], instruction):
            raise ValueError(f"{label}: missing verbatim operation source")
        family = op["template"]
        fields = _OPERATION_FIELDS.get(family)
        if fields is None:
            raise ValueError(f"{label}: unsupported operation {family!r}")
        if any(not isinstance(op.get(k), str) or not op[k].strip() for k in fields):
            raise ValueError(f"{label} requires nonempty {sorted(fields)}")
        allowed = fields | {"template", "object", "quote"}
        row = dict(op)
        if family not in {"E4", "E5"}:
            allowed.add("primary")
            row.setdefault("primary", "auto")
        if family == "E8":
            allowed.add("setting")
        if "part_of" in op and task_contract(family).source_structure == "articulation":
            parent = op["part_of"]
            if (
                not isinstance(parent, str)
                or not parent.strip()
                or parent == op["object"]
            ):
                raise ValueError(f"{label}: part_of needs a distinct scene object ID")
            allowed.add("part_of")
        if family == "E3" and op["mode"] not in ("motion", "fluid"):
            raise ValueError(f"{label}: Unsupported or ambiguous pour mode")
        if family in _TERMINAL_FAMILIES:
            allowed.update(("terminal_behavior", "terminal_quote"))
            terminal = row.setdefault("terminal_behavior", "none")
            if terminal not in ("none", "hold", "return_and_release"):
                raise ValueError(f"{label}: Unsupported terminal behavior")
            if terminal != "none" and not _source_quote(
                op.get("terminal_quote"), instruction
            ):
                raise ValueError(f"{label}: Missing verbatim terminal behavior source")
            if terminal == "none" and "terminal_quote" in op:
                raise ValueError(
                    f"{label}: terminal_quote requires a terminal behavior"
                )
        if extra := op.keys() - allowed:
            raise ValueError(f"{label}: unsupported operation fields {sorted(extra)}")
        result.append(row)
    return result


def _validate_audit_issues(instruction: str, audit: Any, evidence_key: str) -> None:
    if (
        not isinstance(audit, dict)
        or audit.keys() - {"issues", evidence_key}
        or not isinstance(audit.get("issues"), list)
    ):
        raise ValueError("Malformed source audit")
    issues = audit["issues"]
    if any(
        not isinstance(issue, dict)
        or not _source_quote(issue.get("quote"), instruction)
        or not isinstance(issue.get("reason"), str)
        or not issue["reason"].strip()
        for issue in issues
    ):
        raise ValueError("Malformed source audit diagnostic")
    if issues:
        raise ValueError("source_audit: " + json.dumps(issues, ensure_ascii=False))


def _validate_source_audit(
    instruction: str, operations: list[dict[str, Any]], audit: dict[str, Any]
) -> None:
    """Require independently source-anchored agreement on each ordered operation."""
    _validate_audit_issues(instruction, audit, "source_operations")
    if "source_operations" not in audit:
        raise ValueError(
            "source_audit: missing independently reconstructed source_operations"
        )
    reviewed = _operation_rows(audit["source_operations"], instruction)
    if len(reviewed) != len(operations):
        raise ValueError("source_audit: source operation count differs")

    def signature(row):
        return canonical_hash(
            {k: v for k, v in row.items() if k not in {"quote", "terminal_quote"}}
        )

    for index, (proposed, expected) in enumerate(zip(operations, reviewed), start=1):
        if signature(proposed) != signature(expected):
            raise ValueError(
                f"source_audit: operation {index} differs from the source: "
                + json.dumps(expected, ensure_ascii=False)
            )


_AUDIT_SCOPES = ("action", "initial", "always", "before", "step", "effect", "final")
_INTENT_AUDIT_SCHEMA = {
    "type": "object",
    "required": ["issues"],
    "properties": {
        "issues": {"type": "array"},
        "checks": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["quote", "requirement", "scope", "evidence"],
                "properties": {
                    "quote": {"type": "string"},
                    "requirement": {"type": "string"},
                    "scope": {"enum": list(_AUDIT_SCOPES)},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                    "state": {"type": "object"},
                    "spatial": {"type": "object"},
                    "parallel_picks": {"type": "object"},
                },
            },
        },
    },
}

_POSITION_FACTS = {
    "relation",
    "pose_match",
    "view_location",
    "vertical_rise",
    "distance_compare",
    "distance_change",
    "ordered_row",
    "hung_by",
    "spaced_apart",
    "mutually_separate",
}
_OPPOSITE_VALUES = {
    "relation": {frozenset({"inside", "not_inside"})},
    "orientation": (
        {
            frozenset(pair.split())
            for pair in (
                "opening_up opening_down",
                "left_to_right front_to_back",
                "pointing_toward pointing_away_from",
            )
        }
        # Main-axis alignment does not replace a separate cross-section or opening fact.
        | {
            frozenset((vertical, other))
            for vertical in ("upright", "vertical")
            for other in (
                "tilted",
                "flat",
                "lying_on_side",
                "left_to_right",
                "front_to_back",
            )
        }
        | {
            frozenset(pair)
            for pair in combinations(
                (
                    "pointing_left pointing_right pointing_up pointing_down "
                    "pointing_forward pointing_backward"
                ).split(),
                2,
            )
        }
    ),
    "articulated": {frozenset({"open", "closed"})},
    "actuated": {frozenset({"open", "closed"})},
}


def _opposes(fact: dict[str, Any], effect: dict[str, Any]) -> bool:
    kind = fact["kind"]
    if kind != effect["kind"] or fact.get("frame") != effect.get("frame"):
        return False
    if kind == "relation" and (opposite := _AXIS_INVERSES.get(fact["value"])):
        return bool(
            _relation_pairs(fact, fact["value"]) & _relation_pairs(effect, opposite)
        )
    if (
        kind not in _OPPOSITE_VALUES
        or not set(fact.get("subjects", ())).intersection(effect.get("subjects", ()))
        or frozenset({fact.get("value"), effect.get("value")})
        not in _OPPOSITE_VALUES.get(kind, ())
    ):
        return False
    refs, other = set(fact.get("references", ())), set(effect.get("references", ()))
    return bool(refs & other) if kind == "relation" else refs == other


def _carried_facts(
    steps: dict[str, Any],
    seed: dict[str, list[dict[str, Any]]],
    parents: dict[str, str],
    step_arms: dict[str, set[str]],
) -> dict[str, list[dict[str, Any]]]:
    """Keep declaration evidence only until an operation invalidates it."""
    current = dict(seed)

    def affected(fact: dict[str, Any], subjects: set[str]) -> bool:
        roles = set(fact.get("subjects", ()))
        if fact.get("epoch") != "initial" or fact["kind"] != "pose_match":
            roles.update(fact.get("references", ()))
        if "part" in fact:
            roles.add(fact["part"])
        return any(_role_ancestry(role, parents) & subjects for role in roles)

    for ref, step in steps.items():
        verb, subjects, effects = step["verb"], set(step["subjects"]), step["effects"]
        displaced = (
            subjects
            if verb in {"pick", "place", "move_held", "push", "articulate"}
            else set()
        )
        displaced = displaced | {
            role
            for fact in effects
            if fact["kind"] == "transferred"
            for role in fact["subjects"]
        }
        rotated = (subjects if verb in {"orient", "turn", "articulate"} else set()) | {
            role
            for fact in effects
            if fact["kind"] == "orientation"
            for role in fact["subjects"]
        }

        def invalidates(fact: dict[str, Any]) -> bool:
            kind = fact["kind"]
            if any(_opposes(fact, effect) for effect in effects):
                return True
            if _is_control_transition(fact):
                return verb in {
                    "actuate",
                    "press",
                    "push",
                    "turn",
                    "articulate",
                } and affected(fact, subjects)
            if (
                kind == "pose_match"
                and "orientation" in fact["components"]
                and affected(fact, rotated)
            ):
                return True
            if kind == "gripper_relation" and affected(fact, displaced):
                return True
            if kind in _POSITION_FACTS and not (
                kind == "relation" and fact["value"] in {"compatible_with", "fixed_to"}
            ):
                return affected(fact, displaced)
            if kind == "orientation" and verb in {"orient", "turn", "articulate"}:
                return affected(fact, subjects)
            if kind in {"held", "grasped_at", "released"}:
                changes = (
                    {"pick", "grasp"}
                    if kind == "released"
                    else {"place", "release", "handover"}
                )
                return verb in changes and bool(subjects.intersection(fact["subjects"]))
            if kind in {"arms_at_initial_pose", "grippers_open", "gripper_relation"}:
                preserves = (
                    {"return_arms", "open_grippers"}
                    if kind == "arms_at_initial_pose"
                    else (
                        {"return_arms", "open_grippers", "move_gripper"}
                        if kind == "grippers_open"
                        else set()
                    )
                )
                return verb not in preserves and (
                    "auto" in fact["arms"]
                    or bool(step_arms[ref].intersection(fact["arms"]))
                )
            return False

        def retained(fact: dict[str, Any]) -> Iterator[dict[str, Any]]:
            if not invalidates(fact):
                yield fact
                return
            if field := _INDEPENDENT_STATE_MEMBERS.get(fact["kind"]):
                survivors = [
                    member
                    for member in fact[field]
                    if not invalidates({**fact, field: [member]})
                ]
                if survivors:
                    yield {**fact, field: survivors}
            elif fact["kind"] == "relation" and fact["value"] not in _GROUP_RELATIONS:
                # Multiple supporting faces prove a joint contact, not separate placements.
                reference_groups = (
                    [fact["references"]]
                    if fact["value"] in {"supported_by", "on_top_faces"}
                    else [[reference] for reference in fact["references"]]
                )
                for subject in fact["subjects"]:
                    for references in reference_groups:
                        pair = dict(fact, subjects=[subject], references=references)
                        if not invalidates(pair):
                            yield pair

        current = {
            origin: [part for fact in facts for part in retained(fact)]
            for origin, facts in current.items()
        }
        current[ref] = effects
    return current


def _validate_intent_audit(
    instruction: str, plan: dict[str, Any], audit: Any
) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    """Validate evidence and return terminal producers and proven undeclared goals."""
    _validate_audit_issues(instruction, audit, "checks")
    checks = audit.get("checks")
    spans = tuple(source_spans(instruction).values())
    future_sides = {span for span in spans if _FUTURE_SIDES.search(span)}
    witnessed_future_sides: set[str] = set()
    steps = {f"step_{i:02d}": row for i, row in enumerate(plan["steps"], 1)}
    order = {ref: index for index, ref in enumerate(steps)}
    phases = {
        index: (start, end)
        for start, end in _intent_phases(plan["steps"])
        for index in range(start, end)
    }
    conditions = {
        f"condition_{i:02d}": row for i, row in enumerate(plan["conditions"], 1)
    }
    graph_evidence = steps.keys() | conditions.keys()
    roles = {row["id"]: row["kind"] for row in plan["roles"]}
    evidence_ids = graph_evidence | roles.keys() | {f"role_{name}" for name in roles}
    parents = {row["id"]: row["part_of"] for row in plan["roles"] if "part_of" in row}
    invariants = {
        ref: row["fact"] for ref, row in conditions.items() if row["scope"] == "always"
    }

    def check_invariants(origin: str, facts: Iterable[dict[str, Any]]) -> None:
        for fact in facts:
            for invariant in invariants.values():
                # A position locked to an initial snapshot cannot rise. Only
                # these comparisons to the positive threshold exclude zero.
                if _opposes(invariant, fact) or (
                    invariant["kind"] == "pose_match"
                    and invariant["epoch"] == "initial"
                    and "position" in invariant["components"]
                    and fact["kind"] == "vertical_rise"
                    and fact["comparison"] in {"eq", "gt", "ge"}
                    and not set(invariant["subjects"]).isdisjoint(fact["subjects"])
                ):
                    raise AuthoringEvidenceError(
                        f"{origin} contradicts an always condition: {invariant}"
                    )

    for ref, condition in conditions.items():
        if condition["scope"] == "initial":
            check_invariants(f"{ref} initial fact", (condition["fact"],))
    for ref, step in steps.items():
        check_invariants(f"{ref} maintained fact", step.get("maintain", ()))
        check_invariants(f"{ref} effect", step["effects"])
    seed = {
        ref: [row["fact"]] for ref, row in conditions.items() if row["scope"] != "final"
    }
    step_arms = dict(zip(steps, _intent_step_arms(plan), strict=True))
    end = _carried_facts(steps, seed, parents, step_arms)
    final = {
        ref: row["fact"] for ref, row in conditions.items() if row["scope"] == "final"
    }
    surviving = [fact for facts in end.values() for fact in facts]
    earlier = [fact for facts in seed.values() for fact in facts]
    earlier.extend(fact for step in steps.values() for fact in step["effects"])

    def matches(fact: dict[str, Any], facts: list[dict[str, Any]]) -> bool:
        try:
            if fact["kind"] == "relation":
                verify_spatial_claim(fact, facts, roles)
            else:
                verify_state_claim(fact, facts, roles, combine_released=True)
        except AuthoringEvidenceError:
            return False
        return True

    for fact in final.values():
        if any(_opposes(fact, other) for other in (*surviving, *final.values())) or (
            fact["kind"] in _POSITION_FACTS | {"orientation", "gripper_relation"}
            and matches(fact, earlier)
            and not matches(fact, surviving)
        ):
            raise AuthoringEvidenceError(
                f"A final condition contradicts or outlives its ordered effects: {fact}"
            )
        if _is_control_transition(fact) and not matches(fact, surviving):
            raise AuthoringEvidenceError(
                "A final control transition needs its surviving action effect"
            )

    def slot_facts(
        refs: list[str], carried: dict[str, list[dict[str, Any]]], claim: dict[str, Any]
    ) -> list[dict[str, Any]]:
        def describes_slot(fact):
            if fact["kind"] != "relation":
                return False
            subjects = set(fact["subjects"])
            if fact["value"] in _AXIS_INVERSES and claim.get("value") in _AXIS_INVERSES:
                subjects = {
                    subject for subject, _ in _relation_pairs(fact, claim["value"])
                }
            return (
                subjects
                and all(roles[role] == "target" for role in subjects)
                and subjects.isdisjoint(claim.get("subjects", ()))
            )

        return [
            fact
            for ref in refs
            if ref in seed
            for fact in carried.get(ref, ())
            if describes_slot(fact)
        ]

    if not isinstance(checks, list) or not checks:
        raise ValueError("source_audit: no independently reviewed requirements")
    witnessed = False
    reviewed_parallel_groups = set()
    goal_producers: dict[str, set[str]] = {}
    added_conditions: list[dict[str, Any]] = []
    for index, check in enumerate(checks):
        if (
            not isinstance(check, dict)
            or not {"quote", "requirement", "evidence"} <= set(check)
            or set(check)
            - {
                "quote",
                "requirement",
                "evidence",
                "spatial",
                "scope",
                "state",
                "parallel_picks",
            }
            or (
                "scope" in check
                and (
                    not isinstance(check["scope"], str)
                    or check["scope"] not in _AUDIT_SCOPES
                )
            )
            or not isinstance(check["requirement"], str)
            or not check["requirement"].strip()
            or not isinstance(check["evidence"], list)
            or not check["evidence"]
        ):
            raise ValueError(
                "source_audit: each reviewed requirement needs source and graph evidence"
            )
        if not _source_quote(check["quote"], instruction):
            raise ValueError(
                f"source_audit: checks[{index}].quote is not a verbatim source span: "
                f"{repr(check['quote'])[:160]}"
            )
        unknown = [
            ref
            for ref in check["evidence"]
            if not isinstance(ref, str) or ref not in evidence_ids
        ]
        if unknown:
            raise ValueError(f"source_audit: unknown evidence IDs {unknown!r}")
        refs = check["evidence"]
        scope = check.get("scope")
        typed = "spatial" in check or "state" in check
        if "parallel_picks" in check:
            if not _explicit_pick_timing(check["quote"], instruction):
                raise AuthoringEvidenceError(
                    "Parallel picks need explicit source timing for acquisition; "
                    "separate hand-object assignments alone do not request synchrony"
                )
            pairs = check["parallel_picks"]
            if (
                scope != "action"
                or typed
                or not isinstance(pairs, dict)
                or pairs.keys() != {"left", "right"}
                or any(
                    not isinstance(role, str) or role not in roles
                    for role in pairs.values()
                )
            ):
                raise ValueError(
                    "A parallel pick check needs explicit left and right subjects in action scope"
                )
            branches = [steps[ref] for ref in refs if ref in steps]
            if (
                len(refs) != 2
                or len(branches) != 2
                or not (group := branches[0].get("parallel_group"))
                or branches[1].get("parallel_group") != group
                or {row["arms"][0]: row["subjects"][0] for row in branches} != pairs
            ):
                raise AuthoringEvidenceError(
                    "Simultaneous acquisition needs the correct paired parallel picks"
                )
            reviewed_parallel_groups.add(group)
            witnessed = True
            continue
        # Normalize the phase label without promoting role or step evidence.
        if typed and scope == "action":
            scope = "effect"
        elif scope == "initial" and not typed and graph_evidence.isdisjoint(refs):
            scope = "action"
        if typed and scope is None:
            raise ValueError("source_audit: a typed check needs an explicit scope")
        if scope in {None, "action"}:
            if not any(ref not in conditions for ref in refs):
                raise ValueError(
                    "source_audit: a condition check needs an explicit scope"
                )
            witnessed |= not steps.keys().isdisjoint(refs)
            continue
        if scope == "initial":
            refs = [
                ref for ref in refs if conditions.get(ref, {}).get("scope") == "initial"
            ]
            if not refs:
                raise AuthoringEvidenceError(
                    "An initial source requirement needs an initial condition"
                )
            facts = [conditions[ref]["fact"] for ref in refs]
        elif scope in {"always", "step"}:
            facts = [invariants[ref] for ref in refs if ref in invariants]
            if scope == "step":
                facts.extend(
                    fact
                    for ref in refs
                    if ref in steps
                    for fact in steps[ref].get("maintain", ())
                )
            if not facts:
                needed = (
                    "an always condition"
                    if scope == "always"
                    else "maintain or an always condition"
                )
                raise AuthoringEvidenceError(
                    f"A {scope} source requirement needs {needed}"
                )
        elif scope in {"before", "effect"}:
            cited = {ref for ref in refs if ref in steps}
            if not cited:
                raise AuthoringEvidenceError(
                    "A before check needs the step whose incoming state is reviewed"
                    if scope == "before"
                    else "An effect check needs the step producing it"
                )
            if scope == "before" and not typed:
                raise ValueError("A before check needs a typed state or spatial claim")
            latest = max(int(ref.removeprefix("step_")) for ref in cited) - 1
            start, end_of_phase = phases[latest]
            cutoff = start if scope == "before" else end_of_phase
            at_step = _carried_facts(
                dict(list(steps.items())[:cutoff]), seed, parents, step_arms
            )
            sources = refs if scope == "before" else cited
            facts = [fact for ref in sources for fact in at_step.get(ref, ())]
            if scope == "effect" and isinstance(check.get("spatial"), dict):
                facts += slot_facts(refs, at_step, check["spatial"])
        else:
            facts = [final[ref] for ref in refs if ref in final]
            surviving_effects = [
                fact for ref in refs if ref in steps for fact in end.get(ref, ())
            ]
            if not facts and not typed and surviving_effects:
                raise ValueError(
                    "A final check citing a step effect needs a typed claim"
                )
            if typed:
                facts += surviving_effects
                facts += [
                    fact
                    for ref in refs
                    if ref in invariants
                    for fact in end.get(ref, ())
                ]
                if isinstance(check.get("spatial"), dict):
                    facts += slot_facts(refs, end, check["spatial"])
            if not facts:
                raise AuthoringEvidenceError(
                    "A final check needs a final condition or an effect still valid at the end"
                )
        witnessed |= bool(graph_evidence.intersection(refs))
        claims = []
        if "spatial" in check:
            claim = check["spatial"]
            verify_spatial_claim(claim, facts, roles)
            claim = dict(claim)
            _expand_fact_shorthand(claim)
            claims.append(claim)
            if (
                claim.get("value") == "opposite_sides"
                and scope in {"effect", "final"}
                and _FUTURE_SIDES.search(check["quote"])
            ):
                reference = _role_ancestry(claim["references"][0], parents)
                checkpoint = at_step if scope == "effect" else end
                placed = max(
                    (
                        order[ref]
                        for ref, step in steps.items()
                        if step["verb"] == "place"
                        and ref in checkpoint
                        and reference.intersection(step["subjects"])
                    ),
                    default=None,
                )
                if placed is not None:
                    proof = [
                        fact
                        for ref in refs
                        if ref in order and order[ref] >= placed
                        for fact in checkpoint.get(ref, ())
                    ]
                    if matches(claim, proof):
                        witnessed_future_sides.update(
                            span
                            for span in future_sides
                            if check["quote"] in span or span in check["quote"]
                        )
            if (
                scope == "final"
                and claim.get("kind") == "relation"
                and any(_opposes(claim, other) for other in surviving)
            ):
                raise AuthoringEvidenceError(
                    "A final spatial check conflicts with an ordered effect"
                )
        if "state" in check:
            claim = check["state"]
            if not isinstance(claim, dict):
                raise ValueError("A source state claim must be a typed fact")
            kind = claim.get("kind")
            if kind == "untouched_except" and scope == "final":
                facts = [
                    f for ref in refs if ref in invariants for f in end.get(ref, ())
                ]
            if _is_control_transition(claim) and scope == "final":
                facts = surviving_effects
            if kind == "actuated" and scope == "effect":
                actions = (
                    {"push", "actuate"}
                    if _is_control_transition(claim)
                    else {"press", "actuate"}
                )
                facts = [
                    fact
                    for ref in refs
                    if ref in steps and steps[ref]["verb"] in actions
                    for fact in at_step[ref]
                ]
            verify_state_claim(
                claim,
                facts,
                roles,
                combine_released=scope in {"before", "effect", "final"},
            )
            claims.append(claim)
            if scope == "final" and any(_opposes(claim, other) for other in surviving):
                raise AuthoringEvidenceError(
                    "A final state check conflicts with an ordered effect"
                )
        if scope == "final":
            cited_steps = [step for step in steps if step in refs and end[step]]
            reviewed_goals = {ref: final[ref] for ref in refs if ref in final}
            for claim in claims:
                if any(_opposes(claim, goal) for goal in final.values()):
                    raise AuthoringEvidenceError(
                        "Conflicting reviewed terminal conditions"
                    )
                existing = {
                    goal_id: goal
                    for goal_id, goal in final.items()
                    if matches(claim, [goal])
                }
                if existing:
                    reviewed_goals.update(existing)
                elif not matches(claim, list(final.values())):
                    # A surviving audited effect is a source goal, not mere step completion.
                    goal_id = (
                        f"condition_{len(conditions) + len(added_conditions) + 1:02d}"
                    )
                    added_conditions.append(
                        dict(scope="final", fact=claim, quote=check["quote"])
                    )
                    final[goal_id] = reviewed_goals[goal_id] = claim
            if not cited_steps:
                continue
            for goal_id, goal in reviewed_goals.items():
                slots = (
                    slot_facts(refs, end, goal) if goal["kind"] == "relation" else []
                )
                proof = {step: end[step] for step in cited_steps}
                if not matches(
                    goal, slots + [f for facts in proof.values() for f in facts]
                ):
                    continue
                # Remove witnesses that do not contribute to the surviving goal.
                for step in cited_steps:
                    remaining = {
                        key: facts for key, facts in proof.items() if key != step
                    }
                    if matches(
                        goal, slots + [f for facts in remaining.values() for f in facts]
                    ):
                        proof = remaining
                if proof:
                    goal_producers.setdefault(goal_id, set()).update(proof)
    if {
        row["parallel_group"] for row in steps.values() if "parallel_group" in row
    } - reviewed_parallel_groups:
        raise AuthoringEvidenceError(
            "Every parallel pick group needs independently reviewed hand-object pairs"
        )
    if future_sides - witnessed_future_sides:
        raise AuthoringEvidenceError(
            "Opposite sides of a future reference need a surviving opposite_sides effect "
            "after the reference is placed; cite that effect and the future-reference source clause"
        )
    if not witnessed:
        raise ValueError(
            "A source audit needs at least one step or condition, not roles alone"
        )
    return {
        goal: [step for step in steps if step in proven]
        for goal, proven in goal_producers.items()
    }, added_conditions


def _review_scope_guards(audit: Any) -> Counter[str]:
    """Retain terminal, framed and simultaneous claims during citation repair."""
    guards: Counter[str] = Counter()
    checks = audit.get("checks") if isinstance(audit, dict) else None
    for check in checks if isinstance(checks, list) else ():
        if not isinstance(check, dict):
            continue
        final = check.get("scope") == "final"
        if final:
            guards["final"] += 1
        if isinstance(check.get("parallel_picks"), dict):
            guards[canonical_hash(["parallel_picks", check["parallel_picks"]])] += 1
        for name in ("spatial", "state"):
            fact = check.get(name)
            if (
                isinstance(fact, dict)
                and isinstance(fact.get("kind"), str)
                and (final or isinstance(fact.get("frame"), str))
            ):
                guards[canonical_hash([final, name, fact])] += 1
    return guards


def _native_resource_conflicts(
    calls: Iterable[tuple[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Check feasible holdings without rewriting logical resource IDs."""
    from embodichain.lab.task_program.language.semantic_contracts import (
        semantic_planning_catalog,
    )

    catalog = semantic_planning_catalog()["calls"]
    calls = tuple(calls)
    named = {
        resource
        for _, call in calls
        for resource in call.get("resources", {}).values()
        if resource != "auto"
    }
    # Legacy graphs use their own resource labels, not extra left/right aliases.
    if named <= {"left", "right"}:
        named = {"left", "right"}
    elif len(named) == 1:
        named.add("auto")  # One unnamed slot in a partially named two-arm plan.
    auto_choices = tuple((arm,) for arm in sorted(named))
    states = {_ArmState()}
    conflicts = []
    verbs = dict(
        acquire="pick",
        acquire_align="pick",
        held="move_held",
        release="release",
        transfer="handover",
    )
    for node_id, call in calls:
        cid = call.get("call_id", call["kind"])
        contract = catalog[cid]
        effect = contract["effect"]
        if effect == "preserve":
            continue
        args = call.get("arguments", call)
        subject = args.get("object", args.get("handle"))
        res = call.get("resources", {})
        primary = res.get("primary", "auto")
        choices = auto_choices if primary == "auto" else ((primary,),)
        if effect == "transfer":
            source, destination = res["source"], res["destination"]
            choices = (
                ((source, destination),)
                if source != destination and "auto" not in (source, destination)
                else ()
            )

        advanced = set()
        for state in states:
            busy = dict(state.grips)
            if effect == "coordinated_release":
                if not busy and {res.get("left"), res.get("right")} == {
                    "left",
                    "right",
                }:
                    advanced.add(state)
            elif effect == "control":
                if any(choice[0] not in busy for choice in choices):
                    advanced.add(state)
            elif effect == "held" and subject is None:
                if any(choice[0] in busy for choice in choices):
                    advanced.add(state)
            else:
                for choice in choices:
                    following = _advance_arms(state, verbs[effect], [subject], choice)
                    if following is not None:
                        advanced.add(following)
        if advanced:
            states = advanced
        else:
            # Keep the incoming state when recording a bad call, as in goal-route diagnostics.
            before = min(sorted(state.grips) for state in states)
            conflicts.append(
                dict(
                    node_id=node_id,
                    call=cid,
                    object=subject,
                    requested=res,
                    held_before=dict(before),
                    preconditions=contract["preconditions"],
                )
            )
    return conflicts


def _supported_program(operations, *, bindings=None, scene_objects=None):
    """Expand operations already normalized by _operation_rows."""
    roles, goals, milestones, clauses, calls, requirements = {}, {}, [], [], [], set()
    last = None
    holder = {}
    if bindings is None:
        bindings = {}
    scene_objects = scene_objects or {}

    def registered(cid, args, side):
        calls.append(
            dict(
                kind="registered",
                call_id="simulation." + cid,
                arguments=args,
                resources={} if side == "auto" else {"primary": side},
            )
        )

    def bind(kind, **values):
        bindings[f"call_{len(calls)}"] = dict(
            kind=kind, call=canonical_hash(calls[-1]), **values
        )

    def role(name):
        roles.setdefault(name, dict(id=name, kind="object"))

    def milestone(predicate, quote):
        nonlocal last
        mid = f"m{len(milestones)+1}"
        milestones.append(
            dict(
                id=mid,
                after=[last] if last else [],
                achieve=[dict(predicate, id=mid + ".achieve")],
            )
        )
        clauses.append((mid, quote))
        last = mid

    for op in operations:
        template = op["template"]
        contract = task_contract(template)
        obj, quote = op["object"], op["quote"]
        role(obj)
        if parent := op.get("part_of"):
            properties = roles[obj].setdefault("properties", {})
            if properties.setdefault("part_of", parent) != parent:
                raise ValueError(f"Conflicting scene object owners for part {obj!r}")
        side = op.get("primary", "auto")
        if side not in {"auto", "left", "right"}:
            raise ValueError("Invalid primary arm")
        if side == "auto" and obj in holder:
            side = holder[obj]
        resource = {} if side == "auto" else {"primary": side}
        if side != "auto":
            requirements.add(side)
        if template in {"E2", "E3", "E4"} and obj not in holder:
            # Reacquisition ends the earlier placement, not its milestone.
            goals.pop(obj, None)
        if template in {"E2", "E3", "E6", "E7", "E8", "E9"}:
            reuse = template == "E3" and obj in holder
            if template not in {"E2", "E3"} and obj in holder:
                raise ValueError(
                    f"Control arm occupied: {side}, held={holder}, requested={obj}"
                )
            if template == "E2":
                registered("axis_align", {"object": obj}, side)
                orientation = op.get("orientation")
                if orientation not in {"upright", "vertical"}:
                    raise ValueError("Invalid axis intent")
                pred = dict(
                    predicate=(
                        "object_upright" if orientation == "upright" else "orientation"
                    ),
                    subject=obj,
                    arguments={"orientation_goal": orientation},
                )
                bind("orientation", orientation=orientation)
                holder[obj] = side
            elif template == "E3":
                ref = op["reference"]
                role(ref)
                if obj == ref:
                    raise ValueError("Pour needs a distinct receiver")
                mode = op["mode"]
                if not reuse:
                    calls.append(dict(kind="pick", object=obj, resources=resource))
                holder[obj] = side
                registered("move_held_object", {"object": obj, "target": ref}, side)
                registered("pour", {"object": obj}, side)
                bind("pour_motion", reference=ref)
                pred = dict(
                    predicate="pour_motion" if mode == "motion" else "poured",
                    subject=obj,
                    reference=ref,
                )
            elif template == "E6":
                state = op["state"]
                if state not in {"open", "closed"}:
                    raise ValueError("Invalid slide endpoint")
                roles[obj]["kind"] = "articulation"
                registered("articulation_link_slide", {"handle": obj}, side)
                bind("joint_endpoint", state=state, joint_kind="prismatic")
                pred = dict(
                    predicate="articulation_joint_near",
                    subject=obj,
                    arguments=dict(target_state=state, joint_kind="prismatic"),
                )
            elif template == "E7":
                if op["state"] != "open":
                    raise ValueError("E7 only supports opening a hinged door")
                roles[obj]["kind"] = "articulation"
                registered("open_door", {"object": obj, "state": "open"}, side)
                pred = dict(
                    predicate="articulation_joint_near",
                    subject=obj,
                    arguments=dict(target_state="open", joint_kind="revolute"),
                )
            elif template == "E8":
                mode = op["mode"]
                if mode not in {"coarse", "setting"}:
                    raise ValueError("Invalid twist mode")
                args = {"mode": mode}
                if mode == "setting":
                    if type(op.get("setting")) is not int:
                        raise ValueError("Missing calibrated setting")
                    args["setting"] = op["setting"]
                registered("twist", {"object": obj}, side)
                bind("twist", **args)
                pred = dict(
                    predicate="semantic_goal",
                    subject=obj,
                    arguments={"operation": "twist", **args},
                )
            else:
                registered("press", {"object": obj}, side)
                pred = dict(predicate="pressed", subject=obj)
            milestone(pred, quote)
            goals[(obj, template)] = (dict(pred, id=f"final_{obj}_{template}"), quote)
        elif template == "E4":
            src, dst = op["source"], op["destination"]
            if {src, dst} != {"left", "right"}:
                raise ValueError("Invalid transfer resources")
            requirements.update((src, dst))
            if obj not in holder:
                calls.append(dict(kind="pick", object=obj, resources={"primary": src}))
            calls.append(
                dict(
                    kind="hand_over",
                    object=obj,
                    resources={"source": src, "destination": dst},
                )
            )
            holder[obj] = dst
            milestone(
                dict(
                    predicate="handover_complete",
                    subject=obj,
                    arguments=dict(transfer_arm=src, receive_arm=dst),
                ),
                quote,
            )
        else:  # E1 and E5
            ref, rel = op["reference"], op["relation"]
            role(ref)
            if obj == ref or rel not in {
                "on",
                "inside",
                "right_of",
                "left_of",
                "behind",
                "front_of",
                "above",
            }:
                raise ValueError("Invalid support endpoints")
            args = {}
            if template == "E1":
                if obj not in holder:
                    calls.append(dict(kind="pick", object=obj, resources=resource))
                holder.pop(obj, None)
                if side != "auto":
                    args["required_arm"] = side
                if rel not in {"on", "inside"}:
                    registered(
                        "place_relative",
                        dict(object=obj, reference=ref, relation=rel),
                        side,
                    )
                else:
                    calls.append(
                        {
                            "kind": "place",
                            "object": obj,
                            rel: ref,
                            "resources": resource,
                        }
                    )
            else:
                if holder or rel != "on":
                    raise ValueError(
                        "Coordinated release requires free arms and on destination"
                    )
                requirements.update(("left", "right"))
                milestone(dict(predicate="held_by_both_grippers", subject=obj), quote)
                calls.append(
                    dict(
                        kind="registered",
                        call_id="simulation.coordinated_transport",
                        arguments={"object": obj, "target": ref},
                        resources={"left": "left", "right": "right"},
                    )
                )
            pred = dict(
                predicate="relation",
                subject=obj,
                reference=ref,
                relation=rel,
                arguments=args,
            )
            milestone(pred, quote)
            goals[obj] = (dict(pred, id=f"final_{obj}"), quote)
        terminal = op.get("terminal_behavior", "none")
        if terminal != "none":
            terminal_quote = op["terminal_quote"]
            held_side = holder[obj]
            terminal_args = {} if held_side == "auto" else {"required_arm": held_side}
            if terminal == "hold":
                goals[(obj, "held")] = (
                    dict(
                        id=f"final_{obj}_held",
                        predicate="held",
                        subject=obj,
                        arguments=terminal_args,
                    ),
                    terminal_quote,
                )
            else:
                origin = "original_support_" + obj
                registered(
                    "move_held_object", {"object": obj, "target": origin}, held_side
                )
                calls.append(
                    dict(
                        kind="place",
                        object=obj,
                        on=origin,
                        resources={} if held_side == "auto" else {"primary": held_side},
                    )
                )
                bind("return_to_origin", subject=obj)
                milestone(
                    dict(
                        predicate="semantic_goal",
                        subject=obj,
                        arguments={"operation": "return_and_release", **terminal_args},
                    ),
                    terminal_quote,
                )
                holder.pop(obj)
        roles[obj]["affordances"] = sorted(
            set(roles[obj].get("affordances", ())) | contract.required_affordances
        )
    conflicts = _native_resource_conflicts(
        (f"call_{i}", call) for i, call in enumerate(calls, 1)
    )
    if conflicts:
        raise ValueError(
            f"Arm occupied or incompatible resources in native proposal: {conflicts[0]}"
        )
    if not goals:
        # A handover-only instruction can require the transfer without a final hold.
        transfer = milestones[-1]["achieve"][0]
        obj = transfer["subject"]
        goals[(obj, "E4")] = (dict(transfer, id=f"final_{obj}_E4"), clauses[-1][1])
    for goal, _ in goals.values():
        if goal["predicate"] == "held":
            subject = goal["subject"]
            arm = goal["arguments"].get("required_arm")
            if subject not in holder or (arm is not None and holder[subject] != arm):
                raise ValueError(
                    f"Terminal hold of {subject!r} conflicts with a later operation"
                )
    clauses.extend((p["id"], q) for p, q in goals.values())
    objective = dict(
        roles=list(roles.values()),
        milestones=milestones,
        goal=[p for p, _ in goals.values()],
        requirements=[
            dict(
                consumer="embodiment", key="required_arms", value=sorted(requirements)
            ),
            scene_objects_requirement(
                scene_objects,
                {
                    name: name
                    for name, role in roles.items()
                    if "part_of" not in role.get("properties", {})
                },
            ),
        ],
    )
    return objective, clauses, calls


def verify_witness(spec, graph, *, support_targets, bindings=None):
    """Check ordered occurrence effects derived from canonical calls against frozen obligations."""
    from .goal_planning import _facts
    from embodichain.lab.task_program.language.semantic_contracts import (
        validate_semantic_call_payload,
    )

    graph = validate_semantic_task_graph(graph)
    facts = []
    roles = {r.name for r in spec.roles}
    bindings = bindings or {}
    for i, n in enumerate(graph["nodes"]):
        call = n["call"]
        effect = validate_semantic_call_payload(call)["effect"]
        params = call.get("arguments", call)
        if any(
            params.get(k) and params[k] not in roles
            for k in ("object", "handle", "reference")
        ):
            raise ValueError("Undeclared witness role")
        if n["depends_on"] != ([graph["nodes"][i - 1]["id"]] if i else []):
            raise ValueError("Pilot witness requires a dependency chain")
        # This route binds the target symbol to the required support role. Native
        # geometry, contact, release and interval evidence are still pending.
        if call.get("call_id") == "simulation.coordinated_transport":
            args = call["arguments"]
            if (
                args["target"] not in support_targets
                or support_targets[args["target"]] not in roles
                or call["resources"] != {"left": "left", "right": "right"}
            ):
                raise ValueError("Invalid coordinated resource or target binding")
            facts.extend(
                [
                    (
                        n["id"],
                        "acquire",
                        "held_by_both_grippers",
                        {"subject": args["object"]},
                    ),
                    (
                        n["id"],
                        "release",
                        "relation",
                        dict(
                            subject=args["object"],
                            reference=support_targets[args["target"]],
                            relation="on",
                        ),
                    ),
                ]
            )
        else:
            facts.extend((n["id"], "terminal", k, a) for k, a in _facts(call))
            cid = call.get("call_id", call["kind"])
            entity = params.get("object", params.get("handle"))

            def fact(kind, **args):
                facts.append((n["id"], "terminal", kind, dict(subject=entity, **args)))

            if cid == "simulation.press":
                fact("pressed")
            b = bindings.get(n["id"], {})
            if b and canonical_hash(call) == b["call"]:
                if cid == "simulation.axis_align" and b["kind"] == "orientation":
                    fact(
                        (
                            "object_upright"
                            if b["orientation"] == "upright"
                            else "orientation"
                        ),
                        orientation_goal=b["orientation"],
                    )
                elif cid == "simulation.pour" and b["kind"] == "pour_motion":
                    fact("pour_motion", reference=b["reference"])
                elif (
                    cid == "simulation.articulation_link_slide"
                    and b["kind"] == "joint_endpoint"
                ):
                    fact(
                        "articulation_joint_near",
                        target_state=b["state"],
                        joint_kind=b["joint_kind"],
                    )
                elif cid == "simulation.twist" and b["kind"] == "twist":
                    fact(
                        "semantic_goal",
                        operation="twist",
                        **{k: b[k] for k in ("mode", "setting") if k in b},
                    )
                elif cid == "place" and b["kind"] == "return_to_origin":
                    fact(
                        "semantic_goal",
                        operation="return_and_release",
                        required_arm=call.get("resources", {}).get("primary"),
                    )
            if entity is not None and effect in {
                "acquire",
                "acquire_align",
                "held",
                "transfer",
            }:
                arm = call.get("resources", {}).get(
                    "destination" if effect == "transfer" else "primary"
                )
                fact("held", required_arm=arm)

    def matches(p, f):
        return (
            p.expected is True
            and p.operator == "eq"
            and p.name == f[2]
            and all(k in f[3] and f[3][k] == v for k, v in p.arguments.items())
        )

    achieved = {}
    links = {}
    missing = []
    remaining = list(spec.milestones)
    while remaining:
        m = next(m for m in remaining if all(k in achieved for k in m.after))
        start = max((achieved[k] + 1 for k in m.after), default=0)
        idx = next(
            (
                i
                for i in range(start, len(facts))
                if all(matches(p, facts[i]) for p in m.achieve)
            ),
            len(facts),
        )
        achieved[m.id] = idx
        if idx < len(facts):
            links[m.id] = facts[idx][:2]
        else:
            missing.append(m.id)
        remaining.remove(m)
    for p in spec.goal:
        subject = p.arguments["subject"]
        last = next(
            (
                n
                for n in reversed(graph["nodes"])
                if n["call"].get("arguments", n["call"]).get("object") == subject
            ),
            None,
        )
        if (
            p.name
            in {
                "pressed",
                "pour_motion",
                "poured",
                "object_upright",
                "orientation",
                "articulation_joint_near",
            }
            or p.arguments.get("operation") == "twist"
        ):
            candidates = [
                f for f in facts if f[2] == p.name and f[3].get("subject") == subject
            ]
            last = {"id": candidates[-1][0]} if candidates else None
        hit = next(
            (
                f
                for f in reversed(facts)
                if last and f[0] == last["id"] and matches(p, f)
            ),
            None,
        )
        if hit:
            links[p.id] = hit[:2]
        else:
            missing.append(p.id)
    conflicts = _native_resource_conflicts((n["id"], n["call"]) for n in graph["nodes"])
    missing.extend(k for k in ("temporal", "invariants") if getattr(spec, k))
    return dict(
        status="covered" if not missing and not conflicts else "uncovered",
        producers=links,
        uncovered=missing,
        resource_conflicts=conflicts,
        physical_verdict="not_evaluated",
    )


def _declared_support_targets(doc):
    """Allow only support identities declared by frozen goals or milestones."""
    predicates = [
        *doc["objectives"]["goal"],
        *[p for m in doc["objectives"]["milestones"] for p in m["achieve"]],
    ]
    return {
        p["reference"]: p["reference"]
        for p in predicates
        if p["predicate"] == "relation" and p.get("reference")
    }


def _has_groundable_landings(graph):
    last_calls = {}
    for node in graph["nodes"]:
        call = node["call"]
        kind = call.get("call_id", call["kind"])
        if kind not in {
            "pick",
            "hand_over",
            "place",
            "simulation.coordinated_transport",
        }:
            return False
        last_calls[call.get("arguments", call)["object"]] = kind
    return all(
        kind in {"place", "simulation.coordinated_transport"}
        for kind in last_calls.values()
    )


def _scene_projection(doc, graph, packet):
    """Project the rigid objects and destinations of a checked manipulation sequence."""
    from .image_planning import _validate_grounding
    from ..task_spec.publication import project_scene_requirements

    goals = doc["objectives"]["goal"]
    nodes = graph["nodes"]
    from ..task_spec import TaskSpec

    spec = TaskSpec.from_dict(
        {"roles": doc["configuration"]["roles"], **doc["objectives"]}
    )
    coverage = verify_witness(
        spec,
        graph,
        support_targets=_declared_support_targets(doc),
    )
    if coverage["status"] != "covered":
        raise ValueError("Uncovered scene witness")
    achieved = {}
    for milestone in doc["objectives"]["milestones"]:
        producer = coverage["producers"].get(milestone["id"])
        if producer:
            for predicate in milestone["achieve"]:
                if predicate["predicate"] == "relation":
                    achieved[producer[0]] = predicate
    annotations = {}
    for index, n in enumerate(nodes):
        c = n["call"]
        carry = c.get("call_id") == "simulation.coordinated_transport"
        if not carry and c["kind"] not in {"pick", "hand_over", "place"}:
            raise ValueError("Unsupported grounded manipulation call")
        subject = c.get("arguments", c)["object"]
        landing = next(
            (
                row
                for row in nodes[index:]
                if row["id"] in achieved
                and achieved[row["id"]].get("subject") == subject
            ),
            None,
        )
        goal = achieved.get(landing["id"]) if landing else None
        if not goal or goal["relation"] not in {"on", "inside"}:
            raise ValueError("Missing occurrence destination")
        reference, relation = goal["reference"], goal["relation"]
        if (carry and (relation != "on" or c["arguments"]["target"] != reference)) or (
            c["kind"] == "place" and c.get(relation) != reference
        ):
            raise ValueError("Placement witness mismatch")
        annotations[n["id"]] = dict(
            object=subject, reference=reference, relation=relation
        )
    if any(
        c.get("constraint") or c["resolution"] != "resolved"
        for c in doc["interpretation"]["clauses"]
    ):
        raise ValueError(
            "Hybrid scene qualification cannot discharge additional qualifiers"
        )
    roles = [r["id"] for r in doc["configuration"]["roles"]]
    if packet["image_sha256"] != doc["inputs"]["image"]["sha256"]:
        raise ValueError("Grounding belongs to a different source image")
    grounding = _validate_grounding(
        json.dumps(packet["grounding"]),
        [dict(reference_id=r, reference=r, quantifier="one", count=0) for r in roles],
    )
    if set(grounding) != {"objects", "bindings"} or any(
        grounding["bindings"][r] != [r] for r in roles
    ):
        raise ValueError(
            "Grounding must preserve hybrid role IDs and frozen declarations"
        )
    return {
        "scene_requirements": project_scene_requirements(doc),
        "image_grounding": grounding,
        "plan_coverage": {"role_entities": grounding["bindings"]},
        "semantic_graph_validation": {"source": {"annotations": annotations}},
    }


def _ground_frozen_roles(doc, graph, image, services, *, scene_context=None):
    """Request image observations without changing a frozen role or objective."""
    from ..services.compat import structured_call
    from ..services.runtime import service_scope

    instruction = doc["inputs"]["instruction"]
    roles = [r["id"] for r in doc["configuration"]["roles"]]
    errors = []
    prompt = dict(instruction=instruction, roles=roles, validation_errors=errors)
    system_prompt = "Locate each role by inspecting the provided image. For each role return description (including visible shape, size, location) and bbox as ordered integer [xmin,ymin,xmax,ymax] on a 0..1000 grid. For ambiguous or hidden roles return null for bbox. Do not guess coordinates."
    if scene_context is not None:
        prompt["scene_context"] = scene_context
        system_prompt += (
            " Use scene_context as reference data to identify the frozen roles,"
            " not as evidence that an object is visible."
        )
    with service_scope(services) as run:
        for attempt in range(2):
            try:
                raw = structured_call(
                    services=run,
                    role="task_image_grounding",
                    image_path=image,
                    schema={
                        "type": "object",
                        "required": roles,
                        "additionalProperties": False,
                        "properties": {
                            r: {
                                "type": "object",
                                "required": ["description", "bbox"],
                                "additionalProperties": False,
                                "properties": {
                                    "description": {"type": "string"},
                                    "bbox": {
                                        "anyOf": [
                                            {"type": "null"},
                                            {
                                                "type": "array",
                                                "minItems": 4,
                                                "maxItems": 4,
                                                "items": {
                                                    "type": "integer",
                                                    "minimum": 0,
                                                    "maximum": 1000,
                                                },
                                            },
                                        ]
                                    },
                                },
                            }
                            for r in roles
                        },
                    },
                    system_prompt=system_prompt,
                    prompt=json.dumps(prompt, ensure_ascii=False),
                )
                if set(raw) != set(roles):
                    raise ValueError("Expected one entry per frozen role")
                packet = dict(
                    image_sha256=doc["inputs"]["image"]["sha256"],
                    grounding={
                        "objects": [
                            dict(
                                id=r,
                                description=raw[r]["description"],
                                bbox=raw[r]["bbox"],
                                provenance=(
                                    "runtime_required"
                                    if raw[r]["bbox"] is None
                                    else "observed"
                                ),
                            )
                            for r in roles
                        ],
                        "bindings": {r: [r] for r in roles},
                    },
                )
                projection = _scene_projection(doc, graph, packet)
                break
            except (TypeError, ValueError, KeyError) as exc:
                errors.append(str(exc))
                if attempt:
                    raise
    return projection, packet


def publish_hybrid(
    instruction: str,
    task_id: str,
    image: Path,
    output: Path,
    services: Any = None,
    deployment: Path | None = None,
    *,
    scene_grounding: Path | None = None,
    scene_context: Path | None = None,
) -> dict[str, Any]:
    from ..services.runtime import service_scope
    from ..services.compat import VisionFacade, structured_call
    from .image_planning import _json_response

    image = Path(image).expanduser().resolve(strict=True)
    output = Path(output).expanduser().absolute()
    if scene_grounding is not None:
        scene_grounding = Path(scene_grounding).expanduser().resolve(strict=True)
    context = _load_scene_context(scene_context) if scene_context is not None else None
    scene_objects = {entry["id"]: entry for entry in (context or {}).get("objects", [])}
    image_bytes = image.read_bytes()
    prompt = {
        "instruction": instruction,
        "request": (
            "Return JSON {operations:[...]}. Include only requested operations in order. "
            "Every row requires template, object and a verbatim instruction quote. "
            "Use consistent short ASCII role IDs, preserving caller-declared case and identity across steps. "
            "Use primary:auto|left|right on single-arm rows; use auto if unspecified. "
            "E1 pick and place {reference,relation:on|inside|left_of|right_of|behind|front_of|above}; "
            "E2 alignment {orientation:upright|vertical}; E3 pour {reference,mode:motion|fluid}; "
            "E4 explicit handover {source:left|right,destination:left|right}; "
            "E5 explicit two-arm transport with release {reference,relation:on}; "
            "E6 slide {object:part,state:open|closed}; E7 hinged opening {object:part,state:open}; "
            "E8 twist {mode:coarse|setting,setting:integer only for setting}; E9 press {object}. "
            "E6-E9 may include part_of:scene object ID for the asset that owns the operated part. "
            "E2/E3/E4 may state terminal_behavior:hold|return_and_release|none. Require a separate "
            "verbatim terminal_quote for hold or return_and_release; use hold only for an explicit final hold, "
            "and return_and_release only for an explicit put-back, not merely restoring orientation. "
            "Use none when no terminal condition is specified for that operation. If the instruction "
            "then places the object at a named destination, represent that as a following E1 using the "
            "same object and holding arm. Do not invent placement or release. "
            "E3 restores orientation while holding; it does not go back to the original position on its own. "
            "Distinguish a motion with no content transfer from an actual liquid transfer; do not "
            "reinterpret a solid transfer or an ambiguous pour as one of those modes. "
            "Keep on distinct from inside, preserve explicit arm/part/target constraints, and never "
            "invent extra operations. For E5 retain each explicitly requested payload placement in order."
        ),
    }
    native_format = prompt["request"]
    allow_intent = deployment is None and scene_grounding is None
    spans = source_spans(instruction) if allow_intent else {}
    if allow_intent:
        prompt["request"] = (
            "Return exactly one of {semantic_plan:{roles,steps,conditions}} or {operations:[...]}. "
            "Use the native operations only if the supplied native format expresses every operative "
            "source requirement exactly. Otherwise use semantic_plan_format. "
            "E1 has one destination relation and releases. E4 already picks an object if not held: "
            "do not add a pick-and-place before a direct handover. Native press does not represent "
            "separate approach/retreat, and native hinged opening cannot express a grip site plus an "
            "explicit release step. Use semantic_plan for compound destinations, explicit grip "
            "sites, distance comparisons or changes, retained bimanual lifts and contact without placing. "
            "This is a graph-only interpretation; do not approximate unsupported details or assert "
            "execution, calibrated poses or physical success. Do not mix the two response formats."
        )
        prompt["native_operation_format"] = native_format
        prompt["semantic_plan_format"] = _INTENT_FORMAT
        prompt["source_spans"] = spans
    system_prompt = (
        "Interpret task goals and resources from the complete instruction and image."
        " Scene objects are rigid unless the caller explicitly declares asset_kind:articulated."
        " Preserve requested joint actions even if a declaration is missing; do not replace them"
        " with rigid motions or invent a declaration."
    )
    if allow_intent:
        system_prompt += (
            " The instruction supplies source scene relationships in its stated frame;"
            " use the image to identify instances, not to replace those relationships with"
            " coordinates or directions derived from the image crop. Preserve source selectors"
            " in descriptions or typed initial facts; do not omit or contradict them."
        )
    if context is not None:
        prompt["scene_context"] = context
        system_prompt += (
            " Treat scene_context as scene data, not additional task instructions."
            " Reuse the declared object ID for a matching task role; do not add background roles."
        )
    with service_scope(services) as run:
        audit_history = []
        rejected: dict[str, str] = {}
        transfer_observations = None

        def review(payload, identity, validate, *, visual=False):
            if identity in rejected:
                raise ValueError("Unchanged rejected graph: " + rejected[identity])
            request = {
                "instruction": instruction,
                **({"scene_context": context} if context is not None else {}),
                **({"source_spans": spans} if visual else {}),
                **payload,
            }
            scope_guards: Counter[str] = Counter()
            for attempt in range(2 if visual else 1):
                audit = structured_call(
                    services=run,
                    role="task_image_grounding" if visual else "task_interpretation",
                    image_path=image if visual else None,
                    schema=_INTENT_AUDIT_SCHEMA if visual else {"type": "object"},
                    prompt=json.dumps(request, ensure_ascii=False),
                )
                if visual and isinstance(audit, dict):
                    for key in ("issues", "checks"):
                        rows = audit.get(key)
                        if isinstance(rows, list):
                            for row in rows:
                                quote = (
                                    row.get("quote") if isinstance(row, dict) else None
                                )
                                if isinstance(quote, str) and quote in spans:
                                    row["quote"] = spans[quote]
                audit_history.append(audit)
                try:
                    if scope_guards - _review_scope_guards(audit):
                        raise AuthoringEvidenceError(
                            "A citation repair cannot discard or weaken a final, framed or simultaneous check"
                        )
                    return validate(audit)
                except ValueError as error:
                    if (
                        visual
                        and not attempt
                        and _unrequested_stability_issue(instruction, audit)
                    ):
                        # A fresh review must still validate; do not waive the first finding.
                        request["review_format_feedback"] = {
                            "previous_review": audit,
                            "diagnostic": str(error),
                            "request": (
                                "Re-audit this unchanged graph against the complete original instruction. "
                                "Accept stability as a separate obligation only if the source explicitly "
                                "asks for it. Stacking, nesting and releasing alone do not request stable. "
                                "Still report any actual mismatch; otherwise return the complete set of checks."
                            ),
                        }
                        continue
                    if isinstance(error, AuthoringEvidenceError) or (
                        isinstance(audit, dict) and audit.get("issues")
                    ):
                        rejected[identity] = str(error)
                        raise
                    if (
                        attempt
                        or not visual
                        or not isinstance(audit, dict)
                        or audit.get("issues") != []
                    ):
                        raise
                    # Repair the positive review's citations, not the candidate.
                    scope_guards = _review_scope_guards(audit)
                    request["review_format_feedback"] = {
                        "previous_review": audit,
                        "diagnostic": str(error),
                        "request": (
                            "Recheck this same graph. Correct the diagnosed field or citation. "
                            "For quote use a source_spans key directly, not a rewritten fragment. "
                            "Use existing evidence IDs; keep all final, framed and parallel claims. "
                            "Put scope in the check's own JSON field, never in requirement text."
                        ),
                    }

        def audit_calls(operations, calls, bindings):
            from embodichain.lab.task_program.language.semantic_contracts import (
                semantic_planning_catalog,
            )

            catalog = semantic_planning_catalog()["calls"]
            declared_effects = {}
            owners = {
                op["object"]: op["part_of"] for op in operations if "part_of" in op
            }
            for call in calls:
                kind = call.get("call_id", call["kind"])
                if kind in catalog:
                    contract = catalog[kind]
                    declared_effects[kind] = contract["effect"]
                    if contract["effect"] == "control":
                        for field, argument_type in contract["required"].items():
                            if argument_type == "part":
                                part = call["arguments"][field]
                                require_articulation(part, scene_objects, owners)
            review(
                {
                    "operation_format": native_format,
                    "operations": operations,
                    "ordered_calls": calls,
                    "declared_call_effects": declared_effects,
                    "bound_effects": bindings,
                    "request": (
                        "Independently reconstruct each required operation from the complete original instruction. "
                        "Use scene_context only to identify objects, and reuse a proposed role ID only for the same object. "
                        "Check the proposal and actual expanded calls against all source actions, order, subjects, "
                        "targets, relations, mode, orientation, arms, terminal behavior and negative qualifiers. "
                        "Check call effects: a placement releases, and an alignment alone does not provide a measured lift. "
                        "Check every occurrence; a single occurrence cannot cover a repeated action. "
                        "An acquisition or orientation restoration already performed by a call needs no duplicate call. "
                        "Reject unrequested return, placement, release or transfer. Reject requirements not "
                        "expressed by the actual graph rather than treating them as covered. "
                        "If there is any mismatch, return {issues:[{quote:verbatim source clause,reason:specific mismatch}]}. "
                        "Otherwise return {issues:[],source_operations:[independently reconstructed native rows in source order]}, "
                        "including verbatim quote and any terminal_quote as required by operation_format."
                    ),
                },
                canonical_hash({"calls": calls, "bindings": bindings}),
                lambda audit: _validate_source_audit(instruction, operations, audit),
            )

        def without_quotes(row):
            return {
                key: value
                for key, value in row.items()
                if key not in {"quote", "release_quote"}
            }

        def audit_intent(plan):
            nonlocal transfer_observations
            has_transfer = any(
                step["verb"] == "transfer_contents" for step in plan["steps"]
            )
            if has_transfer and transfer_observations is None:
                # Identify visible endpoints without the candidate's role labels.
                transfer_observations = structured_call(
                    services=run,
                    role="task_image_grounding",
                    image_path=image,
                    schema={"type": "object"},
                    prompt=json.dumps(
                        {
                            "instruction": instruction,
                            "request": (
                                "Inspect the original image. For each requested transfer of contents "
                                "identify the source container and recipient separately. Match each "
                                "named object type to its visible shape and distinguishing appearance "
                                "before assigning its function. Do not infer identity from unseen contents. "
                                "Return {transfers:[{source:visible appearance and kind,"
                                "recipient:visible appearance and kind}]} in source order. "
                                "Say ambiguous or not visible if there is insufficient visual evidence."
                            ),
                        },
                        ensure_ascii=False,
                    ),
                )
            return review(
                {
                    "semantic_plan_format": _INTENT_FORMAT,
                    **(
                        {"independent_transfer_observations": transfer_observations}
                        if has_transfer
                        else {}
                    ),
                    "roles": {f"role_{row['id']}": row for row in plan["roles"]},
                    "steps": {
                        f"step_{i:02d}": row for i, row in enumerate(plan["steps"], 1)
                    },
                    "conditions": {
                        f"condition_{i:02d}": row
                        for i, row in enumerate(plan["conditions"], 1)
                    },
                    "request": (
                        "Independently extract the required actions, order, exact object identities, arm use, "
                        "spatial and quantitative conditions, states that must persist, and terminal obligations "
                        "from the COMPLETE original instruction. Treat explicit source spatial descriptions "
                        "as facts in their stated frame. Use the image to identify instances and resolve "
                        "explicitly image-relative selectors, not to infer a different table frame. "
                        "Do not reject a source-supported table-relative location because the object "
                        "looks near the middle of the image crop. "
                        "If no frame is specified, a consistent natural frame such as table for a "
                        "tabletop relation is optional; do not reject its presence or absence unless "
                        "it changes the stated relationship. "
                        "The side of a center is a lateral offset from it, not a far-depth claim. "
                        "Do not require invented positions for roles identified "
                        "by attributes or innate rank. Do not require hidden contents to be visually discernible "
                        "if their identity is stated by the instruction. Then check each obligation against the "
                        "structured graph, respecting specific operative directions over a generic summary. "
                        "Role descriptions may identify instances by attributes and rank, not substitute for "
                        "dynamic action or state evidence. Do not turn descriptive identity into an extra goal. "
                        "Require kind:stable only when the original source explicitly asks for stability, "
                        "balance or preventing toppling. Stacking, nesting, support, and releasing do not "
                        "on their own require a separate stability claim. When requested, the fact must "
                        "name the actual members, not a new aggregate role. "
                        "For objects requested one on each side of a future reference placement, return "
                        "a spatial opposite_sides check scoped to effect or final. Quote the future-reference "
                        "clause itself and cite the placement of the reference, or a later effect that "
                        "establishes the relation. Earlier left/right facts are not evidence after the reference moves. "
                        "In particular, verify named counts, reference-role operands, support/contact faces, "
                        "and required spatial slot constraints in the typed fields, not in names or prose. "
                        "For location relative to a named part, verify the part is the operand and "
                        "part_of links its owner; do not require a second relation to the owner. "
                        "Transfer into a container does not prove the earlier positioning above its opening. "
                        "For a transfer, check the actual visible source and recipient separately against "
                        "the image and independent_transfer_observations. Correct wrong observations from "
                        "the image; do not accept reversed visible operands just because their role IDs match. "
                        "Adjacency plus the matching side relation can express beside a side of an object; "
                        "do not turn a side into a required end. "
                        "For operative starting conditions (e.g. relative pose reuse, support to preserve, "
                        "or a required way to distinguish otherwise similar instances), inspect initial facts. "
                        "For a purely descriptive initial clause, match a faithful role description with scope:action "
                        "or an initial condition with scope:initial. Cite the role ID only for source identity, never as proof of a "
                        "requested action, outcome, positional anchor or ongoing constraint. "
                        "For an explicit starting distance comparison between two items and one anchor, "
                        "check scope:initial and state:{kind:distance_compare,subjects:[item],"
                        "references:[anchor,comparison item],comparison:gt or lt}. Do not confuse it with "
                        "a category like near/far or a change compared to the same item's initial distance. "
                        "For exact on-support centering, require spatial with value:centered_on. "
                        "Near_center_of alone does not center on a support. For a named long-axis match, "
                        "require state with kind:orientation, value:aligned_long_axis and the actual "
                        "reference role, not generic aligned_with. When the source says a part points "
                        "toward, faces, or points away from a named object, use state with kind:orientation, "
                        "value:pointing_toward or pointing_away_from, and the named reference role. "
                        "An axis direction or relative location is not proof of facing the named object. "
                        "For such a starting direction include scope:initial. Centered_on already implies support. "
                        "Check the exact face when specified as a contact requirement; for ordinary "
                        "placement on a flat object's top use on or on_top_faces, without duplicate facts. "
                        "For pressing a held tool against a target, cite the held motion and its "
                        "spatial touching effect with the exact tool part and target. Button actuation "
                        "and placement after release do not prove that contact. "
                        "For conditions that must "
                        "persist throughout, include scope:always in their check and cite an always condition. "
                        "A terminal-only posture or separation fact is not a throughout constraint. "
                        "Use scope:before for the state just before the latest cited step; cite that step and the earlier effect or condition that establishes the state. "
                        "Use scope:step only for during-step maintain, and scope:effect for immediately after the latest cited step. "
                        "For a state requested at task completion use scope:final. Cite its final condition if present, "
                        "and cite producing steps when their effects still prove it after later actions. Include a typed "
                        "state/spatial claim when citing a step. Initial and maintain facts "
                        "do not prove a final result. Every check has a separate scope field, not text in requirement. "
                        "Use scope:action for actions, identity and arm-selection with step or role evidence. Cite affected step arms. "
                        "When two different objects must be picked simultaneously, include a separate "
                        "scope:action check with parallel_picks:{left:left object ID,right:right object ID} "
                        "and cite exactly the two pick steps in the same parallel_group. The quote must "
                        "contain the explicit acquisition timing cue. 'Use both arms' or naming each hand's "
                        "object does not require synchrony; reject an unrequested parallel_group. "
                        "Two arms picking the same object instead use one bimanual pick without parallel_picks. "
                        "Each state or spatial value is one typed fact object, never an array or prose. Split checks across scopes. "
                        "For a no-touch requirement on all unspecified other objects, use scope:always and "
                        "a state fact {kind:untouched_except,subjects:[exempt role IDs]}. A named or prose "
                        "group under untouched does not protect the complement. For pressing a named button "
                        "or control, include a state fact {kind:actuated,value:pressed,subjects:[site role ID]} "
                        "and cite the step which presses that site, not a broad parent face. "
                        "For a control requested to end in its other stable position, instead check "
                        "scope:final and state {kind:actuated,value:other_from_initial,subjects:[control part ID]}. "
                        "Cite the push or actuation on that specific control. A press, an unspecific "
                        "turn, and a final declaration without a producing action are not evidence. "
                        "If the source requires an object not to move, cite the applicable invariant "
                        "or during-step condition and use a state claim of its own initial pose: "
                        "pose_match with the same object as subject and reference, epoch:initial, "
                        "and both position and orientation for a fixed pose. If only one component "
                        "is specified, check only that component. A supporting surface "
                        "named for the reference frame is not the pose reference. "
                        "Required effects may be encoded in the verbs, effects or task-wide conditions. "
                        "An ordinary unquantified pickup is already expressed by pick, even without a "
                        "separate direction or movement step. Do not demand a second unsupported lift. "
                        "If a distinct height, target pose or clearance is requested, check that outcome. "
                        "For an explicit lift followed by moving the same held object away, verify two "
                        "ordered move_held steps: direction:up then direction:away. A single upward step "
                        "with an increased distance from the start does not express the later motion. "
                        "Check extent:short on that later step when a short displacement is requested. "
                        "Check explicit movement directions: lifting away is upward motion, not merely "
                        "clearance, and do not accept numeric thresholds or counts invented for qualitative instructions. "
                        "Check source specificity: a named contact or hanging part does not authorize "
                        "grasped_at that part. Lying on a support does not specify lying on a side. "
                        "Distinct fixtures do not imply a required physical gap. "
                        "An established fact persists until a later step changes it; do not demand duplicate "
                        "final facts or stronger synonyms. An exact placement at a typed slot inherits its "
                        "stated position and support, but mere nearness does not. A legal acquisition witnesses "
                        "that the subject is movable; actual fixed attachment needs its own fact. "
                        "A single concrete role is not evidence of a constraint on every other object. "
                        "Do not require any physical binding, or a terminal hold, release or gripper state "
                        "not requested. Reject promoting an unspecified terminal choice into a requirement. "
                        "Reject concrete semantic gaps, unjustified extra restrictions/actions or wrong "
                        "identity, not alternative but equivalent encodings. When a source leaves a choice "
                        "open (such as which interchangeable object occupies which permitted slot), accept "
                        "specific assignments that meet every explicit constraint. Typed part_of is "
                        "evidence of ownership; it does not require a second fixed_to assertion. "
                        "For an operative object-to-object proximity, on-support placement, or "
                        "direction relative to a named part include a spatial field in its check: "
                        "{kind:relation,value:one of semantic_plan_format.spatial_claim_values,subjects:[role ID],"
                        "references:[role ID],frame:the explicitly requested viewpoint if any}. A reference object or "
                        "surface is not a coordinate frame: on the table or centered on the table names the reference, not a viewpoint. "
                        "Omit frame unless explicitly specified; preserve it when specified. A lateral/depth selector is "
                        "a state:{kind:view_location,...} check, not a spatial relation. "
                        "Cite the step without a spatial field for "
                        "arm-only gripper_relation or descriptive source roles. Seating implies on; "
                        "hanging is a distinct state, "
                        "not an on-support placement. For near_center_of, name the actual reference surface or object. "
                        "Use the exact near_* region when the source identifies it; do not relabel ends as "
                        "proximity to a different role. Placement near a surface does not prove support on it. "
                        "Check a temporary placement with scope:effect at the placement. For the location the next pick starts from, "
                        "use scope:before and cite both the placement and the pick; picking changes the location. "
                        "Cite the explicit typed facts establishing the requested relation for every subject. "
                        "A single-axis middle selector is never proximity to a center. If an exact slot "
                        "supplies the relation, cite both the placement and the slot's relation; do not "
                        "treat merely nearby slots as exact. Omit spatial for other checks. "
                        "A row centered by a specified member does not require the other members "
                        "to have independent center destinations. Placement with its released fact "
                        "witnesses release after placement without a second action. "
                        "For every quote prefer the exact key from source_spans. Reuse one key for "
                        "different requirements in the same sentence. If copying a shorter quote, "
                        "copy a contiguous substring verbatim without expanding any shared wording. "
                        "Return {issues:[{quote:source_spans key or verbatim span,reason:specific actual mismatch}]} for any omission, "
                        "contradiction, unrelated extra action or unjustified claim. Otherwise return "
                        "{issues:[],checks:[{quote:source_spans key or verbatim span,requirement:concrete source meaning,"
                        "scope:action|initial|always|before|step|effect|final,evidence:[step_NN, condition_NN, or a role ID],"
                        "state:optional typed fact,spatial:optional typed relation,"
                        "parallel_picks:optional paired subject map}]}, one check per independent source obligation."
                    ),
                },
                canonical_hash(
                    {
                        "roles": sorted(plan["roles"], key=lambda role: role["id"]),
                        "steps": [without_quotes(row) for row in plan["steps"]],
                        "conditions": sorted(
                            map(without_quotes, plan["conditions"]), key=canonical_hash
                        ),
                    }
                ),
                lambda audit: _validate_intent_audit(instruction, plan, audit),
                visual=True,
            )

        used = 0
        intent = None
        goal_producers = {}
        saw_intent = False
        operations = []
        for attempt in range(4):
            interpreted = False
            try:
                used += 1
                proposal = _json_response(
                    VisionFacade(run, "task_image_grounding").complete(
                        image_path=image,
                        system_prompt=system_prompt,
                        user_prompt=json.dumps(prompt, ensure_ascii=False),
                    )
                )
                if (
                    allow_intent
                    and isinstance(proposal, dict)
                    and "semantic_plan" in proposal
                ):
                    saw_intent = True
                    prompt["previous_semantic_plan"] = proposal["semantic_plan"]
                    if set(proposal) != {"semantic_plan"}:
                        raise ValueError(
                            "Return a semantic_plan without competing native operations"
                        )
                    proposed_intent = validate_intent(
                        proposal["semantic_plan"],
                        instruction,
                        scene_objects=scene_objects,
                    )
                    interpreted = True
                    prompt["previous_semantic_plan"] = proposed_intent
                    goal_producers, added_conditions = audit_intent(proposed_intent)
                    if added_conditions:
                        proposed_intent = validate_intent(
                            dict(
                                proposed_intent,
                                conditions=[
                                    *proposed_intent["conditions"],
                                    *added_conditions,
                                ],
                            ),
                            instruction,
                            scene_objects=scene_objects,
                        )
                    spec, graph, intent_clauses, descriptions = compile_intent(
                        proposed_intent, instruction, task_id, scene_objects
                    )
                    intent = proposed_intent
                    break
                operations = _operation_rows(proposal, instruction)
                bindings = {}
                obj, clauses, calls = _supported_program(
                    operations,
                    bindings=bindings,
                    scene_objects=scene_objects,
                )
                used += 1
                audit_calls(operations, calls, bindings)
                break
            except (KeyError, ValueError, TypeError) as error:
                if (
                    isinstance(error, ArticulationDeclarationError)
                    and not error.retryable
                ):
                    raise
                if attempt == (3 if saw_intent else 1):
                    raise
                prompt["diagnostic"] = str(error)
                if saw_intent:
                    prompt["repair_request"] = (
                        "Return the full corrected semantic_plan. Fix only the findings that "
                        "still apply; keep the other facts, identities, order and literal quotes. "
                        "Remove unsupported facts without replacing them with new constraints."
                    )
                    prompt["source_audit_findings"] = [
                        audit["issues"]
                        for audit in audit_history
                        if isinstance(audit, dict) and audit.get("issues")
                    ]
                    if (attempt == 2 and not interpreted) or str(error).startswith(
                        "Unchanged rejected graph:"
                    ):
                        prompt.pop("previous_semantic_plan", None)
                        prompt["repair_request"] = (
                            "Build a fresh complete semantic_plan from the original instruction and image. "
                            "Use the diagnostics to avoid repeating the rejected interpretation; "
                            "do not copy a candidate that already failed."
                        )
                if operations:
                    prompt["previous_operations"] = operations
        history = []
        support_targets = {
            op["reference"]: op["reference"]
            for op in operations
            if intent is None and op["template"] == "E5"
        }
        if intent is not None:
            bindings = {}
            producers = {node["id"]: node["id"] for node in graph["nodes"]}
            if not graph["success"]["conditions"]["final"]:
                if "terminal_steps" in graph["success"]:
                    goal_producers["final_step_completed"] = graph["success"][
                        "terminal_steps"
                    ]
                else:
                    producers["final_step_completed"] = graph["success"][
                        "terminal_step"
                    ]
            coverage = dict(
                status="source_reviewed",
                producers=producers,
            )
            if goal_producers:
                coverage["goal_producers"] = goal_producers
            result = dict(
                task_spec=spec.to_dict(),
                semantic_task_graph=graph,
                semantic_intent=intent,
                deployment_preflight={},
            )
        for _ in range(2 if intent is None else 0):
            try:
                candidate = author_goal_plan(instruction, task_id, obj, calls)
                spec = TaskSpec.from_dict(candidate["task_spec"])
                candidate_graph = candidate["semantic_task_graph"]
                candidate_graph["planner_route"] = "hybrid"
                feedback = verify_witness(
                    spec,
                    candidate_graph,
                    support_targets=support_targets,
                    bindings=bindings,
                )
                score = len(feedback["uncovered"]) + len(feedback["resource_conflicts"])
                if not history or score < best:
                    best, result, graph, coverage = (
                        score,
                        candidate,
                        candidate_graph,
                        feedback,
                    )
            except (ValueError, TypeError, KeyError) as exc:
                if not history:
                    raise
                feedback = {"status": "invalid", "error": str(exc)}
            history.append(
                dict(calls=calls, coverage=feedback, spec_hash=spec.semantic_hash)
            )
            if feedback["status"] == "covered" or used + 2 > 5:
                break
            used += 2
            revised = structured_call(
                services=run,
                role="task_semantic_planning",
                schema={"type": "object"},
                prompt=json.dumps(
                    dict(
                        request="Return corrected canonical calls only as {calls:[...]}. Preserve valid occurrences and their bindings. Registered calls require kind:registered,call_id,arguments,resources. Never remove requested milestones.",
                        frozen=spec.to_dict(),
                        previous=calls,
                        feedback=feedback,
                    ),
                    ensure_ascii=False,
                ),
            )["calls"]
            try:
                audit_calls(operations, revised, bindings)
            except (ValueError, TypeError, KeyError) as error:
                history.append(
                    dict(
                        calls=revised,
                        coverage={"status": "source_rejected", "error": str(error)},
                        spec_hash=spec.semantic_hash,
                    )
                )
                break
            calls = revised
    if image.read_bytes() != image_bytes:
        raise ValueError("Source image changed during authoring")
    doc = freeze_task_spec(
        spec,
        image_sha256=hashlib.sha256(image_bytes).hexdigest(),
        image_ref="source_image" + image.suffix.lower(),
        role_descriptions=(
            descriptions
            if intent is not None
            else {
                r.name: scene_objects.get(r.name, {}).get("description", r.name)
                for r in spec.roles
            }
        ),
        interpretation=[],
        generalization=[],
        clauses=(
            intent_clauses
            if intent is not None
            else [
                dict(
                    id="source." + k,
                    target=k,
                    quote=q,
                    origin="explicit",
                    resolution="resolved",
                )
                for k, q in clauses
            ]
        ),
    )
    result.update(
        task_spec_v1=doc,
        coverage=coverage,
        planning_history=history,
        template_sources=operations if intent is None else [],
        pending_support_targets=support_targets,
        pending_operation_bindings=bindings,
        source_audits=audit_history,
    )
    if context is not None:
        result["scene_context"] = context
    result["deployment_preflight"].update(
        graph_digest=canonical_hash(graph),
        spec_digest=doc["spec_digest"],
        status=(
            "authoring_only"
            if intent is not None
            else (
                "requires_deployment"
                if coverage["status"] == "covered"
                else "uncovered"
            )
        ),
    )
    if deployment:
        _deployment_check(result, deployment)
    if scene_grounding is not None:
        if coverage["status"] != "covered":
            raise ValueError("Only a covered witness can enter scene qualification")
        packet = json.loads(scene_grounding.read_text())
        result.update(_scene_projection(doc, graph, packet), scene_grounding=packet)
    elif coverage["status"] == "covered" and _has_groundable_landings(graph):
        projection, packet = _ground_frozen_roles(
            doc, graph, image, services, scene_context=context
        )
        if image.read_bytes() != image_bytes:
            raise ValueError("Source image changed during grounding")
        result.update(projection, scene_grounding=packet)
    pending_roles = [
        obj["id"]
        for obj in result.get("image_grounding", {}).get("objects", [])
        if obj["provenance"] != "observed"
    ]
    stage = (
        ("pending_visual_binding" if pending_roles else "ready_for_qualification")
        if "scene_grounding" in result
        else "authoring_only"
    )
    with WorkflowRunArtifacts(output) as artifacts:
        output = artifacts.output_dir
        (output / doc["inputs"]["image"]["ref"]).write_bytes(image_bytes)
        for k, v in result.items():
            write_run_json(output / (k + ".json"), v)
        write_run_json(
            output / "run_manifest.json",
            dict(
                schema_version="goal_authoring/v1",
                route="hybrid",
                stage=stage,
                pending_roles=pending_roles,
                spec_digest=doc["spec_digest"],
                graph_digest=canonical_hash(graph),
                execution_ready=False,
            ),
        )
        from ._review import _hybrid_preview

        (output / "semantic_task_graph.html").write_text(
            _hybrid_preview(result, output / doc["inputs"]["image"]["ref"], None),
            encoding="utf-8",
        )
        if "scene_grounding" not in result:
            from ..task_spec.publication import project_scene_requirements

            write_run_json(
                output / "scene_requirements.json", project_scene_requirements(doc)
            )
        write_run_json(
            output / "task_plan_manifest.json",
            dict(
                schema_version=(
                    "embodichain.hybrid-task-plan/v1"
                    if "scene_grounding" in result
                    else "embodichain.hybrid-authoring/v1"
                ),
                task_id=task_id,
                route="hybrid",
                stage=stage,
                pending_roles=pending_roles,
                execution_ready=False,
                instruction=doc["inputs"]["instruction"],
                image_sha256=doc["inputs"]["image"]["sha256"],
                spec_digest=doc["spec_digest"],
                semantic_graph_sha256=canonical_hash(graph),
                artifacts={
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in output.iterdir()
                    if p.is_file()
                },
            ),
        )
        artifacts.commit()
    return result
