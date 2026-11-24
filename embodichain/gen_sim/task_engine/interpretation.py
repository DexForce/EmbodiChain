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

"""Scene-independent structured interpretation for Task Engine."""

from __future__ import annotations

from embodichain.gen_sim.services.runtime import GenSimServices, service_entry


from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
import json
import math
import os
from pathlib import Path
import re
from time import perf_counter
from typing import Any, TypeAlias

from .ontology import (
    RELATIONS,
    TASK_CONTRACTS,
    TERMINAL_BEHAVIORS,
    TRANSPORT_DIRECTIONS,
)

__all__ = [
    "INSTRUCTION_INTENT_SCHEMA",
    "InstructionDraftResult",
    "InstructionIntent",
    "InstructionCaller",
    "interpret_instruction_draft",
    "validate_instruction_intent",
]

InstructionCaller = Callable[..., Mapping[str, Any]]
InstructionIntent: TypeAlias = dict[str, Any]
TASK_TYPES = frozenset(TASK_CONTRACTS)

_TERMINAL_PARK_CALL = "simulation.park"
_TERMINAL_PARK_KINDS = {f"terminal_park_{arm}": arm for arm in ("left", "right")}
_RELATIONS = RELATIONS
_ARMS = frozenset({"none", "auto", "left_arm", "right_arm"})
_ORIENTATIONS = frozenset({"none", "preserve", "upright", "vertical"})
_TARGET_STATES = frozenset({"none", "open", "closed", "activated"})
_LAYOUTS = frozenset({"none", "line"})
_AXES = frozenset({"none", "world_x", "world_y"})
_DIRECTIONS = TRANSPORT_DIRECTIONS
_TERMINAL_BEHAVIORS = TERMINAL_BEHAVIORS
# Single-arm placements whose released terminal may instead be held.
_HOLDABLE_PLACEMENTS = frozenset({"E1", "E2"})
_SELECTOR_KINDS = frozenset({"none", "scene_ref", "step_result"})
_QUANTIFIERS = frozenset({"one", "all", "count"})
_STEP_KEYS = frozenset(
    {
        "id",
        "task_type",
        "object",
        "target",
        "relation",
        "required_arm",
        "transfer_arm",
        "receive_arm",
        "orientation_goal",
        "target_state",
        "target_setting",
        "layout",
        "axis",
        "direction",
        "terminal_behavior",
        "depends_on",
    }
)
_OPTIONAL_STEP_KEYS = frozenset({"lift_height_m", "pour_goal_mode", "twist_goal_mode"})
_INTENT_TASK_FIELD_REGISTRY = {
    task_type: contract.applicable_intent_fields
    for task_type, contract in TASK_CONTRACTS.items()
}
_INTENT_FIELD_DEFAULTS: dict[str, Any] = {
    "target": None,
    "relation": "none",
    "required_arm": "none",
    "transfer_arm": "none",
    "receive_arm": "none",
    "orientation_goal": "none",
    "target_state": "none",
    "target_setting": 0,
    "layout": "none",
    "axis": "none",
    "direction": "none",
    "terminal_behavior": "none",
}
_SELECTOR_KEYS = frozenset(
    {
        "kind",
        "step_id",
        "reference",
        "quantifier",
        "count",
    }
)
_FORBIDDEN_FIELDS = frozenset(
    {
        "atomic_action",
        "atomic_actions",
        "atomic_skill",
        "atomic_skills",
        "atomicaction",
        "atomicskill",
        "coordinates",
        "bbox",
        "bboxes",
        "grasp_pose",
        "keypoint",
        "keypoints",
        "joint_positions",
        "joints",
        "pose",
        "position",
        "qpos",
        "rotation",
        "target_pose",
        "translation",
        "trajectory",
        "waypoints",
    }
)


class _MissingRequiredTargetError(ValueError):
    """Identify a validation failure that receives targeted repair guidance."""


class _MissingRequiredObjectError(ValueError):
    """Identify a missing manipulated-object selector for targeted repair."""


class _InstructionResourceError(ValueError):
    """Identify explicit held-resource conflicts before scene grounding."""


@dataclass(frozen=True)
class InstructionDraftResult:
    """One validated, scene-independent interpretation and its audit metadata."""

    intent: InstructionIntent
    model: str
    attempts: int
    latency_seconds: float
    normalizations: tuple[dict[str, Any], ...]
    terminal_park: tuple[dict[str, str], ...] = ()


@dataclass(frozen=True)
class _SourceIntent:
    intent: InstructionIntent
    normalizations: list[dict[str, Any]]
    scene_ref_quotes: dict[str, str]
    terminal_park: dict[str, str]


# Object semantics remain open natural-language references until the dedicated
# scene-grounding phase resolves them. All other values are strict protocol
# enums; non-canonical model output is repaired by the model, never guessed by
# a local language alias table.

_SELECTOR_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": sorted(_SELECTOR_KEYS),
    "properties": {
        "kind": {"type": "string", "enum": sorted(_SELECTOR_KINDS)},
        "step_id": {"type": "string"},
        "reference": {"type": "string"},
        "quantifier": {"type": "string", "enum": sorted(_QUANTIFIERS)},
        "count": {"type": "integer", "minimum": 0},
    },
}

_INTENT_OUTPUT_SCHEMA = {
    "title": "ActionEngineInstructionIntent",
    "type": "object",
    "additionalProperties": False,
    "required": ["steps"],
    "properties": {
        "steps": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": sorted(_STEP_KEYS),
                "properties": {
                    "id": {"type": "string"},
                    "task_type": {"type": "string", "enum": sorted(TASK_TYPES)},
                    "object": _SELECTOR_SCHEMA,
                    "target": _SELECTOR_SCHEMA,
                    "relation": {"type": "string", "enum": sorted(_RELATIONS)},
                    "required_arm": {"type": "string", "enum": sorted(_ARMS)},
                    "transfer_arm": {"type": "string", "enum": sorted(_ARMS)},
                    "receive_arm": {"type": "string", "enum": sorted(_ARMS)},
                    "orientation_goal": {
                        "type": "string",
                        "enum": sorted(_ORIENTATIONS),
                    },
                    "target_state": {
                        "type": "string",
                        "enum": sorted(_TARGET_STATES),
                    },
                    "target_setting": {"type": "integer"},
                    "pour_goal_mode": {
                        "type": "string",
                        "enum": ["fluid", "motion"],
                        "description": "MANDATORY for every E3 step, in addition to the 16 common fields. Use motion for a requested pouring gesture without liquid-transfer evidence; fluid for actual liquid transfer. Omit on other E types.",
                    },
                    "twist_goal_mode": {"type": "string", "enum": ["coarse"]},
                    "lift_height_m": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "description": "Minimum vertical increase from the initial object pose in metres, only for targetless E1 lift-and-hold.",
                    },
                    "layout": {"type": "string", "enum": sorted(_LAYOUTS)},
                    "axis": {"type": "string", "enum": sorted(_AXES)},
                    "direction": {
                        "type": "string",
                        "enum": sorted(_DIRECTIONS),
                    },
                    "terminal_behavior": {
                        "type": "string",
                        "enum": sorted(_TERMINAL_BEHAVIORS),
                    },
                    "depends_on": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
            },
        }
    },
}

# Keep a read-only-by-convention public copy for callers that need to configure
# a structured client.  The schema is an input contract, not a persisted task
# graph; ``validate_instruction_intent`` remains the authoritative verifier.
INSTRUCTION_INTENT_SCHEMA = deepcopy(_INTENT_OUTPUT_SCHEMA)

_SOURCE_INTENT_AUDIT_TITLE = "ActionEngineInstructionSourceAudit"


def _quoted_rows_schema(key: str, *values: str) -> dict[str, Any]:
    return {
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": False,
            "required": [key, "quote"],
            "properties": {
                key: {"type": "string", **({"enum": list(values)} if values else {})},
                "quote": {"type": "string"},
            },
        },
    }


_MODEL_INTENT_SCHEMA = {
    **_INTENT_OUTPUT_SCHEMA,
    "properties": {
        **_INTENT_OUTPUT_SCHEMA["properties"],
        "scene_ref_quotes": _quoted_rows_schema("ref"),
    },
}
_UNSUPPORTED_INTENT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["issues"],
    "properties": {"issues": {**_quoted_rows_schema("reason"), "minItems": 1}},
}


@service_entry
def interpret_instruction_draft(
    instruction: str,
    *,
    model: str | None = None,
    caller: InstructionCaller | None = None,
    services: GenSimServices | None = None,
    _allow_terminal_park: bool = False,
) -> InstructionDraftResult:
    """Interpret and audit one instruction without reading or grounding a scene."""
    instruction_text = str(instruction).strip()
    if not instruction_text:
        raise ValueError("instruction must be non-empty.")
    prompt = _instruction_prompt(instruction_text)
    evidence_prompt = (
        "\n\nSOURCE EVIDENCE: alongside steps, quote the verbatim identifying "
        "mention for scene_ref selectors in scene_ref_quotes: [{ref, quote}]. "
        "ref is the one-based step position followed by .object or .target, "
        "for example 1.object. Quote only the identifying phrase, including "
        "any distinguishing qualifiers, not a whole sentence with multiple "
        "referents. Omit evidence when a unique mention cannot be identified."
    )
    schema = deepcopy(_MODEL_INTENT_SCHEMA)
    if _allow_terminal_park:
        schema["properties"]["terminal_park"] = _quoted_rows_schema(
            "arm", *_TERMINAL_PARK_KINDS.values(), "both"
        )
        evidence_prompt += (
            f" {_TERMINAL_PARK_CALL} is available in the graph. Preserve an "
            "explicit return of named arms to their starting posture after all "
            "task actions in terminal_park, not as an E step. Emit "
            "{arm:left|right|both, quote:verbatim source phrase}; use both for "
            "a single request to return both arms. Do not defer any other "
            "requirement or assume an unspecified arm."
        )
    from functools import partial

    invoke = caller or partial(_default_instruction_caller, services=services)
    # An injected caller owns both its transport and model selection.
    selected_model = (
        model if caller is not None else services.model("task_interpretation", model)
    )
    started = perf_counter()
    first_error: Exception | None = None
    for attempt in range(2):
        current_prompt = prompt
        if first_error is not None:
            current_prompt += (
                "\n\nREPAIR OVERRIDE: the previous result failed validation. Return a corrected "
                "JSON object only; do not repeat the sparse response. Every step "
                "must contain all 16 step keys and every selector all 5 selector "
                "keys. Keep semantic fields explicit: E4 requires transfer_arm "
                "and receive_arm. E1/E3 require a target except E1 line arrangement "
                "and E1 metric lift-and-hold. A return-to-origin cleanup belonging to a pour stays within that E3 step; do not add a targetless E1 for it or invent an original-position scene object. Keep all other original steps. Use canonical defaults only for fields that do "
                "not apply. E3 MUST include the additional key pour_goal_mode with value motion or fluid; the 16 common fields alone are NOT sufficient for E3. Validation error: "
                f"{first_error}\n"
                "Copy this complete shape before filling values (shape only; do "
                "not copy its values or step count):\n"
                f"{json.dumps(_instruction_shape_example(), ensure_ascii=False, sort_keys=True)}\n"
                "Selector kind rules:\n"
                f"{_instruction_selector_rules()}"
                f"{_instruction_repair_guidance(first_error)}"
            )
        try:
            response = invoke(
                prompt=current_prompt + evidence_prompt,
                schema=deepcopy(schema),
                model=selected_model,
            )
            draft = _validate_model_intent(
                response, instruction_text, allow_terminal_park=_allow_terminal_park
            )
            terminal_park = _review_source_intent(
                instruction_text,
                prompt + evidence_prompt,
                draft,
                invoke,
                selected_model,
                schema=schema,
                allow_terminal_park=_allow_terminal_park,
            )
            return InstructionDraftResult(
                intent=draft.intent,
                model=selected_model or "injected_caller",
                attempts=attempt + 1,
                latency_seconds=perf_counter() - started,
                normalizations=tuple(draft.normalizations),
                terminal_park=terminal_park,
            )
        except (TypeError, ValueError) as error:
            if attempt:
                raise ValueError(
                    "Instruction intent failed validation after one repair: " f"{error}"
                ) from error
            first_error = error
    raise AssertionError("unreachable")


def _validate_model_intent(
    response: Any, instruction: str, *, allow_terminal_park: bool
) -> _SourceIntent:
    body = dict(_coerce_instruction_response(response))
    references = _audit_quoted_rows(
        instruction, body.pop("scene_ref_quotes", []), "ref"
    )
    parking = _audit_quoted_rows(instruction, body.pop("terminal_park", []), "arm")
    normalized, normalizations = _normalize_instruction_intent_fields(body)
    intent = validate_instruction_intent(normalized)
    _validate_explicit_lift_height(instruction, intent)
    if any(
        step["task_type"] == "E3" and "pour_goal_mode" not in step
        for step in intent["steps"]
    ):
        raise ValueError(
            "New E3 instructions require explicit pour_goal_mode: "
            "motion for an action-only request, fluid for liquid transfer."
        )
    slots = {
        f"{index}.{key}"
        for index, step in enumerate(intent["steps"], start=1)
        for key in ("object", "target")
        if step[key]["kind"] == "scene_ref"
    }
    scene_ref_quotes = dict(references)
    if len(scene_ref_quotes) != len(references) or scene_ref_quotes.keys() - slots:
        raise ValueError("scene_ref_quotes must identify unique scene_ref slots.")

    if parking and not allow_terminal_park:
        raise ValueError("terminal_park requires graph planning.")
    terminal_park: dict[str, str] = {}
    for requested_arm, quote in parking:
        requested = ("left", "right") if requested_arm == "both" else (requested_arm,)
        for arm in requested:
            if arm not in {"left", "right"} or arm in terminal_park:
                raise ValueError(
                    "terminal_park requires a unique declaration per named left/right arm."
                )
            terminal_park[arm] = quote
    return _SourceIntent(intent, normalizations, scene_ref_quotes, terminal_park)


_LIFT_MEASUREMENT = re.compile(
    r"\b(?:lift|raise)\b(?:(?!\b(?:and|or|then)\b)[^.!?;,\n])*?"
    r"\s+(?:\bby\s+(?:(?:more than|at least|over|exactly)\s+)?|"
    r"\b(?:more than|at least|over|exactly)\s+)"
    r"([0-9]+(?:\.[0-9]+)?)\s*(mm|cm|m|millimet(?:er|re)s?|"
    r"centimet(?:er|re)s?|met(?:er|re)s?)\b",
    re.IGNORECASE,
)


def _validate_explicit_lift_height(instruction: str, intent: InstructionIntent) -> None:
    """Independently check a uniquely attributable metric lift in the source."""
    lifts = [
        step
        for step in intent["steps"]
        if step["task_type"] == "E1"
        and step["direction"] == "up"
        and step["terminal_behavior"] == "hold"
        and step["target"]["kind"] == "none"
    ]
    if len(lifts) != 1:
        return
    measurements = []
    for match in _LIFT_MEASUREMENT.finditer(instruction):
        prefix = re.split(r"[.!?;,\n]", instruction[: match.start()])[-1]
        if re.search(
            r"\b(?:not|never|without)\b|n['’]t\b", prefix + match[0], re.IGNORECASE
        ):
            continue
        measurements.append(match.groups())
    if len(measurements) != 1:
        return
    amount, units = measurements[0]
    unit = units.lower()
    scale = 1.0
    if unit == "mm" or unit.startswith("milli"):
        scale = 0.001
    elif unit == "cm" or unit.startswith("centi"):
        scale = 0.01
    height = float(amount) * scale
    if "lift_height_m" not in lifts[0] or not math.isclose(
        lifts[0]["lift_height_m"], height
    ):
        raise ValueError(
            f"lift_height_m must match the explicit source height of {height:g} m."
        )


def _review_source_intent(
    instruction: str,
    prompt: str,
    draft: _SourceIntent,
    invoke: InstructionCaller,
    model: str | None,
    *,
    schema: dict[str, Any],
    allow_terminal_park: bool,
) -> tuple[dict[str, str], ...]:
    """Compare an independent source reconstruction with the proposed typed intent."""
    raw = invoke(
        prompt=(
            f"{prompt}\n\nSOURCE AUDIT: Independently reconstruct the complete "
            "source intent, in source order, as a steps object. Preserve explicit "
            "ordering, actor, destinations, result identity, relations, measurements "
            "and final hold or release. Do not turn a negated action into a step, "
            "or drop an unclear or unsupported requirement. For anything that "
            "cannot be represented faithfully, return only issues with a "
            "verbatim quote and reason."
        ),
        schema={
            "title": _SOURCE_INTENT_AUDIT_TITLE,
            "type": "object",
            "oneOf": [deepcopy(schema), deepcopy(_UNSUPPORTED_INTENT_SCHEMA)],
        },
        model=model,
    )
    response = _coerce_instruction_response(raw)
    if set(response) == {"issues"}:
        issues = _audit_quoted_rows(instruction, response["issues"], "reason")
        if not issues:
            raise ValueError("Source audit issues must be a non-empty list.")
        reasons = [f"{quote!r}: {reason}" for reason, quote in issues]
        raise ValueError(f"Source audit found unsupported intent: {'; '.join(reasons)}")

    reviewed = _validate_model_intent(
        response, instruction, allow_terminal_park=allow_terminal_park
    )
    if draft.terminal_park.keys() != reviewed.terminal_park.keys():
        raise ValueError(
            "Source audit disagrees on terminal_park arms: "
            f"draft={sorted(draft.terminal_park)}, source={sorted(reviewed.terminal_park)}."
        )

    proposed_steps = _canonicalize_intent(draft.intent)["steps"]
    reviewed_steps = _canonicalize_intent(reviewed.intent)["steps"]
    if len(proposed_steps) != len(reviewed_steps):
        source_types = [step["task_type"] for step in reviewed_steps]
        raise ValueError(
            f"Source audit expects {len(reviewed_steps)} steps in order: {source_types}."
        )
    proposed_order = _dependency_ancestors(proposed_steps)
    reviewed_order = _dependency_ancestors(reviewed_steps)
    for index, (proposed, source) in enumerate(
        zip(proposed_steps, reviewed_steps, strict=True), start=1
    ):
        for key in ("object", "target"):
            slot = f"{index}.{key}"
            draft_quote = draft.scene_ref_quotes.get(slot)
            reviewed_quote = reviewed.scene_ref_quotes.get(slot)
            if draft_quote is None or reviewed_quote is None:
                continue

            reference = draft_quote
            if draft_quote != reviewed_quote:
                # Overlap is not identity: both passes must declare the
                # same unique reference inside their respective excerpts.
                reference = proposed[key]["reference"]
                if (
                    reference != source[key]["reference"]
                    or reference not in draft_quote
                    or reference not in reviewed_quote
                    or not _is_unique_source_mention(instruction, reference)
                ):
                    raise ValueError(
                        f"Source audit disagrees at {source['id']} on {key} source mention: "
                        f"draft={draft_quote!r}, source={reviewed_quote!r}."
                    )
            elif not _is_unique_source_mention(instruction, reference):
                continue
            proposed[key]["reference"] = source[key]["reference"] = reference
            # Grounding uses the draft from this attempt, not the audit copies.
            selector = draft.intent["steps"][index - 1][key]
            if (previous := selector["reference"]) != reference:
                selector["reference"] = reference
                draft.normalizations.append(
                    {
                        "path": f"steps[{index - 1}].{key}.reference",
                        "from": previous,
                        "to": reference,
                        "reason": "verified_source_reference",
                    }
                )
        proposed["depends_on"] = sorted(proposed_order[proposed["id"]])
        source["depends_on"] = sorted(reviewed_order[source["id"]])
        if (
            "lift_height_m" in proposed
            and "lift_height_m" in source
            and math.isclose(proposed["lift_height_m"], source["lift_height_m"])
        ):
            proposed["lift_height_m"] = source["lift_height_m"]
        differences = {
            key: source.get(key)
            for key in sorted(proposed.keys() | source.keys())
            if proposed.get(key) != source.get(key)
        }
        if differences:
            raise ValueError(
                f"Source audit disagrees at {source['id']} (reviewed fields: {differences!r})."
            )
    return tuple(
        dict(arm=arm, quote=quote)
        for arm, quote in sorted(reviewed.terminal_park.items())
    )


def _is_unique_source_mention(instruction: str, mention: str) -> bool:
    start = instruction.find(mention)
    return start != -1 and instruction.find(mention, start + 1) == -1


def _dependency_ancestors(steps: Sequence[Mapping[str, Any]]) -> dict[str, set[str]]:
    """Compute reachability on a validated, canonical step DAG."""
    dependencies = {step["id"]: set(step["depends_on"]) for step in steps}
    ancestors: dict[str, set[str]] = {}
    for step_id in TopologicalSorter(dependencies).static_order():
        parents = dependencies[step_id]
        ancestors[step_id] = parents.union(*(ancestors[parent] for parent in parents))
    return ancestors


def _audit_quoted_rows(instruction: str, value: Any, key: str) -> list[tuple[str, str]]:
    if not isinstance(value, list):
        raise ValueError(f"Source audit {key} evidence must be a list.")
    rows = []
    for row in value:
        if not isinstance(row, Mapping) or set(row) != {key, "quote"}:
            raise ValueError(f"Source audit rows require exactly {key} and quote.")
        quote = row["quote"]
        if not isinstance(quote, str) or not quote.strip() or quote not in instruction:
            raise ValueError(
                "Source audit quote must occur verbatim in the instruction."
            )
        rows.append((_nonempty(row[key], f"Source audit {key}"), quote.strip()))
    return rows


def _normalize_instruction_intent_fields(
    value: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Canonicalize defaults and uniquely constrained cross-step continuity.

    The strict public validator deliberately remains unchanged.  This pass is
    confined to the LLM boundary, where weak JSON-mode providers sometimes
    copy a meaningful value into an inapplicable slot such as E4.required_arm.
    Required scene facts and ambiguous arm assignments are never inferred here
    and still fail closed.
    """
    result = deepcopy(dict(value))
    raw_steps = result.get("steps")
    if not isinstance(raw_steps, list):
        return result, []
    changes: list[dict[str, Any]] = []
    for index, raw_step in enumerate(raw_steps):
        if not isinstance(raw_step, dict) or not (
            _STEP_KEYS <= set(raw_step) <= _STEP_KEYS | _OPTIONAL_STEP_KEYS
        ):
            continue
        task_type = raw_step.get("task_type")
        applicable = _INTENT_TASK_FIELD_REGISTRY.get(task_type)
        if applicable is None:
            continue
        for field, configured_default in _INTENT_FIELD_DEFAULTS.items():
            field_applies = field in applicable
            if task_type == "E1" and field in {"target", "relation"}:
                field_applies = raw_step.get("layout") != "line"
            if task_type == "E1" and field == "axis":
                field_applies = raw_step.get("layout") == "line"
            if task_type == "E1" and field == "direction":
                target = raw_step.get("target")
                field_applies = (
                    raw_step.get("direction") == "up"
                    and raw_step.get("terminal_behavior") == "hold"
                    and isinstance(target, Mapping)
                    and target.get("kind") == "none"
                )
            if field_applies:
                continue
            default = (
                _empty_selector()
                if field == "target" and configured_default is None
                else deepcopy(configured_default)
            )
            if raw_step[field] == default:
                continue
            previous = deepcopy(raw_step[field])
            raw_step[field] = default
            changes.append(
                {
                    "path": f"steps[{index}].{field}",
                    "from": previous,
                    "to": deepcopy(default),
                    "reason": f"inapplicable_for_{task_type}",
                }
            )
        target = raw_step.get("target")
        if (
            task_type == "E5"
            and isinstance(target, Mapping)
            and target.get("kind") == "none"
            and raw_step.get("relation") == "none"
            and raw_step.get("direction") == "none"
            and raw_step.get("terminal_behavior") == "hold"
        ):
            raw_step["direction"] = "up"
            changes.append(
                {
                    "path": f"steps[{index}].direction",
                    "from": "none",
                    "to": "up",
                    "reason": "e5_hold_defaults_to_lift",
                }
            )
        if (
            task_type in _HOLDABLE_PLACEMENTS
            and raw_step.get("terminal_behavior") == "place"
        ):
            # Release is the default E1/E2 terminal; only hold is explicit.
            raw_step["terminal_behavior"] = "none"
            changes.append(
                {
                    "path": f"steps[{index}].terminal_behavior",
                    "from": "place",
                    "to": "none",
                    "reason": "release_is_the_default_terminal",
                }
            )
        if task_type == "E4" and raw_step.get("terminal_behavior") == "none":
            terminal = (
                "place"
                if isinstance(target, Mapping) and target.get("kind") != "none"
                else "hold"
            )
            raw_step["terminal_behavior"] = terminal
            changes.append(
                {
                    "path": f"steps[{index}].terminal_behavior",
                    "from": "none",
                    "to": terminal,
                    "reason": "e4_terminal_inferred_from_own_target",
                }
            )
    return result, changes


def _empty_selector() -> dict[str, Any]:
    """Return the canonical selector value for an inapplicable target."""
    return {
        "kind": "none",
        "step_id": "",
        "reference": "",
        "quantifier": "one",
        "count": 0,
    }


def validate_instruction_intent(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the private, non-graph instruction interpretation contract."""
    if not isinstance(value, Mapping):
        raise TypeError("Instruction intent must be a mapping.")
    _reject_forbidden_fields(value)
    if set(value) != {"steps"}:
        raise ValueError("Instruction intent may contain only 'steps'.")
    raw_steps = value.get("steps")
    if not isinstance(raw_steps, Sequence) or isinstance(raw_steps, (str, bytes)):
        raise ValueError("Instruction intent steps must be a list.")
    if not raw_steps:
        raise ValueError("Instruction intent steps must not be empty.")
    steps = []
    ids: set[str] = set()
    dependencies: dict[str, list[str]] = {}
    for index, raw in enumerate(raw_steps):
        context = f"InstructionIntent.steps[{index}]"
        if not isinstance(raw, Mapping):
            raise ValueError(f"{context} must be a mapping.")
        if not (_STEP_KEYS <= set(raw) <= _STEP_KEYS | _OPTIONAL_STEP_KEYS):
            raise ValueError(
                f"{context} requires exactly fields {sorted(_STEP_KEYS)}; "
                f"received {sorted(raw)}."
            )
        step = deepcopy(dict(raw))
        step_id = _nonempty(step["id"], f"{context}.id")
        if step_id in ids:
            raise ValueError(f"Duplicate instruction step ID {step_id!r}.")
        ids.add(step_id)
        step["id"] = step_id
        step["task_type"] = _choice(
            step["task_type"], TASK_TYPES, f"{context}.task_type"
        )
        if "pour_goal_mode" in step:
            if step["task_type"] != "E3":
                raise ValueError(f"{context}.pour_goal_mode is only valid for E3.")
            step["pour_goal_mode"] = _choice(
                step["pour_goal_mode"], {"fluid", "motion"}, f"{context}.pour_goal_mode"
            )
        if "twist_goal_mode" in step:
            if step["task_type"] != "E8":
                raise ValueError(f"{context}.twist_goal_mode is only valid for E8.")
            step["twist_goal_mode"] = _choice(
                step["twist_goal_mode"], {"coarse"}, f"{context}.twist_goal_mode"
            )
        step["object"] = _validate_selector(step["object"], f"{context}.object")
        step["target"] = _validate_selector(step["target"], f"{context}.target")
        step["relation"] = _canonical_relation(step["relation"], f"{context}.relation")
        for key in ("required_arm", "transfer_arm", "receive_arm"):
            step[key] = _canonical_arm(step[key], f"{context}.{key}")
        step["orientation_goal"] = _canonical_orientation(
            step["orientation_goal"], f"{context}.orientation_goal"
        )
        step["target_state"] = _choice(
            step["target_state"], _TARGET_STATES, f"{context}.target_state"
        )
        if isinstance(step["target_setting"], bool) or not isinstance(
            step["target_setting"], int
        ):
            raise ValueError(f"{context}.target_setting must be an integer.")
        step["layout"] = _choice(step["layout"], _LAYOUTS, f"{context}.layout")
        step["axis"] = _choice(step["axis"], _AXES, f"{context}.axis")
        step["direction"] = _choice(
            step["direction"], _DIRECTIONS, f"{context}.direction"
        )
        step["terminal_behavior"] = _choice(
            step["terminal_behavior"],
            _TERMINAL_BEHAVIORS,
            f"{context}.terminal_behavior",
        )
        raw_depends = step["depends_on"]
        if not isinstance(raw_depends, Sequence) or isinstance(
            raw_depends, (str, bytes)
        ):
            raise ValueError(f"{context}.depends_on must be a list.")
        step["depends_on"] = [
            _nonempty(item, f"{context}.depends_on") for item in raw_depends
        ]
        if step_id in step["depends_on"]:
            raise ValueError(f"{context}.depends_on cannot contain its own ID.")
        dependencies[step_id] = step["depends_on"]
        _validate_task_fields(step, context)
        steps.append(step)
    positions = {str(step["id"]): index for index, step in enumerate(steps)}
    for index, step in enumerate(steps):
        for selector_name in ("object", "target"):
            selector = step[selector_name]
            if selector["kind"] != "step_result":
                continue
            reference = str(selector["step_id"])
            if reference not in positions:
                raise ValueError(
                    f"Instruction step {step['id']!r} {selector_name} references "
                    f"unknown step {reference!r}."
                )
            if positions[reference] >= index:
                raise ValueError(
                    f"Instruction step {step['id']!r} {selector_name} must reference "
                    f"a preceding step, not {reference!r}."
                )
    for step_id, depends_on in dependencies.items():
        unknown = set(depends_on) - ids
        if unknown:
            raise ValueError(
                f"Instruction step {step_id!r} has unknown dependencies "
                f"{sorted(unknown)}."
            )
    try:
        TopologicalSorter(dependencies).prepare()
    except CycleError as exc:
        raise ValueError("Instruction intent dependencies contain a cycle.") from exc
    _validate_instruction_resources(steps)
    return {"steps": steps}


def _canonicalize_intent(intent: Mapping[str, Any]) -> dict[str, Any]:
    """Remove arbitrary model step IDs while preserving the explicit DAG order."""
    normalized = validate_instruction_intent(intent)
    steps = normalized["steps"]
    id_map = {step["id"]: f"step_{index + 1:02d}" for index, step in enumerate(steps)}
    for step in steps:
        step["id"] = id_map[step["id"]]
        step["depends_on"] = [id_map[value] for value in step["depends_on"]]
        for selector_name in ("object", "target"):
            selector = step[selector_name]
            if selector["kind"] == "step_result":
                selector["step_id"] = id_map[selector["step_id"]]
    return normalized


def _validate_instruction_resources(steps: Sequence[Mapping[str, Any]]) -> None:
    """Check occupancy and explicit hold continuation in compiler step order.

    Auto arms retain every possible allocation. Only ``step_result`` establishes
    subject identity; scene references may alias after grounding, so a release
    keeps both possibilities rather than asserting noun-based ownership.
    This is an early necessary check, not a grounded resource certificate.
    """
    arms = ("left_arm", "right_arm")
    states: set[tuple[str | None, str | None]] = {(None, None)}
    subjects: dict[str, str] = {}

    def choices(requested: str) -> tuple[int, ...]:
        return (arms.index(requested),) if requested in arms else (0, 1)

    for step in steps:
        step_id = str(step["id"])
        selector = step["object"]
        subject = (
            subjects[str(selector["step_id"])]
            if selector["kind"] == "step_result"
            else step_id
        )
        subjects[step_id] = subject
        contract = TASK_CONTRACTS[str(step["task_type"])]
        next_states: set[tuple[str | None, str | None]] = set()
        for state in states:
            known_owner = tuple(
                index for index, item in enumerate(state) if item == subject
            )
            if contract.resource_mode == "handover":
                for source in choices(str(step["transfer_arm"])):
                    for receiver in choices(str(step["receive_arm"])):
                        if source == receiver or state[receiver] is not None:
                            continue
                        if known_owner and known_owner != (source,):
                            continue
                        updated = list(state)
                        updated[source] = None
                        terminal = str(step["terminal_behavior"])
                        place = terminal == "place" or (
                            terminal == "none" and step["target"]["kind"] != "none"
                        )
                        updated[receiver] = None if place else subject
                        next_states.add(tuple(updated))
            elif contract.resource_mode == "coordinated":
                if any(state) and not (
                    selector["kind"] == "step_result" and state == (subject, subject)
                ):
                    continue
                next_states.add(
                    (None, None)
                    if step["terminal_behavior"] == "place"
                    else (subject, subject)
                )
            else:
                for resource in choices(str(step["required_arm"])):
                    if contract.accepts_incoming_hold:
                        if known_owner and known_owner != (resource,):
                            continue
                        updated = list(state)
                        updated[resource] = (
                            subject if step["terminal_behavior"] == "hold" else None
                        )
                        next_states.add(tuple(updated))
                    elif state[resource] is None:
                        next_states.add(state)
        if not next_states:
            if contract.resource_mode == "handover":
                reason = (
                    "HandOver receiver is occupied or the explicit source arm "
                    "does not own the held step_result"
                )
            elif contract.resource_mode == "coordinated":
                reason = (
                    "E5 requires both arms, but an arm is occupied; continuation "
                    "requires an explicit step_result jointly held by both arms"
                )
            else:
                reason = "the requested arm is occupied by a held object"
            raise _InstructionResourceError(f"Instruction step {step_id!r}: {reason}.")
        states = next_states


def _validate_selector(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping.")
    if set(value) != _SELECTOR_KEYS:
        raise ValueError(
            f"{context} requires exactly fields {sorted(_SELECTOR_KEYS)}; "
            f"received {sorted(value)}."
        )
    selector = deepcopy(dict(value))
    selector["kind"] = _choice(selector["kind"], _SELECTOR_KINDS, f"{context}.kind")
    selector["step_id"] = _selector_string(selector["step_id"], f"{context}.step_id")
    selector["reference"] = _selector_string(
        selector["reference"], f"{context}.reference"
    )
    selector["quantifier"] = _canonical_quantifier(
        selector["quantifier"], f"{context}.quantifier"
    )
    if isinstance(selector["count"], bool) or not isinstance(selector["count"], int):
        raise ValueError(f"{context}.count must be an integer.")
    if selector["count"] < 0:
        raise ValueError(f"{context}.count must be non-negative.")
    kind = selector["kind"]
    if kind == "scene_ref" and not selector["reference"]:
        raise ValueError(f"{context} scene_ref requires a reference.")
    if kind == "step_result":
        if not selector["step_id"]:
            raise ValueError(f"{context} step_result requires step_id.")
        if selector["reference"]:
            raise ValueError(
                f"{context} step_result may identify only a prior step_id."
            )
        if selector["quantifier"] != "one" or selector["count"] != 0:
            raise ValueError(
                f"{context} step_result requires quantifier=one and count=0."
            )
    if kind == "scene_ref" and selector["step_id"]:
        raise ValueError(f"{context} scene_ref cannot carry step_id.")
    if kind == "none" and (selector["step_id"] or selector["reference"]):
        raise ValueError(f"{context} kind=none cannot carry constraints.")
    if kind == "none" and (selector["quantifier"] != "one" or selector["count"] != 0):
        raise ValueError(f"{context} kind=none requires quantifier=one and count=0.")
    if selector["quantifier"] == "one" and selector["count"] != 0:
        raise ValueError(f"{context} quantifier=one requires count=0.")
    if selector["quantifier"] == "all" and selector["count"] != 0:
        raise ValueError(f"{context} quantifier=all requires count=0.")
    if selector["quantifier"] == "count" and selector["count"] < 1:
        raise ValueError(f"{context} quantifier=count requires count>=1.")
    return selector


def _validate_task_fields(step: Mapping[str, Any], context: str) -> None:
    task_type = str(step["task_type"])
    if step["object"]["kind"] == "none":
        raise _MissingRequiredObjectError(
            f"{context} {task_type} requires an object selector."
        )
    target_kind = str(step["target"]["kind"])
    if task_type not in {"E1", "E3", "E4", "E5"} and step["relation"] != "none":
        raise ValueError(f"{context} {task_type} does not accept relation.")
    if task_type == "E3" and step["relation"] != "above":
        raise ValueError(f"{context} E3 relation must be above.")
    target_setting = int(step["target_setting"])
    if task_type != "E8" and target_setting != 0:
        raise ValueError(f"{context} target_setting is only valid for E8.")
    if step.get("twist_goal_mode") == "coarse" and target_setting != 0:
        raise ValueError(
            f"{context} coarse twist uses target_setting=0 as a placeholder, not a numeric goal."
        )
    if task_type != "E1" and step["axis"] != "none":
        raise ValueError(f"{context} axis is only valid for E1 line arrangement.")
    if task_type == "E1" and step["layout"] != "line" and step["axis"] != "none":
        raise ValueError(f"{context} axis is only valid for E1 line arrangement.")
    if task_type not in {"E6", "E7", "E9"} and step["target_state"] != "none":
        raise ValueError(f"{context} target_state is not valid for {task_type}.")
    if task_type != "E4" and step["transfer_arm"] != "none":
        raise ValueError(f"{context} transfer_arm is only valid for E4.")
    if task_type != "E4" and step["receive_arm"] != "none":
        raise ValueError(f"{context} receive_arm is only valid for E4.")
    orientation_goal = str(step["orientation_goal"])
    if orientation_goal == "vertical" and task_type != "E1":
        raise ValueError(
            f"{context} undirected vertical orientation is currently supported for E1 only."
        )
    if task_type == "E2" and orientation_goal != "upright":
        raise ValueError(f"{context} E2 orientation_goal must be upright.")
    if task_type not in {"E1", "E2", "E4"} and orientation_goal != "none":
        raise ValueError(
            f"{context} orientation_goal is only valid for E1, E2, and E4."
        )
    metric_lift = task_type == "E1" and step["direction"] == "up"
    if "lift_height_m" in step and not metric_lift:
        raise ValueError(
            f"{context} lift_height_m is only valid for an upward E1 hold."
        )
    if metric_lift:
        height = step.get("lift_height_m")
        if (
            target_kind != "none"
            or step["relation"] != "none"
            or step["layout"] != "none"
            or step["terminal_behavior"] != "hold"
        ):
            raise ValueError(
                f"{context} an upward E1 lift requires a terminal hold without "
                "a target, relation or layout."
            )
        if type(height) not in {int, float} or not math.isfinite(height) or height <= 0:
            raise ValueError(f"{context} lift_height_m must be finite and positive.")
    elif task_type == "E1" and step["layout"] == "line":
        if target_kind != "none":
            raise ValueError(f"{context} E1 line arrangement cannot carry a target.")
        if step["relation"] != "none":
            raise ValueError(f"{context} E1 line arrangement cannot carry a relation.")
    elif task_type in {"E1", "E3"}:
        if target_kind == "none":
            raise _MissingRequiredTargetError(
                f"{context} {task_type} requires a target selector."
            )
        if step["relation"] == "none" and task_type == "E3":
            raise ValueError(f"{context} {task_type} requires a symbolic relation.")
    elif task_type == "E4":
        terminal = str(step["terminal_behavior"])
        effective_terminal = (
            "place" if terminal == "none" and target_kind != "none" else terminal
        )
        if effective_terminal == "none":
            effective_terminal = "hold"
        if effective_terminal not in _TERMINAL_BEHAVIORS - {"none"}:
            raise ValueError(f"{context} E4 requires terminal_behavior hold/place.")
        if effective_terminal == "place":
            if target_kind == "none" or step["relation"] == "none":
                raise ValueError(
                    f"{context} E4 terminal_behavior=place requires target and relation."
                )
        elif target_kind != "none" or step["relation"] != "none":
            raise ValueError(
                f"{context} E4 terminal_behavior=hold cannot carry target or relation."
            )
    elif task_type == "E5":
        direction = str(step["direction"])
        terminal = str(step["terminal_behavior"])
        if terminal not in _TERMINAL_BEHAVIORS - {"none"}:
            raise ValueError(f"{context} E5 requires terminal_behavior hold/place.")
        if target_kind == "none":
            if step["relation"] != "none":
                raise ValueError(f"{context} E5 relation requires a target selector.")
            if direction == "none" and terminal != "place":
                raise ValueError(
                    f"{context} E5 requires a direction or target relation."
                )
        else:
            if step["relation"] == "none":
                raise ValueError(f"{context} E5 target requires a relation.")
            if direction != "none":
                raise ValueError(
                    f"{context} E5 target relation cannot also carry direction."
                )
    elif target_kind != "none":
        raise ValueError(f"{context} {task_type} does not accept a target selector.")
    if task_type not in {"E4", "E5"}:
        if step["direction"] != "none" and not metric_lift:
            raise ValueError(
                f"{context} direction is only valid for transport or an upward E1 hold."
            )
        if step["terminal_behavior"] != "none" and not (
            task_type in _HOLDABLE_PLACEMENTS and step["terminal_behavior"] == "hold"
        ):
            raise ValueError(
                f"{context} terminal_behavior is only valid for E4/E5, or hold "
                "for E1/E2."
            )
    if task_type == "E4":
        transfer = str(step["transfer_arm"])
        receive = str(step["receive_arm"])
        if transfer not in {"left_arm", "right_arm", "auto"} or receive not in {
            "left_arm",
            "right_arm",
            "auto",
        }:
            raise ValueError(f"{context} E4 requires named arms or auto.")
        if transfer == receive and transfer != "auto":
            raise ValueError(f"{context} E4 transfer and receive arms must differ.")
        if step["required_arm"] not in {"none", "auto"}:
            raise ValueError(
                f"{context} E4 uses transfer_arm/receive_arm, not required_arm."
            )
    if task_type in {"E1", "E2"} and step["required_arm"] == "none":
        raise ValueError(
            f"{context} {task_type} requires required_arm=auto/left_arm/right_arm."
        )
    if task_type == "E5" and step["required_arm"] not in {"none", "auto"}:
        raise ValueError(f"{context} E5 always uses both arms, not required_arm.")
    if task_type == "E6" and step["target_state"] not in {"open", "closed"}:
        raise ValueError(f"{context} E6 target_state must be open or closed.")
    if task_type == "E7" and step["target_state"] != "open":
        raise ValueError(
            f"{context} E7 target_state must be open for a hinged door; "
            "regenerate legacy E7 closing intents from the original instruction."
        )
    if task_type == "E9" and step["target_state"] != "activated":
        raise ValueError(f"{context} E9 target_state must be activated.")
    if step["layout"] == "line" and task_type != "E1":
        raise ValueError(f"{context} only E1 supports layout=line.")


def _instruction_prompt(instruction: str) -> str:
    return (
        "Convert the user's explicit L1-L3 instruction into typed E1-E9 task "
        "intent. Understand synonyms, ellipsis, and pronouns, but "
        "do not invent missing objects. Use step_result for cross-step pronouns "
        "and explicit references to the result of an earlier manipulation. Keep "
        "an independently selected repeated noun as scene_ref; identical text "
        "alone does not prove object identity. A same-entity continuation remains "
        "step_result even when the later clause redescribes its changed state, "
        "such as 'the now-empty tray'. E6 opening and later closing the same part "
        "use the opening step's result; intermediate target references to that "
        "same part also use its step_result. Do not merge independently selected native parts "
        "or distinct roles merely because they belong to one cabinet or appliance. "
        "For placement, preserve containment: placing into or inside a container "
        "requires relation=inside, not on. A location phrase describing that "
        "container, such as 'the bowl left of the drawer', does not change the "
        "placement relation to left_of. A step_result identifies the manipulated object, not its destination. "
        "When another object goes into the same bowl or container, keep that "
        "destination as a scene_ref with the explicit repeated destination phrase; "
        "do not stack the second object on the first. Use a prior step_result "
        "as the target only when that manipulated object itself is the requested "
        "destination, not when the instruction repeats its former destination. "
        "Object directions are robot-relative; arm names are robot body sides. "
        "Preserve each concrete object or target phrase from the instruction as "
        "an open scene_ref.reference. Do not classify it or emit a scene UID. "
        "Do not collapse a coordinated noun into one object: 'knife and fork' "
        "or '刀叉' names two distinct objects. When both share one action and "
        "target, use quantifier=count with count=2 or emit separate steps; "
        "preserve every explicitly named member. "
        "Emit no AtomicSkill, category label, affordance, coordinates, poses, "
        "paths, or reasoning. Encode explicit ordering with depends_on; same-action set "
        "members may remain independent. Use empty strings and 'none' for "
        "inapplicable required fields. A request to retract the transfer arm "
        "E4 owns the complete transfer. For a handover followed by placement in "
        "the same user intent, emit one E4 with target, relation, and "
        "terminal_behavior=place; do not emit a trailing E1. Use "
        "terminal_behavior=hold only when the receiver should keep holding the "
        "object. E1 and E2 release by default (terminal_behavior=none); use "
        "terminal_behavior=hold on E1/E2 only when the instruction requires the "
        "object to stay in the gripper at the end, such as lifting it upright and "
        "holding it, or striking/touching a target with a held tool. For a single-arm "
        "lift by at least a specified height from the object's initial pose followed by a hold, "
        "use E1, direction=up, terminal_behavior=hold, target.kind=none, relation=none "
        "and lift_height_m in metres. Lifting vertically specifies translation, "
        "not the object's orientation. Other E1 holds retain their named target and relation. "
        "Each gripper can hold only one object: a held receiver cannot "
        "accept another handover, and an E5 pickup requires both hands free. "
        "For multiple objects that are handed over and then placed, use per-object "
        "transfer and placement before the next object's handover, unless the "
        "instruction explicitly requires a different order. An E5 continuation "
        "of the jointly held object uses step_result; it does not acquire a "
        "second object. The exact output keys are steps -> id, "
        "task_type, object, target, relation, required_arm, transfer_arm, "
        "receive_arm, orientation_goal, target_state, target_setting, layout, "
        "axis, direction, terminal_behavior, depends_on; each selector has kind, "
        "step_id, reference, quantifier, count.\n\n"
        "For each new E3 instruction, explicitly declare pour_goal_mode. Use "
        "'motion' for an explicitly requested pouring motion, tilt-and-restore "
        "gesture or demonstration (including 倾倒动作 / perform a pouring motion), "
        "even without an additional word meaning only. A container name does not "
        "establish a requested liquid transfer. Use 'fluid' when transfer of actual "
        "liquid, a named fluid or a volume is requested; that explicit transfer "
        "takes precedence over motion wording. Bare ambiguous pouring with no "
        "motion/evidence cue retains fluid mode. Never weaken fluid-transfer intent to "
        "motion merely because fluid simulation may be unavailable. Do not emit "
        "pour_goal_mode for other E types.\n\n"
        "For E8 only, when the user asks for a coarse turn without a printed-label "
        "target, set optional twist_goal_mode='coarse' and target_setting=0 as a "
        "placeholder. This does not request label 0: execution selects a source-calibrated "
        "planning reference, not a new task goal. Omit twist_goal_mode for an explicit "
        "printed-label target and retain that exact numeric target_setting. Do not "
        "weaken an explicit label request into a coarse turn.\n\n"
        "Use orientation_goal=none unless the instruction explicitly requests "
        "upright orientation or preserving the original orientation. Spatial "
        "placement and handover alone do not imply preserve. "
        "For E1 only, use orientation_goal=vertical when the user explicitly "
        "allows either axis direction (upside-down is acceptable), with no mouth-up "
        "or base-down requirement. This means an undirected vertical long axis and "
        "must not require flipping an already vertical object. Otherwise an explicit "
        "request to keep a vessel upright, vertical, or mouth-up requires "
        "orientation_goal=upright, including when the "
        "receiver keeps holding it (terminal_behavior=hold). Do not discard "
        "this orientation requirement because the core operation is handover. "
        "If placement is a separate E1 step after a transfer, the preceding E4 "
        "must use terminal_behavior=hold, target.kind=none and relation=none; "
        "the E1 step owns the placement target and relation. Alternatively, a "
        "single E4 with terminal_behavior=place must itself specify both target "
        "and relation. Never emit terminal_behavior=place with an empty E4 target. "
        "For relative placement, target.reference names the relation anchor, "
        "not the whole destination region or its support surface. For example, "
        "'on the tabletop beside the wooden block' selects 'wooden block' as "
        "target.reference, not 'table' or 'the tabletop beside the wooden block'. "
        "Use the table itself as target only for placement on the table without "
        "another named relation anchor. "
        "Emptying, dumping, or pouring contents from one container into another "
        "is exactly one E3 step: object selects the source container, target "
        "selects the receiving container, and relation=above. Pickup and staging "
        "are internal to that E3 step. "
        "Opening or pulling out a drawer or sliding tray is E6 with target_state=open. "
        "Closing or pushing in that prismatic part is also E6 with target_state=closed. "
        "Opening a hinged cabinet, appliance, or other revolute door is E7 with "
        "target_state=open. Closing a hinged door is not modeled by E6 or E7. "
        "For E8, target_setting is the explicitly requested printed numeric label, "
        "not a rotation angle or an inferred temperature-to-angle conversion. "
        "For labelled OFF/LOW/MED/HIGH controls use setting codes 0/1/2/3; "
        "execution still requires an asset-specific calibrated label map. "
        "Do not guess an unspecified knob setting. "
        f"Instruction:\n{instruction}\n\n"
        f"E1-E9 catalog:\n{json.dumps(_intent_capability_catalog(), ensure_ascii=False, sort_keys=True)}\n\n"
        "Shape-only complete JSON example (do not copy its step count or values; "
        "copy every key, including keys whose value is none/empty/0):\n"
        f"{json.dumps(_instruction_shape_example(), ensure_ascii=False, sort_keys=True)}\n\n"
        "Selector kind rules (these are not extra output fields):\n"
        f"{_instruction_selector_rules()}\n\n"
        "For E5, use target+relation for moving an object relative to another "
        "object. Returning the carried object to its original location uses "
        "target.kind=none, relation=none, direction=none, terminal_behavior=place; "
        "the object's own starting pose is not a separate scene reference. Use direction "
        "for a small robot-relative move. When a request "
        "combines lifting with a horizontal direction, preserve that horizontal "
        "direction: 'carry forward and keep raised' uses direction=front and "
        "terminal_behavior=hold. The hold recipe includes a raised terminal pose. "
        "Use direction=up only for a dual-arm pick, lift, raise, or hold request "
        "without an explicit horizontal direction or another target in E5, "
        "and terminal_behavior=hold. Use hold unless the instruction explicitly "
        "says to put/release the object. For pick "
        "and release at the original location, use direction=none and place. A dual-arm "
        "pick/move/transport request uses E5. Final checklist: every step "
        "has all 16 step keys; every object and target "
        "has all 5 selector keys. For an inapplicable field use the canonical "
        "default shown in the example, never omit the field. E4 must explicitly "
        "state transfer_arm, receive_arm, and terminal_behavior. For an E4 arm "
        "not named in the instruction, set that arm field to auto; do not "
        "invent a left/right side before scene binding. Keep explicit targets for E1/E3 "
        "except E1 line arrangement or metric lift-and-hold. For E1 and E2, copy an explicitly "
        "named left/right arm into required_arm; use required_arm=auto only when "
        "the instruction does not specify an arm, and never use none."
    )


def _instruction_shape_example() -> dict[str, Any]:
    """Return a compact field-complete example for providers with weak schemas."""
    selector = {
        "kind": "scene_ref",
        "step_id": "",
        "reference": "example object A",
        "quantifier": "one",
        "count": 0,
    }
    empty_selector = {
        "kind": "none",
        "step_id": "",
        "reference": "",
        "quantifier": "one",
        "count": 0,
    }
    return {
        "steps": [
            {
                "id": "step_1",
                "task_type": "E2",
                "object": selector,
                "target": empty_selector,
                "relation": "none",
                "required_arm": "auto",
                "transfer_arm": "none",
                "receive_arm": "none",
                "orientation_goal": "upright",
                "target_state": "none",
                "target_setting": 0,
                "layout": "none",
                "axis": "none",
                "direction": "none",
                "terminal_behavior": "none",
                "depends_on": [],
            }
        ]
    }


def _instruction_selector_rules() -> str:
    """Return the mutually exclusive selector encodings for model prompts."""
    step_result = {
        "kind": "step_result",
        "step_id": "step_1",
        "reference": "",
        "quantifier": "one",
        "count": 0,
    }
    return (
        "- kind=none: step_id and reference are empty strings; "
        "quantifier='one'; count=0.\n"
        "- kind=scene_ref: step_id is empty and reference preserves the concrete "
        "object phrase from the user's instruction. Repeated scene_ref text does "
        "not establish cross-step identity.\n"
        "- For an explicit number N of selected objects, use quantifier='count' "
        "and count=N, even when the instruction says 'all N'. For an unnumbered "
        "all-set request, use quantifier='all' and count=0. A singular request "
        "uses quantifier='one' and count=0. Do not infer N from the scene.\n"
        "- kind=step_result: use it only for a pronoun that means exactly one "
        "object, or an explicit continuation of the result of an earlier "
        "instruction step. Set step_id to that prior "
        "step ID and set reference='', quantifier='one', count=0. Do not copy "
        "the prior object's phrase into this selector. Replace step_1 in this "
        f"complete shape with the actual prior step ID: {json.dumps(step_result, sort_keys=True)}\n"
        "A step_result may identify only a prior step_id; it cannot carry any "
        "other object constraint."
    )


def _instruction_repair_guidance(error: Exception) -> str:
    """Add narrow semantic guidance for errors weak JSON-mode models repeat."""
    if isinstance(error, _InstructionResourceError):
        return (
            "\nHeld-resource repair rule: re-read the original instruction. "
            "Do not drop objects, arm requirements, targets, or ordering. When "
            "the instruction requests handover followed by placement for each "
            "object, encode per-object transfer and placement as one E4 with "
            "terminal_behavior=place and its own target/relation, or explicitly "
            "release the held step_result before the next transfer. Use "
            "step_result for a same-object continuation, never infer identity "
            "from repeated noun text. Do not add a release when the instruction "
            "requires holding, or reorder an explicitly ordered clause merely "
            "to avoid the conflict; an impossible request must remain invalid.\n"
        )
    if "quantifier=all requires count=0" in str(error):
        return (
            "\nQuantifier repair rule: re-read the original instruction. If it "
            "contains an explicit number N for the selected set, use "
            "quantifier=count with count=N. If it requests all without an "
            "explicit number, keep quantifier=all and set count=0. Never use "
            "quantifier=all with a nonzero count or invent a number from the "
            "previous model response.\n"
        )
    if "E7 target_state must be open" in str(error):
        return (
            "\nArticulation classification repair rule: re-read the original "
            "instruction. Closing a prismatic drawer or sliding tray is E6 with "
            "target_state=closed; E7 is only for opening a hinged door. "
            "Do not change the requested operation from closing to opening. "
            "Closing a hinged door remains unsupported, not an E6 fallback.\n"
        )
    if "E4 transfer and receive arms must differ" in str(error):
        return (
            "\nSame-arm handover repair rule: transfer_arm and receive_arm must "
            "name different arms. Preserve the explicitly stated transfer arm. "
            "When a later clause clearly continues with the handed object using "
            "the other arm, use that arm as receive_arm. Resolve coreference from "
            "the instruction semantics; identical scene_ref text alone does not "
            "prove that two independently selected objects are the same.\n"
        )
    if "E4 requires named arms or auto" in str(error):
        return (
            "\nHandover arm repair rule: preserve any explicitly named arm. "
            "For an arm not named by the instruction, use auto rather than "
            "guessing left or right; scene grounding will assign a feasible "
            "source arm and the opposite receiver.\n"
        )
    if isinstance(error, _MissingRequiredObjectError):
        return (
            "\nMissing-object repair rule: preserve the selected task_type and "
            "set object to a scene_ref that preserves the explicit manipulated "
            "object phrase from the instruction. For E6 the drawer or sliding tray, "
            "and for E7 the hinged door, is the object selector; target remains none.\n"
        )
    if not isinstance(error, _MissingRequiredTargetError):
        return ""
    if " E3 requires a target selector" in str(error):
        return (
            "\nMissing-target repair rule for E3: keep task_type=E3. object is "
            "the source container whose contents are poured, target is the "
            "receiving container, and relation must be above. An explicit grab "
            "is part of the same E3 task.\n"
        )
    return (
        "\nMissing-target repair rule: an explicit single-arm lift by a measured "
        "height relative to the object's initial pose and held at the end uses "
        "E1 with direction=up, lift_height_m in metres, terminal_behavior=hold, "
        "target.kind=none and relation=none. For a non-line E1 placement, object is "
        "the item being moved and target is the explicit reference object "
        "after the spatial relation in the original instruction. For example, "
        "in 'place it to the left of the striped pedestal', object is the earlier "
        "step_result for 'it', while target selects the striped pedestal; target "
        "must not use kind=none. Use target kind=step_result only when the "
        "reference object itself is exactly the result of a prior step.\n"
    )


def _intent_capability_catalog() -> dict[str, dict[str, Any]]:
    """Return the LLM's thin, import-safe E1-E9 capability view.

    Action Engine's online planning catalog also reports runtime availability
    and therefore imports simulator action classes. Text interpretation only
    needs symbolic E semantics and must remain testable before a simulator
    backend is installed.
    """
    return {
        task_type: {
            "semantics": contract.semantics,
            "applicable_fields": sorted(_INTENT_TASK_FIELD_REGISTRY[task_type]),
        }
        for task_type, contract in TASK_CONTRACTS.items()
    }


def _default_instruction_caller(
    *,
    prompt: str,
    schema: Mapping[str, Any],
    model: str | None,
    services: GenSimServices | None = None,
) -> Mapping[str, Any]:
    from embodichain.gen_sim.services.compat import structured_call

    return structured_call(prompt=prompt, schema=schema, model=model, services=services)


def _instruction_model(explicit: str | None) -> str | None:
    from embodichain.gen_sim.services.legacy import instruction_model

    return instruction_model(explicit)


def _choice(value: Any, allowed: set[str] | frozenset[str], context: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{context} must be one of {sorted(allowed)}.")
    return value


def _selector_string(value: Any, context: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a string.")
    return value.strip()


def _canonical_quantifier(value: Any, context: str) -> str:
    return _choice(value, _QUANTIFIERS, context)


def _canonical_arm(value: Any, context: str) -> str:
    return _choice(value, _ARMS, context)


def _canonical_relation(value: Any, context: str) -> str:
    return _choice(value, _RELATIONS, context)


def _canonical_orientation(value: Any, context: str) -> str:
    return _choice(value, _ORIENTATIONS, context)


def _nonempty(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string.")
    return value.strip()


def _coerce_instruction_response(response: Any) -> Mapping[str, Any]:
    """Coerce common structured-client response wrappers without accepting prose."""
    if isinstance(response, Mapping):
        return dict(response)
    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, Mapping):
            return dict(dumped)
    content = getattr(response, "content", response)
    if isinstance(content, Mapping):
        return dict(content)
    if isinstance(content, list):
        content = "\n".join(
            str(item.get("text", ""))
            for item in content
            if isinstance(item, Mapping) and item.get("type") == "text"
        )
    if not isinstance(content, str):
        raise ValueError(
            f"Instruction model output has unsupported type {type(content).__name__}."
        )
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Instruction model output is not valid JSON: {exc}") from exc
    if not isinstance(parsed, Mapping):
        raise ValueError("Instruction model output must decode to a JSON object.")
    return dict(parsed)


def _reject_forbidden_fields(value: Any) -> None:
    if isinstance(value, Mapping):
        forbidden = _FORBIDDEN_FIELDS & {str(key).strip().lower() for key in value}
        if forbidden:
            raise ValueError(
                f"Instruction intent contains forbidden fields {sorted(forbidden)}."
            )
        for item in value.values():
            _reject_forbidden_fields(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            _reject_forbidden_fields(item)
