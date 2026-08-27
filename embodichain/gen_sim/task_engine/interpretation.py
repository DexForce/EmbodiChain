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

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import json
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

_RELATIONS = RELATIONS
_ARMS = frozenset({"none", "auto", "left_arm", "right_arm"})
_ORIENTATIONS = frozenset({"none", "preserve", "upright"})
_TARGET_STATES = frozenset({"none", "open", "closed", "activated"})
_LAYOUTS = frozenset({"none", "line"})
_AXES = frozenset({"none", "world_x", "world_y"})
_DIRECTIONS = TRANSPORT_DIRECTIONS
_TERMINAL_BEHAVIORS = TERMINAL_BEHAVIORS
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
        "atomicaction",
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
# MiMo's OpenAI-compatible endpoint can spend the whole completion budget in
# hidden reasoning when the request leaves thinking enabled.  A sparse final
# JSON object then looks like a schema failure to the deterministic verifier.
# Keep the budget bounded and turn reasoning off for the text interpretation
# call; the parser must return an auditable object rather than a thought trace.
_MIMO_MAX_COMPLETION_TOKENS = 4096
_GEN_SIM_DIR = Path(__file__).resolve().parents[1]
_GEN_SIM_ENV_PATH = _GEN_SIM_DIR / ".env"
_GEN_CONFIG_PATH = _GEN_SIM_DIR / "simready_pipeline" / "configs" / "gen_config.json"


class _MissingRequiredTargetError(ValueError):
    """Identify a validation failure that receives targeted repair guidance."""


class _MissingRequiredObjectError(ValueError):
    """Identify a missing manipulated-object selector for targeted repair."""


@dataclass(frozen=True)
class InstructionDraftResult:
    """One validated, scene-independent interpretation and its audit metadata."""

    intent: InstructionIntent
    model: str
    attempts: int
    latency_seconds: float
    normalizations: tuple[dict[str, Any], ...]


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


def interpret_instruction_draft(
    instruction: str,
    *,
    model: str | None = None,
    caller: InstructionCaller | None = None,
) -> InstructionDraftResult:
    """Interpret one instruction without reading or grounding a scene."""
    instruction_text = str(instruction).strip()
    if not instruction_text:
        raise ValueError("instruction must be non-empty.")
    prompt = _instruction_prompt(instruction_text)
    invoke = caller or _default_instruction_caller
    # An injected caller owns its transport and does not need provider config.
    selected_model = model if caller is not None else _instruction_model(model)
    if caller is None and selected_model is None:
        raise ValueError(
            "A text LLM model is required through --llm-model, "
            "ACTION_ENGINE_LLM_MODEL, or OPENAI_MODEL."
        )
    started = perf_counter()
    first_error: Exception | None = None
    for attempt in range(2):
        current_prompt = prompt
        if first_error is not None:
            current_prompt += (
                "\n\nREPAIR OVERRIDE: the previous JSON was invalid. Return a corrected "
                "JSON object only; do not repeat the sparse response. Every step "
                "must contain all 16 step keys and every selector all 5 selector "
                "keys. Keep semantic fields explicit: E4 requires transfer_arm "
                "and receive_arm, and E1/E3 require target plus relation (unless "
                "E1 layout=line). Use canonical defaults only for fields that do "
                "not apply. Validation error: "
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
                prompt=current_prompt,
                schema=deepcopy(INSTRUCTION_INTENT_SCHEMA),
                model=selected_model,
            )
            normalized, normalizations = _normalize_instruction_intent_fields(
                _coerce_instruction_response(response)
            )
            intent = validate_instruction_intent(normalized)
            return InstructionDraftResult(
                intent=intent,
                model=selected_model or "injected_caller",
                attempts=attempt + 1,
                latency_seconds=perf_counter() - started,
                normalizations=tuple(normalizations),
            )
        except (TypeError, ValueError) as error:
            if attempt:
                raise ValueError(
                    "Instruction intent failed validation after one repair: " f"{error}"
                ) from error
            first_error = error
    raise AssertionError("unreachable")


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
        if not isinstance(raw_step, dict) or set(raw_step) != _STEP_KEYS:
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
        if set(raw) != _STEP_KEYS:
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
    _validate_dag(dependencies)
    return {"steps": steps}


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
    if task_type == "E2" and orientation_goal != "upright":
        raise ValueError(f"{context} E2 orientation_goal must be upright.")
    if task_type not in {"E1", "E2", "E4"} and orientation_goal != "none":
        raise ValueError(
            f"{context} orientation_goal is only valid for E1, E2, and E4."
        )
    if task_type == "E1" and step["layout"] == "line":
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
        if step["direction"] != "none":
            raise ValueError(f"{context} direction is only valid for E5.")
        if step["terminal_behavior"] != "none":
            raise ValueError(f"{context} terminal_behavior is only valid for E5.")
    if task_type == "E4":
        transfer = str(step["transfer_arm"])
        receive = str(step["receive_arm"])
        if transfer not in {"left_arm", "right_arm"} or receive not in {
            "left_arm",
            "right_arm",
        }:
            raise ValueError(f"{context} E4 requires two explicit arms.")
        if transfer == receive:
            raise ValueError(f"{context} E4 transfer and receive arms must differ.")
        if step["required_arm"] not in {"none", "auto"}:
            raise ValueError(
                f"{context} E4 uses transfer_arm/receive_arm, not required_arm."
            )
    if task_type == "E5" and step["required_arm"] not in {"none", "auto"}:
        raise ValueError(f"{context} E5 always uses both arms, not required_arm.")
    if task_type == "E6" and step["target_state"] != "open":
        raise ValueError(f"{context} E6 target_state must be open.")
    if task_type == "E7" and step["target_state"] != "closed":
        raise ValueError(f"{context} E7 target_state must be closed.")
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
        "alone does not prove object identity. "
        "Object directions are robot-relative; arm names are robot body sides. "
        "Preserve each concrete object or target phrase from the instruction as "
        "an open scene_ref.reference. Do not classify it or emit a scene UID. "
        "Emit no AtomicAction, category label, affordance, coordinates, poses, "
        "paths, or reasoning. Encode explicit ordering with depends_on; same-action set "
        "members may remain independent. Use empty strings and 'none' for "
        "inapplicable required fields. A request to retract the transfer arm "
        "E4 owns the complete transfer. For a handover followed by placement in "
        "the same user intent, emit one E4 with target, relation, and "
        "terminal_behavior=place; do not emit a trailing E1. Use "
        "terminal_behavior=hold only when the receiver should keep holding the "
        "object. The exact output keys are steps -> id, "
        "task_type, object, target, relation, required_arm, transfer_arm, "
        "receive_arm, orientation_goal, target_state, target_setting, layout, "
        "axis, direction, terminal_behavior, depends_on; each selector has kind, "
        "step_id, reference, quantifier, count.\n\n"
        "Use orientation_goal=none unless the instruction explicitly requests "
        "upright orientation or preserving the original orientation. Spatial "
        "placement and handover alone do not imply preserve. "
        "Emptying, dumping, or pouring contents from one container into another "
        "is exactly one E3 step: object selects the source container, target "
        "selects the receiving container, and relation=above. Pickup and staging "
        "are internal to that E3 step. "
        "Opening or pulling out a drawer is E6 with object selecting that drawer "
        "and target_state=open. Closing or pushing in a drawer is E7 with object "
        "selecting that drawer and target_state=closed. "
        f"Instruction:\n{instruction}\n\n"
        f"E1-E9 catalog:\n{json.dumps(_intent_capability_catalog(), ensure_ascii=False, sort_keys=True)}\n\n"
        "Shape-only complete JSON example (do not copy its step count or values; "
        "copy every key, including keys whose value is none/empty/0):\n"
        f"{json.dumps(_instruction_shape_example(), ensure_ascii=False, sort_keys=True)}\n\n"
        "Selector kind rules (these are not extra output fields):\n"
        f"{_instruction_selector_rules()}\n\n"
        "For E5, use target+relation for moving an object relative to another "
        "object, or direction for a small robot-relative move. A dual-arm pick, "
        "lift, raise, or hold request without another target uses direction=up "
        "and terminal_behavior=hold. Use hold unless the instruction explicitly "
        "says to put/release the object. For pick "
        "and release at the original location, use direction=none and place. A dual-arm "
        "pick/move/transport request uses E5. Final checklist: every step "
        "has all 16 step keys; every object and target "
        "has all 5 selector keys. For an inapplicable field use the canonical "
        "default shown in the example, never omit the field. E4 must explicitly "
        "state transfer_arm, receive_arm, and terminal_behavior. E1/E3 must explicitly state target "
        "and relation (except E1 layout=line)."
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
    if "E4 transfer and receive arms must differ" in str(error):
        return (
            "\nSame-arm handover repair rule: transfer_arm and receive_arm must "
            "name different arms. Preserve the explicitly stated transfer arm. "
            "When a later clause clearly continues with the handed object using "
            "the other arm, use that arm as receive_arm. Resolve coreference from "
            "the instruction semantics; identical scene_ref text alone does not "
            "prove that two independently selected objects are the same.\n"
        )
    if isinstance(error, _MissingRequiredObjectError):
        return (
            "\nMissing-object repair rule: preserve the selected task_type and "
            "set object to a scene_ref that preserves the explicit manipulated "
            "object phrase from the instruction. For E6/E7 the drawer, door, or "
            "other articulated part is the object selector; target remains none.\n"
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
        "\nMissing-target repair rule: for a non-line E1 placement, object is "
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
) -> Mapping[str, Any]:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    settings = _load_llm_settings(model=model)
    kwargs: dict[str, Any] = {
        "api_key": settings["api_key"],
        "model": settings["model"],
        "temperature": 0,
        "http_socket_options": (),
    }
    for key in ("base_url", "default_query"):
        if settings[key]:
            kwargs[key] = settings[key]
    if _is_mimo_compatible(settings):
        # MiMo documents ``thinking`` as a provider extension carried in the
        # OpenAI client's extra body.  Disabling it is important here: hidden
        # reasoning can consume the completion and leave only id/object/type.
        kwargs.update(
            {
                "max_completion_tokens": _MIMO_MAX_COMPLETION_TOKENS,
                "extra_body": {"thinking": {"type": "disabled"}},
            }
        )
    client = ChatOpenAI(**kwargs)
    # The full schema remains in the prompt and the local validator is still
    # authoritative even when the provider only offers JSON mode.
    structured = _structured_output_runnable(
        client,
        schema,
        settings=settings,
    )
    schema_prompt = (
        f"{prompt}\n\nReturn one JSON object conforming exactly to this JSON "
        f"Schema:\n{json.dumps(schema, ensure_ascii=False, sort_keys=True)}"
    )
    response = structured.invoke(
        [
            SystemMessage(
                content=(
                    "Return only the requested structured JSON response. Never "
                    "return reasoning, coordinates, or AtomicAction nodes."
                )
            ),
            HumanMessage(content=schema_prompt),
        ]
    )
    return _coerce_instruction_response(response)


def _instruction_model(explicit: str | None) -> str | None:
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    # Keep model selection separate from credential loading.  Reading the local
    # dotenv file is side-effect free and gives generation the documented
    # priority without leaking credentials into TaskSpec metadata.
    for name in ("TASK_ENGINE_LLM_MODEL", "ACTION_ENGINE_LLM_MODEL", "OPENAI_MODEL"):
        for source in (
            os.environ,
            _load_local_env(),
        ):
            value = source.get(name)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _load_local_env() -> dict[str, str]:
    """Read Task Engine model configuration without mutating the environment."""
    return _load_env_file(_GEN_SIM_ENV_PATH)


def _is_mimo_compatible(settings: Mapping[str, Any]) -> bool:
    model = str(settings.get("model", "")).casefold()
    base_url = str(settings.get("base_url", "")).casefold()
    return "mimo" in model or "xiaomimimo.com" in base_url


def _structured_output_runnable(
    client: Any,
    schema: Mapping[str, Any],
    *,
    settings: Mapping[str, Any],
) -> Any:
    if not hasattr(client, "with_structured_output"):
        return client
    method = "json_mode" if _is_mimo_compatible(settings) else "json_schema"
    try:
        return client.with_structured_output(schema, method=method)
    except (TypeError, ValueError):
        if method == "json_mode" and hasattr(client, "bind"):
            from langchain_core.output_parsers import JsonOutputParser

            return (
                client.bind(response_format={"type": "json_object"})
                | JsonOutputParser()
            )
        return client.with_structured_output(schema)


def _load_llm_settings(*, model: str | None) -> dict[str, Any]:
    local_env = _load_local_env()
    config: dict[str, Any] = {}
    if _GEN_CONFIG_PATH.is_file():
        raw = json.loads(_GEN_CONFIG_PATH.read_text(encoding="utf-8"))
        if isinstance(raw, Mapping):
            llm = raw.get("llm", {})
            if isinstance(llm, Mapping):
                configured = llm.get("openai_compatible", {})
                if isinstance(configured, Mapping):
                    config = dict(configured)
    api_key, base_url = _resolve_transport_settings(local_env, config)
    selected_model = (
        (model.strip() if isinstance(model, str) else "")
        or _first_env_value(
            local_env,
            "TASK_ENGINE_LLM_MODEL",
            "ACTION_ENGINE_LLM_MODEL",
            "OPENAI_MODEL",
            "LLM_MODEL",
        )
        or str(config.get("model", "")).strip()
    )
    default_query = config.get("default_query", {}) or {}
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is required for Task Engine interpretation. Set it "
            f"in the process environment or {_GEN_SIM_ENV_PATH}."
        )
    if not selected_model:
        raise ValueError(
            "A text LLM model is required through model=, TASK_ENGINE_LLM_MODEL, "
            f"OPENAI_MODEL, or {_GEN_CONFIG_PATH}."
        )
    if not isinstance(default_query, Mapping):
        raise ValueError("LLM default_query must be a mapping.")
    return {
        "api_key": api_key,
        "model": selected_model,
        "base_url": base_url,
        "default_query": dict(default_query),
    }


def _resolve_transport_settings(
    local_env: Mapping[str, str],
    config: Mapping[str, Any],
) -> tuple[str, str]:
    """Resolve an API key and endpoint from one configuration source."""
    transports = (
        (
            _mapping_value(os.environ, "OPENAI_API_KEY"),
            _mapping_value(
                os.environ,
                "OPENAI_BASE_URL",
                "OPENAI_API_BASE",
                "LLM_URL",
            ),
        ),
        (
            _mapping_value(local_env, "OPENAI_API_KEY"),
            _mapping_value(
                local_env,
                "OPENAI_BASE_URL",
                "OPENAI_API_BASE",
                "LLM_URL",
            ),
        ),
        (
            _mapping_value(config, "api_key"),
            _mapping_value(config, "base_url"),
        ),
    )
    for api_key, base_url in transports:
        if api_key and base_url:
            return api_key, base_url.rstrip("/")
    for api_key, base_url in transports:
        if api_key:
            return api_key, base_url.rstrip("/")
    return "", ""


def _mapping_value(source: Mapping[str, Any], *names: str) -> str:
    for name in names:
        value = source.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _load_env_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"Invalid dotenv key at {path}:{line_number}.")
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        values[key] = value
    return values


def _first_env_value(local_env: Mapping[str, str], *names: str) -> str | None:
    for source in (os.environ, local_env):
        for name in names:
            value = source.get(name)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


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


def _topological_steps(steps: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return a stable topological ordering for validated intent steps."""
    by_id = {str(step["id"]): dict(step) for step in steps}
    effective_dependencies: dict[str, tuple[str, ...]] = {}
    for step_id, step in by_id.items():
        deps = list(str(dep) for dep in step["depends_on"])
        for selector in (step["object"], step["target"]):
            if selector["kind"] == "step_result":
                reference = str(selector["step_id"])
                if reference not in deps:
                    deps.append(reference)
        effective_dependencies[step_id] = tuple(deps)
    pending = set(by_id)
    ordered: list[dict[str, Any]] = []
    original = [str(step["id"]) for step in steps]
    while pending:
        ready = [
            step_id
            for step_id in original
            if step_id in pending
            and all(str(dep) not in pending for dep in effective_dependencies[step_id])
        ]
        if not ready:
            raise ValueError("Instruction intent dependencies contain a cycle.")
        for step_id in ready:
            ordered.append(by_id[step_id])
            pending.remove(step_id)
    return ordered


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


def _validate_dag(dependencies: Mapping[str, Sequence[str]]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise ValueError("Instruction intent dependencies contain a cycle.")
        if node in visited:
            return
        visiting.add(node)
        for dependency in dependencies[node]:
            visit(str(dependency))
        visiting.remove(node)
        visited.add(node)

    for node in dependencies:
        visit(node)


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
