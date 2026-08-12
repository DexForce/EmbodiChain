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

"""Structured language interpretation followed by deterministic scene grounding."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import json
import os
from time import perf_counter
from typing import Any, TypeAlias

from embodichain.gen_sim.action_engine.domain import TASK_TYPES

from .factory import _E_DEFINITIONS
from .planning import (
    _AFFORDANCES,
    GroundedTaskSpec,
    _CATEGORIES,
    _COLORS,
    _Entity,
    _SceneIndex,
    _TaskBuilder,
    plan_grounded_task_spec,
)

__all__ = [
    "INSTRUCTION_INTENT_SCHEMA",
    "InstructionIntent",
    "InstructionCaller",
    "interpret_and_ground_task_spec",
    "validate_instruction_intent",
]

InstructionCaller = Callable[..., Mapping[str, Any]]
InstructionIntent: TypeAlias = dict[str, Any]

_RELATIONS = frozenset(
    {"none", "on", "inside", "above", "left_of", "right_of", "front_of", "behind"}
)
_ARMS = frozenset({"none", "auto", "left_arm", "right_arm"})
_ORIENTATIONS = frozenset({"preserve", "upright"})
_TARGET_STATES = frozenset({"none", "open", "closed", "activated"})
_LAYOUTS = frozenset({"none", "line"})
_AXES = frozenset({"none", "world_x", "world_y"})
_SELECTOR_KINDS = frozenset({"none", "selector", "step_result"})
_SIDES = frozenset({"none", "left", "right", "leftmost", "rightmost"})
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
        "depends_on",
    }
)
_INTENT_TASK_FIELD_REGISTRY = {
    "E1": frozenset(
        {
            "target",
            "relation",
            "required_arm",
            "orientation_goal",
            "layout",
            "axis",
        }
    ),
    "E2": frozenset({"required_arm", "orientation_goal"}),
    "E3": frozenset({"target", "relation", "required_arm"}),
    "E4": frozenset({"transfer_arm", "receive_arm", "orientation_goal"}),
    "E5": frozenset(),
    "E6": frozenset({"required_arm", "target_state"}),
    "E7": frozenset({"required_arm", "target_state"}),
    "E8": frozenset({"required_arm", "target_setting"}),
    "E9": frozenset({"required_arm", "target_state"}),
}
_INTENT_FIELD_DEFAULTS: dict[str, Any] = {
    "target": None,
    "relation": "none",
    "required_arm": "none",
    "transfer_arm": "none",
    "receive_arm": "none",
    "orientation_goal": "preserve",
    "target_state": "none",
    "target_setting": 0,
    "layout": "none",
    "axis": "none",
}
_SELECTOR_KEYS = frozenset(
    {
        "kind",
        "step_id",
        "uid",
        "category",
        "color",
        "side",
        "quantifier",
        "count",
    }
)
_FORBIDDEN_FIELDS = frozenset(
    {
        "atomic_action",
        "atomic_actions",
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
_PROMPT_REDACTED_KEYS = _FORBIDDEN_FIELDS | frozenset(
    {
        "absolute_position",
        "bbox",
        "bounding_box",
        "camera_matrix",
        "center",
        "centroid",
        "coordinates",
        "depth",
        "extrinsics",
        "init_pos",
        "init_rot",
        "intrinsics",
        "location",
        "matrix",
        "orientation",
        "position_xyz",
        "position",
        "quaternion",
        "rotation",
        "scale",
        "transform",
        "translation",
        "world_x",
        "world_y",
        "world_z",
        "x",
        "y",
        "z",
    }
)

# MiMo's OpenAI-compatible endpoint can spend the whole completion budget in
# hidden reasoning when the request leaves thinking enabled.  A sparse final
# JSON object then looks like a schema failure to the deterministic verifier.
# Keep the budget bounded and turn reasoning off for the text interpretation
# call; the parser must return an auditable object rather than a thought trace.
_MIMO_MAX_COMPLETION_TOKENS = 4096


class _MissingRequiredTargetError(ValueError):
    """Identify the one validation failure eligible for local completion."""


# Structured callers are asked for canonical English values.  The verifier
# nevertheless accepts the small set of language aliases users commonly put
# in mock responses; this keeps normalization deterministic and never adds a
# model-defined extension field.
_COLOR_ALIASES = {
    alias.lower(): canonical
    for canonical, aliases in _COLORS.items()
    for alias in (*aliases, "橘色" if canonical == "orange" else "")
    if alias
}
_CATEGORY_ALIASES = {
    alias.lower(): canonical
    for canonical, aliases in _CATEGORIES.items()
    for alias in aliases
}
_CATEGORY_ALIASES.update(
    {
        "pourable_container": "pourable_container",
        "container": "pourable_container",
        "容器": "pourable_container",
    }
)
_SIDE_ALIASES = {
    "左": "left",
    "左边": "left",
    "左侧": "left",
    "左手边": "left",
    "左手侧": "left",
    "右": "right",
    "右边": "right",
    "右侧": "right",
    "右手边": "right",
    "右手侧": "right",
    "最左": "leftmost",
    "最左边": "leftmost",
    "最右": "rightmost",
    "最右边": "rightmost",
}
_QUANTIFIER_ALIASES = {
    "single": "one",
    "one": "one",
    "一个": "one",
    "一": "one",
    "all": "all",
    "全部": "all",
    "所有": "all",
    "都": "all",
    "count": "count",
    "指定数量": "count",
}
_ARM_ALIASES = {
    "左": "left_arm",
    "左手": "left_arm",
    "左臂": "left_arm",
    "left": "left_arm",
    "left hand": "left_arm",
    "left arm": "left_arm",
    "右": "right_arm",
    "右手": "right_arm",
    "右臂": "right_arm",
    "right": "right_arm",
    "right hand": "right_arm",
    "right arm": "right_arm",
    "自动": "auto",
    "默认": "auto",
    "automatic": "auto",
    "none": "none",
    "无": "none",
}
_RELATION_ALIASES = {
    "none": "none",
    "无": "none",
    "on": "on",
    "on top": "on",
    "on_top": "on",
    "on top of": "on",
    "上面": "on",
    "上方": "on",
    "inside": "inside",
    "in": "inside",
    "into": "inside",
    "里面": "inside",
    "内部": "inside",
    "above": "above",
    "上": "above",
    "left": "left_of",
    "left of": "left_of",
    "left_of": "left_of",
    "左边": "left_of",
    "左侧": "left_of",
    "左手边": "left_of",
    "左手侧": "left_of",
    "right": "right_of",
    "right of": "right_of",
    "right_of": "right_of",
    "右边": "right_of",
    "右侧": "right_of",
    "右手边": "right_of",
    "右手侧": "right_of",
    "front of": "front_of",
    "front_of": "front_of",
    "前面": "front_of",
    "前方": "front_of",
    "behind": "behind",
    "后面": "behind",
    "后方": "behind",
}

_SELECTOR_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": sorted(_SELECTOR_KEYS),
    "properties": {
        "kind": {"type": "string", "enum": sorted(_SELECTOR_KINDS)},
        "step_id": {"type": "string"},
        "uid": {"type": "string"},
        "category": {"type": "string", "enum": ["none", *sorted(_CATEGORIES)]},
        "color": {"type": "string", "enum": ["none", *sorted(_COLORS)]},
        "side": {"type": "string", "enum": sorted(_SIDES)},
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


def interpret_and_ground_task_spec(
    task_name: str,
    task_description: str,
    scene_objects: Sequence[Mapping[str, Any]],
    *,
    robot_profile: str,
    model: str | None = None,
    caller: InstructionCaller | None = None,
) -> GroundedTaskSpec:
    """Interpret free language and deterministically resolve it against a scene."""
    task_id = str(task_name).strip()
    instruction = str(task_description).strip()
    if not task_id or not instruction:
        raise ValueError("task_name and task_description must be non-empty.")
    index = _SceneIndex(scene_objects, robot_profile=robot_profile)
    prompt = _instruction_prompt(instruction, index)
    invoke = caller or _default_instruction_caller
    # An injected caller owns its transport and does not need the production
    # model-resolution path (which also loads provider configuration).
    selected_model = model if caller is not None else _instruction_model(model)
    if caller is None and selected_model is None:
        raise ValueError(
            "A text LLM model is required through --llm-model, "
            "ACTION_ENGINE_LLM_MODEL, or OPENAI_MODEL."
        )
    started = perf_counter()
    first_error: Exception | None = None
    intent: dict[str, Any] | None = None
    grounded: GroundedTaskSpec | None = None
    local_completion_fields: tuple[str, ...] = ()
    intent_normalizations: list[dict[str, Any]] = []
    attempts = 0
    for attempt in range(2):
        current_prompt = prompt
        if first_error is not None:
            current_prompt += (
                "\n\nREPAIR OVERRIDE: the previous JSON was invalid. Return a corrected "
                "JSON object only; do not repeat the sparse response. Every step "
                "must contain all 14 step keys and every selector all 8 selector "
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
        attempts += 1
        response_value: Mapping[str, Any] | None = None
        current_normalizations: list[dict[str, Any]] = []
        try:
            response = invoke(
                prompt=current_prompt,
                schema=deepcopy(INSTRUCTION_INTENT_SCHEMA),
                model=selected_model,
            )
            response_value, current_normalizations = (
                _normalize_instruction_intent_fields(
                    _coerce_instruction_response(response)
                )
            )
            intent = validate_instruction_intent(response_value)
            intent_normalizations = current_normalizations
            break
        except (TypeError, ValueError) as error:
            if attempt:
                completed = _complete_missing_explicit_target(
                    response_value,
                    error=error,
                    task_id=task_id,
                    instruction=instruction,
                    scene_objects=scene_objects,
                    robot_profile=robot_profile,
                    index=index,
                )
                if completed is not None:
                    intent, grounded, local_completion_fields = completed
                    intent_normalizations = current_normalizations
                    break
                raise ValueError(
                    "Instruction intent failed validation after one repair: " f"{error}"
                ) from error
            first_error = error
    if intent is None:
        raise AssertionError("unreachable")
    if grounded is None:
        grounded = _ground_intent(task_id, instruction, intent, index)
    grounded.task_spec["metadata"].update(
        {
            "instruction_interpreter": "structured_llm_v1",
            "instruction_model": selected_model or "injected_caller",
            "instruction_call_count": attempts,
            "instruction_latency_seconds": perf_counter() - started,
        }
    )
    if local_completion_fields:
        grounded.task_spec["metadata"].update(
            {
                "instruction_local_completion_count": len(local_completion_fields),
                "instruction_local_completion_fields": list(local_completion_fields),
                "instruction_local_completion_basis": ("deterministic_scene_grounding"),
            }
        )
    if intent_normalizations:
        grounded.task_spec["metadata"][
            "instruction_intent_normalizations"
        ] = intent_normalizations
    return grounded


def _normalize_instruction_intent_fields(
    value: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Canonicalize only fields that the selected E type cannot consume.

    The strict public validator deliberately remains unchanged.  This pass is
    confined to the LLM boundary, where weak JSON-mode providers sometimes
    copy a meaningful value into an inapplicable slot such as E4.required_arm.
    Required semantic fields are never inferred here and still fail closed.
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
    return result, changes


def _empty_selector() -> dict[str, Any]:
    """Return the canonical selector value for an inapplicable target."""
    return {
        "kind": "none",
        "step_id": "",
        "uid": "",
        "category": "none",
        "color": "none",
        "side": "none",
        "quantifier": "one",
        "count": 0,
    }


def _complete_missing_explicit_target(
    value: Mapping[str, Any] | None,
    *,
    error: Exception,
    task_id: str,
    instruction: str,
    scene_objects: Sequence[Mapping[str, Any]],
    robot_profile: str,
    index: _SceneIndex,
) -> tuple[dict[str, Any], GroundedTaskSpec, tuple[str, ...]] | None:
    """Complete one explicit E1 target only when two parsers agree otherwise."""
    if not isinstance(error, _MissingRequiredTargetError) or not isinstance(
        value, Mapping
    ):
        return None
    if set(value) != {"steps"}:
        return None
    steps = value.get("steps")
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
        return None

    missing_indices = [
        step_index
        for step_index, step in enumerate(steps)
        if isinstance(step, Mapping)
        and step.get("task_type") == "E1"
        and step.get("layout") != "line"
        and isinstance(step.get("target"), Mapping)
        and step["target"].get("kind") == "none"
    ]
    if len(missing_indices) != 1:
        return None

    try:
        reference = plan_grounded_task_spec(
            task_name=task_id,
            task_description=instruction,
            scene_objects=scene_objects,
            robot_profile=robot_profile,
        )
    except (TypeError, ValueError):
        return None
    reference_instances = reference.task_spec.get("task_instances", [])
    if len(reference_instances) != len(steps):
        return None
    if [step.get("task_type") for step in steps if isinstance(step, Mapping)] != [
        instance.get("task_type")
        for instance in reference_instances
        if isinstance(instance, Mapping)
    ]:
        return None

    missing_index = missing_indices[0]
    reference_instance = reference_instances[missing_index]
    if not isinstance(reference_instance, Mapping):
        return None
    params = reference_instance.get("params")
    if not isinstance(params, Mapping):
        return None
    target_role = params.get("target_role")
    target_uid = reference.role_bindings.get(str(target_role))
    if not target_uid or target_uid not in index.by_uid:
        return None

    patched = deepcopy(dict(value))
    patched["steps"][missing_index]["target"] = _uid_selector(target_uid)
    try:
        completed_intent = validate_instruction_intent(patched)
        completed_grounding = _ground_intent(
            task_id,
            instruction,
            completed_intent,
            index,
        )
    except (TypeError, ValueError):
        return None
    if not _same_grounded_semantics(completed_grounding, reference):
        return None
    return (
        completed_intent,
        completed_grounding,
        (f"steps[{missing_index}].target",),
    )


def _uid_selector(uid: str) -> dict[str, Any]:
    """Return the canonical selector for one scene-authoritative UID."""
    return {
        "kind": "selector",
        "step_id": "",
        "uid": uid,
        "category": "none",
        "color": "none",
        "side": "none",
        "quantifier": "one",
        "count": 0,
    }


def _same_grounded_semantics(
    candidate: GroundedTaskSpec,
    reference: GroundedTaskSpec,
) -> bool:
    """Compare task meaning after replacing symbolic roles with scene UIDs."""

    def normalized_steps(value: GroundedTaskSpec) -> list[dict[str, Any]]:
        result = []
        for instance in value.task_spec.get("task_instances", []):
            if not isinstance(instance, Mapping):
                return []
            params = deepcopy(dict(instance.get("params", {})))
            for key, parameter in list(params.items()):
                if key.endswith("_role") and isinstance(parameter, str):
                    params[key] = value.role_bindings.get(parameter, parameter)
                elif key.endswith("_roles") and isinstance(parameter, list):
                    params[key] = [
                        value.role_bindings.get(str(role), str(role))
                        for role in parameter
                    ]
            result.append(
                {
                    "task_type": instance.get("task_type"),
                    "params": params,
                }
            )
        return result

    return normalized_steps(candidate) == normalized_steps(reference)


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


def _ground_intent(
    task_id: str,
    instruction: str,
    intent: Mapping[str, Any],
    index: _SceneIndex,
) -> GroundedTaskSpec:
    builder = _TaskBuilder(task_id, instruction, index)
    objects_by_step: dict[str, list[_Entity]] = {}
    task_ids_by_step: dict[str, list[str]] = {}
    # The intent validator restricts object references to preceding instruction
    # steps.  Preserve the explicit dependency DAG for independent operations,
    # while keeping reference grounding deterministic.
    for step in _topological_steps(intent["steps"]):
        step_id = str(step["id"])
        objects = _resolve_reference(
            step["object"],
            index,
            objects_by_step,
            context=f"instruction step {step_id!r} object",
        )
        _validate_compatibility(str(step["task_type"]), objects)
        target_objects = _resolve_reference(
            step["target"],
            index,
            objects_by_step,
            context=f"instruction step {step_id!r} target",
            allow_none=True,
            exclude={item.uid for item in objects},
            allow_support=True,
        )
        if len(target_objects) > 1:
            raise ValueError(f"Instruction step {step_id!r} target is ambiguous.")
        _validate_target_compatibility(
            str(step["task_type"]),
            target_objects[0] if target_objects else None,
            relation=str(step["relation"]),
        )
        # A cross-step selector is an explicit data dependency even when the
        # caller omitted it in ``depends_on``.  This is the deterministic
        # interpretation of pronouns such as ``其``/``it``.
        dependencies_by_step = list(step["depends_on"])
        for selector in (step["object"], step["target"]):
            if selector["kind"] == "step_result":
                reference = str(selector["step_id"])
                if reference not in dependencies_by_step:
                    dependencies_by_step.append(reference)
        dependencies = [
            task_id
            for dependency in dependencies_by_step
            for task_id in task_ids_by_step[str(dependency)]
        ]
        emitted = _emit_step(
            builder,
            step,
            objects,
            target_objects[0] if target_objects else None,
            dependencies,
        )
        objects_by_step[step_id] = objects
        task_ids_by_step[step_id] = emitted
    return builder.build()


def _emit_step(
    builder: _TaskBuilder,
    step: Mapping[str, Any],
    objects: Sequence[_Entity],
    target: _Entity | None,
    dependencies: Sequence[str],
) -> list[str]:
    task_type = str(step["task_type"])
    if step["layout"] == "line":
        roles = [builder._role(entity, "E1") for entity in objects]
        parent = str(step["id"])
        emitted = []
        for slot, entity in enumerate(objects):
            emitted.append(
                builder.add(
                    "E1",
                    entity,
                    params={
                        "target_role": "table",
                        "relation": "on",
                        "layout": "line",
                        "objects_roles": roles,
                        "axis": "world_y" if step["axis"] == "none" else step["axis"],
                        "order_by": "explicit",
                        "order_direction": "given",
                        "order_constraint": "free",
                        "orientation_goal": step["orientation_goal"],
                        "orientation_axis": "none",
                        "nominal_slot_index": slot,
                        "slot_constraint": "free_reassignable",
                        "parent_task_instance_id": parent,
                    },
                    depends_on=dependencies,
                )
            )
        return emitted

    emitted = []
    for entity in objects:
        params: dict[str, Any] = {}
        required_arm = str(step["required_arm"])
        if required_arm in {"left_arm", "right_arm"}:
            params["required_arm"] = required_arm
        if task_type == "E1":
            relation = str(step["relation"])
            if relation == "none":
                # The only unambiguous implicit placement is onto the unique
                # support surface.  A movable target could mean on/inside/
                # beside and must be stated rather than guessed.
                if target is None or target.category != "table":
                    raise ValueError(
                        "E1 omitted relation is only valid for a unique table "
                        "support target."
                    )
                relation = "on"
            params.update(
                {
                    "relation": relation,
                    "relation_frame": "robot",
                    "orientation_goal": step["orientation_goal"],
                    "orientation_axis": "none",
                }
            )
        elif task_type == "E2":
            params.update(
                {
                    "orientation_goal": "upright",
                    "support_role": "table",
                    "upright_local_axis": "long_axis",
                }
            )
        elif task_type == "E3":
            params.update({"relation": "above", "relation_frame": "robot"})
        elif task_type == "E4":
            params.update(
                {
                    "transfer_arm": step["transfer_arm"],
                    "receive_arm": step["receive_arm"],
                    "orientation_goal": step["orientation_goal"],
                }
            )
        elif task_type == "E5":
            params.update({"direction": "up", "terminal_behavior": "hold"})
        elif task_type in {"E6", "E7"}:
            params["target_state"] = step["target_state"]
        elif task_type == "E8":
            params["target_setting"] = int(step["target_setting"])
        elif task_type == "E9":
            params["terminal_state"] = step["target_state"]
        emitted.append(
            builder.add(
                task_type,
                entity,
                target=target,
                params=params,
                depends_on=dependencies,
            )
        )
    return emitted


def _resolve_reference(
    selector: Mapping[str, Any],
    index: _SceneIndex,
    objects_by_step: Mapping[str, Sequence[_Entity]],
    *,
    context: str,
    allow_none: bool = False,
    exclude: set[str] | None = None,
    allow_support: bool = False,
) -> list[_Entity]:
    kind = str(selector["kind"])
    if kind == "none":
        if allow_none:
            return []
        raise ValueError(f"{context} is required.")
    if kind == "step_result":
        step_id = str(selector["step_id"])
        if step_id not in objects_by_step:
            raise ValueError(f"{context} references unavailable step {step_id!r}.")
        objects = list(objects_by_step[step_id])
        if len(objects) != 1:
            raise ValueError(
                f"{context} references step {step_id!r}, which has {len(objects)} objects."
            )
        if exclude and objects[0].uid in exclude:
            raise ValueError(
                f"{context} references the same object as its source; "
                "self-referential placement is not allowed."
            )
        return objects

    excluded = exclude or set()
    source_pool = index.entities if allow_support else index.movable
    pool = [entity for entity in source_pool if entity.uid not in excluded]
    uid = str(selector["uid"])
    category = str(selector["category"])
    color = str(selector["color"])
    if uid:
        if uid not in index.by_uid:
            raise ValueError(f"{context} references unknown scene UID {uid!r}.")
        bound = index.by_uid[uid]
        if bound.uid in excluded:
            pool = []
        else:
            if category != "none" and bound.category != category:
                raise ValueError(
                    f"{context} selector conflicts with UID {uid!r}: "
                    f"category is {bound.category!r}, not {category!r}."
                )
            if color != "none" and bound.color != color:
                raise ValueError(
                    f"{context} selector conflicts with UID {uid!r}: "
                    f"color is {bound.color!r}, not {color!r}."
                )
    # Apply every non-UID constraint to the complete candidate set first.  An
    # explicit UID is a conjunctive assertion, not permission to redefine
    # "leftmost" after narrowing the set to that UID.
    if category != "none":
        pool = [entity for entity in pool if entity.category == category]
    if color != "none":
        pool = [entity for entity in pool if entity.color == color]
    side = str(selector["side"])
    if side == "left":
        pool = [entity for entity in pool if index.left_score(entity) > 0.0]
    elif side == "right":
        pool = [entity for entity in pool if index.left_score(entity) < 0.0]
    elif side in {"leftmost", "rightmost"} and pool:
        scores = [index.left_score(entity) for entity in pool]
        extreme = max(scores) if side == "leftmost" else min(scores)
        tied = [entity for entity in pool if index.left_score(entity) == extreme]
        if len(tied) != 1:
            raise ValueError(f"{context} has an ambiguous {side} object selector.")
        pool = tied
    if uid:
        pool = [entity for entity in pool if entity.uid == uid]
        if not pool:
            if uid in excluded:
                raise ValueError(f"{context} selector references excluded UID {uid!r}.")
            if side in {"left", "right"}:
                raise ValueError(
                    f"{context} selector conflicts with UID {uid!r} in "
                    f"robot-relative {side} side."
                )
            if side in {"leftmost", "rightmost"}:
                raise ValueError(
                    f"{context} selector conflicts with UID {uid!r}: it is not "
                    f"the unique robot-relative {side} candidate."
                )
    pool = sorted(pool, key=lambda item: item.uid)
    if not pool:
        raise ValueError(f"{context} did not match any scene object.")
    quantifier = str(selector["quantifier"])
    count = int(selector["count"])
    if quantifier == "one" and len(pool) != 1:
        raise ValueError(
            f"{context} is ambiguous; matched scene UIDs {[item.uid for item in pool]}."
        )
    if quantifier == "count" and (count < 1 or len(pool) != count):
        raise ValueError(
            f"{context} requested exactly {count} objects but matched {len(pool)}."
        )
    if quantifier == "all" and count not in {0, len(pool)}:
        raise ValueError(
            f"{context} quantifier=all cannot carry count={count}; use count for an exact quantity."
        )
    return pool


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
    selector["uid"] = _selector_string(selector["uid"], f"{context}.uid")
    selector["category"] = _canonical_category(
        selector["category"], f"{context}.category"
    )
    selector["color"] = _canonical_color(selector["color"], f"{context}.color")
    selector["side"] = _canonical_side(selector["side"], f"{context}.side")
    selector["quantifier"] = _canonical_quantifier(
        selector["quantifier"], f"{context}.quantifier"
    )
    if isinstance(selector["count"], bool) or not isinstance(selector["count"], int):
        raise ValueError(f"{context}.count must be an integer.")
    if selector["count"] < 0:
        raise ValueError(f"{context}.count must be non-negative.")
    kind = selector["kind"]
    if kind == "selector" and not any(
        (
            selector["uid"],
            selector["category"] != "none",
            selector["color"] != "none",
            selector["side"] != "none",
        )
    ):
        raise ValueError(f"{context} selector has no identifying constraint.")
    if kind == "step_result":
        if not selector["step_id"]:
            raise ValueError(f"{context} step_result requires step_id.")
        if any(
            (
                selector["uid"],
                selector["category"] != "none",
                selector["color"] != "none",
                selector["side"] != "none",
            )
        ):
            raise ValueError(
                f"{context} step_result may identify only a prior step_id."
            )
        if selector["quantifier"] != "one" or selector["count"] != 0:
            raise ValueError(
                f"{context} step_result requires quantifier=one and count=0."
            )
    if kind == "selector" and selector["step_id"]:
        raise ValueError(f"{context} selector cannot carry step_id.")
    if kind == "none" and any(
        (
            selector["step_id"],
            selector["uid"],
            selector["category"] != "none",
            selector["color"] != "none",
            selector["side"] != "none",
        )
    ):
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
    target_kind = str(step["target"]["kind"])
    if task_type not in {"E1", "E3"} and step["relation"] != "none":
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
    if task_type not in {"E1", "E2", "E4"} and orientation_goal != "preserve":
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
    elif target_kind != "none":
        raise ValueError(f"{context} {task_type} does not accept a target selector.")
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


def _validate_compatibility(task_type: str, objects: Sequence[_Entity]) -> None:
    allowed_categories: dict[str, set[str]] = {
        "E3": {"can", "cup", "bottle", "bowl", "pourable_container"},
        "E5": {"tray", "basket", "bowl", "bucket"},
        "E6": {"drawer", "tray"},
        "E7": {"drawer", "tray"},
        "E8": {"knob"},
        "E9": {"button"},
    }
    allowed = allowed_categories.get(task_type)
    if task_type in {"E2", "E4"}:
        # These two operations are defined by grasp/orient or handover
        # affordances rather than a closed object taxonomy.  When a scene
        # export omits explicit affordances, reject only known non-graspable
        # controls/support surfaces and let the runtime capability preflight
        # make the final decision.
        invalid_categories = {"button", "drawer", "knob", "table"}
        invalid = [
            entity.uid for entity in objects if entity.category in invalid_categories
        ]
        if invalid:
            raise ValueError(
                f"{task_type} is incompatible with non-graspable scene objects "
                f"{invalid}."
            )
    elif allowed is not None:
        invalid = [entity.uid for entity in objects if entity.category not in allowed]
        if invalid:
            raise ValueError(
                f"{task_type} is incompatible with scene objects {invalid}; "
                f"allowed categories are {sorted(allowed)}."
            )
    elif task_type == "E1":
        invalid = [
            entity.uid
            for entity in objects
            if entity.category in {"button", "drawer", "knob", "table"}
        ]
        if invalid:
            raise ValueError(
                f"E1 is incompatible with non-graspable scene objects {invalid}."
            )
    required_affordances = set(_AFFORDANCES.get(task_type, ()))
    for entity in objects:
        # Exported Prompt2Scene objects historically omit affordances.  In that
        # case category compatibility is the available evidence; when a scene
        # explicitly reports affordances, enforce them rather than guessing.
        if entity.affordances:
            missing = required_affordances - set(entity.affordances)
            if missing:
                raise ValueError(
                    f"{task_type} is incompatible with scene object {entity.uid!r}; "
                    f"missing affordances {sorted(missing)}."
                )


def _validate_target_compatibility(
    task_type: str,
    target: _Entity | None,
    *,
    relation: str,
) -> None:
    """Reject target selectors that cannot satisfy the requested E semantics."""
    if task_type == "E3":
        if target is None:
            raise ValueError("E3 requires a target container.")
        containers = {
            "basket",
            "bowl",
            "bucket",
            "can",
            "cup",
            "bottle",
            "pourable_container",
            "tray",
        }
        if target.category not in containers:
            raise ValueError(f"E3 target {target.uid!r} is not a compatible container.")
    if task_type == "E1" and relation == "inside":
        if target is None:
            raise ValueError("E1 inside relation requires a target container.")
        containers = {"basket", "bowl", "bucket", "cup", "drawer", "tray"}
        if target.category not in containers:
            raise ValueError(
                f"E1 inside target {target.uid!r} is not a compatible container."
            )


def _instruction_prompt(instruction: str, index: _SceneIndex) -> str:
    inventory = [
        {
            "uid": entity.uid,
            "role": entity.role,
            "category": entity.category,
            "color": entity.color,
            "description": entity.description,
            "affordances": sorted(entity.affordances),
            "attributes": _prompt_attributes(entity.attributes),
        }
        for entity in index.entities
    ]
    return (
        "Convert the user's explicit L1-L3 instruction into typed E1-E9 task "
        "intent. Understand synonyms, ellipsis, and pronouns such as it/其, but "
        "do not invent missing objects. Use step_result for cross-step pronouns. "
        "Object left/right is robot-relative; arm names are robot body sides. "
        "Prefer an exact inventory UID for a named object. When UID alone "
        "identifies it, set category, color, and side to 'none'; selector fields "
        "are conjunctive constraints, not descriptive metadata. "
        "Use side=left/right for a robot half-space constraint and "
        "leftmost/rightmost only for an ordinal request. Emit no AtomicAction, "
        "UID not present in the inventory, coordinates, poses, paths, or "
        "reasoning. Encode explicit ordering with depends_on; same-action set "
        "members may remain independent. Use empty strings and 'none' for "
        "inapplicable required fields. A request to retract the transfer arm "
        "immediately after an E4 handover is a mandatory runtime retreat/home "
        "barrier for that E4; do "
        "not emit a separate task step for it. The exact output keys are steps -> id, "
        "task_type, object, target, relation, required_arm, transfer_arm, "
        "receive_arm, orientation_goal, target_state, target_setting, layout, "
        "axis, depends_on; each selector has kind, step_id, uid, category, "
        "color, side, quantifier, count.\n\n"
        f"Instruction:\n{instruction}\n\n"
        f"Scene inventory:\n{json.dumps(inventory, ensure_ascii=False, sort_keys=True)}\n\n"
        f"E1-E9 catalog:\n{json.dumps(_intent_capability_catalog(), ensure_ascii=False, sort_keys=True)}\n\n"
        "Shape-only complete JSON example (do not copy its step count or values; "
        "copy every key, including keys whose value is none/empty/0):\n"
        f"{json.dumps(_instruction_shape_example(), ensure_ascii=False, sort_keys=True)}\n\n"
        "Selector kind rules (these are not extra output fields):\n"
        f"{_instruction_selector_rules()}\n\n"
        "Final checklist: every step has all 14 step keys; every object and target "
        "has all 8 selector keys. For an inapplicable field use the canonical "
        "default shown in the example, never omit the field. E4 must explicitly "
        "state transfer_arm and receive_arm. E1/E3 must explicitly state target "
        "and relation (except E1 layout=line)."
    )


def _instruction_shape_example() -> dict[str, Any]:
    """Return a compact field-complete example for providers with weak schemas."""
    selector = {
        "kind": "selector",
        "step_id": "",
        "uid": "",
        "category": "can",
        "color": "purple",
        "side": "none",
        "quantifier": "one",
        "count": 0,
    }
    empty_selector = {
        "kind": "none",
        "step_id": "",
        "uid": "",
        "category": "none",
        "color": "none",
        "side": "none",
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
                "depends_on": [],
            }
        ]
    }


def _instruction_selector_rules() -> str:
    """Return the mutually exclusive selector encodings for model prompts."""
    step_result = {
        "kind": "step_result",
        "step_id": "step_1",
        "uid": "",
        "category": "none",
        "color": "none",
        "side": "none",
        "quantifier": "one",
        "count": 0,
    }
    return (
        "- kind=none: step_id and uid are empty strings; category, color, and "
        "side are 'none'; quantifier='one'; count=0.\n"
        "- kind=selector: step_id is an empty string; use at least one of uid, "
        "category, color, or side to identify scene objects.\n"
        "- kind=step_result: use it only for a pronoun that means exactly one "
        "object from an earlier instruction step. Set step_id to that prior "
        "step ID and set uid='', category='none', color='none', side='none', "
        "quantifier='one', count=0. Do not copy the prior object's UID, "
        "category, color, or side into this selector. Replace step_1 in this "
        f"complete shape with the actual prior step ID: {json.dumps(step_result, sort_keys=True)}\n"
        "A step_result may identify only a prior step_id; it cannot carry any "
        "other object constraint."
    )


def _instruction_repair_guidance(error: Exception) -> str:
    """Add narrow semantic guidance for errors weak JSON-mode models repeat."""
    if not isinstance(error, _MissingRequiredTargetError):
        return ""
    return (
        "\nMissing-target repair rule: for a non-line E1 placement, object is "
        "the item being moved and target is the explicit reference object "
        "after the spatial relation in the original instruction. For example, "
        "in 'place it to the left of the orange can', object is the earlier "
        "step_result for 'it', while target selects the orange can; target "
        "must not use kind=none. Use target kind=step_result only when the "
        "reference object itself is exactly the result of a prior step.\n"
    )


def _prompt_attributes(value: Mapping[str, Any]) -> dict[str, Any]:
    """Keep descriptive scalar attributes while redacting nested geometry."""
    result: dict[str, Any] = {}
    for key, child in value.items():
        name = str(key)
        normalized_name = name.strip().lower().replace("-", "_")
        if normalized_name in _PROMPT_REDACTED_KEYS:
            continue
        if isinstance(child, Mapping):
            nested = _prompt_attributes(child)
            if nested:
                result[name] = nested
        elif isinstance(child, (str, int, float, bool)) and not isinstance(
            child, complex
        ):
            result[name] = child
        # Numeric sequences are intentionally omitted: without a schema they
        # are too easy to mistake for a coordinate or pose vector.
    return result


def _intent_capability_catalog() -> dict[str, dict[str, Any]]:
    """Return the LLM's thin, import-safe E1-E9 capability view.

    ``task_capability_catalog`` also reports runtime availability and therefore
    imports simulator action classes.  Text interpretation only needs the
    symbolic E semantics and must remain testable before a simulator backend is
    installed.
    """
    return {
        task_type: {
            "semantics": str(definition["semantics"]),
            "applicable_fields": sorted(_INTENT_TASK_FIELD_REGISTRY[task_type]),
        }
        for task_type, definition in _E_DEFINITIONS.items()
    }


def _default_instruction_caller(
    *,
    prompt: str,
    schema: Mapping[str, Any],
    model: str | None,
) -> Mapping[str, Any]:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    from embodichain.gen_sim.action_engine.planning.planner import (
        _coerce_model_response,
        _is_mimo_compatible,
        _load_llm_settings,
        _structured_output_runnable,
    )

    settings = _load_llm_settings(model=model)
    kwargs: dict[str, Any] = {
        "api_key": settings["api_key"],
        "model": settings["model"],
        "temperature": 0,
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
                    "Return only the requested structured task intent. Never "
                    "return reasoning, coordinates, or AtomicAction nodes."
                )
            ),
            HumanMessage(content=schema_prompt),
        ]
    )
    return _coerce_model_response(response)


def _instruction_model(explicit: str | None) -> str | None:
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    # Keep model selection separate from credential loading.  Reading the local
    # dotenv file is side-effect free and gives generation the documented
    # priority without leaking credentials into TaskSpec metadata.
    for name in ("ACTION_ENGINE_LLM_MODEL", "OPENAI_MODEL"):
        for source in (
            os.environ,
            _load_local_env(),
        ):
            value = source.get(name)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _load_local_env() -> dict[str, str]:
    """Use the planner's dotenv parser so selection and client setup agree."""
    from embodichain.gen_sim.action_engine.planning.planner import (
        _GEN_SIM_ENV_PATH,
        _load_env_file,
    )

    return _load_env_file(_GEN_SIM_ENV_PATH)


def _choice(value: Any, allowed: set[str] | frozenset[str], context: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{context} must be one of {sorted(allowed)}.")
    return value


def _selector_string(value: Any, context: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a string.")
    return value.strip()


def _canonical_value(
    value: Any,
    aliases: Mapping[str, str],
    allowed: set[str] | frozenset[str],
    context: str,
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a string.")
    text = value.strip()
    canonical = aliases.get(text.lower(), text)
    if canonical not in allowed:
        raise ValueError(f"{context} must be one of {sorted(allowed)}.")
    return canonical


def _canonical_color(value: Any, context: str) -> str:
    return _canonical_value(value, _COLOR_ALIASES, {"none", *_COLORS}, context)


def _canonical_category(value: Any, context: str) -> str:
    return _canonical_value(
        value,
        _CATEGORY_ALIASES,
        {"none", *_CATEGORIES, "pourable_container"},
        context,
    )


def _canonical_side(value: Any, context: str) -> str:
    return _canonical_value(value, _SIDE_ALIASES, _SIDES, context)


def _canonical_quantifier(value: Any, context: str) -> str:
    return _canonical_value(value, _QUANTIFIER_ALIASES, _QUANTIFIERS, context)


def _canonical_arm(value: Any, context: str) -> str:
    return _canonical_value(value, _ARM_ALIASES, _ARMS, context)


def _canonical_relation(value: Any, context: str) -> str:
    return _canonical_value(value, _RELATION_ALIASES, _RELATIONS, context)


def _canonical_orientation(value: Any, context: str) -> str:
    aliases = {
        "upright": "upright",
        "竖直": "upright",
        "直立": "upright",
        "扶正": "upright",
        "preserve": "preserve",
        "保持": "preserve",
        "none": "preserve",
    }
    return _canonical_value(value, aliases, _ORIENTATIONS, context)


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
