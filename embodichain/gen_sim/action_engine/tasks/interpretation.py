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

"""Compatibility bridge from Task Engine drafts to Action Engine TaskSpec v2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from embodichain.gen_sim.task_engine.interpretation import (
    INSTRUCTION_INTENT_SCHEMA,
    InstructionCaller,
    InstructionDraftResult,
    InstructionIntent,
    _default_instruction_caller,
    _instruction_prompt,
    _instruction_selector_rules,
    interpret_instruction_draft,
    validate_instruction_intent,
)

from .assembly import (
    GroundedTaskBuilder,
    GroundedTaskSpec,
    SceneEntity,
    SceneInventory,
    validate_source_compatibility,
    validate_target_compatibility,
)
from .grounding import GroundingCaller, ground_scene_references

__all__ = [
    "GroundingCaller",
    "INSTRUCTION_INTENT_SCHEMA",
    "InstructionCaller",
    "InstructionDraftResult",
    "InstructionIntent",
    "ground_instruction_draft",
    "interpret_and_ground_task_spec",
    "interpret_instruction_draft",
    "validate_instruction_intent",
]


def interpret_and_ground_task_spec(
    task_name: str,
    task_description: str,
    scene_objects: Sequence[Mapping[str, Any]],
    *,
    robot_profile: str,
    model: str | None = None,
    caller: InstructionCaller | None = None,
    grounding_caller: GroundingCaller | None = None,
) -> GroundedTaskSpec:
    """Interpret through Task Engine, then ground through Action Engine."""
    task_id = str(task_name).strip()
    instruction = str(task_description).strip()
    if not task_id or not instruction:
        raise ValueError("task_name and task_description must be non-empty.")
    inventory = SceneInventory(scene_objects, robot_profile=robot_profile)
    draft = interpret_instruction_draft(instruction, model=model, caller=caller)
    invoke = caller or _default_instruction_caller
    selected_model = None if draft.model == "injected_caller" else draft.model
    grounding = ground_scene_references(
        instruction=instruction,
        intent=draft.intent,
        inventory=inventory,
        scene_objects=scene_objects,
        model=selected_model,
        caller=grounding_caller or invoke,
    )
    grounded = _ground_intent(
        task_id,
        instruction,
        draft.intent,
        inventory,
        grounding.bindings,
    )
    grounded.task_spec["metadata"].update(
        {
            "instruction_interpreter": "structured_llm_v2",
            "instruction_model": draft.model,
            "instruction_call_count": draft.attempts,
            "instruction_latency_seconds": draft.latency_seconds,
            "scene_grounding_model": selected_model or "injected_caller",
            "scene_grounding_call_count": grounding.attempts,
            "scene_grounding_latency_seconds": grounding.latency_seconds,
        }
    )
    if draft.normalizations:
        grounded.task_spec["metadata"]["instruction_intent_normalizations"] = list(
            draft.normalizations
        )
    return grounded


def ground_instruction_draft(
    task_id: str,
    instruction: str,
    intent: Mapping[str, Any],
    scene_objects: Sequence[Mapping[str, Any]],
    *,
    robot_profile: str,
    reference_bindings: Mapping[str, Sequence[str]],
) -> GroundedTaskSpec:
    """Lower a Task Engine draft using verified scene bindings."""
    normalized_task_id = str(task_id).strip()
    normalized_instruction = str(instruction).strip()
    if not normalized_task_id or not normalized_instruction:
        raise ValueError("task_id and instruction must be non-empty.")
    inventory = SceneInventory(scene_objects, robot_profile=robot_profile)
    return _ground_intent(
        normalized_task_id,
        normalized_instruction,
        validate_instruction_intent(intent),
        inventory,
        reference_bindings,
    )


def _ground_intent(
    task_id: str,
    instruction: str,
    intent: Mapping[str, Any],
    inventory: SceneInventory,
    scene_bindings: Mapping[str, Sequence[str]],
) -> GroundedTaskSpec:
    builder = GroundedTaskBuilder(
        task_id,
        instruction,
        inventory,
        planner="structured_llm_v2",
    )
    objects_by_step: dict[str, list[SceneEntity]] = {}
    task_ids_by_step: dict[str, list[str]] = {}
    for step in _topological_steps(intent["steps"]):
        step_id = str(step["id"])
        objects = _resolve_reference(
            step["object"],
            inventory,
            objects_by_step,
            context=f"instruction step {step_id!r} object",
            reference_id=f"{step_id}.object",
            scene_bindings=scene_bindings,
        )
        validate_source_compatibility(str(step["task_type"]), objects)
        target_objects = _resolve_reference(
            step["target"],
            inventory,
            objects_by_step,
            context=f"instruction step {step_id!r} target",
            reference_id=f"{step_id}.target",
            scene_bindings=scene_bindings,
            allow_none=True,
            exclude={item.uid for item in objects},
            allow_support=True,
        )
        if len(target_objects) > 1:
            raise ValueError(f"Instruction step {step_id!r} target is ambiguous.")
        validate_target_compatibility(
            str(step["task_type"]),
            target_objects[0] if target_objects else None,
            relation=str(step["relation"]),
        )
        dependencies_by_step = list(step["depends_on"])
        for selector in (step["object"], step["target"]):
            if selector["kind"] == "step_result":
                reference = str(selector["step_id"])
                if reference not in dependencies_by_step:
                    dependencies_by_step.append(reference)
        dependencies = [
            emitted_id
            for dependency in dependencies_by_step
            for emitted_id in task_ids_by_step[str(dependency)]
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
    builder: GroundedTaskBuilder,
    step: Mapping[str, Any],
    objects: Sequence[SceneEntity],
    target: SceneEntity | None,
    dependencies: Sequence[str],
) -> list[str]:
    task_type = str(step["task_type"])
    if step["layout"] == "line":
        roles = [builder._role(entity, "E1") for entity in objects]
        parent = str(step["id"])
        return [
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
            for slot, entity in enumerate(objects)
        ]

    emitted = []
    for entity in objects:
        params: dict[str, Any] = {}
        required_arm = str(step["required_arm"])
        if required_arm in {"left_arm", "right_arm"}:
            params["required_arm"] = required_arm
        if task_type == "E1":
            relation = str(step["relation"])
            if relation == "none":
                if target is None or target not in builder.inventory.support:
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
                    "upright_local_axis": "z",
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
            params.update(
                {
                    "direction": step["direction"],
                    "terminal_behavior": step["terminal_behavior"],
                    "relation": step["relation"],
                    "relation_frame": "robot",
                }
            )
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
    inventory: SceneInventory,
    objects_by_step: Mapping[str, Sequence[SceneEntity]],
    *,
    context: str,
    reference_id: str,
    scene_bindings: Mapping[str, Sequence[str]],
    allow_none: bool = False,
    exclude: set[str] | None = None,
    allow_support: bool = False,
) -> list[SceneEntity]:
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

    if reference_id not in scene_bindings:
        raise ValueError(f"{context} has no verified scene-grounding binding.")
    excluded = exclude or set()
    source_uids = (
        {entity.uid for entity in inventory.entities}
        if allow_support
        else {entity.uid for entity in inventory.interactive}
    )
    resolved_uids = tuple(str(uid) for uid in scene_bindings[reference_id])
    pool = [
        inventory.by_uid[uid]
        for uid in resolved_uids
        if uid in source_uids and uid not in excluded
    ]
    pool = sorted(pool, key=lambda item: item.uid)
    if not pool:
        raise ValueError(f"{context} did not bind an eligible scene object.")
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


def _topological_steps(
    steps: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_id = {str(step["id"]): dict(step) for step in steps}
    effective_dependencies: dict[str, tuple[str, ...]] = {}
    for step_id, step in by_id.items():
        deps = [str(dep) for dep in step["depends_on"]]
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
        # Select one earliest-ready step at a time. Emitting the whole ready
        # frontier lets a later independent step leapfrog an earlier step that
        # becomes ready after its predecessor, changing the instruction's
        # resource-order tie break without any causal reason.
        step_id = ready[0]
        ordered.append(by_id[step_id])
        pending.remove(step_id)
    return ordered
