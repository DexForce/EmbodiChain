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

"""GenSim-owned adjacent Pick/Place qualification with immutable grasp geometry."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch

from embodichain.lab.sim.atomic_actions import JointPositionTarget
from embodichain.lab.sim.atomic_actions.plans import PlanningFailure
from embodichain.lab.task_program.semantics import (
    Pick,
    Place,
    RegisteredSemanticCall,
    EffectAssurance,
    SemanticEffectKind,
)
from embodichain.utils.math import pose_inv
from .alignment_continuation import _PREVIEW, _require_monitor
from .held_alignment_recovery import _RandomState, _cuda_devices
from .pick_safety import _inspect_pick_plan
from .planning_probe import _validate_final_plan_velocity
from .place_recovery import inspect as inspect_place_plan
from .articulation_collision import robot_collision_scope
from ._search_cost import measure_search, record_search

__all__: list[str] = []


@dataclass(frozen=True, slots=True)
class Continuation:
    pick_id: str
    place_id: str
    pick_call: Pick
    place_call: Place | RegisteredSemanticCall
    transport_id: str | None = None
    transport_call: RegisteredSemanticCall | None = None


@dataclass(frozen=True, slots=True)
class Preview:
    route: Continuation
    compiler: Any
    pick_workflow: Any
    place_workflow: Any
    transport_workflow: Any | None = None


def _place_identity(call: Any) -> tuple[str, str] | None:
    if type(call) is Place:
        return call.object.entity_id, "place"
    if (
        type(call) is not RegisteredSemanticCall
        or call.call_id != "simulation.place_relative"
        or call.arguments.get("reference_part") is not None
    ):
        return None
    arguments = call.arguments
    required = {"object", "reference", "relation"}
    if (
        not required <= set(arguments)
        or set(arguments) - required - {"stage_id", "reference_part"}
        or any(
            type(arguments[name]) is not str
            or not arguments[name]
            or arguments[name] != arguments[name].strip()
            for name in required
        )
        or (
            "stage_id" in arguments
            and (
                type(arguments["stage_id"]) is not str
                or arguments["stage_id"] != arguments["stage_id"].strip()
            )
        )
    ):
        return None
    # The registered factory owns displacement, yaw and exact target geometry.
    return arguments["object"], call.call_id


def bind_calls(program: Any, preset: Any) -> tuple[Continuation, ...]:
    if preset is None or preset.effect_assurance is not EffectAssurance.VERIFIED:
        return ()
    result = []
    segments = tuple(program.iter_segments())
    for index, current in enumerate(segments[:-1]):
        following = segments[index + 1]
        if any(
            len(s.calls) != 1 or s.parallel_block is not None
            for s in (current, following)
        ):
            continue
        pick, place = current.calls[0].call, following.calls[0].call
        transport = None
        transported_segment = None
        if (
            type(pick) is Pick
            and type(place) is RegisteredSemanticCall
            and place.call_id == "simulation.move_held_object"
        ):
            arguments = place.arguments
            if (
                index + 2 >= len(segments)
                or set(arguments) - {"object", "target", "reference"}
                or not {"object", "target"} <= set(arguments)
                or arguments["object"] != pick.object.entity_id
                or dict(place.resources) != dict(pick.resources)
                or any(
                    type(value) is not str or not value or value != value.strip()
                    for value in arguments.values()
                )
            ):
                continue
            transport, transported_segment = place, following
            following = segments[index + 2]
            if len(following.calls) != 1 or following.parallel_block is not None:
                continue
            place = following.calls[0].call
        place_identity = _place_identity(place)
        if (
            type(pick) is not Pick
            or place_identity is None
            or pick.object.entity_id != place_identity[0]
            or dict(pick.resources) != dict(place.resources)
            or set(pick.resources) != {"primary"}
            or any(k not in preset.effect_monitors for k in ("pick", place_identity[1]))
            or (
                transport is not None
                and transport.call_id not in preset.effect_monitors
            )
        ):
            continue

        def identity(segment: Any) -> str:
            return f"{program.program_id}/{segment.segment_id}:{segment.calls[0].segment_call_index}"

        result.append(
            Continuation(
                identity(current),
                identity(following),
                pick,
                place,
                (
                    identity(transported_segment)
                    if transported_segment is not None
                    else None
                ),
                transport,
            )
        )
    return tuple(result)


def bind_previews(
    compiler: Any, routes: tuple[Continuation, ...]
) -> dict[str, Preview]:
    return {
        r.pick_id: Preview(
            r,
            compiler,
            compiler.analyze((r.pick_call,), workflow_id=r.pick_id.rsplit(":", 1)[0]),
            compiler.analyze((r.place_call,), workflow_id=r.place_id.rsplit(":", 1)[0]),
            (
                compiler.analyze(
                    (r.transport_call,), workflow_id=r.transport_id.rsplit(":", 1)[0]
                )
                if r.transport_call is not None
                else None
            ),
        )
        for r in routes
    }


def eligible(
    action: Any, request: Any, context: Any, selection: Any, preview: Any
) -> bool:
    if preview is None or _PREVIEW.get():
        return False
    options = request.skill_options
    engine = preview.compiler.integration.engine
    return bool(
        context.batch_size == 1
        and context.env_ids.tolist() == [0]
        and action.robot is engine.robot
        and request.invocation_id == preview.route.pick_id
        and request.invocation_id not in getattr(engine, "_cartesian_calls", ())
        and (getattr(action, "pick_purposes", None) or {}).get(request.invocation_id)
        not in {"pour", "ends"}
        and request.goal.grasp_xpos is None
        and options.fixed_object_to_eef is None
        and options.rotate_upright is None
        and options.obj_upright_direction is None
        and options.pick_object_part == "center"
        and not selection.constrained
        and selection.candidates is not None
        and selection.direction is not None
        and request.motion_policy.strategy == "ik_interp"
        and request.motion_policy.dynamic_collision_mode == "off"
    )


def _same_resources(before: Any, after: Any) -> bool:
    return all(
        old.task_state_key == new.task_state_key and old.target == new.target
        for name in ("motion", "grasp")
        for old, new in (
            (
                before.binding.endpoint("primary", name),
                after.binding.endpoint("primary", name),
            ),
        )
    )


def _project_held(action: Any, request: Any, context: Any, plan: Any, held: Any) -> Any:
    future = context.project(
        qpos=plan.joint_trajectory.positions[:, -1].clone(),
        task=plan.expected_effects.apply(context.task, plan.plan_success),
    )
    arm = request.binding.endpoint("primary", "motion").require_target(
        JointPositionTarget
    )
    eef = action.robot.compute_fk(
        qpos=future.last_qpos[:, list(arm.joint_ids)],
        name=arm.control_part,
        to_matrix=True,
    )
    pose = eef @ pose_inv(held.object_to_eef.to(eef))
    entities = dict(future.scene.entities)
    uid = held.semantics.entity_id
    entities[uid] = replace(entities[uid], pose=pose)
    return replace(future, scene=replace(future.scene, entities=entities))


def _preview_place(
    action: Any,
    request: Any,
    context: Any,
    plan: Any,
    preview: Preview,
    *,
    safety: bool = False,
) -> dict[str, Any]:
    """Expected attachment is private bookkeeping, never acquired-state evidence."""
    key = request.binding.endpoint("primary", "motion").task_state_key
    uid = request.goal.semantics.entity_id
    held = plan.expected_effects.held_object_updates.get(key)
    if held is None or held.semantics.entity_id != uid:
        return {
            "valid": False,
            "status": "unavailable",
            "reason": "predicted_attachment_identity_unavailable",
        }
    pick = preview.compiler.ground(preview.pick_workflow, 0, context)
    _require_monitor(pick, SemanticEffectKind.ATTACH, preview.route.pick_id, uid, key)
    future = _project_held(action, request, context, plan, held)
    engine = preview.compiler.integration.engine
    transport_receipt = None
    if getattr(preview, "transport_workflow", None) is not None:
        transported = preview.compiler.ground(preview.transport_workflow, 0, future)
        _require_monitor(
            transported,
            SemanticEffectKind.ATTACH,
            preview.route.transport_id,
            uid,
            key,
        )
        transport_request = engine.resolve(transported.invocation)
        if not _same_resources(request, transport_request):
            return {
                "valid": False,
                "status": "unavailable",
                "reason": "transport_resource_identity_mismatch",
            }
        original_transform = held.object_to_eef.clone()
        with measure_search("transport_preview"):
            moved = engine.plan(transported.invocation, future)
        if not moved.plan_success.all():
            return {
                "valid": False,
                "status": "unqualified",
                "reason": "transport_planning_failed",
            }
        if not _validate_final_plan_velocity(moved, action.robot)["valid_mask"].all():
            return {
                "valid": False,
                "status": "unqualified",
                "reason": "transport_velocity_rejected",
            }
        effects = moved.expected_effects
        retained = effects.held_object_updates.get(key, held)
        hand_endpoint = transport_request.binding.endpoint("primary", "grasp")
        hand = hand_endpoint.require_target(JointPositionTarget)
        closed = hand_endpoint.joint_positions(
            "grasp",
            num_envs=1,
            device=context.robot.qpos.device,
            dtype=context.robot.qpos.dtype,
        )
        if (
            moved.invocation_id != preview.route.transport_id
            or moved.skill_id != "move_held_object"
            or set(effects.held_object_updates) - {key}
            or effects.coordinated_held_object_updates
            or effects.articulation_joint_updates
            or retained is None
            or retained.semantics.entity_id != uid
            or not torch.equal(retained.object_to_eef, original_transform)
            or not torch.equal(
                moved.joint_trajectory.positions[:, :, list(hand.joint_ids)],
                closed[:, None].expand(
                    -1, moved.joint_trajectory.positions.shape[1], -1
                ),
            )
        ):
            return {
                "valid": False,
                "status": "unavailable",
                "reason": "transport_retention_contract_mismatch",
            }
        transport_receipt = {
            "valid": True,
            "invocation_id": preview.route.transport_id,
            "scope": "canonical_transport_prediction_not_measured_retention",
            "path_qualification": "not_checked",
        }
        future = _project_held(action, transport_request, future, moved, retained)
    grounded = preview.compiler.ground(preview.place_workflow, 0, future)
    _require_monitor(
        grounded, SemanticEffectKind.RELEASE, preview.route.place_id, uid, key
    )
    placed_request = engine.resolve(grounded.invocation)
    if not _same_resources(request, placed_request):
        return {
            "valid": False,
            "status": "unavailable",
            "reason": "continuation_resource_identity_mismatch",
        }
    placed = engine.plan(grounded.invocation, future)
    if not placed.plan_success.all():
        probe = placed.diagnostics.metadata.get("gen_sim_place_waypoint_probe", {})
        return {
            "valid": False,
            "status": "unqualified",
            "reason": "place_planning_failed",
            "unreachable_waypoints": probe.get("kinematic_valid") is False,
        }
    if not _validate_final_plan_velocity(placed, action.robot)["valid_mask"].all():
        return {
            "valid": False,
            "status": "unqualified",
            "reason": "place_velocity_rejected",
        }
    hand_endpoint = placed_request.binding.endpoint("primary", "grasp")
    hand = hand_endpoint.require_target(JointPositionTarget)
    opened = hand_endpoint.joint_positions(
        "open",
        num_envs=1,
        device=context.robot.qpos.device,
        dtype=context.robot.qpos.dtype,
    )
    if (
        placed.invocation_id != preview.route.place_id
        or placed.skill_id != "place"
        or not torch.equal(
            placed.joint_trajectory.positions[:, -1, list(hand.joint_ids)], opened
        )
        or dict(placed.expected_effects.held_object_updates) != {key: None}
        or placed.expected_effects.coordinated_held_object_updates
        or placed.expected_effects.articulation_joint_updates
    ):
        return {
            "valid": False,
            "status": "unavailable",
            "reason": "continuation_release_contract_mismatch",
        }
    if not safety:
        return {
            "valid": True,
            "path_qualification": "not_requested",
            "transport_preview": transport_receipt,
        }
    inspection = inspect_place_plan(
        engine.actions["place"], placed_request, future, placed
    )
    reason = inspection.get("reason", "")
    return {
        **inspection,
        "transport_preview": transport_receipt,
        "status": (
            "qualified"
            if inspection["valid"]
            else (
                "unavailable"
                if (
                    "unavailable" in inspection
                    or "unavailable" in inspection.get("collision", {})
                    or reason.endswith("unavailable")
                )
                else "unqualified"
            )
        ),
    }


def preview_place(
    action: Any,
    request: Any,
    context: Any,
    plan: Any,
    preview: Preview,
    *,
    safety: bool = False,
) -> dict[str, Any]:
    """Predict privately without consuming the caller's RNG or physical state."""
    state = _RandomState.capture(
        _cuda_devices(action.robot.device, context.robot.qpos.device)
    )
    token = _PREVIEW.set(True)
    try:
        with measure_search("place_preview"):
            result = _preview_place(
                action, request, context, plan, preview, safety=safety
            )
        return {
            **result,
            "scope": "private_pick_place_prediction_not_measured_attachment",
            "physical_evidence": False,
            "withdrawal_qualification": "unavailable",
        }
    finally:
        _PREVIEW.reset(token)
        state.restore()


def _failed_continuation(
    action: Any,
    request: Any,
    context: Any,
    initial: dict[str, Any],
    attempts: list[dict[str, Any]],
    *,
    unavailable: bool = False,
) -> Any:
    failed = action.failed_plan(
        request,
        context,
        message="No qualified adjacent Place in the bounded legal grasp pool.",
    )
    return replace(
        failed,
        diagnostics=replace(
            failed.diagnostics,
            failure=PlanningFailure(
                (
                    "pick_continuation_unavailable"
                    if unavailable
                    else "pick_continuation_unqualified"
                ),
                retryable=False,
            ),
            metadata={
                **failed.diagnostics.metadata,
                "gen_sim_pick_continuation": {
                    "initial": initial,
                    "attempts": attempts,
                    "physical_evidence": False,
                    "withdrawal_qualification": "unavailable",
                },
            },
        ),
    )


def recover(
    action: Any,
    request: Any,
    context: Any,
    first: Any,
    selection: Any,
    preview: Preview,
) -> Any:
    if (
        not first.plan_success.all()
        or not _validate_final_plan_velocity(first, action.robot)["valid_mask"].all()
    ):
        return first
    state = _RandomState.capture(
        _cuda_devices(action.robot.device, context.robot.qpos.device)
    )
    token = _PREVIEW.set(True)
    try:
        with robot_collision_scope():
            initial = preview_place(action, request, context, first, preview)
            if initial["valid"]:
                return first
            if initial.get("status") != "unqualified" and not initial.get(
                "unreachable_waypoints", False
            ):
                return _failed_continuation(
                    action, request, context, initial, [], unavailable=True
                )
            pool = selection.candidates
            original = next(
                v
                for v in first.expected_effects.held_object_updates.values()
                if v is not None
            ).grasp_xpos
            # Favor lower legal contacts when a tall object's tool offset exhausts reach.
            indices = sorted(
                pool.valid[0].nonzero().flatten().tolist(),
                key=lambda i: float(pool.poses[0, i, 2, 3]),
            )[:12]
            attempts = []
            for index in indices:
                pose = pool.poses[:, index].clone()
                if torch.equal(pose, original):
                    continue
                record_search("continuation_grasp_candidate")
                state.restore()
                selected = replace(
                    request,
                    goal=replace(request.goal, grasp_xpos=pose),
                    skill_options=replace(
                        request.skill_options, approach_direction=selection.direction[0]
                    ),
                )
                with measure_search("continuation_pick_plan"):
                    alternative = preview.compiler.integration.engine.plan_request(
                        selected, context
                    )
                checked = _inspect_pick_plan(
                    action.robot, selected, context, alternative
                )
                attempts.append({"candidate_id": index, "pick_safety": checked})
                if not alternative.plan_success.all() or not checked["valid"]:
                    continue
                state.restore()
                place = preview_place(
                    action, selected, context, alternative, preview, safety=True
                )
                attempts[-1]["place"] = place
                if place.get("status") == "unavailable":
                    return _failed_continuation(
                        action, request, context, initial, attempts, unavailable=True
                    )
                if place["valid"]:
                    return replace(
                        alternative,
                        diagnostics=replace(
                            alternative.diagnostics,
                            metadata={
                                **alternative.diagnostics.metadata,
                                "gen_sim_pick_continuation": {
                                    "candidate_id": index,
                                    "attempts": attempts,
                                    "place_invocation_id": preview.route.place_id,
                                    "scope": "private_pick_place_prediction_not_measured_attachment",
                                    "physical_evidence": False,
                                    "withdrawal_qualification": "unavailable",
                                },
                            },
                        ),
                    )
            # Do not dispatch a known-unqualified acquisition or broaden the legal pool.
            return _failed_continuation(action, request, context, initial, attempts)
    finally:
        _PREVIEW.reset(token)
        state.restore()
