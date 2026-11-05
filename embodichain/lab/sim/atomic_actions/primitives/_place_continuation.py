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

"""Private Pick-to-release forecasts with source ordering and RNG isolation."""

from __future__ import annotations
from copy import deepcopy
from dataclasses import replace
from typing import Any
import torch
from embodichain.compute.trajectory.limits import joint_velocity_validity
from embodichain.lab.sim.motion.limits import authored_joint_velocity_limits
from embodichain.lab.sim.motion.search import (
    PlanningRandomState,
    measure_search,
    record_search,
)
from embodichain.lab.sim.motion.mesh_collision import robot_collision_query_scope
from ..plans import PlanningFailure
from ._manipulation_checks import inspect_manipulation_plan
from embodichain.utils.math import pose_inv
from ..bindings import JointPositionTarget
from ..place_continuation import PlaceContinuationSession
from ._place_scene import inspect_place_plan

__all__: list[str] = []


def _validate_final_plan_velocity(plan, robot):
    trajectory = plan.joint_trajectory
    if trajectory is None:
        if plan.plan_success.any():
            raise ValueError("Initial plan requires final joint trajectory evidence.")
        return {"valid_mask": plan.plan_success.clone()}
    return {
        "valid_mask": joint_velocity_validity(
            trajectory.positions,
            trajectory.dt,
            authored_joint_velocity_limits(robot, None).to(trajectory.positions),
        )
    }


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
    return replace(
        future, scene=replace(future.scene, entities=entities), is_preview=True
    )


def _preview_place(
    action: Any,
    request: Any,
    context: Any,
    plan: Any,
    preview: PlaceContinuationSession,
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
    preview.validate_source(context)
    future = _project_held(action, request, context, plan, held)
    transport_receipt = None
    if preview.route.transport_invocation_id is not None:
        transport_request = preview.resolve_transport(future)
        if not _same_resources(request, transport_request):
            return {
                "valid": False,
                "status": "unavailable",
                "reason": "transport_resource_identity_mismatch",
            }
        original_transform = held.object_to_eef.clone()
        with measure_search("transport_preview"):
            moved = preview.plan(transport_request, future)
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
            moved.invocation_id != preview.route.transport_invocation_id
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
            "invocation_id": preview.route.transport_invocation_id,
            "scope": "canonical_transport_prediction_not_measured_retention",
            "path_qualification": "not_checked",
        }
        future = _project_held(action, transport_request, future, moved, retained)
    placed_request = preview.resolve_place(future)
    if not _same_resources(request, placed_request):
        return {
            "valid": False,
            "status": "unavailable",
            "reason": "continuation_resource_identity_mismatch",
        }
    placed = preview.plan(placed_request, future)
    if not placed.plan_success.all():
        diagnostics = placed.diagnostics
        keyframes = diagnostics.metadata.get("gen_sim_place_keyframe_preflight", {})
        probe = diagnostics.metadata.get("gen_sim_place_waypoint_probe", {})
        qualified = keyframes.get("valid", probe.get("kinematic_valid"))
        return {
            "valid": False,
            "status": "unqualified",
            "reason": "place_planning_failed",
            "unreachable_waypoints": None if qualified is None else not qualified,
            "place_diagnostics": {
                "backend": diagnostics.backend,
                "messages": list(diagnostics.messages),
                "metadata": deepcopy(dict(diagnostics.metadata)),
                "failure": (
                    None
                    if diagnostics.failure is None
                    else {
                        "code": diagnostics.failure.code,
                        "retryable": diagnostics.failure.retryable,
                    }
                ),
            },
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
        placed.invocation_id != preview.route.place_invocation_id
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
        action,
        placed_request,
        future,
        placed,
        preview.place_scene_inputs(placed_request, future),
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
    preview: PlaceContinuationSession,
    *,
    safety: bool = False,
) -> dict[str, Any]:
    """Predict privately without consuming the caller's RNG or physical state."""
    state = PlanningRandomState.capture(
        (action.robot.device, context.robot.qpos.device)
    )
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
        state.restore()


def prepare_pick_continuation(action, request, context):
    provider = action.planning_services.place_continuation_provider
    if provider is None:
        return None
    session = provider.prepare(request, context)
    recovery = request.recovery_policy.planning_recovery
    if session is None and recovery is not None and recovery.place_continuation:
        raise ValueError("Declared Pick continuation occurrence is unavailable.")
    if session is not None and (
        not isinstance(session, PlaceContinuationSession)
        or session.route.source_invocation_id != request.invocation_id
        or session.route.object_id != request.goal.semantics.entity_id
    ):
        raise ValueError(
            "Pick continuation session does not match its declared occurrence/object."
        )
    return session


def eligible(request, context, selection, preview):
    options = request.skill_options
    return bool(
        preview is not None
        and not context.is_preview
        and context.batch_size == 1
        and context.env_ids.tolist() == [0]
        and not options.linear_segments
        and options.adaptive_grasp not in {"pour", "handover_source"}
        and request.goal.grasp_xpos is None
        and options.fixed_object_to_eef is None
        and options.rotate_upright is None
        and options.obj_upright_direction is None
        and options.pick_object_part == "center"
        and not options.constrained_grasp
        and selection.candidates is not None
        and selection.direction is not None
        and request.motion_policy.strategy == "ik_interp"
        and request.motion_policy.dynamic_collision_mode == "off"
    )


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
    preview: PlaceContinuationSession,
) -> Any:
    if (
        not first.plan_success.all()
        or not _validate_final_plan_velocity(first, action.robot)["valid_mask"].all()
    ):
        return first
    if not preview.route.requires_full_preview:
        return replace(
            first,
            diagnostics=replace(
                first.diagnostics,
                metadata={
                    **first.diagnostics.metadata,
                    "gen_sim_pick_continuation_deferred": {
                        "place_invocation_id": preview.route.place_invocation_id,
                        "scope": "plan_release_from_observed_attachment_not_predicted_state",
                        "physical_evidence": False,
                    },
                },
            ),
        )
    state = PlanningRandomState.capture(
        (action.robot.device, context.robot.qpos.device)
    )
    try:
        with robot_collision_query_scope():
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
                    alternative = preview.plan(
                        selected, replace(context, is_preview=True)
                    )
                checked = inspect_manipulation_plan(
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
                                    "place_invocation_id": preview.route.place_invocation_id,
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
        state.restore()
