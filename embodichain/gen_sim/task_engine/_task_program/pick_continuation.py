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

__all__: list[str] = []


@dataclass(frozen=True, slots=True)
class Continuation:
    pick_id: str
    place_id: str
    pick_call: Pick
    place_call: Place


@dataclass(frozen=True, slots=True)
class Preview:
    route: Continuation
    compiler: Any
    pick_workflow: Any
    place_workflow: Any


def bind_calls(program: Any, preset: Any) -> tuple[Continuation, ...]:
    if preset is None or preset.effect_assurance is not EffectAssurance.VERIFIED:
        return ()
    result = []
    segments = tuple(program.iter_segments())
    for current, following in zip(segments, segments[1:]):
        if any(
            len(s.calls) != 1 or s.parallel_block is not None
            for s in (current, following)
        ):
            continue
        pick, place = current.calls[0].call, following.calls[0].call
        if (
            type(pick) is not Pick
            or type(place) is not Place
            or pick.object != place.object
            or dict(pick.resources) != dict(place.resources)
            or set(pick.resources) != {"primary"}
            or place.inside is None
            or any(k not in preset.effect_monitors for k in ("pick", "place"))
        ):
            continue

        def identity(segment: Any) -> str:
            return f"{program.program_id}/{segment.segment_id}:{segment.calls[0].segment_call_index}"

        result.append(Continuation(identity(current), identity(following), pick, place))
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


def preview_place(
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
        return {"valid": False}
    pick = preview.compiler.ground(preview.pick_workflow, 0, context)
    _require_monitor(pick, SemanticEffectKind.ATTACH, preview.route.pick_id, uid, key)
    future = context.project(
        qpos=plan.joint_trajectory.positions[:, -1],
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
    entities[uid] = replace(entities[uid], pose=pose)
    future = replace(future, scene=replace(future.scene, entities=entities))
    grounded = preview.compiler.ground(preview.place_workflow, 0, future)
    _require_monitor(
        grounded, SemanticEffectKind.RELEASE, preview.route.place_id, uid, key
    )
    engine = preview.compiler.integration.engine
    placed_request = engine.resolve(grounded.invocation)
    for name in ("motion", "grasp"):
        before, after = (
            r.binding.endpoint("primary", name) for r in (request, placed_request)
        )
        if (
            before.task_state_key != after.task_state_key
            or before.target != after.target
        ):
            return {"valid": False}
    placed = engine.plan(grounded.invocation, future)
    if not placed.plan_success.all():
        probe = placed.diagnostics.metadata.get("gen_sim_place_waypoint_probe", {})
        return {
            "valid": False,
            "unreachable_waypoints": probe.get("kinematic_valid") is False,
        }
    if not _validate_final_plan_velocity(placed, action.robot)["valid_mask"].all():
        return {"valid": False}
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
        return {"valid": False}
    return (
        inspect_place_plan(engine.actions["place"], placed_request, future, placed)
        if safety
        else {"valid": True}
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
            if initial["valid"] or not initial.get("unreachable_waypoints", False):
                return first
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
                state.restore()
                selected = replace(
                    request,
                    goal=replace(request.goal, grasp_xpos=pose),
                    skill_options=replace(
                        request.skill_options, approach_direction=selection.direction[0]
                    ),
                )
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
                                },
                            },
                        ),
                    )
            # Do not dispatch a known-unqualified acquisition or broaden the legal pool.
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
                        "pick_continuation_unqualified", retryable=False
                    ),
                    metadata={
                        **failed.diagnostics.metadata,
                        "gen_sim_pick_continuation": {"attempts": attempts},
                    },
                ),
            )
    finally:
        _PREVIEW.reset(token)
        state.restore()
