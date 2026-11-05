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

"""Frozen canonical request/plan facts for prospective attachment boundaries."""

from __future__ import annotations
from dataclasses import replace
from types import SimpleNamespace as NS
import torch
from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    GraspGoal,
    PickUpOptions,
    ControlPartCommandProfile,
    MotionPolicy,
)
from embodichain.lab.sim.atomic_actions.effects import StateDelta
from embodichain.lab.sim.atomic_actions.place_continuation import PlaceContinuationRoute
from ._ordinary_pick_fixture import ordinary_fixture, OrdinaryCandidates


def preview_fixture(directory, mode="success", transport=False):
    generator, context, semantics, calls, ik = ordinary_fixture(directory)
    engine = AtomicActionEngine(
        generator,
        control_profiles={
            "hand": ControlPartCommandProfile.joint_positions(
                open=torch.full((2,), 0.04), grasp=torch.zeros(2)
            )
        },
        grasp_pose_generators={"hand": OrdinaryCandidates()},
    )
    invocation = engine.make_invocation(
        "pick_up",
        GraspGoal(semantics=semantics),
        control_parts={"primary": {"motion": "arm", "grasp": "hand"}},
        motion_policy=MotionPolicy(
            sample_count=20,
            dynamic_collision_mode="off",
            interpolation_mode="fixed_grid",
            acceleration_limit=None,
        ),
        skill_options=PickUpOptions(hand_interp_steps=12),
        invocation_id="pick",
    )
    request = engine.resolve(invocation)
    plan = engine.plan(invocation, context)
    assert plan.plan_success.all()
    key = request.binding.endpoint("primary", "motion").task_state_key
    held = plan.expected_effects.held_object_updates[key]
    route = PlaceContinuationRoute(
        "pick", "place", semantics.entity_id, "transport" if transport else None
    )
    trace = []
    place_request = replace(request, invocation_id="place", skill_id="place")
    transport_request = replace(
        request, invocation_id="transport", skill_id="move_held_object"
    )

    def snapshot(label, ctx):
        trace.append(
            {
                "step": label,
                "qpos": ctx.robot.qpos.tolist(),
                "object_pose": ctx.scene.entities[semantics.entity_id].pose.tolist(),
                "held": ctx.get_held_object(key) is not None,
            }
        )

    def resolve(label, ctx):
        snapshot(label, ctx)
        torch.rand(())
        if mode == "monitor_missing":
            raise ValueError("canonical_effect_monitor_unavailable")
        return place_request if label == "place" else transport_request

    def planning(req, ctx):
        snapshot("plan_" + req.invocation_id, ctx)
        torch.rand(())
        positions = ctx.robot.qpos[:, None].expand(-1, 3, -1).clone()
        positions[:, 1:, 0] += 0.01
        positions[:, :, 3:] = 0.0
        effects = StateDelta()
        if req.invocation_id == "place":
            positions[:, -1, 3:] = 0.04
            effects = StateDelta(held_object_updates={key: None})
            if mode == "release_closed":
                positions[:, -1, 3:] = 0.0
            if mode == "release_effect_missing":
                effects = StateDelta()
        if req.invocation_id == "transport" and mode == "transport_opens":
            positions[:, 1, 3:] = 0.04
        if mode == "velocity":
            positions[:, 1, 0] += 100.0
        dt = torch.tensor([[0.0, 1.0, 1.0]])
        trajectory = replace(
            plan.joint_trajectory,
            positions=positions,
            dt=dt,
            velocities=None,
            accelerations=None,
        )
        # Forecast checker consumes the canonical emitted trajectory/effect facts.
        return NS(
            invocation_id=req.invocation_id,
            skill_id=req.skill_id,
            plan_success=torch.tensor([mode != "planning_failed"]),
            joint_trajectory=trajectory,
            expected_effects=effects,
            diagnostics=plan.diagnostics,
        )

    session = NS(
        route=route,
        validate_source=lambda ctx: snapshot("source", ctx),
        resolve_place=lambda ctx: resolve("place", ctx),
        resolve_transport=lambda ctx: resolve("transport", ctx),
        plan=planning,
        place_scene_inputs=lambda req, ctx: None,
    )
    if mode == "attachment_missing":
        plan = replace(plan, expected_effects=StateDelta())
    return engine.actions["pick_up"], request, context, plan, session, trace


def recovery_fixture(directory, mode):
    action, request, context, plan, session, _ = preview_fixture(directory)
    trace = []
    held = next(iter(plan.expected_effects.held_object_updates.values()))
    poses = held.grasp_xpos[:, None].expand(-1, 15, -1, -1).clone()
    poses[0, :, 2, 3] += torch.tensor(
        [
            0.0,
            0.02,
            -0.03,
            0.01,
            -0.02,
            0.03,
            -0.01,
            0.04,
            -0.04,
            0.05,
            -0.05,
            0.06,
            -0.06,
            0.07,
            -0.07,
        ]
    )
    selection = NS(
        candidates=NS(poses=poses, valid=torch.ones(1, 15, dtype=torch.bool)),
        direction=torch.tensor([[0.0, 0.0, -1.0]]),
    )
    session.route = replace(session.route, requires_full_preview=mode != "deferred")
    counts = {"plan": 0, "preview": 0}

    def planning(selected, current):
        counts["plan"] += 1
        trace.append(
            {
                "plan_z": selected.goal.grasp_xpos[0, 2, 3].item(),
                "random": torch.rand(()).item(),
            }
        )
        return plan

    session.plan = planning

    def forecast(*args, safety=False):
        counts["preview"] += 1
        trace.append(
            {
                "preview": counts["preview"],
                "safety": safety,
                "random": torch.rand(()).item(),
            }
        )
        if mode == "initial_valid":
            return {"valid": True}
        if mode == "initial_unavailable":
            return {"valid": False, "status": "unavailable"}
        if safety and mode == "later_unavailable":
            return {"valid": False, "status": "unavailable"}
        return {
            "valid": bool(safety and mode == "accepted" and counts["plan"] == 2),
            "status": "unqualified",
        }

    def inspect(*args):
        return {
            "valid": mode != "pick_rejected",
            "scope": "controlled_full_command_check",
        }

    return action, request, context, plan, selection, session, trace, forecast, inspect
