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

"""Canonical complete alignment builds with deterministic future Place ports."""

from __future__ import annotations
from dataclasses import replace
from types import SimpleNamespace as NS
import hashlib, json, random
import numpy as np
import torch
from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    ControlPartCommandProfile,
    MotionPolicy,
    RecoveryPolicy,
    MoveHeldObjectOptions,
    MoveHeldObjectPlanningRecovery,
    HeldStagingBinding,
    RetainedCommandAuthority,
    PlaceGoal,
    PlaceOptions,
)
from embodichain.lab.sim.atomic_actions.place_continuation import PlaceContinuationRoute
from embodichain.lab.sim.scene.robot_distal import RobotDistalMeshSnapshot
from ._held_staging_fixture import staging_fixture


def continuation_fixture(directory, mode, *, enabled=False):
    gen, context, goal, generate, check, calls, checks, ik, snapshot, reads = (
        staging_fixture(directory, "original_safe")
    )
    gen._generate_once = generate
    trace = []
    engine = None
    session = None
    places = 0

    class Commands:
        robot = gen.robot

        def authority(self, request):
            if mode == "authority_missing":
                return None
            return RetainedCommandAuthority(
                preset_id="verified",
                predecessor_call_id="pick",
                predecessor_invocation_id="pick",
                entity_id="object",
                resource_id=request.binding.endpoint("primary", "motion").resource_id,
                attachment_monitor_id="builtin.composite_effect",
                attachment_monitor_revision="1",
                retention_monitor_id="builtin.composite_effect",
                retention_monitor_revision="1",
            )

        def capture(self, *args):
            raise AssertionError(
                "strict-safe fixture does not need a native retained command read"
            )

        def verify_current(self, *args):
            raise AssertionError(
                "strict-safe fixture does not need a native retained command read"
            )

    provider = NS(
        robot=gen.robot,
        binding=lambda req: HeldStagingBinding(rigid_entity_id="object"),
        capture_distal_geometry=lambda req, ctx: RobotDistalMeshSnapshot(**snapshot),
    )

    def resolve_place(current):
        trace.append(
            {
                "phase": "ground_place",
                "object_pose": current.scene.entities["object"].pose.tolist(),
                "qpos": current.robot.qpos.tolist(),
            }
        )
        held = current.get_held_object("source_arm")
        if held is None:
            held = next(iter(current.task.held_objects.values()))
        target = (
            current.scene.entities["object"].pose.reshape(1, 4, 4) @ held.object_to_eef
        )
        inv = engine.make_invocation(
            "place",
            PlaceGoal(xpos=target),
            control_parts={"primary": {"motion": "source_arm", "grasp": "source_hand"}},
            motion_policy=policy,
            skill_options=PlaceOptions(hand_interp_steps=12, lift_height=0.1),
            invocation_id="place",
        )
        return engine.resolve(inv)

    def plan_place(request, current):
        nonlocal places
        places += 1
        trace.append(
            {"phase": "plan_place", "attempt": places, "random": float(torch.rand(()))}
        )
        if (mode == "accepted" and places < 3) or mode in {
            "exhausted",
            "exception",
            "cancel",
        }:
            if mode == "exception" and places == 2:
                raise ValueError("controlled_unavailable_place")
            if mode == "cancel" and places == 2:
                raise KeyboardInterrupt()
            return engine.actions["place"].failed_plan(
                request, current, message="controlled_unqualified_place"
            )
        result = engine.plan_request(request, current)
        if mode == "release_mismatch":
            from embodichain.lab.sim.atomic_actions.effects import StateDelta

            result = replace(result, expected_effects=StateDelta())
        return result

    session = NS(
        route=PlaceContinuationRoute("staging", "place", "object"),
        validate_source=lambda ctx: trace.append({"phase": "source_monitor"}),
        resolve_place=resolve_place,
        plan=plan_place,
    )
    continuations = NS(robot=gen.robot, prepare=lambda request, current: session)
    engine = AtomicActionEngine(
        gen,
        control_profiles={
            "source_hand": ControlPartCommandProfile.joint_positions(
                open=torch.tensor([0.04]), grasp=torch.zeros(1)
            )
        },
        held_staging_provider=provider,
        retained_command_provider=Commands(),
        place_continuation_provider=continuations,
    )
    policy = MotionPolicy(
        sample_count=260,
        dynamic_collision_mode="off",
        interpolation_mode="fixed_grid",
        ik_seed_mode="sequential",
        acceleration_limit=None,
    )
    invocation = engine.make_invocation(
        "move_held_object",
        goal,
        control_parts={"primary": {"motion": "source_arm", "grasp": "source_hand"}},
        motion_policy=policy,
        skill_options=MoveHeldObjectOptions(report_retained_attachment=True),
        recovery_policy=RecoveryPolicy(
            planning_recovery=MoveHeldObjectPlanningRecovery(
                staging=True, place_continuation=enabled
            )
        ),
        invocation_id="staging",
    )
    return engine, invocation, context, session, check, calls, checks, ik, trace


def trace_record(plan, calls, checks, ik, trace):
    digest = lambda value: hashlib.sha256(
        value.detach().cpu().numpy().tobytes()
    ).hexdigest()
    return {
        "success": plan.plan_success.tolist(),
        "positions": (
            None
            if plan.joint_trajectory is None
            else digest(plan.joint_trajectory.positions)
        ),
        "dt": (
            None if plan.joint_trajectory is None else digest(plan.joint_trajectory.dt)
        ),
        "metadata": dict(plan.diagnostics.metadata),
        "failure": (
            None if plan.diagnostics.failure is None else plan.diagnostics.failure.code
        ),
        "calls": calls,
        "checks": [
            {
                "frames": len(value),
                "sha256": hashlib.sha256(
                    json.dumps(value, separators=(",", ":")).encode()
                ).hexdigest(),
            }
            for value in checks
        ],
        "ik": ik,
        "trace": trace,
        "rng": [
            digest(torch.get_rng_state()),
            float(np.random.rand()),
            random.random(),
        ],
    }
