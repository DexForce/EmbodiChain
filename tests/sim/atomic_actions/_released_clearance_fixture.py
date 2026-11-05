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

"""Complete released cleanup/park plans with controlled collision facts."""

from __future__ import annotations
from dataclasses import replace
from types import SimpleNamespace as NS
import hashlib, random
import numpy as np
import torch
from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    ControlPartCommandProfile,
    TaskState,
    RecoveryPolicy,
    MotionPolicy,
    MoveEndEffectorOptions,
    EndEffectorPoseGoal,
    JointPositionGoal,
)
from embodichain.lab.sim.atomic_actions.released_continuation import (
    ReleasedHandClearanceRecovery,
    ReleasedHandParkRecovery,
)
from embodichain.lab.sim.motion.mesh_collision import MotionCollisionReport
from ._handover_native_fixture import native_transfer_fixture


def fixture(directory, mode, *, enabled=False, two_parks=False):
    gen, context, *_ = native_transfer_fixture(directory, "clear")
    context = replace(context, task=TaskState.empty(1, "cpu"))
    calls = []
    raw = gen._generate_once

    def generate(targets, options=None):
        calls.append(
            dict(
                control=options.control_part,
                count=options.sample_count,
                targets=[
                    (state.xpos if state.xpos is not None else state.qpos).tolist()
                    for state in targets
                ],
                start=options.start_qpos.tolist(),
            )
        )
        return raw(targets, options=options)

    gen._generate_once = generate

    class Query:
        robot = gen.robot

        def __init__(self):
            self.reads = []

        def inspect(self, positions, *, control_part):
            torch.rand(())
            np.random.rand()
            random.random()
            self.reads.append(
                dict(
                    part=control_part,
                    frames=len(positions),
                    sha256=hashlib.sha256(positions.numpy().tobytes()).hexdigest(),
                )
            )
            if mode == "unavailable":
                raise ValueError("source collision unavailable")
            if mode == "cancel":
                raise KeyboardInterrupt()
            first = len(self.reads) == 1
            # A Park starts above its observed cleanup position, while cleanup
            # starts at the exact source seed. Keep original query order visible.
            park = (
                bool(float(positions[0, 2]) > 0.41) or control_part == "destination_arm"
            )
            bad = mode in {"retreat", "exhausted", "initial"} and (
                first or mode == "exhausted"
            )
            if mode == "height" and park:
                bad = float(positions[0, 2]) < 0.60
            if mode == "secondpark" and control_part == "destination_arm":
                bad = float(positions[0, 2]) < 0.60
            if bad:
                return MotionCollisionReport(
                    False,
                    frame=(
                        0 if park or mode == "initial" else min(10, len(positions) - 1)
                    ),
                    kind="scene",
                    pairs=(("arm", "fixture"),),
                )
            return MotionCollisionReport(True, frames=len(positions))

    query = Query()

    class Provider:
        robot = gen.robot

        def prepare(self, request, current):
            if request.invocation_id != "cleanup":
                return None
            return self.session

        def require_park(self, request, current):
            if request.invocation_id not in {"park0", "park1"}:
                raise ValueError("foreign Park")

    provider = Provider()
    profile = {
        part: ControlPartCommandProfile.joint_positions(
            park=gen.robot.get_qpos(name=part) + 0.01
        )
        for part in ("source_arm", "destination_arm")
    }
    engine = AtomicActionEngine(
        gen,
        control_profiles=profile,
        free_motion_query=query,
        released_hand_provider=provider,
    )
    motion = MotionPolicy(
        sample_count=45,
        interpolation_mode="fixed_grid",
        ik_seed_mode="sequential",
        acceleration_limit=None,
        dynamic_collision_mode="off",
        segment_sample_count_retries=(260, 320),
    )
    poses = torch.eye(4)[None, None].repeat(1, 2, 1, 1)
    poses[0, 0, :3, 3] = torch.tensor([-0.3, 0.0, 0.55])
    poses[0, 1, :3, 3] = torch.tensor([-0.15, 0.0, 0.55])
    invocation = engine.make_invocation(
        "move_end_effector",
        EndEffectorPoseGoal(poses),
        control_parts={"primary": {"motion": "source_arm"}},
        skill_options=MoveEndEffectorOptions(linear_segments=True),
        motion_policy=motion,
        recovery_policy=(
            RecoveryPolicy(planning_recovery=ReleasedHandClearanceRecovery())
            if enabled
            else RecoveryPolicy()
        ),
        invocation_id="cleanup",
    )
    parks = tuple(
        engine.resolve(
            engine.make_invocation(
                "move_joints",
                JointPositionGoal("park"),
                control_parts={"primary": {"motion": part}},
                motion_policy=replace(motion, segment_sample_count_retries=()),
                recovery_policy=(
                    RecoveryPolicy(planning_recovery=ReleasedHandParkRecovery())
                    if enabled
                    else RecoveryPolicy()
                ),
                invocation_id=f"park{i}",
            )
        )
        for i, part in enumerate(
            ("source_arm", "destination_arm") if two_parks else ("source_arm",)
        )
    )

    class Session:
        park_bindings = tuple(request.binding for request in parks)

        def resolve_park(self, index, current):
            return parks[index]

        def plan_park(self, request, current):
            return engine.plan_request(request, current)

    session = Session()
    provider.session = session
    return engine, invocation, context, session, query, calls


def record(plan, query, calls):
    digest = lambda value: hashlib.sha256(value.numpy().tobytes()).hexdigest()
    return dict(
        success=plan.plan_success.tolist(),
        positions=digest(plan.joint_trajectory.positions),
        dt=digest(plan.joint_trajectory.dt),
        commands=plan.commands.frame_count,
        segments=[[s.name, s.start, s.stop] for s in plan.segments],
        recovery=dict(plan.diagnostics.metadata).get("gen_sim_clearance_recovery"),
        calls=calls,
        queries=query.reads,
        rng=[digest(torch.get_rng_state()), float(np.random.rand()), random.random()],
    )
