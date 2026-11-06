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

"""Bounded staged paths retain source command/geometry/RNG boundaries."""

from __future__ import annotations
from dataclasses import replace
from types import SimpleNamespace as NS
from pathlib import Path
import hashlib, json, random
import numpy as np
import pytest
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
)
from embodichain.lab.sim.atomic_actions.primitives import _held_staging
from embodichain.lab.sim.evidence import PhysicalObservationStamp
from embodichain.lab.sim.sensors.command_sampling import (
    capture_robot_commands,
    verify_robot_commands,
)
from embodichain.lab.sim.scene.robot_distal import RobotDistalMeshSnapshot
from embodichain.lab.sim.motion import mesh_collision
from embodichain.lab.sim.motion.mesh_collision import (
    RobotArticulationCollisionSnapshot,
    RobotMeshCollisionSnapshot,
)
from ._held_staging_fixture import staging_fixture


class Commands:
    def __init__(self, robot, stamp, mode):
        self.robot, self.stamp, self.mode = robot, stamp, mode

    def authority(self, request):
        if self.mode == "authority_missing":
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

    def capture(self, request, context):
        actual = (
            context.observed_robot
            if context.observed_robot is not None
            else context.robot
        )
        return capture_robot_commands(
            self.robot,
            expected_qpos=actual.qpos,
            expected_qvel=actual.qvel,
            timestamp=actual.timestamp,
            source_stamp=context.world_source_stamp,
            read_source_stamp=lambda: self.stamp,
        )

    def verify_current(self, observation):
        verify_robot_commands(
            self.robot, observation, read_source_stamp=lambda: self.stamp
        )


def inputs(
    tmp_path,
    monkeypatch,
    mode,
    *,
    missing_articulation=False,
    wrong_articulation=False,
    linear=False,
):
    gen, context, goal, generate, check, calls, checks, ik, snapshot, reads = (
        staging_fixture(tmp_path, mode)
    )
    stamp = PhysicalObservationStamp(
        source_id="staging-fixture",
        source_kind="default.contacts",
        revision=11,
        simulation_time=0.0,
    )
    context = replace(context, world_source_stamp=stamp)
    gen._generate_once = generate

    def factory(*args, **kwargs):
        if kwargs["include_scene"]:
            assert kwargs["include_rigids"] is False
            return lambda positions: {
                **check(positions),
                "scope": "discrete_robot_articulation_meshes",
            }
        return check

    monkeypatch.setattr(mesh_collision, "_make_checker", factory)

    class Staging:
        def __init__(self):
            self.robot = gen.robot
            self.queries = []
            self.queries_at = []
            self.geometry_calls = 0

        def binding(self, request):
            return HeldStagingBinding(
                rigid_entity_id="object", articulation_scene=mode == "articulation"
            )

        def capture_distal_geometry(self, request, current):
            self.geometry_calls += 1
            return RobotDistalMeshSnapshot(**snapshot)

        def prepare_articulation_query(self, request, current):
            self.queries_at.append(len(calls))
            query = (
                RobotMeshCollisionSnapshot(self.robot, control_part="source_arm")
                if wrong_articulation
                else RobotArticulationCollisionSnapshot(
                    self.robot, NS(num_envs=1), control_part="source_arm"
                )
            )
            self.queries.append(query)
            return query

    staging_provider = Staging()
    engine = AtomicActionEngine(
        gen,
        control_profiles={
            "source_hand": ControlPartCommandProfile.joint_positions(
                open=torch.tensor([0.04]), grasp=torch.zeros(1)
            )
        },
        held_staging_provider=(
            NS(
                robot=gen.robot,
                binding=staging_provider.binding,
                capture_distal_geometry=staging_provider.capture_distal_geometry,
            )
            if missing_articulation
            else staging_provider
        ),
        retained_command_provider=Commands(gen.robot, stamp, mode),
    )
    invocation = engine.make_invocation(
        "move_held_object",
        goal,
        control_parts={"primary": {"motion": "source_arm", "grasp": "source_hand"}},
        motion_policy=MotionPolicy(
            sample_count=45,
            dynamic_collision_mode="off",
            interpolation_mode="fixed_grid",
            ik_seed_mode="sequential",
            acceleration_limit=None,
            segment_sample_count_retries=(260, 320) if linear else (),
        ),
        skill_options=MoveHeldObjectOptions(
            report_retained_attachment=True, linear_segments=linear
        ),
        recovery_policy=RecoveryPolicy(
            planning_recovery=MoveHeldObjectPlanningRecovery(staging=True)
        ),
        invocation_id="staging",
    )
    return engine, invocation, context, calls, checks, ik, reads, staging_provider


@pytest.mark.parametrize(
    "mode",
    [
        "accept",
        "cartesian_accept",
        "retained",
        "second",
        "all_reject",
        "target_changed",
        "during_check",
        "velocity",
        "authority_missing",
        "original_safe",
        "articulation",
    ],
)
def test_staging_matches_source_path_order_retained_commands_and_rng(
    tmp_path, monkeypatch, mode
):
    case = mode.removeprefix("cartesian_")
    engine, invocation, context, calls, checks, ik, reads, staging_provider = inputs(
        tmp_path, monkeypatch, case, linear=case != mode
    )
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    plan = engine.plan(invocation, context)
    digest = lambda t: hashlib.sha256(t.numpy().tobytes()).hexdigest()
    hashed = lambda v: hashlib.sha256(
        json.dumps(v, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    actual = {
        "mode": case,
        "success": plan.plan_success.tolist(),
        "positions_sha256": digest(plan.joint_trajectory.positions),
        "dt_sha256": digest(plan.joint_trajectory.dt),
        "segments": [[s.name, s.start, s.stop] for s in plan.segments],
        "metadata": dict(plan.diagnostics.metadata),
        "failure": (
            None if plan.diagnostics.failure is None else plan.diagnostics.failure.code
        ),
        "calls_sha256": hashed(calls),
        "checks_sha256": hashed(checks),
        "ik_sha256": hashed(ik),
        "reads": reads,
        "held_keys": list(plan.expected_effects.held_object_updates),
        "rng": [
            digest(torch.get_rng_state()),
            float(np.random.rand()),
            random.random(),
        ],
    }
    expected = json.loads(
        (
            Path(__file__).parent
            / (
                f"data/held_cartesian_staging_{case}_reference.json"
                if case != mode
                else f"data/held_staging_{case}_reference.json"
            )
        ).read_text()
    )
    assert actual == expected
    assert staging_provider.geometry_calls == (0 if mode == "original_safe" else 1)
    if mode == "articulation":
        assert len(staging_provider.queries) == 2
        assert staging_provider.queries_at == [0, 8]
        assert (
            plan.diagnostics.metadata["held_alignment"]["final_command_check"][
                "collision"
            ]["scope"]
            == "discrete_robot_meshes"
        )
        for query in staging_provider.queries:
            with pytest.raises(RuntimeError, match="closed"):
                query.inspect(context.robot.qpos)
    else:
        assert not staging_provider.queries


def test_transformed_retained_velocity_is_rechecked_without_replanning(
    tmp_path, monkeypatch
):
    engine, invocation, context, calls, checks, ik, reads, provider = inputs(
        tmp_path, monkeypatch, "retained"
    )

    def transform(request, current, plan):
        velocities = plan.joint_trajectory.velocities.clone()
        velocities[:, :, 12] = 0.01
        frames = tuple(
            replace(
                frame,
                commands=tuple(
                    (
                        replace(
                            command,
                            payload=replace(
                                command.payload,
                                velocities=torch.full_like(
                                    command.payload.positions, 0.01
                                ),
                            ),
                        )
                        if command.target.control_part == "source_hand"
                        else command
                    )
                    for command in frame.commands
                ),
            )
            for frame in plan.commands.frames
        )
        return replace(
            plan,
            joint_trajectory=replace(plan.joint_trajectory, velocities=velocities),
            commands=replace(plan.commands, frames=frames),
        )

    plan = engine.plan(invocation, context, plan_transform=transform)
    assert len(calls) == 9
    assert not plan.plan_success.any()
    assert plan.diagnostics.failure.code == "held_staging_safety_failed"
    checked = plan.diagnostics.metadata["final_qualification"]["staging_recovery"][
        "post_transform_check"
    ]
    assert checked["reason"] == "retained_hand_target_velocity_changed"
    assert all(not frame.active_mask.any() for frame in plan.commands.frames)


def test_declared_articulation_scope_never_falls_back_to_robot_only(
    tmp_path, monkeypatch
):
    engine, invocation, context, calls, checks, ik, reads, provider = inputs(
        tmp_path, monkeypatch, "articulation", missing_articulation=True
    )
    plan = engine.plan(invocation, context)
    assert not plan.plan_success.any()
    assert plan.diagnostics.failure.code == "held_alignment_collision_unavailable"
    assert not calls and not checks and not reads and not provider.queries


def test_wrong_owned_articulation_query_is_closed_before_rejection(
    tmp_path, monkeypatch
):
    engine, invocation, context, calls, checks, ik, reads, provider = inputs(
        tmp_path, monkeypatch, "articulation", wrong_articulation=True
    )
    with pytest.raises(TypeError, match="RobotArticulationCollisionSnapshot"):
        engine.plan(invocation, context)
    assert not calls and not checks and not reads
    assert len(provider.queries) == 1
    with pytest.raises(RuntimeError, match="closed"):
        provider.queries[0].inspect(context.robot.qpos)
