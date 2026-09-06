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

"""Candidate materialization and observed atomic execution contracts."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from embodichain.lab.sim.atomic_actions import (
    ActionBinding,
    AtomicActionEngine,
    ControlPartCommandProfile,
    GraspGoal,
    ObjectSemantics,
    RecoveryPolicy,
    EntityState,
    SceneSnapshot,
    TrackingPolicy,
)
from embodichain.lab.sim.motion.trajectory_augmentation import (
    CandidateIdentity,
    CandidateTrajectoryBatch,
    TrajectoryPhase,
    TrajectoryTemplate,
)
from embodichain.lab.sim.atomic_actions.affordance import Affordance

CONTROL_DT = 0.05


class _Robot:
    device = torch.device("cpu")
    dof = 3
    num_instances = 2
    joint_names = ("vertical", "yaw", "finger")
    control_parts = {"arm": object(), "hand": object()}
    mimic_ids = ()
    mimic_parents = ()
    mimic_multipliers = ()
    mimic_offsets = ()

    def __init__(self):
        self.qpos = torch.zeros(2, 3)
        self.targets = self.qpos.clone()
        self.qvel = torch.zeros_like(self.qpos)
        self.velocity_targets = torch.zeros_like(self.qpos)

    def get_qpos(self, name=None, *, target=False):
        value = self.targets if target else self.qpos
        return value[:, self.get_joint_ids(name)].clone()

    def get_qvel(self, name=None, *, target=False):
        value = self.velocity_targets if target else self.qvel
        return value[:, self.get_joint_ids(name)].clone()

    def get_qf(self):
        return torch.zeros_like(self.qpos)

    def get_joint_ids(self, name=None):
        return {None: (0, 1, 2), "arm": (0, 1), "hand": (2,)}[name]

    def compute_fk(self, qpos=None, name=None, to_matrix=True):
        value = self.qpos if qpos is None else qpos
        pose = torch.eye(4).repeat(len(value), 1, 1)
        pose[:, 2, 3] = value[:, 0]
        return pose

    def compute_ik(self, **kwargs):
        raise AssertionError("Candidate execution must not recompute IK")

    def set_qpos(self, value, *, joint_ids, env_ids=None, target=True):
        assert target
        self.targets[:, joint_ids] = value

    def set_qvel(self, value, *, joint_ids, **kwargs):
        self.velocity_targets[:, joint_ids] = value

    def set_qf(self, value, **kwargs):
        pass


def _fixture(*, control_dt=CONTROL_DT):
    from embodichain.lab.trajectory_generation.integrations.atomic_runtime import (
        PickUpRuntimeSource,
    )

    robot = _Robot()
    generator = Mock(robot=robot, device=robot.device)
    generator.planner.cfg.planner_type = "fixture"
    engine = AtomicActionEngine(
        generator,
        control_profiles={
            "hand": ControlPartCommandProfile.joint_positions(
                open=torch.zeros(2, 1), grasp=torch.full((2, 1), 0.03)
            )
        },
    )
    scene = SceneSnapshot(
        timestamp=0.0,
        version=1,
        entities={"cube": EntityState(torch.eye(4).repeat(2, 1, 1))},
    )
    context = engine.initial_context(scene=scene, control_dt=control_dt)
    phases = (
        TrajectoryPhase("transit", 0, 3, allowed_operators=("joint_residual",)),
        TrajectoryPhase("approach", 3, 5, kind="contact"),
        TrajectoryPhase("close", 5, 7, kind="contact"),
        TrajectoryPhase("lift", 7, 9, kind="contact"),
        TrajectoryPhase("hold", 9, 12, kind="hold"),
    )
    qpos = torch.zeros(12, 3)
    qpos[:, 0] = torch.tensor(
        [0.0, 0.05, 0.1, 0.05, 0.0, 0.0, 0.0, 0.09, 0.18, 0.18, 0.18, 0.18]
    )
    qpos[5:, 2] = 0.03
    dt = torch.full((12,), control_dt, dtype=torch.float64)
    dt[0] = 0
    templates = tuple(
        TrajectoryTemplate(
            source_id="atomic_pickup",
            source_revision="v1",
            template_id="reference_0",
            joint_names=robot.joint_names,
            positions=qpos,
            dt=dt,
            phases=phases,
            validator_id="task_success",
            controlled_joint_indices=(0, 1),
        )
        for _ in range(2)
    )
    candidates = tuple(
        CandidateTrajectoryBatch(
            identities=(
                CandidateIdentity(
                    f"case_{i}",
                    "initial",
                    f"candidate_{i}",
                    f"geometry_{i}",
                    "atomic_pickup",
                    "v1",
                    "reference_0",
                ),
            ),
            positions=qpos[None],
            dt=templates[i].dt[None],
            valid_length=torch.tensor([12]),
            phases=(phases,),
            joint_names=robot.joint_names,
            source_row_indices=torch.tensor([i]),
        )
        for i in range(2)
    )

    def invocation(_binding):
        result = engine.make_invocation(
            "pick_up",
            GraspGoal(
                ObjectSemantics(affordance=Affordance(), geometry={}, entity_id="cube"),
                grasp_xpos=torch.eye(4).repeat(2, 1, 1),
            ),
            control_parts={"primary": {"motion": "arm", "grasp": "hand"}},
            tracking_policy=TrackingPolicy.joint_position(
                in_flight_max_abs_error=0.08, terminal_max_abs_error=0.05
            ),
            recovery_policy=RecoveryPolicy(max_replans=1, max_action_retries=0),
            invocation_id="fixture-pickup",
        )
        # A closing gripper intentionally presses against the cube. Native
        # contact verifies this endpoint; arm joints retain feedback tracking.
        return replace(
            result,
            binding=ActionBinding(
                result.binding.owner_id,
                tuple(
                    replace(e, tracking_channels={}) if e.endpoint_id == "grasp" else e
                    for e in result.binding.endpoints
                ),
            ),
        )

    source = PickUpRuntimeSource(
        engine, invocation_factory=invocation, templates=templates
    )
    return robot, engine, context, candidates, source, invocation


def test_materialize_preserves_candidate_commands_and_rebuilds_symbolic_effect():
    robot, engine, context, candidates, source, invocation = _fixture()
    qpos = candidates[0].positions.clone()
    qpos[0, 1, 1] = 0.025
    candidates = (replace(candidates[0], positions=qpos), candidates[1])
    plan = source.materialize(engine.resolve(invocation(None)), context, candidates)
    assert plan.commands.frame_count == 11
    assert plan.segment("transit").stop == 2
    assert plan.segment("hold").stop == 11
    torch.testing.assert_close(plan.joint_trajectory.positions[0], qpos[0, 1:])
    assert not context.task.held_objects
    held = plan.expected_effects.held_object_updates["arm"]
    torch.testing.assert_close(held.object_to_eef, torch.eye(4).repeat(2, 1, 1))
    assert plan.tracking is not None
    assert {s.endpoint_key for s in plan.tracking.frames[0].setpoints} == {
        ("primary", "motion")
    }
    assert plan.planned_scene_version == context.scene.version


@pytest.mark.parametrize(
    "change", ["protected", "clock", "row", "initial", "source", "phase"]
)
def test_materialize_rejects_unqualified_candidate_before_commands(change):
    _, engine, context, candidates, source, invocation = _fixture()
    candidate = candidates[0]
    if change in {"protected", "initial"}:
        q = candidate.positions.clone()
        q[0, 5 if change == "protected" else 0, 0] += 0.01
        candidate = replace(candidate, positions=q)
    elif change == "clock":
        dt = candidate.dt.clone()
        dt[0, 1] += 0.01
        candidate = replace(candidate, dt=dt)
    elif change == "row":
        candidate = replace(candidate, source_row_indices=torch.tensor([1]))
    elif change == "source":
        candidate = replace(
            candidate,
            identities=(replace(candidate.identities[0], source_revision="other"),),
        )
    elif change == "phase":
        phases = list(candidate.phases[0])
        phases[0] = replace(phases[0], allowed_operators=())
        candidate = replace(candidate, phases=(tuple(phases),))
    with pytest.raises(ValueError):
        source.materialize(
            engine.resolve(invocation(None)), context, (candidate, candidates[1])
        )


def test_materialize_tail_batch_keeps_idle_rows_ineligible():
    robot, engine, context, candidates, source, invocation = _fixture()
    plan = source.materialize(
        engine.resolve(invocation(None)), context, (candidates[0], None)
    )
    assert plan.plan_success.tolist() == [True, False]
    assert all(
        frame.active_mask.tolist() == [True, False] for frame in plan.commands.frames
    )
    assert plan.joint_trajectory.positions[1].count_nonzero() == 0


def _runtime_fixture(*, contact_passed=True, control_dt=CONTROL_DT, idle_row=False):
    from embodichain.lab.trajectory_generation.integrations.contact import (
        PickUpContactProfile,
    )
    from embodichain.lab.sim.motion.trajectory_augmentation import (
        ValidationCheck,
        ValidationResult,
    )

    robot, engine, context, candidates, source, invocation = _fixture(
        control_dt=control_dt
    )
    if idle_row:
        candidates = (candidates[0], None)
    sim = SimpleNamespace(
        simulation_time=0.0, sim_config=SimpleNamespace(physics_dt=0.005)
    )
    engine._scene_provider = SimpleNamespace(
        snapshot=lambda **kwargs: replace(context.scene, timestamp=kwargs["timestamp"])
    )
    host = SimpleNamespace(
        adapter=SimpleNamespace(robot=robot, sim=sim), env=None, assert_current=Mock()
    )
    contact = SimpleNamespace(
        robot=robot,
        sim=sim,
        profile=PickUpContactProfile(),
        observations=lambda: {
            "tcp_pose": robot.compute_fk(),
            "object_pose": robot.compute_fk(),
        },
        rollout_validation=lambda row: ValidationResult(
            (
                ValidationCheck(
                    "physical_contacts", "passed" if contact_passed else "failed"
                ),
                ValidationCheck(
                    "held_object_stability", "passed" if contact_passed else "failed"
                ),
            )
        ),
    )
    runtime = source.start(
        host, object(), candidates, contact_validator=contact, control_dt=control_dt
    )
    return robot, sim, engine, candidates, runtime


def test_runtime_executes_candidate_and_commits_effect_only_after_verification():
    robot, sim, engine, candidates, runtime = _runtime_fixture()
    assert not runtime.runner.session.task_state.held_objects
    assert runtime.runner.session.attempt_generation == 0
    for index in range(1, 12):
        expected = torch.cat([c.positions[:, index] for c in candidates])
        runtime.dispatch(expected, (True, True))
        torch.testing.assert_close(robot.targets, expected)
        robot.qpos.copy_(robot.targets)
        sim.simulation_time += CONTROL_DT
        assert not runtime.runner.session.task_state.held_objects
    runtime.finish()
    assert runtime.validation(0).accepted
    assert runtime.runner.session.task_state.held_objects["arm"].env_mask.tolist() == [
        True,
        True,
    ]
    assert runtime.metadata(0)["command_count"] == 11
    assert runtime.metadata(0)["plan_attempts"] == 1


def test_runtime_never_accepts_missing_physical_effect_evidence():
    robot, sim, engine, candidates, runtime = _runtime_fixture(contact_passed=False)
    for index in range(1, 12):
        runtime.dispatch(
            torch.cat([c.positions[:, index] for c in candidates]), (True, True)
        )
        robot.qpos.copy_(robot.targets)
        sim.simulation_time += CONTROL_DT
    with pytest.raises(RuntimeError, match="Runtime"):
        runtime.finish()
    assert not runtime.validation(0).accepted
    assert not runtime.runner.session.task_state.held_objects


def test_recovery_uses_registered_planner_but_never_dispatches_its_candidate(
    monkeypatch,
):
    from embodichain.lab.sim.atomic_actions import TimedTrajectory

    robot, sim, engine, candidates, runtime = _runtime_fixture()
    calls = []

    def fallback(request, context):
        calls.append(context.robot.qpos.clone())
        return engine.actions["pick_up"].build_plan(
            request,
            context,
            success=True,
            trajectory=TimedTrajectory.from_uniform_step(
                context.robot.qpos[:, None].repeat(1, 2, 1),
                env_ids=context.env_ids,
                step_dt=CONTROL_DT,
            ),
        )

    monkeypatch.setattr(engine.actions["pick_up"], "_plan", fallback)
    expected = torch.cat([c.positions[:, 1] for c in candidates])
    runtime.dispatch(expected, (True, True))
    robot.qpos[:, 0] = 0.5
    sim.simulation_time += CONTROL_DT
    with pytest.raises(RuntimeError, match="Runtime"):
        runtime.dispatch(
            torch.cat([c.positions[:, 2] for c in candidates]), (True, True)
        )
    assert len(calls) == 1
    assert runtime.runner.session.attempt_generation == 1
    assert not runtime.validation(0).accepted
    assert runtime.metadata(0)["command_count"] == 1
    assert runtime.metadata(0)["events"]["tracking_diverged"] == 1
    # The recovery plan cannot be sent; ordinary cancellation installs measured hold.
    torch.testing.assert_close(robot.targets, robot.qpos)


def test_runtime_rejects_unexpected_clock_advance():
    robot, sim, engine, candidates, runtime = _runtime_fixture()
    sim.simulation_time += 0.01
    with pytest.raises(RuntimeError, match="clock"):
        runtime.dispatch(
            torch.cat([c.positions[:, 1] for c in candidates]), (True, True)
        )
    assert robot.targets.count_nonzero() == 0


@pytest.mark.parametrize("period", [0.04, 0.05, 0.1, 0.15, 0.2])
def test_runtime_uses_exact_control_period(period):
    robot, sim, engine, candidates, runtime = _runtime_fixture(control_dt=period)
    for index in range(1, 12):
        runtime.dispatch(
            torch.cat([c.positions[:, index] for c in candidates]), (True, True)
        )
        robot.qpos.copy_(robot.targets)
        sim.simulation_time += period
    runtime.finish()
    assert runtime.validation(0).accepted


@pytest.mark.parametrize("idle_row", [False, True])
def test_runtime_commands_zero_velocity_independent_of_batch_occupancy(idle_row):
    robot, sim, engine, candidates, runtime = _runtime_fixture(idle_row=idle_row)
    robot.qvel.fill_(0.2)
    robot.velocity_targets.fill_(0.3)
    expected = robot.qpos.clone()
    for row, candidate in enumerate(candidates):
        if candidate is not None:
            expected[row] = candidate.positions[0, 1]
    runtime.dispatch(expected, (True, not idle_row))
    assert robot.velocity_targets.count_nonzero() == 0


@pytest.mark.parametrize(
    "mismatch", [False, True], ids=["contact_frame", "effect_error"]
)
def test_runtime_verifies_effect_in_the_motion_endpoints_tcp_frame(mismatch):
    robot, sim, engine, candidates, runtime = _runtime_fixture()
    for index in range(1, 12):
        runtime.dispatch(
            torch.cat([c.positions[:, index] for c in candidates]), (True, True)
        )
        robot.qpos.copy_(robot.targets)
        sim.simulation_time += CONTROL_DT
    # The native contact profile can use a different calibrated TCP. The
    # symbolic effect belongs to the motion endpoint's FK frame.
    native_tcp = robot.compute_fk()
    native_tcp[:, 0, 3] += 0.04
    object_pose = robot.compute_fk()
    if mismatch:
        object_pose[:, 2, 3] -= 0.1
    runtime.contact.observations = lambda: {
        "tcp_pose": native_tcp,
        "object_pose": object_pose,
    }
    if mismatch:
        with pytest.raises(RuntimeError, match="effect verification"):
            runtime.finish()
        assert not runtime.validation(0).accepted
        assert not runtime.runner.session.task_state.held_objects
    else:
        runtime.finish()
        assert runtime.validation(0).accepted
    error = runtime.metadata(0)["effect_translation_error_m"]
    assert error == pytest.approx(0.1 if mismatch else 0.0)
