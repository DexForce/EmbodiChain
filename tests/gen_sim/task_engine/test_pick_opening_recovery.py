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

"""Opening a free hand is timed motion, not an instantaneous Pick boundary."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import pickle
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from embodichain.compute.trajectory import differentiate_positions
from embodichain.gen_sim.task_engine._task_program import (
    actions,
    articulation_collision,
    motion,
    pick_safety,
)
from embodichain.lab.sim.atomic_actions import (
    ActionPlan,
    AntipodalAffordance,
    JointPositionTarget,
    MotionPolicy,
    ObjectSemantics,
    PlannerDiagnostics,
    PlanningFailure,
    RecoveryPolicy,
    StateDelta,
    TimedTrajectory,
    TrackingPolicy,
    TrajectorySegment,
)
from embodichain.lab.sim.atomic_actions.primitives.pick_up import (
    GraspGoal,
    PickUp,
    PickUpOptions,
)
from embodichain.lab.sim.atomic_actions.state import HeldObjectState, TaskState


@dataclass(frozen=True)
class _Request:
    goal: GraspGoal
    skill_options: PickUpOptions
    motion_policy: MotionPolicy
    binding: object
    skill_id: str = "pick_up"
    tracking_policy: TrackingPolicy = field(default_factory=TrackingPolicy.timed)


@pytest.fixture
def opening_inputs(monkeypatch):
    qpos = torch.tensor([[0.0, 0.1, 0.7, -0.7, 0.3, 0.4]])
    arm, hand = JointPositionTarget("arm", (0, 1)), JointPositionTarget("hand", (2, 3))
    endpoints = tuple(
        SimpleNamespace(
            task_state_key="active",
            require_target=lambda _, target=target: target,
            joint_positions=lambda key, **kwargs: (
                torch.tensor([[0.0, 0.0]])
                if key == "open"
                else torch.tensor([[0.02, -0.02]])
            ),
        )
        for target in (arm, hand)
    )
    affordance = AntipodalAffordance(
        mesh_vertices=torch.tensor(
            [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]]
        ),
        mesh_triangles=torch.tensor([[0, 1, 2]]),
    )
    affordance.set_custom_config(pick_safety._PICK_GRASP_RULE, ("item", "target"))
    semantics = ObjectSemantics(affordance=affordance, geometry={}, entity_id="item")
    request = _Request(
        GraspGoal(semantics),
        PickUpOptions(hand_interp_steps=12),
        MotionPolicy(
            strategy="ik_interp", dynamic_collision_mode="off", sample_count=260
        ),
        SimpleNamespace(
            endpoint=lambda _, name: endpoints[name == "grasp"], endpoints=endpoints
        ),
    )
    context = SimpleNamespace(
        batch_size=1,
        env_ids=torch.tensor([0]),
        control_dt=0.04,
        require_control_dt=lambda: 0.04,
        robot=SimpleNamespace(qpos=qpos, qvel=torch.zeros_like(qpos)),
        task=TaskState(1, "cpu"),
    )
    robot = SimpleNamespace(
        device=torch.device("cpu"),
        get_joint_ids=lambda name: (0, 1) if name == "arm" else (2, 3),
        get_qpos_limits=lambda name, env_ids=None: torch.tensor(
            [[[-1.0, 1.0], [-1.0, 1.0]]]
        ),
    )
    action = actions.GenSimPickUp()
    action._planning_services = SimpleNamespace(
        device=robot.device,
        robot=robot,
        motion_generator=SimpleNamespace(direct_transport_planning=False),
    )
    monkeypatch.setattr(
        motion, "_joint_velocity_limits", lambda *args: torch.full((1, 6), 2.0)
    )
    collision = []

    def inspect(robot, positions, part):
        collision.append(positions.clone())
        return {"valid": True}

    monkeypatch.setattr(articulation_collision, "check_robot_motion", inspect)
    positions = qpos[:, None].repeat(1, 6, 1)
    positions[:, :, 0] += torch.arange(6) * 0.01
    positions[:, :, 2:4] = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.01, -0.01],
            [0.02, -0.02],
            [0.02, -0.02],
            [0.02, -0.02],
        ]
    )
    trajectory = TimedTrajectory.from_uniform_step(
        positions, env_ids=context.env_ids, step_dt=context.control_dt
    )
    velocities = differentiate_positions(positions, trajectory.dt)
    velocities[:, -1] = 0
    trajectory = replace(trajectory, velocities=velocities)
    held = HeldObjectState(semantics, torch.eye(4)[None], torch.eye(4)[None])
    plan = ActionPlan(
        skill_id="pick_up",
        plan_success=torch.tensor([True]),
        commands=action._joint_command_sequence(
            request, trajectory, active_mask=torch.tensor([True])
        ),
        joint_trajectory=trajectory,
        recovery_policy=RecoveryPolicy(),
        tracking_policy=TrackingPolicy.timed(),
        planned_scene_version=1,
        planned_collision_world_revision=(0,),
        expected_effects=StateDelta(held_object_updates={"active": held}),
        effect_candidates=StateDelta(held_object_updates={"active": held}),
        segments=tuple(
            TrajectorySegment(name, start, stop)
            for name, start, stop in (
                ("approach", 0, 2),
                ("close", 2, 4),
                ("lift", 4, 6),
            )
        ),
        scene_dependencies=frozenset({"item"}),
        scene_dependency_monitor_until={"item": 2},
        diagnostics=PlannerDiagnostics(
            backend="fixture",
            metadata={"affordance_sample": {"grasp": {"candidate_ids": [0]}}},
        ),
    )
    return SimpleNamespace(
        action=action,
        request=request,
        context=context,
        robot=robot,
        plan=plan,
        collision=collision,
    )


def test_observed_closed_hand_gets_bounded_real_opening_prefix(opening_inputs):
    probe = opening_inputs
    assert pick_safety._inspect_pick_plan(
        probe.robot, probe.request, probe.context, probe.plan
    ) == {"valid": False, "reason": "initial_velocity_rejected"}
    candidate, report = pick_safety._prepend_pick_opening(
        probe.action, probe.request, probe.context, probe.plan
    )
    assert candidate is not None and report["accepted"]
    assert report["intervals"] == 9 and report["prefix_frame_count"] == 10
    assert report["duration"] == pytest.approx(0.36)
    assert torch.equal(
        candidate.joint_trajectory.positions[:, 0], probe.context.robot.qpos
    )
    assert torch.equal(
        candidate.joint_trajectory.positions[:, 10:],
        probe.plan.joint_trajectory.positions,
    )
    assert torch.equal(
        candidate.joint_trajectory.velocities[:, 10:],
        probe.plan.joint_trajectory.velocities,
    )
    assert torch.equal(
        candidate.joint_trajectory.dt[:, 10:], probe.plan.joint_trajectory.dt
    )
    assert candidate.scene_dependency_monitor_until["item"] == 12
    assert [(s.name, s.start, s.stop) for s in candidate.segments] == [
        ("approach", 0, 12),
        ("close", 12, 14),
        ("lift", 14, 16),
    ]
    assert pick_safety._inspect_pick_plan(
        probe.robot, probe.request, probe.context, candidate
    )["valid"]


def test_safe_original_keeps_identity_without_prefix_or_rng_draws(
    monkeypatch, opening_inputs
):
    probe = opening_inputs
    probe.context.robot.qpos[:, 2:4] = 0
    first = probe.plan

    def shared(self, request, context):
        actions._PICK_SELECTION.get().constrained = True
        return first

    monkeypatch.setattr(PickUp, "_plan", shared)
    monkeypatch.setattr(
        pick_safety,
        "_prepend_pick_opening",
        lambda *args: pytest.fail("safe plan composed prefix"),
    )
    before = (
        pickle.dumps((random.getstate(), np.random.get_state())),
        torch.random.get_rng_state().clone(),
    )
    assert probe.action._plan(probe.request, probe.context) is first
    assert before[0] == pickle.dumps((random.getstate(), np.random.get_state()))
    assert torch.equal(before[1], torch.random.get_rng_state())


def _rebuild(probe, positions, dt=None):
    previous = probe.plan.joint_trajectory
    trajectory = replace(
        previous, positions=positions, dt=previous.dt if dt is None else dt
    )
    return replace(
        probe.plan,
        joint_trajectory=trajectory,
        commands=probe.action._joint_command_sequence(
            probe.request, trajectory, active_mask=probe.plan.plan_success
        ),
    )


def test_prefix_rebuilds_every_payload_and_schedule_preserving_original_suffix_and_effects(
    opening_inputs,
):
    probe = opening_inputs
    observed = probe.context.robot.qpos.clone()
    original = probe.plan.snapshot()
    candidate, _ = pick_safety._prepend_pick_opening(
        probe.action, probe.request, probe.context, original
    )
    assert candidate is not None
    trajectory = candidate.joint_trajectory
    for index, frame in enumerate(candidate.commands.frames):
        assert torch.equal(
            frame.hold_duration,
            trajectory.dt[:, min(index + 1, trajectory.waypoint_count - 1)],
        )
        for command in frame.commands:
            ids = list(command.target.joint_ids)
            assert torch.equal(
                command.payload.positions, trajectory.positions[:, index, ids]
            )
            assert torch.equal(
                command.payload.velocities, trajectory.velocities[:, index, ids]
            )
    for old, new in zip(original.commands.frames, candidate.commands.frames[10:]):
        assert torch.equal(old.hold_duration, new.hold_duration)
        for old_command, new_command in zip(old.commands, new.commands):
            assert old_command.target == new_command.target
            assert torch.equal(
                old_command.payload.positions, new_command.payload.positions
            )
            assert torch.equal(
                old_command.payload.velocities, new_command.payload.velocities
            )
    assert torch.equal(
        trajectory.positions[:, :10, [0, 1, 4, 5]],
        observed[:, None, [0, 1, 4, 5]].expand(-1, 10, -1),
    )
    assert torch.equal(
        candidate.expected_effects.held_object_updates["active"].object_to_eef,
        original.expected_effects.held_object_updates["active"].object_to_eef,
    )
    assert torch.equal(
        candidate.effect_candidates.held_object_updates["active"].grasp_xpos,
        original.effect_candidates.held_object_updates["active"].grasp_xpos,
    )
    assert torch.equal(probe.context.robot.qpos, observed)
    assert not probe.context.task.held_objects
    assert candidate.tracking is original.tracking is None
    assert candidate.tracking_policy == original.tracking_policy


def test_action_opens_free_hand_once_without_resampling_grasp_or_extra_rng(
    monkeypatch, opening_inputs
):
    probe = opening_inputs
    calls = []

    def shared(self, request, context):
        actions._PICK_SELECTION.get().constrained = True
        calls.append(request)
        return probe.plan

    monkeypatch.setattr(PickUp, "_plan", shared)
    before = (
        pickle.dumps((random.getstate(), np.random.get_state())),
        torch.random.get_rng_state().clone(),
    )
    result = probe.action._plan(probe.request, probe.context)
    assert result.plan_success.all()
    assert len(calls) == 1
    assert (
        result.diagnostics.metadata["gen_sim_pick_safety"]["variant"]
        == "observed_hand_opening"
    )
    assert (
        result.diagnostics.metadata["affordance_sample"]
        == probe.plan.diagnostics.metadata["affordance_sample"]
    )
    assert before[0] == pickle.dumps((random.getstate(), np.random.get_state()))
    assert torch.equal(before[1], torch.random.get_rng_state())


@pytest.mark.parametrize(
    "invalid",
    [
        "held",
        "coordinated",
        "explicit_grasp",
        "downstream",
        "direct",
        "failed_plan",
        "wrong_environment",
        "not_open",
        "arm_boundary",
        "passive",
        "internal_velocity",
        "payload",
        "schedule",
        "unknown_timing",
        "budget",
        "joint_limits",
        "original_collision",
        "prefix_collision",
        "collision_unavailable",
    ],
)
def test_prefix_refuses_any_other_unqualified_plan_or_boundary(
    monkeypatch, opening_inputs, invalid
):
    probe = opening_inputs
    request, plan = probe.request, probe.plan
    if invalid == "held":
        probe.context.task = TaskState(
            1,
            "cpu",
            held_objects={
                "active": plan.expected_effects.held_object_updates["active"]
            },
        )
    elif invalid == "coordinated":
        probe.context.task = SimpleNamespace(
            get_held_object=lambda key: None,
            coordinated_held_objects={
                ("active", "other"): SimpleNamespace(env_mask=torch.tensor([True]))
            },
        )
    elif invalid == "explicit_grasp":
        request = replace(
            request, goal=replace(request.goal, grasp_xpos=torch.eye(4)[None])
        )
    elif invalid == "downstream":
        request = replace(
            request,
            skill_options=replace(
                request.skill_options,
                downstream_object_target_poses=(torch.eye(4)[None],),
            ),
        )
    elif invalid == "direct":
        probe.action.motion_generator.direct_transport_planning = True
    elif invalid == "failed_plan":
        plan = replace(
            plan,
            plan_success=torch.tensor([False]),
            diagnostics=replace(
                plan.diagnostics,
                failure=PlanningFailure("planning_failed", retryable=True),
            ),
        )
    elif invalid == "wrong_environment":
        probe.context.env_ids = torch.tensor([1])
    elif invalid in {
        "not_open",
        "arm_boundary",
        "passive",
        "internal_velocity",
        "joint_limits",
    }:
        positions = plan.joint_trajectory.positions.clone()
        if invalid == "not_open":
            positions[:, :2, 2:4] = 0.03
        elif invalid == "arm_boundary":
            positions[:, 0, 0] += 0.01
        elif invalid == "passive":
            positions[:, 2, 4] += 0.01
        elif invalid == "internal_velocity":
            positions[:, 2, 0] += 0.2
        else:
            probe.robot.get_qpos_limits = lambda **kwargs: torch.tensor(
                [[[-0.2, 0.2], [-0.2, 0.2]]]
            )
        plan = _rebuild(probe, positions)
    elif invalid in {"payload", "schedule"}:
        frames = list(plan.commands.frames)
        if invalid == "payload":
            commands = list(frames[1].commands)
            commands[0] = replace(
                commands[0],
                payload=replace(commands[0].payload, positions=torch.zeros(1, 2)),
            )
            frames[1] = replace(frames[1], commands=tuple(commands))
        else:
            frames[1] = replace(frames[1], hold_duration=torch.tensor([0.08]))
        plan = replace(plan, commands=replace(plan.commands, frames=tuple(frames)))
    elif invalid == "unknown_timing":
        dt = plan.joint_trajectory.dt.clone()
        dt[:, 3] = 0.05
        plan = _rebuild(probe, plan.joint_trajectory.positions, dt)
    elif invalid == "budget":
        request = replace(
            request, skill_options=replace(request.skill_options, hand_interp_steps=9)
        )
    else:

        def collision(robot, positions, part):
            if invalid == "collision_unavailable":
                raise ValueError("no mesh certificate")
            bad = invalid == "original_collision" or bool(
                ((positions[:, 2] > 0.3) & (positions[:, 2] < 0.6)).any()
            )
            return {"valid": not bad}

        monkeypatch.setattr(articulation_collision, "check_robot_motion", collision)
    candidate, report = pick_safety._prepend_pick_opening(
        probe.action, request, probe.context, plan
    )
    assert candidate is None and not report["accepted"]


def test_prefix_retains_zero_velocity_target_policy_and_disabled_scene_dependency(
    opening_inputs,
):
    probe = opening_inputs
    request = replace(
        probe.request,
        motion_policy=replace(probe.request.motion_policy, velocity_targets="zero"),
    )
    plan = replace(probe.plan, scene_dependency_monitor_until={"item": 0})
    candidate, report = pick_safety._prepend_pick_opening(
        probe.action, request, probe.context, plan
    )
    assert candidate is not None and report["accepted"]
    assert torch.equal(
        candidate.joint_trajectory.velocities[:, :10], torch.zeros(1, 10, 6)
    )
    assert candidate.scene_dependency_monitor_until["item"] == 0


def test_feedback_tracking_is_rebuilt_for_every_prefix_and_original_frame(
    opening_inputs,
):
    from embodichain.lab.sim.atomic_actions.tracking import (
        EndpointTrackingChannelBinding,
        EndpointTrackingFeedbackAddress,
        TrackingFeedbackSourceRef,
        TrackingProjectorRef,
        TrackingRuntime,
    )

    probe = opening_inputs
    motion_endpoint, grasp_endpoint = probe.request.binding.endpoints
    arm = motion_endpoint.require_target(JointPositionTarget)
    hand = grasp_endpoint.require_target(JointPositionTarget)
    for name, endpoint, target in (
        ("motion", motion_endpoint, arm),
        ("grasp", grasp_endpoint, hand),
    ):
        endpoint.key = ("primary", name)
        endpoint.destination_key = (target.transport_id, target.target_id)
        endpoint.capabilities = (
            frozenset({"interaction.grasp"}) if name == "grasp" else frozenset()
        )
        endpoint.tracking_channels = {}
    motion_endpoint.tracking_channels["joint.position"] = (
        EndpointTrackingChannelBinding(
            channel_id="joint.position",
            source=TrackingFeedbackSourceRef(
                "planning_context.robot",
                "1",
                EndpointTrackingFeedbackAddress(arm, "joint.position"),
            ),
            projector=TrackingProjectorRef("joint_position_payload", "1"),
        )
    )
    probe.action._planning_services.tracking_runtime = TrackingRuntime.with_builtins()
    request = replace(
        probe.request,
        tracking_policy=TrackingPolicy.joint_position(terminal_max_abs_error=0.25),
    )
    previous_tracking = probe.action._tracking_sequence(request, probe.plan.commands)
    original = replace(
        probe.plan, tracking_policy=request.tracking_policy, tracking=previous_tracking
    )
    candidate, report = pick_safety._prepend_pick_opening(
        probe.action, request, probe.context, original
    )
    assert candidate is not None and report["accepted"]
    assert candidate.tracking.frame_count == candidate.commands.frame_count == 16
    for index, frame in enumerate(candidate.tracking.frames):
        setpoint = frame.setpoints[0]
        assert (
            setpoint.binding.source.source_fingerprint
            == previous_tracking.frames[0]
            .setpoints[0]
            .binding.source.source_fingerprint
        )
        assert torch.equal(
            setpoint.desired.positions,
            candidate.joint_trajectory.positions[:, index, :2],
        )
    for previous, current in zip(
        previous_tracking.frames, candidate.tracking.frames[10:]
    ):
        assert torch.equal(
            previous.setpoints[0].desired.positions,
            current.setpoints[0].desired.positions,
        )


def test_prefix_does_not_remove_missing_velocity_payloads_or_create_an_arm_jump(
    opening_inputs,
):
    probe = opening_inputs
    trajectory = replace(probe.plan.joint_trajectory, velocities=None)
    original = replace(
        probe.plan,
        joint_trajectory=trajectory,
        commands=probe.action._joint_command_sequence(
            probe.request, trajectory, active_mask=probe.plan.plan_success
        ),
    )
    candidate, report = pick_safety._prepend_pick_opening(
        probe.action, probe.request, probe.context, original
    )
    assert candidate is not None and report["accepted"]
    assert candidate.joint_trajectory.velocities is None
    assert all(
        command.payload.velocities is None
        for frame in candidate.commands.frames
        for command in frame.commands
    )
    assert torch.equal(
        candidate.joint_trajectory.positions[:, :10, :2],
        probe.context.robot.qpos[:, None, :2].expand(-1, 10, -1),
    )


def test_shortened_prefix_fails_the_same_full_command_velocity_gate(opening_inputs):
    probe = opening_inputs
    candidate, _ = pick_safety._prepend_pick_opening(
        probe.action, probe.request, probe.context, probe.plan
    )
    dt = candidate.joint_trajectory.dt.clone()
    dt[:, 1:10] = 0.004
    trajectory = replace(candidate.joint_trajectory, dt=dt)
    transformed = replace(
        candidate,
        joint_trajectory=trajectory,
        commands=probe.action._joint_command_sequence(
            probe.request, trajectory, active_mask=probe.plan.plan_success
        ),
    )
    certificate = pick_safety._inspect_pick_plan(
        probe.robot, probe.request, probe.context, transformed
    )
    assert certificate == {"valid": False, "reason": "final_velocity_rejected"}


@pytest.fixture
def native_overshoot_inputs(monkeypatch, opening_inputs):
    probe = opening_inputs
    measured = torch.tensor(
        [
            [
                0.70013028383255,
                -0.6986430883407593,
                0.7082325220108032,
                -0.6903746128082275,
                -0.6920925378799438,
                0.7040384411811829,
            ]
        ]
    )
    qpos = torch.cat(
        (probe.context.robot.qpos[:, :2], measured, probe.context.robot.qpos[:, 4:]),
        dim=1,
    )
    probe.context.robot.qpos, probe.context.robot.qvel = qpos, torch.zeros_like(qpos)
    hand = JointPositionTarget("hand", tuple(range(2, 8)))
    endpoint = probe.request.binding.endpoints[1]
    endpoint.require_target = lambda cls: hand
    endpoint.joint_positions = lambda key, **kwargs: (
        torch.zeros(1, 6)
        if key == "open"
        else torch.tensor([[0.02, -0.02, 0.02, -0.02, -0.02, 0.02]])
    )
    probe.robot.get_joint_ids = lambda name: (
        (0, 1) if name == "arm" else tuple(range(2, 8))
    )
    hand_bounds = torch.tensor([[[-0.7, 0.7]]]).expand(1, 6, 2).clone()
    probe.robot.get_qpos_limits = lambda name, env_ids=None: (
        hand_bounds if name == "hand" else torch.tensor([[[-1.0, 1.0], [-1.0, 1.0]]])
    )
    monkeypatch.setattr(
        motion, "_joint_velocity_limits", lambda *args: torch.full((1, 10), 2.0)
    )
    positions = qpos[:, None].repeat(1, 6, 1)
    positions[:, :, :2] = probe.plan.joint_trajectory.positions[:, :, :2]
    signs = torch.tensor([1.0, -1.0, 1.0, -1.0, -1.0, 1.0])
    positions[:, :, 2:8] = (
        torch.tensor([0.0, 0.0, 0.01, 0.02, 0.02, 0.02])[None, :, None] * signs
    )
    trajectory = TimedTrajectory.from_uniform_step(
        positions, env_ids=probe.context.env_ids, step_dt=0.04
    )
    velocities = differentiate_positions(positions, trajectory.dt)
    velocities[:, -1] = 0
    trajectory = replace(trajectory, velocities=velocities)
    probe.plan = replace(
        probe.plan,
        joint_trajectory=trajectory,
        commands=probe.action._joint_command_sequence(
            probe.request, trajectory, active_mask=probe.plan.plan_success
        ),
    )
    return probe


def test_native_measurement_overshoot_projects_only_first_desired_hand_target(
    native_overshoot_inputs,
):
    probe = native_overshoot_inputs
    actual = probe.context.robot.qpos.clone()
    suffix = probe.plan.joint_trajectory.positions.clone()
    candidate, report = pick_safety._prepend_pick_opening(
        probe.action, probe.request, probe.context, probe.plan
    )
    assert candidate is not None and report["accepted"]
    assert report["intervals"] == 9 and report["duration"] == pytest.approx(0.36)
    assert (
        report["position_limits_source"]
        == "Robot.get_qpos_limits(name=hand, env_ids=context.env_ids)"
    )
    positions = candidate.joint_trajectory.positions
    projected = actual[:, 2:8].clamp(-0.7, 0.7)
    assert torch.equal(positions[:, 0, 2:8], projected)
    expected = actual[:, None, 2:8] * (1 - torch.linspace(0, 1, 10))[None, :, None]
    assert torch.allclose(positions[:, 1:10, 2:8], expected[:, 1:], atol=1e-7, rtol=0)
    assert torch.equal(
        positions[:, :10, [0, 1, 8, 9]],
        actual[:, None, [0, 1, 8, 9]].expand(-1, 10, -1),
    )
    assert torch.equal(positions[:, 10:], suffix)
    assert torch.equal(probe.context.robot.qpos, actual)
    assert pick_safety._inspect_pick_plan(
        probe.robot, probe.request, probe.context, candidate
    )["valid"]


@pytest.mark.parametrize("observed", [0.8, 0.778])
def test_large_overshoot_or_later_out_of_bounds_sample_cannot_be_hidden_by_clamping(
    native_overshoot_inputs, observed
):
    probe = native_overshoot_inputs
    probe.context.robot.qpos[:, 4] = observed
    actual = probe.context.robot.qpos.clone()
    candidate, report = pick_safety._prepend_pick_opening(
        probe.action, probe.request, probe.context, probe.plan
    )
    assert candidate is None and not report["accepted"]
    if observed == 0.8:
        assert report.get("reason") == "first_projection_velocity_rejected"
    else:
        assert report["command_certificate"]["reason"] == "joint_limits_rejected"
    assert torch.equal(probe.context.robot.qpos, actual)


def test_projection_cannot_hide_collision_of_actual_observed_start(
    monkeypatch, native_overshoot_inputs
):
    probe = native_overshoot_inputs

    def collision(robot, positions, part):
        return {"valid": not bool((positions[:, 4] > 0.7).any())}

    monkeypatch.setattr(articulation_collision, "check_robot_motion", collision)
    candidate, report = pick_safety._prepend_pick_opening(
        probe.action, probe.request, probe.context, probe.plan
    )
    assert candidate is None and report["reason"] == "observed_start_collision_rejected"
