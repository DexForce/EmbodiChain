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

from __future__ import annotations

from dataclasses import dataclass, replace
import random
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from embodichain.gen_sim.task_engine._task_program import place_branch as route
from embodichain.gen_sim.task_engine._task_program import place_recovery, motion
from embodichain.gen_sim.task_engine._task_program.adaptive_grasp import ADAPTIVE_GRASP
from embodichain.gen_sim.task_engine._task_program.pick_safety import _PICK_GRASP_RULE
from embodichain.lab.sim.atomic_actions import JointPositionTarget, PlannerDiagnostics
from embodichain.lab.sim.atomic_actions.effects import StateDelta
from embodichain.lab.sim.atomic_actions.goals import SceneEntityPose
from embodichain.lab.sim.atomic_actions.plans import (
    PlanningFailure,
    TimedTrajectory,
    TrajectorySegment,
)
from embodichain.lab.sim.atomic_actions.primitives.place import PlaceGoal, PlaceOptions

_ORIGINAL_INSPECT = place_recovery.inspect


@dataclass(frozen=True)
class Plan:
    plan_success: torch.Tensor
    diagnostics: PlannerDiagnostics
    joint_trajectory: TimedTrajectory
    segments: tuple
    expected_effects: StateDelta
    effect_candidates: StateDelta
    effect_verification: object
    replannable: bool = True
    scene_dependencies: tuple = ("support",)
    scene_dependency_monitor_until: dict | None = None
    scene_dependency_end_segment: str | None = None


def _poses(values):
    result = torch.eye(4)[None, None].repeat(1, len(values), 1, 1)
    result[0, :, 0, 3] = torch.tensor(values)
    return result


def _fixture(monkeypatch):
    down, back = _poses([0.13, 0.23, 0.4, 0.5]), _poses([0.62, 0.7])
    start = torch.tensor([[0.0, 0.9, 0.8]])
    rules = {}
    held = NS(semantics=NS(affordance=NS(get_custom_config=rules.get)))
    arm = JointPositionTarget(control_part="arm", joint_ids=(0,))
    endpoint = NS(task_state_key="left", require_target=lambda _: arm)
    request = NS(
        invocation_id="place:0",
        goal=PlaceGoal(SceneEntityPose("support_target")),
        binding=NS(endpoint=lambda *_: endpoint),
        motion_policy=NS(strategy="ik_interp", dynamic_collision_mode="off"),
        skill_options=PlaceOptions(
            hand_interp_steps=2,
            release_settle_steps=2,
            preserve_current_object_orientation=False,
        ),
    )
    context = NS(
        batch_size=1,
        env_ids=torch.tensor([0]),
        robot=NS(qpos=start),
        get_held_object=lambda _: held,
        scene=NS(
            entities={"support_target": NS(pose=down[:, -1].clone(), confidence=1.0)}
        ),
    )

    def fk(qpos, **kwargs):
        poses = torch.eye(4).expand(*qpos.shape[:-1], 4, 4).clone()
        poses[..., 0, 3] = qpos[..., 0]
        return poses

    robot = NS(
        device=torch.device("cpu"),
        get_qpos_limits=lambda **_: torch.tensor([[[-1.0, 1.0]]]),
        get_solver=lambda _: NS(
            get_iteration_params=lambda: {"pos_eps": 0.005, "rot_eps": 0.0005}
        ),
        compute_fk=fk,
        compute_batch_fk=fk,
    )

    def single(pose, joint_seed, **kwargs):
        torch.rand(1)
        np.random.rand()
        random.random()
        return torch.tensor([True]), pose[:, :1, 3].clone()

    def batch(pose, joint_seed, **kwargs):
        assert torch.equal(joint_seed, torch.full((1, 6, 1), 0.5))
        return torch.ones(1, 6, dtype=torch.bool), pose[:, :, :1, 3].clone()

    robot.compute_ik = Mock(side_effect=single)
    robot.compute_batch_ik = Mock(side_effect=batch)
    generator = NS(robot=robot)
    action = NS(
        is_bound=True,
        ordinary_on_invocations=frozenset({request.invocation_id}),
        device=robot.device,
        robot=robot,
        motion_generator=generator,
        recovery_scene_vertices={"item": object()},
    )
    old = torch.tensor(
        [
            [
                [x, 0.9 if i < 6 else 0.2, 0.8]
                for i, x in enumerate(
                    [0.0, 0.25, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.7]
                )
            ]
        ]
    )
    original = Plan(
        torch.tensor([True]),
        PlannerDiagnostics("mock"),
        TimedTrajectory.from_uniform_step(old, env_ids=context.env_ids, step_dt=0.1),
        (
            TrajectorySegment("approach", 0, 5),
            TrajectorySegment("release", 5, 9),
            TrajectorySegment("retract", 9, 13),
        ),
        StateDelta(held_object_updates={"left": None}),
        StateDelta(),
        object(),
        scene_dependency_monitor_until={"support": 5},
    )
    # Deliberately omit mandatory interior poses from the old phase split.
    original = replace(
        original,
        joint_trajectory=TimedTrajectory.from_uniform_step(
            torch.cat((old[:, :5], old[:, 5:9], old[:, 6:10]), dim=1),
            env_ids=context.env_ids,
            step_dt=0.1,
        ),
    )
    built = []

    def build(selected, current):
        assert selected is request and current is context
        assert route.capture_keyframes(action, down) is down
        assert route.capture_keyframes(action, back) is back
        ok, _ = route.solve_waypoints(
            generator, torch.cat((down, back), 1), start[:, :1], "arm"
        )
        return replace(original, plan_success=ok)

    def rebuild(selected, current, **kwargs):
        assert selected is request and current is context
        built.append(kwargs)
        return replace(
            original,
            plan_success=kwargs["success"],
            joint_trajectory=kwargs["trajectory"],
            expected_effects=kwargs["expected_effects"],
            effect_candidates=kwargs["effect_candidates"],
            effect_verification=kwargs["effect_verification"],
            diagnostics=kwargs["diagnostics"],
        )

    action.build_plan = Mock(side_effect=rebuild)
    failed = replace(
        original,
        plan_success=torch.tensor([False]),
        joint_trajectory=TimedTrajectory.empty(
            batch_size=1, robot_dof=3, device="cpu", env_ids=context.env_ids
        ),
        segments=(),
    )
    action.failed_plan = Mock(
        side_effect=lambda *args, **kwargs: replace(
            failed,
            diagnostics=PlannerDiagnostics(
                "mock",
                failure=PlanningFailure(kwargs["failure_code"], kwargs["retryable"]),
            ),
        )
    )
    inspection = Mock(return_value={"valid": True, "scope": "mock_original_gates"})
    monkeypatch.setattr(place_recovery, "inspect", inspection)
    return (
        action,
        request,
        context,
        build,
        original,
        down,
        back,
        rules,
        inspection,
        built,
    )


def test_mandatory_phase_keyframes_survive_old_split_with_same_timing_and_effects(
    monkeypatch,
):
    action, request, context, build, old, down, back, _, inspection, built = _fixture(
        monkeypatch
    )
    before = route._RandomState.capture(set())
    result = route.plan(action, request, context, build)
    assert result.plan_success.all()
    positions = result.joint_trajectory.positions
    assert positions.shape == old.joint_trajectory.positions.shape
    assert positions[0, :5, 0].tolist() == pytest.approx([0.0, 0.13, 0.23, 0.4, 0.5])
    assert positions[0, 5:9, 0].tolist() == pytest.approx([0.5] * 4)
    assert positions[0, 9:, 0].tolist() == pytest.approx([0.5, 0.56, 0.62, 0.7])
    assert torch.equal(positions[..., 1:], old.joint_trajectory.positions[..., 1:])
    assert torch.equal(result.joint_trajectory.dt, old.joint_trajectory.dt)
    assert result.expected_effects is old.expected_effects
    assert result.effect_candidates is old.effect_candidates
    assert result.effect_verification is old.effect_verification
    assert built[0]["segment_lengths"] == {"approach": 5, "release": 4, "retract": 4}
    receipt = result.diagnostics.metadata["gen_sim_place_branch"]
    assert receipt["accepted"] and receipt["physical_evidence"] is False
    assert receipt["mandatory_counts"] == [4, 2]
    assert receipt["solve_requests"] == 1
    assert (
        action.robot.compute_ik.call_count
        == action.robot.compute_batch_ik.call_count
        == 1
    )
    torch.testing.assert_close(
        action.robot.compute_ik.call_args.kwargs["joint_seed"],
        context.robot.qpos[:, :1],
    )
    inspection.assert_called_once()
    assert torch.equal(before.cpu, torch.get_rng_state())
    assert np.array_equal(before.numpy[1], np.random.get_state()[1])
    assert before.python == random.getstate()
    assert (
        route.solve_waypoints(
            action.motion_generator, down, context.robot.qpos[:, :1], "arm"
        )
        is None
    )
    assert route.capture_keyframes(action, back) is back


@pytest.mark.parametrize(
    "damage",
    [
        "release_ik",
        "batch_ik",
        "finite",
        "limits",
        "fk",
        "emitted_fk",
        "release_fk",
        "scene",
        "velocity",
        "missing_capture",
        "duplicate_solve",
        "too_few_phase_frames",
    ],
)
def test_owned_failure_never_retries_or_dispatches_an_alternative(monkeypatch, damage):
    action, request, context, build, old, down, back, _, inspection, _ = _fixture(
        monkeypatch
    )
    if damage == "release_ik":
        action.robot.compute_ik.side_effect = lambda **_: (
            torch.tensor([False]),
            torch.zeros(1, 1),
        )
    elif damage in {"batch_ik", "finite", "limits"}:
        ok = torch.full((1, 6), damage != "batch_ik", dtype=torch.bool)
        q = torch.cat((down, back), 1)[:, :, :1, 3].clone()
        if damage == "finite":
            q[0, 1, 0] = float("nan")
        elif damage == "limits":
            q[0, 1, 0] = 2.0
        action.robot.compute_batch_ik.side_effect = lambda **_: (ok, q)
    elif damage == "fk":
        action.robot.compute_batch_fk = lambda qpos, **_: _poses([0.9] * qpos.shape[1])
    elif damage == "emitted_fk":
        original_fk = action.robot.compute_batch_fk
        calls = []

        def fk(qpos, **kwargs):
            calls.append(None)
            return (
                original_fk(qpos, **kwargs)
                if len(calls) == 1
                else _poses([0.9] * qpos.shape[1])
            )

        action.robot.compute_batch_fk = fk
    elif damage == "release_fk":
        original_fk = action.robot.compute_fk
        calls = []

        def fk(qpos, **kwargs):
            calls.append(None)
            result = original_fk(qpos, **kwargs)
            if len(calls) > 1:
                result[:, 0, 3] = 0.9
            return result

        action.robot.compute_fk = fk
    elif damage == "velocity":
        monkeypatch.setattr(
            place_recovery,
            "_validate_final_plan_velocity",
            lambda *_: {"valid_mask": torch.tensor([False])},
        )
        inspection.side_effect = _ORIGINAL_INSPECT
    elif damage == "scene":
        inspection.return_value = {"valid": False, "reason": damage + "_rejected"}
    elif damage == "missing_capture":
        build = lambda *_: old
    elif damage == "duplicate_solve":
        first = build

        def build(*args):
            result = first(*args)
            route.solve_waypoints(
                action.motion_generator,
                torch.cat((down, back), 1),
                context.robot.qpos[:, :1],
                "arm",
            )
            return result

    else:
        first = build

        def build(*args):
            return replace(
                first(*args),
                segments=(
                    TrajectorySegment("approach", 0, 3),
                    TrajectorySegment("release", 3, 9),
                    TrajectorySegment("retract", 9, 13),
                ),
            )

    result = route.plan(action, request, context, Mock(side_effect=build))
    assert not result.plan_success.any()
    assert result.joint_trajectory.waypoint_count == 0
    assert result.diagnostics.failure.code == "gen_sim_place_branch_unqualified"
    assert result.diagnostics.failure.retryable is False
    assert not result.diagnostics.metadata["gen_sim_place_branch"]["accepted"]
    assert action.robot.compute_ik.call_count <= 1
    assert action.robot.compute_batch_ik.call_count <= 1
    action.failed_plan.assert_called_once()
    assert (
        route.solve_waypoints(
            action.motion_generator, down, context.robot.qpos[:, :1], "arm"
        )
        is None
    )


@pytest.mark.parametrize(
    "damage",
    [
        "preserve",
        "batch",
        "env",
        "cartesian",
        "dynamic",
        "backend",
        "direct",
        "symmetry",
        "ceiling",
        "pour",
        "ends",
        "constrained",
        "upright",
        "tensor",
        "unbound",
        "unlisted",
        "missing_invocation",
    ],
)
def test_nonowned_contracts_are_not_taken_over(monkeypatch, damage):
    action, request, context, _, old, _, _, rules, _, _ = _fixture(monkeypatch)
    if damage == "preserve":
        request.skill_options = replace(
            request.skill_options, preserve_current_object_orientation=True
        )
    elif damage == "batch":
        context.batch_size = 2
    elif damage == "env":
        context.env_ids = torch.tensor([1])
    elif damage == "dynamic":
        request.motion_policy.dynamic_collision_mode = "auto"
    elif damage == "backend":
        request.motion_policy.strategy = "motion_gen"
    elif damage == "direct":
        action.motion_generator.direct_place_planning = True
    elif damage == "symmetry":
        request.goal = replace(request.goal, tcp_symmetry="z_roll_180")
    elif damage == "ceiling":
        request.skill_options = replace(
            request.skill_options, max_approach_retract_z=1.0
        )
    elif damage in {"pour", "ends"}:
        rules[ADAPTIVE_GRASP] = damage
    elif damage == "constrained":
        rules[_PICK_GRASP_RULE] = ("item", "upright")
    elif damage == "upright":
        request.goal = PlaceGoal(
            SceneEntityPose("support", world_orientation=torch.eye(3))
        )
    elif damage == "tensor":
        request.goal = PlaceGoal(torch.eye(4))
    elif damage == "unbound":
        action.is_bound = False
    elif damage == "unlisted":
        action.ordinary_on_invocations = frozenset()
    elif damage == "missing_invocation":
        request.invocation_id = None
    token = motion._CARTESIAN_APPROACH.set(damage == "cartesian")
    try:
        assert not route.owns(action, request, context)
        build = Mock(return_value=old)
        with pytest.raises(ValueError, match="bound ordinary on"):
            route.plan(action, request, context, build)
        build.assert_not_called()
        action.robot.compute_ik.assert_not_called()
    finally:
        motion._CARTESIAN_APPROACH.reset(token)


def test_unbound_owner_does_not_inspect_request_or_context():
    action = NS(is_bound=False)
    assert not route.owns(action, object(), object())
    build = Mock(return_value=object())
    with pytest.raises(ValueError, match="bound ordinary on"):
        route.plan(action, object(), object(), build)
    build.assert_not_called()


def test_nonplace_owner_does_not_inspect_options_or_context():
    action = NS(is_bound=True)
    assert not route.owns(action, object(), object())
    build = Mock(return_value=object())
    with pytest.raises(ValueError, match="bound ordinary on"):
        route.plan(action, object(), object(), build)
    build.assert_not_called()


def test_real_place_builder_rebuilds_commands_and_tracking_from_new_phase_samples(
    monkeypatch,
):
    from embodichain.gen_sim.task_engine._task_program.actions import GenSimPlace
    from embodichain.lab.sim.atomic_actions import (
        ActionInvocation,
        Affordance,
        AtomicActionEngine,
        ControlPartCommandProfile,
        EntityState,
        HeldObjectState,
        MotionPolicy,
        ObjectSemantics,
        PlanningContext,
        RobotObservation,
        SceneSnapshot,
        TaskState,
    )
    from embodichain.lab.sim.atomic_actions.invocation import PhaseEffectGateRequirement
    from embodichain.lab.sim.atomic_actions.primitives.place import Place
    from embodichain.lab.sim.atomic_actions.tracking import TrackingPolicy

    initial = torch.tensor([[0.0, 0.0, 0.3, 0.0, 1.0, 0.8]])
    arm_ids = (0, 1, 2, 3)

    def encode(pose):
        return torch.cat(
            (
                pose[..., :3, 3],
                torch.atan2(pose[..., 1, 0], pose[..., 0, 0])[..., None],
            ),
            -1,
        )

    def fk(qpos, **_):
        pose = torch.eye(4).expand(*qpos.shape[:-1], 4, 4).clone()
        pose[..., :3, 3] = qpos[..., :3]
        c, s = qpos[..., 3].cos(), qpos[..., 3].sin()
        pose[..., 0, 0], pose[..., 0, 1] = c, -s
        pose[..., 1, 0], pose[..., 1, 1] = s, c
        return pose

    robot = NS(
        device=torch.device("cpu"),
        dof=6,
        control_parts={"arm": object(), "hand": object()},
        get_qpos=lambda: initial.clone(),
        get_joint_ids=lambda name: arm_ids if name == "arm" else (4,),
        get_qpos_limits=lambda **_: torch.tensor([[[-2.0, 2.0]]]).expand(1, 4, 2),
        get_solver=lambda _: NS(
            get_iteration_params=lambda: {"pos_eps": 0.005, "rot_eps": 0.0005}
        ),
        compute_fk=fk,
        compute_batch_fk=fk,
        compute_ik=Mock(
            side_effect=lambda pose, **_: (torch.tensor([True]), encode(pose))
        ),
        compute_batch_ik=Mock(
            side_effect=lambda pose, **_: (
                torch.ones(pose.shape[:2], dtype=torch.bool),
                encode(pose),
            )
        ),
    )
    generator = object.__new__(motion.CheckedMotionGenerator)
    generator.robot, generator.device = robot, robot.device
    generator.planner = NS(
        cfg=NS(planner_type="mock_kinematics"),
        collision_world_info=None,
        preserve_plan_samples=False,
        preserve_failed_plan_positions=False,
        supports_heterogeneous_waypoints=False,
    )
    monkeypatch.setattr(
        motion, "_joint_velocity_limits", lambda *_: torch.full((1, 4), 10.0)
    )
    safety = Mock(return_value={"valid": True, "scope": "mock_scene_only"})
    monkeypatch.setattr(place_recovery, "inspect", safety)
    action = GenSimPlace()
    action.ordinary_on_invocations = frozenset({"ordinary_place:0"})
    engine = AtomicActionEngine(
        generator,
        control_profiles={
            "hand": ControlPartCommandProfile.joint_positions(
                open=torch.zeros(1), grasp=torch.ones(1)
            )
        },
        load_builtins=False,
    )
    engine.register(action)
    binding = engine.bind_control_parts(
        "place", {"primary": {"motion": "arm", "grasp": "hand"}}
    )
    gate = PhaseEffectGateRequirement("source_released", "retract")
    request = action.resolve_request(
        ActionInvocation(
            "place",
            PlaceGoal(SceneEntityPose("destination")),
            binding,
            motion_policy=MotionPolicy(sample_count=22, dynamic_collision_mode="off"),
            tracking_policy=TrackingPolicy.joint_position(),
            skill_options=PlaceOptions(
                hand_interp_steps=2,
                release_settle_steps=2,
                cartesian_waypoint_count=2,
                preserve_current_object_orientation=False,
            ),
            phase_effect_gates=(gate,),
            invocation_id="ordinary_place:0",
            revision=3,
        )
    )
    held = HeldObjectState(
        ObjectSemantics(Affordance(), {}, "item"),
        torch.eye(4)[None],
        torch.eye(4)[None],
    )
    target = fk(torch.tensor([[0.4, 0.1, 0.2, 0.7]]))
    context = PlanningContext(
        robot=RobotObservation(0.0, initial, torch.zeros_like(initial)),
        task=TaskState(batch_size=1, device="cpu", held_objects={"arm": held}),
        scene=SceneSnapshot(
            timestamp=0.0, version=7, entities={"destination": EntityState(target)}
        ),
        env_ids=torch.tensor([0]),
        control_dt=0.1,
    )
    shared_plan = Place._plan
    originals = []

    def capture(self, start, targets, options):
        return route.capture_keyframes(
            self, Place._translation_keyframes(self, start, targets, options)
        )

    def solve(generator, poses, start, part):
        qualified = route.solve_waypoints(generator, poses, start, part)
        assert qualified is not None
        return qualified

    monkeypatch.setattr(GenSimPlace, "_translation_keyframes", capture)
    monkeypatch.setattr(
        motion.CheckedMotionGenerator, "_solve_cartesian_waypoints", solve
    )

    def record_shared(self, selected, current):
        assert self is action and selected is request and current is context
        assert selected.phase_effect_gates == (gate,)
        original = shared_plan(self, selected, current)
        originals.append(original)
        return original

    monkeypatch.setattr(
        GenSimPlace,
        "_plan",
        lambda self, selected, current: self._with_contact_monitor_boundary(
            route.plan(
                self,
                selected,
                current,
                lambda request, context: record_shared(self, request, context),
            )
        ),
    )
    result = action.plan(request, context)
    assert result.plan_success.tolist() == [True]
    assert len(originals) == 1
    original = originals[0]
    actual, old = result.joint_trajectory, original.joint_trajectory
    assert not torch.equal(actual.positions[..., :4], old.positions[..., :4])
    assert torch.equal(actual.positions[..., 4:], old.positions[..., 4:])
    assert torch.equal(actual.dt, old.dt)
    assert result.commands is not original.commands
    assert result.tracking is not original.tracking
    assert result.segments == original.segments
    assert result.invocation_id == original.invocation_id == request.invocation_id
    assert result.invocation_revision == original.invocation_revision == 3
    assert dict(result.expected_effects.held_object_updates) == {"arm": None}
    assert dict(original.expected_effects.held_object_updates) == {"arm": None}
    for field in (
        "held_object_updates",
        "coordinated_held_object_updates",
        "articulation_joint_updates",
    ):
        assert dict(getattr(result.effect_candidates, field)) == dict(
            getattr(original.effect_candidates, field)
        )
    assert result.effect_verification is original.effect_verification
    assert result.scene_dependencies == original.scene_dependencies == ("destination",)
    assert result.scene_dependency_monitor_until == {
        "destination": result.segment("approach").stop
    }
    assert request.phase_effect_gates == (gate,)
    assert context.task.get_held_object("arm") is not None
    assert robot.compute_ik.call_count == robot.compute_batch_ik.call_count == 1
    for index, (frame, tracking) in enumerate(
        zip(result.commands.frames, result.tracking.frames, strict=True)
    ):
        assert len(frame.commands) == 2 and len(tracking.setpoints) == 1
        for command in frame.commands:
            ids = list(command.target.joint_ids)
            assert torch.equal(
                command.payload.positions, actual.positions[:, index, ids]
            )
            assert torch.equal(
                command.payload.velocities, actual.velocities[:, index, ids]
            )
        assert tracking.setpoints[0].endpoint_key == ("primary", "motion")
        assert torch.equal(
            tracking.setpoints[0].desired.positions,
            actual.positions[:, index, list(arm_ids)],
        )
        assert torch.equal(
            frame.hold_duration, actual.dt[:, min(index + 1, actual.waypoint_count - 1)]
        )
    safety.assert_called_once()


def _bound_on(mode="affordance"):
    from embodichain.gen_sim.task_engine._task_program.pick_continuation import (
        Continuation,
    )
    from embodichain.lab.task_program.integrations.simulation.bindings import (
        SimulationRigidObjectBinding,
        SimulationSceneBinding,
        SupportSurfaceAffordanceBinding,
    )
    from embodichain.lab.task_program.semantics import (
        Pick,
        Place,
        RegisteredSemanticCall,
        SceneAffordanceRef,
        SceneObjectRef,
    )

    pick = Pick(SceneObjectRef("item"), resources={"primary": "left"})
    target = (
        SceneAffordanceRef("on__support__item")
        if mode == "affordance"
        else SceneObjectRef("support")
    )
    place = Place(SceneObjectRef("item"), on=target, resources={"primary": "left"})
    if mode == "registered":
        place = RegisteredSemanticCall(
            "simulation.place_relative",
            {"object": "item", "reference": "support", "relation": "on"},
            resources={"primary": "left"},
        )
    route_ = Continuation("pick:0", "place:0", pick, place)
    scene = SimulationSceneBinding(
        registry_id="scene",
        rigid_objects=(
            SimulationRigidObjectBinding("item", "item"),
            SimulationRigidObjectBinding("support", "support"),
        ),
        support_surfaces=(
            SupportSurfaceAffordanceBinding(
                "on__support__item", "support", "on__support__item"
            ),
        ),
    )
    return route_, {"pick:0": "ordinary"}, scene


@pytest.mark.parametrize("mode", ["affordance", "object", "registered"])
@pytest.mark.parametrize("tuple_purposes", [False, True])
def test_binding_owns_only_the_typed_ordinary_on_invocation(mode, tuple_purposes):
    continuation, purposes, scene = _bound_on(mode)
    if tuple_purposes:
        purposes = tuple(purposes.items())
    assert route.bind_on_invocations((continuation,), purposes, scene) == frozenset(
        {"place:0"}
    )


@pytest.mark.parametrize(
    "damage",
    [
        "inside",
        "subject",
        "resource",
        "part",
        "parent",
        "missing_reference",
        "missing_subject",
        "purpose",
        "duplicate_affordance",
        "registered_part",
        "registered_call",
        "registered_extra",
        "self",
        "foreign_place",
    ],
)
def test_binding_rejects_foreign_or_nonordinary_contracts(damage):
    from embodichain.lab.task_program.semantics import (
        Place,
        SceneAffordanceRef,
        SceneObjectRef,
    )

    continuation, purposes, scene = _bound_on(
        "registered" if damage.startswith("registered_") else "affordance"
    )
    call = continuation.place_call
    if damage == "inside":
        call = Place(
            SceneObjectRef("item"),
            inside=SceneObjectRef("support"),
            resources={"primary": "left"},
        )
    elif damage == "subject":
        call = replace(call, object=SceneObjectRef("other"))
    elif damage == "resource":
        call = replace(call, resources={"primary": "right"})
    elif damage in {"part", "parent", "duplicate_affordance"}:
        name = "on__support__item__part_0" if damage == "part" else "on__support__item"
        call = replace(call, on=SceneAffordanceRef(name))
        surface = NS(
            entity_id=name, parent_id="item" if damage == "parent" else "support"
        )
        scene = NS(
            rigid_objects=scene.rigid_objects,
            support_surfaces=(surface,)
            * (2 if damage == "duplicate_affordance" else 1),
        )
    elif damage in {"missing_reference", "missing_subject"}:
        removed = "support" if damage == "missing_reference" else "item"
        scene = NS(
            rigid_objects=tuple(
                b for b in scene.rigid_objects if b.entity_id != removed
            ),
            support_surfaces=scene.support_surfaces,
        )
    elif damage == "purpose":
        purposes["pick:0"] = "pour"
    elif damage == "registered_call":
        call = replace(call, call_id="gen_sim.place_upright")
    elif damage.startswith("registered_"):
        field, value = (
            ("reference_part", "part_0")
            if damage == "registered_part"
            else ("world_yaw", 1.0)
        )
        call = replace(call, arguments={**call.arguments, field: value})
    elif damage == "self":
        call = replace(call, on=SceneObjectRef("item"))
    elif damage == "foreign_place":
        call = object()
    continuation = replace(continuation, place_call=call)
    assert route.bind_on_invocations((continuation,), purposes, scene) == frozenset()


@pytest.mark.parametrize(
    "damage",
    [
        None,
        "phase",
        "resource",
        "missing_release",
        "target",
        "fk",
        "safety",
        "missing_scene",
    ],
)
def test_emitted_inspection_checks_release_pose_and_preserves_all_old_gates(
    monkeypatch, damage
):
    action, request, context, _, original, _, _, _, inspection, _ = _fixture(
        monkeypatch
    )
    if damage == "phase":
        original = replace(original, segments=original.segments[::-1])
    elif damage in {"resource", "missing_release"}:
        original = replace(
            original,
            expected_effects=StateDelta(
                held_object_updates={"other": None} if damage == "resource" else {}
            ),
        )
    elif damage == "target":
        context.scene.entities["support_target"].pose[:, 0, 3] += 0.02
    elif damage == "fk":
        action.robot.compute_fk = lambda **_: _poses([0.9])[:, 0]
    elif damage == "safety":
        inspection.return_value = {"valid": False, "reason": "velocity_rejected"}
    elif damage == "missing_scene":
        context.scene.entities.clear()
        with pytest.raises(KeyError):
            route.inspect_emitted(action, request, context, original)
        return
    result = route.inspect_emitted(action, request, context, original)
    assert result["valid"] is (damage is None)
    if damage is None:
        assert result["release_pose_valid"] and result["physical_evidence"] is False
    action.robot.compute_ik.assert_not_called()
    action.robot.compute_batch_ik.assert_not_called()


@pytest.mark.parametrize("scene", [None, object()])
def test_empty_continuations_do_not_consult_new_scene_or_purpose_dependencies(scene):
    assert route.bind_on_invocations((), object(), scene) == frozenset()


@pytest.mark.parametrize("error", [TypeError, AttributeError, KeyError, IndexError])
def test_programming_errors_propagate_without_fabricating_unqualified_plan(
    monkeypatch, error
):
    action, request, context, _, original, _, _, _, inspection, _ = _fixture(
        monkeypatch
    )
    before = torch.get_rng_state().clone()
    build = Mock(side_effect=error("implementation defect"))
    with pytest.raises(error):
        route.plan(action, request, context, build)
    assert route._SCOPE.get() is None
    assert torch.equal(before, torch.get_rng_state())
    action.failed_plan.assert_not_called()
    inspection.side_effect = error("implementation defect")
    with pytest.raises(error):
        route.inspect_emitted(action, request, context, original)
