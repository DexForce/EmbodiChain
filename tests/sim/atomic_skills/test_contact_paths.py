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

"""Behavioral coverage for one-invocation observed contact paths."""

from __future__ import annotations
from dataclasses import replace
import gc
import pytest
import torch
from embodichain.lab.sim.atomic_skills import (
    Place,
    PickUp,
    PickUpOptions,
    MotionPolicy,
    RecoveryPolicy,
)
from embodichain.lab.sim.atomic_skills.primitives.contact import (
    ContactPlaceGoal,
    ContactPickUpGoal,
    ContactPhase,
    ResolvedContactPhase,
)
from embodichain.lab.sim.atomic_skills.state import RobotObservation
from .test_actions import (
    _bind_action,
    _motion_generator,
    _ACTION_ENGINES,
    _context,
    _invocation,
    _held,
    _torch_interpolation,
)


class _Source:
    def __init__(self, reject_first=False):
        self.reject_first = reject_first
        self.started = []
        self.masks = []

    def resolve(self, phase, request, context, active_mask):
        self.masks.append(active_mask.clone())
        ids = list(request.binding.endpoint("primary", "grasp").target.joint_ids)
        values = context.robot.qpos[:, None, ids].expand(-1, 3, -1).clone()
        values[:, 1:] += 0.001
        return ResolvedContactPhase(values, context.robot.qpos, active_mask, "grasp")

    def begin(self, cursor, context, active_mask):
        self.started.append(cursor.phase_name)

    def hold_steps(self, target):
        return 2

    def finish(self, cursor, context, active_mask):
        mask = active_mask.clone()
        if self.reject_first and cursor.position == 1:
            mask[0] = False
        return mask, ("measured failure",) if self.reject_first else ()


class _GeometrySource(_Source):
    def __init__(self, closed=0.002):
        super().__init__()
        self.closed = closed

    def resolve(self, phase, request, context, active_mask):
        B = context.batch_size
        if phase.kind == "pose":
            values = torch.eye(4).expand(B, 2, 4, 4).clone()
            data = ()
        else:
            values = torch.eye(4).expand(B, 4, 4).clone()
            ids = request.binding.endpoint("primary", "grasp").target.joint_ids
            data = (("closed_qpos", torch.full((B, len(ids)), self.closed)),)
        return ResolvedContactPhase(values, context.robot.qpos, active_mask, data=data)


def _setup(source):
    action = _bind_action(_motion_generator(), Place())
    engine = _ACTION_ENGINES[id(action)]
    engine.planning_services.register_contact_source("fixture", source)
    goal = ContactPlaceGoal(
        "fixture", (ContactPhase("joint", "first"), ContactPhase("joint", "second"))
    )
    invocation = replace(
        _invocation(action, goal),
        motion_policy=MotionPolicy(strategy="ik_interp", dynamic_collision_mode="off"),
        recovery_policy=RecoveryPolicy(
            max_replans=0, max_skill_retries=0, skill_timeout=120
        ),
    )
    return engine, invocation, _context()


def _execute(engine, invocation, context):
    session = engine.start((invocation,), context)
    events = []
    frames = []
    count = 0
    for _ in range(50):
        tick = session.tick(context)
        events.extend(tick.events)
        if tick.command is not None:
            frames.append(tick.command)
            qpos = context.robot.qpos.clone()
            for command in tick.command.commands:
                qpos[:, list(command.target.joint_ids)] = command.payload.positions
            count += 1
            context = replace(
                context,
                robot=RobotObservation(
                    count * context.control_dt, qpos, torch.zeros_like(qpos)
                ),
            )
        if tick.status.value != "running":
            break
    else:
        pytest.fail("Contact path failed to terminate within its finite bound")
    return session, events, frames


def test_contact_path_is_one_invocation_with_multiple_observed_plans():
    source = _Source()
    engine, invocation, context = _setup(source)
    session, events, frames = _execute(engine, invocation, context)
    assert session.status.value == "completed"
    assert len(frames) == 6
    assert len(session.plan_attempts) == 2
    assert source.started == ["first", "second"]
    assert sum(e.kind.value == "skill_completed" for e in events) == 1
    assert sum(e.kind.value == "plan_continued" for e in events) == 1
    assert {a.invocation_index for a in session.plan_attempts} == {0}
    assert session.task_state.get_held_object("arm") is None


def test_measured_phase_failure_does_not_reactivate_or_block_peer():
    source = _Source(reject_first=True)
    engine, invocation, context = _setup(source)
    session, _, frames = _execute(engine, invocation, context)
    assert session.status.value == "completed"
    assert session.eligible_mask.tolist() == [False, True]
    assert source.masks[-1].tolist() == [False, True]
    assert all(frame.active_mask.tolist() == [False, True] for frame in frames[3:])


def test_all_rows_rejected_by_a_contact_check_fail_with_a_normal_terminal_result():
    class RejectAll(_Source):
        def finish(self, cursor, context, active_mask):
            return torch.zeros_like(active_mask), ("All measured checks failed",)

    source = RejectAll()
    engine, invocation, context = _setup(source)
    session, events, frames = _execute(engine, invocation, context)
    assert session.status.value == "failed"
    assert len(frames) == 3
    assert source.started == ["first"]
    assert any(event.kind.value == "session_failed" for event in events)


def test_contact_revision_rejects_before_regrounding_inactive_rows():
    source = _Source()
    engine, invocation, context = _setup(source)
    session = engine.start(
        (invocation,), context, eligible_mask=torch.tensor([False, True])
    )
    with pytest.raises(ValueError, match="does not support invocation revisions"):
        session.revise_current(replace(invocation, revision=1), context=context)
    assert len(source.masks) == 1
    assert source.masks[0].tolist() == [False, True]
    assert len(session.plan_attempts) == 1


def test_preview_does_not_commit_reference_bookkeeping_and_snapshot_is_owned():
    source = _Source()
    engine, invocation, context = _setup(source)
    plan = engine.plan(invocation, context)
    assert source.started == []
    copy = plan.snapshot()
    copy.continuation.resolved.held_qpos.fill_(99)
    assert not bool((plan.continuation.resolved.held_qpos == 99).any())


def test_contact_sources_are_engine_local_and_do_not_leak_episode_lifetimes():
    source = _Source()
    engine, invocation, context = _setup(source)
    del source
    gc.collect()
    with pytest.raises(ValueError, match="expired or belongs"):
        engine.plan(invocation, context)


def test_offline_projection_cannot_fake_future_contact_observations():
    source = _Source()
    engine, invocation, context = _setup(source)
    with pytest.raises(ValueError, match="execution session"):
        engine.compile((invocation,), context)


def test_external_transform_cannot_drop_a_contact_continuation():
    source = _Source()
    engine, invocation, context = _setup(source)
    called = []

    def transform(request, current, plan):
        called.append(True)
        return engine.rebuild_plan_from_trajectory(
            request, current, plan, plan.joint_trajectory
        )

    with pytest.raises(ValueError, match="does not support plan transforms"):
        engine.compile((invocation,), context, plan_transform=transform)
    assert called == []
    assert source.started == []


@pytest.mark.parametrize("skill", [PickUp, Place])
def test_contact_nominal_rejection_precedes_external_source_acquisition(skill):
    class Nominal:
        def instantiate(self, *args, **kwargs):
            pytest.fail(
                "Unsupported contact replay must reject before acquiring external motion"
            )

    source = _GeometrySource()
    action = _bind_action(_motion_generator(), skill())
    engine = _ACTION_ENGINES[id(action)]
    engine.planning_services.register_contact_source("geometry", source)
    goal = (
        ContactPickUpGoal("geometry", (ContactPhase("grasp", "close"),))
        if skill is PickUp
        else ContactPlaceGoal("geometry", (ContactPhase("joint", "open"),))
    )
    with pytest.raises(ValueError, match="cannot replay nominal motion"):
        engine.plan(_invocation(action, goal), _context(), nominal_source=Nominal())
    assert source.started == []


def test_contact_path_rejects_replay_recovery_of_partially_executed_operation():
    source = _Source()
    engine, invocation, context = _setup(source)
    invocation = replace(invocation, recovery_policy=RecoveryPolicy(max_replans=1))
    with pytest.raises(ValueError, match="not replay"):
        engine.plan(invocation, context)


def test_contact_path_rejects_phase_path_overrides_before_observing_geometry():
    source = _Source()
    engine, invocation, context = _setup(source)
    policy = replace(
        invocation.motion_policy,
        interpolation_mode="fixed_grid",
        acceleration_limit=None,
        ik_seed_mode="sequential",
        phase_paths={"approach": "cartesian"},
    )
    with pytest.raises(ValueError, match="fixed sampling policy"):
        engine.plan(replace(invocation, motion_policy=policy), context)
    assert source.masks == []


def test_contact_path_timeout_covers_the_whole_invocation():
    source = _Source()
    engine, invocation, context = _setup(source)
    invocation = replace(
        invocation,
        recovery_policy=RecoveryPolicy(
            max_replans=0, max_skill_retries=0, skill_timeout=4 * context.control_dt
        ),
    )
    session, events, frames = _execute(engine, invocation, context)
    assert session.status.value == "failed"
    assert len(session.plan_attempts) == 2
    assert len(frames) < 6
    assert any(event.kind.value == "skill_timeout" for event in events)


def test_contact_path_cannot_replace_an_existing_rigid_held_relation():
    source = _Source()
    engine, invocation, context = _setup(source)
    held = _held(env_mask=torch.tensor([True, False]))
    context = replace(context, task=replace(context.task, held_objects={"arm": held}))
    plan = engine.plan(invocation, context)
    assert plan.plan_success.tolist() == [False, True]
    assert source.masks[-1].tolist() == [False, True]
    assert source.started == []


def test_contact_pickup_plans_pose_close_and_lift_without_rigid_projection():
    source = _GeometrySource()
    action = _bind_action(_motion_generator(), PickUp())
    engine = _ACTION_ENGINES[id(action)]
    engine.planning_services.register_contact_source("geometry", source)
    invocation = replace(
        _invocation(
            action, ContactPickUpGoal("geometry", (ContactPhase("grasp", "pinch"),))
        ),
        motion_policy=MotionPolicy(strategy="ik_interp", dynamic_collision_mode="off"),
        recovery_policy=RecoveryPolicy(max_replans=0, max_skill_retries=0),
        skill_options=PickUpOptions(
            pre_grasp_distance=0, lift_height=0, grasp_settle_steps=3
        ),
    )
    context = _context()
    plan = engine.plan(invocation, context)
    assert plan.plan_success.tolist() == [True, True]
    assert [segment.name for segment in plan.segments] == ["approach", "close", "lift"]
    close = next(segment for segment in plan.segments if segment.name == "close")
    ids = invocation.binding.endpoint("primary", "grasp").target.joint_ids
    torch.testing.assert_close(
        plan.joint_trajectory.positions[:, close.stop - 1, list(ids)],
        torch.full((2, len(ids)), 0.002),
    )
    assert source.started == []
    assert plan.expected_effects.is_empty
    assert context.task.get_held_object("arm") is None
    invocation = replace(
        invocation,
        goal=ContactPickUpGoal(
            "geometry",
            (ContactPhase("pose", "approach"), ContactPhase("grasp", "pinch")),
        ),
    )
    pose_plan = engine.plan(invocation, context)
    assert pose_plan.plan_success.tolist() == [True, True]
    assert pose_plan.continuation.phase_name == "approach"
    # The observed start and two authored waypoints all occupy the control grid.
    assert pose_plan.commands.frame_count == 3


def test_velocity_qualification_also_rejects_an_unsafe_continued_grasp(monkeypatch):
    source = _GeometrySource(closed=1.0)
    action = _bind_action(_motion_generator(), PickUp())
    engine = _ACTION_ENGINES[id(action)]
    engine.planning_services.register_contact_source("geometry", source)
    context = _context()
    monkeypatch.setattr(
        "embodichain.lab.sim.motion.limits.authored_joint_velocity_limits",
        lambda *_: torch.ones_like(context.robot.qpos),
    )
    invocation = replace(
        _invocation(
            action,
            ContactPickUpGoal(
                "geometry",
                (
                    ContactPhase("pose", "approach"),
                    ContactPhase("grasp", "close"),
                ),
            ),
        ),
        motion_policy=MotionPolicy(
            strategy="ik_interp",
            dynamic_collision_mode="off",
            sample_count=3,
            require_final_velocity=True,
            final_velocity_rejection="mask_only",
        ),
        recovery_policy=RecoveryPolicy(max_replans=0, max_skill_retries=0),
        skill_options=PickUpOptions(
            pre_grasp_distance=0, lift_height=0, hand_interp_steps=2
        ),
    )
    session, events, frames = _execute(engine, invocation, context)
    assert session.status.value == "failed"
    assert len(session.plan_attempts) == 2
    assert session.plan_attempts[-1].plan.plan_success.tolist() == [False, False]
    assert any(event.kind.value == "session_failed" for event in events)
    for frame in frames:
        for command in frame.commands:
            assert not bool((command.payload.positions == 1).any())


@pytest.mark.parametrize(
    "goal,phases",
    [
        (ContactPickUpGoal, (ContactPhase("joint", "x"),)),
        (ContactPlaceGoal, (ContactPhase("grasp", "x"),)),
    ],
)
def test_contact_goal_keeps_pickup_and_placement_intents_distinct(goal, phases):
    with pytest.raises(ValueError):
        goal("fixture", phases)
