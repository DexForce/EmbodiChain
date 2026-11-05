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

"""Canonical bounded Twist planning and logical outcome ownership."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import math
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    ControlPartCommandProfile,
    MotionPolicy,
    RecoveryPolicy,
    TwistAffordance,
    TwistGoal,
    TwistJointTarget,
    TwistOptions,
    ObjectSemantics,
    SceneEntityPose,
    SceneSnapshot,
    EntityState,
    ObservedArticulationJointState,
)
from embodichain.lab.sim.atomic_actions.primitives.twist_recovery import (
    TwistPlanningRecovery,
    TwistMeasuredFeedback,
    TwistRecoveryCursor,
    TwistRecoveryScope,
)
from embodichain.lab.sim.atomic_actions.twist_planning import TwistPlanningSession
from embodichain.toolkits.graspkit.pg_grasp.native_contact import NativeQueryScope
from embodichain.lab.sim.motion.revolute_query import (
    RevolutePathReport,
    RevoluteDependencyReport,
)
from .test_actions import _motion_generator, _context, _torch_interpolation


class _Query:
    source_axis = (0.0, 0.0, 1.0)

    def __init__(self, robot, scope, *, rejected=False, root=True):
        self.robot, self.scope = robot, scope
        self.rejected, self.root = rejected, root
        self.frames = []
        self.closes = 0

    def inspect_free_path(self, positions, frame_indices):
        self.frames.append(tuple(frame_indices))
        return RevolutePathReport(
            True,
            not self.rejected,
            -0.01 if self.rejected else 0.1,
            len(frame_indices),
            ("arm", "pad"),
        )

    def contact_metrics(self, qpos, initial):
        index = len(self.frames) - 1
        # The first-depth roll at pi/2 wins; deeper candidates must not outrank
        # the source's shallow-first ordering even if their numeric score is best.
        fractions = (0.2, 0.01, 0.03, 0.02)
        return (
            fractions[index % 4],
            0.002,
            float(torch.linalg.vector_norm(qpos - initial)),
        )

    def qualify_root_dependency(self, positions, **kwargs):
        return (
            RevoluteDependencyReport(True, "f" * 64, 0.003)
            if self.root
            else RevoluteDependencyReport(False, reason="unproven source")
        )

    def close(self):
        self.closes += 1


class _Provider:
    def __init__(self, robot, cursor, *, rejected=False):
        self.robot, self.cursor = robot, cursor
        self.queries = []
        self.rejected = rejected
        self.foreign = False

    def prepare(self, request, context):
        query = _Query(
            self.robot,
            NativeQueryScope(
                str(len(self.queries)), context.robot.timestamp, context.scene.version
            ),
            rejected=self.rejected,
        )
        self.queries.append(query)
        return TwistPlanningSession(
            self.cursor.snapshot(),
            query,
            "panel",
            "foreign" if self.foreign else request.invocation_id,
            request.revision,
            completed_hold_qpos=(
                torch.full_like(context.robot.qpos, 0.2)
                if self.cursor.snapshot().goal_complete
                else None
            ),
        )


def _setup(tmp_path, *, rejected=False):
    generator = _motion_generator(batch_size=1)
    robot = generator.robot
    path = tmp_path / "robot.urdf"
    path.write_text(
        "<robot>"
        + "".join(
            f'<joint name="{name}"><limit velocity="100"/></joint>'
            for name in robot.joint_names
        )
        + "</robot>"
    )
    robot.cfg = SimpleNamespace(fpath=str(path))
    robot.get_qvel_limits.side_effect = lambda name=None, joint_ids=None: torch.full(
        (1, len(robot.get_joint_ids(name) if joint_ids is None else joint_ids)), 100.0
    )
    robot.get_qpos_limits.side_effect = lambda name=None, **kw: torch.tensor(
        [[[-20.0, 20.0]] * len(robot.get_joint_ids(name))]
    )

    def ik(pose, joint_seed, **kwargs):
        positions = pose[:, :3, 3].detach().cpu().numpy()
        angles = Rotation.from_matrix(
            pose[:, :3, :3].detach().cpu().numpy()
        ).as_rotvec()
        return torch.ones(len(pose), dtype=torch.bool), torch.as_tensor(
            np.concatenate((positions, angles), 1), dtype=joint_seed.dtype
        )

    def fk(qpos=None, **kwargs):
        qpos = torch.zeros(1, 6) if qpos is None else qpos
        pose = torch.eye(4).repeat(len(qpos), 1, 1).to(qpos)
        pose[:, :3, 3] = qpos[:, :3]
        pose[:, :3, :3] = torch.as_tensor(
            Rotation.from_rotvec(qpos[:, 3:6].detach().cpu().numpy()).as_matrix(),
            dtype=qpos.dtype,
        )
        return pose

    robot.compute_ik.side_effect = ik
    robot.compute_fk.side_effect = fk
    robot.compute_batch_fk.side_effect = lambda qpos, **kwargs: fk(
        qpos.reshape(-1, qpos.shape[-1])
    ).reshape(*qpos.shape[:2], 4, 4)
    cursor = TwistRecoveryCursor(
        TwistRecoveryScope("program", "operation", "episode", 0)
    )
    provider = _Provider(robot, cursor, rejected=rejected)
    engine = AtomicActionEngine(
        generator,
        control_profiles={
            "hand": ControlPartCommandProfile.joint_positions(
                open=torch.zeros(2), grasp=torch.ones(2)
            )
        },
        twist_planning_provider=provider,
    )
    point = (0.000000123456789, 0.0, 0.005123456789)
    affordance = TwistAffordance(
        grasp_position=point,
        axis_origin=(0.0, 0.0, 0.0),
        twist_axis=torch.tensor([0.0, 0.0, 1.0]),
        joint_name="knob_joint",
        joint_limits=(-1.0, 1.0),
    )
    goal = TwistGoal(
        ObjectSemantics(entity_id="knob", geometry={}, affordance=affordance),
        SceneEntityPose("knob"),
        joint_target=TwistJointTarget("panel", "knob_joint", 0.3, axis_sign=1),
        contact_reference=point,
    )
    invocation = engine.make_invocation(
        "twist",
        goal,
        control_parts={"primary": {"motion": "arm", "grasp": "hand"}},
        motion_policy=MotionPolicy(
            sample_count=320,
            interpolation_mode="fixed_grid",
            ik_seed_mode="sequential",
            acceleration_limit=None,
            dynamic_collision_mode="off",
        ),
        recovery_policy=RecoveryPolicy(
            max_replans=0,
            max_action_retries=0,
            planning_recovery=TwistPlanningRecovery(),
        ),
        skill_options=TwistOptions(
            hand_interp_steps=12,
            max_twist_angle=math.radians(5),
            contact_bite_depth=0.006,
            contact_settle_steps=5,
        ),
        invocation_id="chunk_1",
    )
    scene = SceneSnapshot(
        timestamp=0.0,
        version=0,
        entities={
            "knob": EntityState(torch.eye(4)),
            "panel": EntityState(torch.eye(4)),
        },
        articulation_joints={
            ("panel", "knob_joint"): ObservedArticulationJointState(torch.zeros(1, 1))
        },
    )
    return engine, invocation, _context(scene=scene, batch_size=1), provider, cursor


def test_bounded_twist_uses_the_standard_engine_and_source_candidate_order(tmp_path):
    engine, invocation, context, provider, cursor = _setup(tmp_path)
    trace = []
    original = provider.robot.compute_ik.side_effect

    def digest(value):
        return hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()

    def observed_ik(*args, **kwargs):
        trace.append(
            {
                "pose": digest(kwargs["pose"]),
                "seed": digest(kwargs["joint_seed"]),
                "rng": digest(torch.get_rng_state()),
            }
        )
        torch.rand(1)
        np.random.random()
        random.random()
        return original(*args, **kwargs)

    provider.robot.compute_ik.side_effect = observed_ik
    torch.manual_seed(38)
    np.random.seed(38)
    random.seed(38)
    from embodichain.lab.sim.motion.search import search_cost_scope

    with search_cost_scope() as cost:
        plan = engine.plan(invocation, context)
    assert cost.counters["twist_candidate"] == 12
    expected = json.loads(
        (Path(__file__).parent / "data/bounded_twist_source_reference.json").read_text()
    )
    assert len(trace) == expected["ik_calls"]
    assert (
        hashlib.sha256(json.dumps(trace, sort_keys=True).encode()).hexdigest()
        == expected["trace_sha256"]
    )
    assert digest(plan.joint_trajectory.positions) == expected["positions_sha256"]
    assert digest(plan.joint_trajectory.dt) == expected["dt_sha256"]
    assert digest(torch.get_rng_state()) == expected["torch_rng"]
    assert (
        hashlib.sha256(
            np.random.get_state()[1].tobytes()
            + repr(np.random.get_state()[2:]).encode()
        ).hexdigest()
        == expected["numpy_rng"]
    )
    assert (
        hashlib.sha256(repr(random.getstate()).encode()).hexdigest()
        == expected["python_rng"]
    )
    assert plan.plan_success.tolist() == [True]
    assert len(provider.queries) == 1 and provider.queries[0].closes == 1
    assert len(provider.queries[0].frames) == 12
    outcome = plan.planning_recovery_outcome
    assert outcome.next_state.selected_index == 1
    assert outcome.next_state.order == (1, 5, 9)
    assert outcome.next_state.selected_bite_depth == pytest.approx(0.002)
    assert plan.joint_trajectory.positions.shape[1] == 330
    assert list(s.name for s in plan.segments) == [
        "approach",
        "reach",
        "close",
        "twist",
        "open",
        "retract",
    ]
    assert plan.scene_dependencies == ("panel",)
    assert cursor.snapshot().version == 0
    cursor.commit_planning(outcome)
    cursor.adopt_plan(plan)
    assert cursor.snapshot().version == 1


def test_table_rejection_keeps_first_logical_candidate_without_physical_success(
    tmp_path,
):
    engine, invocation, context, provider, cursor = _setup(tmp_path, rejected=True)
    plan = engine.plan(invocation, context)
    assert plan.plan_success.tolist() == [False]
    assert plan.planning_recovery_outcome.next_state.selected_index == 0
    assert len(provider.queries[0].frames) == 12 and provider.queries[0].closes == 1
    assert not any(frame.active_mask.any() for frame in plan.commands.frames)
    assert cursor.snapshot().version == 0
    cursor.commit_planning(plan.planning_recovery_outcome)
    assert cursor.snapshot().selected_index == 0


def test_bounded_twist_rejects_outer_retries_before_preparing_a_query(tmp_path):
    engine, invocation, context, provider, _ = _setup(tmp_path)
    invocation = replace(
        invocation, recovery_policy=replace(invocation.recovery_policy, max_replans=1)
    )
    with pytest.raises(ValueError, match="no outer"):
        engine.plan(invocation, context)
    assert provider.queries == []


def test_bounded_twist_closes_foreign_occurrence_query_without_planning(tmp_path):
    engine, invocation, context, provider, _ = _setup(tmp_path)
    provider.foreign = True
    with pytest.raises(ValueError, match="observation"):
        engine.plan(invocation, context)
    assert provider.queries[0].closes == 1
    provider.robot.compute_ik.assert_not_called()


def test_completed_cursor_uses_observed_drive_hold_without_new_candidates(tmp_path):
    engine, invocation, context, provider, cursor = _setup(tmp_path)
    original = engine.plan(invocation, context)
    outcome = original.planning_recovery_outcome
    cursor.commit_planning(outcome)
    cursor.adopt_plan(original)
    state = cursor.snapshot()
    cursor.commit_measurement(
        TwistMeasuredFeedback(
            state.scope,
            state.cursor_id,
            state.version,
            outcome.invocation_id,
            outcome.invocation_revision,
            outcome.plan_id,
            1.0,
            True,
            True,
            True,
            True,
            0.3,
            True,
        )
    )
    # Replaying a pre-feedback observation is rejected, even though the cursor
    # has already reached its externally assessed goal.
    with pytest.raises(ValueError, match="precedes"):
        engine.plan(invocation, context)
    fresh = replace(
        context,
        robot=replace(context.robot, timestamp=1.0),
        scene=replace(context.scene, timestamp=1.0),
    )
    before = provider.robot.compute_ik.call_count
    plan = engine.plan(replace(invocation, invocation_id="chunk_2"), fresh)
    assert provider.robot.compute_ik.call_count == before
    assert provider.queries[-1].frames == [] and provider.queries[-1].closes == 1
    hold_outcome = plan.planning_recovery_outcome
    before_hold = cursor.snapshot()
    assert replace(hold_outcome.next_state, version=before_hold.version) == before_hold
    cursor.commit_planning(hold_outcome)
    cursor.adopt_plan(plan)
    held = cursor.snapshot()
    decision = cursor.commit_measurement(
        TwistMeasuredFeedback(
            held.scope,
            held.cursor_id,
            held.version,
            hold_outcome.invocation_id,
            hold_outcome.invocation_revision,
            hold_outcome.plan_id,
            2.0,
            False,
            False,
            False,
            False,
            0.3,
            False,
        )
    )
    assert not decision.accepted
    assert not cursor.snapshot().goal_complete
    assert plan.joint_trajectory.positions.shape[1] == 1
    torch.testing.assert_close(
        plan.joint_trajectory.positions[:, 0], torch.full_like(context.robot.qpos, 0.2)
    )


def test_transformed_twist_rechecks_qualification_without_replanning(tmp_path):
    engine, invocation, context, provider, _ = _setup(tmp_path)
    calls = []
    original = []

    def transform(request, current, plan):
        calls.append(provider.robot.compute_ik.call_count)
        original.append(plan.joint_trajectory.positions.clone())
        provider.rejected = True
        return plan

    plan = engine.plan(invocation, context, plan_transform=transform)
    assert calls == [provider.robot.compute_ik.call_count]
    assert len(provider.queries) == 2
    assert all(query.closes == 1 for query in provider.queries)
    assert (
        len(provider.queries[0].frames) == 12 and len(provider.queries[1].frames) == 1
    )
    assert not plan.plan_success.any()
    assert not any(frame.active_mask.any() for frame in plan.commands.frames)
    torch.testing.assert_close(plan.rejected_candidate.positions, original[0])


def test_operation_session_binds_occurrences_and_separates_preview_feedback(tmp_path):
    from embodichain.lab.task_program.integrations.simulation.twist_session import (
        TwistOperationSession,
    )

    base, original, context, _, cursor = _setup(tmp_path)
    robot = base.robot
    queries, inputs = [], []

    def prepare(**kwargs):
        inputs.append(kwargs)
        query = _Query(robot, kwargs["scope"])
        queries.append(query)
        return query

    geometry = SimpleNamespace(robot=robot, prepare=prepare)
    scope_open = [True]

    def require_scope():
        if not scope_open[0]:
            raise RuntimeError("replaced episode scope")

    session = TwistOperationSession(
        robot,
        geometry,
        cursor.snapshot().scope,
        invocation_ids=("chunk_1", "chunk_2"),
        parent_entity_id="panel",
        child_entity_id="knob",
        joint_name="knob_joint",
        validate_observation=lambda context: None,
        validate_execution_scope=require_scope,
    )
    engine = AtomicActionEngine(
        base.motion_generator,
        control_profiles=base.control_profiles,
        twist_planning_provider=session,
    )
    invocation = engine.make_invocation(
        "twist",
        original.goal,
        control_parts={"primary": {"motion": "arm", "grasp": "hand"}},
        motion_policy=original.motion_policy,
        recovery_policy=original.recovery_policy,
        skill_options=original.skill_options,
        invocation_id="chunk_1",
    )
    context = replace(
        context, robot=replace(context.robot, root_pose=torch.eye(4)[None])
    )
    with pytest.raises(ValueError, match="declared operation"):
        engine.plan(replace(invocation, invocation_id="foreign"), context)
    assert not queries
    plan = engine.plan(invocation, context)
    assert queries[-1].closes == 1
    assert inputs[-1]["parent_pose"].shape == (4, 4)
    assert inputs[-1]["measured_joint_position"] == 0.0
    assert session.snapshot().version == 0
    preview = session.fork_preview()
    with pytest.raises(ValueError, match="another cursor scope"):
        preview.commit_planning(plan)
    session.commit_planning(plan)
    session.adopt_plan(plan)
    state = session.snapshot()
    outcome = plan.planning_recovery_outcome
    decision = session.commit_measurement(
        TwistMeasuredFeedback(
            state.scope,
            state.cursor_id,
            state.version,
            outcome.invocation_id,
            outcome.invocation_revision,
            outcome.plan_id,
            0.5,
            True,
            True,
            True,
            True,
            0.3,
            True,
        )
    )
    assert decision.accepted and decision.snapshot.goal_complete
    assert not preview.snapshot().goal_complete
    scope_open[0] = False
    with pytest.raises(RuntimeError, match="replaced episode"):
        session.reset_ready()
    scope_open[0] = True
    assert session.snapshot().goal_complete
    session.close()
    with pytest.raises(RuntimeError, match="closed"):
        engine.plan(invocation, context)
    assert preview.snapshot().version == 0
    preview.close()


def test_operation_session_prepares_the_real_motion_query_factory(
    tmp_path, monkeypatch
):
    # The shared fixture lives with the public Motion owner, outside this test package.
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "motion"))
    from _revolute_query_fixture import revolute_query_inputs
    from embodichain.lab.sim.motion.revolute_query import (
        RevolutePathQuery,
        RevolutePathQueryFactory,
    )
    from embodichain.lab.sim.atomic_actions import (
        PlanningContext,
        RobotObservation,
        TaskState,
    )
    from embodichain.lab.task_program.integrations.simulation.twist_session import (
        TwistOperationSession,
    )

    inputs = revolute_query_inputs(tmp_path)
    geometry = RevolutePathQueryFactory(
        inputs.simulation,
        inputs.robot,
        binding=inputs.binding,
        geometry=inputs.geometry,
        source_seal=inputs.seal,
        hand_geometry_cfg=inputs.hand,
        table_uid=inputs.table.uid,
        collision_link_names=inputs.names,
    )
    session = TwistOperationSession(
        inputs.robot,
        geometry,
        TwistRecoveryScope("program", "operation", "episode", 0),
        invocation_ids=("chunk",),
        parent_entity_id="panel",
        child_entity_id="knob",
        joint_name=inputs.binding.joint,
        validate_observation=lambda context: None,
        validate_execution_scope=lambda: None,
    )
    goal = TwistGoal(
        ObjectSemantics(
            TwistAffordance(
                grasp_position=(0, 0, 0), twist_axis=torch.tensor([0.0, 0.0, 1.0])
            ),
            {},
            entity_id="knob",
        ),
        SceneEntityPose("knob"),
        joint_target=TwistJointTarget("panel", inputs.binding.joint, 0.3, axis_sign=1),
    )
    qpos = torch.zeros(1, 3)
    context = PlanningContext(
        RobotObservation(0.0, qpos, torch.zeros_like(qpos), root_pose=inputs.root),
        TaskState(1, "cpu"),
        SceneSnapshot(
            0.0,
            0,
            {"panel": EntityState(inputs.parent), "knob": EntityState(inputs.child)},
            articulation_joints={
                ("panel", inputs.binding.joint): ObservedArticulationJointState(
                    torch.zeros(1, 1)
                )
            },
        ),
        torch.tensor([0]),
        control_dt=0.04,
    )
    try:
        prepared = session.prepare(
            SimpleNamespace(
                skill_id="twist", goal=goal, invocation_id="chunk", revision=0
            ),
            context,
        )
        assert type(prepared.query) is RevolutePathQuery
        try:
            assert prepared.query.robot is inputs.robot
            assert prepared.query.inspect_free_path(
                inputs.positions, tuple(inputs.frames)
            ).supported
        finally:
            prepared.query.close()
        assert session.snapshot().version == 0
    finally:
        session.close()
        geometry.close()


def test_operation_session_closes_query_when_observation_guard_is_replaced(tmp_path):
    from embodichain.lab.task_program.integrations.simulation.twist_session import (
        TwistOperationSession,
    )
    from embodichain.lab.sim.atomic_actions.invocation import ResolvedActionRequest

    base, invocation, context, _, cursor = _setup(tmp_path)
    request = ResolvedActionRequest(
        **{
            name: getattr(invocation, name)
            for name in (
                "skill_id",
                "goal",
                "binding",
                "motion_policy",
                "tracking_policy",
                "recovery_policy",
                "skill_options",
                "invocation_id",
                "revision",
            )
        }
    )
    context = replace(
        context, robot=replace(context.robot, root_pose=torch.eye(4)[None])
    )
    queries = []

    def prepare(**kwargs):
        query = _Query(base.robot, kwargs["scope"])
        queries.append(query)
        return query

    def guard(context):
        if queries:
            raise ValueError("replaced observation owner")

    session = TwistOperationSession(
        base.robot,
        SimpleNamespace(robot=base.robot, prepare=prepare),
        cursor.snapshot().scope,
        invocation_ids=("chunk_1",),
        parent_entity_id="panel",
        child_entity_id="knob",
        joint_name="knob_joint",
        validate_observation=guard,
        validate_execution_scope=lambda: None,
    )
    with pytest.raises(ValueError, match="replaced observation"):
        session.prepare(request, context)
    assert len(queries) == 1 and queries[0].closes == 1
    assert session.snapshot().version == 0


def test_twist_registry_installs_lifecycle_before_execution_and_rejects_overlapping_owners(
    tmp_path,
):
    from embodichain.lab.task_program.integrations.simulation.twist_session import (
        TwistOperationSession,
        TwistOperationRegistry,
    )
    from embodichain.lab.task_program.runtime import TaskProgramPlanRequest
    from embodichain.lab.task_program.semantics.calls import RegisteredSemanticCall

    base, original, context, _, cursor = _setup(tmp_path)
    queries = []

    def prepare(**kwargs):
        query = _Query(base.robot, kwargs["scope"])
        queries.append(query)
        return query

    geometry = SimpleNamespace(robot=base.robot, prepare=prepare)

    def operation(scope):
        return TwistOperationSession(
            base.robot,
            geometry,
            scope,
            invocation_ids=("chunk_1",),
            parent_entity_id="panel",
            child_entity_id="knob",
            joint_name="knob_joint",
            validate_observation=lambda context: None,
            validate_execution_scope=lambda: None,
        )

    live = operation(cursor.snapshot().scope)
    with pytest.raises(ValueError, match="more than one operation owner"):
        TwistOperationRegistry(
            (live, operation(replace(live.scope, operation_id="other")))
        )
    registry = TwistOperationRegistry((live,))
    preview = registry.fork_preview()
    engine = AtomicActionEngine(
        base.motion_generator,
        control_profiles=base.control_profiles,
        twist_planning_provider=registry,
    )
    invocation = engine.make_invocation(
        "twist",
        original.goal,
        control_parts={"primary": {"motion": "arm", "grasp": "hand"}},
        motion_policy=original.motion_policy,
        recovery_policy=original.recovery_policy,
        skill_options=original.skill_options,
        invocation_id="chunk_1",
    )
    request = TaskProgramPlanRequest(
        "workflow", 0, 0, RegisteredSemanticCall(call_id="test.twist"), invocation
    )
    from embodichain.lab.sim.sensors.contact.sampling import ContactEvidenceSampler
    from embodichain.lab.sim.sensors.contact.observation import (
        JointContactObserver,
    )
    from embodichain.lab.task_program.integrations.simulation.control_observation import (
        ControlContactActors,
        ControlObservationWindow,
    )
    from embodichain.lab.sim.atomic_actions.articulation_evidence import (
        TwistEventTarget,
    )

    physical_time, qpos = [0.0], torch.zeros(1)
    data = {
        "is_valid": torch.ones(1, 1, dtype=torch.bool),
        "user_ids": torch.tensor([[[1, 2]]]),
        "impulse": torch.zeros(1, 1),
        "position": torch.zeros(1, 1, 3),
        "normal": torch.zeros(1, 1, 3),
        "distance": torch.zeros(1, 1),
    }
    source = SimpleNamespace(
        num_instances=1,
        contact_capabilities=SimpleNamespace(geometry=True, impulse=True),
        dropped_contacts=0,
        get_data=lambda: data,
        update=lambda: None,
    )
    observer = JointContactObserver(
        ContactEvidenceSampler(source),
        lambda: qpos,
        lambda: physical_time[0],
        joint_velocity=lambda: torch.zeros(1),
    )
    generation = observer.begin_window("prepare")
    observer.capture(refresh=True)
    window = ControlObservationWindow(
        observer,
        generation,
        ControlContactActors(1, 4, (2,)),
        legacy_time_origin=0,
        physics_dt=0.004,
        current_time=lambda: physical_time[0],
        validate_scope=lambda: live.snapshot(),
    )
    live.bind_control_observation(window, TwistEventTarget(0.3, (-1.0, 1.0)))
    observer.set_phase("twist")
    lifecycle = registry.create_plan_lifecycle(request, engine=engine)
    preview_lifecycle = preview.create_plan_lifecycle(request, engine=engine)
    context = replace(
        context, robot=replace(context.robot, root_pose=torch.eye(4)[None])
    )
    execution = engine.start((invocation,), context, plan_lifecycle=lifecycle)
    state = live.snapshot()
    assert state.version == 1 and not state.goal_complete
    for index in range(1, 133):
        token = observer.before_sample()
        physical_time[0] += 0.004
        qpos[0] = 0 if index == 1 else 0.3
        data["impulse"].fill_(0.1 if index <= 2 else 0.0)
        observer.after_sample(0.004, token)
    receipt = live.observe_chunk(
        observed_qpos=float(qpos[0]),
        stable_positions=[float(qpos[0])] * 13,
        stable_velocities=[0.0] * 13,
        feedback_timestamp=physical_time[0],
    )
    assert (
        receipt["accepted"]
        and receipt["goal_complete"]
        and receipt["observation_available"]
    )
    assert receipt["sample_count"] == 132
    assert not receipt["observation_errors"]
    assert not preview_lifecycle.snapshot().goal_complete
    assert queries[-1].closes == 1
    assert execution.abort_plan_lifecycle("explicit cancellation") is None
    with pytest.raises(RuntimeError, match="closed"):
        live.snapshot()
    preview.close()
    registry.close()


def test_bounded_twist_negative_control_keeps_target_identity_and_zero_arc(tmp_path):
    engine, invocation, context, provider, _ = _setup(tmp_path)
    disabled = replace(
        invocation,
        skill_options=replace(invocation.skill_options, rotation_enabled=False),
    )
    plan = engine.plan(disabled, context)
    assert plan.plan_success.all()
    segment = next(value for value in plan.segments if value.name == "twist")
    positions = plan.joint_trajectory.positions[:, segment.start : segment.stop, :6]
    torch.testing.assert_close(
        positions, positions[:, :1].expand_as(positions), atol=1e-6, rtol=0
    )
    assert (
        disabled.goal.joint_target.position
        == invocation.goal.joint_target.position
        == 0.3
    )
    assert provider.queries[-1].closes == 1
    assert not plan.planning_recovery_outcome.next_state.goal_complete
