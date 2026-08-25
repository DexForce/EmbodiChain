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

"""Focused contracts for the public atomic-action adapter."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from embodichain.gen_sim.action_engine.capabilities import HeldObjectHandOver
from embodichain.gen_sim.action_engine.runtime import actions
from embodichain.gen_sim.action_engine.runtime.atomic_compat import (
    ExactTargetMoveHeldObject,
)
from embodichain.gen_sim.action_engine.runtime.actions import AtomicActionAdapter
from embodichain.gen_sim.action_engine.runtime.models import (
    ActionOutcome,
    GroundedAction,
)
from embodichain.gen_sim.action_engine.runtime.state import ExecutionState
from embodichain.lab.sim.atomic_actions import (
    Affordance,
    ActionBinding,
    ActionPlan,
    AntipodalAffordance,
    AxisAlignAffordance,
    AxisAlignGoal,
    CoordinatedPickGoal,
    EndEffectorPoseGoal,
    EntityState,
    GraspGoal,
    HeldObjectPoseGoal,
    HeldObjectState,
    JointPositionGoal,
    ObjectSemantics,
    PlannerDiagnostics,
    RecoveryPolicy,
    RuntimeCommandFrame,
    SceneSnapshot,
    StateDelta,
    TimedCommandSequence,
    TimedTrajectory,
    TrackingPolicy,
)
from embodichain.lab.sim.planners import CuroboPlannerCfg
from embodichain.toolkits.graspkit.pg_grasp import AntipodalGraspPoseGenerator


class _MeshEntity:
    def get_vertices(self, *, env_ids: list[int], scale: bool) -> torch.Tensor:
        assert env_ids == [0]
        assert scale
        return torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )

    def get_triangles(self, *, env_ids: list[int]) -> torch.Tensor:
        assert env_ids == [0]
        return torch.tensor([[0, 1, 2]], dtype=torch.int64)


def _cuboid_vertices(x: float, y: float, z: float) -> torch.Tensor:
    return torch.tensor(
        [
            [sx * x, sy * y, sz * z]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ],
        dtype=torch.float32,
    )


def _rotation_x(degrees: float) -> torch.Tensor:
    angle = torch.deg2rad(torch.tensor(degrees, dtype=torch.float32))
    rotation = torch.eye(3)
    rotation[1, 1] = torch.cos(angle)
    rotation[1, 2] = -torch.sin(angle)
    rotation[2, 1] = torch.sin(angle)
    rotation[2, 2] = torch.cos(angle)
    return rotation


def _rotation_z(degrees: float) -> torch.Tensor:
    angle = torch.deg2rad(torch.tensor(degrees, dtype=torch.float32))
    rotation = torch.eye(3)
    rotation[0, 0] = torch.cos(angle)
    rotation[0, 1] = -torch.sin(angle)
    rotation[1, 0] = torch.sin(angle)
    rotation[1, 1] = torch.cos(angle)
    return rotation


class _PoseEntity:
    def __init__(self, pose: torch.Tensor) -> None:
        self.pose = pose

    def get_local_pose(self, *, to_matrix: bool) -> torch.Tensor:
        assert to_matrix
        return self.pose.clone()


class _PlannerRobot:
    uid = "test_robot"
    dof = 8

    _ids = {
        "physical_left_arm": [0, 1],
        "physical_left_eef": [2, 3],
        "physical_right_arm": [4, 5],
        "physical_right_eef": [6, 7],
    }
    control_parts = _ids

    def get_joint_ids(self, *, name: str) -> list[int]:
        return list(self._ids[name])

    def get_control_part_base_pose(self, *, name: str, to_matrix: bool) -> torch.Tensor:
        assert to_matrix
        pose = torch.eye(4).repeat(2, 1, 1)
        pose[:, 1, 3] = 0.3 if name == "physical_left_arm" else -0.3
        return pose

    def compute_batch_ik(
        self,
        *,
        pose: torch.Tensor,
        name: str,
        joint_seed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del pose, name
        return torch.ones(joint_seed.shape[:2], dtype=torch.bool), joint_seed.clone()


def _commands_for(trajectory: TimedTrajectory) -> TimedCommandSequence:
    """Build timing-only frames for retained test trajectories."""
    active = torch.ones(
        trajectory.batch_size,
        dtype=torch.bool,
        device=trajectory.positions.device,
    )
    frames = tuple(
        RuntimeCommandFrame(
            commands=(),
            active_mask=active,
            env_ids=trajectory.env_ids,
            hold_duration=trajectory.dt[:, index],
        )
        for index in range(trajectory.waypoint_count)
    )
    return TimedCommandSequence(frames=frames, env_ids=trajectory.env_ids)


class _FakeEngine:
    """Minimal endpoint-binding and planning surface for adapter unit tests."""

    binding_owner_id = "action-engine-test"

    def __init__(self, plan=None) -> None:
        self._plan = plan

    def bind_control_parts(self, _skill_id, _endpoints) -> ActionBinding:
        return ActionBinding(owner_id=self.binding_owner_id)

    def plan(self, invocation, context) -> ActionPlan:
        if self._plan is None:
            raise AssertionError("This fake engine has no planning callback.")
        return self._plan(invocation, context)


def test_adapter_registers_gen_sim_compat_actions(
    monkeypatch: Any,
) -> None:
    registered: list[tuple[type, bool]] = []

    class Engine:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def register(self, action: Any, *, replace: bool = False) -> None:
            registered.append((type(action), replace))

    adapter = AtomicActionAdapter(_planner_env())
    monkeypatch.setattr(actions, "AtomicActionEngine", Engine)
    monkeypatch.setattr(adapter, "_generator", lambda: object())
    monkeypatch.setattr(adapter, "_control_profiles", lambda: {})
    monkeypatch.setattr(adapter, "_grasp_pose_generators", lambda **_kwargs: {})

    engine = adapter._engine()

    assert isinstance(engine, Engine)
    assert registered == [
        (ExactTargetMoveHeldObject, True),
        (HeldObjectHandOver, True),
    ]


def test_free_yaw_search_uses_an_internal_reachability_sample_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _planner_env()
    adapter = AtomicActionAdapter(env)
    target = torch.eye(4).repeat(2, 1, 1)
    target[:, :3, 3] = torch.tensor([0.1, -0.2, 0.9])
    semantics = ObjectSemantics(label="can", geometry={}, affordance=Affordance())
    held = HeldObjectState(
        semantics=semantics,
        object_to_eef=torch.eye(4).repeat(2, 1, 1),
        grasp_xpos=torch.eye(4).repeat(2, 1, 1),
    )
    state = ExecutionState(
        last_qpos=torch.zeros(2, 8),
        held_objects={"physical_left_arm": held},
    )
    grounded = GroundedAction(
        "MoveHeldObject",
        "left_arm",
        "arm",
        HeldObjectPoseGoal(object_target_pose=target),
        {},
        target_object_pose=target,
        allow_yaw_search=True,
    )

    def compute_batch_ik(
        *,
        pose: torch.Tensor,
        name: str,
        joint_seed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert name == "physical_left_arm"
        assert pose.shape[:2] == (2, 8)
        success = torch.zeros(2, 8, dtype=torch.bool)
        success[:, 3] = True
        return success, joint_seed.clone()

    monkeypatch.setattr(env.robot, "compute_batch_ik", compute_batch_ik)

    selected = adapter._select_transport_yaw(grounded, state)

    assert selected.target_object_pose is not None
    torch.testing.assert_close(selected.target_object_pose[:, :3, 3], target[:, :3, 3])
    torch.testing.assert_close(
        selected.target_object_pose[:, :3, :3],
        torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]).repeat(
            2, 1, 1
        ),
        atol=1.0e-6,
        rtol=1.0e-6,
    )

    def all_yaws_reachable(
        *,
        pose: torch.Tensor,
        name: str,
        joint_seed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del pose, name
        qpos = joint_seed.clone()
        qpos[:, 0] += 1.0
        return torch.ones(2, 8, dtype=torch.bool), qpos

    monkeypatch.setattr(env.robot, "compute_batch_ik", all_yaws_reachable)

    minimum_rotation = adapter._select_transport_yaw(grounded, state)

    assert minimum_rotation.target_object_pose is not None
    torch.testing.assert_close(minimum_rotation.target_object_pose, target)


def test_adapter_lowers_axis_align_from_live_pose_with_a_stable_seed() -> None:
    vertices = torch.tensor(
        [[x, y, z] for x in (-0.03, 0.03) for y in (-0.06, 0.06) for z in (-0.03, 0.03)]
    )
    semantics = ObjectSemantics(
        label="can",
        entity_id="can",
        geometry={},
        affordance=AxisAlignAffordance(
            mesh_vertices=vertices,
            mesh_triangles=torch.tensor([[0, 1, 2]]),
            internal_axis=torch.tensor([0.0, 1.0, 0.0]),
        ),
    )

    live_pose = torch.eye(4).repeat(2, 1, 1)
    live_pose[:, 2, 3] = 1.10
    parked_pose = live_pose.clone()
    parked_pose[:, 2, 3] -= 100.0
    sampled_poses: list[torch.Tensor] = []
    sampled_random_values: list[float] = []

    class Generator:
        def get_valid_grasp_poses(self, **kwargs: Any):
            poses = kwargs["obj_poses"].clone()
            sampled_poses.append(poses)
            sampled_random_values.append(float(torch.rand(())))
            return [
                (pose.unsqueeze(0), torch.tensor([0.1])) for pose in poses.unbind(dim=0)
            ]

    adapter = AtomicActionAdapter(_planner_env())
    adapter._atomic_engine = SimpleNamespace(
        grasp_pose_generators={"physical_left_eef": Generator()}
    )
    grounded = GroundedAction(
        "AxisAlign",
        "left_arm",
        "arm",
        AxisAlignGoal(semantics=semantics),
        {},
        object_pose=live_pose,
        object_uid="can",
    )
    contexts = [
        SimpleNamespace(
            robot=SimpleNamespace(qpos=torch.zeros(2, 8)),
            scene=SceneSnapshot(
                timestamp=0.0,
                version=version,
                entities={"can": EntityState(parked_pose)},
            ),
        )
        for version in (3, 97)
    ]

    adaptations = [
        adapter._adapt_axis_align_body_grasps(
            grounded,
            context,
            adapter.capabilities.get("AxisAlign"),
        )
        for context in contexts
    ]
    adapted_items = adaptations[0]
    adapted = adapted_items[0]

    assert len(adapted_items) == 1
    assert isinstance(adapted.target, AxisAlignGoal)
    assert adapted.target.semantics.entity_id is None
    assert adapted.target.grasp_xpos is not None
    assert adapted.motion_policy["body_grasp"]["long_axis_index"] == 1
    assert adapted.motion_policy["body_grasp"]["candidate_counts"] == [1, 1]
    assert len(sampled_poses) == 2
    torch.testing.assert_close(sampled_poses[0], live_pose)
    torch.testing.assert_close(sampled_poses[1], live_pose)
    assert not torch.equal(sampled_poses[0], parked_pose)
    assert sampled_random_values[0] == sampled_random_values[1]
    torch.testing.assert_close(
        adaptations[0][0].target.grasp_xpos,
        adaptations[1][0].target.grasp_xpos,
    )


def test_axis_align_body_grasp_does_not_fall_back_to_scene_snapshot_pose() -> None:
    vertices = torch.tensor(
        [[x, y, z] for x in (-0.03, 0.03) for y in (-0.06, 0.06) for z in (-0.03, 0.03)]
    )
    grounded = GroundedAction(
        "AxisAlign",
        "left_arm",
        "arm",
        AxisAlignGoal(
            semantics=ObjectSemantics(
                label="can",
                geometry={},
                affordance=AxisAlignAffordance(
                    mesh_vertices=vertices,
                    mesh_triangles=torch.tensor([[0, 1, 2]]),
                    internal_axis=torch.tensor([0.0, 1.0, 0.0]),
                ),
            )
        ),
        {},
        object_uid="can",
    )
    parked_pose = torch.eye(4).repeat(2, 1, 1)
    parked_pose[:, 2, 3] = -98.9
    context = SimpleNamespace(
        scene=SceneSnapshot(
            timestamp=0.0,
            version=3,
            entities={"can": EntityState(parked_pose)},
        )
    )
    adapter = AtomicActionAdapter(_planner_env())

    with pytest.raises(ValueError, match="grounded live object pose"):
        adapter._adapt_axis_align_body_grasps(
            grounded,
            context,
            adapter.capabilities.get("AxisAlign"),
        )


def _planner_env(
    *,
    table: Any | None = None,
    rigid_objects: dict[str, Any] | None = None,
) -> SimpleNamespace:
    entities = dict(rigid_objects or {})
    if table is not None:
        entities["table"] = table
    return SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        robot=_PlannerRobot(),
        sim=SimpleNamespace(get_rigid_object=entities.get),
        left_arm_joints=[0, 1],
        left_eef_joints=[2, 3],
        right_arm_joints=[4, 5],
        right_eef_joints=[6, 7],
        open_state=torch.zeros(2),
        close_state=torch.ones(2),
        get_agent_arm_control_part=lambda is_left: (
            "physical_left_arm" if is_left else "physical_right_arm"
        ),
        get_agent_eef_control_part=lambda is_left: (
            "physical_left_eef" if is_left else "physical_right_eef"
        ),
    )


def test_semantics_prewarms_vhacd_cache_before_affordance(
    monkeypatch: Any,
) -> None:
    """The lazy shared checker must see V-HACD's pickle, never create CoACD."""
    events: list[str] = []
    observed: dict[str, Any] = {}
    entity = _MeshEntity()
    env = SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        sim=SimpleNamespace(
            get_rigid_object=lambda uid: entity if uid == "cube" else None
        ),
        agent_grasp_runtime_defaults={"max_decomposition_hulls": 8},
    )

    def fake_prepare(**kwargs: Any) -> SimpleNamespace:
        events.append("cache")
        observed.update(kwargs)
        return SimpleNamespace(status="hit")

    def fake_affordance(**kwargs: Any) -> Affordance:
        events.append("affordance")
        observed["affordance_kwargs"] = kwargs
        return Affordance()

    monkeypatch.setattr(
        actions,
        "ensure_vhacd_grasp_collision_cache",
        fake_prepare,
    )
    monkeypatch.setattr(actions, "AntipodalAffordance", fake_affordance)

    adapter = AtomicActionAdapter(env)
    first = adapter.semantics("cube")
    second = adapter.semantics("cube")

    assert first is second
    assert first.entity_id is None
    assert events == ["cache", "affordance"]
    assert observed["max_decomposition_hulls"] == 8
    assert observed["mesh_vertices"].dtype == torch.float32
    assert observed["mesh_triangles"].dtype == torch.int64
    assert set(observed["affordance_kwargs"]) == {
        "object_label",
        "mesh_vertices",
        "mesh_triangles",
    }


def test_planner_policy_uses_curobo_for_single_arm_and_ik_for_dual_arm() -> None:
    adapter = AtomicActionAdapter(_planner_env())
    adapter._atomic_engine = _FakeEngine()
    goal = JointPositionGoal(target=torch.zeros(2, 2))

    single = adapter._invocation(
        GroundedAction("MoveJoints", "left_arm", "arm", goal, {}),
        adapter.capabilities.get("MoveJoints"),
    )
    coordinated_goal = CoordinatedPickGoal(
        semantics=ObjectSemantics(
            label="tray",
            geometry={},
            affordance=AntipodalAffordance(),
        ),
        object_target_pose=torch.eye(4),
        object_initial_pose=torch.eye(4),
    )
    coordinated = adapter._invocation(
        GroundedAction(
            "CoordinatedPickment",
            "coordinated",
            "arm",
            coordinated_goal,
            {},
        ),
        adapter.capabilities.get("CoordinatedPickment"),
    )
    hand = adapter._invocation(
        GroundedAction("MoveJoints", "left_arm", "hand", goal, {}),
        adapter.capabilities.get("MoveJoints"),
    )

    assert adapter.planner_policy["backend"] == "curobo"
    assert single.motion_policy.strategy == "motion_gen"
    assert coordinated.motion_policy.strategy == "ik_interp"
    assert torch.allclose(
        coordinated.skill_options.left_to_right_arm_direction,
        torch.tensor([0.0, -1.0, 0.0]),
    )
    assert hand.motion_policy.strategy == "ik_interp"


def test_coordinated_pickment_uses_engine_scoped_grasp_generator() -> None:
    adapter = AtomicActionAdapter(_planner_env())
    adapter._atomic_engine = _FakeEngine()
    affordance = AntipodalAffordance()
    goal = CoordinatedPickGoal(
        semantics=ObjectSemantics(
            label="tray",
            geometry={},
            affordance=affordance,
        ),
        object_target_pose=torch.eye(4),
        object_initial_pose=torch.eye(4),
    )
    grounded = GroundedAction(
        "CoordinatedPickment",
        "coordinated",
        "arm",
        goal,
        {
            "middle_empty_ratio": 0.7,
            "is_filter_ground_collision": False,
        },
    )

    invocation = adapter._invocation(
        grounded,
        adapter.capabilities.get("CoordinatedPickment"),
    )

    scoped_affordance = invocation.goal.semantics.affordance
    assert isinstance(scoped_affordance, AntipodalAffordance)
    assert scoped_affordance is affordance
    assert invocation.skill_options.middle_empty_ratio == pytest.approx(0.7)


def _coordinated_grounded(
    rotation: torch.Tensor,
    *,
    vertices: torch.Tensor | None = None,
) -> GroundedAction:
    object_pose = torch.eye(4).repeat(2, 1, 1)
    object_pose[:, :3, :3] = rotation
    object_pose[:, :3, 3] = torch.tensor([0.05, 0.0, 0.75])
    mesh_vertices = _cuboid_vertices(0.03, 0.04, 0.20) if vertices is None else vertices
    goal = CoordinatedPickGoal(
        semantics=ObjectSemantics(
            label="test_object",
            geometry={},
            affordance=AntipodalAffordance(
                object_label="test_object",
                mesh_vertices=mesh_vertices,
                mesh_triangles=torch.tensor([[0, 1, 2]], dtype=torch.int64),
            ),
        ),
        object_target_pose=object_pose.clone(),
        object_initial_pose=object_pose.clone(),
    )
    return GroundedAction(
        "CoordinatedPickment",
        "coordinated",
        "arm",
        goal,
        {"middle_empty_ratio": 0.4},
        object_pose=object_pose,
        object_uid="test_object",
    )


def test_coordinated_pickment_geometry_candidates_are_live_and_continuous() -> None:
    adapter = AtomicActionAdapter(_planner_env())
    capability = adapter.capabilities.get("CoordinatedPickment")

    vertical = adapter._adapt_coordinated_pickment_grasps(
        _coordinated_grounded(torch.eye(3)), capability
    )
    tilted = adapter._adapt_coordinated_pickment_grasps(
        _coordinated_grounded(_rotation_x(45.0)), capability
    )
    horizontal = adapter._adapt_coordinated_pickment_grasps(
        _coordinated_grounded(_rotation_x(90.0)), capability
    )
    yawed = adapter._adapt_coordinated_pickment_grasps(
        _coordinated_grounded(_rotation_z(90.0) @ _rotation_x(90.0)),
        capability,
    )

    preferred = [
        candidates[0].cfg["middle_empty_ratio"]
        for candidates in (vertical, tilted, horizontal)
    ]
    assert preferred[0] < preferred[1] < preferred[2]
    assert yawed[0].cfg["middle_empty_ratio"] == pytest.approx(preferred[0])
    for candidates in (vertical, tilted, horizontal, yawed):
        assert candidates
        assert torch.allclose(
            candidates[0].cfg["left_to_right_arm_direction"],
            torch.tensor([0.0, -1.0, 0.0]),
        )
        assert torch.allclose(
            candidates[0].cfg["approach_direction"],
            torch.tensor([0.0, 0.0, -1.0]),
        )


def test_coordinated_pickment_geometry_candidates_are_deterministic_for_tray() -> None:
    adapter = AtomicActionAdapter(_planner_env())
    capability = adapter.capabilities.get("CoordinatedPickment")
    grounded = _coordinated_grounded(
        _rotation_z(31.0),
        vertices=_cuboid_vertices(0.20, 0.14, 0.02),
    )

    first = adapter._adapt_coordinated_pickment_grasps(grounded, capability)
    second = adapter._adapt_coordinated_pickment_grasps(grounded, capability)

    assert [item.cfg["middle_empty_ratio"] for item in first] == pytest.approx(
        [item.cfg["middle_empty_ratio"] for item in second]
    )
    assert (
        first[0].motion_policy["coordinated_grasp"]
        == second[0].motion_policy["coordinated_grasp"]
    )


def test_grasp_generators_follow_mainline_service_contract() -> None:
    adapter = AtomicActionAdapter(_planner_env())

    generators = adapter._grasp_pose_generators()

    assert set(generators) == {"physical_left_eef", "physical_right_eef"}
    generator = generators["physical_left_eef"]
    assert generators["physical_right_eef"] is generator
    assert isinstance(generator, AntipodalGraspPoseGenerator)
    assert generator.algorithm_cfg.sample_count == 10000
    assert generator.algorithm_cfg.approach_direction_samples == 4
    assert generator.algorithm_cfg.max_candidates == 500
    assert generator.collision_cfg.max_decomposition_hulls == 16
    assert generator.collision_cfg.filter_ground_collision is True


def test_coordinated_grasp_generator_honors_ground_filter_policy() -> None:
    adapter = AtomicActionAdapter(_planner_env())

    generators = adapter._grasp_pose_generators(filter_ground_collision=False)

    assert generators["physical_left_eef"] is generators["physical_right_eef"]
    assert (
        generators["physical_left_eef"].collision_cfg.filter_ground_collision is False
    )


def test_retreat_uses_row_local_motion_planner_reachability_search(
    monkeypatch: Any,
) -> None:
    env = _planner_env()
    adapter = AtomicActionAdapter(env)
    reference = torch.eye(4).repeat(2, 1, 1)
    reference[:, 2, 3] = 1.05
    requested = reference.clone()
    requested[:, 2, 3] = 1.35
    height_thresholds = torch.tensor([1.24, 1.00])
    attempted_targets: list[torch.Tensor] = []

    def plan(invocation: Any, _context: Any) -> ActionPlan:
        target = invocation.goal.xpos.clone()
        attempted_targets.append(target)
        height_reachable = target[:, 2, 3] <= height_thresholds
        baseward_reachable = target[:, 1, 3] < -0.05
        success = height_reachable | baseward_reachable
        terminal = target[:, 2, 3, None].repeat(1, 8)
        positions = torch.stack((torch.zeros_like(terminal), terminal), dim=1)
        trajectory = TimedTrajectory.from_uniform_step(
            positions,
            env_ids=torch.arange(2),
            step_dt=0.01,
        )
        return ActionPlan(
            skill_id="move_end_effector",
            plan_success=success,
            commands=_commands_for(trajectory),
            joint_trajectory=trajectory,
            recovery_policy=RecoveryPolicy(),
            tracking_policy=TrackingPolicy.timed(),
            planned_scene_version=0,
            planned_collision_world_revision=(0, 0),
            diagnostics=PlannerDiagnostics(backend="fake"),
            expected_effects=StateDelta(),
        )

    monkeypatch.setattr(adapter, "_engine", lambda: _FakeEngine(plan))
    grounded = GroundedAction(
        "MoveEndEffector",
        "right_arm",
        "arm",
        EndEffectorPoseGoal(xpos=requested),
        {
            "sample_interval": 10,
            "retreat_height": 0.30,
            "minimum_retreat_height": 0.05,
            "retreat_distance": 0.10,
        },
        motion_policy={
            "collision_safety": "required",
            "retreat_reachability_search": True,
            "retreat_reference_pose": reference,
            "minimum_retreat_height": 0.05,
            "retreat_distance": 0.10,
        },
    )

    outcome = adapter.plan(
        grounded,
        ExecutionState(last_qpos=torch.zeros(2, 8)),
    )

    assert len(attempted_targets) > 1
    assert bool(outcome.success.all())
    selected_z = outcome.grounded.target.xpos[:, 2, 3]
    assert selected_z.tolist() == pytest.approx([1.20, 1.35])
    assert outcome.grounded.target.xpos[:, 1, 3].tolist() == pytest.approx([0.0, -0.10])
    search = outcome.planner_trace["reachability_search"]
    assert search["strategy"] == "bounded_motion_planner"
    assert search["selected_target_z"].tolist() == pytest.approx([1.20, 1.35])
    assert len(search["attempts"]) == len(attempted_targets)


def test_curobo_generator_receives_generated_static_obstacles(
    monkeypatch: Any,
) -> None:
    table = object()
    can = object()
    captured: dict[str, Any] = {}

    def fake_motion_generator(*, cfg: Any) -> object:
        captured["cfg"] = cfg
        return object()

    monkeypatch.setattr(actions, "MotionGenerator", fake_motion_generator)
    adapter = AtomicActionAdapter(
        _planner_env(table=table, rigid_objects={"can": can}),
        planner_policy={
            "dynamic_collision": True,
            "dynamic_obstacle_uids": ["can"],
        },
    )

    generator = adapter._generator()

    assert generator is adapter._motion_generator
    planner = captured["cfg"].planner_cfg
    assert isinstance(planner, CuroboPlannerCfg)
    assert planner.world.rigid_objects == {"table": table, "can": can}
    assert planner.world.dynamic_obstacle_names == ["can"]
    assert planner.world.obstacle_representation == "cuboid"
    assert planner.world.collision_cache == {"cuboid": 8, "mesh": 2}


def test_curobo_generator_sizes_collision_cache_for_large_scene(
    monkeypatch: Any,
) -> None:
    rigid_objects = {f"object_{index:02d}": object() for index in range(13)}
    captured: dict[str, Any] = {}

    def fake_motion_generator(*, cfg: Any) -> object:
        captured["cfg"] = cfg
        return object()

    monkeypatch.setattr(actions, "MotionGenerator", fake_motion_generator)
    adapter = AtomicActionAdapter(
        _planner_env(rigid_objects=rigid_objects),
        planner_policy={
            "dynamic_collision": True,
            "dynamic_obstacle_uids": list(rigid_objects),
        },
    )

    adapter._generator()

    planner = captured["cfg"].planner_cfg
    assert planner.world.collision_cache == {"cuboid": 13, "mesh": 2}


def test_dynamic_scene_parks_contact_target_and_held_rows() -> None:
    actual = torch.eye(4).repeat(2, 1, 1)
    actual[:, 2, 3] = torch.tensor([0.7, 0.8])
    entities = {uid: _PoseEntity(actual.clone()) for uid in ("target", "held", "other")}
    adapter = AtomicActionAdapter(
        _planner_env(rigid_objects=entities),
        planner_policy={
            "dynamic_collision": True,
            "dynamic_obstacle_uids": list(entities),
        },
    )
    held_semantics = ObjectSemantics(
        label="held",
        entity=entities["held"],
        geometry={},
        affordance=Affordance(),
    )
    held = HeldObjectState(
        semantics=held_semantics,
        object_to_eef=torch.eye(4).repeat(2, 1, 1),
        grasp_xpos=torch.eye(4).repeat(2, 1, 1),
        env_mask=torch.tensor([True, False]),
    )
    state = ExecutionState(
        last_qpos=torch.zeros(2, 8),
        held_objects={"physical_left_arm": held},
    )
    grounded = GroundedAction(
        "PickUp",
        "right_arm",
        "arm",
        JointPositionGoal(target=torch.zeros(2, 2)),
        {},
        object_uid="target",
    )

    scene = adapter._scene_snapshot(grounded, state)

    assert torch.equal(
        scene.entities["target"].pose[:, 2, 3],
        actual[:, 2, 3] + actions._COLLISION_PARKING_Z_OFFSET,
    )
    assert scene.entities["held"].pose[0, 2, 3] == (
        actual[0, 2, 3] + actions._COLLISION_PARKING_Z_OFFSET
    )
    assert scene.entities["held"].pose[1, 2, 3] == actual[1, 2, 3]
    assert torch.equal(scene.entities["other"].pose, actual)


def test_released_object_returns_to_live_dynamic_collision_pose() -> None:
    actual = torch.eye(4).repeat(2, 1, 1)
    actual[:, 0, 3] = torch.tensor([0.2, 0.4])
    entity = _PoseEntity(actual)
    adapter = AtomicActionAdapter(
        _planner_env(rigid_objects={"released": entity}),
        planner_policy={
            "dynamic_collision": True,
            "dynamic_obstacle_uids": ["released"],
        },
    )
    grounded = GroundedAction(
        "MoveJoints",
        "left_arm",
        "arm",
        JointPositionGoal(target=torch.zeros(2, 2)),
        {},
        object_uid="released",
    )

    scene = adapter._scene_snapshot(
        grounded,
        ExecutionState(last_qpos=torch.zeros(2, 8)),
    )

    assert torch.equal(scene.entities["released"].pose, actual)


def test_default_scene_provider_advances_only_after_material_change() -> None:
    actual = torch.eye(4).repeat(2, 1, 1)
    entity = _PoseEntity(actual.clone())
    adapter = AtomicActionAdapter(
        _planner_env(rigid_objects={"can": entity}),
        planner_policy={
            "dynamic_collision": True,
            "dynamic_obstacle_uids": ["can"],
        },
    )
    grounded = GroundedAction(
        "MoveJoints",
        "left_arm",
        "arm",
        JointPositionGoal(target=torch.zeros(2, 2)),
        {},
        object_uid="can",
    )
    state = ExecutionState(last_qpos=torch.zeros(2, 8))

    first = adapter._scene_snapshot(grounded, state)
    unchanged = adapter._scene_snapshot(grounded, state)
    entity.pose[:, 0, 3] += 0.1
    changed = adapter._scene_snapshot(grounded, state)

    assert first.version == unchanged.version == 0
    assert changed.version == 1
    assert changed.collision_world_revisions(2) == (1, 1)


def test_external_scene_provider_is_used_by_planning_snapshot() -> None:
    pose = torch.eye(4).repeat(2, 1, 1)

    class _Provider:
        def snapshot(self, *, timestamp: float, env_ids: torch.Tensor) -> SceneSnapshot:
            assert timestamp == 0.0
            assert torch.equal(env_ids, torch.tensor([0, 1]))
            return SceneSnapshot(
                timestamp=timestamp,
                version=7,
                entities={"can": actions.EntityState(pose)},
            )

    adapter = AtomicActionAdapter(
        _planner_env(),
        scene_provider=_Provider(),
    )
    grounded = GroundedAction(
        "MoveJoints",
        "left_arm",
        "arm",
        JointPositionGoal(target=torch.zeros(2, 2)),
        {},
        object_uid="can",
    )

    scene = adapter._scene_snapshot(
        grounded,
        ExecutionState(last_qpos=torch.zeros(2, 8)),
    )

    assert scene.version == 7
    assert torch.equal(scene.entities["can"].pose, pose)


def test_start_session_delegates_to_shared_atomic_engine(monkeypatch: Any) -> None:
    adapter = AtomicActionAdapter(_planner_env())
    grounded = GroundedAction(
        "MoveJoints",
        "left_arm",
        "arm",
        JointPositionGoal(target=torch.zeros(2, 2)),
        {},
    )
    state = ExecutionState(last_qpos=torch.zeros(2, 8))
    marker = object()
    captured: dict[str, Any] = {}

    monkeypatch.setattr(adapter, "_planning_context", lambda *_args: "context")
    monkeypatch.setattr(
        adapter,
        "_invocation",
        lambda *_args, **_kwargs: "invocation",
    )

    class _Engine:
        def start(self, invocations: tuple[Any, ...], context: Any) -> object:
            captured["invocations"] = invocations
            captured["context"] = context
            return marker

    monkeypatch.setattr(adapter, "_engine", lambda: _Engine())

    result = adapter.start_session(grounded, state)

    assert result is marker
    assert captured == {"invocations": ("invocation",), "context": "context"}


def test_retreat_parks_intentional_contact_objects() -> None:
    actual = torch.eye(4).repeat(2, 1, 1)
    entities = {
        uid: _PoseEntity(actual.clone()) for uid in ("released", "container", "other")
    }
    adapter = AtomicActionAdapter(
        _planner_env(rigid_objects=entities),
        planner_policy={
            "dynamic_collision": True,
            "dynamic_obstacle_uids": list(entities),
        },
    )
    grounded = GroundedAction(
        "MoveEndEffector",
        "left_arm",
        "arm",
        JointPositionGoal(target=torch.zeros(2, 2)),
        {},
        motion_policy={
            "collision_exclusion_uids": ["released", "container"],
        },
        object_uid="released",
    )

    scene = adapter._scene_snapshot(
        grounded,
        ExecutionState(last_qpos=torch.zeros(2, 8)),
    )

    parked_z = actual[:, 2, 3] + actions._COLLISION_PARKING_Z_OFFSET
    assert torch.equal(scene.entities["released"].pose[:, 2, 3], parked_z)
    assert torch.equal(scene.entities["container"].pose[:, 2, 3], parked_z)
    assert torch.equal(scene.entities["other"].pose, actual)


def test_action_outcome_commits_state_delta_only_for_verified_rows() -> None:
    semantics = ObjectSemantics(
        label="cube",
        entity=object(),
        geometry={},
        affordance=Affordance(),
    )
    held = HeldObjectState(
        semantics=semantics,
        object_to_eef=torch.eye(4).repeat(2, 1, 1),
        grasp_xpos=torch.eye(4).repeat(2, 1, 1),
    )
    prior = ExecutionState(last_qpos=torch.zeros(2, 3))
    trajectory = torch.stack(
        (torch.zeros(2, 3), torch.ones(2, 3)),
        dim=1,
    )
    delta = StateDelta(held_object_updates={"physical_left_arm": held})
    projected = ExecutionState.from_task_state(
        delta.apply(prior.to_task_state(), torch.ones(2, dtype=torch.bool)),
        last_qpos=trajectory[:, -1],
    )
    grounded = GroundedAction(
        "PickUp",
        "left_arm",
        "arm",
        JointPositionGoal(target=torch.zeros(2, 2)),
        {},
    )
    outcome = ActionOutcome(
        trajectory=trajectory,
        success=torch.ones(2, dtype=torch.bool),
        next_state=projected,
        grounded=grounded,
        prior_state=prior,
        expected_effects=delta,
    )

    committed = outcome.state_after(torch.tensor([True, False]))

    assert torch.equal(committed.last_qpos[0], torch.ones(3))
    assert torch.equal(committed.last_qpos[1], torch.zeros(3))
    committed_held = committed.get_held_object("physical_left_arm")
    assert committed_held is not None
    assert torch.equal(committed_held.env_mask, torch.tensor([True, False]))


def test_fallback_rows_keep_the_fallback_plan_effects(monkeypatch: Any) -> None:
    env = _planner_env()
    adapter = AtomicActionAdapter(env)
    semantics = ObjectSemantics(
        label="cube",
        entity=object(),
        geometry={},
        affordance=Affordance(),
    )

    def held_at(x: float) -> HeldObjectState:
        relation = torch.eye(4).repeat(2, 1, 1)
        relation[:, 0, 3] = x
        return HeldObjectState(
            semantics=semantics,
            object_to_eef=relation,
            grasp_xpos=torch.eye(4).repeat(2, 1, 1),
        )

    def action_plan(
        success: torch.Tensor,
        terminal: float,
        held: HeldObjectState,
    ) -> ActionPlan:
        positions = torch.full((2, 2, 8), terminal)
        trajectory = TimedTrajectory.from_uniform_step(
            positions,
            env_ids=torch.arange(2),
            step_dt=0.01,
        )
        return ActionPlan(
            skill_id="pick_up",
            plan_success=success,
            commands=_commands_for(trajectory),
            joint_trajectory=trajectory,
            recovery_policy=RecoveryPolicy(),
            tracking_policy=TrackingPolicy.timed(),
            planned_scene_version=0,
            planned_collision_world_revision=(0, 0),
            diagnostics=PlannerDiagnostics(
                backend="fake",
                metadata={"marker": terminal},
            ),
            expected_effects=StateDelta(
                held_object_updates={"physical_left_arm": held}
            ),
        )

    plans = iter(
        (
            action_plan(torch.tensor([True, False]), 1.0, held_at(1.0)),
            action_plan(torch.tensor([True, True]), 2.0, held_at(2.0)),
        )
    )
    strategies: list[str] = []

    def plan(invocation: Any, _context: Any) -> ActionPlan:
        strategies.append(invocation.motion_policy.strategy)
        return next(plans)

    monkeypatch.setattr(
        adapter,
        "_engine",
        lambda: _FakeEngine(plan),
    )
    grounded = GroundedAction(
        "PickUp",
        "left_arm",
        "arm",
        GraspGoal(semantics=semantics),
        {},
    )

    outcome = adapter.plan(
        grounded,
        ExecutionState(last_qpos=torch.zeros(2, 8)),
    )

    assert strategies == ["motion_gen", "ik_interp"]
    assert torch.equal(outcome.success, torch.tensor([True, True]))
    assert torch.equal(outcome.next_state.last_qpos[0], torch.ones(8))
    assert torch.equal(outcome.next_state.last_qpos[1], torch.full((8,), 2.0))
    held = outcome.next_state.get_held_object("physical_left_arm")
    assert held is not None
    assert held.object_to_eef[0, 0, 3] == 1.0
    assert held.object_to_eef[1, 0, 3] == 2.0
    assert torch.equal(
        outcome.planner_trace["primary_success"], torch.tensor([True, False])
    )
    assert torch.equal(
        outcome.planner_trace["fallback_attempted"], torch.tensor([False, True])
    )
    assert torch.equal(
        outcome.planner_trace["fallback_used"], torch.tensor([False, True])
    )
    assert outcome.planner_trace["primary_action_diagnostics"]["marker"] == 1.0
    assert outcome.planner_trace["fallback_action_diagnostics"]["marker"] == 2.0


def test_collision_required_cleanup_does_not_use_unsafe_fallback(
    monkeypatch: Any,
) -> None:
    pose = torch.eye(4).repeat(2, 1, 1)
    adapter = AtomicActionAdapter(
        _planner_env(rigid_objects={"released": _PoseEntity(pose)}),
        planner_policy={
            "dynamic_collision": True,
            "dynamic_obstacle_uids": ["released"],
        },
    )
    failed_trajectory = TimedTrajectory.from_uniform_step(
        torch.zeros(2, 2, 8),
        env_ids=torch.arange(2),
        step_dt=0.01,
    )
    failed_plan = ActionPlan(
        skill_id="move_joints",
        plan_success=torch.tensor([False, False]),
        commands=_commands_for(failed_trajectory),
        joint_trajectory=failed_trajectory,
        recovery_policy=RecoveryPolicy(),
        tracking_policy=TrackingPolicy.timed(),
        planned_scene_version=1,
        planned_collision_world_revision=(1, 1),
        diagnostics=PlannerDiagnostics(backend="fake"),
        expected_effects=StateDelta(),
    )
    strategies: list[str] = []

    def plan(invocation: Any, _context: Any) -> ActionPlan:
        strategies.append(invocation.motion_policy.strategy)
        assert invocation.motion_policy.dynamic_collision_mode.value == "required"
        return failed_plan

    monkeypatch.setattr(adapter, "_engine", lambda: _FakeEngine(plan))
    grounded = GroundedAction(
        "MoveJoints",
        "left_arm",
        "arm",
        JointPositionGoal(target=torch.zeros(2, 2)),
        {},
        motion_policy={"collision_safety": "required"},
        object_uid="released",
    )

    outcome = adapter.plan(
        grounded,
        ExecutionState(last_qpos=torch.zeros(2, 8)),
    )

    assert strategies == ["motion_gen"]
    assert not bool(outcome.success.any())
    assert outcome.planner_trace["fallback_allowed"] is False
    assert not bool(outcome.planner_trace["fallback_attempted"].any())
    assert not bool(outcome.planner_trace["fallback_used"].any())
    assert outcome.planner_trace["collision_obstacle_positions"]["released"].shape == (
        2,
        3,
    )
