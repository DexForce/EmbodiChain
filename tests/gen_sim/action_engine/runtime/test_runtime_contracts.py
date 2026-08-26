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

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from embodichain.gen_sim.action_engine.config import (
    RuntimePolicyCfg,
    default_runtime_policy,
    resolve_agent_runtime_policy,
    runtime_policy_hash,
)
from embodichain.gen_sim.action_engine.capabilities import (
    HeldObjectHandOverOptions,
    build_atomic_capability_registry,
)
from embodichain.gen_sim.action_engine.compiler import (
    compile_task_agent,
    compile_task_agent_v2,
)
from embodichain.gen_sim.action_engine.cli.run_agent import (
    build_parser as build_run_parser,
)
from embodichain.gen_sim.action_engine.domain import (
    TASK_AGENT_SCHEMA,
    execution_program_hash,
    motion_policy,
    validate_execution_program,
)
from embodichain.gen_sim.action_engine.environment import agent_env as env_module
from embodichain.gen_sim.action_engine.runtime.actions import AtomicActionAdapter
from embodichain.gen_sim.action_engine.runtime.executor import (
    ProgramExecutor,
    _EdgeResult,
    _score_arm_candidate,
)
from embodichain.gen_sim.action_engine.runtime.frames import (
    relation_offset,
    robot_frame_axes,
)
from embodichain.gen_sim.action_engine.runtime.grounding import ActionGrounder
from embodichain.gen_sim.action_engine.runtime.loader import (
    load_agent_execution_program,
    load_execution_program as _load_execution_program,
)
from embodichain.gen_sim.action_engine.runtime.models import (
    ActionOutcome,
    ExecutionEdge,
    ExecutionProgram,
    ExecutionResult,
    GroundedAction,
    SemanticStep,
)
from embodichain.gen_sim.action_engine.runtime.motion_policy import (
    resolve_motion_policy,
)
from embodichain.gen_sim.action_engine.runtime.predicates import evaluate_predicate
from embodichain.gen_sim.action_engine.runtime.recording import RuntimeRecorder
from embodichain.gen_sim.action_engine.runtime.state import ExecutionState
from embodichain.gen_sim.action_engine.runtime import solver_compat
from embodichain.gen_sim.action_engine.protocol import (
    SEED_GRAPH_SCHEMA,
    TASK_SPEC_SCHEMA,
)
from embodichain.gen_sim.action_engine.tasks import instantiate_seed_graph
from embodichain.lab.gym.envs import EmbodiedEnv
from embodichain.lab.sim.atomic_actions import (
    ActionBinding,
    Affordance,
    AntipodalAffordance,
    AxisAlignAffordance,
    AxisAlignGoal,
    CoordinatedPickGoal,
    CoordinatedPlacementGoal,
    CoordinatedPlacementOptions,
    EndEffectorPoseGoal,
    HeldObjectPoseGoal,
    HeldObjectState,
    ObjectSemantics,
    PickUpOptions,
    PourGoal,
    PressAffordance,
    PressGoal,
    PressOptions,
    SlideAffordance,
    SlideGoal,
    StateDelta,
    TwistAffordance,
    TwistGoal,
)

from ..task_fixtures import make_task_spec
from embodichain.lab.sim.solvers import URSolverCfg


def _task_agent(*steps: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": TASK_AGENT_SCHEMA,
        "task": "runtime_contract",
        "goal": "Exercise the deterministic runtime contract.",
        "semantic_steps": list(steps),
    }


def load_execution_program(source: Any, **kwargs: Any) -> ExecutionProgram:
    """Adapt legacy compiler fixtures without weakening the production loader."""
    if isinstance(source, dict) and source.get("schema_version") != SEED_GRAPH_SCHEMA:
        return ExecutionProgram.from_mapping(validate_execution_program(source))
    return _load_execution_program(source, **kwargs)


def _hold_step(step_id: str, object_uid: str, arm: str) -> dict[str, Any]:
    return {
        "id": step_id,
        "operator": "hold_hover",
        "object": object_uid,
        "actor": {"mode": "required", "arm": arm},
        "goal": {},
        "depends_on": [],
    }


class _FakeEntity:
    def __init__(
        self,
        uid: str,
        pose: torch.Tensor,
        vertices: torch.Tensor,
    ) -> None:
        self.uid = uid
        self._pose = pose
        self._vertices = vertices
        self._triangles = torch.tensor(
            [[0, 1, 2], [0, 2, 3]],
            dtype=torch.int64,
        )
        self.lin_vel = torch.zeros(pose.shape[0], 3)
        self.ang_vel = torch.zeros(pose.shape[0], 3)

    def get_local_pose(self, *, to_matrix: bool) -> torch.Tensor:
        assert to_matrix
        return self._pose.clone()

    def get_vertices(
        self,
        *,
        env_ids: list[int],
        scale: bool,
    ) -> torch.Tensor:
        del env_ids, scale
        return self._vertices.clone()

    def get_triangles(self, *, env_ids: list[int]) -> torch.Tensor:
        del env_ids
        return self._triangles.clone()


class _FakeArticulation:
    def __init__(self, uid: str, qpos: float) -> None:
        self.uid = uid
        self.joint_names = ["slide_joint"]
        self.active_joint_ids = [0]
        self.link_names = ["base", "drawer_link", "handle"]
        self.all_joint_names = ["slide_joint", "fixed_handle"]
        self._qpos = torch.tensor([[qpos]], dtype=torch.float32)
        self._limits = torch.tensor([[[0.0, 0.2]]], dtype=torch.float32)
        self._pose = _pose(0.0, 0.0, 0.7)
        self._vertices = _box_vertices(0.05)
        self._triangles = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int64)
        self._joint_info = SimpleNamespace(
            joint_type=SimpleNamespace(name="PRISMATIC"),
            child_link_name="drawer_link",
            parent_link_name="base",
            axis=torch.tensor([1.0, 0.0, 0.0]),
            origin_pose=torch.eye(4),
        )
        self._fixed_info = SimpleNamespace(
            joint_type=SimpleNamespace(name="FIXED"),
            child_link_name="handle",
            parent_link_name="drawer_link",
            axis=torch.tensor([1.0, 0.0, 0.0]),
            origin_pose=torch.eye(4),
        )
        self._entities = [
            SimpleNamespace(
                get_joint_info=lambda name: (
                    self._joint_info if name == "slide_joint" else self._fixed_info
                ),
            )
        ]

    def get_local_pose(self, *, to_matrix: bool) -> torch.Tensor:
        assert to_matrix
        return self._pose.clone()

    def get_link_pose(self, link_name: str, *, to_matrix: bool) -> torch.Tensor:
        assert link_name in self.link_names
        assert to_matrix
        return self._pose.clone()

    def get_link_vert_face(self, link_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        assert link_name == "drawer_link"
        return self._vertices.clone(), self._triangles.clone()

    def get_qpos(self) -> torch.Tensor:
        return self._qpos.clone()

    def get_qpos_limits(self, *, joint_ids: list[int]) -> torch.Tensor:
        return self._limits[:, joint_ids].clone()


class _FakeSim:
    def __init__(
        self,
        entities: dict[str, _FakeEntity],
        articulations: dict[str, _FakeArticulation] | None = None,
    ) -> None:
        self.entities = entities
        self.articulations = articulations or {}

    def get_rigid_object(self, uid: str) -> _FakeEntity | None:
        return self.entities.get(uid)

    def get_rigid_object_uid_list(self) -> list[str]:
        return list(self.entities)

    def get_articulation(self, uid: str) -> _FakeArticulation | None:
        return self.articulations.get(uid)

    def update(self, *, step: int) -> None:
        del step


class _FakeRobot:
    def __init__(self, num_envs: int = 1) -> None:
        self.uid = "fake_robot"
        self.dof = 8
        self._qpos = torch.zeros(num_envs, self.dof)
        self.control_parts = {
            "physical_left_arm": ["l0", "l1"],
            "physical_left_eef": ["lh0", "lh1"],
            "physical_right_arm": ["r0", "r1"],
            "physical_right_eef": ["rh0", "rh1"],
            "dual_arm": ["l0", "l1", "r0", "r1"],
        }
        self._ids = {
            "physical_left_arm": [0, 1],
            "physical_left_eef": [2, 3],
            "physical_right_arm": [4, 5],
            "physical_right_eef": [6, 7],
            "dual_arm": [0, 1, 4, 5],
        }

    def get_qpos(self) -> torch.Tensor:
        return self._qpos.clone()

    def get_joint_ids(self, *, name: str) -> list[int]:
        return list(self._ids[name])

    def get_control_part_base_pose(self, *, name: str, to_matrix: bool) -> torch.Tensor:
        del name
        assert to_matrix
        return torch.eye(4).repeat(self._qpos.shape[0], 1, 1)

    def get_solver(self, *, name: str) -> SimpleNamespace:
        return SimpleNamespace(root_link_name=name.replace("_arm", "_base"))

    def get_link_pose(self, *, link_name: str, to_matrix: bool) -> torch.Tensor:
        assert to_matrix
        pose = torch.eye(4).repeat(self._qpos.shape[0], 1, 1)
        pose[:, 1, 3] = -0.3 if link_name == "physical_left_base" else 0.3
        return pose

    def compute_fk(
        self,
        qpos: torch.Tensor,
        *,
        name: str,
        to_matrix: bool,
    ) -> torch.Tensor:
        del name
        assert to_matrix
        return torch.eye(4).repeat(qpos.shape[0], 1, 1)


class _FakeEnv:
    def __init__(
        self,
        entities: dict[str, _FakeEntity] | None = None,
        articulations: dict[str, _FakeArticulation] | None = None,
    ) -> None:
        self.num_envs = 1
        self.device = torch.device("cpu")
        self.robot = _FakeRobot(self.num_envs)
        self.sim = _FakeSim(entities or {}, articulations)
        self.left_arm_joints = [0, 1]
        self.left_eef_joints = [2, 3]
        self.right_arm_joints = [4, 5]
        self.right_eef_joints = [6, 7]
        self.open_state = torch.tensor([0.0, 0.0])
        self.close_state = torch.tensor([0.7, -0.7])

    def get_agent_arm_control_part(self, is_left: bool) -> str:
        return "physical_left_arm" if is_left else "physical_right_arm"

    def get_agent_eef_control_part(self, is_left: bool) -> str:
        return "physical_left_eef" if is_left else "physical_right_eef"

    def get_current_xpos_agent(self) -> tuple[torch.Tensor, torch.Tensor]:
        left = torch.eye(4).repeat(self.num_envs, 1, 1)
        right = left.clone()
        left[:, 1, 3] = -0.2
        right[:, 1, 3] = 0.2
        return left, right

    def get_current_qpos_agent(self) -> tuple[torch.Tensor, torch.Tensor]:
        qpos = self.robot.get_qpos()
        return qpos[:, self.left_arm_joints], qpos[:, self.right_arm_joints]

    def get_current_gripper_state_agent(self) -> tuple[torch.Tensor, torch.Tensor]:
        qpos = self.robot.get_qpos()
        return qpos[:, self.left_eef_joints], qpos[:, self.right_eef_joints]


def _box_vertices(half_extent: float) -> torch.Tensor:
    h = float(half_extent)
    return torch.tensor(
        [
            [-h, -h, -h],
            [h, -h, -h],
            [h, h, h],
            [-h, h, h],
        ],
        dtype=torch.float32,
    )


def _rect_vertices(x: float, y: float, z: float) -> torch.Tensor:
    return torch.tensor(
        [
            [sx * x, sy * y, sz * z]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ],
        dtype=torch.float32,
    )


def _pose(x: float, y: float, z: float) -> torch.Tensor:
    result = torch.eye(4).unsqueeze(0)
    result[:, :3, 3] = torch.tensor([x, y, z])
    return result


def test_press_grounding_adapts_top_surface_and_depth_to_mainline_contract() -> None:
    entity = _FakeEntity("button", _pose(0.1, -0.2, 0.75), _box_vertices(0.03))
    env = _FakeEnv({"button": entity})
    program = load_execution_program(
        compile_task_agent(
            _task_agent(
                {
                    "id": "press",
                    "operator": "press",
                    "object": "button",
                    "actor": {"mode": "required", "arm": "left_arm"},
                    "goal": {},
                    "depends_on": [],
                }
            )
        )
    )
    semantics = ObjectSemantics(
        affordance=Affordance(),
        geometry={},
        label="button",
        entity=entity,
    )
    step = program.semantic_steps[0]
    action = program.edges[0].actions[0]

    grounded = ActionGrounder(program, env, lambda _uid: semantics).ground(
        action,
        step,
        arm="left_arm",
        state=ExecutionState(last_qpos=env.robot.get_qpos()),
    )

    assert isinstance(grounded.target, PressGoal)
    assert isinstance(grounded.target.semantics.affordance, PressAffordance)
    contact = grounded.target.semantics.affordance.get_press_pose(
        grounded.target.target_pose
    )
    assert torch.allclose(contact[0, :3, 3], torch.tensor([0.1, -0.2, 0.78]))
    assert torch.allclose(contact[0, :3, 2], torch.tensor([0.0, 0.0, -1.0]))

    options = AtomicActionAdapter(env)._build_config(grounded, PressOptions)

    assert options.press_distance == pytest.approx(0.004)


def test_press_grounding_uses_calibrated_prismatic_button_state() -> None:
    task, _ = make_task_spec("E9")
    program = load_execution_program(
        instantiate_seed_graph(task, {"object_01": "button"})
    )
    articulation = _FakeArticulation("button", 0.0)
    articulation._joint_info.axis = torch.tensor([0.0, 0.0, -1.0])
    articulation._limits = torch.tensor([[[0.0, 0.02]]])
    env = _FakeEnv(articulations={"button": articulation})
    env.agent_config = {
        "articulation_settings": {"button": {"slide_joint": [0.0, 0.02]}}
    }
    semantics = ObjectSemantics(affordance=Affordance(), geometry={})
    step = program.semantic_steps[0]

    grounded = ActionGrounder(program, env, lambda _uid: semantics).ground(
        program.edges[0].actions[0],
        step,
        arm="left_arm",
        state=ExecutionState(last_qpos=env.robot.get_qpos()),
    )

    assert isinstance(grounded.target, PressGoal)
    affordance = grounded.target.semantics.affordance
    assert isinstance(affordance, PressAffordance)
    assert torch.allclose(affordance.press_axis, torch.tensor([0.0, 0.0, -1.0]))
    assert grounded.cfg["press_distance"] == pytest.approx(0.02)
    assert grounded.cfg["articulation_target_qpos"].item() == pytest.approx(0.02)


def test_pressed_provider_requires_live_calibrated_joint_state() -> None:
    from embodichain.gen_sim.action_engine.environment.agent_env import ActionEngineEnv

    articulation = _FakeArticulation("button", 0.0)
    articulation._limits = torch.tensor([[[0.0, 0.02]]])
    env = SimpleNamespace(
        sim=_FakeSim({}, {"button": articulation}),
        agent_config={
            "articulation_settings": {"button": {"slide_joint": [0.0, 0.02]}}
        },
        runtime_policy=SimpleNamespace(predicate_fallbacks={"axis_tolerance": 0.03}),
        num_envs=1,
        device="cpu",
    )

    assert not bool(ActionEngineEnv.is_object_pressed(env, "button")[0])
    articulation._qpos[0, 0] = 0.019
    assert bool(ActionEngineEnv.is_object_pressed(env, "button")[0])


def test_loader_regenerates_in_memory_without_execution_artifact(
    tmp_path: Path,
) -> None:
    task = _task_agent(_hold_step("hold", "can", "left_arm"))
    graph = compile_task_agent_v2(task)
    task_spec = {
        "schema_version": TASK_SPEC_SCHEMA,
        "task_id": "runtime_contract",
        "level": "L1",
        "instruction": "Hold the can.",
        "reasoning_type": "none",
        "task_instances": [
            {
                "id": "hold",
                "task_type": "E1",
                "params": {"object_role": "can"},
                "depends_on": [],
                "role": "primary",
            }
        ],
        "success": {"type": "object_held", "object": "can"},
        "oracle": {"reference_seed_graph": graph},
        "metadata": {"role_bindings": {"can": "can"}},
    }
    task_path = tmp_path / "task_spec.json"
    task_path.write_text(json.dumps(task_spec), encoding="utf-8")
    agent_config = {
        "schema_version": "action_engine_config_v2",
        "task_spec": task_path.name,
        "seed_task_graph": "not_written.json",
    }
    config_path = tmp_path / "agent_config.json"
    config_path.write_text(json.dumps(agent_config), encoding="utf-8")

    program = load_agent_execution_program(
        agent_config,
        agent_config_path=config_path,
        regenerate=True,
    )

    assert program.task == "runtime_contract"
    assert program.semantic_steps[0].operator == "hold_hover"
    assert not (tmp_path / "not_written.json").exists()


def test_production_loader_rejects_legacy_mapping() -> None:
    legacy = compile_task_agent(_task_agent(_hold_step("hold", "can", "left_arm")))

    with pytest.raises(ValueError, match="regenerate"):
        _load_execution_program(legacy)


def test_documented_run_command_arguments_remain_compatible() -> None:
    args = build_run_parser().parse_args(
        [
            "--task_name",
            "task4_2",
            "--gym_config",
            "/tmp/fast_gym_config.json",
            "--agent_config",
            "/tmp/agent_config.json",
            "--regenerate",
            "--headless",
            "--seed",
            "17",
        ]
    )

    assert args.task_name == "task4_2"
    assert args.regenerate is True
    assert args.headless is True
    assert args.seed == 17
    assert args.runtime_backend == "independent"
    assert args.failure_policy == "stop"


def test_run_command_accepts_continue_failure_policy() -> None:
    args = build_run_parser().parse_args(
        [
            "--task_name",
            "task4_2",
            "--gym_config",
            "/tmp/fast_gym_config.json",
            "--agent_config",
            "/tmp/agent_config.json",
            "--failure-policy",
            "continue",
        ]
    )

    assert args.failure_policy == "continue"


def test_dual_ur5_policy_uses_short_reach_upright_lifts() -> None:
    upright = motion_policy(("orientation", "upright"))
    ur5_pickup = resolve_motion_policy("dual_ur5", "PickUp", upright)
    ur5_transport = resolve_motion_policy("dual_ur5", "MoveHeldObject", upright)
    ur10_pickup = resolve_motion_policy("dual_ur10", "PickUp", upright)
    ur10_transport = resolve_motion_policy("dual_ur10", "MoveHeldObject", upright)

    assert ur5_pickup["lift_height"] == pytest.approx(0.12)
    assert ur5_transport["staging_lift_height"] == pytest.approx(0.12)
    assert ur10_pickup["lift_height"] == pytest.approx(0.16)
    assert ur10_transport["staging_lift_height"] == pytest.approx(0.25)


def test_joint_state_binding_selects_hand_timing_without_a_named_policy() -> None:
    entity = _FakeEntity("can", _pose(0.0, 0.2, 0.75), _box_vertices(0.03))
    env = _FakeEnv({"can": entity})
    env.agent_initial_object_poses = {"can": entity.get_local_pose(to_matrix=True)}
    program = load_execution_program(
        compile_task_agent(_task_agent(_hold_step("hold", "can", "left_arm")))
    )
    step = program.semantic_steps[0]
    action = next(
        action
        for edge in program.edges
        for action in edge.actions
        if action["target_binding"].get("source") == "gripper_closed"
    )
    grounder = ActionGrounder(program, env, lambda _uid: None)

    grounded = grounder.ground(
        action,
        step,
        arm="left_arm",
        state=ExecutionState(last_qpos=env.robot.get_qpos()),
    )

    assert grounded.cfg["sample_interval"] == 10


def test_e5_synchronized_release_uses_physics_verified_opening_time() -> None:
    entity = _FakeEntity("tray", _pose(0.0, 0.0, 0.75), _box_vertices(0.20))
    env = _FakeEnv({"tray": entity})
    program = load_execution_program(
        compile_task_agent(
            _task_agent(
                {
                    "id": "transport",
                    "operator": "coordinated_transport",
                    "object": "tray",
                    "actor": {
                        "mode": "coordinated",
                        "arms": ["left_arm", "right_arm"],
                    },
                    "goal": {"direction": "front", "terminal_behavior": "place"},
                    "depends_on": [],
                }
            )
        )
    )
    step = program.semantic_steps[0]
    release = next(
        action
        for edge in program.edges
        for action in edge.actions
        if action.get("target_binding", {}).get("coordinated_release_role")
        == "participant"
    )
    grounder = ActionGrounder(program, env, lambda _uid: None)

    grounded = grounder.ground(
        release,
        step,
        arm="left_arm",
        state=_coordinated_held_state(env, entity),
    )

    assert grounded.cfg["sample_interval"] == 60


def test_runtime_policy_discards_legacy_support_z_fallbacks() -> None:
    snapshot = default_runtime_policy("dual_franka").as_mapping()
    snapshot["predicate_fallbacks"].update(
        {
            "support_min_z_offset": 0.02,
            "support_max_z_offset": 0.35,
        }
    )

    policy = RuntimePolicyCfg.from_mapping(snapshot)

    assert "support_min_z_offset" not in policy.predicate_fallbacks
    assert "support_max_z_offset" not in policy.predicate_fallbacks


def test_runtime_policy_v4_migrates_grasp_direction_count() -> None:
    snapshot = default_runtime_policy("dual_franka").as_mapping()
    snapshot["schema_version"] = "action_engine_runtime_policy_v4"
    snapshot["grasp"].pop("n_deviated_approach_directions")
    snapshot_hash = hashlib.sha256(
        json.dumps(
            snapshot,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()

    policy = resolve_agent_runtime_policy(
        {
            "robot_profile": "dual_franka",
            "runtime_policy": snapshot,
            "runtime_policy_hash": snapshot_hash,
        }
    )

    assert policy.schema_version == "action_engine_runtime_policy_v8"
    assert policy.grasp["n_deviated_approach_directions"] == 4


def test_runtime_policy_v5_migrates_support_geometry_thresholds() -> None:
    snapshot = default_runtime_policy("dual_franka").as_mapping()
    snapshot["schema_version"] = "action_engine_runtime_policy_v5"
    snapshot["grounding"]["placement"]["clearance"] = 0.019
    for key in (
        "candidate_count",
        "candidate_offset_fraction",
        "support_margin",
        "recovery_attempts",
    ):
        snapshot["grounding"]["placement"].pop(key)
    for key in (
        "support_stability_samples",
        "support_stability_interval_steps",
        "support_linear_velocity_tolerance",
        "support_angular_velocity_tolerance",
    ):
        snapshot["execution"].pop(key)
    for key in (
        "support_com_margin",
        "support_max_vertical_gap",
        "support_max_penetration",
        "support_min_overlap_ratio",
    ):
        snapshot["predicate_fallbacks"].pop(key)
    snapshot_hash = hashlib.sha256(
        json.dumps(
            snapshot,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()

    policy = resolve_agent_runtime_policy(
        {
            "robot_profile": "dual_franka",
            "runtime_policy": snapshot,
            "runtime_policy_hash": snapshot_hash,
        }
    )

    assert policy.schema_version == "action_engine_runtime_policy_v8"
    assert policy.predicate_fallbacks["support_min_overlap_ratio"] == 0.25
    assert policy.grounding["placement"]["clearance"] == 0.019
    assert policy.grounding["placement"]["candidate_count"] == 5


def test_runtime_recorder_writes_checkpoints_and_rendered_env_graphs(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    program = load_execution_program(
        compile_task_agent(_task_agent(_hold_step("hold", "can", "left_arm")))
    )
    original_seed = deepcopy(program.raw)
    runtime_policy = default_runtime_policy("dual_ur10")
    recorder = RuntimeRecorder(
        program,
        num_envs=2,
        run_id="run-1",
        episode_index=3,
        output_root=tmp_path,
        runtime_policy=runtime_policy.as_mapping(),
        runtime_policy_hash=runtime_policy_hash(runtime_policy),
    )
    step = program.semantic_steps[0]
    recorder.edge(
        program.edges[0].id,
        step,
        assignments=["left_arm", None],
        grounded=[
            GroundedAction(
                action_class="PickUp",
                arm="left_arm",
                control="arm",
                target=None,
                cfg={},
                motion_policy={"obj_upright_direction": torch.tensor([0.0, 0.0, 1.0])},
            )
        ],
        active=torch.tensor([True, False]),
        failed=torch.tensor([False, True]),
        action_steps=4,
        planner_traces=[
            {
                "primary_strategy": "motion_gen",
                "primary_success": torch.tensor([True, False]),
                "fallback_used": torch.tensor([False, True]),
                "planned_trajectory": torch.arange(24, dtype=torch.float32).reshape(
                    2, 3, 4
                ),
            }
        ],
    )
    recorder.step(
        step,
        torch.tensor([True, False]),
        observed=torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        target=torch.tensor([[0.1, 0.2, 0.3], [0.0, 0.0, 0.0]]),
        metadata=[
            {
                "assigned_arm": "left_arm",
                "physical_control_part": "physical_right_arm",
            },
            {"assigned_arm": None, "physical_control_part": None},
        ],
    )

    episode_dir = tmp_path / "runtime_contract" / "run-1" / "episode_0003"
    checkpoint_paths = sorted(episode_dir.glob("env_*/checkpoints/*.json"))
    assert len(checkpoint_paths) == 2
    checkpoint = json.loads(checkpoint_paths[0].read_text(encoding="utf-8"))
    assert checkpoint["semantic_step"]["id"] == "hold"
    assert checkpoint["status"] == "success"
    assert [item["event"] for item in checkpoint["events"]] == [
        "edge",
        "semantic_step",
    ]
    assert checkpoint["events"][0]["actions"][0]["motion_policy"][
        "obj_upright_direction"
    ] == [0.0, 0.0, 1.0]
    assert checkpoint["events"][0]["planner_attempts"] == [
        {
            "primary_strategy": "motion_gen",
            "primary_success": True,
            "fallback_used": False,
            "planned_trajectory": [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
            ],
        }
    ]
    assert checkpoint["events"][1]["assigned_arm"] == "left_arm"
    assert checkpoint["events"][1]["physical_control_part"] == "physical_right_arm"

    rendered_documents: list[dict[str, Any]] = []
    visualization = ModuleType("embodichain.gen_sim.action_engine.graph_visualization")

    def render_task_graph_png(document: dict[str, Any]) -> bytes:
        rendered_documents.append(deepcopy(document))
        return b"\x89PNG\r\n\x1a\nruntime-graph"

    visualization.render_task_graph_png = render_task_graph_png
    monkeypatch.setitem(sys.modules, visualization.__name__, visualization)
    output_dir = recorder.finalize(torch.tensor([True, False]))

    assert output_dir == episode_dir.as_posix()
    assert program.raw == original_seed
    expected_hash = execution_program_hash(original_seed)
    for env_id, expected_status in enumerate(("success", "failed")):
        env_dir = episode_dir / f"env_{env_id:04d}"
        document = json.loads((env_dir / "task_graph.json").read_text(encoding="utf-8"))
        assert document["schema_version"] == original_seed["schema_version"]
        assert document["nodes"] == original_seed["nodes"]
        assert document["edges"] == original_seed["edges"]
        assert document["runtime"]["status"] == expected_status
        assert document["runtime"]["seed_graph_hash"] == expected_hash
        assert document["runtime"]["runtime_policy"] == runtime_policy.as_mapping()
        assert document["runtime"]["runtime_policy_hash"] == runtime_policy_hash(
            runtime_policy
        )
        assert (env_dir / "task_graph.png").read_bytes().startswith(b"\x89PNG")
    assert len(rendered_documents) == 2
    assert not list(episode_dir.rglob("*.tmp"))


def test_runtime_recorder_separates_dynamic_recovery_and_replay_phases(
    tmp_path: Path,
) -> None:
    program = load_execution_program(
        compile_task_agent(_task_agent(_hold_step("hold", "can", "left_arm")))
    )
    recorder = RuntimeRecorder(
        program,
        num_envs=1,
        run_id="phased-recovery",
        output_root=tmp_path,
    )
    primary = program.semantic_steps[0]
    recovery = replace(
        primary,
        id="recovery_e2_hold",
        parent_step_id=primary.id,
    )
    recovery_spec = deepcopy(program.raw["semantic_steps"][0])
    recovery_spec.update(
        {
            "id": recovery.id,
            "parent_step_id": primary.id,
            "role": "recovery",
        }
    )
    recorder.register_step(recovery, recovery_spec)
    active = torch.tensor([True])
    recorder.edge(
        "edge_recovery",
        recovery,
        assignments=["left_arm"],
        grounded=[],
        active=active,
        failed=torch.tensor([False]),
        action_steps=4,
        phase="recovery",
    )
    recorder.step(
        recovery,
        torch.tensor([True]),
        observed=torch.zeros((1, 3)),
        target=None,
        phase="recovery",
    )
    recorder.edge(
        program.edges[0].id,
        primary,
        assignments=["left_arm"],
        grounded=[],
        active=active,
        failed=torch.tensor([False]),
        action_steps=3,
        phase="replay",
    )
    recorder.step(
        primary,
        torch.tensor([True]),
        observed=torch.zeros((1, 3)),
        target=None,
    )

    checkpoints = sorted(
        (recorder.output_dir / "env_0000" / "checkpoints").glob("*.json")
    )
    assert len(checkpoints) == 2
    recovery_checkpoint = next(
        json.loads(path.read_text(encoding="utf-8"))
        for path in checkpoints
        if "recovery_e2_hold" in path.name
    )
    primary_checkpoint = next(
        json.loads(path.read_text(encoding="utf-8"))
        for path in checkpoints
        if path.name.endswith("_hold.json") and "recovery_e2" not in path.name
    )
    assert {event["phase"] for event in recovery_checkpoint["events"]} == {"recovery"}
    assert primary_checkpoint["events"][0]["phase"] == "replay"
    assert primary_checkpoint["events"][-1]["phase"] == "primary"


def test_runtime_recorder_does_not_mask_execution_when_png_rendering_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    program = load_execution_program(
        compile_task_agent(_task_agent(_hold_step("hold", "can", "left_arm")))
    )
    recorder = RuntimeRecorder(
        program,
        num_envs=1,
        run_id="render_failure",
        output_root=tmp_path,
    )
    visualization = ModuleType("embodichain.gen_sim.action_engine.graph_visualization")

    def fail_render(_document: dict[str, Any]) -> bytes:
        raise ValueError("broken renderer")

    visualization.render_task_graph_png = fail_render
    monkeypatch.setitem(sys.modules, visualization.__name__, visualization)

    output_dir = recorder.finalize(torch.tensor([False]))

    record = json.loads(
        (Path(output_dir) / "env_0000" / "task_graph.json").read_text(encoding="utf-8")
    )
    assert record["runtime"]["status"] == "failed"
    assert record["runtime"]["visualization_error"] == ("ValueError: broken renderer")


def test_ready_scheduler_packs_only_declared_opposite_arm_pickups() -> None:
    compiled = compile_task_agent(
        _task_agent(
            _hold_step("left", "can_a", "left_arm"),
            _hold_step("right", "can_b", "right_arm"),
        )
    )
    executor = ProgramExecutor(
        load_execution_program(compiled),
        _FakeEnv(),
        record_runtime=False,
    )
    ready = [edge for edge in executor.program.edges if not edge.depends_on]

    packed = executor._pack_ready_edges(ready)

    assert len(packed) == 2
    assert {executor.step_by_edge[edge.id].object_uid for edge in packed} == {
        "can_a",
        "can_b",
    }


def test_ready_scheduler_serializes_contact_sensitive_orient_pickups() -> None:
    steps = [
        {
            "id": step_id,
            "operator": "orient_object",
            "object": object_uid,
            "actor": {"mode": "auto"},
            "goal": {
                "orientation_goal": "upright",
                "orientation_axis": "none",
            },
            "depends_on": [],
        }
        for step_id, object_uid in (("left", "can_a"), ("right", "can_b"))
    ]
    task_agent = _task_agent(*steps)
    task_agent["allocation_groups"] = [
        {
            "id": "dual_arms_1",
            "semantic_step_ids": ["left", "right"],
            "arm_constraint": "distinct_arms",
        }
    ]
    executor = ProgramExecutor(
        load_execution_program(compile_task_agent(task_agent)),
        _FakeEnv(),
        record_runtime=False,
    )
    ready = [edge for edge in executor.program.edges if not edge.depends_on]

    assert len(executor._pack_ready_edges(ready)) == 1


def test_ready_scheduler_defers_pickups_until_a_carried_payload_is_released() -> None:
    entities = {
        "table": _FakeEntity(
            "table",
            _pose(0.0, 0.0, 0.70),
            _rect_vertices(0.60, 0.40, 0.02),
        ),
        **{
            uid: _FakeEntity(
                uid,
                _pose(x, 0.0, 0.76),
                _rect_vertices(0.03, 0.03, 0.06),
            )
            for uid, x in (("can_a", -0.2), ("can_b", 0.0), ("can_c", 0.2))
        },
    }
    task = _task_agent(
        {
            "id": "line",
            "operator": "arrange_line",
            "objects": ["can_a", "can_b", "can_c"],
            "actor": {"mode": "auto"},
            "goal": {"axis": "world_x", "order_constraint": "free"},
            "depends_on": [],
        }
    )
    executor = ProgramExecutor(
        load_execution_program(compile_task_agent(task)),
        _FakeEnv(entities),
        record_runtime=False,
    )
    pickup_edges = [
        edge
        for edge in executor.program.edges
        if executor._parallel_pickup_candidate(edge)
    ]
    completed = {pickup_edges[0].id, pickup_edges[1].id}
    ready = [
        edge
        for edge in executor.program.edges
        if edge.id not in completed and set(edge.depends_on) <= completed
    ]
    executor._arm_owners["left_arm"][0] = "can_a"
    executor._arm_owners["right_arm"][0] = "can_b"

    packed = executor._pack_ready_edges(ready, completed=completed)

    assert not executor._parallel_pickup_candidate(packed[0])

    executor._arm_owners["right_arm"][0] = None
    packed = executor._pack_ready_edges(ready, completed=completed)
    assert len(packed) == 1
    assert not executor._parallel_pickup_candidate(packed[0])


def test_parallel_pickups_plan_each_arm_at_execution_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _hold_step("first", "can_a", "left_arm")
    second = _hold_step("second", "can_b", "right_arm")
    first["actor"] = {"mode": "auto"}
    second["actor"] = {"mode": "auto"}
    executor = ProgramExecutor(
        load_execution_program(compile_task_agent(_task_agent(first, second))),
        _FakeEnv(
            {
                "can_a": _FakeEntity(
                    "can_a", _pose(0.0, 0.2, 0.75), _box_vertices(0.03)
                ),
                "can_b": _FakeEntity(
                    "can_b", _pose(0.0, -0.2, 0.75), _box_vertices(0.03)
                ),
            }
        ),
        record_runtime=False,
    )
    edges = tuple(
        next(
            edge
            for edge in executor.program.edges
            if edge.id in step.edge_ids
            and edge.actions[0]["atomic_action_class"] == "PickUp"
        )
        for step in executor.program.semantic_steps
    )
    estimate = SimpleNamespace(
        feasible=torch.tensor([True]),
        cost=torch.tensor([0.0]),
    )
    monkeypatch.setattr(executor, "_candidate", lambda *_args, **_kwargs: estimate)
    monkeypatch.setattr(executor, "_report_candidates", lambda *_args: None)
    live_calls: list[tuple[str, str]] = []

    def plan_live(edge, step, arm):
        live_calls.append((step.id, arm))
        grounded = GroundedAction(
            action_class="PickUp",
            arm=arm,
            control="arm",
            target=SimpleNamespace(),
            cfg={},
        )
        return grounded, ActionOutcome(
            trajectory=torch.zeros(1, 1, executor.env.robot.dof),
            success=torch.tensor([True]),
            next_state=ExecutionState(last_qpos=executor.env.robot.get_qpos()),
            grounded=grounded,
        )

    monkeypatch.setattr(executor, "_plan_live_hold", plan_live)
    monkeypatch.setattr(executor.adapter, "execute_trajectory", lambda *_a, **_k: [])
    monkeypatch.setattr(
        executor, "_physical_pickup", lambda _u, _a, _s, attempted: attempted
    )
    monkeypatch.setattr(
        executor, "_rebase_held_state", lambda _u, _a, state, *_args, **_kwargs: state
    )
    monkeypatch.setattr(executor, "_update_ownership", lambda *_args, **_kwargs: None)

    _, failed = executor._execute_parallel_pickups(
        edges,
        failed=torch.tensor([False]),
    )

    assert set(live_calls) == {
        ("first", "right_arm"),
        ("second", "left_arm"),
    }
    assert not bool(failed[0])


def test_required_arm_rejects_wrong_candidate_without_planning() -> None:
    compiled = compile_task_agent(_task_agent(_hold_step("hold", "can", "left_arm")))
    executor = ProgramExecutor(
        load_execution_program(compiled),
        _FakeEnv(),
        record_runtime=False,
    )
    failed = torch.zeros(1, dtype=torch.bool)

    candidate = executor._candidate(
        executor.program.semantic_steps[0],
        "right_arm",
        failed,
    )

    assert not bool(candidate.feasible.any())
    assert bool(torch.isinf(candidate.cost).all())


def test_required_arm_speculative_failure_still_reaches_live_planning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = ProgramExecutor(
        load_execution_program(
            compile_task_agent(_task_agent(_hold_step("hold", "can", "left_arm")))
        ),
        _FakeEnv(
            {"can": _FakeEntity("can", _pose(0.0, 0.2, 0.75), _box_vertices(0.03))}
        ),
        record_runtime=False,
    )
    step = executor.program.semantic_steps[0]
    estimate = SimpleNamespace(
        feasible=torch.tensor([False]),
        cost=torch.tensor([torch.inf]),
    )
    monkeypatch.setattr(executor, "_candidate", lambda *_args, **_kwargs: estimate)
    monkeypatch.setattr(executor, "_report_candidates", lambda *_args: None)

    executor._ensure_assignment(step, torch.tensor([False]))

    assert executor._assignments[step.id] == ["left_arm"]


def test_auto_pickup_outside_deadband_requires_same_side_arm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step_mapping = _hold_step("hold", "can", "left_arm")
    step_mapping["actor"] = {"mode": "auto"}
    executor = ProgramExecutor(
        load_execution_program(compile_task_agent(_task_agent(step_mapping))),
        _FakeEnv(
            {"can": _FakeEntity("can", _pose(0.0, -0.2, 0.75), _box_vertices(0.03))}
        ),
        record_runtime=False,
    )
    step = executor.program.semantic_steps[0]

    def candidate(_step, arm, _failed):
        cost = 10.0 if arm == "left_arm" else 1.0
        return SimpleNamespace(
            feasible=torch.tensor([True]),
            cost=torch.tensor([cost]),
        )

    monkeypatch.setattr(executor, "_candidate", candidate)
    monkeypatch.setattr(executor, "_report_candidates", lambda *_args: None)

    executor._ensure_assignment(step, torch.tensor([False]))

    assert executor._preferred_live_pickup_arm(step, 0) == "left_arm"
    assert executor._assignments[step.id] == ["left_arm"]


def test_auto_pickup_inside_deadband_selects_lower_cost_arm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step_mapping = _hold_step("hold", "can", "left_arm")
    step_mapping["actor"] = {"mode": "auto"}
    executor = ProgramExecutor(
        load_execution_program(compile_task_agent(_task_agent(step_mapping))),
        _FakeEnv(
            {"can": _FakeEntity("can", _pose(0.0, 0.01, 0.75), _box_vertices(0.03))}
        ),
        record_runtime=False,
    )
    step = executor.program.semantic_steps[0]

    def candidate(_step, arm, _failed):
        cost = 10.0 if arm == "left_arm" else 1.0
        return SimpleNamespace(
            feasible=torch.tensor([True]),
            cost=torch.tensor([cost]),
        )

    monkeypatch.setattr(executor, "_candidate", candidate)
    monkeypatch.setattr(executor, "_report_candidates", lambda *_args: None)

    executor._ensure_assignment(step, torch.tensor([False]))

    assert executor._preferred_live_pickup_arm(step, 0) is None
    assert executor._assignments[step.id] == ["right_arm"]


def test_auto_pickup_retry_does_not_cross_sides_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step_mapping = _hold_step("hold", "can", "left_arm")
    step_mapping["actor"] = {"mode": "auto"}
    executor = ProgramExecutor(
        load_execution_program(compile_task_agent(_task_agent(step_mapping))),
        _FakeEnv(
            {"can": _FakeEntity("can", _pose(0.0, 0.2, 0.75), _box_vertices(0.03))}
        ),
        record_runtime=False,
    )
    step = executor.program.semantic_steps[0]
    estimate = SimpleNamespace(
        feasible=torch.tensor([False]),
        cost=torch.tensor([torch.inf]),
    )
    monkeypatch.setattr(executor, "_candidate", lambda *_args, **_kwargs: estimate)
    monkeypatch.setattr(executor, "_report_candidates", lambda *_args: None)
    monkeypatch.setattr(
        executor,
        "_preferred_live_pickup_arm",
        lambda *_args: "left_arm",
    )

    executor._ensure_assignment(step, torch.tensor([False]))
    assert executor._assignments[step.id] == ["left_arm"]

    executor._pickup_retry_exclusions[(step.id, 0)] = {"left_arm"}
    executor._assignments.pop(step.id)
    executor._ensure_assignment(step, torch.tensor([False]))

    assert executor._assignments[step.id] == [None]


def test_auto_pickup_retry_can_explicitly_cross_sides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step_mapping = _hold_step("hold", "can", "left_arm")
    step_mapping["actor"] = {"mode": "auto"}
    executor = ProgramExecutor(
        load_execution_program(compile_task_agent(_task_agent(step_mapping))),
        _FakeEnv(
            {"can": _FakeEntity("can", _pose(0.0, 0.2, 0.75), _box_vertices(0.03))}
        ),
        record_runtime=False,
    )
    executor.runtime_policy.arm_selection.allow_cross_side_fallback = True
    step = executor.program.semantic_steps[0]
    estimate = SimpleNamespace(
        feasible=torch.tensor([False]),
        cost=torch.tensor([torch.inf]),
    )
    monkeypatch.setattr(executor, "_candidate", lambda *_args, **_kwargs: estimate)
    monkeypatch.setattr(executor, "_report_candidates", lambda *_args: None)
    monkeypatch.setattr(
        executor,
        "_preferred_live_pickup_arm",
        lambda *_args: "left_arm",
    )

    executor._pickup_retry_exclusions[(step.id, 0)] = {"left_arm"}
    executor._ensure_assignment(step, torch.tensor([False]))

    assert executor._assignments[step.id] == ["right_arm"]


def test_auto_pickup_runtime_retry_uses_the_other_arm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step_mapping = _hold_step("hold", "can", "left_arm")
    step_mapping["actor"] = {"mode": "auto"}
    executor = ProgramExecutor(
        load_execution_program(compile_task_agent(_task_agent(step_mapping))),
        _FakeEnv(
            {"can": _FakeEntity("can", _pose(0.0, 0.2, 0.75), _box_vertices(0.03))}
        ),
        record_runtime=False,
    )
    executor.runtime_policy.arm_selection.allow_cross_side_fallback = True
    step = executor.program.semantic_steps[0]
    original_edge = executor.edges[step.edge_ids[0]]
    action = {**original_edge.actions[0], "seed_node_id": "pickup_node"}
    edge = replace(original_edge, actions=(action,))
    estimate = SimpleNamespace(
        feasible=torch.tensor([False]),
        cost=torch.tensor([torch.inf]),
    )
    monkeypatch.setattr(executor, "_candidate", lambda *_args, **_kwargs: estimate)
    monkeypatch.setattr(executor, "_report_candidates", lambda *_args: None)
    monkeypatch.setattr(
        executor,
        "_preferred_live_pickup_arm",
        lambda *_args: "left_arm",
    )
    executor._ensure_assignment(step, torch.tensor([False]))
    attempts: list[str | None] = []

    def execute(_edge, _step, *, failed):
        arm = executor._assignments[step.id][0]
        attempts.append(arm)
        return _EdgeResult(
            actions=[],
            failed=torch.tensor([arm == "left_arm"]) | failed,
            grounded=[],
            planner_traces=[],
            executed=torch.tensor([False]),
        )

    decisions = 0

    def record_failure(*_args, **_kwargs):
        nonlocal decisions
        decisions += 1
        return SimpleNamespace(retry=torch.tensor([decisions == 1]))

    executor.runtime_graph = SimpleNamespace(
        graph={"nodes": [{"id": "pickup_node", "precondition": {}}]},
        record_failure=record_failure,
    )
    monkeypatch.setattr(executor, "_execute_edge", execute)

    result = executor._execute_edge_with_retries(
        edge,
        step,
        failed=torch.tensor([False]),
    )

    assert attempts == ["left_arm", "right_arm"]
    assert executor.retry_count == 1
    assert not bool(result.failed[0])


def _held_state(
    env: _FakeEnv,
    entity: _FakeEntity,
    *,
    arm: str = "left_arm",
) -> ExecutionState:
    semantics = ObjectSemantics(
        affordance=Affordance(),
        geometry={},
        label=entity.uid,
        entity=entity,
    )
    left_eef, right_eef = env.get_current_xpos_agent()
    eef = left_eef if arm == "left_arm" else right_eef
    object_pose = entity.get_local_pose(to_matrix=True)
    return ExecutionState(
        last_qpos=env.robot.get_qpos(),
        held_objects={
            f"physical_{arm}": HeldObjectState(
                semantics=semantics,
                object_to_eef=torch.bmm(torch.linalg.inv(object_pose), eef),
                grasp_xpos=eef,
            )
        },
    )


def _coordinated_held_state(
    env: _FakeEnv,
    entity: _FakeEntity,
) -> ExecutionState:
    semantics = ObjectSemantics(
        affordance=Affordance(),
        geometry={},
        label=entity.uid,
        entity=entity,
    )
    left_eef, right_eef = env.get_current_xpos_agent()
    object_pose = entity.get_local_pose(to_matrix=True)
    return ExecutionState(
        last_qpos=env.robot.get_qpos(),
        held_objects={
            "physical_left_arm": HeldObjectState(
                semantics=semantics,
                object_to_eef=torch.bmm(torch.linalg.inv(object_pose), left_eef),
                grasp_xpos=left_eef,
                env_mask=torch.ones(env.num_envs, dtype=torch.bool),
            ),
            "physical_right_arm": HeldObjectState(
                semantics=semantics,
                object_to_eef=torch.bmm(torch.linalg.inv(object_pose), right_eef),
                grasp_xpos=right_eef,
                env_mask=torch.ones(env.num_envs, dtype=torch.bool),
            ),
        },
    )


@pytest.mark.parametrize(
    ("opens", "expected_failed", "expect_held"),
    ((True, False, False), (False, True, True)),
)
def test_explicit_dual_gripper_release_commits_only_after_both_hands_open(
    opens: bool,
    expected_failed: bool,
    expect_held: bool,
) -> None:
    entity = _FakeEntity("tray", _pose(0.0, 0.0, 0.75), _box_vertices(0.2))
    env = _FakeEnv({"tray": entity})
    env.robot._qpos[:, env.left_eef_joints] = env.close_state
    env.robot._qpos[:, env.right_eef_joints] = env.close_state
    state = _coordinated_held_state(env, entity)
    executor = object.__new__(ProgramExecutor)
    executor.env = env
    executor.runtime_policy = default_runtime_policy("dual_ur10")
    executor._assignments = {"task_01": ["coordinated"]}
    executor._step_states = {("task_01", "coordinated"): state}
    executor._object_states = {}
    executor._object_owners = {"tray": ["coordinated"]}
    executor._arm_owners = {
        "left_arm": ["tray"],
        "right_arm": ["tray"],
    }
    executor._orientation_references = {}

    def ground(
        action: dict[str, Any],
        _step: Any,
        *,
        arm: str,
        **_kwargs: Any,
    ) -> GroundedAction:
        return GroundedAction(
            action_class=str(action["atomic_action_class"]),
            arm=arm,
            control="hand",
            target=None,
            cfg={},
        )

    def plan(grounded: GroundedAction, current: ExecutionState) -> ActionOutcome:
        return ActionOutcome(
            trajectory=torch.zeros(1, 2, env.robot.dof),
            success=torch.ones(1, dtype=torch.bool),
            next_state=current,
            grounded=grounded,
        )

    def execute_trajectory(
        _trajectory: torch.Tensor,
        *,
        active: torch.Tensor,
    ) -> list[torch.Tensor]:
        if opens and bool(active.any()):
            env.robot._qpos[:, env.left_eef_joints] = env.open_state
            env.robot._qpos[:, env.right_eef_joints] = env.open_state
        elif bool(active.any()):
            env.robot._qpos[:, env.left_eef_joints] = env.open_state
        return []

    executor.grounder = SimpleNamespace(ground=ground)
    executor.adapter = SimpleNamespace(
        capabilities=build_atomic_capability_registry(),
        plan=plan,
        combine=lambda _outcomes, _masks: (
            torch.zeros(1, 2, env.robot.dof),
            torch.ones(1, dtype=torch.bool),
        ),
        execute_trajectory=execute_trajectory,
    )
    actions = [
        {
            "atomic_action_class": "MoveJoints",
            "actor": {"arm": arm},
            "control": "hand",
            "target_binding": {
                "kind": "joint_state",
                "source": "gripper_open",
                "coordinated_release_role": role,
            },
        }
        for arm, role in (
            ("left_arm", "participant"),
            ("right_arm", "commit"),
        )
    ]

    result = executor._execute_explicit_dual(
        SimpleNamespace(id="release", actions=actions),
        SimpleNamespace(id="task_01", object_uid="tray"),
        torch.zeros(1, dtype=torch.bool),
    )

    released_state = executor._step_states[("task_01", "coordinated")]
    left_held = released_state.get_held_object("physical_left_arm")
    right_held = released_state.get_held_object("physical_right_arm")
    assert result.failed.tolist() == [expected_failed]
    assert (left_held is not None and right_held is not None) is expect_held
    expected_owner = "coordinated" if expect_held else None
    assert executor._object_owners["tray"] == [expected_owner]
    assert executor._arm_owners["left_arm"] == (["tray"] if expect_held else [None])
    assert executor._arm_owners["right_arm"] == (["tray"] if expect_held else [None])


@pytest.mark.parametrize(
    ("closes_both", "expected_failed", "expect_held"),
    ((True, False, True), (False, True, False)),
)
def test_coordinated_pickment_commits_only_after_physical_dual_hold(
    monkeypatch: pytest.MonkeyPatch,
    closes_both: bool,
    expected_failed: bool,
    expect_held: bool,
) -> None:
    entity = _FakeEntity("tray", _pose(0.0, 0.0, 0.75), _box_vertices(0.2))
    env = _FakeEnv({"tray": entity})
    program = load_execution_program(
        compile_task_agent(
            _task_agent(
                {
                    "id": "task_01",
                    "operator": "coordinated_transport",
                    "object": "tray",
                    "actor": {
                        "mode": "coordinated",
                        "arms": ["left_arm", "right_arm"],
                    },
                    "goal": {"direction": "up", "terminal_behavior": "hold"},
                    "depends_on": [],
                }
            )
        )
    )
    executor = ProgramExecutor(program, env, record_runtime=False)
    step = program.semantic_steps[0]
    edge = program.edges[0]
    executor._assignments[step.id] = ["coordinated"]
    prior_state = ExecutionState(last_qpos=env.robot.get_qpos().clone())
    planned_state = _coordinated_held_state(env, entity)
    grounded = GroundedAction(
        action_class="CoordinatedPickment",
        arm="coordinated",
        control="coordinated",
        target=SimpleNamespace(),
        cfg={"postcondition_tolerance": 0.06},
        object_pose=entity.get_local_pose(to_matrix=True),
        target_object_pose=entity.get_local_pose(to_matrix=True),
        object_uid="tray",
    )
    outcome = ActionOutcome(
        trajectory=torch.zeros(1, 1, env.robot.dof),
        success=torch.tensor([True]),
        next_state=planned_state,
        grounded=grounded,
        prior_state=prior_state,
        expected_effects=StateDelta(
            held_object_updates=dict(planned_state.held_objects)
        ),
    )
    monkeypatch.setattr(
        executor.grounder,
        "ground_candidates",
        lambda *_args, **_kwargs: (grounded,),
    )
    monkeypatch.setattr(executor.adapter, "plan", lambda *_args, **_kwargs: outcome)

    def execute_trajectory(*_args: Any, **_kwargs: Any) -> list[torch.Tensor]:
        env.robot._qpos[:, env.left_eef_joints] = env.close_state
        if closes_both:
            env.robot._qpos[:, env.right_eef_joints] = env.close_state
        return []

    monkeypatch.setattr(executor.adapter, "execute_trajectory", execute_trajectory)

    result = executor._execute_coordinated(edge, step, torch.tensor([False]))

    committed = executor._step_states[(step.id, "coordinated")]
    held = tuple(
        committed.get_held_object(f"physical_{arm}")
        for arm in ("left_arm", "right_arm")
    )
    assert result.failed.tolist() == [expected_failed]
    assert all(item is not None for item in held) is expect_held
    expected_owner = "coordinated" if expect_held else None
    assert executor._object_owners["tray"] == [expected_owner]
    assert executor._arm_owners["left_arm"] == (["tray"] if expect_held else [None])
    assert executor._arm_owners["right_arm"] == (["tray"] if expect_held else [None])


def _handover_held_state(
    env: _FakeEnv,
    entity: _FakeEntity,
    *,
    arm: str = "left_arm",
) -> ExecutionState:
    """Build a fixture grasp on the side assigned to the transfer arm."""
    state = _held_state(env, entity, arm=arm)
    held = state.get_held_object(f"physical_{arm}")
    assert held is not None
    _, lateral = robot_frame_axes(env)
    role_axis = lateral if arm == "left_arm" else -lateral
    offset = torch.cat((role_axis, role_axis.new_zeros((int(env.num_envs), 1))), dim=1)
    object_to_eef = held.object_to_eef.clone()
    object_to_eef[:, :3, 3] = offset * 0.02
    replacement = HeldObjectState(
        semantics=held.semantics,
        object_to_eef=object_to_eef,
        grasp_xpos=torch.bmm(entity.get_local_pose(to_matrix=True), object_to_eef),
        env_mask=held.env_mask,
    )
    held_objects = dict(state.held_objects)
    held_objects[f"physical_{arm}"] = replacement
    return state.with_updates(held_objects=held_objects)


def test_handover_grounding_uses_bottom_region_and_diagonal_receive() -> None:
    entities = {
        "can": _FakeEntity(
            "can",
            _pose(0.0, 0.2, 1.2),
            _rect_vertices(0.03, 0.03, 0.10),
        ),
        "table": _FakeEntity("table", _pose(0.0, 0.0, 0.5), _box_vertices(0.5)),
    }
    env = _FakeEnv(entities)
    task = {
        "schema_version": TASK_SPEC_SCHEMA,
        "task_id": "handover",
        "level": "L1",
        "instruction": "Hand over the can.",
        "reasoning_type": "none",
        "task_instances": [
            {
                "id": "task_01",
                "task_type": "E4",
                "params": {
                    "object_role": "can",
                    "transfer_arm": "left_arm",
                    "receive_arm": "right_arm",
                },
                "depends_on": [],
                "role": "primary",
            }
        ],
        "success": {"type": "handover_complete"},
        "oracle": {},
        "metadata": {},
    }
    program = load_execution_program(instantiate_seed_graph(task, {"can": "can"}))
    step = program.semantic_steps[0]
    edge = next(
        edge
        for edge in program.edges
        if edge.actions[0]["atomic_action_class"] == "HandOver"
    )
    state = _handover_held_state(env, entities["can"])
    held = state.get_held_object("physical_left_arm")
    assert held is not None
    grounder = ActionGrounder(program, env, lambda _uid: held.semantics)

    grounded = grounder.ground(
        edge.actions[0],
        step,
        arm="coordinated",
        state=state,
    )
    middle = grounded.cfg["middle_object_pose"]
    final = grounded.cfg["final_object_pose"]
    cfg = AtomicActionAdapter(env)._build_config(grounded, HeldObjectHandOverOptions)

    staging_edge = next(
        edge
        for edge in program.edges
        if edge.actions[0]["target_binding"].get("kind") == "handover_staging"
    )
    staging = grounder.ground(
        staging_edge.actions[0],
        step,
        arm="left_arm",
        state=state,
    )

    assert middle[0, 1, 3] == pytest.approx(0.0)
    torch.testing.assert_close(final, middle)
    torch.testing.assert_close(cfg.middle_object_pose, cfg.final_object_pose)
    assert cfg.receive_pick_object_part == "bottom"
    assert cfg.receive_approach_direction[1] < 0.0
    assert cfg.receive_approach_direction[2] < 0.0
    assert staging.allow_yaw_search


def test_handover_rejects_receiver_motion_during_internal_final_phase() -> None:
    adapter = AtomicActionAdapter(_FakeEnv())
    grounded = GroundedAction(
        action_class="HandOver",
        arm="coordinated",
        control="coordinated",
        target=SimpleNamespace(),
        cfg={"transfer_arm": "left_arm"},
    )
    options = HeldObjectHandOverOptions(retreat_steps=4)
    trajectory = torch.zeros(1, 12, adapter.env.robot.dof)

    assert bool(
        adapter._handover_receiver_hold_mask(
            trajectory,
            grounded,
            options,
            tolerance=1.0e-3,
        )[0]
    )

    trajectory[0, -1, 4] = 0.02
    assert not bool(
        adapter._handover_receiver_hold_mask(
            trajectory,
            grounded,
            options,
            tolerance=1.0e-3,
        )[0]
    )


def _handover_then_place_task() -> dict[str, Any]:
    return {
        "schema_version": TASK_SPEC_SCHEMA,
        "task_id": "handover_then_place",
        "level": "L3",
        "instruction": "Hand over the can and place it beside the target.",
        "reasoning_type": "none",
        "task_instances": [
            {
                "id": "handover",
                "task_type": "E4",
                "params": {
                    "object_role": "can",
                    "transfer_arm": "left_arm",
                    "receive_arm": "right_arm",
                    "orientation_goal": "preserve",
                },
                "depends_on": [],
                "role": "primary",
            },
            {
                "id": "place",
                "task_type": "E1",
                "params": {
                    "object_role": "can",
                    "target_role": "target",
                    "relation": "right_of",
                },
                "depends_on": ["handover"],
                "role": "primary",
            },
        ],
        "success": {
            "op": "all",
            "terms": [
                {"type": "handover_complete", "task_instance_id": "handover"},
                {"type": "semantic_goal", "task_instance_id": "place"},
            ],
        },
        "oracle": {},
        "metadata": {},
    }


def test_handover_continuation_uses_stable_upright_policies() -> None:
    entities = {
        "can": _FakeEntity(
            "can", _pose(0.0, 0.2, 0.75), _rect_vertices(0.03, 0.03, 0.10)
        ),
        "target": _FakeEntity("target", _pose(0.2, 0.0, 0.75), _box_vertices(0.03)),
        "table": _FakeEntity("table", _pose(0.0, 0.0, 0.5), _box_vertices(0.5)),
    }
    env = _FakeEnv(entities)
    task = deepcopy(_handover_then_place_task())
    task["task_instances"][1]["params"]["orientation_goal"] = "upright"
    program = load_execution_program(
        instantiate_seed_graph(
            task,
            {"can": "can", "target": "target"},
        )
    )
    step = next(
        candidate
        for candidate in program.semantic_steps
        if candidate.operator == "place_relative"
    )
    state = _held_state(env, entities["can"], arm="right_arm")
    held = state.get_held_object("physical_right_arm")
    assert held is not None
    grounder = ActionGrounder(program, env, lambda _uid: held.semantics)
    edges = [edge for edge in program.edges if edge.id in step.edge_ids]
    staging = next(
        edge
        for edge in edges
        if edge.actions[0]["target_binding"].get("phase") == "staging"
    )
    final = next(
        edge
        for edge in edges
        if edge.actions[0]["target_binding"].get("phase") == "final"
    )
    release = next(
        edge for edge in edges if edge.actions[0]["atomic_action_class"] == "Place"
    )
    retreat = next(
        edge
        for edge in edges
        if edge.actions[0]["atomic_action_class"] == "MoveEndEffector"
    )
    home = next(
        edge for edge in edges if edge.actions[0]["atomic_action_class"] == "MoveJoints"
    )

    grounded_staging = grounder.ground(
        staging.actions[0], step, arm="right_arm", state=state
    )
    grounded_final = grounder.ground(
        final.actions[0], step, arm="right_arm", state=state
    )
    supported_reference = _pose(0.0, 0.0, 0.90)
    grounded_final_with_reference = grounder.ground(
        final.actions[0],
        step,
        arm="right_arm",
        state=state,
        orientation_reference_pose=supported_reference,
    )
    grounded_release = grounder.ground(
        release.actions[0], step, arm="right_arm", state=state
    )
    grounded_retreat = grounder.ground(
        retreat.actions[0], step, arm="right_arm", state=state
    )
    grounded_home = grounder.ground(home.actions[0], step, arm="right_arm", state=state)
    upright = motion_policy(("orientation", "upright"))
    release_defaults = resolve_motion_policy("dual_ur10", "Place", upright)
    retreat_defaults = resolve_motion_policy("dual_ur10", "MoveEndEffector", upright)

    assert grounded_staging.allow_yaw_search
    assert grounded_final.allow_yaw_search
    assert grounded_final_with_reference.target_object_pose is not None
    assert grounded_final_with_reference.target_object_pose[0, 2, 3] == pytest.approx(
        0.90
    )
    assert (
        grounded_release.cfg["sample_interval"] == release_defaults["sample_interval"]
    )
    assert (
        grounded_release.cfg["post_hold_steps"] == release_defaults["post_hold_steps"]
    )
    assert (
        grounded_retreat.cfg["sample_interval"] == retreat_defaults["sample_interval"]
    )
    assert grounded_retreat.cfg["retreat_height"] == pytest.approx(
        retreat_defaults["retreat_height"]
    )
    assert grounded_retreat.cfg["retreat_height"] == pytest.approx(0.30)
    assert grounded_retreat.motion_policy["clearance_object_uid"] == "can"
    assert grounded_retreat.motion_policy["collision_exclusion_uids"] == [
        "can",
        "target",
    ]
    assert grounded_retreat.motion_policy["collision_safety"] == "required"
    assert grounded_home.motion_policy["collision_safety"] == "required"


def test_preserve_handover_continuation_does_not_enable_yaw_search() -> None:
    entities = {
        "can": _FakeEntity("can", _pose(0.0, 0.2, 0.75), _box_vertices(0.03)),
        "target": _FakeEntity("target", _pose(0.2, 0.0, 0.75), _box_vertices(0.03)),
        "table": _FakeEntity("table", _pose(0.0, 0.0, 0.5), _box_vertices(0.5)),
    }
    env = _FakeEnv(entities)
    task = deepcopy(_handover_then_place_task())
    task["task_instances"][1]["params"]["orientation_goal"] = "preserve"
    program = load_execution_program(
        instantiate_seed_graph(task, {"can": "can", "target": "target"})
    )
    step = next(
        candidate
        for candidate in program.semantic_steps
        if candidate.operator == "place_relative"
    )
    state = _held_state(env, entities["can"], arm="right_arm")
    held = state.get_held_object("physical_right_arm")
    assert held is not None
    grounder = ActionGrounder(program, env, lambda _uid: held.semantics)
    final = next(
        edge
        for edge in program.edges
        if edge.id in step.edge_ids
        and edge.actions[0]["target_binding"].get("phase") == "final"
    )
    reference = _pose(0.0, 0.0, 0.90)

    grounded = grounder.ground(
        final.actions[0],
        step,
        arm="right_arm",
        state=state,
        orientation_reference_pose=reference,
    )

    assert not grounded.allow_yaw_search
    assert grounded.target_object_pose is not None
    torch.testing.assert_close(
        grounded.target_object_pose[:, :3, :3],
        reference[:, :3, :3],
    )


def test_unconstrained_handover_continuation_has_free_yaw() -> None:
    entities = {
        "can": _FakeEntity("can", _pose(0.0, 0.2, 0.75), _box_vertices(0.03)),
        "target": _FakeEntity("target", _pose(0.2, 0.0, 0.75), _box_vertices(0.03)),
        "table": _FakeEntity("table", _pose(0.0, 0.0, 0.5), _box_vertices(0.5)),
    }
    env = _FakeEnv(entities)
    task = deepcopy(_handover_then_place_task())
    task["task_instances"][0]["params"]["orientation_goal"] = "none"
    program = load_execution_program(
        instantiate_seed_graph(task, {"can": "can", "target": "target"})
    )
    step = next(
        candidate
        for candidate in program.semantic_steps
        if candidate.operator == "place_relative"
    )
    state = _held_state(env, entities["can"], arm="right_arm")
    held = state.get_held_object("physical_right_arm")
    assert held is not None
    grounder = ActionGrounder(program, env, lambda _uid: held.semantics)
    grounded = [
        grounder.ground(edge.actions[0], step, arm="right_arm", state=state)
        for edge in program.edges
        if edge.id in step.edge_ids
        and edge.actions[0]["atomic_action_class"] in {"MoveHeldObject", "Place"}
    ]

    assert [item.action_class for item in grounded] == [
        "MoveHeldObject",
        "MoveHeldObject",
        "Place",
    ]
    assert all(item.allow_yaw_search for item in grounded)

    yawed = entities["can"]._pose.clone()
    yawed[:, :3, :3] = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    entities["can"]._pose = yawed
    executor = ProgramExecutor(program, env, settle_steps=0, record_runtime=False)
    executor._target_poses[step.id] = _pose(0.0, 0.2, 0.75)

    assert bool(executor._placement_orientation_satisfied(step, yawed)[0])


def test_dual_franka_handover_uses_explicit_exchange_clearance() -> None:
    entities = {
        "can": _FakeEntity("can", _pose(0.0, 0.2, 1.20), _box_vertices(0.03)),
        "target": _FakeEntity("target", _pose(0.2, 0.0, 0.75), _box_vertices(0.03)),
        "table": _FakeEntity("table", _pose(0.0, 0.0, 0.5), _box_vertices(0.5)),
    }
    env = _FakeEnv(entities)
    env.agent_robot_profile = "dual_franka"
    program = load_execution_program(
        instantiate_seed_graph(
            _handover_then_place_task(),
            {"can": "can", "target": "target"},
        )
    )
    step = next(
        candidate for candidate in program.semantic_steps if candidate.id == "handover"
    )
    state = _handover_held_state(env, entities["can"], arm="left_arm")
    held = state.get_held_object("physical_left_arm")
    assert held is not None
    grounder = ActionGrounder(program, env, lambda _uid: held.semantics)
    staging_edge = next(
        edge
        for edge in program.edges
        if edge.id in step.edge_ids
        and edge.actions[0]["target_binding"].get("kind") == "handover_staging"
    )
    handover_edge = next(
        edge
        for edge in program.edges
        if edge.id in step.edge_ids
        and edge.actions[0]["target_binding"].get("kind") == "handover_goal"
    )
    staging = grounder.ground(
        staging_edge.actions[0], step, arm="left_arm", state=state
    )
    handover = grounder.ground(
        handover_edge.actions[0], step, arm="coordinated", state=state
    )

    assert staging.motion_policy["exchange_clearance"] > 0.0
    assert handover.motion_policy["exchange_clearance"] > 0.0
    assert handover.motion_policy["lift_height"] > 0.0
    assert staging.target_object_pose is not None
    live_object_pose = entities["can"].get_local_pose(to_matrix=True)
    assert handover.motion_policy["middle_object_pose"][0, 2, 3] == pytest.approx(
        live_object_pose[0, 2, 3]
    )


def test_handover_candidates_avoid_occupied_table_center_and_lift_payload() -> None:
    entities = {
        "can": _FakeEntity("can", _pose(0.0, 0.2, 1.03), _box_vertices(0.03)),
        "target": _FakeEntity("target", _pose(0.35, 0.0, 1.03), _box_vertices(0.03)),
        "notebook": _FakeEntity("notebook", _pose(0.0, 0.0, 1.04), _box_vertices(0.05)),
        "table": _FakeEntity("table", _pose(0.0, 0.0, 0.5), _box_vertices(0.5)),
    }
    env = _FakeEnv(entities)
    env.agent_robot_profile = "dual_franka"
    program = load_execution_program(
        instantiate_seed_graph(
            _handover_then_place_task(),
            {"can": "can", "target": "target"},
        )
    )
    step = next(
        candidate for candidate in program.semantic_steps if candidate.id == "handover"
    )
    state = _handover_held_state(env, entities["can"], arm="left_arm")
    held = state.get_held_object("physical_left_arm")
    assert held is not None
    grounder = ActionGrounder(program, env, lambda _uid: held.semantics)
    staging_edge = next(
        edge
        for edge in program.edges
        if edge.id in step.edge_ids
        and edge.actions[0]["target_binding"].get("kind") == "handover_staging"
    )
    handover_edge = next(
        edge
        for edge in program.edges
        if edge.id in step.edge_ids
        and edge.actions[0]["target_binding"].get("kind") == "handover_goal"
    )

    staging = grounder.ground(
        staging_edge.actions[0], step, arm="left_arm", state=state
    )
    candidates = grounder.ground_candidates(
        handover_edge.actions[0], step, arm="coordinated", state=state
    )

    assert staging.target_object_pose is not None
    assert torch.linalg.vector_norm(staging.target_object_pose[0, :2, 3]) > 0.10
    assert float(staging.target_object_pose[0, 2, 3]) >= 1.15
    assert len(candidates) == 4
    assert all(
        torch.linalg.vector_norm(candidate.cfg["middle_object_pose"][0, :2, 3]) > 0.10
        for candidate in candidates[:2]
    )


def test_on_placement_grounding_samples_bounded_live_support_poses() -> None:
    entities = {
        "payload": _FakeEntity(
            "payload",
            _pose(0.0, -0.20, 0.79),
            _rect_vertices(0.03, 0.03, 0.03),
        ),
        "support": _FakeEntity(
            "support",
            _pose(0.0, 0.0, 0.75),
            _rect_vertices(0.20, 0.15, 0.01),
        ),
    }
    env = _FakeEnv(entities)
    program = load_execution_program(
        compile_task_agent(
            _task_agent(
                {
                    "id": "place",
                    "operator": "place_relative",
                    "object": "payload",
                    "actor": {"mode": "required", "arm": "left_arm"},
                    "goal": {
                        "reference_object": "support",
                        "relation": "on",
                        "orientation_goal": "none",
                    },
                    "depends_on": [],
                }
            )
        )
    )
    step = program.semantic_steps[0]
    edge = next(
        item
        for item in program.edges
        if item.id in step.edge_ids
        and item.actions[0]["target_binding"].get("kind") == "semantic_goal"
        and item.actions[0]["target_binding"].get("phase", "final") != "staging"
    )
    grounder = ActionGrounder(program, env, lambda _uid: None)

    candidates = grounder.ground_candidates(
        edge.actions[0],
        step,
        arm="left_arm",
        state=ExecutionState(last_qpos=env.robot.get_qpos()),
    )

    assert len(candidates) == 5
    assert [item.motion_policy["placement_candidate_index"] for item in candidates] == [
        0,
        1,
        2,
        3,
        4,
    ]
    offsets = [item.motion_policy["placement_xy_offset"][0] for item in candidates]
    assert len({tuple(float(value) for value in offset) for offset in offsets}) == 5
    support_lower = torch.tensor([-0.20, -0.15])
    support_upper = torch.tensor([0.20, 0.15])
    for item in candidates:
        center = item.target_object_pose[0, :2, 3]
        assert torch.all(center >= support_lower)
        assert torch.all(center <= support_upper)


def test_on_placement_candidates_respect_support_geometry_origin() -> None:
    support_vertices = _rect_vertices(0.10, 0.08, 0.01)
    support_vertices[:, 0] += 0.25
    entities = {
        "payload": _FakeEntity(
            "payload",
            _pose(0.0, -0.20, 0.79),
            _rect_vertices(0.03, 0.03, 0.03),
        ),
        "support": _FakeEntity(
            "support",
            _pose(0.10, 0.0, 0.75),
            support_vertices,
        ),
    }
    env = _FakeEnv(entities)
    program = load_execution_program(
        compile_task_agent(
            _task_agent(
                {
                    "id": "place",
                    "operator": "place_relative",
                    "object": "payload",
                    "actor": {"mode": "required", "arm": "left_arm"},
                    "goal": {
                        "reference_object": "support",
                        "relation": "on",
                        "orientation_goal": "none",
                    },
                    "depends_on": [],
                }
            )
        )
    )
    step = program.semantic_steps[0]
    edge = next(
        item
        for item in program.edges
        if item.id in step.edge_ids
        and item.actions[0]["target_binding"].get("kind") == "semantic_goal"
        and item.actions[0]["target_binding"].get("phase", "final") != "staging"
    )
    candidates = ActionGrounder(program, env, lambda _uid: None).ground_candidates(
        edge.actions[0],
        step,
        arm="left_arm",
        state=ExecutionState(last_qpos=env.robot.get_qpos()),
    )

    support_world = support_vertices[:, :2] + torch.tensor([0.10, 0.0])
    lower = support_world.min(dim=0).values + 0.002
    upper = support_world.max(dim=0).values - 0.002
    payload_local = entities["payload"]._vertices[:, :2]
    for candidate in candidates:
        origin = candidate.target_object_pose[0, :2, 3]
        assert torch.all(origin + payload_local.min(dim=0).values >= lower)
        assert torch.all(origin + payload_local.max(dim=0).values <= upper)


def test_build_stack_root_compiles_to_generic_table_support() -> None:
    program = load_execution_program(
        compile_task_agent(
            _task_agent(
                {
                    "id": "stack",
                    "operator": "build_stack",
                    "objects": ["base", "nested"],
                    "actor": {"mode": "auto"},
                    "goal": {
                        "anchor": "table_center",
                        "stack_mode": "nested",
                        "orientation_goal": "none",
                    },
                    "depends_on": [],
                }
            )
        )
    )

    root, child = program.semantic_steps
    assert root.goal["relation"] == "on"
    assert root.goal["reference_object"] == "table"
    assert root.postcondition["reference_object"] == "table"
    assert child.goal["relation"] == "inside"
    assert child.goal["reference_object"] == "base"


def test_executor_tries_next_placement_pose_after_planning_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entities = {
        "payload": _FakeEntity(
            "payload",
            _pose(0.0, -0.20, 0.79),
            _rect_vertices(0.03, 0.03, 0.03),
        ),
        "support": _FakeEntity(
            "support",
            _pose(0.0, 0.0, 0.75),
            _rect_vertices(0.20, 0.15, 0.01),
        ),
    }
    env = _FakeEnv(entities)
    executor = ProgramExecutor(
        load_execution_program(
            compile_task_agent(
                _task_agent(
                    {
                        "id": "place",
                        "operator": "place_relative",
                        "object": "payload",
                        "actor": {"mode": "required", "arm": "left_arm"},
                        "goal": {
                            "reference_object": "support",
                            "relation": "on",
                            "orientation_goal": "none",
                        },
                        "depends_on": [],
                    }
                )
            )
        ),
        env,
        settle_steps=0,
        record_runtime=False,
    )
    step = executor.program.semantic_steps[0]
    edge = next(
        item
        for item in executor.program.edges
        if item.id in step.edge_ids
        and item.actions[0]["target_binding"].get("kind") == "semantic_goal"
        and item.actions[0]["target_binding"].get("phase", "final") != "staging"
    )
    state = ExecutionState(last_qpos=env.robot.get_qpos())

    def plan(grounded: GroundedAction, _state: ExecutionState) -> ActionOutcome:
        index = int(grounded.motion_policy["placement_candidate_index"])
        return ActionOutcome(
            trajectory=torch.zeros(1, 1, env.robot.dof),
            success=torch.tensor([index == 1]),
            next_state=state,
            grounded=grounded,
        )

    monkeypatch.setattr(executor.adapter, "plan", plan)
    grounded, outcome = executor._ground_and_plan_candidates(
        edge.actions[0],
        step,
        arm="left_arm",
        state=state,
        active=torch.tensor([True]),
    )

    assert bool(outcome.success[0])
    assert grounded.motion_policy["placement_candidate_index"] == 1
    assert outcome.planner_trace["selected_grounding_candidate"] == 1
    assert len(outcome.planner_trace["grounding_candidates"]) == 2


def test_post_release_candidate_search_skips_the_released_pose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entities = {
        "payload": _FakeEntity(
            "payload",
            _pose(0.0, -0.20, 0.79),
            _rect_vertices(0.03, 0.03, 0.03),
        ),
        "support": _FakeEntity(
            "support",
            _pose(0.0, 0.0, 0.75),
            _rect_vertices(0.20, 0.15, 0.01),
        ),
    }
    env = _FakeEnv(entities)
    executor = ProgramExecutor(
        load_execution_program(
            compile_task_agent(
                _task_agent(
                    {
                        "id": "place",
                        "operator": "place_relative",
                        "object": "payload",
                        "actor": {"mode": "required", "arm": "left_arm"},
                        "goal": {
                            "reference_object": "support",
                            "relation": "on",
                            "orientation_goal": "none",
                        },
                        "depends_on": [],
                    }
                )
            )
        ),
        env,
        settle_steps=0,
        record_runtime=False,
    )
    step = executor.program.semantic_steps[0]
    edge = next(
        item
        for item in executor.program.edges
        if item.id in step.edge_ids
        and item.actions[0]["target_binding"].get("kind") == "semantic_goal"
        and item.actions[0]["target_binding"].get("phase", "final") != "staging"
    )
    state = ExecutionState(last_qpos=env.robot.get_qpos())
    executor._placement_candidate_history[(step.id, "left_arm")] = {0}

    def plan(grounded: GroundedAction, _state: ExecutionState) -> ActionOutcome:
        return ActionOutcome(
            trajectory=torch.zeros(1, 1, env.robot.dof),
            success=torch.tensor([True]),
            next_state=state,
            grounded=grounded,
        )

    monkeypatch.setattr(executor.adapter, "plan", plan)
    grounded, outcome = executor._ground_and_plan_candidates(
        edge.actions[0],
        step,
        arm="left_arm",
        state=state,
        active=torch.tensor([True]),
    )

    assert grounded.motion_policy["placement_candidate_index"] == 1
    assert outcome.planner_trace["grounding_candidates"][0] == {
        "candidate_index": 0,
        "status": "previously_released",
    }


def test_unstable_placement_recovery_replays_pick_before_another_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entities = {
        "payload": _FakeEntity(
            "payload",
            _pose(0.0, 0.0, 0.79),
            _rect_vertices(0.03, 0.03, 0.03),
        ),
        "support": _FakeEntity(
            "support",
            _pose(0.0, 0.0, 0.75),
            _rect_vertices(0.10, 0.08, 0.01),
        ),
    }
    env = _FakeEnv(entities)
    executor = ProgramExecutor(
        load_execution_program(
            compile_task_agent(
                _task_agent(
                    {
                        "id": "place",
                        "operator": "place_relative",
                        "object": "payload",
                        "actor": {"mode": "required", "arm": "left_arm"},
                        "goal": {
                            "reference_object": "support",
                            "relation": "on",
                            "orientation_goal": "none",
                        },
                        "depends_on": [],
                    }
                )
            )
        ),
        env,
        settle_steps=0,
        record_runtime=False,
    )
    step = executor.program.semantic_steps[0]
    replayed_actions: list[str] = []
    verification_count = 0

    def ensure_assignment(_step: SemanticStep, failed: torch.Tensor) -> None:
        executor._assignments[_step.id] = [
            None if bool(failed[env_id]) else "left_arm"
            for env_id in range(len(failed))
        ]

    def execute(
        edge: ExecutionEdge,
        _step: SemanticStep,
        *,
        failed: torch.Tensor,
    ) -> _EdgeResult:
        replayed_actions.append(str(edge.actions[0]["atomic_action_class"]))
        return _EdgeResult([], failed.clone(), [], executed=~failed)

    def verify(
        _step: SemanticStep,
        failed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nonlocal verification_count
        verification_count += 1
        success = torch.tensor([verification_count == 2]) & ~failed
        return failed | ~success, success, executor._entity_pose("payload")[:, :3, 3]

    monkeypatch.setattr(executor, "_ensure_assignment", ensure_assignment)
    monkeypatch.setattr(executor, "_execute_edge_with_retries", execute)
    monkeypatch.setattr(executor, "_verify_step", verify)
    recorder = RuntimeRecorder(
        executor.program,
        num_envs=1,
        enabled=False,
    )

    recovery = executor._recover_unstable_placement(
        step,
        torch.tensor([True]),
        recorder=recorder,
    )

    first_action = str(
        executor.edges[step.edge_ids[0]].actions[0]["atomic_action_class"]
    )
    assert replayed_actions.count(first_action) == 2
    assert verification_count == 2
    assert bool(recovery.succeeded[0])
    assert not bool(recovery.failed[0])
    assert recovery.failure_events == []


def test_unstable_placement_recovery_reports_its_own_planning_blocker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entities = {
        "payload": _FakeEntity(
            "payload",
            _pose(0.0, 0.0, 0.79),
            _rect_vertices(0.03, 0.03, 0.03),
        ),
        "support": _FakeEntity(
            "support",
            _pose(0.0, 0.0, 0.75),
            _rect_vertices(0.10, 0.08, 0.01),
        ),
    }
    executor = ProgramExecutor(
        load_execution_program(
            compile_task_agent(
                _task_agent(
                    {
                        "id": "place",
                        "operator": "place_relative",
                        "object": "payload",
                        "actor": {"mode": "required", "arm": "left_arm"},
                        "goal": {
                            "reference_object": "support",
                            "relation": "on",
                            "orientation_goal": "none",
                        },
                        "depends_on": [],
                    }
                )
            )
        ),
        _FakeEnv(entities),
        settle_steps=0,
        record_runtime=False,
    )
    step = executor.program.semantic_steps[0]

    def fail_assignment(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("no plan")

    monkeypatch.setattr(executor, "_ensure_assignment", fail_assignment)

    recovery = executor._recover_unstable_placement(
        step,
        torch.tensor([True]),
        recorder=RuntimeRecorder(executor.program, num_envs=1, enabled=False),
    )

    assert bool(recovery.failed[0])
    assert bool(recovery.covered_failures[0])
    assert len(recovery.failure_events) == 1
    event = recovery.failure_events[0]
    assert event["failure_type"] == "search_exhausted"
    assert event["phase"] == "recovery"
    assert event["origin_edge_id"] == step.edge_ids[-1]
    assert event["blocking_edge_id"] == step.edge_ids[0]


def test_handover_height_accounts_for_obstacle_and_tool_envelope() -> None:
    entities = {
        "can": _FakeEntity("can", _pose(0.0, 0.2, 1.03), _box_vertices(0.03)),
        "target": _FakeEntity("target", _pose(0.35, 0.0, 1.03), _box_vertices(0.03)),
        "shelf": _FakeEntity(
            "shelf",
            _pose(0.0, 0.0, 1.05),
            _rect_vertices(0.40, 0.35, 0.05),
        ),
        "table": _FakeEntity("table", _pose(0.0, 0.0, 0.5), _box_vertices(0.5)),
    }
    env = _FakeEnv(entities)
    env.agent_robot_profile = "dual_franka"
    program = load_execution_program(
        instantiate_seed_graph(
            _handover_then_place_task(),
            {"can": "can", "target": "target"},
        )
    )
    step = next(
        candidate for candidate in program.semantic_steps if candidate.id == "handover"
    )
    state = _handover_held_state(env, entities["can"])
    held = state.get_held_object("physical_left_arm")
    assert held is not None
    grounder = ActionGrounder(program, env, lambda _uid: held.semantics)
    staging = next(
        edge.actions[0]
        for edge in program.edges
        if edge.id in step.edge_ids
        and edge.actions[0]["target_binding"].get("kind") == "handover_staging"
    )

    grounded = grounder.ground(staging, step, arm="left_arm", state=state)

    assert grounded.target_object_pose is not None
    obstacle_top = 1.10
    object_bottom = -0.03
    object_clearance = 0.06
    tool_vertical_envelope = 0.025 + 0.04
    expected_height = (
        obstacle_top + object_clearance + tool_vertical_envelope - object_bottom
    )
    assert grounded.target_object_pose[0, 2, 3] == pytest.approx(expected_height)


def test_handover_workspace_rejects_points_outside_shared_reach() -> None:
    entities = {
        "can": _FakeEntity("can", _pose(0.0, 0.2, 1.20), _box_vertices(0.03)),
        "target": _FakeEntity("target", _pose(0.2, 0.0, 0.75), _box_vertices(0.03)),
        "table": _FakeEntity("table", _pose(0.0, 0.0, 0.5), _box_vertices(0.5)),
    }
    env = _FakeEnv(entities)
    program = load_execution_program(
        instantiate_seed_graph(
            _handover_then_place_task(),
            {"can": "can", "target": "target"},
        )
    )
    step = next(
        candidate for candidate in program.semantic_steps if candidate.id == "handover"
    )
    state = _handover_held_state(env, entities["can"])
    held = state.get_held_object("physical_left_arm")
    assert held is not None
    grounder = ActionGrounder(program, env, lambda _uid: held.semantics)
    staging = next(
        edge.actions[0]
        for edge in program.edges
        if edge.id in step.edge_ids
        and edge.actions[0]["target_binding"].get("kind") == "handover_staging"
    )
    staging = {
        **staging,
        "motion_policy_config": {"exchange_maximum_reach": 0.20},
    }

    with pytest.raises(ValueError, match="reachable intersection"):
        grounder.ground(staging, step, arm="left_arm", state=state)


def test_handover_grounding_preserves_the_original_object_affordance() -> None:
    entities = {
        "can": _FakeEntity("can", _pose(0.0, 0.2, 1.20), _box_vertices(0.03)),
        "target": _FakeEntity("target", _pose(0.2, 0.0, 0.75), _box_vertices(0.03)),
        "table": _FakeEntity("table", _pose(0.0, 0.0, 0.5), _box_vertices(0.5)),
    }
    env = _FakeEnv(entities)
    program = load_execution_program(
        instantiate_seed_graph(
            _handover_then_place_task(),
            {"can": "can", "target": "target"},
        )
    )
    step = next(
        candidate for candidate in program.semantic_steps if candidate.id == "handover"
    )
    state = _handover_held_state(env, entities["can"])
    held = state.get_held_object("physical_left_arm")
    assert held is not None
    held.object_to_eef[:, 1, 3] *= -1.0
    grounder = ActionGrounder(program, env, lambda _uid: held.semantics)
    action = next(
        edge.actions[0]
        for edge in program.edges
        if edge.id in step.edge_ids
        and edge.actions[0]["target_binding"].get("kind") == "handover_goal"
    )

    grounded = grounder.ground(action, step, arm="coordinated", state=state)

    assert grounded.target.semantics.affordance is held.semantics.affordance


def test_robot_relative_left_uses_live_right_to_left_arm_axis() -> None:
    env = _FakeEnv()

    forward, lateral = robot_frame_axes(env)
    offset = relation_offset(
        env,
        "left_of",
        frame="robot",
        forward_distance=0.10,
        lateral_distance=0.12,
        dtype=torch.float32,
        device=env.device,
    )

    torch.testing.assert_close(forward, torch.tensor([[-1.0, 0.0]]))
    torch.testing.assert_close(lateral, torch.tensor([[0.0, -1.0]]))
    assert offset is not None
    torch.testing.assert_close(offset, torch.tensor([[0.0, -0.12, 0.0]]))


def test_directional_verification_rejects_grounded_target_on_wrong_side() -> None:
    entities = {
        "can": _FakeEntity("can", _pose(0.0, 0.12, 0.75), _box_vertices(0.03)),
        "target": _FakeEntity("target", _pose(0.0, 0.0, 0.75), _box_vertices(0.03)),
    }
    env = _FakeEnv(entities)
    program = load_execution_program(
        compile_task_agent(
            _task_agent(
                {
                    "id": "place",
                    "operator": "place_relative",
                    "object": "can",
                    "actor": {"mode": "required", "arm": "left_arm"},
                    "goal": {
                        "reference_object": "target",
                        "relation": "left_of",
                    },
                    "depends_on": [],
                }
            )
        )
    )
    executor = ProgramExecutor(program, env, settle_steps=0, record_runtime=False)
    step = program.semantic_steps[0]
    step.goal["relation_frame"] = "robot"
    executor._targets[step.id] = entities["can"].get_local_pose(to_matrix=True)[
        :, :3, 3
    ]
    executor._policies[step.id] = {
        "postcondition_tolerance": 0.08,
        "relation_clearance": 0.01,
    }

    failed, success, _ = executor._verify_step(step, torch.tensor([False]))

    assert bool(failed[0])
    assert not bool(success[0])


def test_directional_verification_accepts_support_height_settling() -> None:
    entities = {
        "can": _FakeEntity("can", _pose(0.0, -0.12, 0.75), _box_vertices(0.03)),
        "target": _FakeEntity("target", _pose(0.0, 0.0, 0.75), _box_vertices(0.03)),
    }
    env = _FakeEnv(entities)
    program = load_execution_program(
        compile_task_agent(
            _task_agent(
                {
                    "id": "place",
                    "operator": "place_relative",
                    "object": "can",
                    "actor": {"mode": "required", "arm": "left_arm"},
                    "goal": {
                        "reference_object": "target",
                        "relation": "left_of",
                    },
                    "depends_on": [],
                }
            )
        )
    )
    executor = ProgramExecutor(program, env, settle_steps=0, record_runtime=False)
    step = program.semantic_steps[0]
    step.goal["relation_frame"] = "robot"
    executor._targets[step.id] = torch.tensor([[0.0, -0.12, 0.90]])
    executor._policies[step.id] = {
        "postcondition_tolerance": 0.08,
        "relation_clearance": 0.01,
    }

    failed, success, _ = executor._verify_step(step, torch.tensor([False]))

    assert not bool(failed[0])
    assert bool(success[0])


def test_legacy_released_above_relation_verifies_as_physical_support() -> None:
    entities = {
        "payload": _FakeEntity(
            "payload",
            _pose(0.0, 0.0, 0.79),
            _rect_vertices(0.03, 0.03, 0.03),
        ),
        "support": _FakeEntity(
            "support",
            _pose(0.0, 0.0, 0.75),
            _rect_vertices(0.10, 0.08, 0.01),
        ),
    }
    env = _FakeEnv(entities)
    program = load_execution_program(
        compile_task_agent(
            _task_agent(
                {
                    "id": "place",
                    "operator": "place_relative",
                    "object": "payload",
                    "actor": {"mode": "required", "arm": "left_arm"},
                    "goal": {
                        "reference_object": "support",
                        "relation": "on",
                    },
                    "depends_on": [],
                }
            )
        )
    )
    executor = ProgramExecutor(program, env, settle_steps=0, record_runtime=False)
    step = program.semantic_steps[0]
    step.goal["relation"] = "above"
    executor._targets[step.id] = torch.tensor([[0.0, 0.0, 1.0]])

    failed, success, _ = executor._verify_step(step, torch.tensor([False]))

    assert not bool(failed[0])
    assert bool(success[0])


def test_handover_retreat_clears_exchange_toward_transfer_workspace() -> None:
    entities = {
        "can": _FakeEntity("can", _pose(0.0, 0.2, 1.106), _box_vertices(0.03)),
        "target": _FakeEntity("target", _pose(0.2, 0.0, 0.75), _box_vertices(0.03)),
        "table": _FakeEntity("table", _pose(0.0, 0.0, 0.5), _box_vertices(0.5)),
    }
    env = _FakeEnv(entities)
    env.agent_robot_profile = "dual_franka"
    high_left = _pose(0.0, 0.2, 1.106)
    right = _pose(0.0, -0.2, 0.8)
    env.get_current_xpos_agent = lambda: (high_left, right)
    program = load_execution_program(
        instantiate_seed_graph(
            _handover_then_place_task(),
            {"can": "can", "target": "target"},
        )
    )
    step = next(
        candidate for candidate in program.semantic_steps if candidate.id == "handover"
    )
    state = _held_state(env, entities["can"], arm="left_arm")
    held = state.get_held_object("physical_left_arm")
    assert held is not None
    grounder = ActionGrounder(program, env, lambda _uid: held.semantics)
    retreat_edge = next(
        edge
        for edge in program.edges
        if edge.id in step.edge_ids
        and edge.actions[0]["target_binding"].get("source") == "handover"
    )

    grounded = grounder.ground(
        retreat_edge.actions[0], step, arm="left_arm", state=state
    )

    torch.testing.assert_close(
        grounded.target.xpos[0, :2, 3],
        torch.tensor([0.0, 0.10]),
    )
    assert grounded.target.xpos[0, 2, 3] == pytest.approx(1.206)
    assert grounded.cfg["retreat_distance"] == pytest.approx(0.10)
    assert grounded.cfg["maximum_eef_height"] == pytest.approx(1.50)


def test_handover_retreat_and_home_block_receiver_continuation() -> None:
    entities = {
        "can": _FakeEntity("can", _pose(0.0, 0.2, 0.75), _box_vertices(0.03)),
        "target": _FakeEntity("target", _pose(0.2, 0.0, 0.75), _box_vertices(0.03)),
        "table": _FakeEntity("table", _pose(0.0, 0.0, 0.5), _box_vertices(0.5)),
    }
    program = load_execution_program(
        instantiate_seed_graph(
            _handover_then_place_task(),
            {"can": "can", "target": "target"},
        )
    )
    executor = ProgramExecutor(program, _FakeEnv(entities), record_runtime=False)
    handover = next(
        step for step in program.semantic_steps if step.operator == "handover"
    )
    handover_edges = [edge for edge in program.edges if edge.id in handover.edge_ids]
    retreat = next(
        edge
        for edge in handover_edges
        if edge.actions[0]["target_binding"].get("source") == "handover"
    )
    home = next(
        edge
        for edge in handover_edges
        if edge.actions[0]["target_binding"].get("operation") == "handover_home"
    )

    assert executor._edge_failure_policy(retreat) == "safety_required"
    assert executor._edge_failure_policy(home) == "best_effort"


def test_release_retreat_is_required_and_exact_home_is_best_effort() -> None:
    entities = {
        "can": _FakeEntity("can", _pose(0.0, 0.2, 0.75), _box_vertices(0.03)),
        "target": _FakeEntity("target", _pose(0.2, 0.0, 0.75), _box_vertices(0.03)),
    }
    program = load_execution_program(
        compile_task_agent_v2(
            _task_agent(
                {
                    "id": "place",
                    "operator": "place_relative",
                    "object": "can",
                    "actor": {"mode": "required", "arm": "left_arm"},
                    "goal": {
                        "reference_object": "target",
                        "relation": "left_of",
                    },
                    "depends_on": [],
                }
            )
        )
    )
    executor = ProgramExecutor(program, _FakeEnv(entities), record_runtime=False)
    step = program.semantic_steps[0]
    edges = [edge for edge in program.edges if edge.id in step.edge_ids]
    retreat = next(
        edge
        for edge in edges
        if edge.actions[0]["atomic_action_class"] == "MoveEndEffector"
    )
    home = next(
        edge for edge in edges if edge.actions[0]["atomic_action_class"] == "MoveJoints"
    )

    assert executor._edge_failure_policy(retreat) == "safety_required"
    assert executor._edge_failure_policy(home) == "best_effort"


def test_best_effort_home_does_not_veto_required_arm_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entities = {
        "can": _FakeEntity("can", _pose(0.0, -0.2, 0.75), _box_vertices(0.03)),
        "target": _FakeEntity("target", _pose(0.0, 0.0, 0.75), _box_vertices(0.03)),
    }
    graph = compile_task_agent_v2(
        _task_agent(
            {
                "id": "place",
                "operator": "place_relative",
                "object": "can",
                "actor": {"mode": "required", "arm": "left_arm"},
                "goal": {"reference_object": "target", "relation": "left_of"},
                "depends_on": [],
            }
        )
    )
    executor = ProgramExecutor(
        load_execution_program(graph),
        _FakeEnv(entities),
        record_runtime=False,
    )
    step = executor.program.semantic_steps[0]

    def ground(action: dict[str, Any], *_args: Any, **_kwargs: Any) -> GroundedAction:
        return GroundedAction(
            action_class=str(action["atomic_action_class"]),
            arm="left_arm",
            control=str(action["control"]),
            target=SimpleNamespace(xpos=None),
            cfg={},
        )

    def plan(grounded: GroundedAction, state: ExecutionState) -> ActionOutcome:
        return ActionOutcome(
            trajectory=torch.zeros(1, 1, executor.env.robot.dof),
            success=torch.tensor([grounded.action_class != "MoveJoints"]),
            next_state=state,
            grounded=grounded,
            planner_trace={"primary_strategy": "motion_gen"},
        )

    monkeypatch.setattr(executor.grounder, "ground", ground)
    monkeypatch.setattr(executor, "_with_downstream_targets", lambda *args: args[-1])
    monkeypatch.setattr(executor.adapter, "plan", plan)

    candidate = executor._candidate(step, "left_arm", torch.tensor([False]))

    assert bool(candidate.feasible[0])
    assert any("best-effort action degraded" in item for item in candidate.warnings)


def test_best_effort_home_exception_does_not_fail_semantic_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = compile_task_agent_v2(
        _task_agent(
            {
                "id": "place",
                "operator": "place_relative",
                "object": "can",
                "actor": {"mode": "required", "arm": "left_arm"},
                "goal": {"reference_object": "target", "relation": "left_of"},
                "depends_on": [],
            }
        )
    )
    executor = ProgramExecutor(
        load_execution_program(graph),
        _FakeEnv(),
        settle_steps=0,
        record_runtime=False,
    )
    monkeypatch.setattr(
        executor,
        "_ensure_assignment",
        lambda step, _failed: executor._assignments.setdefault(step.id, ["left_arm"]),
    )

    def execute(edge: ExecutionEdge, _step: SemanticStep, *, failed: torch.Tensor):
        if executor._edge_failure_policy(edge) == "best_effort":
            raise RuntimeError("home search failed")
        return SimpleNamespace(
            actions=[],
            failed=failed.clone(),
            grounded=[],
            planner_traces=[],
            executed=~failed,
        )

    monkeypatch.setattr(executor, "_execute_edge_with_retries", execute)
    monkeypatch.setattr(
        executor,
        "_verify_step",
        lambda _step, failed: (failed, ~failed, torch.zeros(1, 3)),
    )

    result = executor.run()

    assert bool(result.success[0])
    assert len(result.failure_events) == 1
    assert result.failure_events[0]["failure_type"] == "search_exhausted"
    assert result.failure_events[0]["failure_policy"] == "best_effort"
    assert result.failure_events[0]["fatal"] is False
    assert result.failure_events[0]["evidence"]["exception"].endswith(
        "home search failed"
    )


def test_candidate_failure_reports_real_blocking_safety_edge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entities = {
        "can": _FakeEntity("can", _pose(0.0, -0.2, 0.75), _box_vertices(0.03)),
        "target": _FakeEntity("target", _pose(0.0, 0.0, 0.75), _box_vertices(0.03)),
    }
    executor = ProgramExecutor(
        load_execution_program(
            compile_task_agent_v2(
                _task_agent(
                    {
                        "id": "place",
                        "operator": "place_relative",
                        "object": "can",
                        "actor": {"mode": "required", "arm": "left_arm"},
                        "goal": {
                            "reference_object": "target",
                            "relation": "left_of",
                        },
                        "depends_on": [],
                    }
                )
            )
        ),
        _FakeEnv(entities),
        record_runtime=False,
    )
    step = executor.program.semantic_steps[0]

    def ground(action: dict[str, Any], *_args: Any, **_kwargs: Any) -> GroundedAction:
        return GroundedAction(
            action_class=str(action["atomic_action_class"]),
            arm="left_arm",
            control=str(action["control"]),
            target=SimpleNamespace(xpos=None),
            cfg={},
        )

    def plan(grounded: GroundedAction, state: ExecutionState) -> ActionOutcome:
        return ActionOutcome(
            trajectory=torch.zeros(1, 1, executor.env.robot.dof),
            success=torch.tensor([grounded.action_class != "MoveEndEffector"]),
            next_state=state,
            grounded=grounded,
            planner_trace={"primary_strategy": "motion_gen"},
        )

    monkeypatch.setattr(executor.grounder, "ground", ground)
    monkeypatch.setattr(executor, "_with_downstream_targets", lambda *args: args[-1])
    monkeypatch.setattr(executor.adapter, "plan", plan)
    candidate = executor._candidate(step, "left_arm", torch.tensor([False]))
    executor._assignments[step.id] = [None]
    executor._report_candidates(step, (candidate,))
    first_edge = executor.edges[step.edge_ids[0]]

    events = executor._failure_events(
        first_edge,
        step,
        torch.tensor([True]),
        postcondition=False,
        executed=torch.tensor([False]),
        fallen_transition=torch.tensor([False]),
    )

    assert len(events) == 1
    event = events[0]
    assert event["failure_type"] == "search_exhausted"
    assert event["failure_policy"] == "safety_required"
    assert event["atomic_action"] == "MoveEndEffector"
    assert event["blocking_edge_id"] != first_edge.id
    assert event["planning_stage"] == "candidate_suffix"
    assert "not a geometric proof" in event["reason"]


def test_on_relation_rejects_preserve_orientation_drift() -> None:
    rotated = _pose(0.0, 0.0, 0.82)
    rotated[:, :3, :3] = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    )
    entities = {
        "can": _FakeEntity("can", rotated, _rect_vertices(0.03, 0.03, 0.06)),
        "notebook": _FakeEntity(
            "notebook",
            _pose(0.0, 0.0, 0.75),
            _rect_vertices(0.10, 0.08, 0.01),
        ),
    }
    executor = ProgramExecutor(
        load_execution_program(
            compile_task_agent(
                _task_agent(
                    {
                        "id": "place",
                        "operator": "place_relative",
                        "object": "can",
                        "actor": {"mode": "required", "arm": "left_arm"},
                        "goal": {
                            "reference_object": "notebook",
                            "relation": "on",
                            "orientation_goal": "preserve",
                        },
                        "depends_on": [],
                    }
                )
            )
        ),
        _FakeEnv(entities),
        settle_steps=0,
        record_runtime=False,
    )
    step = executor.program.semantic_steps[0]
    executor._orientation_references[step.id] = _pose(0.0, 0.0, 0.82)
    executor._policies[step.id] = {
        "preserve_orientation_tolerance": torch.pi / 12,
    }

    failed, success, _ = executor._verify_step(step, torch.tensor([False]))

    assert bool(failed[0])
    assert not bool(success[0])
    assert executor._orientation_errors[step.id][0] > torch.pi / 12


def test_inside_relation_accepts_settling_orientation_drift() -> None:
    rotated = _pose(0.02, -0.02, 0.72)
    rotated[:, :3, :3] = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    )
    entities = {
        "can": _FakeEntity("can", rotated, _rect_vertices(0.03, 0.03, 0.06)),
        "basket": _FakeEntity(
            "basket",
            _pose(0.0, 0.0, 0.70),
            _rect_vertices(0.10, 0.10, 0.08),
        ),
    }
    executor = ProgramExecutor(
        load_execution_program(
            compile_task_agent(
                _task_agent(
                    {
                        "id": "place",
                        "operator": "place_relative",
                        "object": "can",
                        "actor": {"mode": "required", "arm": "left_arm"},
                        "goal": {
                            "reference_object": "basket",
                            "relation": "inside",
                            "orientation_goal": "preserve",
                        },
                        "depends_on": [],
                    }
                )
            )
        ),
        _FakeEnv(entities),
        settle_steps=0,
        record_runtime=False,
    )
    step = executor.program.semantic_steps[0]
    executor._orientation_references[step.id] = _pose(0.02, -0.02, 0.72)

    failed, success, _ = executor._verify_step(step, torch.tensor([False]))

    assert not bool(failed[0])
    assert bool(success[0])
    assert step.id not in executor._orientation_errors


@pytest.mark.parametrize(
    ("orientation_goal", "expected_success"),
    (("upright", False), ("none", True)),
)
def test_on_relation_applies_only_the_requested_orientation_goal(
    orientation_goal: str,
    expected_success: bool,
) -> None:
    fallen = _pose(0.0, 0.0, 0.79)
    fallen[:, :3, :3] = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    )
    entities = {
        "can": _FakeEntity("can", fallen, _rect_vertices(0.03, 0.03, 0.06)),
        "notebook": _FakeEntity(
            "notebook",
            _pose(0.0, 0.0, 0.75),
            _rect_vertices(0.10, 0.08, 0.01),
        ),
    }
    executor = ProgramExecutor(
        load_execution_program(
            compile_task_agent(
                _task_agent(
                    {
                        "id": "place",
                        "operator": "place_relative",
                        "object": "can",
                        "actor": {"mode": "required", "arm": "left_arm"},
                        "goal": {
                            "reference_object": "notebook",
                            "relation": "on",
                            "orientation_goal": orientation_goal,
                        },
                        "depends_on": [],
                    }
                )
            )
        ),
        _FakeEnv(entities),
        settle_steps=0,
        record_runtime=False,
    )
    step = executor.program.semantic_steps[0]

    failed, success, _ = executor._verify_step(step, torch.tensor([False]))

    assert bool(success[0]) is expected_success
    assert bool(failed[0]) is not expected_success


def test_support_stability_window_rejects_motion_after_initial_contact() -> None:
    payload = _FakeEntity(
        "payload",
        _pose(0.0, 0.0, 0.79),
        _rect_vertices(0.03, 0.03, 0.03),
    )
    support = _FakeEntity(
        "support",
        _pose(0.0, 0.0, 0.75),
        _rect_vertices(0.10, 0.08, 0.01),
    )
    env = _FakeEnv({"payload": payload, "support": support})
    update_count = 0

    def update(*, step: int) -> None:
        nonlocal update_count
        del step
        update_count += 1
        payload.lin_vel[:, 0] = 0.10

    env.sim.update = update
    executor = ProgramExecutor(
        load_execution_program(
            compile_task_agent(
                _task_agent(
                    {
                        "id": "place",
                        "operator": "place_relative",
                        "object": "payload",
                        "actor": {"mode": "required", "arm": "left_arm"},
                        "goal": {
                            "reference_object": "support",
                            "relation": "on",
                            "orientation_goal": "none",
                        },
                        "depends_on": [],
                    }
                )
            )
        ),
        env,
        settle_steps=0,
        record_runtime=False,
    )
    step = executor.program.semantic_steps[0]

    failed, success, _ = executor._verify_step(step, torch.tensor([False]))

    assert update_count == executor.support_stability_samples - 1
    assert bool(failed[0])
    assert not bool(success[0])


def test_support_stability_reads_real_rigid_object_body_state() -> None:
    payload = _FakeEntity(
        "payload",
        _pose(0.0, 0.0, 0.79),
        _rect_vertices(0.03, 0.03, 0.03),
    )
    del payload.lin_vel
    del payload.ang_vel
    payload.body_state = torch.zeros(1, 13)
    executor = ProgramExecutor(
        load_execution_program(
            compile_task_agent(_task_agent(_hold_step("hold", "payload", "left_arm")))
        ),
        _FakeEnv({"payload": payload}),
        settle_steps=0,
        record_runtime=False,
    )

    assert bool(executor._entity_motion_stable("payload")[0])
    payload.body_state[:, 7] = 0.10
    assert not bool(executor._entity_motion_stable("payload")[0])


def test_final_support_revalidation_detects_later_chain_damage() -> None:
    payload = _FakeEntity(
        "payload",
        _pose(0.0, 0.0, 0.79),
        _rect_vertices(0.03, 0.03, 0.03),
    )
    support = _FakeEntity(
        "support",
        _pose(0.0, 0.0, 0.75),
        _rect_vertices(0.10, 0.08, 0.01),
    )
    executor = ProgramExecutor(
        load_execution_program(
            compile_task_agent(
                _task_agent(
                    {
                        "id": "place",
                        "operator": "place_relative",
                        "object": "payload",
                        "actor": {"mode": "required", "arm": "left_arm"},
                        "goal": {
                            "reference_object": "support",
                            "relation": "on",
                            "orientation_goal": "none",
                        },
                        "depends_on": [],
                    }
                )
            )
        ),
        _FakeEnv({"payload": payload, "support": support}),
        settle_steps=0,
        record_runtime=False,
    )
    step = executor.program.semantic_steps[0]
    failed, success, _ = executor._verify_step(step, torch.tensor([False]))
    assert not bool(failed[0])
    assert bool(success[0])

    payload._pose[:, 2, 3] += 0.20
    failures = executor._revalidate_support_relations()

    assert bool(failures[step.id][0])


def test_support_relation_state_rejects_cycles() -> None:
    executor = ProgramExecutor(
        load_execution_program(
            compile_task_agent(
                _task_agent(
                    {
                        "id": "place_b",
                        "operator": "place_relative",
                        "object": "b",
                        "actor": {"mode": "required", "arm": "left_arm"},
                        "goal": {
                            "reference_object": "a",
                            "relation": "on",
                            "orientation_goal": "none",
                        },
                        "depends_on": [],
                    }
                )
            )
        ),
        _FakeEnv(
            {
                "a": _FakeEntity("a", _pose(0.0, 0.0, 0.75), _box_vertices(0.03)),
                "b": _FakeEntity("b", _pose(0.0, 0.0, 0.81), _box_vertices(0.03)),
            }
        ),
        settle_steps=0,
        record_runtime=False,
    )
    step_b = executor.program.semantic_steps[0]
    step_a = replace(step_b, id="prior", object_uid="a")
    executor._commit_support_relation(step_a, "b", torch.tensor([True]))

    cycle_free = executor._support_cycle_free("b", "a", torch.tensor([True]))

    assert not bool(cycle_free[0])


def test_standalone_handover_assigns_its_pickup_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entities = {
        "can": _FakeEntity("can", _pose(0.0, 0.2, 0.75), _box_vertices(0.03)),
        "table": _FakeEntity("table", _pose(0.0, 0.0, 0.5), _box_vertices(0.5)),
    }
    task = _handover_then_place_task()
    task["level"] = "L1"
    task["task_instances"] = [task["task_instances"][0]]
    task["success"] = {"type": "handover_complete"}
    program = load_execution_program(instantiate_seed_graph(task, {"can": "can"}))
    executor = ProgramExecutor(program, _FakeEnv(entities), record_runtime=False)
    step = program.semantic_steps[0]
    calls: list[str] = []

    def candidate(_step: SemanticStep, arm: str, _failed: torch.Tensor) -> Any:
        calls.append(arm)
        return SimpleNamespace(feasible=torch.tensor([True]))

    monkeypatch.setattr(executor, "_candidate", candidate)
    monkeypatch.setattr(executor, "_report_candidates", lambda *_args: None)

    executor._ensure_assignment(step, torch.zeros(1, dtype=torch.bool))

    assert calls == ["left_arm"]
    assert executor._assignments[step.id] == ["left_arm"]


def test_standalone_handover_candidate_stops_before_coordinated_transfer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entities = {
        "can": _FakeEntity("can", _pose(0.0, 0.2, 0.75), _box_vertices(0.03)),
        "table": _FakeEntity("table", _pose(0.0, 0.0, 0.5), _box_vertices(0.5)),
    }
    task = _handover_then_place_task()
    task["level"] = "L1"
    task["task_instances"] = [task["task_instances"][0]]
    task["success"] = {"type": "handover_complete"}
    env = _FakeEnv(entities)
    program = load_execution_program(instantiate_seed_graph(task, {"can": "can"}))
    executor = ProgramExecutor(program, env, record_runtime=False)
    step = program.semantic_steps[0]
    planned_actions: list[str] = []

    def ground(
        action: dict[str, Any],
        _step: SemanticStep,
        *,
        arm: str,
        state: ExecutionState,
        reference_eef_pose: torch.Tensor | None = None,
        orientation_reference_pose: torch.Tensor | None = None,
    ) -> GroundedAction:
        del state, reference_eef_pose, orientation_reference_pose
        return GroundedAction(
            action_class=str(action["atomic_action_class"]),
            arm=arm,
            control=str(action["control"]),
            target=SimpleNamespace(),
            cfg={},
        )

    def plan(grounded: GroundedAction, state: ExecutionState) -> ActionOutcome:
        planned_actions.append(grounded.action_class)
        return ActionOutcome(
            trajectory=torch.zeros(1, 1, env.robot.dof),
            success=torch.tensor([True]),
            next_state=state,
            grounded=grounded,
        )

    monkeypatch.setattr(executor.grounder, "ground", ground)
    monkeypatch.setattr(executor, "_with_downstream_targets", lambda *args: args[-1])
    monkeypatch.setattr(executor.adapter, "plan", plan)

    candidate = executor._candidate(
        step,
        "left_arm",
        torch.zeros(1, dtype=torch.bool),
    )

    assert bool(candidate.feasible[0])
    assert planned_actions == ["PickUp", "MoveHeldObject"]
    assert set(candidate.plans) == set(step.edge_ids[:2])


def test_failed_handover_keeps_transfer_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entities = {
        "can": _FakeEntity("can", _pose(0.0, 0.2, 0.75), _box_vertices(0.03)),
        "table": _FakeEntity("table", _pose(0.0, 0.0, 0.5), _box_vertices(0.5)),
    }
    env = _FakeEnv(entities)
    task = _handover_then_place_task()
    task["level"] = "L1"
    task["task_instances"] = [task["task_instances"][0]]
    task["success"] = {"type": "handover_complete"}
    program = load_execution_program(
        instantiate_seed_graph(
            task,
            {"can": "can"},
        )
    )
    step = program.semantic_steps[0]
    edge = next(
        candidate
        for candidate in program.edges
        if candidate.actions[0]["atomic_action_class"] == "HandOver"
    )
    state = _held_state(env, entities["can"], arm="left_arm")
    executor = ProgramExecutor(program, env, record_runtime=False)
    executor._assignments[step.id] = ["coordinated"]
    executor._object_owners["can"] = ["left_arm"]
    executor._arm_owners["left_arm"] = ["can"]
    executor._object_states[("can", "left_arm")] = state
    grounded = GroundedAction(
        action_class="HandOver",
        arm="coordinated",
        control="coordinated",
        target=SimpleNamespace(),
        cfg={},
    )
    failed_outcome = ActionOutcome(
        trajectory=torch.zeros(1, 1, env.robot.dof),
        success=torch.tensor([False]),
        next_state=state,
        grounded=grounded,
    )
    monkeypatch.setattr(
        executor.grounder,
        "ground",
        lambda *_args, **_kwargs: grounded,
    )
    monkeypatch.setattr(
        executor.adapter, "plan", lambda *_args, **_kwargs: failed_outcome
    )
    monkeypatch.setattr(
        executor.adapter,
        "execute_trajectory",
        lambda *_args, **_kwargs: [],
    )

    result = executor._execute_coordinated(edge, step, torch.tensor([False]))

    assert bool(result.failed[0])
    assert executor._object_owners["can"] == ["left_arm"]
    assert executor._arm_owners["left_arm"] == ["can"]
    assert executor._arm_owners["right_arm"] == [None]
    assert ("can", "left_arm") in executor._object_states
    assert ("can", "right_arm") not in executor._object_states


def test_handover_commits_receiver_ownership_only_after_physical_verification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entities = {
        "can": _FakeEntity("can", _pose(0.0, 0.2, 0.75), _box_vertices(0.03)),
        "table": _FakeEntity("table", _pose(0.0, 0.0, 0.5), _box_vertices(0.5)),
    }
    env = _FakeEnv(entities)
    task = _handover_then_place_task()
    task["level"] = "L1"
    task["task_instances"] = [task["task_instances"][0]]
    task["success"] = {"type": "handover_complete"}
    program = load_execution_program(instantiate_seed_graph(task, {"can": "can"}))
    step = program.semantic_steps[0]
    edge = next(
        candidate
        for candidate in program.edges
        if candidate.actions[0]["atomic_action_class"] == "HandOver"
    )
    transfer_state = _held_state(env, entities["can"], arm="left_arm")
    receiver_state = _held_state(env, entities["can"], arm="right_arm")
    executor = ProgramExecutor(program, env, record_runtime=False)
    executor._assignments[step.id] = ["coordinated"]
    executor._object_owners["can"] = ["left_arm"]
    executor._arm_owners["left_arm"] = ["can"]
    executor._object_states[("can", "left_arm")] = transfer_state
    grounded = GroundedAction(
        action_class="HandOver",
        arm="coordinated",
        control="coordinated",
        target=SimpleNamespace(),
        cfg={},
        motion_policy={"held_position_tolerance": 0.03},
    )
    successful_outcome = ActionOutcome(
        trajectory=torch.zeros(1, 1, env.robot.dof),
        success=torch.tensor([True]),
        next_state=receiver_state,
        grounded=grounded,
    )
    observed_poses: list[torch.Tensor] = []

    def ground_candidates(*_args, **_kwargs):
        observed_poses.append(entities["can"].get_local_pose(to_matrix=True))
        return (grounded,)

    monkeypatch.setattr(
        executor.grounder,
        "ground_candidates",
        ground_candidates,
    )
    monkeypatch.setattr(
        executor.adapter, "plan", lambda *_args, **_kwargs: successful_outcome
    )
    monkeypatch.setattr(
        executor.adapter,
        "execute_trajectory",
        lambda *_args, **_kwargs: [],
    )

    entities["can"]._pose[:, 0, 3] += 0.30
    result = executor._execute_coordinated(edge, step, torch.tensor([False]))

    assert observed_poses[0][0, 0, 3] == pytest.approx(0.30)
    assert result.planner_traces[0]["execution_replanned_from_live_state"] is True
    assert bool(result.failed[0])
    assert executor._object_owners["can"] == [None]
    assert executor._arm_owners["left_arm"] == [None]
    assert executor._arm_owners["right_arm"] == [None]
    assert ("can", "right_arm") not in executor._object_states


def test_handover_defers_clearance_verification_to_retreat_action() -> None:
    adapter = AtomicActionAdapter(_FakeEnv())

    assert adapter.capabilities.get("HandOver").verifier_hook is None
    assert adapter.capabilities.get("MoveEndEffector").verifier_hook is not None


def test_axis_align_then_handover_reacquires_with_a_separate_transfer_policy() -> None:
    entity = _FakeEntity("can", _pose(0.0, 0.2, 0.75), _rect_vertices(0.10, 0.03, 0.03))
    env = _FakeEnv(
        {
            "can": entity,
            "table": _FakeEntity("table", _pose(0.0, 0.0, 0.5), _box_vertices(0.5)),
        }
    )
    task = {
        "schema_version": TASK_SPEC_SCHEMA,
        "task_id": "orient_then_handover",
        "level": "L3",
        "instruction": "Orient the can, then hand it over.",
        "reasoning_type": "none",
        "task_instances": [
            {
                "id": "task_01",
                "task_type": "E2",
                "params": {
                    "object_role": "can",
                    "required_arm": "right_arm",
                    "orientation_goal": "upright",
                    "support_role": "table",
                },
                "depends_on": [],
                "role": "primary",
            },
            {
                "id": "task_02",
                "task_type": "E4",
                "params": {
                    "object_role": "can",
                    "transfer_arm": "right_arm",
                    "receive_arm": "left_arm",
                },
                "depends_on": ["task_01"],
                "role": "primary",
            },
        ],
        "success": {"type": "handover_complete", "task_instance_id": "task_02"},
        "oracle": {},
        "metadata": {},
    }
    program = load_execution_program(instantiate_seed_graph(task, {"can": "can"}))
    orient_step = next(
        candidate for candidate in program.semantic_steps if candidate.id == "task_01"
    )
    handover_step = next(
        candidate for candidate in program.semantic_steps if candidate.id == "task_02"
    )
    orient_edge = next(
        candidate
        for candidate in program.edges
        if candidate.id in orient_step.edge_ids
        and candidate.actions[0]["atomic_action_class"] == "AxisAlign"
    )
    orient_lift_edge = next(
        candidate
        for candidate in program.edges
        if candidate.id in orient_step.edge_ids
        and candidate.actions[0]["target_binding"].get("operation") == "lift_clear"
    )
    orient_retreat_edge = next(
        candidate
        for candidate in program.edges
        if candidate.id in orient_step.edge_ids
        and candidate.actions[0]["target_binding"].get("operation")
        == "retreat_after_lift"
    )
    handover_edge = next(
        candidate
        for candidate in program.edges
        if candidate.id in handover_step.edge_ids
        if candidate.actions[0]["atomic_action_class"] == "PickUp"
    )
    vertices = _rect_vertices(0.10, 0.03, 0.03)
    semantics = ObjectSemantics(
        affordance=AntipodalAffordance(
            object_label="can",
            mesh_vertices=vertices,
        ),
        geometry={"mesh_vertices": vertices},
        label="can",
        entity=entity,
    )
    grounder = ActionGrounder(program, env, lambda _uid: semantics)

    orient_alignment = grounder.ground(
        orient_edge.actions[0],
        orient_step,
        arm="right_arm",
        state=ExecutionState(last_qpos=env.robot.get_qpos()),
    )
    release_pose = _pose(0.05, 0.2, 0.78)
    orient_lift = grounder.ground(
        orient_lift_edge.actions[0],
        orient_step,
        arm="right_arm",
        state=ExecutionState(last_qpos=env.robot.get_qpos()),
        reference_eef_pose=release_pose,
    )
    orient_retreat = grounder.ground(
        orient_retreat_edge.actions[0],
        orient_step,
        arm="right_arm",
        state=ExecutionState(last_qpos=env.robot.get_qpos()),
        reference_eef_pose=orient_lift.target.xpos,
    )
    handover_pickup = grounder.ground(
        handover_edge.actions[0],
        handover_step,
        arm="right_arm",
        state=ExecutionState(last_qpos=env.robot.get_qpos()),
    )

    assert "approach_direction_mode" not in orient_alignment.cfg
    assert "approach_direction_mode" not in handover_pickup.cfg
    assert handover_pickup.cfg["pick_object_part"] == "top"
    assert isinstance(orient_alignment.target, AxisAlignGoal)
    assert isinstance(orient_lift.target, EndEffectorPoseGoal)
    assert torch.equal(
        orient_lift.target.xpos[:, :2, 3],
        release_pose[:, :2, 3],
    )
    assert bool((orient_lift.target.xpos[:, 2, 3] > release_pose[:, 2, 3]).all())
    assert "retreat_reachability_search" not in orient_lift.motion_policy
    assert isinstance(orient_retreat.target, EndEffectorPoseGoal)
    retreat_distance = torch.linalg.vector_norm(
        orient_retreat.target.xpos[:, :2, 3] - orient_lift.target.xpos[:, :2, 3],
        dim=1,
    )
    torch.testing.assert_close(
        retreat_distance, torch.full_like(retreat_distance, 0.20)
    )
    assert bool(
        (orient_retreat.target.xpos[:, 0, 3] > orient_lift.target.xpos[:, 0, 3]).all()
    )
    assert torch.equal(
        orient_retreat.target.xpos[:, 2, 3],
        orient_lift.target.xpos[:, 2, 3],
    )
    assert bool(
        (
            orient_retreat.target.xpos[:, :2, 3] != orient_lift.target.xpos[:, :2, 3]
        ).any()
    )
    assert orient_alignment.target.grasp_xpos is None
    assert torch.equal(
        orient_alignment.target.object_target_pose,
        orient_alignment.target_object_pose,
    )
    assert orient_alignment.target_object_pose is not None
    assert isinstance(
        orient_alignment.target.semantics.affordance,
        AxisAlignAffordance,
    )
    assert torch.equal(
        orient_alignment.target.semantics.affordance.internal_axis,
        torch.tensor([0.0, 0.0, 1.0]),
    )
    assert orient_alignment.cfg["target_axis"] == (0.0, 0.0, 1.0)
    assert handover_pickup.target.grasp_xpos is None
    assert isinstance(
        handover_pickup.target.semantics.affordance,
        AntipodalAffordance,
    )
    assert handover_pickup.target.semantics.affordance is semantics.affordance


def test_pour_grounding_targets_receiver_without_physical_contents() -> None:
    task = {
        "schema_version": TASK_SPEC_SCHEMA,
        "task_id": "pour_grounding",
        "level": "L1",
        "instruction": "Pour the ball from the cup into the bin.",
        "reasoning_type": "none",
        "task_instances": [
            {
                "id": "task_01",
                "task_type": "E3",
                "params": {
                    "source_role": "cup",
                    "target_role": "bin",
                    "required_arm": "right_arm",
                },
                "depends_on": [],
                "role": "primary",
            }
        ],
        "success": {"type": "poured"},
        "oracle": {},
        "metadata": {},
    }
    bindings = {"cup": "source", "bin": "target"}
    program = load_execution_program(instantiate_seed_graph(task, bindings))
    step = program.semantic_steps[0]
    edges = {edge.actions[0]["atomic_action_class"]: edge for edge in program.edges}
    staging_edge = next(
        edge
        for edge in program.edges
        if edge.actions[0]["atomic_action_class"] == "MoveHeldObject"
        and edge.actions[0]["target_binding"].get("phase") == "final"
    )
    source = _FakeEntity("source", _pose(-0.2, 0.0, 0.7), _box_vertices(0.05))
    target = _FakeEntity("target", _pose(0.2, 0.0, 0.7), _box_vertices(0.10))
    env = _FakeEnv({"source": source, "target": target})
    env.agent_initial_object_poses = {
        "source": source.get_local_pose(to_matrix=True),
        "target": target.get_local_pose(to_matrix=True),
    }
    semantics = ObjectSemantics(
        affordance=AntipodalAffordance(
            object_label="source",
            mesh_vertices=_box_vertices(0.05),
        ),
        geometry={},
        label="source",
        entity=source,
    )
    grounder = ActionGrounder(program, env, lambda _uid: semantics)
    state = ExecutionState(last_qpos=env.robot.get_qpos())

    pickup = grounder.ground(
        edges["PickUp"].actions[0], step, arm="right_arm", state=state
    )
    staging = grounder.ground(
        staging_edge.actions[0],
        step,
        arm="right_arm",
        state=state,
    )
    pouring = grounder.ground(
        edges["Pour"].actions[0], step, arm="right_arm", state=state
    )

    assert isinstance(pickup.target.semantics.affordance, AxisAlignAffordance)
    assert torch.allclose(
        pickup.target.semantics.affordance.internal_axis,
        torch.tensor([0.0, 1.0, 0.0]),
    )
    assert staging.target_object_pose is not None
    assert staging.target_object_pose[0, 0, 3] == pytest.approx(0.2)
    assert staging.target_object_pose[0, 2, 3] > 0.8
    assert isinstance(pouring.target, PourGoal)
    assert pouring.cfg["rotate_angle"] == pytest.approx(torch.pi / 2.0)

    assert step.postcondition["verification"] == "action_completion"


@pytest.mark.parametrize(
    ("task_type", "initial_qpos", "direction", "target_state"),
    (("E6", 0.0, "pull", "open"), ("E7", 0.2, "push", "closed")),
)
def test_articulation_grounding_reuses_slide_and_observes_joint_state(
    task_type: str,
    initial_qpos: float,
    direction: str,
    target_state: str,
) -> None:
    task, _ = make_task_spec(task_type)
    graph = instantiate_seed_graph(task, {"object_01": "drawer"})
    program = load_execution_program(graph)
    articulation = _FakeArticulation("drawer", initial_qpos)
    env = _FakeEnv(articulations={"drawer": articulation})
    grounder = ActionGrounder(
        program,
        env,
        lambda _uid: ObjectSemantics(affordance=Affordance(), geometry={}),
    )
    step = program.semantic_steps[0]
    edge = program.edges[0]

    grounded = grounder.ground(
        edge.actions[0],
        step,
        arm="right_arm",
        state=ExecutionState(last_qpos=env.robot.get_qpos()),
    )

    assert isinstance(grounded.target, SlideGoal)
    assert grounded.target.grasp_xpos is not None
    assert isinstance(grounded.target.semantics.affordance, SlideAffordance)
    assert grounded.cfg["direction"] == direction
    assert grounded.cfg["translation_distance"] == pytest.approx(0.2)
    assert grounded.cfg["articulation_joint_name"] == "slide_joint"
    assert grounded.cfg["articulation_initial_qpos"].item() == pytest.approx(
        initial_qpos
    )
    assert torch.allclose(
        grounded.target.semantics.affordance.translation_axis,
        torch.tensor([-1.0, 0.0, 0.0]),
    )
    assert edge.actions[0]["failure_policy"] == "task_required"

    predicate = {
        "type": "articulation_joint_near",
        "object": "drawer",
        "target_state": target_state,
    }
    assert not bool(evaluate_predicate(env, predicate)[0])
    articulation._qpos[0, 0] = 0.19 if target_state == "open" else 0.01
    assert bool(evaluate_predicate(env, predicate)[0])


def test_turn_knob_requires_setting_map_and_reuses_twist() -> None:
    task = {
        "schema_version": TASK_SPEC_SCHEMA,
        "task_id": "turn_knob",
        "level": "L1",
        "instruction": "Turn the knob to setting two.",
        "reasoning_type": "none",
        "task_instances": [
            {
                "id": "task_01",
                "task_type": "E8",
                "params": {
                    "object_role": "knob",
                    "target_setting": 2,
                    "required_arm": "right_arm",
                },
                "depends_on": [],
                "role": "primary",
            }
        ],
        "success": {"type": "articulation_joint_near"},
        "oracle": {},
        "metadata": {},
    }
    program = load_execution_program(instantiate_seed_graph(task, {"knob": "dial"}))
    articulation = _FakeArticulation("dial", 0.0)
    articulation._joint_info.joint_type = SimpleNamespace(name="REVOLUTE")
    articulation._limits = torch.tensor([[[-1.0, 1.0]]])
    env = _FakeEnv(articulations={"dial": articulation})
    env.agent_config = {
        "articulation_settings": {"dial": {"slide_joint": [-1.0, 0.0, 1.0]}}
    }
    grounder = ActionGrounder(
        program,
        env,
        lambda _uid: ObjectSemantics(affordance=Affordance(), geometry={}),
    )
    step = program.semantic_steps[0]

    grounded = grounder.ground(
        program.edges[0].actions[0],
        step,
        arm="right_arm",
        state=ExecutionState(last_qpos=env.robot.get_qpos()),
    )

    assert isinstance(grounded.target, TwistGoal)
    assert isinstance(grounded.target.semantics.affordance, TwistAffordance)
    assert grounded.cfg["twist_angle"] == pytest.approx(1.0)
    assert grounded.cfg["articulation_joint_name"] == "slide_joint"
    assert grounded.cfg["articulation_initial_qpos"].item() == pytest.approx(0.0)
    assert grounded.target.semantics.affordance.grasp_position == pytest.approx(
        (0.04, 0.0, 0.0)
    )
    predicate = {
        **step.postcondition,
        "joint_name": "slide_joint",
        "target_qpos": 1.0,
    }
    assert not bool(evaluate_predicate(env, predicate)[0])
    articulation._qpos[0, 0] = 0.99
    assert bool(evaluate_predicate(env, predicate)[0])

    env.agent_config = {"articulation_settings": {}}
    with pytest.raises(ValueError, match="explicit setting_values"):
        ActionGrounder(
            program,
            env,
            lambda _uid: ObjectSemantics(affordance=Affordance(), geometry={}),
        ).ground(
            program.edges[0].actions[0],
            program.semantic_steps[0],
            arm="right_arm",
            state=ExecutionState(last_qpos=env.robot.get_qpos()),
        )


def test_handover_clearance_verifier_checks_distance_and_transfer_side() -> None:
    entities = {
        "can": _FakeEntity("can", _pose(0.0, -0.2, 1.0), _box_vertices(0.03)),
        "target": _FakeEntity("target", _pose(0.2, 0.0, 0.75), _box_vertices(0.03)),
        "table": _FakeEntity("table", _pose(0.0, 0.0, 0.5), _box_vertices(0.5)),
    }
    env = _FakeEnv(entities)
    program = load_execution_program(
        instantiate_seed_graph(
            _handover_then_place_task(),
            {"can": "can", "target": "target"},
        )
    )
    executor = ProgramExecutor(program, env, record_runtime=False)
    hook = executor.adapter.capabilities.get("MoveEndEffector").verifier_hook
    assert hook is not None
    _, lateral = robot_frame_axes(env)
    grounded = GroundedAction(
        action_class="MoveEndEffector",
        arm="left_arm",
        control="arm",
        target=SimpleNamespace(),
        cfg={},
        motion_policy={
            "clearance_object_uid": "can",
            "transfer_arm": "left_arm",
            "transfer_role_axis": torch.cat(
                (lateral, lateral.new_zeros((1, 1))), dim=1
            ),
            "minimum_transfer_clearance": 0.10,
            "minimum_transfer_lateral_clearance": 0.06,
        },
    )
    outcome = SimpleNamespace(grounded=grounded)
    attempted = torch.tensor([True])

    assert not bool(
        hook(
            executor=executor,
            step=program.semantic_steps[0],
            arm="left_arm",
            outcome=outcome,
            attempted=attempted,
        )[0]
    )

    clear_left = _pose(0.0, -0.4, 1.0)
    env.get_current_xpos_agent = lambda: (clear_left, _pose(0.0, 0.2, 1.0))
    assert bool(
        hook(
            executor=executor,
            step=program.semantic_steps[0],
            arm="left_arm",
            outcome=outcome,
            attempted=attempted,
        )[0]
    )


@pytest.mark.parametrize("arm", ["left_arm", "right_arm"])
def test_handover_source_policy_uses_pickup_default_top_down_approach(
    arm: str,
) -> None:
    action = GroundedAction(
        action_class="PickUp",
        arm=arm,
        control="arm",
        target=SimpleNamespace(),
        cfg={"pick_object_part": "top"},
    )

    cfg = AtomicActionAdapter(_FakeEnv())._build_config(action, PickUpOptions)

    assert cfg.pick_object_part == "top"
    torch.testing.assert_close(
        cfg.approach_direction,
        torch.tensor([0.0, 0.0, -1.0]),
    )


@pytest.mark.parametrize(
    ("transfer_arm", "expected_world_y"),
    [("left_arm", -1.0), ("right_arm", 1.0)],
)
def test_handover_receiver_uses_the_mirrored_diagonal_approach(
    transfer_arm: str,
    expected_world_y: float,
) -> None:
    env = _FakeEnv()
    action = GroundedAction(
        action_class="HandOver",
        arm="coordinated",
        control="coordinated",
        target=SimpleNamespace(),
        cfg={
            "transfer_arm": transfer_arm,
            "middle_object_pose": torch.eye(4).unsqueeze(0),
            "final_object_pose": torch.eye(4).unsqueeze(0),
        },
    )

    cfg = AtomicActionAdapter(env)._build_config(action, HeldObjectHandOverOptions)

    diagonal = 2.0**-0.5
    assert cfg.receive_approach_direction[0] == pytest.approx(0.0)
    assert cfg.receive_approach_direction[1] == pytest.approx(
        expected_world_y * diagonal
    )
    assert cfg.receive_approach_direction[2] == pytest.approx(-diagonal)
    _, lateral = robot_frame_axes(env)
    receive_side = "right_arm" if transfer_arm == "left_arm" else "left_arm"
    receiver_outward = lateral[0] if receive_side == "left_arm" else -lateral[0]
    pre_grasp_offset = -cfg.receive_approach_direction[:2] * cfg.pre_grasp_distance
    assert torch.dot(pre_grasp_offset, receiver_outward) > 0.0


@pytest.mark.parametrize(
    ("transfer_arm", "expected_x"),
    [("left_arm", 1.0), ("right_arm", -1.0)],
)
def test_handover_receiver_approach_tracks_rotated_robot_lateral_axis(
    monkeypatch: pytest.MonkeyPatch,
    transfer_arm: str,
    expected_x: float,
) -> None:
    env = _FakeEnv()

    def get_link_pose(*, link_name: str, to_matrix: bool) -> torch.Tensor:
        assert to_matrix
        pose = torch.eye(4).unsqueeze(0)
        pose[:, 0, 3] = 0.3 if link_name == "physical_left_base" else -0.3
        return pose

    monkeypatch.setattr(env.robot, "get_link_pose", get_link_pose)
    action = GroundedAction(
        action_class="HandOver",
        arm="coordinated",
        control="coordinated",
        target=SimpleNamespace(),
        cfg={
            "transfer_arm": transfer_arm,
            "middle_object_pose": torch.eye(4).unsqueeze(0),
            "final_object_pose": torch.eye(4).unsqueeze(0),
        },
    )

    cfg = AtomicActionAdapter(env)._build_config(action, HeldObjectHandOverOptions)

    diagonal = 2.0**-0.5
    assert cfg.receive_approach_direction[0] == pytest.approx(expected_x * diagonal)
    assert cfg.receive_approach_direction[1] == pytest.approx(0.0)
    assert cfg.receive_approach_direction[2] == pytest.approx(-diagonal)


@pytest.mark.parametrize(
    ("arm", "outward_x"),
    [("left_arm", 1.0), ("right_arm", -1.0)],
)
def test_legacy_handover_transfer_mode_tracks_a_rotated_live_base_line(
    monkeypatch: pytest.MonkeyPatch,
    arm: str,
    outward_x: float,
) -> None:
    env = _FakeEnv()

    def get_link_pose(*, link_name: str, to_matrix: bool) -> torch.Tensor:
        assert to_matrix
        pose = torch.eye(4).unsqueeze(0)
        pose[:, 0, 3] = 0.3 if link_name == "physical_left_base" else -0.3
        return pose

    monkeypatch.setattr(env.robot, "get_link_pose", get_link_pose)
    action = GroundedAction(
        action_class="PickUp",
        arm=arm,
        control="arm",
        target=SimpleNamespace(),
        cfg={"approach_direction_mode": "handover_transfer"},
    )

    cfg = AtomicActionAdapter(env)._build_config(action, PickUpOptions)

    outward = torch.tensor([outward_x, 0.0])
    assert torch.dot(cfg.approach_direction[:2], outward) < 0.0
    assert cfg.approach_direction[2] < 0.0


def test_pickup_is_replanned_from_live_pose_and_screens_downstream_targets(
    monkeypatch: Any,
) -> None:
    entity = _FakeEntity("can", _pose(0.0, 0.2, 0.75), _box_vertices(0.03))
    env = _FakeEnv({"can": entity})
    executor = ProgramExecutor(
        load_execution_program(
            compile_task_agent(_task_agent(_hold_step("hold", "can", "left_arm")))
        ),
        env,
        record_runtime=False,
    )
    plan_calls: list[GroundedAction] = []

    def ground(
        action,
        step,
        *,
        arm,
        state,
        reference_eef_pose=None,
        orientation_reference_pose=None,
    ):
        del reference_eef_pose, orientation_reference_pose
        action_class = action["atomic_action_class"]
        target_pose = (
            _pose(0.0, 0.2, 0.85) if action_class == "MoveHeldObject" else None
        )
        return GroundedAction(
            action_class=action_class,
            arm=arm,
            control=str(action.get("control", "arm")),
            target=SimpleNamespace(xpos=None),
            cfg={"planned_object_pose": entity.get_local_pose(to_matrix=True)},
            target_object_pose=target_pose,
        )

    def plan(grounded: GroundedAction, state: ExecutionState) -> ActionOutcome:
        plan_calls.append(grounded)
        next_state = (
            _held_state(env, entity) if grounded.action_class == "PickUp" else state
        )
        return ActionOutcome(
            trajectory=torch.zeros(1, 1, env.robot.dof),
            success=torch.tensor([True]),
            next_state=next_state,
            grounded=grounded,
        )

    monkeypatch.setattr(executor.grounder, "ground", ground)
    monkeypatch.setattr(executor.adapter, "plan", plan)
    monkeypatch.setattr(executor.adapter, "execute_trajectory", lambda *a, **k: [])
    step = executor.program.semantic_steps[0]
    failed = torch.tensor([False])

    executor._ensure_assignment(step, failed)
    planned_call_count = len(plan_calls)
    executor._candidate_cache.clear()
    entity._pose[:, 0, 3] += 0.25
    edge_result = executor._execute_edge(
        executor.edges[step.edge_ids[0]], step, failed=failed
    )

    assert len(plan_calls) == planned_call_count + 1
    assert planned_call_count == len(step.edge_ids)
    assert len(plan_calls[0].cfg["downstream_object_target_poses"]) == 1
    assert plan_calls[-1].cfg["planned_object_pose"][0, 0, 3] == pytest.approx(0.25)
    assert plan_calls[-1].cfg["downstream_object_target_poses"]
    assert edge_result.planner_traces[0]["execution_replanned_from_live_state"]
    assert not edge_result.planner_traces[0]["speculative_candidate_available"]
    assert bool(edge_result.failed[0])
    assert executor._object_owners["can"] == [None]


def test_completed_step_releases_speculative_candidate_plans() -> None:
    executor = object.__new__(ProgramExecutor)
    executor._candidate_cache = {
        ("completed", "left_arm"): object(),
        ("next", "right_arm"): object(),
    }
    executor._candidate_failures = {
        ("completed", "left_arm"): "failed",
        ("next", "right_arm"): "pending",
    }
    executor._candidate_diagnostics = {
        "completed": {"large": "trace"},
        "next": {"small": "trace"},
    }
    executor._candidate_blockers = {
        "completed": ({"reason": "old"},),
        "next": ({"reason": "new"},),
    }
    executor._reported_candidates = {"completed", "next"}

    executor._release_candidate_plans("completed")

    assert set(executor._candidate_cache) == {("next", "right_arm")}
    assert set(executor._candidate_failures) == {("next", "right_arm")}
    assert set(executor._candidate_diagnostics) == {"next"}
    assert set(executor._candidate_blockers) == {"next"}
    assert executor._reported_candidates == {"next"}


def test_live_pickup_planning_exception_is_a_retryable_edge_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entity = _FakeEntity("can", _pose(0.0, 0.2, 0.75), _box_vertices(0.03))
    executor = ProgramExecutor(
        load_execution_program(
            compile_task_agent(_task_agent(_hold_step("hold", "can", "left_arm")))
        ),
        _FakeEnv({"can": entity}),
        record_runtime=False,
    )
    step = executor.program.semantic_steps[0]
    edge = executor.edges[step.edge_ids[0]]
    executor._assignments[step.id] = ["left_arm"]
    monkeypatch.setattr(
        executor.grounder,
        "ground",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("no IK")),
    )

    result = executor._execute_edge(edge, step, failed=torch.tensor([False]))

    assert bool(result.failed[0])
    assert result.actions == []
    assert result.planner_traces[0]["primary_strategy"] == "live_pickup_replan"
    assert result.planner_traces[0]["exception"] == "RuntimeError: no IK"


def test_pickup_candidate_screens_handover_successor_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later handover staging pose participates in pickup grasp screening."""
    entity = _FakeEntity("can", _pose(0.0, 0.2, 0.75), _box_vertices(0.03))
    env = _FakeEnv({"can": entity})
    executor = ProgramExecutor(
        load_execution_program(
            compile_task_agent(_task_agent(_hold_step("pickup", "can", "left_arm")))
        ),
        env,
        record_runtime=False,
    )
    pickup_step = executor.program.semantic_steps[0]
    handover_step = SemanticStep(
        id="handover",
        parent_step_id="handover",
        operator="handover",
        object_uid="can",
        actor={"mode": "required", "arm": "left_arm"},
        goal={"transfer_arm": "left_arm", "receive_arm": "right_arm"},
        depends_on=(pickup_step.id,),
        postcondition={},
        edge_ids=("handover_staging",),
    )
    handover_edge = ExecutionEdge(
        id="handover_staging",
        source="pickup_done",
        target="handover_done",
        actions=(
            {
                "atomic_action_class": "MoveHeldObject",
                "actor": {"mode": "required", "arm": "left_arm"},
                "control": "arm",
                "target_binding": {
                    "kind": "handover_staging",
                    "transfer_arm": "left_arm",
                    "receive_arm": "right_arm",
                },
                "motion_policy": motion_policy(),
            },
        ),
    )
    executor.steps[handover_step.id] = handover_step
    executor.edges[handover_edge.id] = handover_edge

    existing_target = _pose(0.0, 0.2, 0.85)
    handover_target = _pose(0.0, 0.0, 1.15)
    grounded = GroundedAction(
        action_class="PickUp",
        arm="left_arm",
        control="arm",
        target=SimpleNamespace(),
        cfg={"downstream_object_target_poses": (existing_target,)},
    )

    def ground(
        _action: Any,
        candidate: SemanticStep,
        *,
        arm: str,
        state: ExecutionState,
        reference_eef_pose: torch.Tensor | None = None,
        orientation_reference_pose: torch.Tensor | None = None,
    ) -> GroundedAction:
        del arm, state, reference_eef_pose, orientation_reference_pose
        target = handover_target if candidate.id == handover_step.id else None
        return GroundedAction(
            action_class="MoveHeldObject",
            arm="left_arm",
            control="arm",
            target=SimpleNamespace(),
            cfg={},
            target_object_pose=target,
        )

    monkeypatch.setattr(executor.grounder, "ground", ground)
    result = executor._with_downstream_targets(
        pickup_step,
        pickup_step.edge_ids[0],
        "left_arm",
        ExecutionState(last_qpos=env.robot.get_qpos()),
        grounded,
    )

    targets = result.cfg["downstream_object_target_poses"]
    assert len(targets) == 2
    assert torch.equal(targets[0], existing_target)
    assert torch.equal(targets[1], handover_target)


def test_object_held_predicate_checks_live_gripper_and_tcp_geometry() -> None:
    entity = _FakeEntity("can", _pose(0.0, 0.2, 0.75), _box_vertices(0.03))
    env = _FakeEnv({"can": entity})
    env.robot._qpos[:, env.left_eef_joints] = env.close_state
    state = _held_state(env, entity)
    left_eef, _ = env.get_current_xpos_agent()
    env.get_current_xpos_agent = lambda: (left_eef, None)

    held = evaluate_predicate(
        env,
        {"type": "object_held", "object": "can"},
        held_owners={"can": ["left_arm"]},
        held_states={("can", "left_arm"): state},
    )

    assert bool(held[0])
    env.robot._qpos[:, env.left_eef_joints] = env.open_state
    assert not bool(
        evaluate_predicate(
            env,
            {"type": "object_held", "object": "can"},
            held_owners={"can": ["left_arm"]},
            held_states={("can", "left_arm"): state},
        )[0]
    )
    env.robot._qpos[:, env.left_eef_joints] = (env.open_state + env.close_state) / 2
    assert bool(
        evaluate_predicate(
            env,
            {"type": "object_held", "object": "can"},
            held_owners={"can": ["left_arm"]},
            held_states={("can", "left_arm"): state},
        )[0]
    )
    env.close_state = torch.tensor([0.0, 0.0])
    env.open_state = torch.tensor([0.04, 0.04])
    env.robot._qpos[:, env.left_eef_joints] = env.open_state
    assert not bool(
        evaluate_predicate(
            env,
            {"type": "object_held", "object": "can"},
            held_owners={"can": ["left_arm"]},
            held_states={("can", "left_arm"): state},
        )[0]
    )


def test_coordinated_held_predicate_uses_per_arm_held_relations() -> None:
    entity = _FakeEntity("tray", _pose(0.0, 0.0, 0.75), _box_vertices(0.2))
    env = _FakeEnv({"tray": entity})
    env.robot._qpos[:, env.left_eef_joints] = env.close_state
    env.robot._qpos[:, env.right_eef_joints] = env.close_state
    state = _coordinated_held_state(env, entity)
    predicate = {"type": "object_held_by_both_grippers", "object": "tray"}

    assert bool(evaluate_predicate(env, predicate, coordinated_state=state)[0])

    env.robot._qpos[:, env.right_eef_joints] = env.open_state
    assert not bool(evaluate_predicate(env, predicate, coordinated_state=state)[0])


def test_both_grippers_open_uses_the_live_reset_posture() -> None:
    env = _FakeEnv()
    physical_open = torch.tensor([[0.20, 0.45]])
    env.left_arm_init_gripper_state = physical_open.clone()
    env.right_arm_init_gripper_state = physical_open.clone()
    env.robot._qpos[:, env.left_eef_joints] = physical_open
    env.robot._qpos[:, env.right_eef_joints] = physical_open

    opened = evaluate_predicate(
        env,
        {"type": "both_grippers_open", "tolerance": 0.08},
    )
    assert opened.tolist() == [True]

    env.robot._qpos[:, env.right_eef_joints] += 0.20
    opened = evaluate_predicate(
        env,
        {"type": "both_grippers_open", "tolerance": 0.08},
    )
    assert opened.tolist() == [False]


def test_object_supported_by_requires_overlap_and_vertical_contact() -> None:
    support_z = 0.75
    payload_z = support_z + 0.05 + 0.02 + 0.005
    payload = _FakeEntity(
        "payload",
        _pose(0.002, -0.002, payload_z),
        _box_vertices(0.02),
    )
    support = _FakeEntity(
        "support",
        _pose(0.0, 0.0, support_z),
        _box_vertices(0.05),
    )
    env = _FakeEnv({"payload": payload, "support": support})
    predicate = {
        "type": "object_supported_by",
        "object": "payload",
        "support": "support",
    }

    assert bool(evaluate_predicate(env, predicate)[0])

    payload._pose = _pose(0.002, -0.002, payload_z - 1.0)
    assert not bool(evaluate_predicate(env, predicate)[0])

    payload._pose = _pose(0.002, -0.002, payload_z + 1.0)
    assert not bool(evaluate_predicate(env, predicate)[0])

    payload._pose = _pose(0.081, 0.0, payload_z)
    assert not bool(evaluate_predicate(env, predicate)[0])


def test_object_supported_by_uses_local_not_mesh_wide_support_height() -> None:
    support_vertices = torch.tensor(
        [
            [-0.10, -0.10, -0.05],
            [-0.02, -0.02, 0.05],
            [0.02, -0.02, 0.05],
            [0.02, 0.02, 0.05],
            [-0.02, 0.02, 0.05],
            [0.40, 0.00, 0.40],
        ],
        dtype=torch.float32,
    )
    payload = _FakeEntity(
        "payload",
        _pose(0.0, 0.0, 0.075),
        _box_vertices(0.02),
    )
    support = _FakeEntity("support", _pose(0.0, 0.0, 0.0), support_vertices)
    env = _FakeEnv({"payload": payload, "support": support})

    supported = evaluate_predicate(
        env,
        {
            "type": "object_supported_by",
            "object": "payload",
            "support": "support",
        },
    )

    assert bool(supported[0])


def test_poured_predicate_requires_observed_contents_inside_target() -> None:
    source = _FakeEntity("source", _pose(-0.2, 0.0, 0.7), _box_vertices(0.08))
    target = _FakeEntity("target", _pose(0.2, 0.0, 0.7), _box_vertices(0.12))
    content = _FakeEntity("content", _pose(0.2, 0.0, 0.7), _box_vertices(0.01))
    env = _FakeEnv({"source": source, "target": target, "content": content})
    predicate = {
        "type": "poured",
        "object": "source",
        "reference_object": "target",
        "contents": [{"object": "content"}],
    }

    assert bool(evaluate_predicate(env, predicate)[0])

    content._pose = _pose(-0.2, 0.0, 0.7)
    assert not bool(evaluate_predicate(env, predicate)[0])

    with pytest.raises(ValueError, match="independently observable"):
        evaluate_predicate(env, {**predicate, "contents": []})


def test_object_supported_by_uses_live_center_of_mass_projection() -> None:
    payload = _FakeEntity(
        "payload",
        _pose(0.04, 0.0, 0.125),
        _rect_vertices(0.08, 0.02, 0.02),
    )
    payload.body_data = SimpleNamespace(
        com_pose=torch.tensor([[0.04, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    )
    support = _FakeEntity(
        "support",
        _pose(0.0, 0.0, 0.05),
        _rect_vertices(0.05, 0.05, 0.05),
    )
    env = _FakeEnv({"payload": payload, "support": support})

    supported = evaluate_predicate(
        env,
        {
            "type": "object_supported_by",
            "object": "payload",
            "support": "support",
        },
    )

    assert not bool(supported[0])


def test_physical_pickup_rebases_a_compliant_grasp_from_live_pose() -> None:
    entity = _FakeEntity("can", _pose(0.0, 0.2, 0.75), _box_vertices(0.03))
    env = _FakeEnv({"can": entity})
    env.robot._qpos[:, env.left_eef_joints] = env.close_state
    state = _held_state(env, entity)
    entity._pose[:, 0, 3] += 0.055
    executor = ProgramExecutor(
        load_execution_program(
            compile_task_agent(_task_agent(_hold_step("hold", "can", "left_arm")))
        ),
        env,
        record_runtime=False,
    )

    physical = executor._physical_pickup(
        "can",
        "left_arm",
        state,
        torch.tensor([True]),
    )
    state = executor._rebase_held_state(
        "can",
        "left_arm",
        state,
        physical,
        from_planned_qpos=False,
    )

    assert bool(physical[0])
    left_eef, _ = env.get_current_xpos_agent()
    rebased_eef = torch.bmm(
        entity.get_local_pose(to_matrix=True),
        state.get_held_object("physical_left_arm").object_to_eef,
    )
    assert torch.allclose(rebased_eef, left_eef)


def test_physical_hold_accepts_configured_held_position_tolerance() -> None:
    entity = _FakeEntity("can", _pose(0.0, 0.2, 0.75), _box_vertices(0.03))
    env = _FakeEnv({"can": entity})
    env.robot._qpos[:, env.left_eef_joints] = env.close_state
    state = _held_state(env, entity)
    executor = ProgramExecutor(
        load_execution_program(
            compile_task_agent(_task_agent(_hold_step("hold", "can", "left_arm")))
        ),
        env,
        record_runtime=False,
    )
    executor._object_owners["can"] = ["left_arm"]
    entity._pose[:, 0, 3] += 0.055

    held = executor._physical_hold(
        "can",
        "left_arm",
        state,
        torch.tensor([True]),
    )

    assert bool(held[0])


def test_physical_pickup_rejects_large_grasp_slip() -> None:
    entity = _FakeEntity("can", _pose(0.0, 0.2, 0.75), _box_vertices(0.03))
    env = _FakeEnv({"can": entity})
    env.robot._qpos[:, env.left_eef_joints] = env.close_state
    state = _held_state(env, entity)
    entity._pose[:, 0, 3] += 0.08
    executor = ProgramExecutor(
        load_execution_program(
            compile_task_agent(_task_agent(_hold_step("hold", "can", "left_arm")))
        ),
        env,
        record_runtime=False,
    )

    physical = executor._physical_pickup(
        "can",
        "left_arm",
        state,
        torch.tensor([True]),
    )

    assert not bool(physical[0])


def test_physical_pickup_rejects_offset_even_when_object_was_lifted() -> None:
    entity = _FakeEntity("can", _pose(0.0, 0.2, 0.75), _box_vertices(0.03))
    env = _FakeEnv({"can": entity})
    env.robot._qpos[:, env.left_eef_joints] = env.close_state
    state = _held_state(env, entity)
    entity._pose[:, 0, 3] += 0.08
    entity._pose[:, 2, 3] += 0.08
    executor = ProgramExecutor(
        load_execution_program(
            compile_task_agent(_task_agent(_hold_step("hold", "can", "left_arm")))
        ),
        env,
        record_runtime=False,
    )

    physical = executor._physical_pickup(
        "can",
        "left_arm",
        state,
        torch.tensor([True]),
    )

    assert not bool(physical[0])


def test_physical_hold_detects_loss_and_releases_runtime_ownership() -> None:
    entity = _FakeEntity("can", _pose(0.0, 0.2, 0.75), _box_vertices(0.03))
    env = _FakeEnv({"can": entity})
    env.robot._qpos[:, env.left_eef_joints] = env.close_state
    state = _held_state(env, entity)
    executor = ProgramExecutor(
        load_execution_program(
            compile_task_agent(_task_agent(_hold_step("hold", "can", "left_arm")))
        ),
        env,
        record_runtime=False,
    )
    executor._object_owners["can"] = ["left_arm"]
    executor._arm_owners["left_arm"] = ["can"]
    executor._object_states[("can", "left_arm")] = state
    entity._pose[:, 0, 3] += 0.08

    held = executor._physical_hold(
        "can",
        "left_arm",
        state,
        torch.tensor([True]),
    )
    executor._release_ownership("can", "left_arm", ~held)

    assert not bool(held[0])
    assert executor._object_owners["can"] == [None]
    assert executor._arm_owners["left_arm"] == [None]
    assert ("can", "left_arm") not in executor._object_states


def test_rebase_held_state_uses_fk_qpos_not_stale_eef_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entity = _FakeEntity("can", _pose(0.0, 0.2, 0.75), _box_vertices(0.03))
    env = _FakeEnv({"can": entity})
    state = _held_state(env, entity, arm="left_arm")
    expected_eef = _pose(0.31, -0.17, 1.06)

    def compute_fk(
        qpos: torch.Tensor,
        *,
        name: str,
        to_matrix: bool,
    ) -> torch.Tensor:
        del name
        assert to_matrix
        return expected_eef.repeat(qpos.shape[0], 1, 1)

    monkeypatch.setattr(env.robot, "compute_fk", compute_fk)
    executor = ProgramExecutor(
        load_execution_program(
            compile_task_agent(_task_agent(_hold_step("hold", "can", "left_arm")))
        ),
        env,
        record_runtime=False,
    )

    state = executor._rebase_held_state(
        "can",
        "left_arm",
        state,
        torch.tensor([True]),
    )
    held = state.get_held_object("physical_left_arm")
    assert held is not None
    expected_relation = torch.bmm(
        torch.linalg.inv(entity.get_local_pose(to_matrix=True)),
        expected_eef,
    )
    assert torch.allclose(held.object_to_eef, expected_relation)


def test_upright_transport_state_tracks_selected_target_pose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entity = _FakeEntity("can", _pose(0.0, 0.2, 0.75), _box_vertices(0.03))
    env = _FakeEnv({"can": entity})
    state = _held_state(env, entity, arm="left_arm")
    expected_eef = _pose(0.27, -0.11, 1.04)
    target_pose = _pose(0.05, 0.01, 0.92)

    def compute_fk(
        qpos: torch.Tensor,
        *,
        name: str,
        to_matrix: bool,
    ) -> torch.Tensor:
        del name
        assert to_matrix
        return expected_eef.repeat(qpos.shape[0], 1, 1)

    monkeypatch.setattr(env.robot, "compute_fk", compute_fk)
    entity._pose = target_pose.clone()
    executor = ProgramExecutor(
        load_execution_program(
            compile_task_agent(_task_agent(_hold_step("hold", "can", "left_arm")))
        ),
        env,
        record_runtime=False,
    )
    synchronized = executor._rebase_held_state(
        "can",
        "left_arm",
        state,
        torch.tensor([True]),
    )

    held = synchronized.get_held_object("physical_left_arm")
    assert held is not None
    assert torch.allclose(
        held.object_to_eef,
        torch.bmm(torch.linalg.inv(target_pose), expected_eef),
    )
    assert torch.allclose(held.grasp_xpos, expected_eef)


def test_existing_object_owner_reserves_same_arm(monkeypatch: Any) -> None:
    entity = _FakeEntity("can", _pose(0.0, 0.2, 0.75), _box_vertices(0.03))
    env = _FakeEnv({"can": entity})
    executor = ProgramExecutor(
        load_execution_program(
            compile_task_agent(_task_agent(_hold_step("hold", "can", "left_arm")))
        ),
        env,
        record_runtime=False,
    )
    original = executor.program.semantic_steps[0]
    continuation = replace(
        original,
        id="continuation",
        actor={"mode": "auto"},
    )
    executor._object_owners["can"] = ["left_arm"]
    executor._arm_owners["left_arm"] = ["can"]
    executor._object_states[("can", "left_arm")] = _held_state(env, entity)

    monkeypatch.setattr(
        executor,
        "_candidate",
        lambda step, arm, failed: SimpleNamespace(
            feasible=torch.tensor([True]),
            cost=torch.tensor([0.0 if arm == "right_arm" else 10.0]),
            warnings=(),
        ),
    )
    executor._ensure_assignment(continuation, torch.tensor([False]))

    assert executor._assignments["continuation"] == ["left_arm"]
    assert bool(executor._resource_conflicts(continuation, "right_arm")[0])


def test_new_task_group_hydrates_predecessor_held_state() -> None:
    entity = _FakeEntity("can", _pose(0.0, 0.2, 0.75), _box_vertices(0.03))
    env = _FakeEnv({"can": entity})
    executor = ProgramExecutor(
        load_execution_program(
            compile_task_agent(_task_agent(_hold_step("hold", "can", "right_arm")))
        ),
        env,
        record_runtime=False,
    )
    held_state = _held_state(env, entity, arm="right_arm")
    executor._object_states[("can", "right_arm")] = held_state
    continuation = replace(
        executor.program.semantic_steps[0],
        id="place_after_handover",
        actor={"mode": "required", "arm": "right_arm"},
    )

    hydrated = executor._state_for(continuation, "right_arm")

    assert hydrated.get_held_object("physical_right_arm") is not None
    assert torch.equal(hydrated.last_qpos, env.robot.get_qpos())


def test_place_uses_preceding_or_live_eef_pose_not_original_grasp() -> None:
    entities = {
        "can": _FakeEntity("can", _pose(0.0, 0.2, 0.75), _box_vertices(0.03)),
        "basket": _FakeEntity("basket", _pose(0.0, 0.0, 0.70), _box_vertices(0.10)),
    }
    env = _FakeEnv(entities)
    program = load_execution_program(
        compile_task_agent(
            _task_agent(
                {
                    "id": "place",
                    "operator": "place_relative",
                    "object": "can",
                    "actor": {"mode": "required", "arm": "left_arm"},
                    "goal": {
                        "reference_object": "basket",
                        "relation": "inside",
                    },
                    "depends_on": [],
                }
            )
        )
    )
    state = _held_state(env, entities["can"])
    held_object = state.get_held_object("physical_left_arm")
    assert held_object is not None
    replacement = HeldObjectState(
        semantics=held_object.semantics,
        object_to_eef=held_object.object_to_eef,
        grasp_xpos=_pose(0.0, -0.3, 0.75),
        env_mask=held_object.env_mask,
    )
    held_objects = dict(state.held_objects)
    held_objects["physical_left_arm"] = replacement
    state = state.with_updates(held_objects=held_objects)
    held_object = replacement
    edge = next(
        edge
        for edge in program.edges
        if edge.actions[0]["atomic_action_class"] == "Place"
    )
    grounder = ActionGrounder(
        program,
        env,
        lambda uid: held_object.semantics,
    )
    reference = _pose(0.0, 0.4, 0.85)

    planned = grounder.ground(
        edge.actions[0],
        program.semantic_steps[0],
        arm="left_arm",
        state=state,
        reference_eef_pose=reference,
    )
    live = grounder.ground(
        edge.actions[0],
        program.semantic_steps[0],
        arm="left_arm",
        state=state,
    )

    assert torch.equal(planned.target.xpos, reference)
    assert torch.equal(live.target.xpos, env.get_current_xpos_agent()[0])
    assert not torch.equal(live.target.xpos, held_object.grasp_xpos)


def test_inside_target_preserves_pre_pick_supported_height() -> None:
    entities = {
        "can": _FakeEntity("can", _pose(0.0, 0.2, 1.40), _box_vertices(0.03)),
        "basket": _FakeEntity("basket", _pose(0.0, 0.0, 0.70), _box_vertices(0.10)),
    }
    env = _FakeEnv(entities)
    program = load_execution_program(
        compile_task_agent(
            _task_agent(
                {
                    "id": "place",
                    "operator": "place_relative",
                    "object": "can",
                    "actor": {"mode": "required", "arm": "left_arm"},
                    "goal": {
                        "reference_object": "basket",
                        "relation": "inside",
                    },
                    "depends_on": [],
                }
            )
        )
    )
    step = program.semantic_steps[0]
    final = next(
        edge
        for edge in program.edges
        if edge.actions[0]["atomic_action_class"] == "MoveHeldObject"
    )
    state = _held_state(env, entities["can"])
    held = state.get_held_object("physical_left_arm")
    assert held is not None
    grounder = ActionGrounder(program, env, lambda _uid: held.semantics)
    supported_pose = _pose(0.0, 0.2, 0.75)

    grounded = grounder.ground(
        final.actions[0],
        step,
        arm="left_arm",
        state=state,
        orientation_reference_pose=supported_pose,
    )

    assert grounded.target_object_pose is not None
    assert grounded.target_object_pose[0, 2, 3] == pytest.approx(0.75)


def test_coordinated_step_rejects_an_arm_reserved_by_terminal_hold() -> None:
    entity = _FakeEntity("shared_box", _pose(0.0, 0.0, 0.75), _box_vertices(0.05))
    executor = ProgramExecutor(
        load_execution_program(
            compile_task_agent(
                _task_agent(
                    {
                        "id": "transport",
                        "operator": "coordinated_transport",
                        "object": "shared_box",
                        "actor": {
                            "mode": "coordinated",
                            "arms": ["left_arm", "right_arm"],
                        },
                        "goal": {
                            "direction": "front",
                            "terminal_behavior": "hold",
                        },
                        "depends_on": [],
                    }
                )
            )
        ),
        _FakeEnv({"shared_box": entity}),
        record_runtime=False,
    )
    executor._arm_owners["left_arm"] = ["held_can"]
    step = executor.program.semantic_steps[0]

    executor._ensure_assignment(step, torch.tensor([False]))

    assert executor._assignments[step.id] == [None]


def test_failure_propagates_only_to_dependent_branch(monkeypatch: Any) -> None:
    program = load_execution_program(
        compile_task_agent(
            _task_agent(
                _hold_step("left", "can_a", "left_arm"),
                _hold_step("right", "can_b", "right_arm"),
            )
        )
    )
    executor = ProgramExecutor(
        program, _FakeEnv(), settle_steps=0, record_runtime=False
    )
    monkeypatch.setattr(
        executor,
        "_pack_ready_edges",
        lambda ready, **_kwargs: (ready[0],),
    )
    monkeypatch.setattr(
        executor,
        "_ensure_assignment",
        lambda step, failed: executor._assignments.setdefault(
            step.id, [step.actor["arm"]]
        ),
    )
    active_by_step: dict[str, list[bool]] = {"left": [], "right": []}

    def execute(edge, step, *, failed):
        active_by_step[step.id].append(not bool(failed[0]))
        action_failed = failed.clone()
        if step.id == "left" and edge.id == step.edge_ids[0]:
            action_failed[:] = True
        return SimpleNamespace(actions=[], failed=action_failed, grounded=[])

    monkeypatch.setattr(executor, "_execute_edge", execute)
    monkeypatch.setattr(
        executor,
        "_verify_step",
        lambda step, failed: (
            failed,
            ~failed,
            torch.zeros(1, 3),
        ),
    )

    result = executor.run()

    assert any(active_by_step["right"])
    assert bool(result.semantic_success["right"][0])
    assert not bool(result.success[0])


def _run_dependent_failure_policy_chain(
    monkeypatch: pytest.MonkeyPatch,
    *,
    failure_policy: str,
    record_root: Path | None = None,
) -> tuple[ExecutionResult, dict[str, list[list[bool]]]]:
    first = _hold_step("first", "can_a", "left_arm")
    second = _hold_step("second", "can_b", "right_arm")
    second["depends_on"] = ["first"]
    env = _FakeEnv()
    env.num_envs = 2
    env.robot = _FakeRobot(env.num_envs)
    executor = ProgramExecutor(
        load_execution_program(compile_task_agent(_task_agent(first, second))),
        env,
        settle_steps=0,
        record_runtime=record_root is not None,
        record_root=record_root,
        failure_policy=failure_policy,
    )
    monkeypatch.setattr(
        executor,
        "_ensure_assignment",
        lambda step, failed: executor._assignments.setdefault(
            step.id,
            [
                None if bool(failed[env_id]) else str(step.actor["arm"])
                for env_id in range(env.num_envs)
            ],
        ),
    )
    active_by_step: dict[str, list[list[bool]]] = {"first": [], "second": []}

    def execute(
        edge: ExecutionEdge,
        step: SemanticStep,
        *,
        failed: torch.Tensor,
    ) -> _EdgeResult:
        active = ~failed
        active_by_step[step.id].append(active.tolist())
        result_failed = failed.clone()
        if step.id == "first" and edge.id == step.edge_ids[0]:
            result_failed[1] = True
        return _EdgeResult(
            actions=[],
            failed=result_failed,
            grounded=[],
            executed=active,
        )

    monkeypatch.setattr(executor, "_execute_edge", execute)
    monkeypatch.setattr(
        executor,
        "_step_runtime_metadata",
        lambda _step: [{} for _ in range(env.num_envs)],
    )
    monkeypatch.setattr(
        executor,
        "_verify_step",
        lambda _step, failed: (
            failed,
            ~failed,
            torch.zeros(env.num_envs, 3),
        ),
    )
    return executor.run(run_id=failure_policy), active_by_step


def test_stop_failure_policy_blocks_only_failed_environment_downstream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, active_by_step = _run_dependent_failure_policy_chain(
        monkeypatch,
        failure_policy="stop",
    )

    assert all(active == [True, False] for active in active_by_step["second"])
    assert result.semantic_success["second"].tolist() == [True, False]
    assert result.success.tolist() == [True, False]


def test_continue_failure_policy_executes_downstream_without_clearing_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, active_by_step = _run_dependent_failure_policy_chain(
        monkeypatch,
        failure_policy="continue",
    )

    assert all(active == [True, True] for active in active_by_step["second"])
    assert result.semantic_success["first"].tolist() == [True, False]
    assert result.semantic_success["second"].tolist() == [True, True]
    assert result.success.tolist() == [True, False]


def test_continue_failure_policy_records_failed_then_executed_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    visualization = ModuleType("embodichain.gen_sim.action_engine.graph_visualization")
    visualization.render_task_graph_png = lambda _document: b"\x89PNG\r\n\x1a\n"
    monkeypatch.setitem(sys.modules, visualization.__name__, visualization)

    result, _ = _run_dependent_failure_policy_chain(
        monkeypatch,
        failure_policy="continue",
        record_root=tmp_path,
    )

    env_dir = Path(result.record_dir) / "env_0001"
    checkpoints = {
        document["semantic_step"]["id"]: document
        for path in env_dir.joinpath("checkpoints").glob("*.json")
        for document in [json.loads(path.read_text(encoding="utf-8"))]
    }
    task_graph = json.loads(
        env_dir.joinpath("task_graph.json").read_text(encoding="utf-8")
    )

    assert checkpoints["first"]["status"] == "failed"
    assert checkpoints["second"]["status"] == "success"
    assert checkpoints["first"]["failure_policy"] == "continue"
    assert any(
        event["event"] == "edge"
        and event["semantic_step_id"] == "second"
        and event["status"] == "executed"
        for event in task_graph["runtime"]["events"]
    )
    assert task_graph["runtime"]["failure_policy"] == "continue"
    assert task_graph["runtime"]["status"] == "failed"


def test_resource_ordering_waits_without_propagating_semantic_failure() -> None:
    task = {
        "schema_version": TASK_SPEC_SCHEMA,
        "task_id": "resource_ordering",
        "level": "L3",
        "instruction": "Stand both cans, then hand over the second can.",
        "reasoning_type": "none",
        "task_instances": [
            {
                "id": "task_01",
                "task_type": "E2",
                "params": {
                    "object_role": "first",
                    "required_arm": "right_arm",
                },
                "depends_on": [],
                "role": "primary",
            },
            {
                "id": "task_02",
                "task_type": "E2",
                "params": {
                    "object_role": "second",
                    "required_arm": "left_arm",
                },
                "depends_on": [],
                "role": "primary",
            },
            {
                "id": "task_03",
                "task_type": "E4",
                "params": {
                    "object_role": "second",
                    "transfer_arm": "left_arm",
                    "receive_arm": "right_arm",
                },
                "depends_on": ["task_02"],
                "role": "primary",
            },
        ],
        "success": {"type": "all_complete"},
        "oracle": {},
        "metadata": {},
    }
    program = load_execution_program(
        instantiate_seed_graph(
            task,
            {"first": "first_can", "second": "second_can"},
        )
    )
    executor = ProgramExecutor(program, _FakeEnv(), record_runtime=False)
    handover_entry = next(
        edge
        for edge in program.edges
        if executor.step_by_edge[edge.id].id == "task_03"
        and all(
            executor.step_by_edge[dependency].id != "task_03"
            for dependency in edge.depends_on
        )
    )
    dependencies = {
        executor.step_by_edge[dependency].id: dependency
        for dependency in handover_entry.depends_on
    }
    failures = {
        dependency: torch.tensor([step_id == "task_01"])
        for step_id, dependency in dependencies.items()
    }

    assert not bool(executor._dependency_failures(handover_entry, failures)[0])

    failures[dependencies["task_02"]][:] = True
    assert bool(executor._dependency_failures(handover_entry, failures)[0])


def test_v2_executor_retries_one_complete_atomic_action_twice(
    monkeypatch: Any,
) -> None:
    task, requirements = make_task_spec("E9")
    bindings = {
        item["role_id"]: f"uid_{item['role_id']}" for item in requirements["objects"]
    }
    program = load_execution_program(instantiate_seed_graph(task, bindings))
    executor = ProgramExecutor(
        program,
        _FakeEnv(),
        settle_steps=0,
        record_runtime=False,
    )
    monkeypatch.setattr(
        executor,
        "_ensure_assignment",
        lambda step, failed: executor._assignments.setdefault(step.id, ["left_arm"]),
    )
    attempts = 0

    def execute(_edge, _step, *, failed):
        nonlocal attempts
        attempts += 1
        action_failed = failed.clone()
        if attempts < 3:
            action_failed[:] = True
        return SimpleNamespace(actions=[], failed=action_failed, grounded=[])

    monkeypatch.setattr(executor, "_execute_edge", execute)
    monkeypatch.setattr(
        executor,
        "_verify_step",
        lambda step, failed: (failed, ~failed, torch.zeros(1, 3)),
    )

    result = executor.run()

    assert attempts == 3
    assert result.retry_count == 2
    assert bool(result.success[0])
    assert result.failure_events == []


def test_v2_executor_stops_at_transition_budget() -> None:
    task, requirements = make_task_spec("E1")
    bindings = {
        item["role_id"]: f"uid_{item['role_id']}" for item in requirements["objects"]
    }
    program = load_execution_program(instantiate_seed_graph(task, bindings))
    executor = ProgramExecutor(
        program,
        _FakeEnv(),
        max_transitions=0,
        settle_steps=0,
        record_runtime=False,
    )

    with pytest.raises(RuntimeError, match="max_transitions"):
        executor.run()


def test_failed_arrangement_records_candidate_diagnostics_without_marker_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entities = {
        "table": _FakeEntity(
            "table",
            _pose(0.0, 0.0, 0.70),
            _rect_vertices(0.60, 0.40, 0.02),
        ),
        "can_a": _FakeEntity(
            "can_a",
            _pose(-0.15, 0.0, 0.76),
            _rect_vertices(0.03, 0.03, 0.06),
        ),
        "can_b": _FakeEntity(
            "can_b",
            _pose(0.15, 0.0, 0.76),
            _rect_vertices(0.03, 0.03, 0.06),
        ),
    }
    program = load_execution_program(
        compile_task_agent(
            _task_agent(
                {
                    "id": "line",
                    "operator": "arrange_line",
                    "objects": ["can_a", "can_b"],
                    "actor": {"mode": "auto"},
                    "goal": {
                        "axis": "world_x",
                        "order_constraint": "free",
                    },
                    "depends_on": [],
                }
            )
        )
    )
    executor = ProgramExecutor(
        program,
        _FakeEnv(entities),
        settle_steps=0,
        record_root=tmp_path,
    )
    infeasible = SimpleNamespace(
        feasible=torch.tensor([False]),
        cost=torch.tensor([torch.inf]),
        plans={},
        warnings=("No IK solutions found for downstream target poses.",),
    )
    monkeypatch.setattr(executor, "_candidate", lambda *_args, **_kwargs: infeasible)

    result = executor.run(run_id="no_candidate")

    assert not bool(result.success[0])
    record_path = Path(result.record_dir) / "env_0000" / "task_graph.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["runtime"]["status"] == "failed"
    first_event = record["runtime"]["events"][0]
    assert first_event["status"] == "failed"
    assert first_event["diagnostics"] == [
        "No IK solutions found for downstream target poses."
    ]


def test_arrange_line_builds_live_slots_for_compiler_operator_name() -> None:
    entities = {
        "table": _FakeEntity(
            "table",
            _pose(0.0, 0.0, 0.70),
            _rect_vertices(0.60, 0.40, 0.02),
        ),
        "can_a": _FakeEntity(
            "can_a",
            _pose(-0.15, 0.0, 0.76),
            _rect_vertices(0.03, 0.03, 0.06),
        ),
        "can_b": _FakeEntity(
            "can_b",
            _pose(0.15, 0.0, 0.76),
            _rect_vertices(0.03, 0.03, 0.06),
        ),
    }
    compiled = compile_task_agent(
        _task_agent(
            {
                "id": "line",
                "operator": "arrange_line",
                "objects": ["can_a", "can_b"],
                "actor": {"mode": "auto"},
                "goal": {
                    "axis": "world_x",
                    "order_constraint": "free",
                },
                "depends_on": [],
            }
        )
    )

    executor = ProgramExecutor(
        load_execution_program(compiled),
        _FakeEnv(entities),
        record_runtime=False,
    )

    assert executor.arrangement is not None
    assert executor.arrangement.positions.shape == (1, 2, 3)
    assert {step.operator for step in executor.program.semantic_steps} == {
        "arrange_line"
    }


def test_free_arrangement_matches_live_object_order_without_crossing() -> None:
    entities = {
        "table": _FakeEntity(
            "table",
            _pose(0.0, 0.0, 0.70),
            _rect_vertices(0.60, 0.40, 0.02),
        ),
        "can_a": _FakeEntity(
            "can_a",
            _pose(0.0, 0.20, 0.78),
            _rect_vertices(0.03, 0.03, 0.06),
        ),
        "can_b": _FakeEntity(
            "can_b",
            _pose(0.0, 0.00, 0.78),
            _rect_vertices(0.03, 0.03, 0.06),
        ),
        "can_c": _FakeEntity(
            "can_c",
            _pose(0.0, -0.20, 0.78),
            _rect_vertices(0.03, 0.03, 0.06),
        ),
    }
    executor = ProgramExecutor(
        load_execution_program(
            compile_task_agent(
                _task_agent(
                    {
                        "id": "line",
                        "operator": "arrange_line",
                        "objects": ["can_a", "can_b", "can_c"],
                        "actor": {"mode": "auto"},
                        "goal": {
                            "axis": "world_y",
                            "order_constraint": "free",
                        },
                        "depends_on": [],
                    }
                )
            )
        ),
        _FakeEnv(entities),
        record_runtime=False,
    )
    arrangement = executor.arrangement

    assert arrangement is not None
    assert int(arrangement.assignments["line__01"][0]) == 2
    assert int(arrangement.assignments["line__02"][0]) == 1
    assert int(arrangement.assignments["line__03"][0]) == 0
    assert arrangement.spacing[0] == pytest.approx(0.1648528)


def test_arm_candidate_score_softly_penalizes_cross_zone_motion() -> None:
    source = _pose(0.0, -0.30, 0.78)
    target = _pose(0.0, -0.20, 0.78)
    kwargs = {
        "motion_cost": torch.tensor([torch.pi]),
        "source_pose": source,
        "target_pose": target,
        "workspace_center_xy": torch.tensor([[0.0, 0.0]]),
        "workspace_half_width": torch.tensor([0.40]),
        "robot_lateral_axis": torch.tensor([[0.0, -1.0]]),
        "policy": default_runtime_policy("dual_ur10").arm_selection,
    }

    left = _score_arm_candidate(arm="left_arm", **kwargs)
    right = _score_arm_candidate(arm="right_arm", **kwargs)

    assert left["normalized_motion_cost"][0] == pytest.approx(1.0)
    assert left["pickup_crossing_penalty"][0] == pytest.approx(0.0)
    assert left["placement_crossing_penalty"][0] == pytest.approx(0.0)
    assert right["pickup_crossing_penalty"][0] > 0.0
    assert right["placement_crossing_penalty"][0] > 0.0
    assert right["total_cost"][0] > left["total_cost"][0]


def test_preserve_grounding_uses_pre_pickup_orientation_reference() -> None:
    entities = {
        "table": _FakeEntity(
            "table",
            _pose(0.0, 0.0, 0.70),
            _rect_vertices(0.60, 0.40, 0.02),
        ),
        "can_a": _FakeEntity(
            "can_a",
            _pose(0.0, -0.10, 0.78),
            _rect_vertices(0.03, 0.03, 0.06),
        ),
        "can_b": _FakeEntity(
            "can_b",
            _pose(0.0, 0.10, 0.78),
            _rect_vertices(0.03, 0.03, 0.06),
        ),
    }
    executor = ProgramExecutor(
        load_execution_program(
            compile_task_agent(
                _task_agent(
                    {
                        "id": "line",
                        "operator": "arrange_line",
                        "objects": ["can_a", "can_b"],
                        "actor": {"mode": "auto"},
                        "goal": {
                            "axis": "world_y",
                            "order_constraint": "free",
                            "orientation_goal": "preserve",
                        },
                        "depends_on": [],
                    }
                )
            )
        ),
        _FakeEnv(entities),
        record_runtime=False,
    )
    step = executor.program.semantic_steps[0]
    final_edge = next(
        edge
        for edge in executor.program.edges
        if edge.id in step.edge_ids
        and edge.actions[0]["atomic_action_class"] == "MoveHeldObject"
        and edge.actions[0]["target_binding"].get("phase") == "final"
    )
    reference = entities[step.object_uid].get_local_pose(to_matrix=True)
    disturbed = reference.clone()
    disturbed[:, :3, :3] = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    )
    entities[step.object_uid]._pose = disturbed

    grounded = executor.grounder.ground(
        final_edge.actions[0],
        step,
        arm="left_arm",
        state=ExecutionState(last_qpos=executor.env.robot.get_qpos()),
        orientation_reference_pose=reference,
    )

    assert grounded.target_object_pose is not None
    assert torch.allclose(
        grounded.target_object_pose[:, :3, :3],
        reference[:, :3, :3],
    )


def test_arrange_line_verifies_planar_slot_without_height_coupling() -> None:
    entities = {
        "table": _FakeEntity(
            "table",
            _pose(0.0, 0.0, 0.70),
            _rect_vertices(0.60, 0.40, 0.02),
        ),
        "can_a": _FakeEntity(
            "can_a",
            _pose(-0.055, -0.190, 0.755),
            _rect_vertices(0.03, 0.03, 0.06),
        ),
        "can_b": _FakeEntity(
            "can_b",
            _pose(0.15, 0.0, 0.76),
            _rect_vertices(0.03, 0.03, 0.06),
        ),
    }
    executor = ProgramExecutor(
        load_execution_program(
            compile_task_agent(
                _task_agent(
                    {
                        "id": "line",
                        "operator": "arrange_line",
                        "objects": ["can_a", "can_b"],
                        "actor": {"mode": "auto"},
                        "goal": {
                            "axis": "world_y",
                            "order_constraint": "free",
                        },
                        "depends_on": [],
                    }
                )
            )
        ),
        _FakeEnv(entities),
        settle_steps=0,
        record_runtime=False,
    )
    step = executor.program.semantic_steps[0]
    executor._targets[step.id] = torch.tensor([[0.0, -0.216, 0.842]])
    executor._policies[step.id] = {
        "line_axis_tolerance": 0.06,
        "line_perpendicular_tolerance": 0.06,
    }

    failed, success, _ = executor._verify_step(step, torch.tensor([False]))

    assert not bool(failed[0])
    assert bool(success[0])


def test_arrange_line_rejects_preserve_orientation_drift() -> None:
    rotated = _pose(0.0, -0.190, 0.755)
    rotated[:, :3, :3] = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    )
    entities = {
        "table": _FakeEntity(
            "table",
            _pose(0.0, 0.0, 0.70),
            _rect_vertices(0.60, 0.40, 0.02),
        ),
        "can_a": _FakeEntity(
            "can_a",
            rotated,
            _rect_vertices(0.03, 0.03, 0.06),
        ),
        "can_b": _FakeEntity(
            "can_b",
            _pose(0.15, 0.0, 0.76),
            _rect_vertices(0.03, 0.03, 0.06),
        ),
    }
    executor = ProgramExecutor(
        load_execution_program(
            compile_task_agent(
                _task_agent(
                    {
                        "id": "line",
                        "operator": "arrange_line",
                        "objects": ["can_a", "can_b"],
                        "actor": {"mode": "auto"},
                        "goal": {
                            "axis": "world_y",
                            "order_constraint": "free",
                            "orientation_goal": "preserve",
                        },
                        "depends_on": [],
                    }
                )
            )
        ),
        _FakeEnv(entities),
        settle_steps=0,
        record_runtime=False,
    )
    step = executor.program.semantic_steps[0]
    executor._targets[step.id] = rotated[:, :3, 3].clone()
    executor._orientation_references[step.id] = _pose(0.0, -0.190, 0.755)
    executor._policies[step.id] = {
        "line_axis_tolerance": 0.06,
        "line_perpendicular_tolerance": 0.06,
        "preserve_orientation_tolerance": torch.pi / 12,
    }

    failed, success, _ = executor._verify_step(step, torch.tensor([False]))

    assert bool(failed[0])
    assert not bool(success[0])


def test_shared_container_placements_receive_non_overlapping_live_slots() -> None:
    entities = {
        "basket": _FakeEntity(
            "basket",
            _pose(0.0, 0.0, 0.72),
            _rect_vertices(0.25, 0.18, 0.08),
        ),
        "cube": _FakeEntity(
            "cube",
            _pose(-0.15, 0.0, 0.76),
            _rect_vertices(0.03, 0.03, 0.03),
        ),
        "cup": _FakeEntity(
            "cup",
            _pose(0.15, 0.0, 0.76),
            _rect_vertices(0.035, 0.035, 0.06),
        ),
    }
    steps = [
        {
            "id": f"place_{uid}",
            "operator": "place_relative",
            "object": uid,
            "actor": {"mode": "auto"},
            "goal": {"relation": "inside", "reference_object": "basket"},
            "depends_on": [],
        }
        for uid in ("cube", "cup")
    ]
    task = {
        "schema_version": TASK_AGENT_SCHEMA,
        "task": "two_in_basket",
        "goal": "Place both objects in the basket.",
        "semantic_steps": steps,
    }
    executor = ProgramExecutor(
        load_execution_program(compile_task_agent(task)),
        _FakeEnv(entities),
        record_runtime=False,
    )
    targets = [
        executor.placements[step.id].positions[step.id][0, :2]
        for step in executor.program.semantic_steps
    ]

    assert set(executor.placements) == {"place_cube", "place_cup"}
    assert torch.linalg.vector_norm(targets[0] - targets[1]) > 0.05


@pytest.mark.parametrize(
    ("direction", "expected_position"),
    (
        ("front_left", (0.16, 0.16, 0.85)),
        ("up", (0.0, 0.0, 0.91)),
    ),
)
def test_coordinated_transport_direction_is_grounded_from_live_pose(
    direction: str,
    expected_position: tuple[float, float, float],
) -> None:
    entities = {
        "shared_box": _FakeEntity(
            "shared_box",
            _pose(0.0, 0.0, 0.75),
            _rect_vertices(0.10, 0.06, 0.03),
        )
    }
    env = _FakeEnv(entities)
    env.agent_initial_object_poses = {"shared_box": _pose(9.0, 9.0, 9.0)}
    program = load_execution_program(
        compile_task_agent(
            _task_agent(
                {
                    "id": "transport",
                    "operator": "coordinated_transport",
                    "object": "shared_box",
                    "actor": {
                        "mode": "coordinated",
                        "arms": ["left_arm", "right_arm"],
                    },
                    "goal": {
                        "direction": direction,
                        "terminal_behavior": "hold",
                    },
                    "depends_on": [],
                }
            )
        )
    )

    def semantics(uid: str) -> ObjectSemantics:
        entity = entities[uid]
        return ObjectSemantics(
            affordance=Affordance(),
            geometry={
                "mesh_vertices": entity.get_vertices(env_ids=[0], scale=True),
                "mesh_triangles": entity.get_triangles(env_ids=[0]),
            },
            label=uid,
            entity=entity,
        )

    step = program.semantic_steps[0]
    edge = program.edges[0]
    grounded = ActionGrounder(program, env, semantics).ground(
        edge.actions[0],
        step,
        arm="coordinated",
        state=ExecutionState(last_qpos=env.robot.get_qpos()),
    )

    assert isinstance(grounded.target, CoordinatedPickGoal)
    assert torch.allclose(
        grounded.target.object_initial_pose,
        entities["shared_box"].get_local_pose(to_matrix=True),
    )
    assert not torch.allclose(
        grounded.target.object_initial_pose,
        env.agent_initial_object_poses["shared_box"],
    )
    assert torch.allclose(
        grounded.target.object_target_pose[0, :3, 3],
        torch.tensor(expected_position),
    )


def test_coordinated_payload_monitor_rejects_drift_and_carrier_tilt() -> None:
    class _BatchedVerticesEntity(_FakeEntity):
        def get_vertices(
            self,
            *,
            env_ids: list[int],
            scale: bool,
        ) -> torch.Tensor:
            return super().get_vertices(env_ids=env_ids, scale=scale).unsqueeze(0)

    entities = {
        "tray": _BatchedVerticesEntity(
            "tray",
            _pose(0.0, 0.0, 0.75),
            _rect_vertices(0.20, 0.14, 0.02),
        ),
        "bottle": _FakeEntity(
            "bottle",
            _pose(0.0, 0.0, 0.80),
            _rect_vertices(0.03, 0.03, 0.08),
        ),
        "cup": _FakeEntity(
            "cup",
            _pose(0.06, 0.0, 0.79),
            _rect_vertices(0.025, 0.025, 0.06),
        ),
    }
    program = compile_task_agent(
        _task_agent(
            {
                "id": "carry",
                "operator": "coordinated_transport",
                "object": "tray",
                "actor": {
                    "mode": "coordinated",
                    "arms": ["left_arm", "right_arm"],
                },
                "goal": {
                    "terminal_behavior": "place",
                    "payloads": [
                        {"object": "bottle", "slot": "center"},
                        {"object": "cup", "slot": "center"},
                    ],
                },
                "depends_on": [],
            }
        )
    )
    executor = ProgramExecutor(
        load_execution_program(program),
        _FakeEnv(entities),
        record_runtime=False,
    )
    step = executor.program.semantic_steps[0]
    executor._capture_payloads(step)

    assert bool(executor._verify_payloads(step)[0])
    entities["bottle"]._pose[:, 0, 3] += 0.20
    assert not bool(executor._verify_payloads(step)[0])
    entities["bottle"]._pose = _pose(0.0, 0.0, 0.80)
    entities["cup"]._pose[:, 1, 3] += 0.20
    assert not bool(executor._verify_payloads(step)[0])
    entities["cup"]._pose = _pose(0.06, 0.0, 0.79)
    entities["tray"]._pose[:, :3, :3] = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    )
    assert not bool(executor._verify_payloads(step)[0])


def test_lay_flat_surface_height_uses_rotated_live_mesh() -> None:
    entities = {
        "table": _FakeEntity(
            "table",
            _pose(0.0, 0.0, 0.70),
            _rect_vertices(0.60, 0.40, 0.02),
        ),
        "rod": _FakeEntity(
            "rod",
            _pose(0.2, 0.0, 0.80),
            _rect_vertices(0.02, 0.03, 0.10),
        ),
    }
    env = _FakeEnv(entities)
    program = load_execution_program(
        compile_task_agent(
            _task_agent(
                {
                    "id": "place",
                    "operator": "place_relative",
                    "object": "rod",
                    "actor": {"mode": "auto"},
                    "goal": {
                        "reference_object": "table",
                        "relation": "on",
                        "orientation_goal": "lay_flat",
                        "orientation_axis": "none",
                    },
                    "depends_on": [],
                }
            )
        )
    )
    step = program.semantic_steps[0]
    edge = next(
        edge
        for edge in program.edges
        if edge.actions[0]["atomic_action_class"] == "MoveHeldObject"
    )
    grounder = ActionGrounder(
        program,
        env,
        lambda uid: ObjectSemantics(
            affordance=Affordance(),
            geometry={},
            label=uid,
            entity=entities[uid],
        ),
    )
    grounded = grounder.ground(
        edge.actions[0],
        step,
        arm="left_arm",
        state=ExecutionState(last_qpos=env.robot.get_qpos()),
    )

    assert isinstance(grounded.target, HeldObjectPoseGoal)
    table_top = 0.70 + 0.02
    rotated_rod_half_height = 0.02
    surface_clearance = 0.005
    expected_surface_z = table_top + rotated_rod_half_height + surface_clearance
    assert grounded.target.object_target_pose[0, 2, 3] == pytest.approx(
        expected_surface_z,
        abs=1.0e-5,
    )


def test_orient_object_anchors_final_pose_to_support_not_live_lift_height() -> None:
    entities = {
        "table": _FakeEntity(
            "table",
            _pose(0.0, 0.0, 0.70),
            _rect_vertices(0.60, 0.40, 0.02),
        ),
        "bottle": _FakeEntity(
            "bottle",
            _pose(0.10, 0.20, 1.30),
            _rect_vertices(0.02, 0.03, 0.10),
        ),
    }
    env = _FakeEnv(entities)
    env.agent_initial_object_poses = {"bottle": _pose(0.10, 0.20, 0.78)}
    program = load_execution_program(
        compile_task_agent(
            _task_agent(
                {
                    "id": "orient",
                    "operator": "orient_object",
                    "object": "bottle",
                    "actor": {"mode": "auto"},
                    "goal": {
                        "orientation_goal": "upright",
                        "orientation_axis": "none",
                    },
                    "depends_on": [],
                }
            )
        )
    )
    step = program.semantic_steps[0]
    edges = {
        edge.actions[0]["target_binding"].get("phase"): edge
        for edge in program.edges
        if edge.actions[0]["atomic_action_class"] == "MoveHeldObject"
    }
    grounder = ActionGrounder(
        program,
        env,
        lambda uid: ObjectSemantics(
            affordance=Affordance(),
            geometry={},
            label=uid,
            entity=entities[uid],
        ),
    )

    staging = grounder.ground(
        edges["staging"].actions[0],
        step,
        arm="left_arm",
        state=ExecutionState(last_qpos=env.robot.get_qpos()),
    )
    final = grounder.ground(
        edges["final"].actions[0],
        step,
        arm="left_arm",
        state=ExecutionState(last_qpos=env.robot.get_qpos()),
    )
    expected_final_z = 0.70 + 0.02 + 0.10 + 0.05
    assert final.target_object_pose[0, 2, 3] == pytest.approx(expected_final_z)
    assert final.target_object_pose[0, :2, 3].tolist() == pytest.approx([0.10, 0.20])
    assert staging.target_object_pose[0, 2, 3] > final.target_object_pose[0, 2, 3]
    assert staging.target_object_pose[0, :2, 3].tolist() == pytest.approx([0.10, 0.20])


def test_orient_grounding_uses_mature_robot_profile_policy() -> None:
    entities = {
        "table": _FakeEntity(
            "table",
            _pose(0.0, 0.0, 0.70),
            _rect_vertices(0.60, 0.40, 0.02),
        ),
        "bottle": _FakeEntity(
            "bottle",
            _pose(0.10, 0.20, 0.78),
            _rect_vertices(0.02, 0.03, 0.10),
        ),
    }
    env = _FakeEnv(entities)
    env.agent_robot_profile = "dual_ur10"
    env.agent_initial_object_poses = {"bottle": _pose(0.10, 0.20, 0.78)}
    program = load_execution_program(
        compile_task_agent(
            _task_agent(
                {
                    "id": "orient",
                    "operator": "orient_object",
                    "object": "bottle",
                    "actor": {"mode": "auto"},
                    "goal": {
                        "orientation_goal": "upright",
                        "orientation_axis": "none",
                        "support_object": "table",
                    },
                    "depends_on": [],
                }
            )
        )
    )
    step = program.semantic_steps[0]
    pickup_edge = next(
        edge
        for edge in program.edges
        if edge.actions[0]["atomic_action_class"] == "PickUp"
    )
    final_edge = next(
        edge
        for edge in program.edges
        if edge.actions[0]["target_binding"].get("phase") == "final"
    )
    grounder = ActionGrounder(
        program,
        env,
        lambda uid: ObjectSemantics(
            affordance=Affordance(),
            geometry={},
            label=uid,
            entity=entities[uid],
        ),
    )

    pickup = grounder.ground(
        pickup_edge.actions[0],
        step,
        arm="left_arm",
        state=ExecutionState(last_qpos=env.robot.get_qpos()),
    )
    final = grounder.ground(
        final_edge.actions[0],
        step,
        arm="left_arm",
        state=ExecutionState(last_qpos=env.robot.get_qpos()),
    )

    table_top = 0.72
    bottle_half_height = 0.10
    expected_z = table_top + bottle_half_height + 0.05
    assert torch.equal(
        pickup.motion_policy["obj_upright_direction"],
        torch.tensor([0.0, 0.0, 1.0]),
    )
    assert pickup.motion_policy["rotate_upright"] == pytest.approx(torch.pi / 4)
    assert "upright_yaw_samples" not in pickup.motion_policy
    assert final.target_object_pose[0, 2, 3] == pytest.approx(expected_z)
    assert final.motion_policy["upright_local_axis"] == "long_axis"
    assert final.allow_yaw_search


def test_orient_verification_requires_upright_pose_near_initial_xy() -> None:
    bottle = _FakeEntity(
        "bottle",
        _pose(0.10, 0.20, 0.823),
        _rect_vertices(0.02, 0.03, 0.10),
    )
    env = _FakeEnv({"bottle": bottle})
    env.agent_initial_object_poses = {"bottle": _pose(0.10, 0.20, 0.78)}
    executor = ProgramExecutor(
        load_execution_program(
            compile_task_agent(
                _task_agent(
                    {
                        "id": "orient",
                        "operator": "orient_object",
                        "object": "bottle",
                        "actor": {"mode": "auto"},
                        "goal": {
                            "orientation_goal": "upright",
                            "orientation_axis": "none",
                        },
                        "depends_on": [],
                    }
                )
            )
        ),
        env,
        settle_steps=0,
        record_runtime=False,
    )
    step = executor.program.semantic_steps[0]
    executor._policies[step.id] = {
        "upright_max_tilt": torch.pi / 12,
        "upright_xy_tolerance": 0.05,
        "upright_local_axis": "z",
    }

    failed, success, _ = executor._verify_step(step, torch.tensor([False]))
    assert bool(success[0])
    assert not bool(failed[0])

    bottle._pose[:, :3, :3] = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    )
    failed, success, _ = executor._verify_step(step, torch.tensor([False]))
    assert not bool(success[0])
    assert bool(failed[0])


@pytest.mark.parametrize("yaw", [0.0, torch.pi / 3, torch.pi, -torch.pi / 2])
def test_orient_verification_accepts_any_upright_yaw(yaw: float) -> None:
    pose = _pose(0.10, 0.20, 0.823)
    pose[:, :3, :3] = torch.tensor(
        [
            [torch.cos(torch.tensor(yaw)), -torch.sin(torch.tensor(yaw)), 0.0],
            [torch.sin(torch.tensor(yaw)), torch.cos(torch.tensor(yaw)), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    bottle = _FakeEntity(
        "bottle",
        pose,
        _rect_vertices(0.02, 0.03, 0.10),
    )
    env = _FakeEnv({"bottle": bottle})
    env.agent_initial_object_poses = {"bottle": _pose(0.10, 0.20, 0.78)}
    executor = ProgramExecutor(
        load_execution_program(
            compile_task_agent(
                _task_agent(
                    {
                        "id": "orient",
                        "operator": "orient_object",
                        "object": "bottle",
                        "actor": {"mode": "auto"},
                        "goal": {
                            "orientation_goal": "upright",
                            "orientation_axis": "none",
                        },
                        "depends_on": [],
                    }
                )
            )
        ),
        env,
        settle_steps=0,
        record_runtime=False,
    )
    step = executor.program.semantic_steps[0]
    executor._target_poses[step.id] = _pose(0.10, 0.20, 0.823)
    executor._policies[step.id] = {
        "upright_max_tilt": torch.pi / 12,
        "upright_xy_tolerance": 0.05,
        "upright_local_axis": "z",
    }

    failed, success, _ = executor._verify_step(step, torch.tensor([False]))

    assert bool(success[0])
    assert not bool(failed[0])
    assert executor.retry_count == 0


def test_orient_verification_accepts_grounded_live_xy_anchor() -> None:
    bottle = _FakeEntity(
        "bottle",
        _pose(0.15, -0.10, 0.823),
        _rect_vertices(0.02, 0.03, 0.10),
    )
    env = _FakeEnv({"bottle": bottle})
    executor = ProgramExecutor(
        load_execution_program(
            compile_task_agent(
                _task_agent(
                    {
                        "id": "orient",
                        "operator": "orient_object",
                        "object": "bottle",
                        "actor": {"mode": "auto"},
                        "goal": {
                            "orientation_goal": "upright",
                            "orientation_axis": "none",
                            "position_anchor": "live_xy",
                        },
                        "depends_on": [],
                    }
                )
            )
        ),
        env,
        settle_steps=0,
        record_runtime=False,
    )
    step = executor.program.semantic_steps[0]
    executor._targets[step.id] = torch.tensor([[0.15, -0.10, 0.823]])
    executor._policies[step.id] = {
        "upright_max_tilt": torch.pi / 12,
        "upright_xy_tolerance": 0.05,
        "upright_local_axis": "z",
    }

    failed, success, _ = executor._verify_step(step, torch.tensor([False]))

    assert bool(success[0])
    assert not bool(failed[0])


def test_long_axis_upright_is_undirected_but_explicit_axis_is_not() -> None:
    pose = _pose(0.0, 0.0, 0.75)
    pose[:, :3, 1] = torch.tensor([0.0, 0.0, -1.0])
    pose[:, :3, 2] = torch.tensor([0.0, 1.0, 0.0])
    entity = _FakeEntity("can", pose, _rect_vertices(0.03, 0.10, 0.03))
    env = _FakeEnv({"can": entity})

    assert bool(
        evaluate_predicate(
            env,
            {
                "type": "object_upright",
                "object": "can",
                "local_axis": "long_axis",
            },
        )[0]
    )
    assert not bool(
        evaluate_predicate(
            env,
            {
                "type": "object_upright",
                "object": "can",
                "local_axis": "long_axis",
                "directed": True,
            },
        )[0]
    )
    assert not bool(
        evaluate_predicate(
            env,
            {
                "type": "object_upright",
                "object": "can",
                "local_axis": "y",
            },
        )[0]
    )


def test_orient_object_uses_solver_roots_when_control_groups_share_root() -> None:
    entities = {
        "left_object": _FakeEntity(
            "left_object",
            _pose(0.0, -0.20, 0.8),
            _rect_vertices(0.02, 0.02, 0.08),
        ),
        "right_object": _FakeEntity(
            "right_object",
            _pose(0.0, 0.20, 0.8),
            _rect_vertices(0.02, 0.02, 0.08),
        ),
    }
    env = _FakeEnv(entities)
    env.agent_initial_object_poses = {
        uid: entity.get_local_pose(to_matrix=True) for uid, entity in entities.items()
    }
    execution = compile_task_agent(
        _task_agent(
            *[
                {
                    "id": uid,
                    "operator": "orient_object",
                    "object": uid,
                    "actor": {"mode": "auto"},
                    "goal": {
                        "orientation_goal": "upright",
                        "orientation_axis": "none",
                    },
                    "depends_on": [],
                }
                for uid in ("left_object", "right_object")
            ]
        )
    )
    executor = ProgramExecutor(
        load_execution_program(execution), env, record_runtime=False
    )

    torch.testing.assert_close(
        env.robot.get_control_part_base_pose(name="physical_left_arm", to_matrix=True),
        env.robot.get_control_part_base_pose(name="physical_right_arm", to_matrix=True),
    )
    assert executor._preferred_in_place_arm(executor.steps["left_object"], 0) == (
        "left_arm"
    )
    assert executor._preferred_in_place_arm(executor.steps["right_object"], 0) == (
        "right_arm"
    )


def test_orient_object_arm_preference_follows_translated_robot_and_table() -> None:
    entities = {
        "table": _FakeEntity(
            "table",
            _pose(1.50, -0.70, 0.70),
            _rect_vertices(0.50, 0.40, 0.02),
        ),
        "left_object": _FakeEntity(
            "left_object",
            _pose(1.70, -0.70, 0.80),
            _rect_vertices(0.02, 0.02, 0.08),
        ),
        "right_object": _FakeEntity(
            "right_object",
            _pose(1.30, -0.70, 0.80),
            _rect_vertices(0.02, 0.02, 0.08),
        ),
    }
    env = _FakeEnv(entities)
    env.robot.get_link_pose = lambda *, link_name, to_matrix: _pose(
        1.80 if link_name == "physical_left_base" else 1.20,
        -0.70,
        0.0,
    )
    env.agent_initial_object_poses = {
        uid: entity.get_local_pose(to_matrix=True) for uid, entity in entities.items()
    }
    execution = compile_task_agent(
        _task_agent(
            *[
                {
                    "id": uid,
                    "operator": "orient_object",
                    "object": uid,
                    "actor": {"mode": "auto"},
                    "goal": {
                        "orientation_goal": "upright",
                        "orientation_axis": "none",
                    },
                    "depends_on": [],
                }
                for uid in ("left_object", "right_object")
            ]
        )
    )
    executor = ProgramExecutor(
        load_execution_program(execution), env, record_runtime=False
    )

    center, _, lateral = executor._arm_selection_workspace(
        executor.steps["left_object"]
    )
    torch.testing.assert_close(center, torch.tensor([[1.50, -0.70]]))
    torch.testing.assert_close(lateral, torch.tensor([[1.0, 0.0]]))
    assert executor._preferred_in_place_arm(executor.steps["left_object"], 0) == (
        "left_arm"
    )
    assert executor._preferred_in_place_arm(executor.steps["right_object"], 0) == (
        "right_arm"
    )


def test_coordinated_placement_uses_live_typed_target_and_profile_parts() -> None:
    entities = {
        "placing": _FakeEntity(
            "placing",
            _pose(0.0, 0.1, 0.75),
            _box_vertices(0.04),
        ),
        "support": _FakeEntity(
            "support",
            _pose(0.0, -0.1, 0.75),
            _box_vertices(0.06),
        ),
    }
    env = _FakeEnv(entities)
    compiled = compile_task_agent(
        _task_agent(
            {
                "id": "place",
                "operator": "coordinated_place",
                "object": "placing",
                "actor": {
                    "mode": "coordinated",
                    "arms": ["left_arm", "right_arm"],
                },
                "goal": {
                    "support_object": "support",
                    "relation": "on",
                    "release": True,
                },
                "depends_on": [],
            }
        )
    )
    program = load_execution_program(compiled)

    def semantics(uid: str) -> ObjectSemantics:
        return ObjectSemantics(
            affordance=Affordance(),
            geometry={},
            label=uid,
            entity=entities[uid],
        )

    step = program.semantic_steps[0]
    assert [action["atomic_action_class"] for action in program.edges[0].actions] == [
        "PickUp",
        "PickUp",
    ]
    edge = next(
        edge
        for edge in program.edges
        if edge.actions[0]["atomic_action_class"] == "CoordinatedPlacement"
    )
    grounder = ActionGrounder(program, env, semantics)
    grounded = grounder.ground(
        edge.actions[0],
        step,
        arm="coordinated",
        state=ExecutionState(last_qpos=env.robot.get_qpos()),
    )

    assert isinstance(grounded.target, CoordinatedPlacementGoal)
    assert grounded.target.release is True
    assert torch.allclose(
        grounded.target.support_object_target_pose,
        entities["support"].get_local_pose(to_matrix=True),
    )

    adapter = AtomicActionAdapter(env)
    cfg = adapter._build_config(grounded, CoordinatedPlacementOptions)
    bound_endpoints: dict[str, dict[str, str]] = {}

    class _BindingEngine:
        binding_owner_id = "runtime-contract-test"

        def bind_control_parts(
            self,
            _skill_id: str,
            endpoints: dict[str, dict[str, str]],
        ) -> ActionBinding:
            bound_endpoints.update(deepcopy(endpoints))
            return ActionBinding(owner_id=self.binding_owner_id)

    adapter._atomic_engine = _BindingEngine()
    binding = adapter._binding(
        grounded,
        adapter.capabilities.get("CoordinatedPlacement"),
    )
    assert binding.owner_id == "runtime-contract-test"
    assert bound_endpoints == {
        "placing": {
            "motion": "physical_left_arm",
            "grasp": "physical_left_eef",
        },
        "support": {
            "motion": "physical_right_arm",
            "grasp": "physical_right_eef",
        },
    }
    assert cfg.release is True


def test_online_environment_preserves_result_and_disables_terminations(
    monkeypatch: Any,
) -> None:
    installed: list[Any] = []
    initialization_order: list[tuple[str, Any]] = []

    def fake_super_init(self: Any, cfg: Any, **kwargs: Any) -> None:
        del kwargs
        initialization_order.append(("super", cfg.robot))
        self.cfg = cfg
        self.robot = object()
        self.ignore_terminations_during_agent = True

    def fake_repair(robot_cfg: Any) -> int:
        initialization_order.append(("repair", robot_cfg))
        return 1

    def fake_install(robot: Any) -> int:
        initialization_order.append(("install", robot))
        installed.append(robot)
        return 1

    monkeypatch.setattr(EmbodiedEnv, "__init__", fake_super_init)
    monkeypatch.setattr(
        env_module,
        "repair_action_engine_ur5_solver_cfg",
        fake_repair,
    )
    monkeypatch.setattr(
        env_module,
        "install_action_engine_solver_compat",
        fake_install,
    )
    monkeypatch.setattr(
        env_module.ActionEngineEnv,
        "_capture_runtime_state",
        lambda self: None,
    )
    robot_cfg = object()
    cfg = SimpleNamespace(ignore_terminations=False, robot=robot_cfg)
    env = env_module.ActionEngineEnv(
        cfg,
        agent_config={"schema_version": "action_engine_config_v2"},
        task_name="task",
        agent_config_path="/tmp/agent_config.json",
    )
    result = ExecutionResult(
        actions=[],
        success=torch.tensor([True]),
        semantic_success={},
    )

    assert cfg.ignore_terminations is True
    assert installed == [env.robot]
    assert [name for name, _ in initialization_order] == [
        "repair",
        "super",
        "install",
    ]
    assert initialization_order[0][1] is robot_cfg
    assert env._normalize_demo_action_list(result) is result


def test_environment_passes_failure_policy_to_program_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    expected = object()

    class FakeExecutor:
        def __init__(self, _program: Any, _env: Any, **kwargs: Any) -> None:
            captured.update(kwargs)

        def run(self, **kwargs: Any) -> Any:
            captured["run"] = kwargs
            return expected

    monkeypatch.setattr(
        env_module,
        "load_agent_execution_program",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(env_module, "ProgramExecutor", FakeExecutor)
    env = SimpleNamespace(
        agent_config={},
        agent_config_path="/tmp/agent_config.json",
        runtime_policy=object(),
        last_execution=None,
    )

    result = env_module.ActionEngineEnv.create_demo_action_list.__wrapped__(
        env,
        failure_policy="continue",
        runtime_run_id="run",
        episode_index=3,
    )

    assert result is expected
    assert captured["failure_policy"] == "continue"
    assert captured["run"] == {"run_id": "run", "episode_index": 3}


def test_solver_compat_repairs_only_stale_action_engine_ur_dh_defaults() -> None:
    stale_ur5 = URSolverCfg()
    stale_ur5.ur_type = "ur5"
    custom_ur5 = URSolverCfg(ur_type="ur5")
    custom_ur5.d1 = 0.1
    ur10 = URSolverCfg()
    robot_cfg = SimpleNamespace(
        solver_cfg={
            "left": stale_ur5,
            "left_alias": stale_ur5,
            "custom": custom_ur5,
            "right": ur10,
        }
    )
    expected = URSolverCfg(ur_type="ur5")
    dh_fields = ("d1", "a2", "a3", "d4", "d5", "d6")

    assert solver_compat.repair_action_engine_ur5_solver_cfg(robot_cfg) == 1
    assert tuple(getattr(stale_ur5, name) for name in dh_fields) == pytest.approx(
        tuple(getattr(expected, name) for name in dh_fields)
    )
    assert custom_ur5.d1 == pytest.approx(0.1)
    assert ur10.ur_type == "ur10"
    assert solver_compat.repair_action_engine_ur5_solver_cfg(robot_cfg) == 0


def test_solver_compat_uses_true_tcp_inverse_and_restores_solver(
    monkeypatch: Any,
) -> None:
    class FakeSolver:
        def __init__(self) -> None:
            self.device = torch.device("cpu")
            self.tcp_xpos = np.array(
                [
                    [0.0, -1.0, 0.0, 0.1],
                    [1.0, 0.0, 0.0, 0.2],
                    [0.0, 0.0, 1.0, 0.3],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            self.received: torch.Tensor | None = None
            self.saw_identity = False

        def get_ik(self, target_xpos: torch.Tensor, **kwargs: Any) -> str:
            del kwargs
            self.received = target_xpos
            self.saw_identity = np.allclose(self.tcp_xpos, np.eye(4))
            return "ok"

    monkeypatch.setattr(solver_compat, "PytorchSolver", FakeSolver)
    solver = FakeSolver()
    original_tcp = solver.tcp_xpos.copy()
    robot = SimpleNamespace(_solvers={"left": solver, "alias": solver})
    target = torch.eye(4).unsqueeze(0)

    assert solver_compat.install_pytorch_solver_tcp_compat(robot) == 1
    assert solver.get_ik(target_xpos=target) == "ok"
    assert solver.saw_identity
    assert torch.allclose(
        solver.received,
        target @ torch.linalg.inv(torch.as_tensor(original_tcp)),
    )
    assert np.allclose(solver.tcp_xpos, original_tcp)
    assert solver_compat.install_pytorch_solver_tcp_compat(robot) == 0


def test_solver_compat_aligns_ur5_analytic_ik_with_urdf_ee_frame(
    monkeypatch: Any,
) -> None:
    class FakeSolver:
        def __init__(self, ur_type: str) -> None:
            self.cfg = SimpleNamespace(ur_type=ur_type)
            self.device = torch.device("cpu")
            self.tcp_xpos = np.array(
                [
                    [0.0, -1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.2],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            self.received: torch.Tensor | None = None

        def get_ik(
            self,
            target_xpos: torch.Tensor,
            qpos_seed: torch.Tensor | None = None,
            **kwargs: Any,
        ) -> str:
            del kwargs, qpos_seed
            self.received = target_xpos
            return "ok"

    monkeypatch.setattr(solver_compat, "URSolver", FakeSolver)
    ur5 = FakeSolver("ur5")
    ur10 = FakeSolver("ur10")
    robot = SimpleNamespace(_solvers={"left": ur5, "alias": ur5, "right": ur10})
    target = torch.eye(4).unsqueeze(0)
    target[:, :3, 3] = torch.tensor([0.3, -0.2, 0.8])
    qpos_seed = torch.zeros((1, 6))

    assert solver_compat.install_ur5_solver_frame_compat(robot) == 1
    assert ur5.get_ik(target, qpos_seed) == "ok"

    tcp = torch.as_tensor(ur5.tcp_xpos)
    analytic_to_urdf = torch.eye(4)
    analytic_to_urdf[0, 3] = -0.01
    expected = target @ torch.linalg.inv(tcp) @ torch.linalg.inv(analytic_to_urdf) @ tcp
    assert torch.allclose(ur5.received, expected)
    assert ur10.received is None
    assert solver_compat.install_ur5_solver_frame_compat(robot) == 0
