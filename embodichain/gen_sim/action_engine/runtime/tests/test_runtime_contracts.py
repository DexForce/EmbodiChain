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
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from embodichain.gen_sim.action_engine.compiler import compile_task_agent
from embodichain.gen_sim.action_engine.cli.run_agent import (
    build_parser as build_run_parser,
)
from embodichain.gen_sim.action_engine.domain import (
    TASK_AGENT_SCHEMA,
    execution_program_hash,
)
from embodichain.gen_sim.action_engine.env import agent_env as env_module
from embodichain.gen_sim.action_engine.runtime.actions import AtomicActionAdapter
from embodichain.gen_sim.action_engine.runtime.executor import ProgramExecutor
from embodichain.gen_sim.action_engine.runtime.grounding import ActionGrounder
from embodichain.gen_sim.action_engine.runtime.loader import (
    load_agent_execution_program,
    load_execution_program,
)
from embodichain.gen_sim.action_engine.runtime.models import (
    ActionOutcome,
    ExecutionResult,
    GroundedAction,
)
from embodichain.gen_sim.action_engine.runtime.predicates import evaluate_predicate
from embodichain.gen_sim.action_engine.runtime.recording import RuntimeRecorder
from embodichain.gen_sim.action_engine.runtime import solver_compat
from embodichain.lab.gym.envs import EmbodiedEnv
from embodichain.lab.sim.atomic_actions import (
    Affordance,
    CoordinatedPickmentTarget,
    CoordinatedPlacementCfg,
    CoordinatedPlacementTarget,
    HeldObjectPoseTarget,
    HeldObjectState,
    ObjectSemantics,
    WorldState,
)


def _task_agent(*steps: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": TASK_AGENT_SCHEMA,
        "task": "runtime_contract",
        "goal": "Exercise the deterministic runtime contract.",
        "semantic_steps": list(steps),
    }


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


class _FakeSim:
    def __init__(self, entities: dict[str, _FakeEntity]) -> None:
        self.entities = entities

    def get_rigid_object(self, uid: str) -> _FakeEntity | None:
        return self.entities.get(uid)

    def get_rigid_object_uid_list(self) -> list[str]:
        return list(self.entities)


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
    def __init__(self, entities: dict[str, _FakeEntity] | None = None) -> None:
        self.num_envs = 1
        self.device = torch.device("cpu")
        self.robot = _FakeRobot(self.num_envs)
        self.sim = _FakeSim(entities or {})
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
        left[:, 1, 3] = 0.2
        right[:, 1, 3] = -0.2
        return left, right

    def get_current_qpos_agent(self) -> tuple[torch.Tensor, torch.Tensor]:
        qpos = self.robot.get_qpos()
        return qpos[:, self.left_arm_joints], qpos[:, self.right_arm_joints]


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


def test_loader_regenerates_in_memory_without_execution_artifact(
    tmp_path: Path,
) -> None:
    task = _task_agent(_hold_step("hold", "can", "left_arm"))
    task_path = tmp_path / "task_agent.json"
    task_path.write_text(json.dumps(task), encoding="utf-8")
    agent_config = {
        "schema_version": "action_engine_config_v1",
        "task_agent": task_path.name,
        "execution_program": "not_written.json",
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


def test_runtime_recorder_writes_checkpoints_and_rendered_env_graphs(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    program = load_execution_program(
        compile_task_agent(_task_agent(_hold_step("hold", "can", "left_arm")))
    )
    original_seed = deepcopy(program.raw)
    recorder = RuntimeRecorder(
        program,
        num_envs=2,
        run_id="run-1",
        episode_index=3,
        output_root=tmp_path,
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
    )
    recorder.step(
        step,
        torch.tensor([True, False]),
        observed=torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        target=torch.tensor([[0.1, 0.2, 0.3], [0.0, 0.0, 0.0]]),
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
        assert document["runtime"]["execution_program_hash"] == expected_hash
        assert (env_dir / "task_graph.png").read_bytes().startswith(b"\x89PNG")
    assert len(rendered_documents) == 2
    assert not list(episode_dir.rglob("*.tmp"))


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


def _held_state(env: _FakeEnv, entity: _FakeEntity) -> WorldState:
    semantics = ObjectSemantics(
        affordance=Affordance(),
        geometry={},
        label=entity.uid,
        entity=entity,
    )
    left_eef, _ = env.get_current_xpos_agent()
    object_pose = entity.get_local_pose(to_matrix=True)
    return WorldState(
        last_qpos=env.robot.get_qpos(),
        held_object=HeldObjectState(
            semantics=semantics,
            object_to_eef=torch.bmm(torch.linalg.inv(object_pose), left_eef),
            grasp_xpos=left_eef,
        ),
    )


def test_candidate_plan_is_reused_and_screens_downstream_targets(
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

    def ground(action, step, *, arm, state, reference_eef_pose=None):
        del reference_eef_pose
        action_class = action["atomic_action_class"]
        target_pose = (
            _pose(0.0, 0.2, 0.85) if action_class == "MoveHeldObject" else None
        )
        return GroundedAction(
            action_class=action_class,
            arm=arm,
            control=str(action.get("control", "arm")),
            target=SimpleNamespace(xpos=None),
            cfg={},
            target_object_pose=target_pose,
        )

    def plan(grounded: GroundedAction, state: WorldState) -> ActionOutcome:
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
    edge_result = executor._execute_edge(
        executor.edges[step.edge_ids[0]], step, failed=failed
    )

    assert len(plan_calls) == planned_call_count == len(step.edge_ids)
    assert len(plan_calls[0].cfg["downstream_object_target_poses"]) == 1
    assert bool(edge_result.failed[0])
    assert executor._object_owners["can"] == [None]


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


def test_physical_pickup_rebases_a_compliant_grasp_from_live_pose() -> None:
    entity = _FakeEntity("can", _pose(0.0, 0.2, 0.75), _box_vertices(0.03))
    env = _FakeEnv({"can": entity})
    env.robot._qpos[:, env.left_eef_joints] = env.close_state
    state = _held_state(env, entity)
    entity._pose[:, 0, 3] += 0.02
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

    assert bool(physical[0])
    left_eef, _ = env.get_current_xpos_agent()
    rebased_eef = torch.bmm(
        entity.get_local_pose(to_matrix=True),
        state.held_object.object_to_eef,
    )
    assert torch.allclose(rebased_eef, left_eef)


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
    state.held_object.grasp_xpos = _pose(0.0, -0.3, 0.75)
    edge = next(
        edge
        for edge in program.edges
        if edge.actions[0]["atomic_action_class"] == "Place"
    )
    grounder = ActionGrounder(
        program,
        env,
        lambda uid: state.held_object.semantics,
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
    assert not torch.equal(live.target.xpos, state.held_object.grasp_xpos)


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
    monkeypatch.setattr(executor, "_pack_ready_edges", lambda ready: (ready[0],))
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


def test_coordinated_transport_diagonal_is_grounded_from_live_pose() -> None:
    entities = {
        "shared_box": _FakeEntity(
            "shared_box",
            _pose(0.0, 0.0, 0.75),
            _rect_vertices(0.10, 0.06, 0.03),
        )
    }
    env = _FakeEnv(entities)
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
                        "direction": "front_left",
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
        state=WorldState(last_qpos=env.robot.get_qpos()),
    )

    assert isinstance(grounded.target, CoordinatedPickmentTarget)
    default_relation_distance = 0.16
    assert torch.allclose(
        grounded.target.object_target_pose[0, :3, 3],
        torch.tensor(
            [
                default_relation_distance,
                default_relation_distance,
                0.75,
            ]
        ),
    )


def test_coordinated_payload_monitor_rejects_drift_and_carrier_tilt() -> None:
    entities = {
        "tray": _FakeEntity(
            "tray",
            _pose(0.0, 0.0, 0.75),
            _rect_vertices(0.20, 0.14, 0.02),
        ),
        "bottle": _FakeEntity(
            "bottle",
            _pose(0.0, 0.0, 0.80),
            _rect_vertices(0.03, 0.03, 0.08),
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
                    "payloads": [{"object": "bottle", "slot": "center"}],
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
        state=WorldState(last_qpos=env.robot.get_qpos()),
    )

    assert isinstance(grounded.target, HeldObjectPoseTarget)
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
        state=WorldState(last_qpos=env.robot.get_qpos()),
    )
    final = grounder.ground(
        edges["final"].actions[0],
        step,
        arm="left_arm",
        state=WorldState(last_qpos=env.robot.get_qpos()),
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
        state=WorldState(last_qpos=env.robot.get_qpos()),
    )
    final = grounder.ground(
        final_edge.actions[0],
        step,
        arm="left_arm",
        state=WorldState(last_qpos=env.robot.get_qpos()),
    )

    table_top = 0.72
    bottle_half_height = 0.10
    expected_z = table_top + bottle_half_height + 0.05
    assert torch.equal(
        pickup.motion_policy["obj_upright_direction"],
        torch.tensor([0.0, 0.0, 1.0]),
    )
    assert pickup.motion_policy["rotate_upright"] == pytest.approx(torch.pi / 4)
    assert final.target_object_pose[0, 2, 3] == pytest.approx(expected_z)
    assert final.motion_policy["upright_local_axis"] == "long_axis"


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
                "local_axis": "y",
            },
        )[0]
    )


def test_orient_object_maps_world_y_sides_to_robot_view_arms() -> None:
    entities = {
        "left_object": _FakeEntity(
            "left_object",
            _pose(0.0, 0.20, 0.8),
            _rect_vertices(0.02, 0.02, 0.08),
        ),
        "right_object": _FakeEntity(
            "right_object",
            _pose(0.0, -0.20, 0.8),
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
                for uid in entities
            ]
        )
    )
    executor = ProgramExecutor(
        load_execution_program(execution), env, record_runtime=False
    )

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
        state=WorldState(last_qpos=env.robot.get_qpos()),
    )

    assert isinstance(grounded.target, CoordinatedPlacementTarget)
    assert grounded.target.release is True
    assert torch.allclose(
        grounded.target.support_object_target_pose,
        entities["support"].get_local_pose(to_matrix=True),
    )

    adapter = AtomicActionAdapter(env)
    cfg = adapter._build_config(grounded, CoordinatedPlacementCfg)
    assert cfg.placing_arm_control_part == "physical_left_arm"
    assert cfg.support_arm_control_part == "physical_right_arm"
    assert cfg.placing_hand_control_part == "physical_left_eef"
    assert cfg.support_hand_control_part == "physical_right_eef"


def test_online_environment_preserves_result_and_disables_terminations(
    monkeypatch: Any,
) -> None:
    installed: list[Any] = []

    def fake_super_init(self: Any, cfg: Any, **kwargs: Any) -> None:
        del kwargs
        self.cfg = cfg
        self.robot = object()
        self.ignore_terminations_during_agent = True

    monkeypatch.setattr(EmbodiedEnv, "__init__", fake_super_init)
    monkeypatch.setattr(
        env_module,
        "install_pytorch_solver_tcp_compat",
        lambda robot: installed.append(robot),
    )
    monkeypatch.setattr(
        env_module.ActionEngineEnv,
        "_capture_runtime_state",
        lambda self: None,
    )
    cfg = SimpleNamespace(ignore_terminations=False)
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
    assert env._normalize_demo_action_list(result) is result


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
