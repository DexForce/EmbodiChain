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

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
import torch

from embodichain.gen_sim.action_agent_pipeline.agents.compile_agent import CompileAgent
from embodichain.gen_sim.action_agent_pipeline.generation.seed_task_graph import (
    make_arrangement_seed_task_graph,
    make_relative_seed_task_graph,
    make_stacking_seed_task_graph,
    seed_task_graph_hash,
)
from embodichain.gen_sim.action_agent_pipeline.graph_visualization import (
    render_seed_task_graph_png,
)
from embodichain.gen_sim.action_agent_pipeline.generation.prompt_builders import (
    make_agent_config,
)
from embodichain.gen_sim.action_agent_pipeline.generation.success_specs import (
    _make_arrangement_success_spec,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.motion_policy import (
    resolve_motion_policy,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.arrangement_runtime import (
    ArrangementRuntimePlan,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.action_parts import (
    _select_arm_parts,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.parallel_execution import (
    _merge_inactive_world_state_qpos,
    build_parallel_action_stream,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.symbolic_grounding import (
    ground_symbolic_action,
    select_auto_arm_from_candidates,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.task_graph_artifact import (
    RuntimeTaskGraphRecorder,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.task_graph import (
    AgentGraphEdge,
    _classify_execution_failure,
    _physical_control_parts_for_assignments,
    _postcondition_target_positions,
)
from embodichain.lab.sim.atomic_actions import WorldState


class _Object:
    def __init__(
        self,
        positions: list[list[float]],
        *,
        half_extents: tuple[float, float, float] = (0.04, 0.04, 0.05),
    ) -> None:
        self.pose = torch.eye(4).repeat(len(positions), 1, 1)
        self.pose[:, :3, 3] = torch.tensor(positions)
        x, y, z = half_extents
        self.vertices = torch.tensor(
            [
                [sx * x, sy * y, sz * z]
                for sx in (-1.0, 1.0)
                for sy in (-1.0, 1.0)
                for sz in (-1.0, 1.0)
            ],
            dtype=torch.float32,
        )

    def get_local_pose(self, *, to_matrix: bool) -> torch.Tensor:
        assert to_matrix
        return self.pose

    def get_vertices(self, *, env_ids, scale: bool) -> torch.Tensor:
        assert scale
        return self.vertices.unsqueeze(0).repeat(len(env_ids), 1, 1)


class _Sim:
    def __init__(self, objects: dict[str, _Object]) -> None:
        self.objects = objects

    def get_rigid_object(self, uid: str):
        return self.objects.get(uid)

    def get_rigid_object_uid_list(self) -> list[str]:
        return list(self.objects)


def test_agent_config_references_only_executable_seed() -> None:
    config = make_agent_config()

    assert config["TaskAgent"] == {"seed_task_graph": "seed_task_graph.json"}


def test_seed_hash_is_independent_of_config_geometry() -> None:
    first = _relative_spec(release_position=[0.1, 0.2, 0.3])
    second = _relative_spec(release_position=[9.0, 8.0, 7.0])

    first_seed = make_relative_seed_task_graph("same_task", first)
    second_seed = make_relative_seed_task_graph("same_task", second)

    assert first_seed == second_seed
    assert seed_task_graph_hash(first_seed) == seed_task_graph_hash(second_seed)
    assert "position" not in str(first_seed)


def test_arrangement_and_stacking_expand_each_object_into_a_step() -> None:
    arrangement_steps = tuple(
        SimpleNamespace(
            runtime_uid=f"can_{index}",
            slot_index=2 - index,
            orientation_goal="preserve",
            orientation_axis="none",
        )
        for index in range(3)
    )
    arrangement = make_arrangement_seed_task_graph(
        "line",
        SimpleNamespace(
            task_description="arrange cans by size",
            order_by="size",
            order_direction="ascending",
            axis="world_x",
            anchor="center",
            semantic_order=(),
            steps=arrangement_steps,
        ),
    )
    stacking = make_stacking_seed_task_graph(
        "stack",
        SimpleNamespace(
            stack_mode="on_top",
            anchor_runtime_uid="bucket",
            steps=(
                SimpleNamespace(
                    runtime_uid="cup",
                    support_runtime_uid="bucket",
                    layer_index=0,
                    orientation_goal="preserve",
                    orientation_axis="none",
                ),
                SimpleNamespace(
                    runtime_uid="headphones",
                    support_runtime_uid="cup",
                    layer_index=1,
                    orientation_goal="preserve",
                    orientation_axis="none",
                ),
            ),
        ),
    )

    assert len(arrangement["semantic_steps"]) == 3
    assert [
        step["goal"]["nominal_slot_index"] for step in arrangement["semantic_steps"]
    ] == [0, 1, 2]
    assert [step["object"] for step in arrangement["semantic_steps"]] == [
        "can_2",
        "can_1",
        "can_0",
    ]
    assert all(
        step["goal"]["slot_constraint"] == "required"
        for step in arrangement["semantic_steps"]
    )
    assert [
        edge["actions"][0]["atomic_action_class"] for edge in arrangement["edges"][:6]
    ] == [
        "PickUp",
        "MoveHeldObject",
        "MoveHeldObject",
        "Place",
        "MoveEndEffector",
        "MoveJoints",
    ]
    assert [
        arrangement["edges"][index]["actions"][0]["target_binding"]["phase"]
        for index in (1, 2)
    ] == ["staging", "final"]
    assert arrangement["semantic_steps"][1]["depends_on"] == [
        arrangement["semantic_steps"][0]["id"]
    ]
    assert len(stacking["semantic_steps"]) == 2
    assert stacking["semantic_steps"][1]["goal"]["reference_object"] == "cup"


def test_auto_arm_selection_is_per_environment_and_deterministic() -> None:
    assignments, failed = select_auto_arm_from_candidates(
        torch.tensor([True, False, True, False]),
        torch.tensor([False, True, True, False]),
        torch.tensor([2.0, 99.0, 1.0, 99.0]),
        torch.tensor([99.0, 3.0, 1.0, 99.0]),
    )

    assert assignments == ["left_arm", "right_arm", "left_arm", None]
    assert failed.tolist() == [False, False, False, True]


def test_auto_arm_remains_unresolved_in_relative_seed() -> None:
    seed = make_relative_seed_task_graph(
        "auto_arm",
        _relative_spec(first_arm="auto"),
    )

    assert seed["semantic_steps"][0]["actor"] == {"mode": "auto"}
    assert seed["nodes"][1]["semantic"] == "Holding `object_a`"
    assert seed["nodes"][2]["semantic"] == ("`object_a` held at its semantic goal")
    assert all(
        edge["actions"][0]["actor"] == {"mode": "auto"} for edge in seed["edges"]
    )


def test_coordinated_place_seed_releases_and_retracts_both_arms() -> None:
    placement = SimpleNamespace(
        intent="coordinated_pickment",
        moved_runtime_uid="tray",
        reference_runtime_uid="table",
        relation="on",
        reference_is_initial_pose=False,
        orientation_goal="preserve",
        orientation_axis="none",
        orientation_align_to_runtime_uid=None,
        arm_request="auto",
        step_id="s01_transport_tray",
        depends_on=(),
    )
    seed = make_relative_seed_task_graph(
        "coordinated_place",
        SimpleNamespace(
            intent="coordinated_pickment",
            placements=(placement,),
            coordinated_direction="front",
            coordinated_terminal_behavior="place",
        ),
    )

    assert [len(edge["actions"]) for edge in seed["edges"]] == [1, 2, 2, 2]
    assert {
        action["target_binding"]["source"] for action in seed["edges"][1]["actions"]
    } == {"gripper_open"}
    assert seed["semantic_steps"][0]["actor"]["mode"] == "coordinated"


def test_coordinated_release_edge_grounds_one_action_per_arm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from embodichain.gen_sim.action_agent_pipeline.runtime import task_graph
    from embodichain.gen_sim.action_agent_pipeline.runtime.graph_compiler import (
        compile_agent_graph_spec,
    )

    placement = SimpleNamespace(
        intent="coordinated_pickment",
        moved_runtime_uid="tray",
        reference_runtime_uid="table",
        relation="on",
        reference_is_initial_pose=False,
        orientation_goal="preserve",
        orientation_axis="none",
        orientation_align_to_runtime_uid=None,
        arm_request="auto",
        step_id="s01_transport_tray",
        depends_on=(),
    )
    seed = make_relative_seed_task_graph(
        "coordinated_runtime",
        SimpleNamespace(
            intent="coordinated_pickment",
            placements=(placement,),
            coordinated_direction="front",
            coordinated_terminal_behavior="place",
        ),
    )
    graph = compile_agent_graph_spec(seed)
    step = next(iter(graph.semantic_steps.values()))
    release_edge = graph.edges[step.edge_ids[1]]
    env = SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        agent_robot_profile="dual_franka",
        sim=_Sim(
            {
                "tray": _Object([[0.0, 0.0, 0.2], [0.1, 0.0, 0.2]]),
                "table": _Object([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            }
        ),
        agent_initial_object_poses={},
    )
    captured = {}

    def execute_parallel_atomic_actions(**kwargs):
        captured.update(kwargs)
        return {
            "actions": [],
            "world_states": {},
            "arm_actions": {"left": None, "right": None},
            "failed_env_mask": kwargs["failed_env_mask"],
        }

    monkeypatch.setattr(
        task_graph,
        "execute_parallel_atomic_actions",
        execute_parallel_atomic_actions,
    )
    _, grounded_actions = graph._execute_coordinated_edge(
        release_edge,
        step,
        env=env,
        world_states={},
        failed=torch.zeros(2, dtype=torch.bool),
        runtime_kwargs={},
    )

    assert [action.action_spec["robot_name"] for action in grounded_actions] == [
        "left_arm",
        "right_arm",
    ]
    assert captured["left_arm_action"]["target_qpos"]["state"] == "open"
    assert captured["right_arm_action"]["target_qpos"]["state"] == "open"


def test_parallel_arm_masks_hold_inactive_environments() -> None:
    class Robot:
        control_parts = {}

        def get_qpos(self):
            return torch.zeros((2, 4), dtype=torch.float32)

    env = SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        robot=Robot(),
        left_arm_joints=[0, 1],
        left_eef_joints=[],
        right_arm_joints=[2, 3],
        right_eef_joints=[],
    )
    left = np.ones((2, 2, 2), dtype=np.float32)
    right = np.full((2, 2, 2), 2.0, dtype=np.float32)

    result = build_parallel_action_stream(
        left,
        right,
        env=env,
        left_active_env_mask=torch.tensor([True, False]),
        right_active_env_mask=torch.tensor([False, True]),
        return_result=True,
    )
    final = result["actions"][-1]

    assert final[0].tolist() == [1.0, 1.0, 0.0, 0.0]
    assert final[1].tolist() == [0.0, 0.0, 2.0, 2.0]


def test_inactive_arm_world_state_keeps_previous_qpos() -> None:
    previous = WorldState(last_qpos=torch.zeros((2, 4), dtype=torch.float32))
    candidate = WorldState(last_qpos=torch.ones((2, 4), dtype=torch.float32))

    merged = _merge_inactive_world_state_qpos(
        candidate,
        previous_state=previous,
        current_qpos=np.zeros((2, 4), dtype=np.float32),
        active_env_mask=torch.tensor([True, False]),
    )

    assert merged.last_qpos[0].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert merged.last_qpos[1].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_auto_arm_candidate_checks_transport_reachability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from embodichain.gen_sim.action_agent_pipeline.runtime import task_graph
    from embodichain.gen_sim.action_agent_pipeline.runtime.action_runtime_types import (
        ExecutedAtomicAction,
    )
    from embodichain.gen_sim.action_agent_pipeline.runtime.graph_compiler import (
        compile_agent_graph_spec,
    )

    seed = make_relative_seed_task_graph("candidate", _relative_spec())
    seed["semantic_steps"][0]["actor"] = {"mode": "auto"}
    for edge in seed["edges"]:
        edge["actions"][0]["actor"] = {"mode": "auto"}
        edge["resources"] = [
            "arm:auto" if resource.startswith("arm:") else resource
            for resource in edge["resources"]
        ]
    graph = compile_agent_graph_spec(seed)
    step = next(iter(graph.semantic_steps.values()))
    env = SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        agent_robot_profile="dual_franka",
        sim=_Sim(
            {
                "object_a": _Object([[0.0, 0.0, 0.2], [0.0, 0.0, 0.2]]),
                "object_c": _Object([[0.5, 0.0, 0.2], [0.5, 0.0, 0.2]]),
            }
        ),
        agent_initial_object_poses={},
    )
    saw_downstream_target = False

    def plan(action_spec, *, state, **kwargs):
        nonlocal saw_downstream_target
        action_class = action_spec["atomic_action_class"]
        arm = action_spec["robot_name"]
        if action_class == "PickUp":
            saw_downstream_target = bool(
                kwargs["pickup_downstream_object_target_specs"][arm]
            )
        failed = torch.tensor(
            [False, arm == "left_arm" and action_class == "MoveHeldObject"]
        )
        cost = 1.0 if arm == "left_arm" else 2.0
        trajectory = np.zeros((2, 2, 1), dtype=np.float32)
        trajectory[:, 1, 0] = cost
        return ExecutedAtomicAction(
            action=trajectory,
            next_state=state,
            robot_name=arm,
            control="arm",
            failed_env_mask=failed,
            atomic_action_class=action_class,
        )

    monkeypatch.setattr(task_graph, "_execute_atomic_action_result", plan)
    assignments, failed = graph._select_step_arms(
        step,
        env=env,
        world_states={"left": None, "right": None},
        failed=torch.zeros(2, dtype=torch.bool),
        runtime_kwargs={},
    )

    assert saw_downstream_target
    assert assignments == ["left_arm", "right_arm"]
    assert failed.tolist() == [False, False]


def test_required_opposite_arm_steps_compile_to_pickup_fork_join() -> None:
    from embodichain.gen_sim.action_agent_pipeline.runtime.graph_compiler import (
        compile_agent_graph_spec,
    )

    seed = make_relative_seed_task_graph(
        "parallel_pickups",
        _parallel_relative_spec(),
    )
    first_step, second_step = seed["semantic_steps"]
    edge_by_id = {edge["id"]: edge for edge in seed["edges"]}
    first_pick = edge_by_id[first_step["edge_ids"][0]]
    second_pick = edge_by_id[second_step["edge_ids"][0]]
    first_tail = edge_by_id[first_step["edge_ids"][-1]]
    second_transport = edge_by_id[second_step["edge_ids"][1]]

    assert first_pick["actions"][0]["atomic_action_class"] == "PickUp"
    assert second_pick["actions"][0]["atomic_action_class"] == "PickUp"
    assert first_pick["depends_on"] == second_pick["depends_on"] == []
    assert set(first_pick["resources"]).isdisjoint(second_pick["resources"])
    assert first_tail["target"] == second_pick["target"]
    assert second_transport["source"] == second_pick["target"]
    assert second_transport["depends_on"] == [
        first_step["edge_ids"][-1],
        second_step["edge_ids"][0],
    ]

    graph = compile_agent_graph_spec(seed)
    ready = [
        graph.edges[edge_id]
        for edge_id in graph.edges
        if not graph.edges[edge_id].depends_on
    ]
    packed = graph._pack_ready_edges(ready)
    assert [edge.id for edge in packed] == [first_pick["id"], second_pick["id"]]
    assert render_seed_task_graph_png(seed).startswith(b"\x89PNG\r\n\x1a\n")


def test_auto_dual_arm_request_compiles_to_distinct_pickup_group() -> None:
    from embodichain.gen_sim.action_agent_pipeline.runtime.graph_compiler import (
        compile_agent_graph_spec,
    )

    seed = make_relative_seed_task_graph(
        "auto_parallel_pickups",
        _parallel_relative_spec(auto=True),
    )
    first_step, second_step = seed["semantic_steps"]
    first_pick = seed["edges"][0]
    second_pick = seed["edges"][1]

    assert first_step["actor"] == second_step["actor"] == {"mode": "auto"}
    assert seed["allocation_groups"] == [
        {
            "id": "g01_dual_pickup",
            "semantic_step_ids": [first_step["id"], second_step["id"]],
            "arm_constraint": "distinct_arms",
            "execution_policy": "parallel_if_feasible",
            "parallel_action_classes": ["PickUp"],
            "workspace_policy": "shared_target_serial",
        }
    ]
    assert first_pick["depends_on"] == second_pick["depends_on"] == []
    graph = compile_agent_graph_spec(seed)
    assert (
        len(
            graph._pack_ready_edges(
                [graph.edges[first_pick["id"]], graph.edges[second_pick["id"]]]
            )
        )
        == 2
    )


def test_auto_dual_pickup_joint_selection_uses_one_feasible_permutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from embodichain.gen_sim.action_agent_pipeline.runtime.graph_compiler import (
        compile_agent_graph_spec,
    )

    graph = compile_agent_graph_spec(
        make_relative_seed_task_graph(
            "auto_joint_selection",
            _parallel_relative_spec(auto=True),
        )
    )
    ready = [edge for edge in graph.edges.values() if not edge.depends_on]
    env = SimpleNamespace(num_envs=2, device=torch.device("cpu"))

    def plan(step, *, arm, failed, **kwargs):
        del kwargs
        first = step.id.startswith("s01")
        feasible = torch.tensor(
            [first == (arm == "left_arm"), first != (arm == "left_arm")],
            dtype=torch.bool,
        )
        return feasible & ~failed, torch.ones(2, dtype=torch.float32)

    monkeypatch.setattr(graph, "_plan_arm_candidate", plan)
    assignments, selection_failed = graph._select_parallel_pickup_arms(
        ready,
        env=env,
        world_states={"left": None, "right": None},
        failed=torch.zeros(2, dtype=torch.bool),
        runtime_kwargs={},
        arrangement_plan=None,
    )

    steps = [graph.semantic_step_by_edge[edge.id] for edge in ready]
    assert assignments[steps[0].id] == ["left_arm", "right_arm"]
    assert assignments[steps[1].id] == ["right_arm", "left_arm"]
    assert selection_failed.tolist() == [False, False]


def test_auto_dual_pickup_partitions_opposite_environment_assignments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from embodichain.gen_sim.action_agent_pipeline.runtime import task_graph
    from embodichain.gen_sim.action_agent_pipeline.runtime.graph_compiler import (
        compile_agent_graph_spec,
    )

    graph = compile_agent_graph_spec(
        make_relative_seed_task_graph(
            "auto_partitioned_execution",
            _parallel_relative_spec(auto=True),
        )
    )
    ready = [edge for edge in graph.edges.values() if not edge.depends_on]
    env = SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        agent_robot_profile="dual_franka",
        sim=_Sim(
            {
                "cube": _Object([[-0.2, 0.0, 0.1], [0.2, 0.0, 0.1]]),
                "cup": _Object([[0.2, 0.0, 0.1], [-0.2, 0.0, 0.1]]),
                "basket": _Object([[0.0, 0.3, 0.1], [0.0, 0.3, 0.1]]),
            }
        ),
        agent_initial_object_poses={},
    )
    steps = [graph.semantic_step_by_edge[edge.id] for edge in ready]
    assignments = {
        steps[0].id: ["left_arm", "right_arm"],
        steps[1].id: ["right_arm", "left_arm"],
    }
    calls = []

    def execute_parallel_atomic_actions(**kwargs):
        calls.append(kwargs["left_active_env_mask"].tolist())
        state = SimpleNamespace(last_qpos=torch.zeros((2, 1)))
        executed = SimpleNamespace(next_state=state)
        return {
            "actions": [],
            "world_states": kwargs["world_states"],
            "arm_actions": {"left": executed, "right": executed},
            "failed_env_mask": kwargs["failed_env_mask"],
        }

    monkeypatch.setattr(
        task_graph,
        "execute_parallel_atomic_actions",
        execute_parallel_atomic_actions,
    )
    graph._execute_parallel_pickup_edges(
        ready,
        assignments_by_step=assignments,
        env=env,
        world_states={},
        failed=torch.zeros(2, dtype=torch.bool),
        runtime_kwargs={},
        arrangement_plan=None,
    )

    assert calls == [[True, False], [False, True]]
    assert set(graph._step_arm_world_states) == {
        (steps[0].id, "left_arm"),
        (steps[0].id, "right_arm"),
        (steps[1].id, "left_arm"),
        (steps[1].id, "right_arm"),
    }


def test_parallel_pickup_runtime_packs_both_required_arms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from embodichain.gen_sim.action_agent_pipeline.runtime import task_graph
    from embodichain.gen_sim.action_agent_pipeline.runtime.graph_compiler import (
        compile_agent_graph_spec,
    )

    graph = compile_agent_graph_spec(
        make_relative_seed_task_graph(
            "parallel_runtime",
            _parallel_relative_spec(),
        )
    )
    ready = [
        graph.edges[edge_id]
        for edge_id in graph.edges
        if not graph.edges[edge_id].depends_on
    ]
    batch = graph._pack_ready_edges(ready)
    env = SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        agent_robot_profile="dual_franka",
        sim=_Sim(
            {
                "cube": _Object([[-0.2, 0.0, 0.1]]),
                "cup": _Object([[0.2, 0.0, 0.1]]),
                "basket": _Object([[0.0, 0.3, 0.1]]),
            }
        ),
        agent_initial_object_poses={},
    )
    captured = {}

    def execute_parallel_atomic_actions(**kwargs):
        captured.update(kwargs)
        return {
            "actions": [],
            "world_states": kwargs["world_states"],
            "arm_actions": {"left": None, "right": None},
            "failed_env_mask": kwargs["failed_env_mask"],
        }

    monkeypatch.setattr(
        task_graph,
        "execute_parallel_atomic_actions",
        execute_parallel_atomic_actions,
    )
    assignments = {
        step.id: [str(step.actor["arm"])] for step in graph.semantic_steps.values()
    }
    graph._execute_parallel_pickup_edges(
        batch,
        assignments_by_step=assignments,
        env=env,
        world_states={},
        failed=torch.tensor([False]),
        runtime_kwargs={},
        arrangement_plan=None,
    )

    assert captured["left_arm_action"]["target_object"]["obj_name"] == "cube"
    assert captured["right_arm_action"]["target_object"]["obj_name"] == "cup"
    assert captured["left_active_env_mask"].tolist() == [True]
    assert captured["right_active_env_mask"].tolist() == [True]


def test_action_dag_runtime_executes_parallel_pickups_before_serial_transports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from embodichain.gen_sim.action_agent_pipeline.runtime import task_graph
    from embodichain.gen_sim.action_agent_pipeline.runtime.graph_compiler import (
        compile_agent_graph_spec,
    )

    graph = compile_agent_graph_spec(
        make_relative_seed_task_graph(
            "dag_runtime",
            _parallel_relative_spec(),
        )
    )

    class Robot:
        def get_qpos(self):
            return torch.zeros((1, 4), dtype=torch.float32)

    eef_pose = torch.eye(4).unsqueeze(0)
    env = SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        agent_robot_profile="dual_franka",
        robot=Robot(),
        sim=_Sim(
            {
                "cube": _Object([[-0.2, 0.0, 0.1]]),
                "cup": _Object([[0.2, 0.0, 0.1]]),
                "basket": _Object([[0.0, 0.3, 0.1]]),
            }
        ),
        agent_initial_object_poses={},
        get_current_xpos_agent=lambda: (eef_pose, eef_pose),
    )
    calls = []

    class Recorder:
        output_dir = tmp_path

        def __init__(self, *args, **kwargs):
            pass

        def begin_step(self, *args, **kwargs):
            pass

        def record_edge(self, *args, **kwargs):
            pass

        def complete_step(self, *args, **kwargs):
            pass

        def finalize(self, *args, **kwargs):
            pass

    def select(step, *, failed, **kwargs):
        arm = str(step.actor["arm"])
        return [None if bool(failed[0]) else arm], torch.zeros_like(failed)

    def parallel(edges, *, failed, world_states, **kwargs):
        calls.append(tuple(edge.id for edge in edges))
        grounded = {}
        for edge in edges:
            step = graph.semantic_step_by_edge[edge.id]
            grounded[edge.id] = (
                ground_symbolic_action(
                    edge.symbolic_actions[0],
                    step,
                    env=env,
                    arm=str(step.actor["arm"]),
                ),
            )
        return (
            {
                "actions": [],
                "world_states": dict(world_states),
                "arm_actions": {},
                "failed_env_mask": failed.clone(),
            },
            grounded,
        )

    def serial(edge, step, *, failed, world_states, **kwargs):
        calls.append((edge.id,))
        grounded = ground_symbolic_action(
            edge.symbolic_actions[0],
            step,
            env=env,
            arm=str(step.actor["arm"]),
        )
        return (
            {
                "actions": [],
                "world_states": dict(world_states),
                "arm_actions": {},
                "failed_env_mask": failed.clone(),
            },
            (grounded,),
        )

    def complete(step, *, failed, **kwargs):
        success = ~failed
        observed = env.sim.get_rigid_object(step.object_uid).pose[:, :3, 3]
        return failed, success, observed, None, None

    monkeypatch.setattr(task_graph, "RuntimeTaskGraphRecorder", Recorder)
    monkeypatch.setattr(graph, "_select_step_arms", select)
    monkeypatch.setattr(graph, "_execute_parallel_pickup_edges", parallel)
    monkeypatch.setattr(graph, "_execute_symbolic_edge", serial)
    monkeypatch.setattr(graph, "_complete_semantic_step", complete)

    result = graph.run(env=env, semantic_step_settle_steps=0)
    steps = list(graph.semantic_steps.values())

    assert calls[0] == (steps[0].edge_ids[0], steps[1].edge_ids[0])
    assert [edge_id for call in calls[1:] for edge_id in call] == [
        *steps[0].edge_ids[1:],
        *steps[1].edge_ids[1:],
    ]
    assert result.runtime_success.tolist() == [True]
    assert all(
        success.tolist() == [True] for success in result.semantic_step_success.values()
    )


def test_required_arm_candidate_preflights_release_retreat_and_home(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from embodichain.gen_sim.action_agent_pipeline.runtime import task_graph
    from embodichain.gen_sim.action_agent_pipeline.runtime.action_runtime_types import (
        ExecutedAtomicAction,
    )
    from embodichain.gen_sim.action_agent_pipeline.runtime.graph_compiler import (
        compile_agent_graph_spec,
    )

    graph = compile_agent_graph_spec(
        make_relative_seed_task_graph("full_path", _relative_spec())
    )
    step = next(iter(graph.semantic_steps.values()))
    env = SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        agent_robot_profile="dual_franka",
        sim=_Sim(
            {
                "object_a": _Object([[0.0, 0.0, 0.2]]),
                "object_c": _Object([[0.5, 0.0, 0.2]]),
            }
        ),
        agent_initial_object_poses={},
    )
    planned_classes = []

    def plan(action_spec, *, state, **kwargs):
        planned_classes.append(action_spec["atomic_action_class"])
        return ExecutedAtomicAction(
            action=np.zeros((1, 2, 1), dtype=np.float32),
            next_state=state,
            robot_name=action_spec["robot_name"],
            control=action_spec["control"],
            failed_env_mask=torch.tensor([False]),
            atomic_action_class=action_spec["atomic_action_class"],
        )

    monkeypatch.setattr(task_graph, "_execute_atomic_action_result", plan)
    assignments, failed = graph._select_step_arms(
        step,
        env=env,
        world_states={"left": None, "right": None},
        failed=torch.tensor([False]),
        runtime_kwargs={},
    )

    assert assignments == ["left_arm"]
    assert failed.tolist() == [False]
    assert planned_classes == [
        "PickUp",
        "MoveHeldObject",
        "Place",
        "MoveEndEffector",
        "MoveJoints",
    ]


def test_dynamic_retreat_caps_target_at_profile_workspace_height() -> None:
    from embodichain.gen_sim.action_agent_pipeline.runtime.graph_compiler import (
        compile_agent_graph_spec,
    )

    graph = compile_agent_graph_spec(
        make_relative_seed_task_graph("dynamic_retreat", _relative_spec())
    )
    step = next(iter(graph.semantic_steps.values()))
    retreat_edge = next(
        graph.edges[edge_id]
        for edge_id in step.edge_ids
        if graph.edges[edge_id].symbolic_actions[0]["atomic_action_class"]
        == "MoveEndEffector"
    )
    eef_pose = torch.eye(4).unsqueeze(0)
    eef_pose[:, 2, 3] = 0.98
    env = SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        agent_robot_profile="dual_franka",
        sim=_Sim(
            {
                "object_a": _Object([[0.0, 0.0, 0.2]]),
                "object_c": _Object([[0.5, 0.0, 0.2]]),
            }
        ),
        agent_initial_object_poses={},
        get_current_xpos_agent=lambda: (eef_pose, eef_pose),
    )

    grounded = ground_symbolic_action(
        retreat_edge.symbolic_actions[0],
        step,
        env=env,
        arm="left_arm",
    )

    assert grounded.action_spec["target_pose"]["reference"] == "absolute"
    assert grounded.action_spec["target_pose"]["position_by_env"][0][
        2
    ] == pytest.approx(1.10)
    assert grounded.motion_policy["resolved_retreat_height_by_env"] == pytest.approx(
        [0.12]
    )


def test_inside_postcondition_uses_live_container_relation() -> None:
    from embodichain.gen_sim.action_agent_pipeline.runtime.graph_compiler import (
        compile_agent_graph_spec,
    )

    placement = SimpleNamespace(
        intent="place_relative",
        moved_runtime_uid="payload",
        reference_runtime_uid="basket",
        relation="inside",
        reference_is_initial_pose=False,
        orientation_goal="preserve",
        orientation_axis="none",
        orientation_align_to_runtime_uid=None,
        arm_request="left",
        step_id="s01_inside",
        depends_on=(),
    )
    graph = compile_agent_graph_spec(
        make_relative_seed_task_graph(
            "inside_relation",
            SimpleNamespace(
                intent="place_relative",
                placements=(placement,),
                coordinated_direction=None,
                coordinated_terminal_behavior=None,
            ),
        )
    )
    step = next(iter(graph.semantic_steps.values()))
    env = SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        sim=_Sim(
            {
                "payload": _Object([[0.02, 0.01, 0.12]]),
                "basket": _Object([[0.0, 0.0, 0.0]]),
            }
        ),
    )

    failed, success, _, distance, tolerance = graph._complete_semantic_step(
        step,
        env=env,
        failed=torch.tensor([False]),
        target_positions=torch.tensor([[0.0, 0.0, 0.90]]),
        motion_policy={"postcondition_tolerance": 0.01},
        settle_steps=0,
    )

    assert failed.tolist() == [False]
    assert success.tolist() == [True]
    assert distance is None
    assert tolerance is None


def test_cleanup_failure_is_degraded_but_not_fatal() -> None:
    failed, cleanup_failed = _classify_execution_failure(
        torch.tensor([False, True]),
        torch.tensor([True, True]),
        cleanup=True,
    )
    fatal_failed, fatal_cleanup = _classify_execution_failure(
        torch.tensor([False, True]),
        torch.tensor([True, True]),
        cleanup=False,
    )

    assert failed.tolist() == [False, True]
    assert cleanup_failed.tolist() == [True, False]
    assert fatal_failed.tolist() == [True, True]
    assert fatal_cleanup.tolist() == [False, False]


def test_semantic_goal_target_is_not_overwritten_by_release_edge() -> None:
    semantic_target = torch.tensor([[0.1, 0.2, 0.3]])
    release_target = torch.tensor([[0.1, 0.2, 0.39]])
    semantic_pose = torch.eye(4).unsqueeze(0)
    semantic_pose[:, :3, 3] = semantic_target
    release_pose = torch.eye(4).unsqueeze(0)
    release_pose[:, :3, 3] = release_target
    semantic_edge = AgentGraphEdge(
        id="e01_transport",
        source="v0",
        target="v1",
        symbolic_actions=(
            {
                "atomic_action_class": "MoveHeldObject",
                "target_binding": {"kind": "semantic_goal"},
            },
        ),
    )
    release_edge = AgentGraphEdge(
        id="e02_release",
        source="v1",
        target="v2",
        symbolic_actions=(
            {
                "atomic_action_class": "Place",
                "target_binding": {"kind": "current_held_pose"},
            },
        ),
    )

    resolved_semantic = _postcondition_target_positions(
        semantic_edge,
        arm_actions={
            "left": SimpleNamespace(resolved_object_target_pose=semantic_pose)
        },
        grounded_actions=(SimpleNamespace(target_object_pose=semantic_target),),
    )
    resolved_release = _postcondition_target_positions(
        release_edge,
        arm_actions={"left": SimpleNamespace(resolved_object_target_pose=release_pose)},
        grounded_actions=(SimpleNamespace(target_object_pose=None),),
    )

    assert torch.equal(resolved_semantic, semantic_target)
    assert resolved_release is None


def test_auto_arm_skips_candidate_planning_when_every_environment_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from embodichain.gen_sim.action_agent_pipeline.runtime.graph_compiler import (
        compile_agent_graph_spec,
    )

    seed = make_relative_seed_task_graph(
        "failed_auto_candidate",
        _relative_spec(first_arm="auto"),
    )
    graph = compile_agent_graph_spec(seed)
    step = next(iter(graph.semantic_steps.values()))

    def unexpected_plan(**kwargs):
        raise AssertionError("candidate planning must not run for failed environments")

    monkeypatch.setattr(graph, "_plan_arm_candidate", unexpected_plan)
    assignments, selection_failed = graph._select_step_arms(
        step,
        env=SimpleNamespace(num_envs=2, device=torch.device("cpu")),
        world_states={"left": None, "right": None},
        failed=torch.tensor([True, True]),
        runtime_kwargs={},
    )

    assert assignments == [None, None]
    assert selection_failed.tolist() == [False, False]


def test_reversed_semantic_arm_slots_resolve_to_physical_control_parts() -> None:
    env = SimpleNamespace(
        left_arm_joints=[4, 5],
        left_eef_joints=[6],
        right_arm_joints=[0, 1],
        right_eef_joints=[2],
        get_agent_arm_control_part=lambda is_left: (
            "right_arm" if is_left else "left_arm"
        ),
        get_agent_eef_control_part=lambda is_left: (
            "right_eef" if is_left else "left_eef"
        ),
    )

    semantic_left = _select_arm_parts(env, "left_arm")
    semantic_right = _select_arm_parts(env, "right_arm")
    physical_assignments = _physical_control_parts_for_assignments(
        env,
        ["left_arm", "right_arm", None, "coordinated"],
    )

    assert semantic_left[0] is True
    assert semantic_left[1:3] == ("right_arm", "right_eef")
    assert semantic_left[3:] == ([4, 5], [6])
    assert semantic_right[0] is False
    assert semantic_right[1:3] == ("left_arm", "left_eef")
    assert semantic_right[3:] == ([0, 1], [2])
    assert physical_assignments == [
        "right_arm",
        "left_arm",
        None,
        "coordinated",
    ]


def test_grounding_reads_moved_object_and_live_reference_again() -> None:
    seed = make_relative_seed_task_graph("move_twice", _relative_spec(two_steps=True))
    env = SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        agent_robot_profile="dual_franka",
        sim=_Sim(
            {
                "object_a": _Object([[0.0, 0.0, 0.2], [1.0, 0.0, 0.2]]),
                "object_c": _Object([[0.5, 0.0, 0.2], [1.5, 0.0, 0.2]]),
            }
        ),
        agent_initial_object_poses={},
    )
    second_step_data = seed["semantic_steps"][1]
    second_step = SimpleNamespace(
        id=second_step_data["id"],
        operator=second_step_data["operator"],
        object_uid=second_step_data["object"],
        goal=second_step_data["goal"],
    )
    env.sim.objects["object_a"].pose[:, 1, 3] = -0.4
    env.sim.objects["object_c"].pose[:, 0, 3] += 0.25

    grounded = ground_symbolic_action(
        (
            seed["edges"][second_step_data["edge_ids"][1]]["actions"][0]
            if isinstance(second_step_data["edge_ids"][1], int)
            else next(
                edge["actions"][0]
                for edge in seed["edges"]
                if edge["id"] == second_step_data["edge_ids"][1]
            )
        ),
        second_step,
        env=env,
        arm="right_arm",
    )

    assert grounded.object_pose[:, 1, 3].tolist() == pytest.approx([-0.4, -0.4])
    assert grounded.reference_pose[:, 0, 3].tolist() == pytest.approx([0.75, 1.75])
    assert grounded.target_object_pose[:, 0].tolist() == pytest.approx([0.75, 1.75])
    assert grounded.target_object_pose[:, 1].tolist() == pytest.approx([-0.18, -0.18])


def test_motion_policy_requires_known_profile_and_policy() -> None:
    assert (
        resolve_motion_policy("dual_franka", "default_pickup")["pre_grasp_distance"] > 0
    )
    assert (
        resolve_motion_policy("dual_franka", "default_transport")[
            "postcondition_tolerance"
        ]
        > 0
    )
    with pytest.raises(ValueError, match="robot profile"):
        resolve_motion_policy("unknown", "default_pickup")
    with pytest.raises(ValueError, match="not registered"):
        resolve_motion_policy("dual_franka", "unknown")


def test_runtime_arrangement_slots_are_centered_per_environment() -> None:
    seed = make_arrangement_seed_task_graph(
        "runtime_line",
        SimpleNamespace(
            task_description="arrange freely",
            order_by="none",
            order_direction="ascending",
            axis="world_y",
            anchor="table_center",
            semantic_order=(),
            steps=tuple(
                SimpleNamespace(
                    runtime_uid=f"object_{index}",
                    slot_index=index,
                    orientation_goal="preserve",
                    orientation_axis="none",
                )
                for index in range(3)
            ),
        ),
    )
    table = _Object(
        [[0.0, 0.0, 0.0], [0.3, -0.2, 0.0]],
        half_extents=(0.6, 0.5, 0.04),
    )
    objects = {
        f"object_{index}": _Object(
            [[0.1 * index, 0.0, 0.1], [0.3, -0.2 + 0.1 * index, 0.1]]
        )
        for index in range(3)
    }
    env = SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        sim=_Sim({"table": table, **objects}),
    )
    from embodichain.gen_sim.action_agent_pipeline.runtime.graph_compiler import (
        compile_agent_graph_spec,
    )

    graph = compile_agent_graph_spec(seed)
    plan = ArrangementRuntimePlan(
        env=env,
        semantic_steps=list(graph.semantic_steps.values()),
    )

    assert plan.slot_positions[:, :, 1].mean(dim=1).tolist() == pytest.approx(
        [0.0, -0.2]
    )
    assert plan.slot_positions[0, :, 1].tolist() == pytest.approx(
        [-plan.spacing[0].item(), 0.0, plan.spacing[0].item()]
    )
    assert plan.slot_positions[1, :, 1].tolist() != pytest.approx(
        plan.slot_positions[0, :, 1].tolist()
    )
    table_lower = plan.table_bounds[:, 0, :2]
    table_upper = plan.table_bounds[:, 1, :2]
    assert bool((plan.slot_positions[:, :, :2] >= table_lower[:, None, :]).all())
    assert bool((plan.slot_positions[:, :, :2] <= table_upper[:, None, :]).all())


def test_runtime_arrangement_searches_a_safe_parallel_row() -> None:
    seed = make_arrangement_seed_task_graph(
        "offset_line",
        SimpleNamespace(
            task_description="form a row",
            order_by="none",
            order_direction="ascending",
            axis="world_y",
            anchor="table_center",
            semantic_order=(),
            steps=tuple(
                SimpleNamespace(
                    runtime_uid=f"object_{index}",
                    slot_index=index,
                    orientation_goal="preserve",
                    orientation_axis="none",
                )
                for index in range(3)
            ),
        ),
    )
    env = SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        sim=_Sim(
            {
                "table": _Object(
                    [[0.0, 0.0, 0.0]],
                    half_extents=(0.6, 0.5, 0.04),
                ),
                "fixed_obstacle": _Object(
                    [[0.0, 0.0, 0.1]],
                    half_extents=(0.06, 0.22, 0.08),
                ),
                **{
                    f"object_{index}": _Object([[0.3, 0.1 * index, 0.1]])
                    for index in range(3)
                },
            }
        ),
    )
    from embodichain.gen_sim.action_agent_pipeline.runtime.graph_compiler import (
        compile_agent_graph_spec,
    )

    graph = compile_agent_graph_spec(seed)
    plan = ArrangementRuntimePlan(
        env=env,
        semantic_steps=list(graph.semantic_steps.values()),
    )

    assert abs(float(plan.slot_positions[0, :, 0].mean())) >= 0.125


def test_arrangement_success_is_relational_and_seed_derived() -> None:
    steps = tuple(
        SimpleNamespace(
            runtime_uid=f"object_{index}",
            slot_index=index,
            orientation_goal="preserve",
            orientation_axis="none",
        )
        for index in range(3)
    )
    free_seed = make_arrangement_seed_task_graph(
        "free_success",
        SimpleNamespace(
            task_description="form a row",
            order_by="none",
            order_direction="ascending",
            axis="world_y",
            anchor="table_center",
            semantic_order=(),
            steps=steps,
        ),
    )
    ordered_seed = make_arrangement_seed_task_graph(
        "ordered_success",
        SimpleNamespace(
            task_description="arrange by size",
            order_by="size",
            order_direction="ascending",
            axis="world_y",
            anchor="table_center",
            semantic_order=tuple(step.runtime_uid for step in steps),
            steps=steps,
        ),
    )

    free_success = _make_arrangement_success_spec(free_seed)
    ordered_success = _make_arrangement_success_spec(ordered_seed)
    free_types = [term["type"] for term in free_success["terms"]]
    ordered_types = [term["type"] for term in ordered_success["terms"]]

    assert "object_xy_near" not in str(free_success)
    assert "target_xy" not in str(ordered_success)
    assert "objects_ordered" not in free_types
    assert "objects_ordered" in ordered_types


@pytest.mark.parametrize(
    "schema_version",
    ["seed_task_graph_v2", "seed_task_graph_v3", "seed_task_graph_v4"],
)
def test_legacy_seed_requires_regeneration(schema_version: str) -> None:
    legacy = make_relative_seed_task_graph("legacy_seed", _relative_spec())
    legacy["schema_version"] = schema_version

    with pytest.raises(ValueError, match="--overwrite"):
        seed_task_graph_hash(legacy)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("target_xy", [0.1, 0.2]),
        ("active_arm", "left_arm"),
        ("trajectory", [[0.0, 1.0]]),
    ],
)
def test_seed_v3_rejects_grounded_arrangement_fields(
    field_name: str,
    value,
) -> None:
    seed = make_arrangement_seed_task_graph(
        "coordinate_free",
        SimpleNamespace(
            task_description="form a row",
            order_by="none",
            order_direction="ascending",
            axis="world_y",
            anchor="table_center",
            semantic_order=(),
            steps=(
                SimpleNamespace(
                    runtime_uid="object_a",
                    slot_index=0,
                    orientation_goal="preserve",
                    orientation_axis="none",
                ),
            ),
        ),
    )
    invalid = deepcopy(seed)
    invalid["semantic_steps"][0]["goal"][field_name] = value

    with pytest.raises(ValueError, match="grounded|geometric"):
        seed_task_graph_hash(invalid)


def test_free_arrangement_rematches_only_unfinished_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from embodichain.gen_sim.action_agent_pipeline.runtime.graph_compiler import (
        compile_agent_graph_spec,
    )

    seed = make_arrangement_seed_task_graph(
        "free_rematch",
        SimpleNamespace(
            task_description="form a row",
            order_by="none",
            order_direction="ascending",
            axis="world_y",
            anchor="table_center",
            semantic_order=(),
            steps=tuple(
                SimpleNamespace(
                    runtime_uid=f"object_{index}",
                    slot_index=index,
                    orientation_goal="preserve",
                    orientation_axis="none",
                )
                for index in range(3)
            ),
        ),
    )
    graph = compile_agent_graph_spec(seed)
    env = SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        agent_robot_profile="dual_franka",
        sim=_Sim(
            {
                "table": _Object(
                    [[0.0, 0.0, 0.0]],
                    half_extents=(0.6, 0.5, 0.04),
                ),
                **{
                    f"object_{index}": _Object([[0.0, 0.1 * index, 0.1]])
                    for index in range(3)
                },
            }
        ),
    )
    plan = ArrangementRuntimePlan(
        env=env,
        semantic_steps=list(graph.semantic_steps.values()),
    )
    steps = list(graph.semantic_steps.values())
    plan.mark_completed(steps[0].id, torch.tensor([True]))

    def candidate(step, *, arrangement_plan, **kwargs):
        slot = int(arrangement_plan.resolved_slots[step.id][0].item())
        expected = 2 if step.id == steps[1].id else 1
        feasible = torch.tensor([slot == expected])
        return feasible, torch.tensor([float(slot + 1)])

    monkeypatch.setattr(graph, "_plan_arm_candidate", candidate)
    reassigned = graph._reassign_free_slots(
        env=env,
        world_states={"left": None, "right": None},
        failed=torch.tensor([False]),
        trigger_mask=torch.tensor([True]),
        runtime_kwargs={},
        arrangement_plan=plan,
    )

    assert reassigned.tolist() == [True]
    assert int(plan.resolved_slots[steps[0].id][0]) == 0
    assert int(plan.resolved_slots[steps[1].id][0]) == 2
    assert int(plan.resolved_slots[steps[2].id][0]) == 1
    assert plan.metadata(steps[1])[0]["slot_reassigned"] is True


def test_compile_agent_rejects_precomputed_task_graph() -> None:
    agent = CompileAgent(task_name="legacy")

    with pytest.raises(ValueError, match="--overwrite"):
        agent.generate(task_graph={"task": "legacy"})


def test_recorder_writes_every_environment_json_and_png(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from embodichain.gen_sim.action_agent_pipeline.runtime import task_graph_artifact

    seed = make_relative_seed_task_graph("recording", _relative_spec())
    env = SimpleNamespace(num_envs=2, agent_robot_profile="dual_franka")
    monkeypatch.setattr(task_graph_artifact, "_outputs_root", lambda: tmp_path)
    recorder = RuntimeTaskGraphRecorder(
        seed,
        env=env,
        run_id="test_run",
        episode_index=3,
    )

    step_data = seed["semantic_steps"][0]
    step = SimpleNamespace(id=step_data["id"])
    recorder.begin_step(
        step,
        assignments=["left_arm", None],
        object_pose=torch.eye(4).repeat(2, 1, 1),
        reference_pose=None,
        active_mask=torch.tensor([True, True]),
        selection_failed_mask=torch.tensor([False, True]),
    )
    recorder.record_edge(
        step_data["edge_ids"][0],
        assignments=["left_arm", None],
        grounded_actions=(
            SimpleNamespace(
                action_spec={"robot_name": "left_arm"},
                object_pose=torch.eye(4).repeat(2, 1, 1),
                reference_pose=None,
                target_object_pose=None,
                motion_policy={"sample_interval": 45},
            ),
        ),
        failed_before=torch.tensor([False, True]),
        failed_after=torch.tensor([False, True]),
        grounding_failed=torch.tensor([False, True]),
        action_steps=1,
        arm_actions={},
    )
    recorder.complete_step(
        step.id,
        success=torch.tensor([True, False]),
        failed_mask=torch.tensor([False, True]),
        observed_positions=torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        target_positions=torch.tensor([[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]]),
        position_error=torch.tensor([0.0, 0.5]),
        tolerance=0.08,
    )
    recorder.finalize(torch.tensor([False, True]))

    for env_id in range(2):
        env_dir = (
            tmp_path
            / "recording"
            / "runs"
            / "test_run"
            / "episode_0003"
            / f"env_{env_id:04d}"
        )
        assert (env_dir / "task_graph.json").is_file()
        assert (
            (env_dir / "task_graph.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        )
        document = json.loads((env_dir / "task_graph.json").read_text(encoding="utf-8"))
        postcondition = document["semantic_steps"][0]["runtime"]["postcondition"]
        assert postcondition["observed_object_position"] is not None
        assert postcondition["tolerance"] == pytest.approx(0.08)
        edge_runtime = document["edges"][0]["actions"][0]["runtime"]
        assert edge_runtime["status"] == ("executed" if env_id == 0 else "failed")
        if env_id == 1:
            assert edge_runtime["failure_reason"] == "no feasible arm candidate"


def test_recorder_accepts_partitioned_arm_actions_by_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from embodichain.gen_sim.action_agent_pipeline.runtime import task_graph_artifact

    seed = make_relative_seed_task_graph(
        "partitioned_recording",
        _relative_spec(first_arm="auto"),
    )
    env = SimpleNamespace(num_envs=2, agent_robot_profile="dual_franka")
    monkeypatch.setattr(task_graph_artifact, "_outputs_root", lambda: tmp_path)
    recorder = RuntimeTaskGraphRecorder(
        seed,
        env=env,
        run_id="partitioned_run",
        episode_index=0,
    )
    step_data = seed["semantic_steps"][0]
    step = SimpleNamespace(id=step_data["id"])
    assignments = ["left_arm", "right_arm"]
    object_pose = torch.eye(4).repeat(2, 1, 1)
    recorder.begin_step(
        step,
        assignments=assignments,
        object_pose=object_pose,
        reference_pose=None,
        active_mask=torch.tensor([True, True]),
        selection_failed_mask=torch.tensor([False, False]),
    )
    grounded_actions = tuple(
        SimpleNamespace(
            action_spec={"robot_name": arm},
            object_pose=object_pose,
            reference_pose=None,
            target_object_pose=None,
            motion_policy={"sample_interval": 45},
        )
        for arm in assignments
    )
    planned_action = np.zeros((2, 45, 3), dtype=np.float32)
    arm_actions_by_env = [
        {"left": SimpleNamespace(action=planned_action), "right": None},
        {"left": None, "right": SimpleNamespace(action=planned_action)},
    ]

    recorder.record_edge(
        step_data["edge_ids"][0],
        assignments=assignments,
        grounded_actions=grounded_actions,
        failed_before=torch.tensor([False, False]),
        failed_after=torch.tensor([False, False]),
        grounding_failed=torch.tensor([False, False]),
        action_steps=45,
        arm_actions=arm_actions_by_env,
    )

    for env_id, expected_arm in enumerate(assignments):
        runtime = recorder.documents[env_id]["edges"][0]["actions"][0]["runtime"]
        assert runtime["assigned_arm"] == expected_arm
        assert runtime["planning"]["status"] == "planned"
        assert runtime["planning"]["trajectory_step_count"] == 45


def test_recorder_finalizes_aborted_episode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from embodichain.gen_sim.action_agent_pipeline.runtime import task_graph_artifact

    seed = make_relative_seed_task_graph("aborted_recording", _relative_spec())
    env = SimpleNamespace(num_envs=1, agent_robot_profile="dual_franka")
    monkeypatch.setattr(task_graph_artifact, "_outputs_root", lambda: tmp_path)
    recorder = RuntimeTaskGraphRecorder(
        seed,
        env=env,
        run_id="aborted_run",
        episode_index=0,
    )

    recorder.finalize(None, aborted_reason="RuntimeError: injected")

    env_dir = (
        tmp_path
        / "aborted_recording"
        / "runs"
        / "aborted_run"
        / "episode_0000"
        / "env_0000"
    )
    document = json.loads((env_dir / "task_graph.json").read_text(encoding="utf-8"))
    assert document["status"] == "aborted"
    assert document["failure_reason"] == "RuntimeError: injected"
    assert (env_dir / "task_graph.png").is_file()


def _relative_spec(
    *,
    release_position: list[float] | None = None,
    two_steps: bool = False,
    first_arm: str = "left",
):
    def placement(step_id: str, relation: str, arm: str, depends_on=()):
        return SimpleNamespace(
            intent="place_relative",
            moved_runtime_uid="object_a",
            reference_runtime_uid="object_c",
            relation=relation,
            reference_is_initial_pose=False,
            orientation_goal="preserve",
            orientation_axis="none",
            orientation_align_to_runtime_uid=None,
            arm_request=arm,
            step_id=step_id,
            depends_on=depends_on,
            release_position=release_position,
        )

    placements = [placement("s01_left", "left_of", first_arm)]
    if two_steps:
        placements.append(placement("s02_right", "right_of", "right", ("s01_left",)))
    return SimpleNamespace(
        intent="place_relative",
        placements=tuple(placements),
        coordinated_direction=None,
        coordinated_terminal_behavior=None,
    )


def _parallel_relative_spec(*, auto: bool = False):
    def placement(
        step_id: str,
        object_uid: str,
        arm: str,
        depends_on=(),
    ):
        return SimpleNamespace(
            intent="place_relative",
            moved_runtime_uid=object_uid,
            reference_runtime_uid="basket",
            relation="inside",
            reference_is_initial_pose=False,
            orientation_goal="preserve",
            orientation_axis="none",
            orientation_align_to_runtime_uid=None,
            arm_request="auto" if auto else arm,
            step_id=step_id,
            depends_on=depends_on,
        )

    return SimpleNamespace(
        intent="place_relative",
        task_description="Use both arms to move the two objects into the basket.",
        placements=(
            placement("s01_cube_inside", "cube", "left"),
            placement(
                "s02_cup_inside",
                "cup",
                "right",
                ("s01_cube_inside",),
            ),
        ),
        coordinated_direction=None,
        coordinated_terminal_behavior=None,
    )
