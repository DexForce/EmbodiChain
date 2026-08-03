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

from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch

from embodichain.gen_sim.action_agent_pipeline.runtime.graph_compiler import (
    compile_agent_graph_spec,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.symbolic_grounding import (
    ground_symbolic_action,
)
from embodichain.gen_sim.action_agent_pipeline.runtime import (
    task_graph as task_graph_module,
)
from embodichain.gen_sim.action_engine.cli.run_agent import build_parser
from embodichain.gen_sim.action_engine.compiler import compile_task_agent
from embodichain.gen_sim.action_engine.protocol import TASK_AGENT_SCHEMA
from embodichain.gen_sim.action_engine.runtime import (
    load_execution_program,
    lower_to_pipeline_seed,
)


class _Object:
    def __init__(
        self,
        half_extents: tuple[float, float, float],
        *,
        rotation: torch.Tensor | None = None,
    ) -> None:
        x, y, z = half_extents
        self.vertices = torch.tensor(
            [[-x, -y, -z], [x, y, z]],
            dtype=torch.float32,
        )
        self.pose = torch.eye(4, dtype=torch.float32)
        if rotation is not None:
            self.pose[:3, :3] = rotation

    def get_vertices(self, *, env_ids, scale):
        return self.vertices.unsqueeze(0)

    def get_local_pose(self, *, to_matrix):
        return self.pose.unsqueeze(0)


class _Sim:
    def __init__(self, objects: dict[str, _Object]) -> None:
        self.objects = objects

    def get_rigid_object(self, uid: str) -> _Object | None:
        return self.objects.get(uid)


def _env(**objects: _Object) -> SimpleNamespace:
    return SimpleNamespace(
        device=torch.device("cpu"),
        sim=_Sim(objects),
    )


def _program(*steps: dict) -> dict:
    return {
        "schema_version": TASK_AGENT_SCHEMA,
        "task": "pipeline_parity",
        "goal": "Verify pipeline parity.",
        "semantic_steps": list(steps),
        "allocation_groups": [],
    }


def test_orient_program_lowers_to_valid_mature_pipeline_graph() -> None:
    execution = load_execution_program(
        compile_task_agent(
            _program(
                {
                    "id": "s01_tea",
                    "operator": "orient_object",
                    "object": "tea",
                    "goal": {"orientation_goal": "upright"},
                },
                {
                    "id": "s02_can",
                    "operator": "orient_object",
                    "object": "can",
                    "goal": {"orientation_goal": "upright"},
                },
            )
        )
    )
    env = _env(
        tea=_Object((0.03, 0.03, 0.12)),
        can=_Object((0.04, 0.04, 0.08)),
        table=_Object((0.6, 0.4, 0.03)),
    )
    env.agent_robot_profile = "dual_franka"
    seed = lower_to_pipeline_seed(execution, env=env)

    graph = compile_agent_graph_spec(seed, task_name="pipeline_parity")

    assert seed["route"] == "object_manipulation"
    assert [step["operator"] for step in seed["semantic_steps"]] == [
        "place_relative",
        "place_relative",
    ]
    assert [step["goal"]["upright_local_axis"] for step in seed["semantic_steps"]] == [
        "z",
        "z",
    ]
    assert [edge["actions"][0]["atomic_action_class"] for edge in seed["edges"]] == [
        "PickUp",
        "MoveHeldObject",
        "MoveHeldObject",
        "Place",
        "MoveEndEffector",
        "MoveJoints",
    ] * 2
    assert len(graph.edges) == len(execution.edges)

    first_step = graph.semantic_steps["s01_tea"]
    pickup = ground_symbolic_action(
        graph.edges[first_step.edge_ids[0]].symbolic_actions[0],
        first_step,
        env=env,
        arm="left_arm",
    )
    assert pickup.action_spec["cfg"]["obj_upright_direction"] == [0.0, 0.0, 1.0]
    assert pickup.action_spec["cfg"]["rotate_upright"] == pytest.approx(torch.pi / 4)


def test_arrangement_resolves_table_axis_and_strips_engine_actor_metadata() -> None:
    execution = load_execution_program(
        compile_task_agent(
            _program(
                {
                    "id": "line",
                    "operator": "arrange_line",
                    "objects": ["a", "b"],
                    "goal": {
                        "axis": "table_long_axis",
                        "participation": "both_arms",
                    },
                }
            )
        )
    )
    seed = lower_to_pipeline_seed(
        execution,
        env=_env(
            a=_Object((0.03, 0.03, 0.04)),
            b=_Object((0.03, 0.03, 0.04)),
            table=_Object((0.4, 0.7, 0.03)),
        ),
    )

    assert seed["route"] == "arrangement_line"
    assert {step["goal"]["axis"] for step in seed["semantic_steps"]} == {"world_y"}
    assert all(step["actor"] == {"mode": "auto"} for step in seed["semantic_steps"])
    assert all(
        edge["actions"][0]["actor"] == {"mode": "auto"} for edge in seed["edges"]
    )


def test_pipeline_backend_rejects_cross_route_composition() -> None:
    execution = load_execution_program(
        compile_task_agent(
            _program(
                {
                    "id": "line",
                    "operator": "arrange_line",
                    "objects": ["a", "b"],
                    "goal": {},
                },
                {
                    "id": "place",
                    "operator": "place_relative",
                    "object": "c",
                    "goal": {"reference_object": "table", "relation": "on"},
                    "depends_on": ["line"],
                },
            )
        )
    )

    with pytest.raises(ValueError, match="one runtime route family"):
        lower_to_pipeline_seed(execution)


def test_pipeline_backend_is_cli_default() -> None:
    args = build_parser().parse_args(
        [
            "--task_name",
            "task",
            "--gym_config",
            "gym.json",
            "--agent_config",
            "agent.json",
        ]
    )

    assert args.runtime_backend == "pipeline"


def test_independent_semantic_step_runs_after_sibling_selection_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution = load_execution_program(
        compile_task_agent(
            _program(
                {
                    "id": "first",
                    "operator": "orient_object",
                    "object": "a",
                    "goal": {
                        "orientation_goal": "upright",
                        "upright_local_axis": "z",
                    },
                },
                {
                    "id": "second",
                    "operator": "orient_object",
                    "object": "b",
                    "goal": {
                        "orientation_goal": "upright",
                        "upright_local_axis": "z",
                    },
                },
            )
        )
    )
    graph = compile_agent_graph_spec(lower_to_pipeline_seed(execution))
    selected_steps: list[str] = []

    def select_step_arms(self, step, *, failed, **_kwargs):
        selected_steps.append(step.id)
        selection_failed = torch.tensor(
            [step.id == "first"],
            dtype=torch.bool,
        )
        assignments = [None if bool(selection_failed[0]) else "left_arm"]
        assert not bool(failed[0])
        return assignments, selection_failed

    def execute_symbolic_edge(self, _edge, _step, *, failed, world_states, **_kwargs):
        return (
            {
                "actions": [],
                "world_states": world_states,
                "failed_env_mask": failed.clone(),
                "arm_actions": {},
                "arm_actions_by_env": [{}],
            },
            (
                SimpleNamespace(
                    target_object_pose=None,
                    motion_policy={},
                ),
            ),
        )

    def complete_semantic_step(self, _step, *, failed, **_kwargs):
        return failed, ~failed, torch.zeros((1, 3)), None, None

    class Recorder:
        output_dir = Path("runtime-record")

        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def begin_step(self, *_args, **_kwargs) -> None:
            pass

        def record_edge(self, *_args, **_kwargs) -> None:
            pass

        def complete_step(self, *_args, **_kwargs) -> None:
            pass

        def finalize(self, *_args, **_kwargs) -> None:
            pass

    grounded = SimpleNamespace(
        object_pose=None,
        reference_pose=None,
        target_object_pose=None,
        motion_policy={},
    )
    graph._select_step_arms = MethodType(select_step_arms, graph)
    graph._execute_symbolic_edge = MethodType(execute_symbolic_edge, graph)
    graph._complete_semantic_step = MethodType(complete_semantic_step, graph)
    monkeypatch.setattr(task_graph_module, "RuntimeTaskGraphRecorder", Recorder)
    monkeypatch.setattr(
        task_graph_module,
        "init_parallel_world_states",
        lambda _env: {"left": None, "right": None},
    )
    monkeypatch.setattr(
        task_graph_module,
        "ground_symbolic_action",
        lambda *_args, **_kwargs: grounded,
    )

    result = graph.run(
        env=SimpleNamespace(num_envs=1, device=torch.device("cpu")),
        semantic_step_settle_steps=0,
    )

    assert selected_steps == ["first", "second"]
    assert not bool(result.semantic_step_success["first"][0])
    assert bool(result.semantic_step_success["second"][0])
    assert not bool(result.runtime_success[0])
