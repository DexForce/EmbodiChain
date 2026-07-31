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

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from embodichain.gen_sim.action_agent_pipeline.agents.compile_agent import CompileAgent
from embodichain.gen_sim.action_agent_pipeline.env_adapters.tableware.agent_env import (
    AgenticGenSimEnv,
)
from embodichain.gen_sim.action_agent_pipeline.generation.seed_task_graph import (
    make_relative_seed_task_graph,
)
from embodichain.gen_sim.action_agent_pipeline.runtime import task_graph
from embodichain.gen_sim.action_agent_pipeline.runtime.graph_compiler import (
    compile_agent_graph_from_file,
    compile_agent_graph_spec,
    load_agent_graph_bundle,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.success_evaluator import (
    _object_lifted,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.task_graph import (
    AgentTaskGraph,
    _execution_kwargs,
)


class _RigidObject:
    def __init__(self, heights: list[float]) -> None:
        self.pose = torch.eye(4).repeat(len(heights), 1, 1)
        self.set_heights(heights)

    def set_heights(self, heights: list[float]) -> None:
        self.pose[:, 2, 3] = torch.tensor(heights, dtype=torch.float32)

    def get_local_pose(self, *, to_matrix: bool) -> torch.Tensor:
        assert to_matrix
        return self.pose


class _ObjectSim:
    def __init__(self, objects: dict[str, _RigidObject]) -> None:
        self.objects = objects

    def get_rigid_object_uid_list(self) -> list[str]:
        return list(self.objects)

    def get_rigid_object(self, uid: str) -> _RigidObject:
        return self.objects[uid]


def test_seed_files_require_strict_json(tmp_path: Path) -> None:
    path = tmp_path / "seed_task_graph.json"
    path.write_text('```json\n{"task": "wrapped"}\n```', encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        load_agent_graph_bundle(path)


def test_graph_compiler_enforces_expected_task_name(tmp_path: Path) -> None:
    path = tmp_path / "seed_task_graph.json"
    seed = make_relative_seed_task_graph("actual", _parallel_spec())
    path.write_text(json.dumps(seed), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match"):
        compile_agent_graph_from_file(path, task_name="expected")


def test_removed_action_module_is_not_a_compiler_parameter() -> None:
    assert "action_module" not in inspect.signature(compile_agent_graph_spec).parameters
    assert (
        "action_module"
        not in inspect.signature(compile_agent_graph_from_file).parameters
    )


def test_compile_agent_rejects_llm_wrapped_seed_json() -> None:
    agent = CompileAgent(task_name="wrapped")

    with pytest.raises(json.JSONDecodeError):
        agent.generate(seed_task_graph='```json\n{"task": "wrapped"}\n```')


def test_execution_kwargs_do_not_leak_scheduler_controls() -> None:
    renderer = object()
    filtered = _execution_kwargs(
        {
            "runtime_graph_renderer": renderer,
            "runtime_run_id": "run",
            "strict_serial": True,
            "semantic_step_settle_steps": 0,
            "allow_grasp_annotation": True,
        }
    )

    assert filtered == {"allow_grasp_annotation": True}


def test_object_height_snapshot_is_not_overwritten_by_live_updates() -> None:
    rigid_object = _RigidObject([0.10, 0.20])
    env = SimpleNamespace(
        sim=_ObjectSim({"can": rigid_object}),
        obj_info={},
    )
    AgenticGenSimEnv.update_obj_info(
        env,
        reset_ids=torch.tensor([0, 1]),
        capture_initial=True,
    )
    rigid_object.set_heights([0.35, 0.45])

    AgenticGenSimEnv.update_obj_info(env)

    assert env.obj_info["can"]["height"].tolist() == pytest.approx([0.10, 0.20])
    assert env.obj_info["can"]["current_height"].tolist() == pytest.approx([0.35, 0.45])


def test_partial_reset_updates_only_selected_initial_height() -> None:
    rigid_object = _RigidObject([0.10, 0.20])
    env = SimpleNamespace(
        sim=_ObjectSim({"can": rigid_object}),
        obj_info={},
    )
    AgenticGenSimEnv.update_obj_info(
        env,
        reset_ids=torch.tensor([0, 1]),
        capture_initial=True,
    )
    rigid_object.set_heights([0.30, 0.40])

    AgenticGenSimEnv.update_obj_info(
        env,
        reset_ids=torch.tensor([0]),
        capture_initial=True,
    )

    assert env.obj_info["can"]["height"].tolist() == pytest.approx([0.30, 0.20])
    assert env.obj_info["can"]["current_height"].tolist() == pytest.approx([0.30, 0.40])


def test_lift_success_uses_explicit_initial_height_snapshot() -> None:
    rigid_object = _RigidObject([0.25, 0.25])
    env = SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        sim=_ObjectSim({"can": rigid_object}),
        agent_initial_object_heights={"can": torch.tensor([0.10, 0.20])},
        obj_info={"can": {"height": torch.tensor([0.25, 0.25])}},
    )

    success = _object_lifted(
        env,
        {"type": "object_lifted", "object": "can", "min_height": 0.10},
    )

    assert success.tolist() == [True, False]


def test_semantic_completion_reads_declared_stack_postcondition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_spec = {}

    def evaluate(env, spec):
        captured_spec.update(spec)
        return torch.tensor([True])

    monkeypatch.setattr(task_graph, "evaluate_configured_success", evaluate)
    graph = object.__new__(AgentTaskGraph)
    env = SimpleNamespace(
        sim=_ObjectSim({"item": _RigidObject([0.20])}),
    )
    step = SimpleNamespace(
        id="s01",
        object_uid="item",
        goal={"relation": "on", "reference_object": "wrong_support"},
        postcondition={
            "type": "stack_layer_supported",
            "reference_object": "declared_support",
        },
    )

    graph._complete_semantic_step(
        step,
        env=env,
        failed=torch.tensor([False]),
        target_positions=None,
        motion_policy=None,
        settle_steps=0,
    )

    assert captured_spec["support"] == "declared_support"


def test_joint_auto_assignment_records_both_step_starts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = compile_agent_graph_spec(
        make_relative_seed_task_graph("parallel_recording", _parallel_spec())
    )
    recorded_starts = []

    class Recorder:
        output_dir = tmp_path

        def __init__(self, *args, **kwargs) -> None:
            pass

        def begin_step(self, step, **kwargs) -> None:
            recorded_starts.append((step.id, list(kwargs["assignments"])))

        def record_edge(self, *args, **kwargs) -> None:
            pass

        def complete_step(self, *args, **kwargs) -> None:
            pass

        def finalize(self, *args, **kwargs) -> None:
            pass

    grounded = SimpleNamespace(
        action_spec={"robot_name": "left_arm"},
        object_pose=torch.eye(4).unsqueeze(0),
        reference_pose=None,
        target_object_pose=None,
        motion_policy={},
    )

    def select_parallel(edges, *, failed, **kwargs):
        steps = [graph.semantic_step_by_edge[edge.id] for edge in edges]
        return {
            steps[0].id: ["left_arm"],
            steps[1].id: ["right_arm"],
        }, torch.zeros_like(failed)

    def execute_parallel(edges, *, failed, world_states, **kwargs):
        return (
            {
                "actions": [],
                "world_states": dict(world_states),
                "arm_actions": {},
                "failed_env_mask": failed.clone(),
            },
            {edge.id: (grounded,) for edge in edges},
        )

    def execute_serial(edge, step, *, failed, world_states, **kwargs):
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
        return failed, ~failed, torch.zeros((1, 3)), None, None

    monkeypatch.setattr(task_graph, "RuntimeTaskGraphRecorder", Recorder)
    monkeypatch.setattr(task_graph, "init_parallel_world_states", lambda env: {})
    monkeypatch.setattr(
        task_graph, "ground_symbolic_action", lambda *args, **kwargs: grounded
    )
    monkeypatch.setattr(graph, "_select_parallel_pickup_arms", select_parallel)
    monkeypatch.setattr(graph, "_execute_parallel_pickup_edges", execute_parallel)
    monkeypatch.setattr(graph, "_execute_symbolic_edge", execute_serial)
    monkeypatch.setattr(graph, "_complete_semantic_step", complete)
    env = SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        agent_robot_profile="dual_franka",
    )

    graph.run(env=env, semantic_step_settle_steps=0)

    expected_steps = list(graph.semantic_steps)
    assert [step_id for step_id, _ in recorded_starts] == expected_steps
    assert recorded_starts[0][1] == ["left_arm"]
    assert recorded_starts[1][1] == ["right_arm"]


def _parallel_spec() -> SimpleNamespace:
    def placement(
        step_id: str,
        object_uid: str,
        depends_on: tuple[str, ...] = (),
    ) -> SimpleNamespace:
        return SimpleNamespace(
            intent="place_relative",
            moved_runtime_uid=object_uid,
            reference_runtime_uid="basket",
            relation="inside",
            reference_is_initial_pose=False,
            orientation_goal="preserve",
            orientation_axis="none",
            orientation_align_to_runtime_uid=None,
            arm_request="auto",
            step_id=step_id,
            depends_on=depends_on,
        )

    return SimpleNamespace(
        intent="place_relative",
        task_description="Move both objects into the basket.",
        parallel_pickup_requested=True,
        placements=(
            placement("s01_cube_inside", "cube"),
            placement("s02_cup_inside", "cup", ("s01_cube_inside",)),
        ),
        coordinated_direction=None,
        coordinated_terminal_behavior=None,
    )
