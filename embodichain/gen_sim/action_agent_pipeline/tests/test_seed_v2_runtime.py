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
from embodichain.gen_sim.action_agent_pipeline.generation.prompt_builders import (
    make_agent_config,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.motion_policy import (
    resolve_motion_policy,
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
from embodichain.lab.sim.atomic_actions import WorldState


class _Object:
    def __init__(self, positions: list[list[float]]) -> None:
        self.pose = torch.eye(4).repeat(len(positions), 1, 1)
        self.pose[:, :3, 3] = torch.tensor(positions)

    def get_local_pose(self, *, to_matrix: bool) -> torch.Tensor:
        assert to_matrix
        return self.pose


class _Sim:
    def __init__(self, objects: dict[str, _Object]) -> None:
        self.objects = objects

    def get_rigid_object(self, uid: str):
        return self.objects.get(uid)


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
    assert [step["goal"]["slot_index"] for step in arrangement["semantic_steps"]] == [
        0,
        1,
        2,
    ]
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
