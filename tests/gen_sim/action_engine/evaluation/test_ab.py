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
import json
from types import SimpleNamespace

import pytest
import torch

from embodichain.gen_sim.action_engine.evaluation import run_strict_ab, state_digest
from embodichain.gen_sim.action_engine.evaluation.ab import _graph_difference
from embodichain.gen_sim.action_engine.tasks import instantiate_seed_graph

from ..task_fixtures import make_task_level, make_task_spec


class _Env:
    def __init__(self, route: str, seed: int, config: dict) -> None:
        self.route = route
        self.seed = seed
        self.config = config
        self.closed = False

    def reset(self, *, seed: int) -> None:
        self.seed = seed

    def close(self) -> None:
        self.closed = True


class _Executor:
    def __init__(self, graph: dict, env: _Env) -> None:
        self.graph = graph
        self.env = env

    def run(self, **_kwargs):
        return SimpleNamespace(
            success=torch.tensor([True]),
            actions=[torch.tensor([[0.0]]), torch.tensor([[1.0]])],
            retry_count=0,
            recovery_count=0,
            revision_count=0,
            runtime_revisions=[],
            record_dir=f"records/{self.env.route}",
        )


def _inputs():
    task, requirements = make_task_spec("E1")
    bindings = {
        item["role_id"]: f"uid_{item['role_id']}" for item in requirements["objects"]
    }
    offline = instantiate_seed_graph(task, bindings)
    online = deepcopy(offline)
    online["planner_route"] = "online"
    return task, offline, online


def test_state_digest_is_mapping_order_stable() -> None:
    assert state_digest(
        {"qpos": torch.tensor([1.0]), "objects": {"a": [2.0]}}
    ) == state_digest({"objects": {"a": [2.0]}, "qpos": torch.tensor([1.0])})


def test_strict_ab_writes_isolated_branches_and_comparison(tmp_path) -> None:
    task, offline, online = _inputs()
    created = []

    def env_factory(**kwargs):
        env = _Env(**kwargs)
        created.append(env)
        return env

    result = run_strict_ab(
        task,
        offline,
        online,
        env_factory=env_factory,
        executor_factory=lambda graph, env: _Executor(graph, env),
        snapshot_reader=lambda env: {
            "robot_qpos": torch.tensor([float(env.seed)]),
            "object_poses": {"object": torch.eye(4)},
        },
        output_dir=tmp_path,
        seed=123,
        shared_config={"robot": "same"},
        planning_metrics={
            "offline": {"planning_seconds": 0.1, "vlm_call_count": 0},
            "online": {"planning_seconds": 0.2, "vlm_call_count": 2},
        },
    )

    assert result.comparison_path.is_file()
    assert (result.offline_dir / "seed_task_graph.json").is_file()
    assert (result.online_dir / "seed_task_graph.json").is_file()
    assert result.comparison["initial_state_digest"] == result.initial_state_digest
    assert result.comparison["branches"]["offline"]["planning_seconds"] == 0.1
    assert result.comparison["branches"]["online"]["vlm_call_count"] == 2
    assert all(env.closed for env in created)


def test_graph_difference_reports_changed_task_group_fields() -> None:
    _task_spec, offline, online = _inputs()
    online["task_groups"][0]["goal"] = deepcopy(online["task_groups"][0]["goal"])
    online["task_groups"][0]["goal"]["relation"] = "right_of"

    difference = _graph_difference(offline, online)

    assert difference["changed_task_groups"] == [
        {"id": offline["task_groups"][0]["id"], "changed_fields": ["goal"]}
    ]
    assert (
        difference["task_group_difference"]["changed_groups"]
        == difference["changed_task_groups"]
    )


def test_strict_ab_finalizes_two_branch_videos_and_revision_files(tmp_path) -> None:
    task, offline, online = _inputs()
    finalized = []

    def finalizer(**kwargs):
        route = kwargs["route"]
        path = kwargs["branch_dir"] / "video.mp4"
        path.write_bytes(route.encode("ascii"))
        finalized.append(route)
        return [path.as_posix()]

    result = run_strict_ab(
        task,
        offline,
        online,
        env_factory=lambda **kwargs: _Env(**kwargs),
        executor_factory=lambda graph, env: _Executor(graph, env),
        snapshot_reader=lambda env: {
            "robot_qpos": torch.tensor([float(env.seed)]),
            "object_poses": {"object": torch.eye(4)},
        },
        branch_finalizer=finalizer,
        output_dir=tmp_path,
        seed=11,
    )

    assert finalized == ["offline", "online"]
    for route, branch_dir in (
        ("offline", result.offline_dir),
        ("online", result.online_dir),
    ):
        assert (branch_dir / "video.mp4").read_bytes() == route.encode("ascii")
        assert (branch_dir / "runtime_revisions.json").is_file()
        assert result.comparison["branches"][route]["video_paths"] == [
            (branch_dir / "video.mp4").as_posix()
        ]


def test_strict_ab_aborts_before_execution_on_state_mismatch(tmp_path) -> None:
    task, offline, online = _inputs()
    executions = []

    with pytest.raises(RuntimeError, match="initial state mismatch"):
        run_strict_ab(
            task,
            offline,
            online,
            env_factory=lambda **kwargs: _Env(**kwargs),
            executor_factory=lambda graph, env: executions.append((graph, env)),
            snapshot_reader=lambda env: {
                "robot_qpos": torch.tensor([float(env.seed)]),
                "object_poses": {env.route: torch.eye(4)},
            },
            output_dir=tmp_path,
            seed=5,
        )
    assert executions == []


def test_strict_ab_rejects_incomplete_state_snapshot(tmp_path) -> None:
    task, offline, online = _inputs()

    with pytest.raises(ValueError, match="missing required state"):
        run_strict_ab(
            task,
            offline,
            online,
            env_factory=lambda **kwargs: _Env(**kwargs),
            executor_factory=lambda graph, env: _Executor(graph, env),
            snapshot_reader=lambda env: {"robot_qpos": torch.tensor([0.0])},
            output_dir=tmp_path,
            seed=5,
        )


def test_strict_ab_requires_articulation_and_camera_digest_components(tmp_path) -> None:
    task, offline, online = _inputs()

    with pytest.raises(ValueError, match="articulation_state.*camera_calibration"):
        run_strict_ab(
            task,
            offline,
            online,
            env_factory=lambda **kwargs: _Env(**kwargs),
            executor_factory=lambda graph, env: _Executor(graph, env),
            snapshot_reader=lambda env: {
                "robot_qpos": torch.tensor([float(env.seed)]),
                "object_poses": {"object": torch.eye(4)},
            },
            output_dir=tmp_path,
            seed=5,
            strict_state_digest=True,
        )


def test_strict_ab_reuses_prepared_identical_resets(tmp_path) -> None:
    task, offline, online = _inputs()
    environments = {
        route: _Env(route=route, seed=17, config={}) for route in ("offline", "online")
    }
    snapshots = {
        route: {
            "robot_qpos": torch.tensor([17.0]),
            "object_poses": {"object": torch.eye(4)},
        }
        for route in environments
    }

    result = run_strict_ab(
        task,
        offline,
        online,
        executor_factory=lambda graph, env: _Executor(graph, env),
        snapshot_reader=lambda _env: pytest.fail("prepared snapshots must be reused"),
        output_dir=tmp_path,
        seed=17,
        prepared_environments=environments,
        prepared_snapshots=snapshots,
    )

    assert result.initial_state_digest == state_digest(snapshots["offline"])
    assert all(env.closed for env in environments.values())


def test_strict_ab_stops_both_branches_when_global_preflight_fails(tmp_path) -> None:
    task, offline, online = _inputs()
    runs: list[str] = []

    class PreflightExecutor(_Executor):
        def preflight(self) -> bool:
            if self.env.route == "online":
                raise ValueError("online capability unavailable")
            return True

        def run(self, **kwargs):
            runs.append(self.env.route)
            return super().run(**kwargs)

    with pytest.raises(RuntimeError, match="no branch was allowed to move"):
        run_strict_ab(
            task,
            offline,
            online,
            env_factory=lambda **kwargs: _Env(**kwargs),
            executor_factory=lambda graph, env: PreflightExecutor(graph, env),
            snapshot_reader=lambda env: {
                "robot_qpos": torch.tensor([float(env.seed)]),
                "object_poses": {"object": torch.eye(4)},
            },
            output_dir=tmp_path,
            seed=17,
        )
    assert runs == []


def test_strict_ab_keeps_other_branch_running_after_execution_failure(tmp_path) -> None:
    task, offline, online = _inputs()
    runs: list[str] = []

    class IsolatedExecutor(_Executor):
        def preflight(self) -> bool:
            return True

        def run(self, **kwargs):
            runs.append(self.env.route)
            if self.env.route == "offline":
                raise RuntimeError("offline execution failed")
            return super().run(**kwargs)

    result = run_strict_ab(
        task,
        offline,
        online,
        env_factory=lambda **kwargs: _Env(**kwargs),
        executor_factory=lambda graph, env: IsolatedExecutor(graph, env),
        snapshot_reader=lambda env: {
            "robot_qpos": torch.tensor([float(env.seed)]),
            "object_poses": {"object": torch.eye(4)},
        },
        output_dir=tmp_path,
        seed=17,
    )

    assert runs == ["offline", "online"]
    assert result.comparison["branches"]["offline"]["success_rate"] == 0.0
    assert result.comparison["branches"]["online"]["success_rate"] == 1.0


def test_strict_ab_surfaces_video_finalizer_failure_and_closes(tmp_path) -> None:
    task, offline, online = _inputs()
    environments = []

    def factory(**kwargs):
        environment = _Env(**kwargs)
        environments.append(environment)
        return environment

    def finalizer(**kwargs):
        if kwargs["route"] == "offline":
            raise OSError("recorder did not produce a file")
        return []

    with pytest.raises(RuntimeError, match="branch video finalization failed"):
        run_strict_ab(
            task,
            offline,
            online,
            env_factory=factory,
            executor_factory=lambda graph, env: _Executor(graph, env),
            snapshot_reader=lambda env: {
                "robot_qpos": torch.tensor([float(env.seed)]),
                "object_poses": {"object": torch.eye(4)},
            },
            branch_finalizer=finalizer,
            output_dir=tmp_path,
            seed=19,
        )

    assert len(environments) == 2
    assert all(environment.closed for environment in environments)
    comparison = json.loads((tmp_path / "comparison.json").read_text())
    assert set(comparison["video_finalization_errors"]) == {"offline"}


def test_strict_ab_require_branch_videos_checks_normalized_artifact(tmp_path) -> None:
    task, offline, online = _inputs()

    with pytest.raises(RuntimeError, match="video.mp4"):
        run_strict_ab(
            task,
            offline,
            online,
            env_factory=lambda **kwargs: _Env(**kwargs),
            executor_factory=lambda graph, env: _Executor(graph, env),
            snapshot_reader=lambda env: {
                "robot_qpos": torch.tensor([float(env.seed)]),
                "object_poses": {"object": torch.eye(4)},
            },
            branch_finalizer=lambda **_kwargs: [],
            output_dir=tmp_path,
            seed=19,
            require_branch_videos=True,
        )


def test_strict_ab_closes_prepared_environments_on_graph_validation_error(
    tmp_path,
) -> None:
    task, offline, online = _inputs()
    environments = {
        route: _Env(route=route, seed=23, config={}) for route in ("offline", "online")
    }
    invalid_online = deepcopy(online)
    invalid_online["planner_route"] = "offline"

    with pytest.raises(ValueError, match="explicit offline and online"):
        run_strict_ab(
            task,
            offline,
            invalid_online,
            executor_factory=lambda graph, env: _Executor(graph, env),
            snapshot_reader=lambda _env: pytest.fail("validation must happen first"),
            output_dir=tmp_path,
            seed=23,
            prepared_environments=environments,
        )

    assert all(environment.closed for environment in environments.values())


def test_strict_l4_ab_requires_and_records_private_oracle(tmp_path) -> None:
    task, requirements = make_task_level("L4")
    bindings = {
        item["role_id"]: f"uid_{item['role_id']}" for item in requirements["objects"]
    }
    offline = instantiate_seed_graph(task, bindings)
    online = deepcopy(offline)
    online["planner_route"] = "online"

    with pytest.raises(ValueError, match="private-oracle"):
        run_strict_ab(
            task,
            offline,
            online,
            env_factory=lambda **kwargs: _Env(**kwargs),
            executor_factory=lambda graph, env: _Executor(graph, env),
            snapshot_reader=lambda env: {
                "robot_qpos": torch.tensor([float(env.seed)]),
                "object_poses": {"object": torch.eye(4)},
            },
            output_dir=tmp_path,
            seed=7,
        )

    result = run_strict_ab(
        task,
        offline,
        online,
        env_factory=lambda **kwargs: _Env(**kwargs),
        executor_factory=lambda graph, env: _Executor(graph, env),
        snapshot_reader=lambda env: {
            "robot_qpos": torch.tensor([float(env.seed)]),
            "object_poses": {"object": torch.eye(4)},
        },
        success_evaluator=lambda **_kwargs: torch.tensor([True]),
        output_dir=tmp_path,
        seed=7,
    )
    assert all(
        branch["success_source"] == "private_oracle"
        for branch in result.comparison["branches"].values()
    )
