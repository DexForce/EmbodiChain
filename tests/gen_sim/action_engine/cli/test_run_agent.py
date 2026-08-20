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

from embodichain.gen_sim.action_engine.cli.run_agent import (
    _ABWorkerConfig,
    _SerializedABBranch,
    _capture_ab_initial_frame,
    _prepare_ab_branches,
    _publish_task_engine_report,
    _task_engine_exit_code,
)
from embodichain.gen_sim.action_engine.runtime import (
    ExecutionReport,
    build_execution_provenance,
)


class record_camera_data:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, *args, **kwargs) -> None:
        self.calls.append((args, kwargs))


class _FakeEnv:
    def __init__(self, recorder=None) -> None:
        self.unwrapped = self
        self.event_manager = SimpleNamespace(
            _mode_functor_cfgs={
                "interval": (
                    [
                        SimpleNamespace(
                            func=recorder,
                            params={"name": "record_cam_audience_view"},
                        )
                    ]
                    if recorder is not None
                    else []
                )
            }
        )


def test_capture_ab_initial_frame_invokes_only_audience_recorder() -> None:
    recorder = record_camera_data()
    env = _FakeEnv(recorder)

    _capture_ab_initial_frame(env)

    assert len(recorder.calls) == 1
    args, kwargs = recorder.calls[0]
    assert args == (env, None)
    assert kwargs == {"name": "record_cam_audience_view"}


def test_capture_ab_initial_frame_requires_audience_recorder() -> None:
    with pytest.raises(RuntimeError, match="audience recorder"):
        _capture_ab_initial_frame(_FakeEnv())


def _worker_config(route: str) -> _ABWorkerConfig:
    return _ABWorkerConfig(
        route=route,
        gym_config={},
        env_options={},
        gym_id="ActionEngine-v1",
        agent_config={},
        agent_config_path="agent_config.json",
        task_name="smoke",
        runtime_backend="independent",
        seed=7,
        camera_uids=("vlm_front",),
        staging_dir=f"/tmp/ab/{route}/video",
    )


class _MemoryAwareFakeWorker:
    instances = []

    def __init__(self, config: _ABWorkerConfig) -> None:
        self.config = config
        self.closed = False
        self.startup_snapshot = {
            "robot_qpos": [0.0, 1.0],
            "object_poses": {"object": [0.0, 0.0, 0.0]},
        }
        self.startup_observation = {"route": config.route}
        self.events = []
        self.instances.append(self)
        if (
            config.route == "online"
            and Path(config.staging_dir).parent.name == config.route
        ):
            raise RuntimeError("CUDA out of memory")

    def preflight(self, graph):
        self.events.append(("preflight", graph))
        return True

    def run(self, graph, **kwargs):
        self.events.append(("run", graph, kwargs))
        return SimpleNamespace(success=True)

    def finalize(self, branch_dir: Path, *, episode_index: int):
        self.events.append(("finalize", branch_dir, episode_index))
        return [(branch_dir / "video.mp4").as_posix()]

    def close(self):
        self.closed = True


def test_ab_serializes_workers_after_startup_oom() -> None:
    _MemoryAwareFakeWorker.instances = []
    branches, snapshots = _prepare_ab_branches(
        {"offline": _worker_config("offline"), "online": _worker_config("online")},
        worker_factory=_MemoryAwareFakeWorker,
        prefer_serial=False,
    )

    assert set(branches) == {"offline", "online"}
    assert all(isinstance(branch, _SerializedABBranch) for branch in branches.values())
    assert snapshots["offline"] == snapshots["online"]
    for route, branch in branches.items():
        assert branch.preflight({"route": route}) is True
        branch.run(
            {"route": route},
            run_id=f"run-{route}",
            episode_index=0,
            record_root=Path("/tmp/ab/runtime"),
        )
        assert branch.finalize(Path(f"/tmp/ab/{route}"), episode_index=0) == [
            f"/tmp/ab/{route}/video.mp4"
        ]
        branch.close()

    phases = [
        Path(worker.config.staging_dir).parent.name
        for worker in _MemoryAwareFakeWorker.instances
        if worker.config.route == "offline"
    ]
    assert phases == ["offline", "probe", "preflight", "execute"]


@pytest.mark.parametrize(
    ("status", "success"),
    [("succeeded", True), ("failed", False)],
)
def test_task_engine_report_is_mirrored_into_bundle_only_when_enabled(
    tmp_path: Path,
    status: str,
    success: bool,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    agent_config = bundle / "agent_config.json"
    report = ExecutionReport(
        task_id="task",
        plan_hash="0" * 64,
        action_graph_hash="1" * 64,
        status=status,
        run_id="run",
        episode_id="0",
        provenance=build_execution_provenance(episode_seed=7),
        environments=(
            {
                "env_id": "0",
                "success": success,
                "semantic_success": {"task_01": success},
                "action_count": 3,
                "retry_count": 0,
                "recovery_count": 0,
                "revision_count": 0,
                "failures": [],
            },
        ),
        action_count=3,
        record_dir=(tmp_path / "runtime-records").as_posix(),
    )

    assert _publish_task_engine_report(agent_config, report, enabled=False) is None
    assert not (bundle / "execution_report.json").exists()

    path = _publish_task_engine_report(agent_config, report, enabled=True)

    assert path == bundle / "execution_report.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == status
    assert payload["record_dir"] == report.record_dir


def test_task_engine_exit_code_uses_report_status() -> None:
    success = SimpleNamespace(status="succeeded")
    failure = SimpleNamespace(status="failed")

    assert _task_engine_exit_code(False, [success]) == 0
    assert _task_engine_exit_code(False, [success, failure]) == 1
    assert _task_engine_exit_code(True, []) == 1
