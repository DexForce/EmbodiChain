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
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from embodichain.lab.gym.envs.episode_context import EpisodeContext
from embodichain.lab.gym.envs.managers.record import record_camera_data_async

from embodichain.lab.gym.envs.managers import FunctorCfg
from embodichain.lab.gym.envs.managers.record import record_camera_data
from embodichain.lab.sim.sensors import CameraCfg


class _CameraHandle:
    group_id = 7


class _Simulation:
    def __init__(self) -> None:
        self.camera = _CameraHandle()
        self.sensor_cfg: CameraCfg | None = None

    def add_sensor(self, sensor_cfg: CameraCfg) -> _CameraHandle:
        self.sensor_cfg = sensor_cfg
        return self.camera


class _Environment:
    def __init__(self) -> None:
        self.sim = _Simulation()
        self.camera_group_ids: list[int] = []

    def add_camera_group_id(self, group_id: int) -> None:
        self.camera_group_ids.append(group_id)


def test_record_camera_is_marked_for_visualization() -> None:
    env = _Environment()
    cfg = FunctorCfg(func=record_camera_data, params={"name": "episode_camera"})

    record_camera_data(cfg, env)

    assert env.sim.sensor_cfg is not None
    assert env.sim.sensor_cfg.visualization_role == "record"


@pytest.fixture
def camera_env():
    context = EpisodeContext(4)
    context._begin_episode([0, 1, 2, 3])
    context._finish_initialization([0, 1, 2, 3])
    rgb = torch.stack([torch.full((4, 6, 4), i, dtype=torch.uint8) for i in range(4)])
    camera = SimpleNamespace(
        group_id=1, update=lambda **kwargs: None, get_data=lambda: {"color": rgb}
    )
    env = SimpleNamespace(
        num_envs=4,
        device=torch.device("cpu"),
        episode_context=context,
        sim=SimpleNamespace(add_sensor=lambda **kwargs: camera),
        add_camera_group_id=lambda group_id: None,
    )
    return env, rgb


@pytest.mark.parametrize("kind", [record_camera_data, record_camera_data_async])
def test_videos_commit_valid_rows_and_keep_other_live_buffers(
    camera_env, kind, tmp_path, monkeypatch
):
    from embodichain.lab.gym.envs.managers import record

    env, rgb = camera_env
    writes = []

    def save(frames, directory, name, fps):
        writes.append((name, [frame.copy() for frame in frames]))
        (Path(directory) / (name + ".mp4")).write_bytes(b"video")

    monkeypatch.setattr(record, "images_to_video", save)
    recorder = kind(
        FunctorCfg(func=kind, params={"name": "camera", "save_path": str(tmp_path)}),
        env,
    )
    recorder(env, None, name="camera")
    rgb.fill_(99)
    with env.episode_context.check("layout", [3]) as finish:
        finish(accepted=False, reasons="no layout")
    recorder.save_and_clear([1, 3])
    assert [len(row) for row in recorder._frames_list] == [1, 0, 1, 1]
    recorder.save_and_clear([3], purpose="diagnostic")
    recorder.discard_and_clear([0])
    assert len(recorder._frames_list[2]) == 1
    recorder.finalize()
    recorder.close()
    assert len(writes) == 2
    assert sorted(int(frames[0][0, 0, 0]) for _, frames in writes) == [1, 3]
    normal = list((tmp_path / "env_001").glob("*.json"))
    diagnostic = list((tmp_path / "diagnostics/env_003").glob("*.json"))
    assert len(normal) == len(diagnostic) == 1
    assert json.loads(normal[0].read_text())["episode_context"]["recordable"]
    assert not json.loads(diagnostic[0].read_text())["episode_context"]["recordable"]
    assert json.loads(diagnostic[0].read_text())["artifact_purpose"] == "diagnostic"
    assert not (tmp_path / "env_002").exists()


def test_sync_video_io_failure_retains_owned_job_for_retry(
    camera_env, tmp_path, monkeypatch
):
    from embodichain.lab.gym.envs.managers import record

    env, rgb = camera_env
    recorder = record_camera_data(
        FunctorCfg(
            func=record_camera_data,
            params={"name": "camera", "save_path": str(tmp_path)},
        ),
        env,
    )
    recorder(env, [3], name="camera")
    calls = []

    def save(frames, directory, name, fps):
        calls.append(name)
        if len(calls) == 1:
            raise OSError("disk full")
        assert np.all(frames[0] == 3)
        (Path(directory) / (name + ".mp4")).write_bytes(b"video")

    monkeypatch.setattr(record, "images_to_video", save)
    with pytest.raises(OSError, match="disk full"):
        recorder.save_and_clear([3])
    assert len(recorder._jobs) == 1
    rgb.fill_(99)
    recorder.save_and_clear([3])
    recorder.finalize()
    assert calls[0] == calls[1]
    assert not recorder._jobs
    assert len(list(tmp_path.rglob("*.mp4"))) == 1


def test_async_writer_owns_provenance_before_partial_reset(
    camera_env, tmp_path, monkeypatch
):
    from embodichain.lab.gym.envs.managers import record

    env, _ = camera_env
    entered, release = threading.Event(), threading.Event()

    def save(frames, directory, name, fps):
        entered.set()
        assert release.wait(timeout=5)
        (Path(directory) / (name + ".mp4")).write_bytes(b"video")

    monkeypatch.setattr(record, "images_to_video", save)
    recorder = record_camera_data_async(
        FunctorCfg(
            func=record_camera_data_async,
            params={"name": "camera", "save_path": str(tmp_path)},
        ),
        env,
    )
    recorder(env, [1], name="camera")
    recorder.save_and_clear([1])
    try:
        assert entered.wait(timeout=5)
        env.episode_context._begin_episode([1])
        env.episode_context._finish_initialization([1])
    finally:
        release.set()
        recorder.finalize()
    sidecar = json.loads(next(tmp_path.rglob("*.json")).read_text())
    assert sidecar["episode_context"]["episode"] == 0
    assert env.episode_context.snapshot([1])[0]["episode"] == 1


@pytest.mark.parametrize("kind", [record_camera_data, record_camera_data_async])
def test_close_discards_uncommitted_rows(camera_env, kind, tmp_path, monkeypatch):
    from embodichain.lab.gym.envs.managers import record

    env, _ = camera_env
    monkeypatch.setattr(
        record,
        "images_to_video",
        lambda *a, **k: pytest.fail("Close committed live frames"),
    )
    recorder = kind(
        FunctorCfg(func=kind, params={"name": "camera", "save_path": str(tmp_path)}),
        env,
    )
    recorder(env, None, name="camera")
    recorder.finalize()
    recorder.finalize()
    assert not any(recorder._frames_list)
    assert not list(tmp_path.rglob("*.mp4"))
