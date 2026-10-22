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

import gc
from pathlib import Path
from typing import Any
import weakref
import numpy as np
import pytest
import torch
from embodichain.gen_sim.task_engine._task_program import twist_recording
from embodichain.gen_sim.task_engine._task_program.twist_recording import (
    E8CameraRecorder,
)
from embodichain.lab.gym.envs.managers import FunctorCfg
from embodichain.lab.sim.sensors import CameraCfg


class _Camera:
    group_id = 7

    def __init__(self) -> None:
        self.frames = [torch.zeros((1, 16, 32, 4), dtype=torch.uint8)]
        self.updates: list[dict[str, Any]] = []
        self.count = 0

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)

    def get_data(self) -> dict[str, torch.Tensor]:
        frame = self.frames[self.count % len(self.frames)]
        self.count += 1
        return {"color": frame}


class _Simulation:
    def __init__(self) -> None:
        self.camera = _Camera()
        self.sensor_cfg: CameraCfg | None = None

    def add_sensor(self, sensor_cfg: CameraCfg) -> _Camera:
        self.sensor_cfg = sensor_cfg
        return self.camera


class _Environment:
    def __init__(self) -> None:
        self.sim = _Simulation()
        self.camera_group_ids: list[int] = []

    def add_camera_group_id(self, group_id: int) -> None:
        self.camera_group_ids.append(group_id)


class _Writer:
    def __init__(self) -> None:
        self.frame_refs: list[weakref.ReferenceType[np.ndarray]] = []
        self.closed = 0
        self.append_error: BaseException | None = None
        self.close_error: BaseException | None = None

    def append_data(self, frame: np.ndarray) -> None:
        if self.append_error is not None:
            raise self.append_error
        self.frame_refs.append(weakref.ref(frame))

    def close(self) -> None:
        self.closed += 1
        if self.close_error is not None:
            raise self.close_error


def _recorder(
    tmp_path: Path, resolution: tuple[int, int] = (32, 16)
) -> tuple[E8CameraRecorder, _Environment]:
    env = _Environment()
    cfg = FunctorCfg(
        func=E8CameraRecorder,
        params={
            "name": "e8_knob_view",
            "resolution": resolution,
            "save_path": str(tmp_path / "videos"),
        },
    )
    return E8CameraRecorder(cfg, env), env


def _mock_writer(
    monkeypatch: pytest.MonkeyPatch, writer: _Writer
) -> list[tuple[str, dict[str, Any]]]:
    calls = []

    def open_writer(path: str, **kwargs: Any) -> _Writer:
        calls.append((path, kwargs))
        return writer

    monkeypatch.setattr(twist_recording.imageio, "get_writer", open_writer)
    monkeypatch.setattr(
        twist_recording,
        "count_frames_and_secs",
        lambda path: (len(writer.frame_refs), len(writer.frame_refs) / 20),
    )
    return calls


def test_capture_keeps_no_completed_frame_objects(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recorder, env = _recorder(tmp_path)
    writer = _Writer()
    calls = _mock_writer(monkeypatch, writer)
    for _ in range(100):
        recorder(env, None, "e8_knob_view")
        assert recorder._frames == []
    gc.collect()
    assert len(writer.frame_refs) == 100
    assert all(reference() is None for reference in writer.frame_refs)
    assert calls[0][1] == {"fps": 20, "quality": 5}
    assert all(update == {"fetch_only": True} for update in env.sim.camera.updates)
    recorder.save_and_clear()
    assert writer.closed == 1
    assert (tmp_path / "videos/episode_0_e8_knob_view.mp4").is_file()
    assert recorder._frame_count == 0
