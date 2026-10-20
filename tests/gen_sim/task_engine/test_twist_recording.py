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

import cv2
import numpy as np
import pytest
import torch

from dexsim.utility import images_to_video
from embodichain.gen_sim.task_engine._task_program import twist_recording
from embodichain.gen_sim.task_engine._task_program.twist_recording import (
    E8CameraRecorder,
)
from embodichain.lab.gym.envs.managers import FunctorCfg
from embodichain.lab.gym.envs.managers.record import record_camera_data
from embodichain.lab.sim.sensors import CameraCfg
from embodichain.utils.string import string_to_callable


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


def test_private_route_reuses_existing_camera_and_recorder_dispatch(
    tmp_path: Path,
) -> None:
    recorder, env = _recorder(tmp_path)
    route = (
        "embodichain.gen_sim.task_engine._task_program.twist_recording:E8CameraRecorder"
    )
    assert string_to_callable(route) is E8CameraRecorder
    assert isinstance(recorder, record_camera_data)
    assert env.sim.sensor_cfg is not None
    assert env.sim.sensor_cfg.visualization_role == "record"
    assert env.camera_group_ids == [7]
    assert recorder._writer is None


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


def test_empty_commit_and_repeated_finalize_create_no_video(tmp_path: Path) -> None:
    recorder, env = _recorder(tmp_path)
    recorder.save_and_clear()
    recorder.finalize()
    recorder.finalize()
    recorder.close()
    assert not (tmp_path / "videos").exists()
    with pytest.raises(RuntimeError, match="finalized"):
        recorder(env, None, "e8_knob_view")


def test_discard_removes_only_its_partial_and_preserves_other_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recorder, env = _recorder(tmp_path)
    writer = _Writer()
    _mock_writer(monkeypatch, writer)
    recorder(env, None, "e8_knob_view")
    partial = recorder._partial_path
    assert partial is not None
    other = partial.parent / "other.partial.mp4"
    other.write_bytes(b"preserved")
    recorder.discard_and_clear()
    recorder.finalize()
    assert writer.closed == 1
    assert not partial.exists()
    assert other.read_bytes() == b"preserved"
    assert not (tmp_path / "videos/episode_0_e8_knob_view.mp4").exists()


@pytest.mark.parametrize("failure", ["append", "close", "frame_count"])
def test_encoder_failure_never_publishes_partial_video(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str
) -> None:
    recorder, env = _recorder(tmp_path)
    writer = _Writer()
    _mock_writer(monkeypatch, writer)
    if failure == "append":
        writer.append_error = OSError("encoder append failed")
        with pytest.raises(OSError, match="append failed"):
            recorder(env, None, "e8_knob_view")
        with pytest.raises(RuntimeError, match="failed E8 camera"):
            recorder.save_and_clear()
    else:
        recorder(env, None, "e8_knob_view")
        if failure == "close":
            writer.close_error = OSError("encoder close failed")
            with pytest.raises(OSError, match="close failed"):
                recorder.save_and_clear()
        else:
            monkeypatch.setattr(
                twist_recording, "count_frames_and_secs", lambda path: (0, 0)
            )
            with pytest.raises(RuntimeError, match="frame count mismatch"):
                recorder.save_and_clear()
    assert recorder._frames == []
    assert recorder._partial_path is not None
    assert recorder._partial_path.is_file()
    assert not (tmp_path / "videos/episode_0_e8_knob_view.mp4").exists()
    recorder.finalize()
    recorder.finalize()


def test_committed_video_is_never_overwritten(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recorder, env = _recorder(tmp_path)
    _mock_writer(monkeypatch, _Writer())
    path = tmp_path / "videos/episode_0_e8_knob_view.mp4"
    path.parent.mkdir()
    path.write_bytes(b"user video")
    with pytest.raises(FileExistsError, match="overwrite"):
        recorder(env, None, "e8_knob_view")
    recorder.finalize()
    assert path.read_bytes() == b"user video"


def _decoded_frames(path: Path) -> tuple[float, list[np.ndarray]]:
    capture = cv2.VideoCapture(str(path))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        frames = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return fps, frames
    finally:
        capture.release()


@pytest.mark.parametrize("resolution", [(960, 540), (640, 360)])
def test_streamed_rgb_order_dimensions_and_quality_match_existing_utility(
    tmp_path: Path, resolution: tuple[int, int]
) -> None:
    width, height = resolution
    recorder, env = _recorder(tmp_path, resolution)
    colors = [(240, 15, 15), (15, 240, 15), (15, 15, 240), (180, 180, 30)]
    rgb_frames = [
        np.full((height, width, 3), color, dtype=np.uint8) for color in colors
    ]
    env.sim.camera.frames = [
        torch.from_numpy(
            np.concatenate(
                (frame, np.full((height, width, 1), 255, dtype=np.uint8)), axis=2
            )
        ).unsqueeze(0)
        for frame in rgb_frames
    ]
    for _ in rgb_frames:
        recorder(env, None, "e8_knob_view")
        assert recorder._frames == []
    recorder.save_and_clear()
    recorder.finalize()
    images_to_video(
        rgb_frames, str(tmp_path / "reference"), "original", fps=20, verbose=False
    )
    fps, streamed = _decoded_frames(tmp_path / "videos/episode_0_e8_knob_view.mp4")
    original_fps, original = _decoded_frames(tmp_path / "reference/original.mp4")
    assert fps == original_fps == 20
    assert len(streamed) == len(original) == len(rgb_frames)
    for frame, expected in zip(streamed, original, strict=True):
        np.testing.assert_array_equal(frame, expected)
    assert streamed[0].shape == original[0].shape
    assert [
        int(np.argmax(frame[frame.shape[0] // 2, frame.shape[1] // 2]))
        for frame in streamed[:3]
    ] == [0, 1, 2]
