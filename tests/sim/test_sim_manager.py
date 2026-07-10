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

from types import SimpleNamespace

import numpy as np
import pytest

from embodichain.lab.sim.sim_manager import SimulationManager, _WindowRecordState

DEFAULT_LOOK_AT = (
    (2.6, -2.2, 1.6),
    (0.0, 0.0, 0.45),
    (0.0, 0.0, 1.0),
)


class FakeCamera:
    """Simple camera stub for recorder unit tests."""

    def __init__(self) -> None:
        self._is_open = False
        self.last_pose: np.ndarray | None = None
        self.render_count = 0

    def is_open(self) -> bool:
        return self._is_open

    def open_camera(self) -> None:
        self._is_open = True

    def close_camera(self) -> None:
        self._is_open = False

    def set_world_pose(self, pose: np.ndarray) -> None:
        self.last_pose = np.asarray(pose, dtype=np.float32)

    def render(self) -> None:
        self.render_count += 1

    def get_rgb_map(self) -> np.ndarray:
        return np.full((4, 4, 4), 7, dtype=np.uint8)


class FakeThreadRuntime:
    """Runtime loop stub for viewer-timed recording."""

    def __init__(self) -> None:
        self.add_loop_calls: list[tuple[object, float]] = []

    def add_loop(self, callback, time_step: float) -> str:
        self.add_loop_calls.append((callback, time_step))
        return "loop_handle"


class FakeWorld:
    """World stub exposing the render-thread loop API."""

    def __init__(self) -> None:
        self.thread_runtime = FakeThreadRuntime()

    def thread_rt(self) -> FakeThreadRuntime:
        return self.thread_runtime


class FakeEnv:
    """Environment stub that creates fake cameras."""

    def __init__(self) -> None:
        self.created_cameras: list[FakeCamera] = []

    def create_camera(self, name: str, width: int, height: int) -> FakeCamera:
        camera = FakeCamera()
        self.created_cameras.append(camera)
        return camera


def _make_sim_manager(window: object | None = None) -> SimulationManager:
    """Create a minimally initialized simulation manager for recorder tests."""
    sim = object.__new__(SimulationManager)
    sim.instance_id = 0
    sim.sim_config = SimpleNamespace(width=64, height=48)
    sim._window = window
    sim._window_record_state = None
    sim._window_record_camera = None
    sim._window_record_save_threads = []
    sim._env = FakeEnv()
    sim._world = FakeWorld()
    return sim


def test_start_window_record_rejects_invalid_parameters() -> None:
    sim = _make_sim_manager()

    with pytest.raises(RuntimeError, match="FPS must be positive"):
        sim.start_window_record(fps=0, look_at=DEFAULT_LOOK_AT)

    with pytest.raises(RuntimeError, match="max_memory must be positive"):
        sim.start_window_record(fps=20, max_memory=0, look_at=DEFAULT_LOOK_AT)


def test_start_window_record_rejects_concurrent_sessions() -> None:
    sim = _make_sim_manager()
    sim._window_record_state = _WindowRecordState(
        time_step=0.1,
        max_memory_bytes=1024,
        output_dir="/tmp",
        video_name="existing",
        save_kwargs={"fps": 20},
    )

    with pytest.raises(RuntimeError, match="already active"):
        sim.start_window_record(look_at=DEFAULT_LOOK_AT)


def test_headless_recording_uses_sim_time_and_captures_frames() -> None:
    sim = _make_sim_manager()

    assert sim.start_window_record(look_at=DEFAULT_LOOK_AT, fps=5, max_memory=1)
    state = sim._window_record_state
    assert state is not None
    assert state.capture_from_sim_update is True
    assert state.loop_handle is None
    assert sim._window_record_camera is not None
    assert sim._window_record_camera.is_open() is True
    assert state.fixed_pose is not None
    assert sim._world.thread_runtime.add_loop_calls == []

    sim._step_window_record_from_sim_update(state, physics_dt=0.1)
    assert len(state.frames) == 0

    sim._step_window_record_from_sim_update(state, physics_dt=0.1)
    assert len(state.frames) == 1
    assert sim._window_record_camera.render_count == 1
    np.testing.assert_allclose(
        sim._window_record_camera.last_pose,
        state.fixed_pose,
    )


def test_stop_window_record_waits_for_background_export(monkeypatch) -> None:
    sim = _make_sim_manager()
    assert sim.start_window_record(look_at=DEFAULT_LOOK_AT, fps=5, max_memory=1)
    state = sim._window_record_state
    assert state is not None
    state.frames.append(np.zeros((4, 4, 3), dtype=np.uint8))

    save_call: dict[str, object] = {}

    def fake_save_window_record_worker(
        frames: list[np.ndarray],
        output_dir: str,
        video_name: str,
        save_kwargs: dict[str, object],
    ) -> None:
        save_call["frame_count"] = len(frames)
        save_call["output_dir"] = output_dir
        save_call["video_name"] = video_name
        save_call["save_kwargs"] = save_kwargs

    monkeypatch.setattr(
        sim, "_save_window_record_worker", fake_save_window_record_worker
    )

    assert sim.stop_window_record() is True
    sim.wait_window_record_saves()

    assert save_call["frame_count"] == 1
    assert save_call["save_kwargs"] == {"fps": 5}
    assert sim._window_record_save_threads == []


def test_format_window_camera_pose_as_look_at() -> None:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = (1.0, 2.0, 3.0)
    pose[:3, 1] = (0.0, -1.0, 0.0)
    pose[:3, 2] = (0.0, 0.0, -1.0)

    snippet = SimulationManager._format_window_camera_pose(pose)

    assert "window.set_look_at(" in snippet
    assert "eye=np.array([1.0, 2.0, 3.0]" in snippet
    assert "look_at=np.array([1.0, 2.0, 4.0]" in snippet
    assert "up=np.array([0.0, -1.0, 0.0]" in snippet


def test_window_camera_pose_look_at_conversion_round_trips() -> None:
    eye = np.array([2.845114, 0.15252542, 2.8951859], dtype=np.float32)
    target = np.array([2.1446962, 0.11497631, 2.1824413], dtype=np.float32)
    up = np.array([0.71172267, 0.038155168, -0.70142359], dtype=np.float32)

    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    camera_up = np.cross(right, forward)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 0] = right
    pose[:3, 1] = camera_up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye

    converted_eye, converted_target, converted_up = (
        SimulationManager._window_camera_pose_to_look_at(pose)
    )
    reconstructed_forward = converted_target - converted_eye
    reconstructed_forward /= np.linalg.norm(reconstructed_forward)
    reconstructed_right = np.cross(reconstructed_forward, converted_up)
    reconstructed_right /= np.linalg.norm(reconstructed_right)
    reconstructed_up = np.cross(reconstructed_right, reconstructed_forward)
    reconstructed_pose = np.eye(4, dtype=np.float32)
    reconstructed_pose[:3, 0] = reconstructed_right
    reconstructed_pose[:3, 1] = reconstructed_up
    reconstructed_pose[:3, 2] = -reconstructed_forward
    reconstructed_pose[:3, 3] = converted_eye

    np.testing.assert_allclose(reconstructed_pose, pose, atol=1e-6)


def test_print_window_camera_pose_uses_raw_matrix_when_requested(capsys) -> None:
    pose = np.eye(4, dtype=np.float32)
    sim = _make_sim_manager(window=SimpleNamespace(get_pose_matrix=lambda: pose))

    snippet = sim.print_window_camera_pose(convert_to_look_at=False)

    assert snippet is not None
    assert snippet.startswith("window_pose = np.array([")
    assert "window.set_look_at" not in capsys.readouterr().out


def test_window_camera_pose_hotkey_registers_p_control(monkeypatch) -> None:
    class FakeWindow:
        def __init__(self) -> None:
            self.controls: list[object] = []

        def add_input_control(self, control: object) -> None:
            self.controls.append(control)

    window = FakeWindow()
    sim = _make_sim_manager(window=window)
    sim._window_camera_pose_input_control = None
    calls: list[bool] = []
    monkeypatch.setattr(
        sim,
        "print_window_camera_pose",
        lambda convert_to_look_at=True: calls.append(convert_to_look_at),
    )

    assert sim.enable_window_camera_pose_hotkey() is True
    assert len(window.controls) == 1

    from dexsim.types import InputKey

    control = window.controls[0]
    control.on_key_down(InputKey.SCANCODE_P.value)
    control.on_key_down(InputKey.SCANCODE_O.value)
    assert calls == [True]
