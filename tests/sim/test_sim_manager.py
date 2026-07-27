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
from unittest.mock import MagicMock

import dexsim
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


class FakeEntityGizmo:
    """Entity-gizmo stub with external-target registration."""

    def __init__(self) -> None:
        self.active = True
        self.external_targets: list[tuple[int, object, object, object]] = []

    def register_external_target(
        self,
        target_id: int,
        target_type: object,
        target: object,
        actor_type: object,
    ) -> object:
        self.external_targets.append((target_id, target_type, target, actor_type))
        return dexsim.interaction.EntityGizmoResult.SUCCESS


class FakeWorld:
    """World stub exposing the render-thread loop API."""

    def __init__(self) -> None:
        self.thread_runtime = FakeThreadRuntime()
        self.entity_gizmo: object | None = None
        self.entity_gizmo_configs: list[object | None] = []
        self.window = SimpleNamespace(add_input_control=lambda control: None)
        self.window_open_count = 0
        self.window_closed = False

    def thread_rt(self) -> FakeThreadRuntime:
        return self.thread_runtime

    def enable_entity_gizmo(self, config: object | None = None) -> object:
        self.entity_gizmo_configs.append(config)
        self.entity_gizmo = FakeEntityGizmo()
        return self.entity_gizmo

    def disable_entity_gizmo(self) -> None:
        if self.entity_gizmo is not None:
            self.entity_gizmo.active = False
        self.entity_gizmo = None

    def get_entity_gizmo(self) -> object | None:
        return self.entity_gizmo

    def open_window(self) -> None:
        self.window_open_count += 1
        self.window_closed = False

    def get_windows(self) -> object:
        return self.window

    def close_window(self) -> None:
        self.window_closed = True


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
    sim.sim_config = SimpleNamespace(
        width=64,
        height=48,
        enable_entity_gizmo_on_window_open=True,
    )
    sim._window = window
    sim._entity_gizmo_config = None
    sim._window_record_state = None
    sim._window_record_camera = None
    sim._window_record_save_threads = []
    sim._window_record_hotkey_cfg = None
    sim._window_camera_pose_hotkey_cfg = None
    sim._window_record_input_control = None
    sim._window_camera_pose_input_control = None
    sim._env = FakeEnv()
    sim._world = FakeWorld()
    sim._default_plane = object()
    sim.is_window_opened = window is not None
    return sim


def test_window_camera_pose_to_look_at_uses_dexsim_world_up() -> None:
    """Captured look-at snippets preserve DexSim's default Z-up controls."""
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    eye, look_at, up = SimulationManager._window_camera_pose_to_look_at(pose)

    np.testing.assert_allclose(eye, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(look_at, [1.0, 2.0, 2.0])
    np.testing.assert_allclose(up, [0.0, 0.0, 1.0])


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


def test_entity_gizmo_lifecycle_delegates_to_dexsim_world() -> None:
    sim = _make_sim_manager()
    config = object()

    controller = sim.enable_entity_gizmo(config)

    assert controller is sim._world.get_entity_gizmo()
    assert sim._world.entity_gizmo_configs == [config]
    assert sim.get_entity_gizmo() is controller
    assert sim.has_entity_gizmo() is True
    assert sim.disable_entity_gizmo() is True
    assert controller.active is False
    assert sim.has_entity_gizmo() is False
    assert sim.disable_entity_gizmo() is False


def test_entity_gizmo_registers_default_plane_as_static_exclusion() -> None:
    sim = _make_sim_manager()

    controller = sim.enable_entity_gizmo()

    assert controller.external_targets == [
        (
            SimulationManager._DEFAULT_PLANE_GIZMO_TARGET_ID,
            dexsim.interaction.EntityGizmoTargetType.RIGID_BODY,
            sim._default_plane,
            dexsim.types.ActorType.STATIC,
        )
    ]


def test_open_window_enables_entity_gizmo_by_default() -> None:
    sim = _make_sim_manager()

    sim.open_window()

    assert sim.is_window_opened is True
    assert sim._world.window_open_count == 1
    assert sim.has_entity_gizmo() is True
    assert sim._world.entity_gizmo_configs == [None]


def test_open_window_supports_view_only_opt_out() -> None:
    sim = _make_sim_manager()

    sim.open_window(enable_entity_gizmo=False)

    assert sim.is_window_opened is True
    assert sim.has_entity_gizmo() is False
    assert sim._world.entity_gizmo_configs == []


def test_open_window_view_only_opt_out_disables_active_controller() -> None:
    sim = _make_sim_manager()
    controller = sim.enable_entity_gizmo()

    sim.open_window(enable_entity_gizmo=False)

    assert controller.active is False
    assert sim.has_entity_gizmo() is False


def test_open_window_respects_configured_entity_gizmo_default() -> None:
    sim = _make_sim_manager()
    sim.sim_config.enable_entity_gizmo_on_window_open = False

    sim.open_window()

    assert sim.is_window_opened is True
    assert sim.has_entity_gizmo() is False


def test_open_window_tolerates_dexsim_without_entity_gizmo_api() -> None:
    sim = _make_sim_manager()
    window = object()
    sim._world = SimpleNamespace(
        open_window=lambda: None,
        get_windows=lambda: window,
    )

    sim.open_window()

    assert sim.is_window_opened is True
    assert sim._window is window
    assert sim.has_entity_gizmo() is False


def test_open_window_preserves_active_entity_gizmo_configuration() -> None:
    sim = _make_sim_manager(window=object())
    config = object()
    controller = sim.enable_entity_gizmo(config)

    sim.open_window()

    assert sim.get_entity_gizmo() is controller
    assert sim._world.entity_gizmo_configs == [config]
    assert sim._world.window_open_count == 0


def test_reopened_window_restores_last_entity_gizmo_configuration() -> None:
    sim = _make_sim_manager(window=object())
    config = object()
    sim.enable_entity_gizmo(config)
    sim.close_window()

    sim.open_window()

    assert sim.has_entity_gizmo() is True
    assert sim._world.entity_gizmo_configs == [config, config]


def test_close_window_disables_entity_gizmo() -> None:
    sim = _make_sim_manager(window=object())
    controller = sim.enable_entity_gizmo()

    sim.close_window()

    assert controller.active is False
    assert sim.has_entity_gizmo() is False
    assert sim._world.window_closed is True
    assert sim.is_window_opened is False


def test_reset_objects_state_includes_soft_and_cloth_assets() -> None:
    sim = object.__new__(SimulationManager)
    sim._robots = {}
    sim._articulations = {}
    sim._rigid_objects = {}
    sim._rigid_object_groups = {}
    sim._lights = {}
    sim._sensors = {}
    sim._soft_objects = {"soft": MagicMock()}
    sim._cloth_objects = {"cloth": MagicMock()}

    sim.reset_objects_state(env_ids=[1])

    sim._soft_objects["soft"].reset.assert_called_once_with([1])
    sim._cloth_objects["cloth"].reset.assert_called_once_with([1])
