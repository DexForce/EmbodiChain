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
        self.window_closed = False

    def thread_rt(self) -> FakeThreadRuntime:
        return self.thread_runtime

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


class FakeWindow:
    """Window stub that records registered input controls."""

    def __init__(self) -> None:
        self.controls: list[object] = []

    def add_input_control(self, control: object) -> None:
        self.controls.append(control)


class FakeRigidEntity:
    """DexSim entity stub identified by its raycast user ID."""

    def __init__(self, user_id: int) -> None:
        self._user_id = user_id

    def get_user_id(self) -> int:
        return self._user_id


class FakeRigidObject:
    """Rigid-object stub that records runtime body-type transitions."""

    def __init__(self, body_type: str, user_id: int) -> None:
        self.body_type = body_type
        self._entities = [FakeRigidEntity(user_id)]
        self.body_type_history: list[str] = []

    def set_body_type(self, body_type: str) -> None:
        self.body_type = body_type
        self.body_type_history.append(body_type)


class FakeGizmo:
    """Gizmo stub that exposes whether cleanup ran."""

    def __init__(self) -> None:
        self.destroyed = False

    def destroy(self) -> None:
        self.destroyed = True


def _make_sim_manager(window: object | None = None) -> SimulationManager:
    """Create a minimally initialized simulation manager for recorder tests."""
    sim = object.__new__(SimulationManager)
    sim.instance_id = 0
    sim.sim_config = SimpleNamespace(width=64, height=48)
    sim._window = window
    sim._window_record_state = None
    sim._window_record_camera = None
    sim._window_record_save_threads = []
    sim._window_record_input_control = None
    sim._window_camera_pose_input_control = None
    sim._window_gizmo_hotkey_enabled = True
    sim._window_gizmo_input_control = None
    sim._window_gizmo_state = None
    sim._gizmo_controller = None
    sim._gizmos = {}
    sim._rigid_objects = {}
    sim._robots = {}
    sim._sensors = {}
    sim._arenas = []
    sim._env = FakeEnv()
    sim._world = FakeWorld()
    sim.is_window_opened = window is not None
    return sim


def _add_fake_rigid_object(
    sim: SimulationManager, body_type: str = "dynamic", user_id: int = 17
) -> FakeRigidObject:
    """Add one fake rigid object and replace gizmo creation with a stub."""
    rigid_object = FakeRigidObject(body_type=body_type, user_id=user_id)
    sim._rigid_objects["cube"] = rigid_object

    def enable_fake_gizmo(uid: str, *args, **kwargs) -> FakeGizmo:
        gizmo = FakeGizmo()
        sim._gizmos[uid] = gizmo
        return gizmo

    sim.enable_gizmo = enable_fake_gizmo
    return rigid_object


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


def test_window_gizmo_toggles_dynamic_object_and_restores_body_type() -> None:
    sim = _make_sim_manager()
    rigid_object = _add_fake_rigid_object(sim)
    selected_object = FakeRigidEntity(user_id=17)

    assert sim.toggle_selected_rigid_object_gizmo(selected_object) is True
    assert rigid_object.body_type == "kinematic"
    assert sim.has_gizmo("cube") is True

    gizmo = sim.get_gizmo("cube")
    assert sim.toggle_selected_rigid_object_gizmo(None) is False
    assert rigid_object.body_type == "dynamic"
    assert sim.has_gizmo("cube") is False
    assert gizmo.destroyed is True


def test_window_gizmo_preserves_original_kinematic_body_type() -> None:
    sim = _make_sim_manager()
    rigid_object = _add_fake_rigid_object(sim, body_type="kinematic")

    assert sim.toggle_selected_rigid_object_gizmo(FakeRigidEntity(17)) is True
    sim.disable_gizmo("cube")

    assert rigid_object.body_type == "kinematic"
    assert rigid_object.body_type_history == ["kinematic", "kinematic"]


def test_window_gizmo_rejects_missing_unregistered_and_static_selections() -> None:
    sim = _make_sim_manager()
    rigid_object = _add_fake_rigid_object(sim, body_type="static")

    assert sim.toggle_selected_rigid_object_gizmo(None) is False
    assert sim.toggle_selected_rigid_object_gizmo(FakeRigidEntity(99)) is False
    assert sim.toggle_selected_rigid_object_gizmo(FakeRigidEntity(17)) is False

    assert rigid_object.body_type == "static"
    assert rigid_object.body_type_history == []
    assert sim._window_gizmo_state is None


def test_window_gizmo_rolls_back_body_type_when_enable_fails() -> None:
    sim = _make_sim_manager()
    rigid_object = _add_fake_rigid_object(sim)
    sim.enable_gizmo = lambda uid, *args, **kwargs: None

    assert sim.toggle_selected_rigid_object_gizmo(FakeRigidEntity(17)) is False

    assert rigid_object.body_type == "dynamic"
    assert rigid_object.body_type_history == ["kinematic", "dynamic"]
    assert sim._window_gizmo_state is None


def test_enable_gizmo_returns_none_when_creation_fails(monkeypatch) -> None:
    import embodichain.lab.sim.sim_manager as sim_manager_module

    sim = _make_sim_manager()
    _add_fake_rigid_object(sim)
    del sim.enable_gizmo

    class FailingGizmo:
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("gizmo creation failed")

    monkeypatch.setattr(sim_manager_module, "Gizmo", FailingGizmo)

    assert sim.enable_gizmo("cube") is None
    assert sim.has_gizmo("cube") is False


def test_close_window_disables_gizmo_and_restores_body_type() -> None:
    sim = _make_sim_manager(window=FakeWindow())
    rigid_object = _add_fake_rigid_object(sim)
    assert sim.toggle_selected_rigid_object_gizmo(FakeRigidEntity(17)) is True

    sim.close_window()

    assert rigid_object.body_type == "dynamic"
    assert sim.has_gizmo("cube") is False
    assert sim._window_gizmo_state is None
    assert sim._world.window_closed is True


def test_window_gizmo_hotkey_registration_is_idempotent(monkeypatch) -> None:
    from dexsim.types import InputKey

    window = FakeWindow()
    sim = _make_sim_manager(window=window)
    toggle_calls: list[object | None] = []
    monkeypatch.setattr(
        sim,
        "toggle_selected_rigid_object_gizmo",
        lambda selected: toggle_calls.append(selected),
    )

    assert sim.enable_window_gizmo_hotkey() is True
    assert sim.enable_window_gizmo_hotkey() is True
    assert len(window.controls) == 1

    window.controls[0].on_key_down(InputKey.SCANCODE_G.value)
    assert toggle_calls == [None]
