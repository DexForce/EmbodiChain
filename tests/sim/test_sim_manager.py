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
import queue

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

import embodichain.lab.sim.sim_manager as sim_manager_module
from embodichain.lab.sim.profiler import Profiler
from embodichain.lab.sim.sim_manager import (
    SimulationManager,
    SimulationManagerCfg,
    _WindowRecordState,
)
from embodichain.lab.visualization import (
    GizmoCommand,
    PointCloudOverlay,
    SceneOverlays,
    VisualizationCfg,
)

DEFAULT_LOOK_AT = (
    (2.6, -2.2, 1.6),
    (0.0, 0.0, 0.45),
    (0.0, 0.0, 1.0),
)

pytestmark = pytest.mark.no_sim


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
        self.physics_updates: list[float] = []

    def thread_rt(self) -> FakeThreadRuntime:
        return self.thread_runtime

    def is_physics_manually_update(self) -> bool:
        return True

    def update(self, physics_dt: float) -> None:
        self.physics_updates.append(physics_dt)


class FakeEnv:
    """Environment stub that creates fake cameras."""

    def __init__(self) -> None:
        self.created_cameras: list[FakeCamera] = []

    def create_camera(self, name: str, width: int, height: int) -> FakeCamera:
        camera = FakeCamera()
        self.created_cameras.append(camera)
        return camera


class FakeVisualizationRuntime:
    """Visualization runtime stub for lifecycle unit tests."""

    def __init__(self) -> None:
        self.is_running = True
        self.capture_calls: list[dict[str, object]] = []
        self.refresh_count = 0
        self.stopped = False

    def capture(self, **kwargs: object) -> bool:
        self.capture_calls.append(kwargs)
        return True

    def refresh_scene(self) -> SimpleNamespace:
        self.refresh_count += 1
        return SimpleNamespace(scene_revision=self.refresh_count)

    def stop(self) -> None:
        self.is_running = False
        self.stopped = True


class FakeInteractiveGizmo:
    """Minimal shared target controller for browser-command routing tests."""

    def __init__(self) -> None:
        self.owner: str | None = None
        self.requested_poses: list[torch.Tensor] = []

    def begin_interaction(self, source_id: str) -> bool:
        if self.owner not in {None, source_id}:
            return False
        self.owner = source_id
        return True

    def request_local_pose(self, pose: torch.Tensor, *, source_id: str) -> bool:
        if self.owner not in {None, source_id}:
            return False
        self.requested_poses.append(pose.clone())
        return True

    def end_interaction(self, source_id: str) -> bool:
        if self.owner != source_id:
            return False
        self.owner = None
        return True


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


def _make_visualization_sim_manager() -> (
    tuple[SimulationManager, FakeVisualizationRuntime]
):
    """Create a manager with a running fake Viser runtime."""
    sim = object.__new__(SimulationManager)
    runtime = FakeVisualizationRuntime()
    sim.sim_config = SimpleNamespace(
        physics_dt=0.01,
        visualization=SimpleNamespace(backend="viser"),
    )
    sim.device = SimpleNamespace(type="cpu")
    sim.profiler = Profiler(None, torch.device("cpu"))
    sim._is_initialized_gpu_physics = False
    sim._world = FakeWorld()
    sim._window_record_state = None
    sim._visualization_runtime = runtime
    sim._visualization_overlays = None
    sim._visualization_topology_revision = 2
    sim._visualization_manifest_topology_revision = 1
    sim._visualization_sim_step = 0
    sim._visualization_sim_time = 0.0
    sim._visualization_error_reported = False
    return sim, runtime


def test_flush_cleanup_queue_returns_immediately_when_no_destroy_is_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleanup_queue: queue.Queue = queue.Queue()
    collect = MagicMock()
    wait_scene_destruction = MagicMock()
    monkeypatch.setattr(SimulationManager, "_cleanup_queue", cleanup_queue)
    monkeypatch.setattr(gc, "collect", collect)
    monkeypatch.setattr(
        SimulationManager, "wait_scene_destruction", wait_scene_destruction
    )

    SimulationManager.flush_cleanup_queue()

    collect.assert_not_called()
    wait_scene_destruction.assert_not_called()


def test_flush_cleanup_queue_waits_after_running_pending_destroy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleanup_queue: queue.Queue = queue.Queue()
    destroy = MagicMock()
    cleanup_queue.put(destroy)
    collect = MagicMock()
    wait_scene_destruction = MagicMock()
    monkeypatch.setattr(SimulationManager, "_cleanup_queue", cleanup_queue)
    monkeypatch.setattr(gc, "collect", collect)
    monkeypatch.setattr(
        SimulationManager, "wait_scene_destruction", wait_scene_destruction
    )

    SimulationManager.flush_cleanup_queue()

    destroy.assert_called_once_with()
    collect.assert_called_once_with()
    wait_scene_destruction.assert_called_once_with()


def test_sim_update_refreshes_dirty_visualization_and_captures_current_state() -> None:
    sim, runtime = _make_visualization_sim_manager()

    sim.update(step=2)

    assert runtime.refresh_count == 1
    assert sim._visualization_manifest_topology_revision == 2
    assert [call["sim_step"] for call in runtime.capture_calls] == [1, 2]
    assert [call["sim_time"] for call in runtime.capture_calls] == [0.01, 0.02]
    assert [call["capture_camera_images"] for call in runtime.capture_calls] == [
        False,
        True,
    ]
    assert all(call["overlays"] is None for call in runtime.capture_calls)


def test_sim_manager_persists_overlays_across_automatic_captures() -> None:
    sim, runtime = _make_visualization_sim_manager()
    overlays = SceneOverlays(
        point_clouds=(
            PointCloudOverlay(
                "workspace",
                np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            ),
        )
    )

    sim.set_visualization_overlays(overlays)
    sim.update(step=1)

    assert sim.visualization_overlays is overlays
    assert runtime.capture_calls[-1]["overlays"] is overlays

    sim.set_visualization_overlays(None)
    assert sim.visualization_overlays is None
    assert runtime.capture_calls[-1]["overlays"] is None


def test_sim_manager_routes_viser_gizmo_commands_in_local_arena_frame() -> None:
    sim = object.__new__(SimulationManager)
    gizmo = FakeInteractiveGizmo()
    commands = tuple(
        GizmoCommand(
            run_id="run",
            scene_revision=2,
            sequence=sequence,
            gizmo_id="cube",
            phase=phase,
            client_id="client-a",
            position=np.array([2.0 + sequence, 0.0, 0.5], dtype=np.float32),
            wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )
        for sequence, phase in enumerate(("start", "update", "end"), start=1)
    )
    sim.sim_config = SimpleNamespace(
        visualization=SimpleNamespace(allow_commands=True),
    )
    sim.device = torch.device("cpu")
    sim._gizmos = {"cube": gizmo}
    sim.__dict__["arena_offsets"] = torch.tensor([[2.0, 0.0, 0.0]])
    sim._visualization_runtime = SimpleNamespace(
        exporter=SimpleNamespace(run_id="run", scene_revision=2),
        drain_gizmo_commands=lambda: commands,
    )

    accepted = sim.process_visualization_commands()

    assert accepted == 3
    assert gizmo.owner is None
    np.testing.assert_allclose(
        gizmo.requested_poses[-1][0, :3, 3],
        [3.0, 0.0, 0.5],
    )


def test_simulation_config_nests_viser_server_under_visualization() -> None:
    cfg = SimulationManagerCfg()

    assert cfg.visualization.viser_server.port == 8080
    assert not hasattr(cfg, "viser_server")


def test_simulation_config_forces_headless_mode_for_viser() -> None:
    cfg = SimulationManagerCfg(
        headless=False,
        visualization=VisualizationCfg(backend="viser"),
    )

    assert cfg.headless


def test_headless_simulation_does_not_enable_viser() -> None:
    cfg = SimulationManagerCfg(headless=True)

    assert cfg.visualization.backend == "none"


@pytest.mark.parametrize(
    ("backend", "runtime_active", "expected"),
    [
        ("none", False, True),
        ("none", True, False),
        ("viser", False, False),
    ],
)
def test_native_window_availability_depends_on_visualization_backend(
    backend: str,
    runtime_active: bool,
    expected: bool,
) -> None:
    sim = object.__new__(SimulationManager)
    sim.sim_config = SimpleNamespace(
        visualization=SimpleNamespace(backend=backend),
    )
    sim._visualization_runtime = object() if runtime_active else None

    assert sim.can_open_native_window() is expected


def test_open_window_skips_viser_backend() -> None:
    sim = object.__new__(SimulationManager)
    sim.sim_config = SimpleNamespace(
        visualization=SimpleNamespace(backend="viser"),
    )
    sim._visualization_runtime = None
    sim._world = MagicMock()

    opened = sim.open_window()

    assert not opened
    sim._world.open_window.assert_not_called()


def test_open_window_allows_native_backend() -> None:
    sim = object.__new__(SimulationManager)
    sim.sim_config = SimpleNamespace(
        visualization=SimpleNamespace(backend="none"),
    )
    sim._visualization_runtime = None
    sim._world = MagicMock()
    sim._window_record_hotkey_cfg = None
    sim._window_record_input_control = None
    sim._window_camera_pose_hotkey_cfg = None
    sim._window_camera_pose_input_control = None
    sim.is_window_opened = False

    opened = sim.open_window()

    assert opened
    sim._world.open_window.assert_called_once_with()
    assert sim.is_window_opened


def test_open_window_is_idempotent() -> None:
    sim = object.__new__(SimulationManager)
    sim.sim_config = SimpleNamespace(
        visualization=SimpleNamespace(backend="none"),
    )
    sim._visualization_runtime = None
    sim._world = MagicMock()
    sim.is_window_opened = True

    opened = sim.open_window()

    assert opened
    sim._world.open_window.assert_not_called()


def test_start_visualization_rejects_open_native_window() -> None:
    sim = object.__new__(SimulationManager)
    sim.sim_config = SimpleNamespace(
        visualization=SimpleNamespace(backend="viser"),
    )
    sim.is_window_opened = True
    sim._visualization_runtime = None

    with pytest.raises(RuntimeError, match="native DexSim window"):
        sim.start_visualization()


def test_constructor_only_declares_spawn_scene(monkeypatch) -> None:
    lifecycle: list[str] = []
    spawn_scene = MagicMock()
    world = MagicMock()
    world.get_physics_scene.return_value = MagicMock()
    world.get_env.return_value = MagicMock()

    monkeypatch.setattr(
        sim_manager_module.os, "makedirs", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(sim_manager_module.wp, "init", lambda: None)
    monkeypatch.setattr(sim_manager_module.dexsim, "World", lambda _cfg: world)
    monkeypatch.setattr(
        sim_manager_module,
        "SpawnScene",
        lambda *_args, **_kwargs: spawn_scene,
    )
    monkeypatch.setattr(
        sim_manager_module.dexsim, "set_physics_config", lambda **_kwargs: None
    )
    monkeypatch.setattr(
        sim_manager_module.dexsim,
        "set_physics_gpu_memory_config",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        SimulationManager, "_convert_sim_config", lambda _self, _cfg: object()
    )
    monkeypatch.setattr(
        SimulationManager, "enable_physics", lambda _self, _enable: None
    )
    monkeypatch.setattr(
        SimulationManager,
        "_init_sim_resources",
        lambda _self: lifecycle.append("resources"),
    )
    monkeypatch.setattr(
        SimulationManager,
        "_declare_spawn_default_plane",
        lambda _self: lifecycle.append("plane"),
    )
    monkeypatch.setattr(
        SimulationManager,
        "set_default_background",
        lambda _self: lifecycle.append("background"),
    )
    monkeypatch.setattr(
        SimulationManager,
        "set_default_global_lighting",
        lambda _self: lifecycle.append("lighting"),
    )

    def start_visualization(sim: SimulationManager) -> None:
        lifecycle.append(f"visualization:{sim.num_envs}")

    monkeypatch.setattr(
        SimulationManager,
        "start_visualization",
        start_visualization,
    )

    sim = object.__new__(SimulationManager)
    SimulationManager.__init__(sim, SimulationManagerCfg(num_envs=3))

    assert lifecycle == [
        "resources",
        "background",
        "plane",
        "lighting",
    ]
    assert sim._spawn_scene is spawn_scene
    assert sim._arenas == []


@pytest.mark.parametrize(
    ("backend", "device", "expected_gpu_init_calls"),
    [
        pytest.param("default", torch.device("cpu"), 0, id="default-host"),
        pytest.param("default", torch.device("cuda"), 1, id="default-accelerator"),
        pytest.param("newton", torch.device("cpu"), 0, id="newton-host"),
        pytest.param("newton", torch.device("cuda"), 0, id="newton-accelerator"),
    ],
)
def test_prepare_initializes_default_gpu_runtime(
    backend: str,
    device: torch.device,
    expected_gpu_init_calls: int,
) -> None:
    result = MagicMock()
    spawn_scene = MagicMock()
    spawn_scene.builder.is_finalized = False
    spawn_scene.builder.result = None
    spawn_scene.commit.return_value = result
    spawn_scene.arena_names = ["arena_0"]

    sim = object.__new__(SimulationManager)
    sim.physics = SimpleNamespace(name=backend)
    sim.device = device
    sim._world = MagicMock()
    sim._spawn_scene = spawn_scene
    sim._default_plane = object()
    sim._pending_sensor_attachments = []

    sim.prepare()

    assert sim._world.init_gpu_physics.call_count == expected_gpu_init_calls


def test_remove_asset_marks_visualization_topology_dirty() -> None:
    sim, runtime = _make_visualization_sim_manager()
    rigid_object = MagicMock()
    spawn_scene = MagicMock()
    spawn_scene.__contains__.return_value = True
    spawn_scene.result = object()
    sim._spawn_scene = spawn_scene
    sim.prepare = MagicMock()
    sim._rigid_objects = {"cube": rigid_object}
    sim._articulations = {}
    sim._robots = {}
    sim._lights = {}

    assert sim.remove_asset("cube")

    spawn_scene.remove.assert_called_once_with("cube")
    sim.prepare.assert_called_once_with()
    rigid_object.destroy.assert_not_called()
    assert "cube" not in sim._rigid_objects
    assert sim._visualization_topology_revision == 3
    sim.stop_visualization()
    assert runtime.stopped


def test_add_stereo_camera_marks_visualization_topology_dirty() -> None:
    sim = object.__new__(SimulationManager)
    sensor = object.__new__(sim_manager_module.StereoCamera)
    sim.device = torch.device("cpu")
    sim._sensors = {}
    sim._visualization_topology_revision = 2
    sim.SUPPORTED_SENSOR_TYPES = {
        "StereoCamera": lambda cfg, device: sensor,
    }
    cfg = SimpleNamespace(sensor_type="StereoCamera", uid="cam_high")

    assert sim.add_sensor(cfg) is sensor
    assert sim._visualization_topology_revision == 3


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
