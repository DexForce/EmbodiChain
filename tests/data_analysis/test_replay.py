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
import subprocess
import sys
import threading
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from embodichain.data_analysis.replay import ReplayViewer
from embodichain.data_analysis.recording import write_recording


class _FakeBackend:
    instances: list[_FakeBackend] = []

    def __init__(self, cfg: object, *, allow_commands: bool = False) -> None:
        del allow_commands
        self.cfg = cfg
        self.endpoint = f"http://{cfg.host}:{cfg.port}"
        self.thread_ids: list[int] = []
        self.manifests: list[object] = []
        self.frames: list[object] = []
        self.camera_frames: list[object] = []
        self.controls: list[tuple[int, int, bool]] = []
        self.stopped = False
        self._sink = None
        self.frame_event = threading.Event()
        self.render_hook = None
        self.instances.append(self)

    def _track(self) -> None:
        self.thread_ids.append(threading.get_ident())

    def start(self) -> None:
        self._track()

    def set_replay_control_command_sink(self, sink: object) -> None:
        self._sink = sink

    def publish_manifest(self, manifest: object) -> None:
        self._track()
        self.manifests.append(manifest)

    def publish_frame(self, frame: object) -> bool:
        self._track()
        self.frames.append(frame)
        if self.render_hook is not None:
            self.render_hook()
        self.frame_event.set()
        return True

    def publish_camera_images(self, frame: object) -> bool:
        self._track()
        self.camera_frames.append(frame)
        return True

    def publish_replay_control(
        self, *, step: int, max_step: int, visible: bool
    ) -> None:
        self._track()
        self.controls.append((step, max_step, visible))

    def poll(self) -> None:
        self._track()

    def stop(self) -> None:
        self._track()
        self.stopped = True


def _arrays(offset: float = 0.0) -> dict[str, np.ndarray]:
    timestamps = np.array([0.0, 0.1, 0.4], dtype=np.float64)
    positions = np.array(
        [[offset, 0.0, 0.0], [offset + 0.1, 0.0, 0.1], [offset + 0.2, 0.0, 0.0]],
        dtype=np.float32,
    )
    return {
        "timestamps": timestamps,
        "qpos": np.zeros((3, 2), dtype=np.float32),
        "tcp_position": positions,
        "object_position": positions + np.array([0.0, 0.1, 0.0], dtype=np.float32),
    }


def _record(
    tmp_path: Path, episode_id: str, *, offset: float = 0.0, camera: bool = False
) -> dict[str, object]:
    scene = {
        "run_id": episode_id,
        "scene_revision": 0,
        "nodes": [],
        "geometries": [],
        "cameras": [],
        "gizmos": [],
        "joint_controls": [],
        "schema_version": 5,
        "up_direction": "+z",
        "length_unit": "meter",
    }
    frames = np.zeros((3, 2, 2, 3), dtype=np.uint8) if camera else None
    artifacts = write_recording(
        tmp_path / episode_id, _arrays(offset), scene, camera=frames
    )
    return {"episode_id": episode_id, "artifacts": artifacts}


def test_seek_uses_nearest_sample_and_all_backend_mutations_share_owner_thread(
    tmp_path: Path,
) -> None:
    _FakeBackend.instances.clear()
    viewer = ReplayViewer(port=0, _backend_factory=_FakeBackend)
    try:
        viewer.load(_record(tmp_path, "primary", camera=True))
        assert viewer.seek(0.26) == 2
        backend = _FakeBackend.instances[-1]
        assert backend.frames[-1].sim_step == 2
        assert backend.camera_frames[-1].sim_step == 2
        assert len(set(backend.thread_ids)) == 1
        assert viewer.url.startswith("http://127.0.0.1:")
    finally:
        viewer.close()
    assert backend.stopped


def test_episode_switch_and_compare_publish_new_manifest_and_two_paths(
    tmp_path: Path,
) -> None:
    _FakeBackend.instances.clear()
    viewer = ReplayViewer(port=0, _backend_factory=_FakeBackend)
    try:
        first = _record(tmp_path, "first")
        second = _record(tmp_path, "second", offset=1.0)
        viewer.load(first)
        viewer.load(second, compare_record=first)
        backend = _FakeBackend.instances[-1]
        assert len(backend.manifests) == 2
        overlays = backend.frames[-1].overlays.trajectories
        assert [overlay.overlay_id for overlay in overlays] == [
            "primary-tcp-path",
            "compare-tcp-path",
        ]
    finally:
        viewer.close()


class _ControlledClock:
    def __init__(self, value: float) -> None:
        self.value = value
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self.value

    def advance(self, seconds: float) -> None:
        with self._lock:
            self.value += seconds


class _CameraState:
    def __init__(self) -> None:
        self.thread_ids: list[int] = []
        self._position = None
        self._look_at = None
        self._up_direction = None

    def _set(self, name: str, value: object) -> None:
        self.thread_ids.append(threading.get_ident())
        setattr(self, f"_{name}", np.asarray(value))

    position = property(
        lambda self: self._position, lambda self, value: self._set("position", value)
    )
    look_at = property(
        lambda self: self._look_at, lambda self, value: self._set("look_at", value)
    )
    up_direction = property(
        lambda self: self._up_direction,
        lambda self, value: self._set("up_direction", value),
    )


class _ClientServer:
    def __init__(self) -> None:
        self.client = SimpleNamespace(camera=_CameraState())
        self.connect_callback = None
        self.gui = SimpleNamespace(
            add_checkbox=lambda *args, **kwargs: SimpleNamespace(
                on_update=lambda callback: callback
            )
        )

    def on_client_connect(self, callback: object) -> object:
        self.connect_callback = callback
        return callback

    def get_clients(self) -> dict[str, object]:
        return {"client": self.client}


class _FramingBackend(_FakeBackend):
    def __init__(self, cfg: object, *, allow_commands: bool = False) -> None:
        super().__init__(cfg, allow_commands=allow_commands)
        self._server = _ClientServer()


def test_playback_uses_absolute_irregular_timestamps_without_render_drift(
    tmp_path: Path,
) -> None:
    _FakeBackend.instances.clear()
    clock = _ControlledClock(100.0)
    viewer = ReplayViewer(port=0, _backend_factory=_FakeBackend, _clock=clock)
    try:
        viewer.load(_record(tmp_path, "irregular"))
        backend = _FakeBackend.instances[-1]
        backend.frame_event.clear()
        backend.render_hook = lambda: clock.advance(0.08)

        viewer.play()
        assert not backend.frame_event.wait(timeout=0.03)
        clock.advance(0.1)
        assert backend.frame_event.wait(timeout=0.2)
        assert backend.frames[-1].sim_step == 1
        backend.frame_event.clear()

        # Rendering step 1 advanced the wall clock from 100.10 to 100.18. The
        # final sample remains due at the absolute recording offset 100.40.
        clock.advance(0.21)
        assert not backend.frame_event.wait(timeout=0.03)
        clock.advance(0.011)
        assert backend.frame_event.wait(timeout=0.2)
        assert backend.frames[-1].sim_step == 2
        assert clock() == pytest.approx(100.481)
    finally:
        viewer.close()


def test_loaded_episode_frames_existing_and_new_clients_from_recorded_bounds(
    tmp_path: Path,
) -> None:
    _FakeBackend.instances.clear()
    viewer = ReplayViewer(port=0, _backend_factory=_FramingBackend)
    try:
        viewer.load(_record(tmp_path, "framed"))
        backend = _FakeBackend.instances[-1]
        first_camera = backend._server.client.camera
        np.testing.assert_allclose(first_camera.look_at, [0.1, 0.05, 0.05])
        assert len(set(first_camera.thread_ids)) == 1
        assert first_camera.thread_ids[0] == backend.thread_ids[0]

        new_client = SimpleNamespace(camera=_CameraState())
        backend._server.connect_callback(new_client)
        assert new_client.camera.thread_ids == []
        viewer.seek(0.0)  # wakes the owner thread after the connect callback
        assert new_client.camera.thread_ids == [backend.thread_ids[0]] * 3
        np.testing.assert_allclose(new_client.camera.look_at, [0.1, 0.05, 0.05])
    finally:
        viewer.close()


def test_malformed_trajectory_is_rejected_without_replacing_loaded_episode(
    tmp_path: Path,
) -> None:
    _FakeBackend.instances.clear()
    viewer = ReplayViewer(port=0, _backend_factory=_FakeBackend)
    try:
        valid = _record(tmp_path, "valid")
        viewer.load(valid)
        bad_path = tmp_path / "bad.npz"
        np.savez(bad_path, timestamps=np.array([0.0]))
        with pytest.raises(ValueError, match="Trajectory requires"):
            viewer.load(
                {"episode_id": "bad", "artifacts": {"trajectory": str(bad_path)}}
            )
        assert viewer.episode_id == "valid"
    finally:
        viewer.close()


def test_import_is_offline_when_dexsim_imports_are_blocked() -> None:
    script = """
import builtins
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'dexsim' or name.startswith('dexsim.'):
        raise AssertionError(f'forbidden import: {name}')
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
from embodichain.data_analysis.replay import ReplayViewer
import embodichain.lab as lab
assert lab.__all__ == ['devices', 'task_program', 'gym', 'sim', 'visualization']
print(ReplayViewer.__name__)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ReplayViewer")
