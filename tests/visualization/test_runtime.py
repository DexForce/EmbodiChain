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

import threading
from dataclasses import dataclass, field
from time import monotonic, sleep

import numpy as np

from embodichain.lab.visualization import (
    CameraImage,
    CameraImageCaptureResult,
    CameraImageFrame,
    CaptureResult,
    GizmoCommand,
    GizmoCommandQueue,
    LatestFrameQueue,
    SceneFrame,
    SceneManifest,
    VisualizationCfg,
    VisualizationRuntime,
)
from embodichain.lab.visualization.backends.base import VisualizationBackend


def _frame(sequence: int, scene_revision: int = 1) -> SceneFrame:
    return SceneFrame(
        run_id="run",
        scene_revision=scene_revision,
        sequence=sequence,
        sim_step=sequence,
        sim_time=float(sequence),
        node_ids=(),
        positions=np.empty((0, 3), dtype=np.float32),
        wxyz=np.empty((0, 4), dtype=np.float32),
        visible=np.empty((0,), dtype=np.bool_),
    )


def test_latest_frame_queue_replaces_unconsumed_frame() -> None:
    frames = LatestFrameQueue()

    assert not frames.put_latest(_frame(1))
    assert frames.put_latest(_frame(2))

    assert frames.get_nowait().sequence == 2


def _gizmo_command(sequence: int, phase: str) -> GizmoCommand:
    return GizmoCommand(
        run_id="run",
        scene_revision=1,
        sequence=sequence,
        gizmo_id="cube",
        phase=phase,
        client_id="client-a",
        position=np.array([float(sequence), 0.0, 0.0], dtype=np.float32),
        wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )


def test_gizmo_command_queue_coalesces_updates_but_retains_lifecycle() -> None:
    commands = GizmoCommandQueue(maxsize=3)

    commands.put(_gizmo_command(1, "start"))
    commands.put(_gizmo_command(2, "update"))
    commands.put(_gizmo_command(3, "update"))
    commands.put(_gizmo_command(4, "end"))

    drained = commands.drain()

    assert [(command.sequence, command.phase) for command in drained] == [
        (1, "start"),
        (3, "update"),
        (4, "end"),
    ]


@dataclass
class _Exporter:
    published: threading.Event
    scene_revision: int = 0
    dynamic_capture_flags: list[bool] = field(default_factory=list)
    image_capture_count: int = 0

    @property
    def has_cameras(self) -> bool:
        return True

    @property
    def has_deformables(self) -> bool:
        return True

    def build_manifest(self) -> SceneManifest:
        self.scene_revision += 1
        return SceneManifest(
            run_id="run",
            scene_revision=self.scene_revision,
            nodes=(),
            geometries=(),
        )

    def capture(self, **kwargs: object) -> CaptureResult:
        self.dynamic_capture_flags.append(
            bool(kwargs.get("capture_dynamic_geometry", False))
        )
        return CaptureResult(
            frame=_frame(1, self.scene_revision),
            capture_seconds=0.001,
        )

    def capture_camera_images(self, **_: object) -> CameraImageCaptureResult:
        self.image_capture_count += 1
        return CameraImageCaptureResult(
            frame=CameraImageFrame(
                run_id="run",
                scene_revision=self.scene_revision,
                sequence=1,
                sim_step=1,
                sim_time=0.01,
                images=(
                    CameraImage(
                        camera_id="env:0/camera:test",
                        image=np.zeros((2, 3, 3), dtype=np.uint8),
                    ),
                ),
            ),
            capture_seconds=0.002,
        )


def test_runtime_captures_images_on_every_step_without_fps_limit() -> None:
    """A ``None`` image FPS captures at every eligible simulation step."""
    published = threading.Event()
    image_published = threading.Event()
    exporter = _Exporter(published)
    runtime = VisualizationRuntime(
        exporter,
        VisualizationCfg(backend="viser", sensor_image_fps=None),
        backend=_Backend(published, image_published),
    )

    runtime.start()
    runtime.capture(sim_step=1, sim_time=0.01)
    runtime.capture(sim_step=2, sim_time=0.02)
    runtime.stop()

    assert exporter.image_capture_count == 2


class _Backend(VisualizationBackend):
    def __init__(
        self,
        published: threading.Event,
        image_published: threading.Event,
    ) -> None:
        self.published = published
        self.image_published = image_published
        self.thread_ids: list[int] = []
        self.stopped = False

    @property
    def endpoint(self) -> str:
        return "http://localhost:1234"

    @property
    def client_count(self) -> int:
        return 0

    def _record_thread(self) -> None:
        self.thread_ids.append(threading.get_ident())

    def start(self) -> None:
        self._record_thread()

    def publish_manifest(self, manifest: SceneManifest) -> None:
        self._record_thread()

    def publish_frame(self, frame: SceneFrame) -> bool:
        self._record_thread()
        self.published.set()
        return True

    def publish_camera_images(self, frame: CameraImageFrame) -> bool:
        self._record_thread()
        self.image_published.set()
        return True

    def poll(self) -> None:
        self._record_thread()

    def stop(self) -> None:
        self._record_thread()
        self.stopped = True


def test_runtime_keeps_backend_lifecycle_on_update_thread() -> None:
    published = threading.Event()
    image_published = threading.Event()
    backend = _Backend(published, image_published)
    cfg = VisualizationCfg(backend="viser")
    exporter = _Exporter(published)
    runtime = VisualizationRuntime(exporter, cfg, backend=backend)

    runtime.start()
    assert runtime.endpoint == "http://localhost:1234"
    assert runtime.health.status == "running"
    assert runtime.health.published_scene_revision == 1
    assert runtime.capture(sim_step=1, sim_time=0.01, force=True)
    assert published.wait(timeout=2.0)
    assert image_published.wait(timeout=2.0)
    runtime._next_capture_time = 0.0
    runtime._next_deformable_capture_time = float("inf")
    runtime._next_image_capture_time = float("inf")
    assert runtime.capture(sim_step=2, sim_time=0.02)
    assert exporter.dynamic_capture_flags == [True, False]
    runtime.refresh_scene()
    deadline = monotonic() + 2.0
    while runtime.health.published_scene_revision != 2 and monotonic() < deadline:
        sleep(0.01)
    assert runtime.health.published_scene_revision == 2
    runtime.stop()

    assert backend.stopped
    assert len(set(backend.thread_ids)) == 1
    assert runtime.health.status == "stopped"
    assert runtime.stats.captured_frames == 2
    assert runtime.stats.published_frames >= 1
    assert runtime.stats.captured_image_frames == 1
    assert runtime.stats.published_image_frames == 1
