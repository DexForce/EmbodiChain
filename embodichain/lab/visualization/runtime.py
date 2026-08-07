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

import queue
import threading
from collections import deque
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Generic, TypeVar

from .backends.base import VisualizationBackend
from .cfg import VisualizationCfg
from .protocol import (
    CameraImageFrame,
    GizmoCommand,
    JointControlCommand,
    JointControlProvider,
    SceneFrame,
    SceneManifest,
    SceneOverlays,
    estimate_camera_image_frame_bytes,
    estimate_frame_bytes,
    estimate_manifest_bytes,
)
from .scene_exporter import SceneExporter

__all__ = [
    "GizmoCommandQueue",
    "JointControlCommandQueue",
    "LatestFrameQueue",
    "RuntimeHealth",
    "RuntimeStats",
    "VisualizationRuntime",
]


FrameT = TypeVar("FrameT")


class LatestFrameQueue(Generic[FrameT]):
    """A one-slot queue where producers replace an unconsumed old frame."""

    def __init__(self) -> None:
        self._queue: queue.Queue[FrameT] = queue.Queue(maxsize=1)

    def put_latest(self, frame: FrameT) -> bool:
        """Enqueue ``frame`` and return whether an older frame was dropped."""
        dropped = False
        try:
            self._queue.put_nowait(frame)
            return dropped
        except queue.Full:
            pass
        try:
            self._queue.get_nowait()
            dropped = True
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            # Another producer won the race. Its newer-or-equal sample is retained.
            dropped = True
        return dropped

    def get(self, timeout: float | None = None) -> FrameT:
        """Return the queued frame, waiting for up to ``timeout`` seconds."""
        return self._queue.get(timeout=timeout)

    def get_nowait(self) -> FrameT:
        """Return the queued frame without blocking."""
        return self._queue.get_nowait()

    def clear(self) -> None:
        """Discard any queued frame."""
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass


class GizmoCommandQueue:
    """Bounded command queue that coalesces high-rate drag updates.

    Drag lifecycle commands are retained. When the queue reaches its soft
    capacity, an older ``update`` for the same Gizmo/client is replaced first.
    """

    def __init__(self, maxsize: int = 256) -> None:
        if maxsize <= 0:
            raise ValueError("maxsize must be greater than zero.")
        self._maxsize = maxsize
        self._commands: deque[GizmoCommand] = deque()
        self._lock = threading.Lock()

    def put(self, command: GizmoCommand) -> None:
        """Enqueue a command without blocking the Viser callback thread."""
        with self._lock:
            if command.phase == "update":
                for index in range(len(self._commands) - 1, -1, -1):
                    queued = self._commands[index]
                    if (
                        queued.phase == "update"
                        and queued.gizmo_id == command.gizmo_id
                        and queued.client_id == command.client_id
                    ):
                        self._commands[index] = command
                        return
            if len(self._commands) >= self._maxsize:
                for index, queued in enumerate(self._commands):
                    if queued.phase == "update":
                        del self._commands[index]
                        break
            self._commands.append(command)

    def drain(self) -> tuple[GizmoCommand, ...]:
        """Return and clear all queued commands in arrival order."""
        with self._lock:
            commands = tuple(self._commands)
            self._commands.clear()
        return commands

    def clear(self) -> None:
        """Discard all queued commands."""
        with self._lock:
            self._commands.clear()


class JointControlCommandQueue:
    """Bounded queue that keeps only the newest value for each joint control."""

    def __init__(self, maxsize: int = 256) -> None:
        if maxsize <= 0:
            raise ValueError("maxsize must be greater than zero.")
        self._maxsize = maxsize
        self._commands: deque[JointControlCommand] = deque()
        self._lock = threading.Lock()

    def put(self, command: JointControlCommand) -> None:
        """Enqueue a value without blocking the Viser callback thread."""
        with self._lock:
            for index in range(len(self._commands) - 1, -1, -1):
                if self._commands[index].control_id == command.control_id:
                    del self._commands[index]
                    self._commands.append(command)
                    return
            if len(self._commands) >= self._maxsize:
                self._commands.popleft()
            self._commands.append(command)

    def drain(self) -> tuple[JointControlCommand, ...]:
        """Return and clear all queued commands in arrival order."""
        with self._lock:
            commands = tuple(self._commands)
            self._commands.clear()
        return commands

    def clear(self) -> None:
        """Discard all queued commands."""
        with self._lock:
            self._commands.clear()


@dataclass(frozen=True)
class RuntimeStats:
    """Snapshot of scene and camera-image capture/upload telemetry."""

    captured_frames: int = 0
    published_frames: int = 0
    dropped_frames: int = 0
    rejected_frames: int = 0
    manifest_bytes: int = 0
    frame_bytes: int = 0
    capture_seconds: float = 0.0
    upload_seconds: float = 0.0
    captured_image_frames: int = 0
    published_image_frames: int = 0
    dropped_image_frames: int = 0
    rejected_image_frames: int = 0
    image_bytes: int = 0
    image_capture_seconds: float = 0.0
    image_upload_seconds: float = 0.0


@dataclass(frozen=True)
class RuntimeHealth:
    """Current visualization runtime health and connection state."""

    status: str
    running: bool
    endpoint: str | None
    client_count: int
    published_scene_revision: int
    worker_error: str | None = None


class VisualizationRuntime:
    """Run scene capture and a visualization backend without blocking simulation.

    The simulation thread calls :meth:`capture`. Viser creation and all handle
    mutations occur on one private update thread. The frame queue always keeps
    the newest sample, preventing visualization overload from accumulating lag.

    Args:
        exporter: Scene exporter bound to a simulation manager.
        cfg: Visualization, frame rate, and Viser server configuration.
        backend: Optional backend injection hook used by tests and alternate UIs.
    """

    def __init__(
        self,
        exporter: SceneExporter,
        cfg: VisualizationCfg,
        backend: VisualizationBackend | None = None,
    ) -> None:
        if cfg.backend != "viser":
            raise ValueError("VisualizationRuntime currently requires backend='viser'.")
        self.exporter = exporter
        self.cfg = cfg
        if backend is None:
            from .backends.viser import ViserBackend

            backend = ViserBackend(
                cfg.viser_server,
                allow_commands=cfg.allow_commands,
            )
        self._backend = backend
        self._gizmo_commands = GizmoCommandQueue()
        self._joint_control_commands = JointControlCommandQueue()
        self._backend.set_gizmo_command_sink(self._enqueue_gizmo_command)
        self._backend.set_joint_control_command_sink(
            self._enqueue_joint_control_command
        )
        self._backend.set_replay_control_command_sink(
            self._enqueue_replay_control_command
        )
        self._frames: LatestFrameQueue[SceneFrame] = LatestFrameQueue()
        self._camera_images: LatestFrameQueue[CameraImageFrame] = LatestFrameQueue()
        self._replay_control_states: LatestFrameQueue[tuple[int, int, bool]] = (
            LatestFrameQueue()
        )
        self._replay_control_commands: LatestFrameQueue[int] = LatestFrameQueue()
        self._manifests: queue.Queue[SceneManifest] = queue.Queue()
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._worker_error: BaseException | None = None
        self._published_scene_revision = 0
        self._next_capture_time = 0.0
        self._next_image_capture_time = 0.0
        self._next_deformable_capture_time = 0.0
        self._stats = RuntimeStats()
        self._stats_lock = threading.Lock()

    def _enqueue_gizmo_command(self, command: GizmoCommand) -> None:
        if self.cfg.allow_commands:
            self._gizmo_commands.put(command)

    def drain_gizmo_commands(self) -> tuple[GizmoCommand, ...]:
        """Drain browser Gizmo commands for simulation-thread processing."""
        if not self.cfg.allow_commands:
            return ()
        return self._gizmo_commands.drain()

    def _enqueue_joint_control_command(self, command: JointControlCommand) -> None:
        if self.cfg.allow_commands:
            self._joint_control_commands.put(command)

    def drain_joint_control_commands(self) -> tuple[JointControlCommand, ...]:
        """Drain browser joint commands for simulation-thread processing."""
        if not self.cfg.allow_commands:
            return ()
        return self._joint_control_commands.drain()

    def _enqueue_replay_control_command(self, step: int) -> None:
        if self.cfg.allow_commands:
            self._replay_control_commands.put_latest(step)

    def drain_replay_control_command(self) -> int | None:
        """Return the newest browser replay seek, if one is pending.

        Returns:
            Requested trajectory step, or ``None`` when no seek is pending.
        """
        if not self.cfg.allow_commands:
            return None
        try:
            return self._replay_control_commands.get_nowait()
        except queue.Empty:
            return None

    def publish_replay_control(
        self,
        *,
        step: int,
        max_step: int,
        visible: bool = True,
    ) -> None:
        """Asynchronously publish trajectory replay progress to Viser.

        Args:
            step: Current trajectory step.
            max_step: Largest valid trajectory step.
            visible: Whether the replay control should be visible.

        Raises:
            RuntimeError: If the visualization runtime is not running.
            ValueError: If the step range is invalid.
        """
        if not self.is_running:
            raise RuntimeError("VisualizationRuntime.start() must be called first.")
        if max_step < 0 or not 0 <= step <= max_step:
            raise ValueError("Replay step must satisfy 0 <= step <= max_step.")
        self._raise_worker_error()
        self._replay_control_states.put_latest((step, max_step, visible))

    def set_joint_control_provider(
        self,
        provider: JointControlProvider | None,
    ) -> None:
        """Install a simulation-thread joint source for future scene captures.

        Registering a provider does not publish a new manifest by itself. The
        caller must refresh the scene after registration so the backend can
        build its controls.
        """
        self.exporter.set_joint_control_provider(provider)

    @property
    def endpoint(self) -> str | None:
        """Local browser endpoint after :meth:`start` returns."""
        return self._backend.endpoint

    @property
    def is_running(self) -> bool:
        """Whether the visualization update thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def stats(self) -> RuntimeStats:
        """Return an immutable telemetry snapshot."""
        with self._stats_lock:
            return replace(self._stats)

    @property
    def health(self) -> RuntimeHealth:
        """Return runtime, endpoint, client, and worker health information."""
        running = self.is_running
        error = self._worker_error
        if error is not None:
            status = "failed"
        elif running and self._ready_event.is_set():
            status = "running"
        elif running:
            status = "starting"
        else:
            status = "stopped"
        return RuntimeHealth(
            status=status,
            running=running,
            endpoint=self.endpoint,
            client_count=self._backend.client_count if running else 0,
            published_scene_revision=self._published_scene_revision,
            worker_error=repr(error) if error is not None else None,
        )

    def _update_stats(self, **changes: int | float) -> None:
        with self._stats_lock:
            values = self._stats.__dict__.copy()
            for key, delta in changes.items():
                values[key] += delta
            self._stats = RuntimeStats(**values)

    def _raise_worker_error(self) -> None:
        if self._worker_error is not None:
            raise RuntimeError(
                "Visualization update thread failed."
            ) from self._worker_error

    def start(self, timeout: float = 10.0) -> None:
        """Build the initial manifest and start the backend update thread."""
        if self.is_running:
            return
        manifest = self.exporter.build_manifest()
        self._update_stats(manifest_bytes=estimate_manifest_bytes(manifest))
        self._stop_event.clear()
        self._ready_event.clear()
        self._worker_error = None
        self._thread = threading.Thread(
            target=self._run,
            args=(manifest,),
            name="embodichain-visualization",
            daemon=True,
        )
        self._thread.start()
        if not self._ready_event.wait(timeout=timeout):
            self.stop(timeout=timeout)
            raise TimeoutError("Timed out while starting the visualization backend.")
        self._raise_worker_error()
        self._next_capture_time = perf_counter()
        self._next_image_capture_time = perf_counter()
        self._next_deformable_capture_time = perf_counter()

    def refresh_scene(self) -> SceneManifest:
        """Capture and asynchronously publish a new topology revision."""
        if not self.is_running:
            raise RuntimeError("VisualizationRuntime.start() must be called first.")
        self._raise_worker_error()
        manifest = self.exporter.build_manifest()
        self._frames.clear()
        self._camera_images.clear()
        self._manifests.put_nowait(manifest)
        self._update_stats(manifest_bytes=estimate_manifest_bytes(manifest))
        return manifest

    def capture(
        self,
        *,
        sim_step: int,
        sim_time: float,
        overlays: SceneOverlays | None = None,
        force: bool = False,
        capture_camera_images: bool = True,
    ) -> bool:
        """Capture a due frame and enqueue it without waiting for Viser.

        Args:
            sim_step: Current simulation step.
            sim_time: Current simulation time in seconds.
            overlays: Optional backend-neutral debug overlays.
            force: Ignore the configured scene FPS limiter.
            capture_camera_images: Whether camera images may be captured in
                this call. Simulation batches disable this for intermediate
                physics substeps.

        Returns:
            ``True`` when a frame was captured, otherwise ``False`` when limited.
        """
        if not self.is_running:
            raise RuntimeError("VisualizationRuntime.start() must be called first.")
        self._raise_worker_error()
        now = perf_counter()
        pose_due = force or now >= self._next_capture_time
        deformable_due = self.exporter.has_deformables and (
            force or now >= self._next_deformable_capture_time
        )
        scene_due = pose_due or deformable_due
        image_due = (
            capture_camera_images
            and self.exporter.has_cameras
            and (
                force
                or self.cfg.sensor_image_fps is None
                or now >= self._next_image_capture_time
            )
        )
        if not scene_due and not image_due:
            return False
        if scene_due:
            if pose_due:
                self._next_capture_time = now + 1.0 / self.cfg.scene_fps
            if deformable_due:
                self._next_deformable_capture_time = now + 1.0 / self.cfg.soft_body_fps
            result = self.exporter.capture(
                sim_step=sim_step,
                sim_time=sim_time,
                overlays=overlays,
                capture_dynamic_geometry=deformable_due,
            )
            dropped = self._frames.put_latest(result.frame)
            self._update_stats(
                captured_frames=1,
                dropped_frames=int(dropped),
                frame_bytes=estimate_frame_bytes(result.frame),
                capture_seconds=result.capture_seconds,
            )
        if image_due:
            if self.cfg.sensor_image_fps is not None:
                self._next_image_capture_time = now + 1.0 / self.cfg.sensor_image_fps
            image_result = self.exporter.capture_camera_images(
                sim_step=sim_step,
                sim_time=sim_time,
            )
            if image_result.frame.images:
                image_dropped = self._camera_images.put_latest(image_result.frame)
                self._update_stats(
                    captured_image_frames=1,
                    dropped_image_frames=int(image_dropped),
                    image_bytes=estimate_camera_image_frame_bytes(image_result.frame),
                    image_capture_seconds=image_result.capture_seconds,
                )
        return True

    def _publish_pending_manifests(self) -> None:
        while True:
            try:
                manifest = self._manifests.get_nowait()
            except queue.Empty:
                return
            self._backend.publish_manifest(manifest)
            self._published_scene_revision = manifest.scene_revision

    def _publish_pending_camera_images(self) -> None:
        try:
            frame = self._camera_images.get_nowait()
        except queue.Empty:
            return
        started = perf_counter()
        accepted = self._backend.publish_camera_images(frame)
        self._update_stats(
            published_image_frames=int(accepted),
            rejected_image_frames=int(not accepted),
            image_upload_seconds=perf_counter() - started,
        )

    def _publish_pending_replay_control(self) -> None:
        try:
            step, max_step, visible = self._replay_control_states.get_nowait()
        except queue.Empty:
            return
        self._backend.publish_replay_control(
            step=step,
            max_step=max_step,
            visible=visible,
        )

    def _run(self, initial_manifest: SceneManifest) -> None:
        try:
            self._backend.start()
            self._backend.publish_manifest(initial_manifest)
            self._published_scene_revision = initial_manifest.scene_revision
            self._ready_event.set()
            while not self._stop_event.is_set():
                self._publish_pending_manifests()
                self._publish_pending_camera_images()
                self._publish_pending_replay_control()
                try:
                    frame = self._frames.get(timeout=0.05)
                except queue.Empty:
                    self._publish_pending_camera_images()
                    self._publish_pending_replay_control()
                    self._backend.poll()
                    continue
                # A topology refresh and its first frame can be queued while this
                # thread is blocked above. Publish the manifest before that frame.
                self._publish_pending_manifests()
                self._publish_pending_camera_images()
                self._publish_pending_replay_control()
                started = perf_counter()
                accepted = self._backend.publish_frame(frame)
                upload_seconds = perf_counter() - started
                self._update_stats(
                    published_frames=int(accepted),
                    rejected_frames=int(not accepted),
                    upload_seconds=upload_seconds,
                )
            self._publish_pending_manifests()
            self._publish_pending_camera_images()
            self._publish_pending_replay_control()
            try:
                final_frame = self._frames.get_nowait()
            except queue.Empty:
                final_frame = None
            if final_frame is not None:
                started = perf_counter()
                accepted = self._backend.publish_frame(final_frame)
                self._update_stats(
                    published_frames=int(accepted),
                    rejected_frames=int(not accepted),
                    upload_seconds=perf_counter() - started,
                )
            self._publish_pending_camera_images()
            self._publish_pending_replay_control()
        except BaseException as error:
            self._worker_error = error
            self._ready_event.set()
        finally:
            try:
                self._backend.stop()
            except BaseException as error:
                if self._worker_error is None:
                    self._worker_error = error

    def stop(self, timeout: float = 10.0) -> None:
        """Stop the backend and reclaim its server port and update thread."""
        thread = self._thread
        if thread is None:
            return
        self._stop_event.set()
        thread.join(timeout=timeout)
        if thread.is_alive():
            raise TimeoutError("Timed out while stopping the visualization backend.")
        self._thread = None
        self._frames.clear()
        self._camera_images.clear()
        self._replay_control_states.clear()
        self._replay_control_commands.clear()
        self._gizmo_commands.clear()
        self._joint_control_commands.clear()
        self._raise_worker_error()

    def __enter__(self) -> VisualizationRuntime:
        self.start()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.stop()
