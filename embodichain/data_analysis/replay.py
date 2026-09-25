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

"""Offline Viser replay for portable episode recordings."""

from __future__ import annotations

import json
import queue
import socket
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from time import monotonic
from typing import Any, Callable, Mapping

import numpy as np

from embodichain.lab.visualization.backends.viser import ViserBackend
from embodichain.lab.visualization.cfg import ViserServerCfg
from embodichain.lab.visualization.protocol import (
    CameraImage,
    CameraImageFrame,
    CameraSpec,
    FrameOverlay,
    MeshGeometry,
    SceneFrame,
    SceneManifest,
    SceneNode,
    SceneOverlays,
    TrajectoryOverlay,
)

from .recording import load_trajectory

__all__ = ["ReplayViewer"]


@dataclass(frozen=True)
class _Episode:
    episode_id: str
    trajectory: dict[str, np.ndarray]
    manifest: SceneManifest
    camera_frames: np.ndarray | None
    camera_timestamps: np.ndarray | None


@dataclass(frozen=True)
class _Command:
    name: str
    value: object
    response: queue.Queue[object]


def _free_port(host: str) -> int:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as stream:
        stream.bind((host, 0))
        return int(stream.getsockname()[1])


def _manifest(value: Mapping[str, Any], *, episode_id: str) -> SceneManifest:
    geometries = tuple(
        MeshGeometry(
            geometry_id=str(item["geometry_id"]),
            vertices=np.asarray(item["vertices"], dtype=np.float32),
            faces=np.asarray(item["faces"], dtype=np.uint32),
            color=tuple(item.get("color", (90, 200, 255))),
        )
        for item in value.get("geometries", [])
    )
    nodes = tuple(
        SceneNode(
            node_id=str(item["node_id"]),
            path=str(item["path"]),
            parent_id=item.get("parent_id"),
            env_id=int(item["env_id"]),
            kind=str(item["kind"]),
            geometry_id=str(item["geometry_id"]),
            dynamic_geometry=bool(item.get("dynamic_geometry", False)),
            visible=bool(item.get("visible", True)),
        )
        for item in value.get("nodes", [])
    )
    cameras = tuple(
        CameraSpec(
            camera_id=str(item["camera_id"]),
            sensor_uid=str(item["sensor_uid"]),
            env_id=int(item["env_id"]),
            path=str(item["path"]),
            fov_y=float(item["fov_y"]),
            aspect=float(item["aspect"]),
            near=float(item["near"]),
            far=float(item["far"]),
            role=str(item.get("role", "sensor")),
        )
        for item in value.get("cameras", [])
    )
    return SceneManifest(
        run_id=episode_id,
        scene_revision=int(value.get("scene_revision", 0)),
        nodes=nodes,
        geometries=geometries,
        cameras=cameras,
        up_direction=str(value.get("up_direction", "+z")),
        length_unit=str(value.get("length_unit", "meter")),
    )


def _read_episode(record: Mapping[str, Any]) -> _Episode:
    episode_id = record.get("episode_id")
    artifacts = record.get("artifacts")
    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError("Replay record requires a non-empty episode_id.")
    if not isinstance(artifacts, Mapping) or not isinstance(
        artifacts.get("trajectory"), str
    ):
        raise ValueError("Replay record requires a trajectory artifact path.")
    trajectory = load_trajectory(artifacts["trajectory"])
    scene_path = artifacts.get("scene")
    if scene_path is None:
        scene_value: Mapping[str, Any] = {}
    elif isinstance(scene_path, str):
        loaded = json.loads(Path(scene_path).read_text(encoding="utf-8"))
        if not isinstance(loaded, Mapping):
            raise ValueError("Scene artifact must contain a JSON object.")
        scene_value = loaded
    else:
        raise ValueError("Scene artifact path must be a string.")
    manifest = _manifest(scene_value, episode_id=episode_id)
    camera_frames = None
    camera_timestamps = None
    camera_path = artifacts.get("camera")
    if camera_path is not None:
        if not isinstance(camera_path, str):
            raise ValueError("Camera artifact path must be a string.")
        with np.load(camera_path, allow_pickle=False) as archive:
            if set(archive.files) != {"frames", "timestamps"}:
                raise ValueError("Camera artifact requires frames and timestamps.")
            camera_frames = archive["frames"]
            camera_timestamps = archive["timestamps"]
        if (
            camera_frames.dtype != np.uint8
            or camera_frames.ndim != 4
            or camera_frames.shape[-1] != 3
            or camera_timestamps.ndim != 1
            or len(camera_frames) != len(camera_timestamps)
            or len(camera_frames) == 0
            or not np.isfinite(camera_timestamps).all()
            or np.any(np.diff(camera_timestamps) <= 0.0)
        ):
            raise ValueError("Camera artifact contains invalid or unaligned samples.")
        if not manifest.cameras:
            height, width = camera_frames.shape[1:3]
            camera = CameraSpec(
                camera_id="record-camera",
                sensor_uid="record-camera",
                env_id=0,
                path="/world/record-camera",
                fov_y=float(np.pi / 3.0),
                aspect=float(width / height),
                near=0.01,
                far=100.0,
                role="record",
            )
            manifest = replace(manifest, cameras=(camera,))
    _validate_scene_samples(trajectory, manifest)
    return _Episode(episode_id, trajectory, manifest, camera_frames, camera_timestamps)


def _validate_scene_samples(
    trajectory: Mapping[str, np.ndarray], manifest: SceneManifest
) -> None:
    sample_count = len(trajectory["timestamps"])
    node_count = len(manifest.nodes)
    if "scene_positions" in trajectory and trajectory["scene_positions"].shape != (
        sample_count,
        node_count,
        3,
    ):
        raise ValueError("scene_positions do not match the scene manifest nodes.")
    camera_count = len(manifest.cameras)
    for key, width in (("camera_positions", 3), ("camera_wxyz", 4)):
        if key in trajectory and trajectory[key].shape != (
            sample_count,
            camera_count,
            width,
        ):
            raise ValueError(f"{key} do not match the scene manifest cameras.")


def _camera_view(episode: _Episode) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.concatenate(
        [episode.trajectory["tcp_position"], episode.trajectory["object_position"]],
        axis=0,
    )
    points = points[np.isfinite(points).all(axis=1)]
    if len(points):
        lower = points.min(axis=0)
        upper = points.max(axis=0)
        center = (lower + upper) * 0.5
        extent = float(np.linalg.norm(upper - lower))
    else:
        center = np.zeros(3, dtype=np.float32)
        extent = 1.0
    direction = np.array([-1.2, -1.4, 0.9], dtype=np.float32)
    direction /= np.linalg.norm(direction)
    eye = center + direction * max(0.8, extent * 2.2)
    return eye, center, np.array([0.0, 0.0, 1.0], dtype=np.float32)


def _set_client_camera(client: object, episode: _Episode) -> None:
    eye, center, up = _camera_view(episode)
    client.camera.position = eye
    client.camera.look_at = center
    client.camera.up_direction = up


class ReplayViewer:
    """Replay portable recordings through one thread-owned Viser backend."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        *,
        _backend_factory: Callable[..., object] = ViserBackend,
        _clock: Callable[[], float] = monotonic,
    ) -> None:
        if not host:
            raise ValueError("host must not be empty.")
        if not 0 <= port <= 65_535:
            raise ValueError("port must be between 0 and 65535.")
        selected_port = _free_port(host) if port == 0 else port
        self._commands: queue.Queue[_Command] = queue.Queue()
        self._ready = threading.Event()
        self._startup_error: BaseException | None = None
        self._url: str | None = None
        self._episode_id: str | None = None
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            args=(host, selected_port, _backend_factory, _clock),
            name="embodichain-offline-replay",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout=10.0):
            raise TimeoutError("Timed out starting offline Viser replay.")
        if self._startup_error is not None:
            raise RuntimeError(
                "Failed to start offline Viser replay."
            ) from self._startup_error

    @property
    def url(self) -> str:
        """Return the local Viser endpoint."""
        if self._url is None:
            raise RuntimeError("Replay viewer is not running.")
        return self._url

    @property
    def episode_id(self) -> str | None:
        """Return the currently loaded episode ID."""
        return self._episode_id

    def load(
        self, record: Mapping[str, Any], compare_record: Mapping[str, Any] | None = None
    ) -> None:
        """Load an episode and optional time-aligned comparison episode."""
        episode = _read_episode(record)
        compare = None if compare_record is None else _read_episode(compare_record)
        self._call("load", (episode, compare))
        self._episode_id = episode.episode_id

    def seek(self, time_s: float) -> int:
        """Seek to the sample nearest ``time_s`` and return its index."""
        if not np.isfinite(time_s):
            raise ValueError("Replay time must be finite.")
        return int(self._call("seek_time", float(time_s)))

    def play(self, playing: bool = True) -> None:
        """Start or pause recorded-cadence playback."""
        self._call("play", bool(playing))

    def close(self) -> None:
        """Stop playback and release the Viser server. Safe to call repeatedly."""
        if self._closed:
            return
        self._closed = True
        try:
            self._call("close", None)
        finally:
            self._thread.join(timeout=10.0)
            self._url = None

    def __enter__(self) -> ReplayViewer:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _call(self, name: str, value: object) -> object:
        if self._closed and name != "close":
            raise RuntimeError("Replay viewer is closed.")
        response: queue.Queue[object] = queue.Queue(maxsize=1)
        self._commands.put(_Command(name, value, response))
        result = response.get(timeout=10.0)
        if isinstance(result, BaseException):
            raise result
        return result

    def _run(
        self,
        host: str,
        port: int,
        backend_factory: Callable[..., object],
        clock: Callable[[], float],
    ) -> None:
        backend = None
        episode: _Episode | None = None
        compare: _Episode | None = None
        index = 0
        playing = False
        play_origin_wall = clock()
        play_origin_time = 0.0
        try:
            backend = backend_factory(
                ViserServerCfg(host=host, port=port, label="EmbodiChain Replay"),
                allow_commands=False,
            )
            backend.set_replay_control_command_sink(
                lambda step: self._enqueue_seek(step)
            )
            backend.start()
            self._install_client_framing(backend)
            self._url = backend.endpoint
        except BaseException as exc:
            self._startup_error = exc
            self._ready.set()
            return
        self._ready.set()
        running = True
        while running:
            timeout = 0.05
            if playing and episode is not None:
                timestamps = episode.trajectory["timestamps"]
                if index < len(timestamps) - 1:
                    deadline = (
                        play_origin_wall
                        + float(timestamps[index + 1])
                        - play_origin_time
                    )
                    timeout = min(0.01, max(0.0, deadline - clock()))
            try:
                command = self._commands.get(timeout=timeout)
            except queue.Empty:
                backend.poll()
                if playing and episode is not None:
                    timestamps = episode.trajectory["timestamps"]
                    if index >= len(episode.trajectory["timestamps"]) - 1:
                        playing = False
                    elif (
                        clock()
                        >= play_origin_wall
                        + float(timestamps[index + 1])
                        - play_origin_time
                    ):
                        index += 1
                        self._publish(backend, episode, compare, index)
                continue
            try:
                if command.name == "load":
                    episode, compare = command.value
                    assert isinstance(episode, _Episode)
                    index = 0
                    playing = False
                    backend.publish_manifest(episode.manifest)
                    self._install_play_control(backend)
                    self._publish(backend, episode, compare, index)
                    server = getattr(backend, "_server", None)
                    if server is not None:
                        clients = server.get_clients()
                        clients = (
                            clients.values()
                            if isinstance(clients, Mapping)
                            else clients
                        )
                        for client in clients:
                            _set_client_camera(client, episode)
                    command.response.put(None)
                elif command.name == "seek_time":
                    if episode is None:
                        raise RuntimeError("Load an episode before seeking.")
                    timestamps = episode.trajectory["timestamps"]
                    index = int(np.argmin(np.abs(timestamps - float(command.value))))
                    self._publish(backend, episode, compare, index)
                    if playing:
                        play_origin_wall = clock()
                        play_origin_time = float(timestamps[index])
                    command.response.put(index)
                elif command.name == "seek_index":
                    if episode is None:
                        raise RuntimeError("Load an episode before seeking.")
                    index = int(
                        np.clip(
                            int(command.value),
                            0,
                            len(episode.trajectory["timestamps"]) - 1,
                        )
                    )
                    self._publish(backend, episode, compare, index)
                    if playing:
                        play_origin_wall = clock()
                        play_origin_time = float(
                            episode.trajectory["timestamps"][index]
                        )
                    command.response.put(index)
                elif command.name == "play":
                    requested_playing = bool(command.value)
                    if requested_playing:
                        if episode is None:
                            raise RuntimeError("Load an episode before playing.")
                        play_origin_wall = clock()
                        play_origin_time = float(
                            episode.trajectory["timestamps"][index]
                        )
                    playing = requested_playing
                    command.response.put(None)
                elif command.name == "frame_client":
                    if episode is not None:
                        _set_client_camera(command.value, episode)
                    command.response.put(None)
                elif command.name == "close":
                    backend.stop()
                    command.response.put(None)
                    running = False
                else:
                    raise RuntimeError(f"Unknown replay command {command.name!r}.")
            except BaseException as exc:
                command.response.put(exc)

    def _enqueue_seek(self, step: int) -> None:
        response: queue.Queue[object] = queue.Queue(maxsize=1)
        self._commands.put(_Command("seek_index", int(step), response))

    def _install_play_control(self, backend: object) -> None:
        server = getattr(backend, "_server", None)
        if server is None:
            return
        control = server.gui.add_checkbox("Play recording", initial_value=False)

        @control.on_update
        def _(event: object) -> None:
            response: queue.Queue[object] = queue.Queue(maxsize=1)
            self._commands.put(_Command("play", bool(event.target.value), response))

    def _install_client_framing(self, backend: object) -> None:
        server = getattr(backend, "_server", None)
        if server is None or not hasattr(server, "on_client_connect"):
            return

        @server.on_client_connect
        def _(client: object) -> None:
            response: queue.Queue[object] = queue.Queue(maxsize=1)
            self._commands.put(_Command("frame_client", client, response))

    @staticmethod
    def _publish(
        backend: object, episode: _Episode, compare: _Episode | None, index: int
    ) -> None:
        trajectory = episode.trajectory
        timestamps = trajectory["timestamps"]
        node_count = len(episode.manifest.nodes)
        positions = trajectory.get(
            "scene_positions",
            np.zeros((len(timestamps), node_count, 3), dtype=np.float32),
        )[index]
        wxyz = trajectory.get(
            "scene_wxyz",
            np.tile(
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                (len(timestamps), node_count, 1),
            ),
        )[index]
        visible = trajectory.get(
            "scene_visible", np.zeros((len(timestamps), node_count), dtype=np.bool_)
        )[index]
        camera_count = len(episode.manifest.cameras)
        camera_positions = trajectory.get(
            "camera_positions",
            np.zeros((len(timestamps), camera_count, 3), dtype=np.float32),
        )[index]
        camera_wxyz = trajectory.get(
            "camera_wxyz",
            np.tile(
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                (len(timestamps), camera_count, 1),
            ),
        )[index]
        paths = [
            TrajectoryOverlay(
                "primary-tcp-path", trajectory["tcp_position"], color=(255, 170, 30)
            )
        ]
        frames = [
            FrameOverlay(
                "primary-tcp",
                trajectory["tcp_position"][index],
                np.array([1.0, 0.0, 0.0, 0.0]),
            )
        ]
        if compare is not None:
            compare_index = int(
                np.argmin(np.abs(compare.trajectory["timestamps"] - timestamps[index]))
            )
            paths.append(
                TrajectoryOverlay(
                    "compare-tcp-path",
                    compare.trajectory["tcp_position"],
                    color=(70, 160, 255),
                )
            )
            frames.append(
                FrameOverlay(
                    "compare-tcp",
                    compare.trajectory["tcp_position"][compare_index],
                    np.array([1.0, 0.0, 0.0, 0.0]),
                )
            )
        frame = SceneFrame(
            run_id=episode.manifest.run_id,
            scene_revision=episode.manifest.scene_revision,
            sequence=index,
            sim_step=index,
            sim_time=float(timestamps[index]),
            node_ids=tuple(node.node_id for node in episode.manifest.nodes),
            positions=positions,
            wxyz=wxyz,
            visible=visible,
            camera_ids=tuple(camera.camera_id for camera in episode.manifest.cameras),
            camera_positions=camera_positions,
            camera_wxyz=camera_wxyz,
            overlays=SceneOverlays(frames=tuple(frames), trajectories=tuple(paths)),
        )
        if not backend.publish_frame(frame):
            raise RuntimeError("Viser rejected the offline replay frame.")
        backend.publish_replay_control(
            step=index, max_step=len(timestamps) - 1, visible=True
        )
        if episode.camera_frames is not None and episode.camera_timestamps is not None:
            camera_index = int(
                np.argmin(np.abs(episode.camera_timestamps - timestamps[index]))
            )
            camera_id = episode.manifest.cameras[0].camera_id
            backend.publish_camera_images(
                CameraImageFrame(
                    run_id=episode.manifest.run_id,
                    scene_revision=episode.manifest.scene_revision,
                    sequence=index,
                    sim_step=index,
                    sim_time=float(timestamps[index]),
                    images=(
                        CameraImage(camera_id, episode.camera_frames[camera_index]),
                    ),
                )
            )
