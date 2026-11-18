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

"""Scene-preview state, portable inspection and camera output helpers."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
from itertools import product
from pathlib import Path
import queue
import threading
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from embodichain.lab.sim import SimulationManager

__all__: list[str] = []


class ReloadPreview(Exception):
    """Request a fresh owned scene rather than a partial solver-state restore."""


class PreviewControls:
    """Apply frontend commands only on the owning simulation thread."""

    def __init__(self, running: bool) -> None:
        self.running = running
        self.stopped = False
        self.pending_steps = 0
        self.steps = 0
        self.commands: queue.Queue[str] = queue.Queue(maxsize=64)
        self._stop_requested = threading.Event()

    def enqueue(self, command: str) -> None:
        if command not in {"play", "pause", "step", "reset", "stop", "toggle"}:
            raise ValueError(f"Unknown preview command: {command}")
        if command == "stop":
            self._stop_requested.set()
            return
        try:
            self.commands.put_nowait(command)
        except queue.Full:
            pass

    def update(self, runtime: object | None) -> None:
        commands = list(runtime.drain_preview_commands()) if runtime is not None else []
        if self._stop_requested.is_set() or "stop" in commands:
            self.stopped = True
            return
        while True:
            try:
                commands.append(self.commands.get_nowait())
            except queue.Empty:
                break
        for command in commands:
            if command == "reset":
                raise ReloadPreview()
            if command == "stop":
                self.stopped = True
            elif command in {"play", "pause", "toggle"}:
                self.running = (
                    not self.running if command == "toggle" else command == "play"
                )
                self.pending_steps = 0
            elif command == "step":
                self.running = False
                self.pending_steps += 1
        if runtime is not None:
            runtime.publish_preview_control(running=self.running, steps=self.steps)

    def advance(self) -> bool:
        if self.running:
            return True
        if self.pending_steps:
            self.pending_steps -= 1
            return True
        return False

    def after_step(self) -> None:
        self.steps += 1


def source_digest(path: Path) -> str:
    """Fingerprint the entry file; USD dependency qualification remains with import."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_import(imported: Mapping[str, object]) -> list[dict[str, Any]]:
    """Describe resources without interpreting producer-specific custom data."""
    uids = getattr(imported, "resource_uids", {})
    kinds = getattr(imported, "resource_types", {})
    paths = getattr(imported, "resource_source_paths", {})
    keys = tuple(dict.fromkeys((*uids, *imported)))
    rows = []
    for key in keys:
        asset = imported.get(key)
        cfg = getattr(asset, "cfg", None)
        rows.append(
            {
                "source_key": key,
                "prim_path": paths.get(key, key),
                "uid": uids.get(key, getattr(cfg, "uid", None)),
                "kind": kinds.get(
                    key, type(asset).__name__ if asset is not None else None
                ),
                "body_type": getattr(cfg, "body_type", None),
                "joint_names": list(getattr(asset, "joint_names", ())),
            }
        )
    return rows


def camera_view(sim: SimulationManager) -> tuple[list[float], list[float]]:
    """Fit imported geometry using the same resolved meshes and poses as Viser."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    from embodichain.lab.visualization.scene_exporter import SceneExporter
    from embodichain.lab.visualization import VisualizationCfg

    selection = sim.sim_config.visualization
    env_ids = selection.env_ids if selection.backend == "viser" else [0]
    exporter = SceneExporter(sim, VisualizationCfg(env_ids=env_ids))
    manifest = exporter.build_manifest()
    frame = exporter.capture(sim_step=0, sim_time=sim.simulation_time).frame
    poses = {uid: i for i, uid in enumerate(frame.node_ids)}
    bounds = {}
    for geometry in manifest.geometries:
        vertices = geometry.vertices
        if len(vertices) and np.isfinite(vertices).all():
            bounds[geometry.geometry_id] = np.array(
                list(product(*zip(vertices.min(0), vertices.max(0))))
            )
    corners = []
    volumetric = []
    for node in manifest.nodes:
        index = poses.get(node.node_id)
        local = bounds.get(node.geometry_id)
        if index is None or local is None or not frame.visible[index]:
            continue
        rotation = Rotation.from_quat(np.roll(frame.wxyz[index], -1)).as_matrix()
        transformed = local @ rotation.T + frame.positions[index]
        corners.append(transformed)
        if np.all(np.ptp(local, axis=0) > 1e-7):
            volumetric.append(transformed)
    if not corners:
        return [2.0, -3.0, 2.0], [0.0, 0.0, 0.0]
    # Flat context meshes such as an authored floor must not drown out the
    # bounded objects. Purely planar scenes still fit their complete geometry.
    points = np.concatenate(volumetric or corners)
    target = (points.min(0) + points.max(0)) / 2
    radius = max(float(np.linalg.norm(points.max(0) - points.min(0))) / 2, 0.1)
    eye = target + radius * np.array([1.6, -2.0, 1.3])
    return eye.tolist(), target.tolist()


def screenshot(
    sim: SimulationManager,
    output: Path,
    eye: list[float],
    target: list[float],
    resolution: tuple[int, int],
) -> None:
    """Render an offscreen camera without integrating physics."""
    import uuid
    from PIL import Image
    from embodichain.lab.sim.sensors import CameraCfg

    width, height = resolution
    camera = sim.add_sensor(
        CameraCfg(
            uid="preview_capture_" + uuid.uuid4().hex,
            width=width,
            height=height,
            intrinsics=(
                float(max(width, height)),
                float(max(width, height)),
                width / 2,
                height / 2,
            ),
            extrinsics=CameraCfg.ExtrinsicsCfg(
                eye=tuple(eye), target=tuple(target), up=(0.0, 0.0, 1.0)
            ),
        )
    )
    sim.sync_render_state()
    for _ in range(8):
        camera.update()
    pixels = camera.get_data()["color"][0, :, :, :3].detach().cpu().numpy()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".in_progress")
    try:
        Image.fromarray(pixels).save(temporary, format="PNG")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def install_native_controls(
    sim: SimulationManager, controls: PreviewControls
) -> object:
    """Forward window shortcuts to the same queue as browser controls."""
    from dexsim.engine import ObjectManipulator
    from dexsim.types import InputKey

    class Input(ObjectManipulator):
        def on_key_down(self, key: int) -> None:
            for name, command in (
                ("SCANCODE_SPACE", "toggle"),
                ("SCANCODE_N", "step"),
                ("SCANCODE_HOME", "reset"),
                ("SCANCODE_ESCAPE", "stop"),
            ):
                if key == getattr(InputKey, name).value:
                    controls.enqueue(command)
                    break

    handler = Input()
    sim.add_custom_window_control([handler])
    return handler
