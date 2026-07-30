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
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..cfg import ViserServerCfg
from ..protocol import (
    CameraImageFrame,
    CameraSpec,
    FrameOverlay,
    GizmoCommand,
    GizmoSpec,
    GizmoState,
    MeshGeometry,
    PointCloudOverlay,
    SceneFrame,
    SceneManifest,
    SceneNode,
    TargetOverlay,
    TrajectoryOverlay,
)
from ..scene_exporter import safe_path_component
from .base import VisualizationBackend

__all__ = ["ViserBackend"]


@dataclass
class _MeshBatch:
    handle: object
    node_ids: tuple[str, ...]
    frame_indices: np.ndarray
    env_ids: np.ndarray
    frame_visible: np.ndarray


@dataclass
class _DynamicMesh:
    handle: object
    node: SceneNode
    geometry: MeshGeometry
    frame_index: int
    frame_visible: bool


@dataclass
class _GizmoHandle:
    handle: object
    spec: GizmoSpec


@dataclass(frozen=True)
class _GuiEvent:
    category: str
    value: object


@dataclass(frozen=True)
class _GizmoEvent:
    gizmo_id: str | None
    phase: str
    client_id: str
    position: np.ndarray | None = None
    wxyz: np.ndarray | None = None


class ViserBackend(VisualizationBackend):
    """Map backend-neutral scene snapshots onto a Viser server.

    Static triangle data is uploaded only when a manifest is published. Nodes
    sharing a geometry are represented by one batched mesh, while frame updates
    change only batched positions, quaternions, and opacities. Deformable nodes
    use individual meshes that are recreated only when a low-frequency dynamic
    vertex update is present.

    Args:
        cfg: Viser server binding configuration.
        server_factory: Optional dependency injection hook used by unit tests.
        allow_commands: Whether transform-control drags may mutate simulation.
    """

    _INDIVIDUAL_ENV_CONTROL_LIMIT = 16

    def __init__(
        self,
        cfg: ViserServerCfg,
        server_factory: Callable[..., object] | None = None,
        *,
        allow_commands: bool = False,
    ) -> None:
        self.cfg = cfg
        self.allow_commands = allow_commands
        self._server_factory = server_factory
        self._server: object | None = None
        self._endpoint: str | None = None
        self._thread_id: int | None = None
        self._run_id: str | None = None
        self._scene_revision = -1
        self._frame_node_ids: tuple[str, ...] = ()
        self._frame_camera_ids: tuple[str, ...] = ()
        self._camera_frame_indices: dict[str, int] = {}
        self._mesh_batches: dict[str, _MeshBatch] = {}
        self._dynamic_meshes: dict[str, _DynamicMesh] = {}
        self._gizmo_handles: dict[str, _GizmoHandle] = {}
        self._gizmo_states: dict[str, GizmoState] = {}
        self._gizmo_owners: dict[str, str] = {}
        self._gizmo_drag_poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._gizmo_sequence = 0
        self._world_handle: object | None = None
        self._ground_grid_handle: object | None = None
        self._camera_handles: dict[str, object] = {}
        self._camera_specs: dict[str, CameraSpec] = {}
        self._latest_camera_images: dict[str, np.ndarray] = {}
        self._selected_camera_env: int | None = None
        self._selected_camera_uid: str | None = None
        self._show_camera_frustum = True
        self._show_camera_rgb = True
        self._camera_env_dropdown: object | None = None
        self._camera_uid_dropdown: object | None = None
        self._camera_image_handle: object | None = None
        self._overlay_handles: dict[tuple[str, str], object] = {}
        self._overlay_base_visibility: dict[tuple[str, str], bool] = {}
        self._env_visibility: dict[int, bool] = {}
        self._show_all_envs = True
        self._selected_scene_env: int | None = None
        self._overlay_visibility: dict[str, bool] = {
            "frames": True,
            "trajectories": True,
            "targets": True,
            "point_clouds": True,
        }
        self._gui_events: queue.SimpleQueue[_GuiEvent] = queue.SimpleQueue()
        self._gizmo_events: queue.SimpleQueue[_GizmoEvent] = queue.SimpleQueue()

    @property
    def endpoint(self) -> str | None:
        """Return the browser endpoint after the server starts."""
        return self._endpoint

    @property
    def client_count(self) -> int:
        """Return the number of connected Viser clients."""
        if self._server is None:
            return 0
        return len(self._server.get_clients())

    def _assert_update_thread(self) -> None:
        current = threading.get_ident()
        if self._thread_id is None:
            self._thread_id = current
        elif self._thread_id != current:
            raise RuntimeError(
                "Viser handles may only be modified by the update thread."
            )

    def start(self) -> None:
        """Start the Viser server on the visualization worker thread."""
        self._assert_update_thread()
        if self._server is not None:
            return
        if self._server_factory is None:
            import viser

            self._server_factory = viser.ViserServer
        self._server = self._server_factory(
            host=self.cfg.host,
            port=self.cfg.port,
            label=self.cfg.label,
            verbose=self.cfg.verbose,
        )
        port = self._server.get_port()
        display_host = (
            "localhost" if self.cfg.host in {"0.0.0.0", "::"} else self.cfg.host
        )
        self._endpoint = f"http://{display_host}:{port}"
        if self.allow_commands and hasattr(self._server, "on_client_disconnect"):

            @self._server.on_client_disconnect
            def _(client: object) -> None:
                client_id = getattr(client, "client_id", None)
                if client_id is not None:
                    self._gizmo_events.put(
                        _GizmoEvent(
                            gizmo_id=None,
                            phase="disconnect",
                            client_id=str(client_id),
                        )
                    )

    def _register_visibility_controls(self, manifest: SceneManifest) -> None:
        previous_env_visibility = self._env_visibility
        while True:
            try:
                self._gui_events.get_nowait()
            except queue.Empty:
                break
        self._server.gui.reset()
        self._camera_image_handle = None
        self._camera_env_dropdown = None
        self._camera_uid_dropdown = None
        self._server.gui.add_markdown(
            f"**Run:** `{manifest.run_id}`  \n**Scene revision:** {manifest.scene_revision}"
        )
        if self.allow_commands:
            self._server.gui.add_markdown(
                "⚠️ **Interactive Gizmos enabled.** Dragging a transform control "
                "mutates the simulation."
            )
        env_ids = sorted(
            {node.env_id for node in manifest.nodes}
            | {camera.env_id for camera in manifest.cameras}
            | {gizmo.env_id for gizmo in manifest.gizmos}
        )
        if self._selected_scene_env not in env_ids:
            self._selected_scene_env = env_ids[0] if env_ids else None
        if len(env_ids) > self._INDIVIDUAL_ENV_CONTROL_LIMIT:
            self._env_visibility = {
                env_id: self._show_all_envs or env_id == self._selected_scene_env
                for env_id in env_ids
            }
        else:
            self._env_visibility = {
                env_id: previous_env_visibility.get(env_id, True) for env_id in env_ids
            }
        with self._server.gui.add_folder("Environments"):
            if len(env_ids) > self._INDIVIDUAL_ENV_CONTROL_LIMIT:
                show_all = self._server.gui.add_checkbox(
                    "Show all environments",
                    initial_value=self._show_all_envs,
                )

                @show_all.on_update
                def _(event: object) -> None:
                    self._gui_events.put(
                        _GuiEvent(
                            category="environment_all",
                            value=bool(event.target.value),
                        )
                    )

                selected = self._server.gui.add_dropdown(
                    "Selected environment",
                    options=[str(env_id) for env_id in env_ids],
                    initial_value=str(self._selected_scene_env),
                )

                @selected.on_update
                def _(event: object) -> None:
                    self._gui_events.put(
                        _GuiEvent(
                            category="environment_selected",
                            value=int(event.target.value),
                        )
                    )

            else:
                for env_id in env_ids:
                    checkbox = self._server.gui.add_checkbox(
                        f"Environment {env_id}",
                        initial_value=self._env_visibility[env_id],
                    )

                    @checkbox.on_update
                    def _(event: object, selected_env_id: int = env_id) -> None:
                        self._gui_events.put(
                            _GuiEvent(
                                category="environment",
                                value=(selected_env_id, bool(event.target.value)),
                            )
                        )

        self._register_camera_controls(manifest)

        with self._server.gui.add_folder("Overlays"):
            for category, label in (
                ("frames", "Frames"),
                ("trajectories", "Trajectories"),
                ("targets", "Targets"),
                ("point_clouds", "Point clouds"),
            ):
                checkbox = self._server.gui.add_checkbox(
                    label, initial_value=self._overlay_visibility[category]
                )

                @checkbox.on_update
                def _(event: object, selected_category: str = category) -> None:
                    self._gui_events.put(
                        _GuiEvent(
                            category="overlay",
                            value=(selected_category, bool(event.target.value)),
                        )
                    )

    @staticmethod
    def _event_client_id(event: object) -> str | None:
        client_id = getattr(event, "client_id", None)
        if client_id is None:
            client = getattr(event, "client", None)
            client_id = getattr(client, "client_id", None)
        if client_id is None:
            return None
        return str(client_id)

    def _queue_gizmo_event(
        self,
        event: object,
        *,
        gizmo_id: str,
        phase: str,
    ) -> None:
        client_id = self._event_client_id(event)
        if client_id is None:
            return
        target = event.target
        self._gizmo_events.put(
            _GizmoEvent(
                gizmo_id=gizmo_id,
                phase=phase,
                client_id=client_id,
                position=np.asarray(target.position, dtype=np.float32).copy(),
                wxyz=np.asarray(target.wxyz, dtype=np.float32).copy(),
            )
        )

    def _create_gizmo_handle(self, spec: GizmoSpec) -> object:
        if self.allow_commands:
            handle = self._server.scene.add_transform_controls(
                spec.path,
                scale=spec.scale,
                line_width=spec.line_width,
                depth_test=True,
                opacity=0.9,
                visible=spec.visible,
            )

            @handle.on_drag_start
            def _(event: object, gizmo_id: str = spec.gizmo_id) -> None:
                self._queue_gizmo_event(
                    event,
                    gizmo_id=gizmo_id,
                    phase="start",
                )

            @handle.on_update
            def _(event: object, gizmo_id: str = spec.gizmo_id) -> None:
                self._queue_gizmo_event(
                    event,
                    gizmo_id=gizmo_id,
                    phase="update",
                )

            @handle.on_drag_end
            def _(event: object, gizmo_id: str = spec.gizmo_id) -> None:
                self._queue_gizmo_event(
                    event,
                    gizmo_id=gizmo_id,
                    phase="end",
                )

            return handle
        return self._server.scene.add_frame(
            spec.path,
            axes_length=spec.scale,
            axes_radius=max(0.001, spec.scale * 0.04),
            visible=spec.visible,
        )

    def _publish_gizmo_command(self, event: _GizmoEvent, phase: str) -> None:
        sink = getattr(self, "_gizmo_command_sink", None)
        if (
            sink is None
            or self._run_id is None
            or event.gizmo_id is None
            or event.position is None
            or event.wxyz is None
        ):
            return
        self._gizmo_sequence += 1
        sink(
            GizmoCommand(
                run_id=self._run_id,
                scene_revision=self._scene_revision,
                sequence=self._gizmo_sequence,
                gizmo_id=event.gizmo_id,
                phase=phase,
                client_id=event.client_id,
                position=event.position,
                wxyz=event.wxyz,
            )
        )

    def _restore_gizmo_pose(self, gizmo_id: str) -> None:
        state = self._gizmo_states.get(gizmo_id)
        gizmo_handle = self._gizmo_handles.get(gizmo_id)
        drag_pose = self._gizmo_drag_poses.get(gizmo_id)
        if gizmo_handle is None or (state is None and drag_pose is None):
            return
        if drag_pose is not None:
            position, wxyz = drag_pose
        else:
            position, wxyz = state.position, state.wxyz
        gizmo_handle.handle.position = position
        gizmo_handle.handle.wxyz = wxyz

    def _remember_gizmo_drag(self, event: _GizmoEvent) -> None:
        if (
            event.gizmo_id is not None
            and event.position is not None
            and event.wxyz is not None
        ):
            self._gizmo_drag_poses[event.gizmo_id] = (
                event.position,
                event.wxyz,
            )

    def _release_client_gizmos(self, client_id: str) -> None:
        for gizmo_id, owner in tuple(self._gizmo_owners.items()):
            if owner != client_id:
                continue
            drag_pose = self._gizmo_drag_poses.get(gizmo_id)
            state = self._gizmo_states.get(gizmo_id)
            if drag_pose is not None or state is not None:
                position, wxyz = (
                    drag_pose if drag_pose is not None else (state.position, state.wxyz)
                )
                self._publish_gizmo_command(
                    _GizmoEvent(
                        gizmo_id=gizmo_id,
                        phase="end",
                        client_id=client_id,
                        position=position,
                        wxyz=wxyz,
                    ),
                    "end",
                )
            self._gizmo_owners.pop(gizmo_id, None)
            self._gizmo_drag_poses.pop(gizmo_id, None)

    def _apply_gizmo_events(self) -> None:
        while True:
            try:
                event = self._gizmo_events.get_nowait()
            except queue.Empty:
                return
            if event.phase == "disconnect":
                self._release_client_gizmos(event.client_id)
                continue
            gizmo_id = event.gizmo_id
            if gizmo_id is None or gizmo_id not in self._gizmo_handles:
                continue
            owner = self._gizmo_owners.get(gizmo_id)
            if event.phase == "start":
                if owner not in {None, event.client_id}:
                    self._restore_gizmo_pose(gizmo_id)
                    continue
                self._gizmo_owners[gizmo_id] = event.client_id
                self._remember_gizmo_drag(event)
                self._publish_gizmo_command(event, "start")
            elif event.phase == "update":
                if owner is None:
                    self._gizmo_owners[gizmo_id] = event.client_id
                    self._publish_gizmo_command(event, "start")
                elif owner != event.client_id:
                    self._restore_gizmo_pose(gizmo_id)
                    continue
                self._remember_gizmo_drag(event)
                self._publish_gizmo_command(event, "update")
            elif event.phase == "end":
                if owner != event.client_id:
                    self._restore_gizmo_pose(gizmo_id)
                    continue
                self._publish_gizmo_command(event, "end")
                self._gizmo_owners.pop(gizmo_id, None)
                self._gizmo_drag_poses.pop(gizmo_id, None)

    def _camera_uids_for_env(self, env_id: int) -> list[str]:
        return sorted(
            {
                spec.sensor_uid
                for spec in self._camera_specs.values()
                if spec.env_id == env_id
            }
        )

    def _reconcile_camera_selection(self) -> None:
        env_ids = sorted({spec.env_id for spec in self._camera_specs.values()})
        if not env_ids:
            self._selected_camera_env = None
            self._selected_camera_uid = None
            return
        if self._selected_camera_env not in env_ids:
            self._selected_camera_env = env_ids[0]
        camera_uids = self._camera_uids_for_env(self._selected_camera_env)
        if self._selected_camera_uid not in camera_uids:
            self._selected_camera_uid = camera_uids[0]

    def _selected_camera_id(self) -> str | None:
        for camera_id, spec in self._camera_specs.items():
            if (
                spec.env_id == self._selected_camera_env
                and spec.sensor_uid == self._selected_camera_uid
            ):
                return camera_id
        return None

    def _selected_camera_image(self) -> np.ndarray:
        camera_id = self._selected_camera_id()
        if camera_id is None or camera_id not in self._latest_camera_images:
            return np.zeros((120, 160, 3), dtype=np.uint8)
        return self._latest_camera_images[camera_id]

    def _replace_camera_image_handle(self) -> None:
        if self._camera_image_handle is not None:
            self._camera_image_handle.remove()
            self._camera_image_handle = None
        if not self._show_camera_rgb or self._selected_camera_uid is None:
            return
        label = (
            f"Environment {self._selected_camera_env} / "
            f"{self._selected_camera_uid} RGB"
        )
        self._camera_image_handle = self._server.gui.add_image(
            self._selected_camera_image(),
            label=label,
            format="jpeg",
            jpeg_quality=80,
        )

    def _register_camera_controls(self, manifest: SceneManifest) -> None:
        self._reconcile_camera_selection()
        if not manifest.cameras:
            return
        env_options = [
            str(env_id) for env_id in sorted({c.env_id for c in manifest.cameras})
        ]
        camera_options = self._camera_uids_for_env(self._selected_camera_env)
        with self._server.gui.add_folder("Cameras"):
            self._camera_env_dropdown = self._server.gui.add_dropdown(
                "Environment",
                options=env_options,
                initial_value=str(self._selected_camera_env),
            )

            @self._camera_env_dropdown.on_update
            def _(event: object) -> None:
                self._gui_events.put(
                    _GuiEvent(category="camera_environment", value=event.target.value)
                )

            self._camera_uid_dropdown = self._server.gui.add_dropdown(
                "Camera",
                options=camera_options,
                initial_value=self._selected_camera_uid,
            )

            @self._camera_uid_dropdown.on_update
            def _(event: object) -> None:
                self._gui_events.put(
                    _GuiEvent(category="camera_uid", value=event.target.value)
                )

            frustum_checkbox = self._server.gui.add_checkbox(
                "Camera frustum",
                initial_value=self._show_camera_frustum,
            )

            @frustum_checkbox.on_update
            def _(event: object) -> None:
                self._gui_events.put(
                    _GuiEvent(
                        category="camera_frustum",
                        value=bool(event.target.value),
                    )
                )

            rgb_checkbox = self._server.gui.add_checkbox(
                "RGB preview",
                initial_value=self._show_camera_rgb,
            )

            @rgb_checkbox.on_update
            def _(event: object) -> None:
                self._gui_events.put(
                    _GuiEvent(
                        category="camera_rgb",
                        value=bool(event.target.value),
                    )
                )

        self._replace_camera_image_handle()

    def publish_manifest(self, manifest: SceneManifest) -> None:
        """Replace the browser's static scene topology.

        Args:
            manifest: Static scene nodes, geometries, cameras, and metadata.
        """
        self._assert_update_thread()
        if self._server is None:
            raise RuntimeError("ViserBackend.start() must be called before publishing.")
        while True:
            try:
                self._gizmo_events.get_nowait()
            except queue.Empty:
                break
        self._gizmo_owners.clear()
        self._gizmo_drag_poses.clear()
        self._server.scene.set_up_direction(manifest.up_direction)
        if self._world_handle is None:
            self._world_handle = self._server.scene.add_frame(
                "/world",
                axes_length=0.35,
                axes_radius=0.012,
            )
        if self._ground_grid_handle is None:
            self._ground_grid_handle = self._server.scene.add_grid(
                "/world/default_ground",
                width=1000.0,
                height=1000.0,
                plane="xy",
                cell_color=(120, 120, 120),
                cell_thickness=1.0,
                cell_size=1.0,
                section_color=(170, 170, 170),
                section_thickness=1.5,
                section_size=10.0,
                fade_distance=100.0,
                shadow_opacity=0.15,
            )
        for handle in self._overlay_handles.values():
            handle.remove()
        self._overlay_handles.clear()
        self._overlay_base_visibility.clear()
        geometry_by_id = {
            geometry.geometry_id: geometry for geometry in manifest.geometries
        }
        self._frame_node_ids = tuple(node.node_id for node in manifest.nodes)
        node_indices = {
            node_id: index for index, node_id in enumerate(self._frame_node_ids)
        }
        self._frame_camera_ids = tuple(camera.camera_id for camera in manifest.cameras)
        self._camera_frame_indices = {
            camera_id: index for index, camera_id in enumerate(self._frame_camera_ids)
        }
        nodes_by_geometry: dict[str, list[SceneNode]] = defaultdict(list)
        dynamic_nodes: list[SceneNode] = []
        for node in manifest.nodes:
            if node.dynamic_geometry:
                dynamic_nodes.append(node)
            else:
                nodes_by_geometry[node.geometry_id].append(node)

        removed_geometry_ids = set(self._mesh_batches) - set(nodes_by_geometry)
        for geometry_id in removed_geometry_ids:
            self._mesh_batches.pop(geometry_id).handle.remove()

        for geometry_id, nodes in nodes_by_geometry.items():
            geometry = geometry_by_id[geometry_id]
            count = len(nodes)
            batch = self._mesh_batches.get(geometry_id)
            if batch is None:
                handle = self._server.scene.add_batched_meshes_simple(
                    f"/scene/mesh_batches/{safe_path_component(geometry_id)}",
                    vertices=geometry.vertices,
                    faces=geometry.faces,
                    batched_wxyzs=np.tile(
                        np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                        (count, 1),
                    ),
                    batched_positions=np.zeros((count, 3), dtype=np.float32),
                    batched_colors=geometry.color,
                    batched_opacities=np.asarray(
                        [1.0 if node.visible else 0.0 for node in nodes],
                        dtype=np.float32,
                    ),
                    side="double",
                )
                batch = _MeshBatch(
                    handle=handle,
                    node_ids=(),
                    frame_indices=np.empty((0,), dtype=np.int64),
                    env_ids=np.empty((0,), dtype=np.int64),
                    frame_visible=np.empty((0,), dtype=np.bool_),
                )
                self._mesh_batches[geometry_id] = batch
            batch.node_ids = tuple(node.node_id for node in nodes)
            batch.frame_indices = np.asarray(
                [node_indices[node_id] for node_id in batch.node_ids],
                dtype=np.int64,
            )
            batch.env_ids = np.asarray([node.env_id for node in nodes], dtype=np.int64)
            batch.frame_visible = np.asarray(
                [node.visible for node in nodes], dtype=np.bool_
            )
            batch.handle.batched_wxyzs = np.tile(
                np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                (count, 1),
            )
            batch.handle.batched_positions = np.zeros((count, 3), dtype=np.float32)

        for dynamic_mesh in self._dynamic_meshes.values():
            dynamic_mesh.handle.remove()
        self._dynamic_meshes.clear()
        for node in dynamic_nodes:
            geometry = geometry_by_id[node.geometry_id]
            handle = self._server.scene.add_mesh_simple(
                node.path,
                vertices=geometry.vertices,
                faces=geometry.faces,
                color=geometry.color,
                side="double",
                visible=node.visible and self._env_visibility.get(node.env_id, True),
            )
            self._dynamic_meshes[node.node_id] = _DynamicMesh(
                handle=handle,
                node=node,
                geometry=geometry,
                frame_index=node_indices[node.node_id],
                frame_visible=node.visible,
            )

        camera_specs = {camera.camera_id: camera for camera in manifest.cameras}
        removed_camera_ids = set(self._camera_handles) - set(camera_specs)
        for camera_id in removed_camera_ids:
            self._camera_handles.pop(camera_id).remove()
            self._latest_camera_images.pop(camera_id, None)
        for camera_id, spec in camera_specs.items():
            if self._camera_specs.get(camera_id) == spec:
                continue
            old_handle = self._camera_handles.pop(camera_id, None)
            if old_handle is not None:
                old_handle.remove()
            self._camera_handles[camera_id] = self._server.scene.add_camera_frustum(
                spec.path,
                fov=spec.fov_y,
                aspect=spec.aspect,
                scale=min(0.3, spec.far * 0.1),
                line_width=2.0,
                color=(70, 180, 255),
                visible=False,
            )
        self._camera_specs = camera_specs
        self._reconcile_camera_selection()

        gizmo_specs = {gizmo.gizmo_id: gizmo for gizmo in manifest.gizmos}
        removed_gizmo_ids = set(self._gizmo_handles) - set(gizmo_specs)
        for gizmo_id in removed_gizmo_ids:
            self._gizmo_handles.pop(gizmo_id).handle.remove()
            self._gizmo_states.pop(gizmo_id, None)
            self._gizmo_owners.pop(gizmo_id, None)
            self._gizmo_drag_poses.pop(gizmo_id, None)
        for gizmo_id, spec in gizmo_specs.items():
            current = self._gizmo_handles.get(gizmo_id)
            if current is not None and current.spec == spec:
                continue
            if current is not None:
                current.handle.remove()
                self._gizmo_owners.pop(gizmo_id, None)
                self._gizmo_drag_poses.pop(gizmo_id, None)
            self._gizmo_handles[gizmo_id] = _GizmoHandle(
                handle=self._create_gizmo_handle(spec),
                spec=spec,
            )

        self._run_id = manifest.run_id
        self._scene_revision = manifest.scene_revision
        self._register_visibility_controls(manifest)
        for batch in self._mesh_batches.values():
            self._apply_mesh_visibility(batch)
        for dynamic_mesh in self._dynamic_meshes.values():
            self._apply_dynamic_mesh_visibility(dynamic_mesh)
        self._apply_camera_visibility()
        self._apply_gizmo_visibility()
        self._server.flush()

    def _apply_gui_events(self) -> None:
        self._apply_gizmo_events()
        changed = False
        camera_selection_changed = False
        camera_rgb_changed = False
        while True:
            try:
                event = self._gui_events.get_nowait()
            except queue.Empty:
                break
            if event.category == "environment":
                env_id, visible = event.value
                self._env_visibility[int(env_id)] = bool(visible)
            elif event.category == "environment_all":
                self._show_all_envs = bool(event.value)
                self._env_visibility = {
                    env_id: self._show_all_envs or env_id == self._selected_scene_env
                    for env_id in self._env_visibility
                }
            elif event.category == "environment_selected":
                self._selected_scene_env = int(event.value)
                if not self._show_all_envs:
                    self._env_visibility = {
                        env_id: env_id == self._selected_scene_env
                        for env_id in self._env_visibility
                    }
            elif event.category == "overlay":
                category, visible = event.value
                self._overlay_visibility[str(category)] = bool(visible)
            elif event.category == "camera_environment":
                self._selected_camera_env = int(event.value)
                camera_uids = self._camera_uids_for_env(self._selected_camera_env)
                if self._selected_camera_uid not in camera_uids:
                    self._selected_camera_uid = camera_uids[0]
                if self._camera_uid_dropdown is not None:
                    self._camera_uid_dropdown.options = camera_uids
                    self._camera_uid_dropdown.value = self._selected_camera_uid
                camera_selection_changed = True
            elif event.category == "camera_uid":
                self._selected_camera_uid = str(event.value)
                camera_selection_changed = True
            elif event.category == "camera_frustum":
                self._show_camera_frustum = bool(event.value)
            elif event.category == "camera_rgb":
                self._show_camera_rgb = bool(event.value)
                camera_rgb_changed = True
            changed = True
        if not changed:
            return
        for batch in self._mesh_batches.values():
            self._apply_mesh_visibility(batch)
        for dynamic_mesh in self._dynamic_meshes.values():
            self._apply_dynamic_mesh_visibility(dynamic_mesh)
        for key, handle in self._overlay_handles.items():
            handle.visible = self._overlay_base_visibility.get(
                key, True
            ) and self._overlay_visibility.get(key[0], True)
        self._apply_camera_visibility()
        self._apply_gizmo_visibility()
        if camera_selection_changed or camera_rgb_changed:
            self._replace_camera_image_handle()

    def _apply_camera_visibility(self) -> None:
        selected_camera_id = self._selected_camera_id()
        for camera_id, handle in self._camera_handles.items():
            spec = self._camera_specs[camera_id]
            handle.visible = (
                self._show_camera_frustum
                and camera_id == selected_camera_id
                and self._env_visibility.get(spec.env_id, False)
            )

    def _apply_gizmo_visibility(self) -> None:
        for gizmo_id, gizmo_handle in self._gizmo_handles.items():
            state = self._gizmo_states.get(gizmo_id)
            frame_visible = (
                state.visible if state is not None else gizmo_handle.spec.visible
            )
            gizmo_handle.handle.visible = frame_visible and self._env_visibility.get(
                gizmo_handle.spec.env_id,
                False,
            )

    def _apply_mesh_visibility(self, batch: _MeshBatch) -> None:
        env_visible = np.asarray(
            [self._env_visibility.get(int(env_id), False) for env_id in batch.env_ids],
            dtype=np.bool_,
        )
        batch.handle.batched_opacities = (batch.frame_visible & env_visible).astype(
            np.float32
        )

    def _apply_dynamic_mesh_visibility(self, dynamic_mesh: _DynamicMesh) -> None:
        dynamic_mesh.handle.visible = (
            dynamic_mesh.frame_visible
            and self._env_visibility.get(
                dynamic_mesh.node.env_id,
                False,
            )
        )

    def _replace_dynamic_mesh(
        self,
        dynamic_mesh: _DynamicMesh,
        *,
        vertices: np.ndarray,
        position: np.ndarray,
        wxyz: np.ndarray,
    ) -> None:
        dynamic_mesh.handle.remove()
        dynamic_mesh.handle = self._server.scene.add_mesh_simple(
            dynamic_mesh.node.path,
            vertices=vertices,
            faces=dynamic_mesh.geometry.faces,
            color=dynamic_mesh.geometry.color,
            side="double",
            position=position,
            wxyz=wxyz,
            visible=dynamic_mesh.frame_visible
            and self._env_visibility.get(dynamic_mesh.node.env_id, False),
        )

    @staticmethod
    def _segments(points: np.ndarray) -> np.ndarray:
        if len(points) < 2:
            return np.empty((0, 2, 3), dtype=np.float32)
        return np.stack((points[:-1], points[1:]), axis=1)

    def _update_frame_overlay(
        self,
        category: str,
        overlay: FrameOverlay | TargetOverlay,
        active: set[tuple[str, str]],
    ) -> None:
        key = (category, overlay.overlay_id)
        active.add(key)
        self._overlay_base_visibility[key] = overlay.visible
        handle = self._overlay_handles.get(key)
        if handle is None:
            path_component = safe_path_component(overlay.overlay_id)
            handle = self._server.scene.add_frame(
                f"/overlays/{category}/{path_component}",
                axes_length=overlay.axes_length,
                axes_radius=getattr(overlay, "axes_radius", overlay.axes_length * 0.04),
            )
            self._overlay_handles[key] = handle
        handle.position = overlay.position
        handle.wxyz = overlay.wxyz
        handle.visible = overlay.visible and self._overlay_visibility[category]

    def _update_trajectory(
        self, overlay: TrajectoryOverlay, active: set[tuple[str, str]]
    ) -> None:
        category = "trajectories"
        key = (category, overlay.overlay_id)
        active.add(key)
        self._overlay_base_visibility[key] = overlay.visible
        segments = self._segments(overlay.points)
        handle = self._overlay_handles.get(key)
        if handle is None:
            path_component = safe_path_component(overlay.overlay_id)
            handle = self._server.scene.add_line_segments(
                f"/overlays/{category}/{path_component}",
                points=segments,
                colors=overlay.color,
                line_width=overlay.line_width,
            )
            self._overlay_handles[key] = handle
        else:
            handle.points = segments
            handle.colors = np.asarray(overlay.color, dtype=np.uint8)
            handle.line_width = overlay.line_width
        handle.visible = overlay.visible and self._overlay_visibility[category]

    def _update_point_cloud(
        self, overlay: PointCloudOverlay, active: set[tuple[str, str]]
    ) -> None:
        category = "point_clouds"
        key = (category, overlay.overlay_id)
        active.add(key)
        self._overlay_base_visibility[key] = overlay.visible
        handle = self._overlay_handles.get(key)
        if handle is None:
            path_component = safe_path_component(overlay.overlay_id)
            handle = self._server.scene.add_point_cloud(
                f"/overlays/{category}/{path_component}",
                points=overlay.points,
                colors=overlay.colors,
                point_size=overlay.point_size,
            )
            self._overlay_handles[key] = handle
        else:
            handle.points = overlay.points
            handle.colors = overlay.colors
            handle.point_size = overlay.point_size
        handle.visible = overlay.visible and self._overlay_visibility[category]

    def publish_frame(self, frame: SceneFrame) -> bool:
        """Publish a dynamic scene frame.

        Args:
            frame: Pose, visibility, deformable, camera, and overlay updates.

        Returns:
            Whether the frame matched the active run and scene revision.
        """
        self._assert_update_thread()
        if self._server is None:
            raise RuntimeError("ViserBackend.start() must be called before publishing.")
        if frame.run_id != self._run_id or frame.scene_revision != self._scene_revision:
            return False
        if (
            frame.node_ids != self._frame_node_ids
            or frame.camera_ids != self._frame_camera_ids
        ):
            return False
        self._apply_gui_events()
        dynamic_updates = {mesh.node_id: mesh for mesh in frame.dynamic_meshes}
        if set(dynamic_updates) - set(self._dynamic_meshes):
            return False
        if any(
            update.vertices.shape != dynamic_mesh.geometry.vertices.shape
            for node_id, update in dynamic_updates.items()
            for dynamic_mesh in (self._dynamic_meshes[node_id],)
        ):
            return False
        for batch in self._mesh_batches.values():
            indices = batch.frame_indices
            batch.handle.batched_positions = frame.positions[indices]
            batch.handle.batched_wxyzs = frame.wxyz[indices]
            frame_visible = frame.visible[indices]
            if not np.array_equal(batch.frame_visible, frame_visible):
                batch.frame_visible = frame_visible
                self._apply_mesh_visibility(batch)
        for node_id, dynamic_mesh in self._dynamic_meshes.items():
            index = dynamic_mesh.frame_index
            dynamic_mesh.frame_visible = bool(frame.visible[index])
            update = dynamic_updates.get(node_id)
            if update is None:
                dynamic_mesh.handle.position = frame.positions[index]
                dynamic_mesh.handle.wxyz = frame.wxyz[index]
                self._apply_dynamic_mesh_visibility(dynamic_mesh)
                continue
            self._replace_dynamic_mesh(
                dynamic_mesh,
                vertices=update.vertices,
                position=frame.positions[index],
                wxyz=frame.wxyz[index],
            )

        for camera_id, handle in self._camera_handles.items():
            index = self._camera_frame_indices[camera_id]
            handle.position = frame.camera_positions[index]
            handle.wxyz = frame.camera_wxyz[index]
        self._apply_camera_visibility()

        gizmo_states = {gizmo.gizmo_id: gizmo for gizmo in frame.gizmos}
        if set(gizmo_states) != set(self._gizmo_handles):
            return False
        self._gizmo_states = gizmo_states
        for gizmo_id, gizmo_handle in self._gizmo_handles.items():
            state = gizmo_states[gizmo_id]
            if gizmo_id not in self._gizmo_owners:
                gizmo_handle.handle.position = state.position
                gizmo_handle.handle.wxyz = state.wxyz
        self._apply_gizmo_visibility()

        active_overlays: set[tuple[str, str]] = set()
        for overlay in frame.overlays.frames:
            self._update_frame_overlay("frames", overlay, active_overlays)
        for overlay in frame.overlays.targets:
            self._update_frame_overlay("targets", overlay, active_overlays)
        for overlay in frame.overlays.trajectories:
            self._update_trajectory(overlay, active_overlays)
        for overlay in frame.overlays.point_clouds:
            self._update_point_cloud(overlay, active_overlays)

        stale = set(self._overlay_handles) - active_overlays
        for key in stale:
            self._overlay_handles.pop(key).remove()
            self._overlay_base_visibility.pop(key, None)
        return True

    def publish_camera_images(self, frame: CameraImageFrame) -> bool:
        """Publish low-frequency camera RGB previews.

        Args:
            frame: Camera image updates for the active scene revision.

        Returns:
            Whether the frame matched the active run and scene revision.
        """
        self._assert_update_thread()
        if self._server is None:
            raise RuntimeError("ViserBackend.start() must be called before publishing.")
        if frame.run_id != self._run_id or frame.scene_revision != self._scene_revision:
            return False
        self._apply_gui_events()
        for image in frame.images:
            if image.camera_id in self._camera_specs:
                self._latest_camera_images[image.camera_id] = image.image
        selected_camera_id = self._selected_camera_id()
        if (
            self._show_camera_rgb
            and self._camera_image_handle is not None
            and selected_camera_id in self._latest_camera_images
        ):
            self._camera_image_handle.image = self._latest_camera_images[
                selected_camera_id
            ]
        return True

    def poll(self) -> None:
        """Apply queued browser GUI events."""
        self._assert_update_thread()
        self._apply_gui_events()

    def stop(self) -> None:
        """Stop the Viser server and release all browser handles."""
        self._assert_update_thread()
        if self._server is None:
            return
        self._server.flush()
        self._server.stop()
        self._server = None
        self._endpoint = None
        self._run_id = None
        self._scene_revision = -1
        self._frame_node_ids = ()
        self._frame_camera_ids = ()
        self._camera_frame_indices.clear()
        self._mesh_batches.clear()
        self._dynamic_meshes.clear()
        self._gizmo_handles.clear()
        self._gizmo_states.clear()
        self._gizmo_owners.clear()
        self._gizmo_drag_poses.clear()
        self._gizmo_sequence = 0
        self._world_handle = None
        self._ground_grid_handle = None
        self._camera_handles.clear()
        self._camera_specs.clear()
        self._latest_camera_images.clear()
        self._camera_image_handle = None
        self._camera_env_dropdown = None
        self._camera_uid_dropdown = None
        self._overlay_handles.clear()
        self._overlay_base_visibility.clear()
        while True:
            try:
                self._gizmo_events.get_nowait()
            except queue.Empty:
                break
        self._thread_id = None
