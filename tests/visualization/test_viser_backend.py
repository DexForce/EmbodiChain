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

from embodichain.lab.visualization import (
    CameraImage,
    CameraImageFrame,
    CameraSpec,
    DynamicMeshUpdate,
    GizmoSpec,
    GizmoState,
    JointControlSpec,
    JointControlState,
    MeshGeometry,
    SceneFrame,
    SceneManifest,
    SceneNode,
    ViserServerCfg,
)
from embodichain.lab.visualization.backends.viser import ViserBackend

REPLAY_INITIAL_STEP = 2
REPLAY_TARGET_STEP = 7
REPLAY_MAX_STEP = 9


class _Handle(SimpleNamespace):
    def remove(self) -> None:
        self.removed = True
        if hasattr(self, "visible"):
            self.visible = False


class _TransformControls(_Handle):
    def on_update(self, callback: object) -> object:
        self.update_callback = callback
        return callback

    def on_drag_start(self, callback: object) -> object:
        self.drag_start_callback = callback
        return callback

    def on_drag_end(self, callback: object) -> object:
        self.drag_end_callback = callback
        return callback


class _Folder(_Handle):
    def __enter__(self) -> _Folder:
        self.gui._folder_stack.append(self.label)
        return self

    def __exit__(self, *args: object) -> None:
        assert self.gui._folder_stack.pop() == self.label


class _Checkbox:
    def __init__(self, initial_value: bool) -> None:
        self.value = initial_value

    def on_update(self, callback: object) -> object:
        self.callback = callback
        return callback


class _Dropdown:
    def __init__(self, options: list[str], initial_value: str) -> None:
        self.options = options
        self.value = initial_value

    def on_update(self, callback: object) -> object:
        self.callback = callback
        return callback


class _ValueControl(_Handle):
    def on_update(self, callback: object) -> object:
        self.callback = callback
        return callback


class _Button(_Handle):
    def on_click(self, callback: object) -> object:
        self.callback = callback
        return callback


class _Gui:
    def __init__(self) -> None:
        self.checkboxes: dict[str, _Checkbox] = {}
        self.dropdowns: dict[str, _Dropdown] = {}
        self.sliders: dict[str, _ValueControl] = {}
        self.numbers: dict[str, _ValueControl] = {}
        self.buttons: dict[str, _Button] = {}
        self.folders: dict[str, _Folder] = {}
        self.image_handles: list[_Handle] = []
        self._folder_stack: list[str] = []

    def reset(self) -> None:
        self.checkboxes.clear()
        self.dropdowns.clear()
        self.sliders.clear()
        self.numbers.clear()
        self.buttons.clear()
        self.folders.clear()
        self.image_handles.clear()
        self._folder_stack.clear()

    def add_markdown(self, content: str) -> _Handle:
        return _Handle(content=content)

    def add_folder(self, label: str, **kwargs: object) -> _Folder:
        folder = _Folder(
            gui=self,
            label=label,
            parent_folder=self._folder_stack[-1] if self._folder_stack else None,
            removed=False,
            **kwargs,
        )
        self.folders[label] = folder
        return folder

    def add_checkbox(self, label: str, initial_value: bool) -> _Checkbox:
        checkbox = _Checkbox(initial_value)
        self.checkboxes[label] = checkbox
        return checkbox

    def add_dropdown(
        self,
        label: str,
        options: list[str],
        initial_value: str,
    ) -> _Dropdown:
        dropdown = _Dropdown(options, initial_value)
        self.dropdowns[label] = dropdown
        return dropdown

    def add_slider(
        self,
        label: str,
        *,
        min: float,
        max: float,
        step: float,
        initial_value: float,
        marks: tuple[object, ...],
        disabled: bool,
        hint: str,
    ) -> _ValueControl:
        slider = _ValueControl(
            value=initial_value,
            min=min,
            max=max,
            step=step,
            marks=marks,
            disabled=disabled,
            hint=hint,
        )
        self.sliders[label] = slider
        return slider

    def add_number(
        self,
        label: str,
        *,
        initial_value: float,
        min: float | None,
        max: float | None,
        step: float,
        disabled: bool,
        hint: str,
    ) -> _ValueControl:
        number = _ValueControl(
            value=initial_value,
            min=min,
            max=max,
            step=step,
            disabled=disabled,
            hint=hint,
        )
        self.numbers[label] = number
        return number

    def add_button(self, label: str) -> _Button:
        button = _Button(label=label)
        self.buttons[label] = button
        return button

    def add_image(self, image: np.ndarray, **kwargs: object) -> _Handle:
        kwargs.setdefault("visible", True)
        kwargs.setdefault(
            "parent_folder",
            self._folder_stack[-1] if self._folder_stack else None,
        )
        handle = _Handle(image=image, removed=False, **kwargs)
        self.image_handles.append(handle)
        return handle


class _Scene:
    def __init__(self) -> None:
        self.mesh_uploads = 0
        self.dynamic_mesh_uploads = 0
        self.mesh_handles: list[_Handle] = []
        self.dynamic_mesh_handles: list[_Handle] = []
        self.camera_handles: list[_Handle] = []
        self.grid_handles: list[_Handle] = []
        self.transform_controls: list[_TransformControls] = []

    def reset(self) -> None:
        pass

    def set_up_direction(self, direction: str) -> None:
        self.up_direction = direction

    def add_frame(self, name: str, **kwargs: object) -> _Handle:
        kwargs.setdefault("visible", True)
        return _Handle(name=name, **kwargs)

    def add_batched_meshes_simple(self, name: str, **kwargs: object) -> _Handle:
        self.mesh_uploads += 1
        handle = _Handle(name=name, visible=True, **kwargs)
        self.mesh_handles.append(handle)
        return handle

    def add_mesh_simple(self, name: str, **kwargs: object) -> _Handle:
        self.dynamic_mesh_uploads += 1
        kwargs.setdefault("visible", True)
        handle = _Handle(name=name, removed=False, **kwargs)
        self.dynamic_mesh_handles.append(handle)
        return handle

    def add_camera_frustum(self, name: str, **kwargs: object) -> _Handle:
        handle = _Handle(name=name, removed=False, **kwargs)
        self.camera_handles.append(handle)
        return handle

    def add_grid(self, name: str, **kwargs: object) -> _Handle:
        handle = _Handle(name=name, removed=False, **kwargs)
        self.grid_handles.append(handle)
        return handle

    def add_transform_controls(
        self,
        name: str,
        **kwargs: object,
    ) -> _TransformControls:
        kwargs.setdefault(
            "position",
            np.zeros((3,), dtype=np.float32),
        )
        kwargs.setdefault(
            "wxyz",
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )
        handle = _TransformControls(name=name, removed=False, **kwargs)
        self.transform_controls.append(handle)
        return handle


class _Server:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.scene = _Scene()
        self.gui = _Gui()
        self.stopped = False
        self.disconnect_callback: object | None = None

    def get_port(self) -> int:
        return 8765

    def get_clients(self) -> dict[str, object]:
        return {}

    def flush(self) -> None:
        pass

    def on_client_disconnect(self, callback: object) -> object:
        self.disconnect_callback = callback
        return callback

    def stop(self) -> None:
        self.stopped = True


def test_viser_backend_adds_one_meter_default_ground_grid() -> None:
    server = _Server()
    backend = ViserBackend(ViserServerCfg(port=8765), server_factory=lambda **_: server)

    backend.start()
    backend.publish_manifest(SceneManifest("run", 1, (), ()))
    backend.publish_manifest(SceneManifest("run", 2, (), ()))

    assert len(server.scene.grid_handles) == 1
    grid = server.scene.grid_handles[0]
    assert grid.name == "/world/default_ground"
    assert grid.plane == "xy"
    assert grid.width == 1000.0
    assert grid.height == 1000.0
    assert grid.cell_size == 1.0
    assert grid.section_size == 10.0

    backend.stop()


def test_viser_backend_uploads_static_mesh_once_and_updates_only_poses() -> None:
    server = _Server()
    backend = ViserBackend(ViserServerCfg(port=8765), server_factory=lambda **_: server)
    geometry = MeshGeometry(
        geometry_id="sha256:mesh",
        vertices=np.array(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]],
            dtype=np.float32,
        ),
        faces=np.array([[0, 1, 2]], dtype=np.uint32),
    )
    nodes = tuple(
        SceneNode(
            node_id=f"env:{env_id}/rigid:cube",
            path=f"/envs/{env_id}/rigid_objects/cube",
            parent_id=f"env:{env_id}",
            env_id=env_id,
            kind="rigid_object",
            geometry_id=geometry.geometry_id,
        )
        for env_id in range(2)
    )
    cameras = tuple(
        CameraSpec(
            camera_id=f"env:{env_id}/camera:camera",
            sensor_uid="camera",
            env_id=env_id,
            path=f"/envs/{env_id}/cameras/camera",
            fov_y=0.8,
            aspect=4.0 / 3.0,
            near=0.01,
            far=10.0,
        )
        for env_id in range(2)
    )
    manifest = SceneManifest("run", 1, nodes, (geometry,), cameras)
    frame = SceneFrame(
        run_id="run",
        scene_revision=1,
        sequence=1,
        sim_step=1,
        sim_time=0.01,
        node_ids=tuple(node.node_id for node in nodes),
        positions=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32),
        wxyz=np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        visible=np.ones((2,), dtype=np.bool_),
        camera_ids=tuple(camera.camera_id for camera in cameras),
        camera_positions=np.array([[0.0, 0.0, 1.0], [2.0, 0.0, 1.0]], dtype=np.float32),
        camera_wxyz=np.array(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
    )

    backend.start()
    backend.publish_manifest(manifest)
    assert backend.publish_frame(frame)
    assert backend.publish_frame(frame)
    original_handle = server.scene.mesh_handles[0]
    assert server.scene.camera_handles[0].visible
    assert not server.scene.camera_handles[1].visible
    image_frame = CameraImageFrame(
        run_id="run",
        scene_revision=1,
        sequence=1,
        sim_step=1,
        sim_time=0.01,
        images=tuple(
            CameraImage(
                camera_id=camera.camera_id,
                image=np.full((2, 3, 3), env_id + 1, dtype=np.uint8),
            )
            for env_id, camera in enumerate(cameras)
        ),
    )
    assert backend.publish_camera_images(image_frame)
    visible_images = [
        handle
        for handle in server.gui.image_handles
        if not handle.removed and handle.visible
    ]
    assert len(visible_images) == 1
    assert np.all(visible_images[0].image == 1)
    camera_environment = server.gui.dropdowns["Environment"]
    camera_environment.callback(SimpleNamespace(target=SimpleNamespace(value="1")))
    backend.poll()
    assert not server.scene.camera_handles[0].visible
    assert server.scene.camera_handles[1].visible
    visible_images = [
        handle
        for handle in server.gui.image_handles
        if not handle.removed and handle.visible
    ]
    assert len(visible_images) == 1
    assert np.all(visible_images[0].image == 2)

    environment_one = server.gui.checkboxes["Environment 1"]
    environment_one.callback(SimpleNamespace(target=SimpleNamespace(value=False)))
    backend.poll()
    assert not server.scene.camera_handles[1].visible

    refreshed_manifest = SceneManifest("run", 2, nodes, (geometry,), cameras)
    refreshed_frame = SceneFrame(
        run_id="run",
        scene_revision=2,
        sequence=2,
        sim_step=2,
        sim_time=0.02,
        node_ids=tuple(node.node_id for node in nodes),
        positions=frame.positions + 1.0,
        wxyz=frame.wxyz,
        visible=frame.visible,
        camera_ids=frame.camera_ids,
        camera_positions=frame.camera_positions,
        camera_wxyz=frame.camera_wxyz,
    )
    backend.publish_manifest(refreshed_manifest)
    assert backend.publish_frame(refreshed_frame)

    assert server.scene.mesh_uploads == 1
    assert server.scene.mesh_handles[0] is original_handle
    assert server.gui.checkboxes["Environment 1"].value is False
    np.testing.assert_allclose(original_handle.batched_opacities, [1.0, 0.0])

    backend.publish_manifest(SceneManifest("run", 3, (), ()))
    assert original_handle.removed
    assert all(handle.removed for handle in server.scene.camera_handles[:2])
    backend.stop()

    assert server.scene.mesh_uploads == 1
    assert server.stopped


def test_viser_backend_groups_sensor_and_record_camera_previews() -> None:
    server = _Server()
    backend = ViserBackend(ViserServerCfg(port=8765), server_factory=lambda **_: server)
    cameras = tuple(
        CameraSpec(
            camera_id=f"env:0/camera:{sensor_uid}",
            sensor_uid=sensor_uid,
            env_id=0,
            path=f"/envs/0/cameras/{sensor_uid}",
            fov_y=0.8,
            aspect=4.0 / 3.0,
            near=0.01,
            far=10.0,
            role=role,
        )
        for sensor_uid, role in (
            ("cam_high", "sensor"),
            ("record_camera", "record"),
        )
    )
    manifest = SceneManifest("run", 1, (), (), cameras)

    backend.start()
    backend.publish_manifest(manifest)

    preview_folder = server.gui.folders["RGB previews"]
    record_folder = server.gui.folders["Record cameras"]
    sensor_folder = server.gui.folders["Sensor cameras"]
    assert preview_folder.expand_by_default is True
    assert preview_folder.visible is True
    assert record_folder.parent_folder == "RGB previews"
    assert sensor_folder.parent_folder == "RGB previews"
    assert record_folder.expand_by_default is True
    assert sensor_folder.expand_by_default is True
    previews = {
        handle.label: handle
        for handle in server.gui.image_handles
        if not handle.removed and handle.visible
    }
    assert previews["record_camera RGB"].parent_folder == "Record cameras"
    assert previews["cam_high RGB"].parent_folder == "Sensor cameras"

    image_frame = CameraImageFrame(
        run_id="run",
        scene_revision=1,
        sequence=1,
        sim_step=1,
        sim_time=0.01,
        images=tuple(
            CameraImage(
                camera_id=camera.camera_id,
                image=np.full((2, 3, 3), index + 1, dtype=np.uint8),
            )
            for index, camera in enumerate(cameras)
        ),
    )
    assert backend.publish_camera_images(image_frame)
    previews = {
        handle.label: handle.image
        for handle in server.gui.image_handles
        if not handle.removed
    }
    assert np.all(previews["cam_high RGB"] == 1)
    assert np.all(previews["record_camera RGB"] == 2)

    rgb_preview = server.gui.checkboxes["RGB previews"]
    rgb_preview.callback(SimpleNamespace(target=SimpleNamespace(value=False)))
    backend.poll()
    assert preview_folder.visible is False
    assert record_folder.visible is False
    assert sensor_folder.visible is False
    assert not any(
        not handle.removed and handle.visible for handle in server.gui.image_handles
    )
    backend.stop()


def test_viser_backend_replay_slider_emits_client_seek_once() -> None:
    server = _Server()
    backend = ViserBackend(
        ViserServerCfg(port=8765),
        server_factory=lambda **_: server,
        allow_commands=True,
    )
    commands: list[int] = []
    backend.set_replay_control_command_sink(commands.append)

    backend.start()
    backend.publish_manifest(SceneManifest("run", 1, (), ()))
    backend.publish_replay_control(
        step=REPLAY_INITIAL_STEP,
        max_step=REPLAY_MAX_STEP,
        visible=True,
    )
    slider = server.gui.sliders["Frame"]
    slider.callback(
        SimpleNamespace(
            client_id="client-a",
            target=SimpleNamespace(value=REPLAY_TARGET_STEP),
        )
    )
    backend.poll()

    assert commands == [REPLAY_TARGET_STEP]
    backend.stop()


def test_viser_backend_ignores_server_originated_replay_slider_update() -> None:
    """Viser invokes callbacks with no client when backend code sets value."""
    server = _Server()
    backend = ViserBackend(
        ViserServerCfg(port=8765),
        server_factory=lambda **_: server,
        allow_commands=True,
    )
    commands: list[int] = []
    backend.set_replay_control_command_sink(commands.append)

    backend.start()
    backend.publish_manifest(SceneManifest("run", 1, (), ()))
    backend.publish_replay_control(
        step=REPLAY_INITIAL_STEP,
        max_step=REPLAY_MAX_STEP,
        visible=True,
    )
    slider = server.gui.sliders["Frame"]
    slider.callback(
        SimpleNamespace(
            client_id=None,
            target=SimpleNamespace(value=REPLAY_TARGET_STEP),
        )
    )
    backend.poll()

    assert commands == []
    backend.stop()


def test_viser_backend_replay_slider_tracks_progress_and_hides() -> None:
    server = _Server()
    backend = ViserBackend(ViserServerCfg(port=8765), server_factory=lambda **_: server)

    backend.start()
    backend.publish_manifest(SceneManifest("run", 1, (), ()))
    backend.publish_replay_control(
        step=REPLAY_INITIAL_STEP,
        max_step=REPLAY_MAX_STEP,
        visible=True,
    )
    folder = server.gui.folders["Replay control"]
    slider = server.gui.sliders["Frame"]
    backend.publish_replay_control(
        step=REPLAY_TARGET_STEP,
        max_step=REPLAY_MAX_STEP,
        visible=True,
    )
    assert slider.value == REPLAY_TARGET_STEP

    backend.publish_replay_control(
        step=REPLAY_TARGET_STEP,
        max_step=REPLAY_MAX_STEP,
        visible=False,
    )
    assert folder.removed is True
    assert slider.removed is True
    backend.stop()


def test_viser_backend_batches_large_scenes_without_per_env_gui_nodes() -> None:
    num_envs = 1024
    server = _Server()
    backend = ViserBackend(ViserServerCfg(port=8765), server_factory=lambda **_: server)
    geometry = MeshGeometry(
        geometry_id="sha256:large-batch",
        vertices=np.array(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]],
            dtype=np.float32,
        ),
        faces=np.array([[0, 1, 2]], dtype=np.uint32),
    )
    nodes = tuple(
        SceneNode(
            node_id=f"env:{env_id}/robot:robot/link:base",
            path=f"/envs/{env_id}/robots/robot/links/base",
            parent_id=f"env:{env_id}/robot:robot",
            env_id=env_id,
            kind="robot_link",
            geometry_id=geometry.geometry_id,
        )
        for env_id in range(num_envs)
    )
    positions = np.zeros((num_envs, 3), dtype=np.float32)
    positions[:, 0] = np.arange(num_envs, dtype=np.float32)
    frame = SceneFrame(
        run_id="large",
        scene_revision=1,
        sequence=1,
        sim_step=1,
        sim_time=0.01,
        node_ids=tuple(node.node_id for node in nodes),
        positions=positions,
        wxyz=np.tile(
            np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            (num_envs, 1),
        ),
        visible=np.ones((num_envs,), dtype=np.bool_),
    )

    backend.start()
    backend.publish_manifest(SceneManifest("large", 1, nodes, (geometry,)))
    assert backend.publish_frame(frame)

    assert server.scene.mesh_uploads == 1
    handle = server.scene.mesh_handles[0]
    assert handle.batched_positions.shape == (num_envs, 3)
    assert "Show all environments" in server.gui.checkboxes
    assert "Environment 1023" not in server.gui.checkboxes

    show_all = server.gui.checkboxes["Show all environments"]
    show_all.callback(SimpleNamespace(target=SimpleNamespace(value=False)))
    backend.poll()
    assert np.count_nonzero(handle.batched_opacities) == 1

    selected = server.gui.dropdowns["Selected environment"]
    selected.callback(SimpleNamespace(target=SimpleNamespace(value="17")))
    backend.poll()
    assert handle.batched_opacities[17] == 1.0
    assert np.count_nonzero(handle.batched_opacities) == 1
    backend.stop()


def test_viser_backend_recreates_dynamic_mesh_only_for_vertex_updates() -> None:
    server = _Server()
    backend = ViserBackend(ViserServerCfg(port=8765), server_factory=lambda **_: server)
    vertices = np.array(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]],
        dtype=np.float32,
    )
    geometry = MeshGeometry(
        geometry_id="sha256:cloth",
        vertices=vertices,
        faces=np.array([[0, 1, 2]], dtype=np.uint32),
    )
    node = SceneNode(
        node_id="env:0/cloth:flag",
        path="/envs/0/cloth_objects/flag",
        parent_id="env:0",
        env_id=0,
        kind="cloth_object",
        geometry_id=geometry.geometry_id,
        dynamic_geometry=True,
    )
    manifest = SceneManifest("run", 1, (node,), (geometry,))

    def make_frame(
        sequence: int,
        dynamic_meshes: tuple[DynamicMeshUpdate, ...] = (),
    ) -> SceneFrame:
        return SceneFrame(
            run_id="run",
            scene_revision=1,
            sequence=sequence,
            sim_step=sequence,
            sim_time=sequence * 0.01,
            node_ids=(node.node_id,),
            positions=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
            wxyz=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            visible=np.ones((1,), dtype=np.bool_),
            dynamic_meshes=dynamic_meshes,
        )

    backend.start()
    backend.publish_manifest(manifest)
    initial_handle = server.scene.dynamic_mesh_handles[-1]

    assert backend.publish_frame(make_frame(1))
    assert server.scene.dynamic_mesh_uploads == 1
    np.testing.assert_allclose(initial_handle.position, [1.0, 2.0, 3.0])

    updated_vertices = vertices + np.array([0.0, 0.0, 0.2], dtype=np.float32)
    assert backend.publish_frame(
        make_frame(
            2,
            (DynamicMeshUpdate(node_id=node.node_id, vertices=updated_vertices),),
        )
    )
    updated_handle = server.scene.dynamic_mesh_handles[-1]
    assert initial_handle.removed
    assert server.scene.dynamic_mesh_uploads == 2
    np.testing.assert_allclose(updated_handle.vertices, updated_vertices)
    np.testing.assert_allclose(updated_handle.position, [1.0, 2.0, 3.0])

    environment_zero = server.gui.checkboxes["Environment 0"]
    environment_zero.callback(SimpleNamespace(target=SimpleNamespace(value=False)))
    backend.poll()
    assert not updated_handle.visible

    backend.publish_manifest(SceneManifest("run", 2, (), ()))
    assert updated_handle.removed
    backend.stop()


def test_viser_backend_emits_owned_gizmo_drag_commands() -> None:
    server = _Server()
    commands = []
    backend = ViserBackend(
        ViserServerCfg(port=8765),
        server_factory=lambda **_: server,
        allow_commands=True,
    )
    backend.set_gizmo_command_sink(commands.append)
    spec = GizmoSpec(
        gizmo_id="cube",
        target_uid="cube",
        target_type="rigid_object",
        control_part=None,
        env_id=0,
        path="/interactions/gizmos/cube",
    )
    manifest = SceneManifest("run", 1, (), (), gizmos=(spec,))
    frame = SceneFrame(
        run_id="run",
        scene_revision=1,
        sequence=1,
        sim_step=1,
        sim_time=0.01,
        node_ids=(),
        positions=np.empty((0, 3), dtype=np.float32),
        wxyz=np.empty((0, 4), dtype=np.float32),
        visible=np.empty((0,), dtype=np.bool_),
        gizmos=(
            GizmoState(
                gizmo_id="cube",
                position=np.array([0.0, 0.0, 1.0], dtype=np.float32),
                wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            ),
        ),
    )

    backend.start()
    backend.publish_manifest(manifest)
    assert backend.publish_frame(frame)
    handle = server.scene.transform_controls[0]

    def event(client_id: str, position: list[float]) -> SimpleNamespace:
        return SimpleNamespace(
            client_id=client_id,
            target=SimpleNamespace(
                position=np.asarray(position, dtype=np.float32),
                wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            ),
        )

    handle.drag_start_callback(event("client-a", [0.0, 0.0, 1.0]))
    handle.drag_start_callback(event("client-b", [5.0, 0.0, 1.0]))
    handle.update_callback(event("client-a", [0.2, 0.0, 1.0]))
    backend.poll()

    assert [(command.client_id, command.phase) for command in commands] == [
        ("client-a", "start"),
        ("client-a", "update"),
    ]

    handle.position = np.array([0.2, 0.0, 1.0], dtype=np.float32)
    assert backend.publish_frame(frame)
    np.testing.assert_allclose(handle.position, [0.2, 0.0, 1.0])

    handle.drag_end_callback(event("client-a", [0.2, 0.0, 1.0]))
    backend.poll()
    assert commands[-1].phase == "end"

    assert backend.publish_frame(frame)
    np.testing.assert_allclose(handle.position, [0.0, 0.0, 1.0])

    handle.drag_start_callback(event("client-a", [0.0, 0.0, 1.0]))
    handle.update_callback(event("client-a", [0.4, 0.0, 1.0]))
    backend.poll()
    server.disconnect_callback(SimpleNamespace(client_id="client-a"))
    backend.poll()
    assert commands[-1].phase == "end"
    np.testing.assert_allclose(commands[-1].position, [0.4, 0.0, 1.0])
    backend.stop()


def test_viser_backend_keeps_gizmos_read_only_without_command_permission() -> None:
    server = _Server()
    backend = ViserBackend(
        ViserServerCfg(port=8765),
        server_factory=lambda **_: server,
    )
    spec = GizmoSpec(
        gizmo_id="cube",
        target_uid="cube",
        target_type="rigid_object",
        control_part=None,
        env_id=0,
        path="/interactions/gizmos/cube",
    )

    backend.start()
    backend.publish_manifest(SceneManifest("run", 1, (), (), gizmos=(spec,)))

    assert not server.scene.transform_controls
    backend.stop()


def test_viser_backend_controls_articulation_joints_in_display_units() -> None:
    server = _Server()
    commands = []
    backend = ViserBackend(
        ViserServerCfg(port=8765),
        server_factory=lambda **_: server,
        allow_commands=True,
    )
    backend.set_joint_control_command_sink(commands.append)
    revolute = JointControlSpec(
        control_id="door-hinge",
        articulation_uid="door",
        env_id=0,
        joint_id=0,
        joint_name="base_to_lower_arm",
        joint_type="revolute",
        lower=-np.pi / 2.0,
        upper=np.pi / 2.0,
        step=np.pi / 180.0,
        initial_value=0.0,
    )
    prismatic = JointControlSpec(
        control_id="door-slide",
        articulation_uid="door",
        env_id=0,
        joint_id=1,
        joint_name="slide",
        joint_type="prismatic",
        lower=0.0,
        upper=None,
        step=0.001,
        initial_value=0.2,
    )

    def frame(
        sequence: int,
        hinge_value: float,
        applied_sequence: int,
    ) -> SceneFrame:
        return SceneFrame(
            run_id="run",
            scene_revision=1,
            sequence=sequence,
            sim_step=sequence,
            sim_time=sequence * 0.01,
            node_ids=(),
            positions=np.empty((0, 3), dtype=np.float32),
            wxyz=np.empty((0, 4), dtype=np.float32),
            visible=np.empty((0,), dtype=np.bool_),
            joint_controls=(
                JointControlState(
                    "door-hinge",
                    value=hinge_value,
                    applied_sequence=applied_sequence,
                ),
                JointControlState("door-slide", value=0.2),
            ),
        )

    backend.start()
    backend.publish_manifest(
        SceneManifest(
            "run",
            1,
            (),
            (),
            joint_controls=(revolute, prismatic),
        )
    )
    assert backend.publish_frame(frame(1, -1.0e-9, 0))

    hinge = server.gui.sliders["lower arm (°)"]
    slide = server.gui.numbers["slide (m)"]
    assert hinge.min == -90.0
    assert hinge.max == 90.0
    assert hinge.step == 1.0
    assert hinge.value == 0.0
    assert not np.signbit(hinge.value)
    assert hinge.marks == ()
    assert "base_to_lower_arm" in hinge.hint
    assert "range -90.00 … 90.00 °" in hinge.hint
    assert slide.min == 0.0
    assert slide.max is None

    hinge.value = 45.0
    hinge.callback(SimpleNamespace(client_id="client-a", target=hinge))
    backend.poll()

    assert len(commands) == 1
    assert commands[0].control_id == "door-hinge"
    np.testing.assert_allclose(commands[0].value, np.pi / 4.0)

    assert backend.publish_frame(frame(2, 0.0, 0))
    assert hinge.value == 45.0
    assert backend.publish_frame(frame(3, np.pi / 6.0, commands[0].sequence))
    np.testing.assert_allclose(hinge.value, 30.0)

    reset = server.gui.buttons["Reset articulation"]
    reset.callback(SimpleNamespace(client_id="client-a"))
    backend.poll()
    assert commands[-2].control_id == "door-hinge"
    assert commands[-2].value == 0.0
    assert commands[-1].control_id == "door-slide"
    assert commands[-1].value == 0.2
    backend.stop()


def test_viser_backend_disambiguates_compact_joint_labels() -> None:
    specs = tuple(
        JointControlSpec(
            control_id=f"arm-{joint_id}",
            articulation_uid="robot",
            env_id=0,
            joint_id=joint_id,
            joint_name=joint_name,
            joint_type="revolute",
            lower=-1.0,
            upper=1.0,
            step=np.pi / 180.0,
            initial_value=0.0,
        )
        for joint_id, joint_name in enumerate(("base_to_arm", "shoulder_to_arm"))
    )

    assert ViserBackend._joint_control_labels(specs) == {
        "arm-0": "arm [0] (°)",
        "arm-1": "arm [1] (°)",
    }
