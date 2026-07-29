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
    MeshGeometry,
    SceneFrame,
    SceneManifest,
    SceneNode,
    ViserServerCfg,
)
from embodichain.lab.visualization.backends.viser import ViserBackend


class _Handle(SimpleNamespace):
    def remove(self) -> None:
        self.removed = True


class _Folder:
    def __enter__(self) -> _Folder:
        return self

    def __exit__(self, *args: object) -> None:
        pass


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


class _Gui:
    def __init__(self) -> None:
        self.checkboxes: dict[str, _Checkbox] = {}
        self.dropdowns: dict[str, _Dropdown] = {}
        self.image_handles: list[_Handle] = []

    def reset(self) -> None:
        self.checkboxes.clear()
        self.dropdowns.clear()
        self.image_handles.clear()

    def add_markdown(self, content: str) -> _Handle:
        return _Handle(content=content)

    def add_folder(self, label: str) -> _Folder:
        return _Folder()

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

    def add_image(self, image: np.ndarray, **kwargs: object) -> _Handle:
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

    def reset(self) -> None:
        pass

    def set_up_direction(self, direction: str) -> None:
        self.up_direction = direction

    def add_frame(self, name: str, **kwargs: object) -> _Handle:
        return _Handle(name=name, visible=True, **kwargs)

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


class _Server:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.scene = _Scene()
        self.gui = _Gui()
        self.stopped = False

    def get_port(self) -> int:
        return 8765

    def get_clients(self) -> dict[str, object]:
        return {}

    def flush(self) -> None:
        pass

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
    assert np.all(server.gui.image_handles[-1].image == 1)
    camera_environment = server.gui.dropdowns["Environment"]
    camera_environment.callback(SimpleNamespace(target=SimpleNamespace(value="1")))
    backend.poll()
    assert not server.scene.camera_handles[0].visible
    assert server.scene.camera_handles[1].visible
    assert np.all(server.gui.image_handles[-1].image == 2)

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
