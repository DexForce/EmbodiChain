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

from dataclasses import dataclass, field, replace
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from dexsim.scene.objects import SpawnedRigidBody
from embodichain.lab.visualization.protocol import MeshMarkerOverlay
from embodichain.lab.visualization.markers import _native
from embodichain.lab.visualization.markers._native import NativeMarkerRenderer


# These explicit SDK stand-ins keep consumer transactions testable with the
# released DexSim wheel. Actual descriptor/native compatibility is covered by
# the GPU integration tests and DexSim's own contract tests, not this fake.
@dataclass
class _SdkGeometryDesc:
    vertices: np.ndarray
    triangles: np.ndarray

    @classmethod
    def mesh(cls, *, vertices: np.ndarray, triangles: np.ndarray):
        return cls(vertices, triangles)


@dataclass
class _SdkMaterialDesc:
    name: str
    base_color: tuple[float, float, float, float] | None = None
    double_sided: bool | None = None
    unlit: bool | None = field(default=None, kw_only=True)
    alpha_mode: str | None = field(default=None, kw_only=True)
    depth_write: bool | None = field(default=None, kw_only=True)


@dataclass
class _SdkRenderDesc(_SdkGeometryDesc):
    render_mode: str = field(default="scene", kw_only=True)
    cast_shadow: bool = field(default=True, kw_only=True)
    pickable: bool = field(default=True, kw_only=True)
    material: _SdkMaterialDesc | None = None

    @classmethod
    def from_geometry(
        cls,
        geometry: _SdkGeometryDesc,
        *,
        render_mode: str = "scene",
        cast_shadow: bool = True,
        pickable: bool = True,
        material: _SdkMaterialDesc | None = None,
    ):
        return cls(
            geometry.vertices,
            geometry.triangles,
            render_mode=render_mode,
            cast_shadow=cast_shadow,
            pickable=pickable,
            material=material,
        )


@dataclass
class _SdkMeshObjectDesc:
    name: str
    renders: list[_SdkRenderDesc] = field(default_factory=list)
    physics: object | None = None
    per_env: bool = True
    collisions: list[object] = field(default_factory=list)
    scale: np.ndarray = field(default_factory=lambda: np.ones(3, dtype=np.float32))


@pytest.fixture(autouse=True)
def sdk_descriptors(monkeypatch):
    sdk = SimpleNamespace(
        GeometryDesc=_SdkGeometryDesc,
        RenderDesc=_SdkRenderDesc,
        MaterialDesc=_SdkMaterialDesc,
        MeshObjectDesc=_SdkMeshObjectDesc,
    )

    def import_sdk(name):
        assert name == "dexsim.spawn"
        return sdk

    monkeypatch.setattr(_native, "import_module", import_sdk)
    return sdk


class NativeActor:
    """Render-state storage behind the real stable SpawnedRigidBody facade."""

    def __init__(self, desc):
        self.color = tuple(desc.renders[0].material.base_color)
        self.visible = True
        self.pose = np.eye(4)
        self.scale = (1, 1, 1)

    def set_world_pose(self, pose, sync=True):
        self.pose = pose.copy()

    def set_scale(self, x, y, z, sync=True):
        self.scale = (x, y, z)

    def set_visible(self, visible):
        self.visible = visible

    def get_material(self):
        return self

    def set_base_color(self, color):
        self.color = tuple(color)


class Scene:
    """Fake Scene allocation/removal while retaining real stable handles."""

    def __init__(self):
        self.handles = {}
        self.created = 0
        self.fail = False
        self.closed = False
        self.removed = []

    def add_mesh_object(self, desc, *, arena_name="default"):
        assert arena_name == "default"
        assert isinstance(desc, _SdkMeshObjectDesc)
        if self.fail:
            raise RuntimeError("upload failed")
        self.created += 1
        assert desc.name not in self.handles
        handle = SpawnedRigidBody(
            path=desc.name,
            object_desc=desc,
            _render_actor=NativeActor(desc),
            _physics_binding=None,
            _arena_index=0,
        )
        handle._result_ref = weakref.ref(self)
        self.handles[handle.path] = handle
        return handle

    def remove_mesh_object(self, path):
        assert not self.closed, "must not remove resources after Scene.close()"
        self.removed.append(path)
        self.handles.pop(path).discard()

    def close(self):
        self.closed = True
        for handle in self.handles.values():
            handle.discard()
        self.handles.clear()


def marker():
    return MeshMarkerOverlay(
        "marker",
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        np.array([[0, 1, 2]]),
        np.array([1, 2, 3]),
        np.array([0.5, 0.5, 0.5, 0.5]),
    )


def test_updates_reuse_scene_handles_and_keep_world_pose_scale_and_alpha():
    scene = Scene()
    renderer = NativeMarkerRenderer(scene)
    renderer.publish("g", (marker(),))
    handle = next(iter(scene.handles.values()))
    native = handle.native()
    np.testing.assert_allclose(native.pose[:3, 3], [1, 2, 3])
    np.testing.assert_allclose(native.pose[:3, :3], [[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    renderer.publish(
        "g",
        (
            replace(
                marker(), color=(0, 1, 0, 0.2), scale=np.array([2, 3, 4]), visible=False
            ),
        ),
    )
    assert scene.created == 1
    assert native.color == (0, 1, 0, 0.2) and native.scale == (2, 3, 4)
    np.testing.assert_array_equal(handle.desc.scale, [2, 3, 4])
    assert not native.visible
    path = handle.path
    renderer.remove("g")
    assert scene.removed == [path]
    assert scene.handles == {} and not handle.is_valid
    renderer.close()


def test_mesh_descriptor_requests_overlay_without_physics():
    scene = Scene()
    renderer = NativeMarkerRenderer(scene)
    snapshot = replace(marker(), color=(0.2, 0.4, 0.6, 0.3))
    renderer.publish("g", (snapshot,))
    handle = next(iter(scene.handles.values()))
    desc = handle.desc
    assert desc.physics is None and desc.collisions == [] and not desc.per_env
    (render,) = desc.renders
    assert isinstance(render, _SdkRenderDesc)
    np.testing.assert_array_equal(render.vertices, snapshot.vertices)
    np.testing.assert_array_equal(render.triangles, snapshot.faces)
    assert render.render_mode == "overlay"
    assert render.cast_shadow is False and render.pickable is False
    material = render.material
    assert isinstance(material, _SdkMaterialDesc)
    assert material.base_color == snapshot.color
    assert material.unlit and material.alpha_mode == "blend"
    assert material.depth_write is False and material.double_sided


def test_failed_geometry_replacement_retains_previous_scene_handles():
    scene = Scene()
    renderer = NativeMarkerRenderer(scene)
    renderer.publish("g", (marker(),))
    previous = dict(scene.handles)
    scene.fail = True
    with pytest.raises(RuntimeError, match="upload failed"):
        renderer.publish("g", (replace(marker(), vertices=marker().vertices * 2),))
    assert scene.handles == previous
    renderer.close()
    assert scene.handles == {}


def test_scene_paths_are_unique_across_groups_environments_and_renderers():
    scene = Scene()
    first = NativeMarkerRenderer(scene)
    second = NativeMarkerRenderer(scene)
    first.publish("g", (marker(), replace(marker(), overlay_id="marker/env1")))
    first.publish("h", (marker(),))
    second.publish("g", (marker(),))
    assert len(scene.handles) == 4
    first.remove("g")
    assert len(scene.handles) == 2
    second.close()
    first.close()
    assert scene.handles == {}


def test_geometry_replacement_releases_old_handle_after_success():
    scene = Scene()
    renderer = NativeMarkerRenderer(scene)
    renderer.publish("g", (marker(),))
    old_handle = next(iter(scene.handles.values()))
    renderer.publish("g", (replace(marker(), vertices=marker().vertices * 2),))
    assert scene.created == 2 and len(scene.handles) == 1
    assert old_handle.path not in scene.handles and not old_handle.is_valid
    renderer.close()
    assert scene.handles == {}


def test_later_creation_failure_rolls_back_updates_and_new_handles(monkeypatch):
    scene = Scene()
    renderer = NativeMarkerRenderer(scene)
    renderer.publish("g", (marker(),))
    original = next(iter(scene.handles.values()))
    native = original.native()
    before = dict(scene.handles)
    original_create = scene.add_mesh_object

    def create(*args, **kwargs):
        if scene.created == 2:
            raise RuntimeError("upload failed")
        return original_create(*args, **kwargs)

    monkeypatch.setattr(scene, "add_mesh_object", create)
    with pytest.raises(RuntimeError, match="upload failed"):
        renderer.publish(
            "g",
            (
                replace(marker(), position=np.array([9, 8, 7]), color=(1, 0, 0, 0.2)),
                replace(marker(), overlay_id="new"),
                replace(marker(), overlay_id="failed"),
            ),
        )
    assert scene.handles == before
    np.testing.assert_allclose(native.pose[:3, 3], [1, 2, 3])
    assert native.color == marker().color


def test_apply_failure_removes_staged_handle_and_restores_previous_state(monkeypatch):
    scene = Scene()
    renderer = NativeMarkerRenderer(scene)
    renderer.publish("g", (marker(),))
    original = next(iter(scene.handles.values()))
    native = original.native()
    before = dict(scene.handles)
    set_visible = NativeActor.set_visible

    def fail_new_handle(handle, visible):
        if handle is not native:
            raise RuntimeError("visibility update failed")
        set_visible(handle, visible)

    monkeypatch.setattr(NativeActor, "set_visible", fail_new_handle)
    with pytest.raises(RuntimeError, match="visibility update failed"):
        renderer.publish(
            "g",
            (
                replace(marker(), scale=np.array([3, 2, 1]), visible=False),
                replace(marker(), overlay_id="new"),
            ),
        )
    assert scene.handles == before
    assert native.visible and native.scale == (1, 1, 1)
    np.testing.assert_array_equal(original.desc.scale, [1, 1, 1])


def test_remove_and_close_are_idempotent_after_scene_close():
    scene = Scene()
    renderer = NativeMarkerRenderer(scene)
    renderer.publish("g", (marker(),))
    renderer.publish("h", (marker(),))
    scene.close()
    renderer.remove("g")
    renderer.remove("g")
    renderer.close()
    renderer.close()
    assert scene.removed == []
    assert renderer._groups == {}


def test_missing_scene_capability_fails_before_allocation(monkeypatch):
    scene = Scene()
    monkeypatch.setattr(scene, "add_mesh_object", None)
    with pytest.raises(RuntimeError, match="Install a DexSim build"):
        NativeMarkerRenderer(scene)
    assert not scene.handles


def test_missing_spawn_module_fails_with_installation_guidance(monkeypatch):
    def missing_sdk(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(_native, "import_module", missing_sdk)
    with pytest.raises(RuntimeError, match="Install a DexSim build"):
        NativeMarkerRenderer(Scene())


def test_updates_follow_stable_handle_after_native_rebind():
    scene = Scene()
    renderer = NativeMarkerRenderer(scene)
    renderer.publish("g", (marker(),))
    handle = next(iter(scene.handles.values()))
    old_native = handle.native()
    replacement = SpawnedRigidBody(
        path=handle.path,
        object_desc=handle.desc,
        _render_actor=NativeActor(handle.desc),
        _physics_binding=None,
        _arena_index=0,
    )
    handle.rebind(replacement)
    renderer.publish("g", (replace(marker(), position=np.array([4, 5, 6])),))
    assert scene.created == 1
    assert next(iter(scene.handles.values())) is handle
    np.testing.assert_array_equal(handle.native().pose[:3, 3], [4, 5, 6])
    np.testing.assert_array_equal(old_native.pose[:3, 3], [1, 2, 3])
    renderer.close()
    assert not handle.is_valid


@pytest.mark.parametrize(
    "owner, field_name",
    [
        ("RenderDesc", "render_mode"),
        ("MaterialDesc", "unlit"),
        ("MaterialDesc", "alpha_mode"),
        ("MaterialDesc", "depth_write"),
        (None, "MeshObjectDesc"),
    ],
)
def test_missing_descriptor_capability_fails_before_allocation(
    monkeypatch, sdk_descriptors, owner, field_name
):
    target = sdk_descriptors if owner is None else getattr(sdk_descriptors, owner)
    monkeypatch.delattr(target, field_name)
    scene = Scene()
    with pytest.raises(RuntimeError, match="Install a DexSim build"):
        NativeMarkerRenderer(scene)
    assert not scene.handles
