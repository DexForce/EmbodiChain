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

from dataclasses import replace
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from embodichain.lab.visualization.protocol import MeshMarkerOverlay
from embodichain.lab.visualization.markers._native import NativeMarkerRenderer


class Handle:
    def __init__(self, name, color):
        self.name, self.color, self.visible = name, color, True

    def get_name(self):
        return self.name

    def set_world_pose(self, pose):
        self.pose = pose.copy()

    def set_scale(self, x, y, z):
        self.scale = (x, y, z)

    def set_visible(self, visible):
        self.visible = visible

    def get_material(self):
        return self

    def set_base_color(self, color):
        self.color = tuple(color)


class Arena:
    def __init__(self):
        self.handles = {}
        self.created = 0
        self.fail = False
        self.descriptions = []

    def create_render_actor(self, desc):
        if self.fail:
            raise RuntimeError("upload failed")
        self.created += 1
        assert desc.name not in self.handles
        self.descriptions.append(desc)
        handle = Handle(desc.name, desc.renders[0].material.base_color)
        self.handles[handle.name] = handle
        return handle

    def remove_actor(self, name):
        self.handles.pop(name)


@pytest.fixture(autouse=True)
def spawn_api(monkeypatch):
    # Recording the public Spawn boundary keeps this consumer suite CPU-only,
    # including when the installed wheel predates generic render descriptors.
    class GeometryDesc:
        @staticmethod
        def mesh(*, vertices, triangles):
            return SimpleNamespace(vertices=vertices, triangles=triangles)

    class RenderDesc:
        @staticmethod
        def from_geometry(geometry, **kwargs):
            return SimpleNamespace(**vars(geometry), **kwargs)

    api = SimpleNamespace(
        GeometryDesc=GeometryDesc,
        RenderDesc=RenderDesc,
        MaterialDesc=SimpleNamespace,
        ObjectDesc=SimpleNamespace,
        create_render_actor=lambda arena, desc: arena.create_render_actor(desc),
    )
    monkeypatch.setitem(sys.modules, "dexsim.spawn", api)
    return api


def marker():
    return MeshMarkerOverlay(
        "marker",
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        np.array([[0, 1, 2]]),
        np.array([1, 2, 3]),
        np.array([0.5, 0.5, 0.5, 0.5]),
    )


def test_updates_reuse_native_resources_and_keep_world_pose_scale_and_alpha():
    arena = Arena()
    renderer = NativeMarkerRenderer(arena)
    renderer.publish("g", (marker(),))
    handle = next(iter(arena.handles.values()))
    np.testing.assert_allclose(handle.pose[:3, 3], [1, 2, 3])
    np.testing.assert_allclose(handle.pose[:3, :3], [[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    renderer.publish(
        "g",
        (
            replace(
                marker(), color=(0, 1, 0, 0.2), scale=np.array([2, 3, 4]), visible=False
            ),
        ),
    )
    assert arena.created == 1
    assert handle.color == (0, 1, 0, 0.2) and handle.scale == (2, 3, 4)
    assert not handle.visible
    renderer.remove("g")
    assert arena.handles == {}
    renderer.close()


def test_failed_geometry_replacement_retains_previous_native_handles():
    arena = Arena()
    renderer = NativeMarkerRenderer(arena)
    renderer.publish("g", (marker(),))
    previous = dict(arena.handles)
    arena.fail = True
    with pytest.raises(RuntimeError, match="upload failed"):
        renderer.publish("g", (replace(marker(), vertices=marker().vertices * 2),))
    assert arena.handles == previous
    renderer.close()
    assert arena.handles == {}


def test_generic_render_descriptor_keeps_markers_out_of_physics_and_sensors():
    arena = Arena()
    renderer = NativeMarkerRenderer(arena)
    snapshot = replace(marker(), color=(0.2, 0.4, 0.6, 0.3))
    renderer.publish("g", (snapshot,))
    desc = arena.descriptions[0]
    assert desc.physics is None
    assert desc.per_env is False
    assert len(desc.renders) == 1
    render = desc.renders[0]
    np.testing.assert_array_equal(render.vertices, snapshot.vertices)
    np.testing.assert_array_equal(render.triangles, snapshot.faces)
    assert render.render_mode == "overlay"
    assert render.cast_shadow is False and render.pickable is False
    assert render.material.base_color == snapshot.color
    assert render.material.unlit is True
    assert render.material.alpha_mode == "blend"
    assert render.material.depth_write is False
    assert render.material.double_sided is True


def test_native_names_are_unique_across_groups_environments_and_renderers():
    arena = Arena()
    first = NativeMarkerRenderer(arena)
    second = NativeMarkerRenderer(arena)
    first.publish("g", (marker(), replace(marker(), overlay_id="marker/env1")))
    first.publish("h", (marker(),))
    second.publish("g", (marker(),))
    assert len(arena.handles) == 4
    first.remove("g")
    assert len(arena.handles) == 2
    second.close()
    first.close()
    assert arena.handles == {}


def test_geometry_replacement_releases_old_actor_after_success():
    arena = Arena()
    renderer = NativeMarkerRenderer(arena)
    renderer.publish("g", (marker(),))
    old_name = next(iter(arena.handles))
    renderer.publish("g", (replace(marker(), vertices=marker().vertices * 2),))
    assert arena.created == 2
    assert len(arena.handles) == 1
    assert old_name not in arena.handles
    renderer.close()
    assert arena.handles == {}


def test_later_creation_failure_rolls_back_updates_and_new_actors(
    monkeypatch, spawn_api
):
    arena = Arena()
    renderer = NativeMarkerRenderer(arena)
    renderer.publish("g", (marker(),))
    original = next(iter(arena.handles.values()))
    before = dict(arena.handles)

    def create(arena, desc):
        if arena.created == 2:
            raise RuntimeError("upload failed")
        return arena.create_render_actor(desc)

    monkeypatch.setattr(spawn_api, "create_render_actor", create)
    with pytest.raises(RuntimeError, match="upload failed"):
        renderer.publish(
            "g",
            (
                replace(marker(), position=np.array([9, 8, 7]), color=(1, 0, 0, 0.2)),
                replace(marker(), overlay_id="new"),
                replace(marker(), overlay_id="failed"),
            ),
        )
    assert arena.handles == before
    np.testing.assert_allclose(original.pose[:3, 3], [1, 2, 3])
    assert original.color == marker().color


def test_missing_spawn_factory_fails_explicitly(monkeypatch, spawn_api):
    monkeypatch.delattr(spawn_api, "create_render_actor")
    with pytest.raises(RuntimeError, match="create_render_actor"):
        NativeMarkerRenderer(Arena())


def test_missing_native_binding_error_is_preserved_without_leaks(
    monkeypatch, spawn_api
):
    def unsupported(arena, desc):
        raise RuntimeError("Install a DexSim build with MaterialInst.set_unlit")

    monkeypatch.setattr(spawn_api, "create_render_actor", unsupported)
    arena = Arena()
    renderer = NativeMarkerRenderer(arena)
    with pytest.raises(RuntimeError, match="MaterialInst.set_unlit"):
        renderer.publish("g", (marker(),))
    assert arena.handles == {}


def test_apply_failure_removes_staged_actor_and_restores_previous_state(monkeypatch):
    arena = Arena()
    renderer = NativeMarkerRenderer(arena)
    renderer.publish("g", (marker(),))
    original = next(iter(arena.handles.values()))
    before = dict(arena.handles)
    set_visible = Handle.set_visible

    def fail_new_handle_once(handle, visible):
        if handle is not original:
            raise RuntimeError("visibility update failed")
        set_visible(handle, visible)

    monkeypatch.setattr(Handle, "set_visible", fail_new_handle_once)
    with pytest.raises(RuntimeError, match="visibility update failed"):
        renderer.publish(
            "g",
            (
                replace(marker(), scale=np.array([3, 2, 1]), visible=False),
                replace(marker(), overlay_id="new"),
            ),
        )
    assert arena.handles == before
    assert original.visible
    assert original.scale == (1, 1, 1)


def test_missing_spawn_module_fails_with_installation_guidance(monkeypatch):
    monkeypatch.setitem(sys.modules, "dexsim.spawn", None)
    with pytest.raises(RuntimeError, match="Install a DexSim build"):
        NativeMarkerRenderer(Arena())
