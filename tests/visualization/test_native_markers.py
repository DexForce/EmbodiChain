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
    def __init__(self, name):
        self.name, self.color, self.visible = name, (1, 1, 1, 1), True
        self.node = self
        self.events = []

    def get_name(self):
        return self.name

    def get_render_body(self):
        return self

    def set_raytrace_visible(self, enabled):
        self.raytrace = enabled
        self.events.append("routing")

    def set_shadow(self, enabled):
        self.shadow = enabled

    def set_pickable(self, enabled):
        self.pickable = enabled

    def add_mesh(self, vertices, indices, *, auto_build):
        assert not auto_build
        self.vertices, self.indices = vertices.copy(), indices.copy()
        self.events.append("mesh")
        return 0

    def default_material(self):
        return self

    def set_material(self, mesh_id, material, *, owned):
        assert mesh_id == 0 and material is self and owned
        self.events.append("owned_material")
        return 1

    def set_unlit(self, enabled):
        self.unlit = enabled

    def set_alpha_mode(self, mode):
        self.alpha_mode = mode

    def set_depth_write(self, enabled):
        self.depth_write = enabled

    def set_double_sided(self, enabled):
        self.double_sided = enabled

    def build(self):
        self.events.append("build")
        return True

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

    def create_actor(self, name, create_render_body, attach_scene):
        assert create_render_body and not attach_scene
        if self.fail:
            raise RuntimeError("upload failed")
        self.created += 1
        assert name not in self.handles
        handle = Handle(name)
        self.handles[name] = handle
        return handle

    def attach(self, node):
        node.events.append("attach")
        return 0

    def remove_actor(self, name):
        self.handles.pop(name)


@pytest.fixture(autouse=True)
def native_api(monkeypatch):
    api = SimpleNamespace(
        RenderBody=Handle, MaterialInst=Handle, AlphaMode=SimpleNamespace(BLEND="blend")
    )
    monkeypatch.setitem(sys.modules, "dexsim.engine", api)
    # A marker must not depend on any Spawn factory or descriptor.
    monkeypatch.setitem(sys.modules, "dexsim.spawn", None)
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


def test_plain_mesh_object_configures_overlay_before_build_without_physics():
    arena = Arena()
    renderer = NativeMarkerRenderer(arena)
    snapshot = replace(marker(), color=(0.2, 0.4, 0.6, 0.3))
    renderer.publish("g", (snapshot,))
    handle = next(iter(arena.handles.values()))
    np.testing.assert_array_equal(handle.vertices, snapshot.vertices)
    np.testing.assert_array_equal(handle.indices, snapshot.faces.flatten())
    assert (
        handle.raytrace is False and handle.shadow is False and handle.pickable is False
    )
    assert handle.color == snapshot.color
    assert handle.unlit and handle.alpha_mode == "blend"
    assert not handle.depth_write and handle.double_sided
    assert handle.events == ["routing", "mesh", "owned_material", "build", "attach"]


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


def test_later_creation_failure_rolls_back_updates_and_new_actors(monkeypatch):
    arena = Arena()
    renderer = NativeMarkerRenderer(arena)
    renderer.publish("g", (marker(),))
    original = next(iter(arena.handles.values()))
    before = dict(arena.handles)

    original_create = arena.create_actor

    def create(*args):
        if arena.created == 2:
            raise RuntimeError("upload failed")
        return original_create(*args)

    monkeypatch.setattr(arena, "create_actor", create)
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


@pytest.mark.parametrize("stage", ["add_mesh", "set_unlit", "build", "attach"])
def test_creation_failure_removes_incomplete_actor(monkeypatch, stage):
    arena = Arena()
    renderer = NativeMarkerRenderer(arena)
    owner = arena if stage == "attach" else Handle
    if stage == "set_unlit":

        def fail(*args, **kwargs):
            raise RuntimeError("material failed")

    else:

        def fail(*args, **kwargs):
            return False if stage == "build" else -1

    monkeypatch.setattr(owner, stage, fail)
    with pytest.raises(RuntimeError):
        renderer.publish("g", (marker(),))
    assert arena.handles == {}
    assert renderer._groups == {}


def test_missing_native_binding_fails_before_allocation(monkeypatch, native_api):
    monkeypatch.delattr(native_api, "AlphaMode")
    arena = Arena()
    with pytest.raises(RuntimeError, match="Install a DexSim build"):
        NativeMarkerRenderer(arena)
    assert not arena.handles


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


def test_missing_native_module_fails_with_installation_guidance(monkeypatch):
    monkeypatch.setitem(sys.modules, "dexsim.engine", None)
    with pytest.raises(RuntimeError, match="Install a DexSim build"):
        NativeMarkerRenderer(Arena())
