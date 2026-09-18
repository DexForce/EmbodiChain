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

    def create_debug_mesh(self, vertices, triangles, rgba):
        if self.fail:
            raise RuntimeError("upload failed")
        self.created += 1
        handle = Handle(str(self.created), rgba)
        self.handles[handle.name] = handle
        return handle

    def remove_actor(self, name):
        self.handles.pop(name)


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


def test_missing_engine_capability_fails_explicitly():
    with pytest.raises(RuntimeError, match="create_debug_mesh"):
        NativeMarkerRenderer(object())
