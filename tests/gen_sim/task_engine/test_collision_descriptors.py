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

"""Provider-free qualification of retained collision materialization inputs."""

from __future__ import annotations

from types import SimpleNamespace
import weakref

import numpy as np
import pytest
import trimesh

from embodichain.gen_sim.task_engine._task_program import (
    drawer_geometry,
    scene_geometry,
)


def _fixture(monkeypatch, approximation="convexDecomposition", scale=2.0):
    from dexsim.scene import SpawnedArticulation
    from dexsim.spawn.descs import (
        ArticulationDesc,
        CollisionApproximation,
        CollisionDesc,
        GeometryType,
        LinkDesc,
        UsdSourceDesc,
    )

    shell = trimesh.creation.annulus(r_min=0.2, r_max=0.3, height=0.5)
    local = np.eye(4, dtype=np.float32)
    local[:3, 3] = (0.4, 0.1, 0.2)
    collider = CollisionDesc(
        geometry_type=GeometryType.MESH,
        vertices=np.asarray(shell.vertices, dtype=np.float32),
        triangles=np.asarray(shell.faces, dtype=np.int32),
        local_transform=local,
        approximation=CollisionApproximation(approximation),
        usd=UsdSourceDesc(file_path="source.usdc", prim_path="/root/base/shell"),
    )
    desc = ArticulationDesc(
        name="native_cabinet",
        links=[
            LinkDesc(
                name="base", parent_name=None, pose=np.eye(4), collisions=[collider]
            )
        ],
        usd=UsdSourceDesc(file_path="source.usdc"),
        body_scale=np.full(3, scale, dtype=np.float32),
    )

    class Scene:
        backend = "dexsim"
        arenas = SimpleNamespace(names=["env0"])

        def get_paths(self):
            return list(self.handles)

        def get(self, path):
            return self.handles[path]

        def get_link_desc(self, path, name):
            return next(link for link in self.get(path).desc.links if link.name == name)

    scene = Scene()
    handle = SpawnedArticulation(
        path="env0/native_cabinet",
        articulation_desc=desc,
        render_articulation=object(),
        _physics_binding=SimpleNamespace(get_link_names=lambda _order: ["base"]),
        _arena_index=0,
    )
    scene.handles = {handle.path: handle}
    handle._result_ref = weakref.ref(scene)
    native = SimpleNamespace(
        cfg=SimpleNamespace(fpath="source.usdc", body_scale=(scale,) * 3)
    )
    simulation = SimpleNamespace(
        spawn_result=scene,
        physics_backend="default",
        get_articulation=lambda _uid: native,
    )
    bindings = SimpleNamespace(
        rigid_objects=(),
        articulations=[SimpleNamespace(entity_id="cabinet", simulation_uid=desc.name)],
        links=[
            SimpleNamespace(
                entity_id="base_frame",
                articulation_id="cabinet",
                native_link_name="base",
            )
        ],
        rigidized_articulations=(),
    )
    monkeypatch.setattr(
        drawer_geometry, "collision_meshes", lambda _cfg: {("base", "shell"): shell}
    )
    return simulation, bindings, desc, collider


def test_decomposed_source_keeps_native_parts_and_scales_once(monkeypatch):
    from dexsim.kit.meshproc import convex_cache

    simulation, bindings, desc, collider = _fixture(monkeypatch)
    vertices = collider.vertices.copy()
    transform = collider.local_transform.copy()
    calls = []

    def decompose(points, triangles, **kwargs):
        calls.append((points.copy(), triangles.copy(), kwargs))
        return "existing-cache.obj"

    parts = []
    for x in (0.15, 0.65):
        part = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
        part.apply_translation((x, 0.1, 0.2))
        parts.append((np.asarray(part.vertices), np.asarray(part.faces)))
    monkeypatch.setattr(convex_cache, "convex_decomposition_to_obj", decompose)
    monkeypatch.setattr(convex_cache, "load_obj_as_convex_parts", lambda _path: parts)
    result = scene_geometry.capture_scene_vertices(simulation, bindings)
    assert result.unavailable is None
    assert len(result.components) == 2
    assert all(
        component.owner_id == "cabinet" and component.pose_entity_id == "base_frame"
        for component in result.components
    )
    assert (
        result.components[0].vertices[0, :, 0].max()
        < result.components[1].vertices[0, :, 0].min()
    )
    np.testing.assert_allclose(calls[0][0], vertices + transform[:3, 3], atol=1e-7)
    assert calls[0][2] == {
        "algorithm": collider.decomp_algorithm,
        "max_convex_hulls": collider.decomp_max_hulls,
        "max_vert_per_convex_hull": collider.decomp_vertex_limit,
    }
    np.testing.assert_allclose(
        result.components[0].vertices[0].numpy().min(0),
        parts[0][0].min(0) * desc.body_scale,
    )
    np.testing.assert_array_equal(collider.vertices, vertices)
    np.testing.assert_array_equal(collider.local_transform, transform)


def test_convex_hull_uses_whole_collider_semantics(monkeypatch):
    simulation, bindings, desc, collider = _fixture(monkeypatch, "convexHull")
    collider.local_transform[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    result = scene_geometry.capture_scene_vertices(simulation, bindings)
    assert result.unavailable is None
    assert len(result.components) == 1
    points = result.components[0].vertices[0].numpy()
    expected = (
        collider.vertices @ collider.local_transform[:3, :3].T
        + collider.local_transform[:3, 3]
    ) * desc.body_scale
    np.testing.assert_allclose(points.min(0), expected.min(0), atol=1e-7)
    np.testing.assert_allclose(points.max(0), expected.max(0), atol=1e-7)


@pytest.mark.parametrize(
    "damage",
    [
        "identity",
        "ambiguous",
        "scale",
        "source",
        "provenance",
        "no_scene",
        "invalid",
        "file",
        "primitive",
        "unsupported",
        "empty",
        "link",
    ],
)
def test_descriptor_failure_never_falls_back_to_raw_source(monkeypatch, damage):
    simulation, bindings, desc, collider = _fixture(monkeypatch, "convexHull")

    def forbidden_raw_reader(_cfg):
        raise AssertionError("Invalid retained geometry must not use raw fallback")

    monkeypatch.setattr(drawer_geometry, "collision_meshes", forbidden_raw_reader)
    if damage == "identity":
        desc.name = "unrelated"
    elif damage == "ambiguous":
        handle = next(iter(simulation.spawn_result.handles.values()))
        simulation.spawn_result.handles["duplicate"] = handle
    elif damage == "scale":
        desc.body_scale[0] *= 2
    elif damage == "source":
        desc.usd.file_path = "different.usdc"
    elif damage == "provenance":
        desc.usd = None
    elif damage == "no_scene":
        simulation.spawn_result = None
    elif damage == "invalid":
        collider.vertices[0, 0] = np.nan
    elif damage == "file":
        collider.vertices = collider.triangles = None
        collider.file_path = "other.obj"
    elif damage == "primitive":
        from dexsim.spawn.descs import GeometryType

        collider.geometry_type = GeometryType.CUBE
    elif damage == "unsupported":
        from dexsim.spawn.descs import CollisionApproximation

        collider.approximation = CollisionApproximation.NONE
    elif damage == "empty":
        desc.links[0].collisions.clear()
    else:
        bindings.links.clear()
    result = scene_geometry.capture_scene_vertices(simulation, bindings)
    assert result.unavailable is not None
    assert result.components == ()


def test_each_submitted_convex_part_uses_hull_not_source_is_convex(monkeypatch):
    from dexsim.kit.meshproc import convex_cache

    simulation, bindings, _desc, collider = _fixture(monkeypatch)
    part = (collider.vertices + collider.local_transform[:3, 3], collider.triangles)
    monkeypatch.setattr(
        convex_cache, "convex_decomposition_to_obj", lambda *_a, **_kw: "part.obj"
    )
    monkeypatch.setattr(convex_cache, "load_obj_as_convex_parts", lambda _path: [part])
    result = scene_geometry.capture_scene_vertices(simulation, bindings)
    assert result.unavailable is None
    assert len(result.components) == 1


def test_disabled_retained_shape_is_conservatively_included(monkeypatch):
    simulation, bindings, _desc, collider = _fixture(monkeypatch, "convexHull")
    collider.enable_collision = False
    result = scene_geometry.capture_scene_vertices(simulation, bindings)
    assert result.unavailable is None
    assert len(result.components) == 1


def test_missing_convex_parts_is_unavailable(monkeypatch):
    from dexsim.kit.meshproc import convex_cache

    simulation, bindings, _desc, _collider = _fixture(monkeypatch)
    monkeypatch.setattr(
        convex_cache, "convex_decomposition_to_obj", lambda *_a, **_kw: "empty.obj"
    )
    monkeypatch.setattr(convex_cache, "load_obj_as_convex_parts", lambda _path: [])
    result = scene_geometry.capture_scene_vertices(simulation, bindings)
    assert result.unavailable is not None
    assert result.components == ()


def test_nondefault_backend_retains_previous_fail_closed_route(monkeypatch):
    simulation, bindings, _desc, _collider = _fixture(monkeypatch, "convexHull")
    simulation.physics_backend = "newton"
    simulation.spawn_result.backend = "newton"
    result = scene_geometry.capture_scene_vertices(simulation, bindings)
    assert "nonconvex_unavailable" in result.unavailable
