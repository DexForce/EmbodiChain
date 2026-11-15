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

"""Exact native mesh order and input ownership for shared geometric consumers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from embodichain.lab.sim.scene.rigid_mesh import RigidMeshSnapshot, capture_rigid_mesh


@pytest.mark.parametrize("semantic", [False, True])
def test_native_capture_owns_unprocessed_batched_vertices_and_triangles(semantic):
    vertices = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 0.0]],
            [[4.0, 0.0, 0.0], [6.0, 0.0, 0.0], [4.0, 3.0, 0.0], [4.0, 0.0, 0.0]],
        ]
    )
    triangles = torch.tensor(
        [[[3, 1, 2], [0, 2, 1]], [[0, 2, 1], [3, 1, 2]]], dtype=torch.int32
    )
    calls = []

    def get_vertices(*, scale):
        assert scale is True
        calls.append("vertices")
        return vertices

    def get_triangles():
        calls.append("triangles")
        return triangles

    def pose(**kwargs):
        raise AssertionError("Geometry capture must not read poses")

    native = SimpleNamespace(
        get_vertices=get_vertices, get_triangles=get_triangles, get_local_pose=pose
    )
    if semantic:
        native.get_semantic_mesh = lambda: (get_vertices(scale=True), get_triangles())

        def no_visual_mesh(**kwargs):
            raise AssertionError("Gaussian appearance has no triangle mesh")

        native.get_vertices = native.get_triangles = no_visual_mesh
    simulation = SimpleNamespace(
        get_rigid_object=lambda uid: native if uid == "object" else None
    )
    snapshot = capture_rigid_mesh(simulation, "object", batch_size=2)
    assert calls == ["vertices", "triangles"]
    assert snapshot.batch_size == 2
    assert torch.equal(snapshot.vertices, vertices)
    assert torch.equal(snapshot.triangles, triangles)
    assert (
        snapshot.vertices.shape[1] == 4
    )  # Duplicate vertices and winding remain authored.
    vertices.fill_(99)
    triangles.zero_()
    assert snapshot.vertices[0, 1, 0] == 2
    assert snapshot.triangles[0, 0].tolist() == [3, 1, 2]
    detached = snapshot.snapshot()
    snapshot.vertices.zero_()
    assert detached.vertices[1, 1, 0] == 6


def test_mesh_snapshot_revalidates_mutable_inputs_and_never_fabricates_topology():
    vertices = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    triangles = torch.tensor([[[0, 1, 2]]])
    snapshot = RigidMeshSnapshot("object", vertices, triangles)
    snapshot.triangles[0, 0, 0] = 3
    with pytest.raises(ValueError, match="index"):
        snapshot.snapshot()
    with pytest.raises(ValueError, match="triangles"):
        RigidMeshSnapshot("object", vertices, torch.empty(1, 0, 3, dtype=torch.long))
    with pytest.raises(ValueError, match="finite"):
        RigidMeshSnapshot("object", vertices * float("nan"), triangles)
    simulation = SimpleNamespace(get_rigid_object=lambda uid: None)
    with pytest.raises(ValueError, match="exact bound"):
        capture_rigid_mesh(simulation, "missing", batch_size=1)


@pytest.mark.parametrize("mismatch", ["expected_batch", "triangle_batch"])
def test_capture_rejects_batch_mismatch_before_copy(monkeypatch, mismatch):
    vertices = torch.zeros(2, 3, 3)
    triangle_batch = 1 if mismatch == "triangle_batch" else 2
    triangles = torch.zeros(triangle_batch, 1, 3, dtype=torch.int64)
    native = SimpleNamespace(
        get_vertices=lambda **kwargs: vertices, get_triangles=lambda: triangles
    )
    simulation = SimpleNamespace(get_rigid_object=lambda uid: native)

    def unexpected_copy(*args, **kwargs):
        raise AssertionError("Invalid mesh metadata must reject before copying")

    monkeypatch.setattr(torch.Tensor, "cpu", unexpected_copy)
    monkeypatch.setattr(torch.Tensor, "clone", unexpected_copy)
    expected_batch = 1 if mismatch == "expected_batch" else 2
    message = "expected observed batch" if mismatch == "expected_batch" else "triangles"
    with pytest.raises(ValueError, match=message):
        capture_rigid_mesh(simulation, "object", batch_size=expected_batch)


@pytest.mark.parametrize("damage", ["vertex_shape", "vertex_dtype", "triangle_dtype"])
def test_snapshot_rejects_invalid_metadata_before_copy(monkeypatch, damage):
    vertices = torch.zeros(1, 3, 2 if damage == "vertex_shape" else 3)
    triangles = torch.zeros(1, 1, 3, dtype=torch.int64)
    if damage == "vertex_dtype":
        vertices = vertices.to(torch.int64)
    if damage == "triangle_dtype":
        triangles = triangles.to(torch.float32)

    def unexpected_copy(*args, **kwargs):
        raise AssertionError("Invalid mesh metadata must reject before copying")

    monkeypatch.setattr(torch.Tensor, "cpu", unexpected_copy)
    monkeypatch.setattr(torch.Tensor, "clone", unexpected_copy)
    message = "triangles" if damage == "triangle_dtype" else "coordinates"
    with pytest.raises(ValueError, match=message):
        RigidMeshSnapshot("object", vertices, triangles)
