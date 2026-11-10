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

import pytest
from dexsim.engine import BoxGeometry
from dexsim.spawn.descs import CollisionDesc, GeometryDesc
from dexsim.types import RigidBodyShape

from embodichain.lab.sim.objects import RigidObject


class _FakePhysicalBody:
    def __init__(self, geometries):
        self.geometries = geometries

    def get_shape_count(self):
        return len(self.geometries)

    def get_shape_geometry(self, shape_idx):
        return self.geometries[shape_idx]

    def get_shape_name(self, shape_idx):
        return f"collision_{shape_idx}"


class _UnavailableGeometryBody(_FakePhysicalBody):
    def get_shape_geometry(self, shape_idx):
        raise RuntimeError("SDF dispatch is unavailable")


class _FakePhysicalEntity:
    def __init__(self, geometries):
        self.physical_body = _FakePhysicalBody(geometries)

    def get_physical_body(self):
        return self.physical_body


class _DescriptorEntity:
    """New-style SpawnedRigidBody shape-query fixture."""

    def __init__(self, collision):
        self.object_desc = SimpleNamespace(
            collisions=[collision],
            body_scale=(1.0, 1.0, 1.0),
        )
        self._physics_binding = None

    def native(self):
        return None


def _box_geometry(half_extents):
    geometry = BoxGeometry()
    geometry.half_extents = half_extents
    return geometry


def test_get_collision_shapes_snapshots_physical_box_geometry():
    rigid_object = RigidObject.__new__(RigidObject)
    rigid_object.uid = "fixture"
    rigid_object._entities = [_FakePhysicalEntity([_box_geometry([0.1, 0.2, 0.3])])]

    shapes = rigid_object.get_collision_shapes()

    assert len(shapes) == 1
    assert shapes[0].name == "collision_0"
    assert shapes[0].shape_type == RigidBodyShape.BOX
    assert shapes[0].half_extents.tolist() == pytest.approx([0.1, 0.2, 0.3])


def test_get_collision_shapes_rejects_batched_topology_mismatch():
    rigid_object = RigidObject.__new__(RigidObject)
    rigid_object.uid = "fixture"
    rigid_object._entities = [
        _FakePhysicalEntity([_box_geometry([0.1, 0.2, 0.3])]),
        _FakePhysicalEntity(
            [
                _box_geometry([0.1, 0.2, 0.3]),
                _box_geometry([0.4, 0.5, 0.6]),
            ]
        ),
    ]

    with pytest.raises(ValueError, match="different collision-shape topology"):
        rigid_object.get_collision_shapes()


def test_get_collision_shapes_reports_unavailable_sdf_geometry():
    rigid_object = RigidObject.__new__(RigidObject)
    rigid_object.uid = "sdf_object"
    entity = _FakePhysicalEntity([])
    entity.physical_body = _UnavailableGeometryBody([object()])
    rigid_object._entities = [entity]

    with pytest.raises(RuntimeError, match="canonical collision mesh"):
        rigid_object.get_collision_shapes()


def test_get_collision_shapes_reads_new_spawned_body_descriptor():
    collision = CollisionDesc.from_geometry(GeometryDesc.cube((0.2, 0.4, 0.6)))
    rigid_object = RigidObject.__new__(RigidObject)
    rigid_object.uid = "descriptor_fixture"
    rigid_object._entities = [_DescriptorEntity(collision)]

    shapes = rigid_object.get_collision_shapes()

    assert len(shapes) == 1
    assert shapes[0].shape_type == RigidBodyShape.BOX
    assert shapes[0].half_extents.tolist() == pytest.approx([0.1, 0.2, 0.3])


def test_articulation_link_collision_reads_native_physical_shape_not_render_mesh():
    from embodichain.lab.sim.objects.backends import (
        collision_shapes_from_articulation_link,
    )

    geometry = _box_geometry([0.1, 0.2, 0.3])
    native = SimpleNamespace(
        get_physical_body=lambda name: (
            _FakePhysicalBody([geometry]) if name == "arm" else None
        )
    )
    entity = SimpleNamespace(native=lambda: native)
    shapes = collision_shapes_from_articulation_link(entity, "arm", object_uid="robot")
    assert shapes[0].shape_type == RigidBodyShape.BOX
    assert shapes[0].half_extents.tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert (
        collision_shapes_from_articulation_link(entity, "empty", object_uid="robot")
        == []
    )


def test_articulation_link_collision_reads_newton_named_physical_mesh():
    import numpy as np
    from embodichain.lab.sim.objects.backends import (
        collision_shapes_from_articulation_link,
    )

    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    triangles = np.array([[0, 1, 2]])
    entity = SimpleNamespace(
        native=lambda: None,
        _physics_binding=SimpleNamespace(
            get_collision_mesh=lambda name: (vertices, triangles)
        ),
    )
    shapes = collision_shapes_from_articulation_link(entity, "arm", object_uid="robot")
    vertices[:] = 99
    assert shapes[0].vertices[1, 0] == 1
    assert shapes[0].triangles.tolist() == [[0, 1, 2]]


def test_newton_link_keeps_convex_pieces_and_triangle_types():
    import torch
    from types import SimpleNamespace
    import numpy as np
    import newton
    from embodichain.lab.sim.objects.backends.collision import (
        collision_shapes_from_articulation_link,
    )

    mesh = SimpleNamespace(
        vertices=np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32
        ),
        indices=np.array([0, 1, 2, 0, 3, 1, 0, 2, 3, 1, 3, 2]),
    )
    model = SimpleNamespace(
        shape_flags=np.array(
            [
                int(newton.ShapeFlags.COLLIDE_SHAPES),
                int(newton.ShapeFlags.COLLIDE_SHAPES),
            ]
        ),
        shape_type=np.array(
            [int(newton.GeoType.CONVEX_MESH), int(newton.GeoType.MESH)]
        ),
        shape_scale=np.array([[2, 3, 4], [1, 1, 1]], dtype=np.float32),
        shape_transform=np.array(
            [[0, 0, 0, 0, 0, 0, 1], [1, 2, 3, 0, 0, 0, 1]], dtype=np.float32
        ),
        shape_source=[mesh, mesh],
    )
    binding = SimpleNamespace(
        _runtime=SimpleNamespace(model=model),
        _links=[SimpleNamespace(name="arm", shape_ids=(0, 1))],
    )
    shapes = collision_shapes_from_articulation_link(
        SimpleNamespace(_physics_binding=binding), "arm", object_uid="robot"
    )
    assert len(shapes) == 2
    assert shapes[0].shape_type.name == "CONVEX" and shapes[1].shape_type.name == "MESH"
    torch.testing.assert_close(shapes[0].vertices[1], torch.tensor([2.0, 0.0, 0.0]))
    torch.testing.assert_close(
        shapes[1].local_pose[:3, 3], torch.tensor([1.0, 2.0, 3.0])
    )

    model.shape_flags[1] = 0
    enabled = collision_shapes_from_articulation_link(
        SimpleNamespace(_physics_binding=binding), "arm", object_uid="robot"
    )
    assert len(enabled) == 1 and enabled[0].shape_type.name == "CONVEX"
