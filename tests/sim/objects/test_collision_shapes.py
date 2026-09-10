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

import pytest
from dexsim.engine import BoxGeometry
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
