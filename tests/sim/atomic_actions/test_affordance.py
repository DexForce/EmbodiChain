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

"""Tests for atomic_actions.affordance (Affordance, AntipodalAffordance, InteractionPoints)."""

from __future__ import annotations

import pytest
import torch

from embodichain.lab.sim.atomic_actions.affordance import (
    Affordance,
    AntipodalAffordance,
    InteractionPoints,
)


class TestAffordance:
    def test_default_object_label_is_empty(self):
        assert Affordance().object_label == ""

    def test_custom_config_get_set(self):
        aff = Affordance()
        aff.set_custom_config("k", 1)
        assert aff.get_custom_config("k") == 1
        assert aff.get_custom_config("missing") is None
        assert aff.get_custom_config("missing", "d") == "d"

    def test_base_get_batch_size_is_one(self):
        assert Affordance().get_batch_size() == 1


class TestAntipodalAffordance:
    def test_stores_mesh_fields_directly(self):
        v = torch.randn(8, 3)
        t = torch.randint(0, 8, (5, 3))
        aff = AntipodalAffordance(mesh_vertices=v, mesh_triangles=t)
        assert aff.mesh_vertices is v
        assert aff.mesh_triangles is t

    def test_no_geometry_alias_field(self):
        # The redesign removes the shared-geometry-dict footgun.
        aff = AntipodalAffordance()
        assert not hasattr(aff, "geometry") or getattr(aff, "geometry", None) is None


class TestInteractionPoints:
    def test_default_points_shape(self):
        assert InteractionPoints().points.shape == (1, 3)

    def test_get_batch_size_matches_points(self):
        ip = InteractionPoints(points=torch.randn(4, 3))
        assert ip.get_batch_size() == 4

    def test_get_points_by_type_returns_subset(self):
        pts = torch.tensor([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
        ip = InteractionPoints(points=pts, point_types=["push", "poke", "push"])
        result = ip.get_points_by_type("push")
        assert result is not None and result.shape == (2, 3)
        assert torch.equal(result[0], pts[0])
        assert torch.equal(result[1], pts[2])

    def test_get_points_by_type_returns_none_for_missing(self):
        ip = InteractionPoints(points=torch.zeros(2, 3), point_types=["push", "push"])
        assert ip.get_points_by_type("poke") is None

    def test_approach_direction_inverts_normal(self):
        normals = torch.tensor([[0.0, 0, 1.0], [1.0, 0, 0]])
        ip = InteractionPoints(points=torch.zeros(2, 3), normals=normals)
        assert torch.equal(ip.get_approach_direction(0), torch.tensor([0.0, 0, -1.0]))
        assert torch.equal(ip.get_approach_direction(1), torch.tensor([-1.0, 0, 0]))

    def test_approach_direction_default_when_no_normals(self):
        ip = InteractionPoints(points=torch.zeros(1, 3))
        assert torch.equal(ip.get_approach_direction(0), torch.tensor([0.0, 0, 1.0]))
