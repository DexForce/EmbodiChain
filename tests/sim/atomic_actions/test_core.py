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

"""Tests for atomic action core module (Affordance, InteractionPoints, ObjectSemantics, ActionCfg)."""

from __future__ import annotations

import pytest
import torch

from embodichain.lab.sim.atomic_actions.core import (
    ActionCfg,
    Affordance,
    InteractionPoints,
    ObjectSemantics,
)

# ---------------------------------------------------------------------------
# Affordance
# ---------------------------------------------------------------------------


class TestAffordance:
    """Tests for the Affordance base dataclass."""

    def test_default_values(self):
        aff = Affordance()
        assert aff.object_label == ""
        assert aff.geometry == {}
        assert aff.custom_config == {}

    def test_mesh_vertices_returns_tensor(self):
        vertices = torch.randn(10, 3)
        aff = Affordance(geometry={"mesh_vertices": vertices})
        assert torch.equal(aff.mesh_vertices, vertices)

    def test_mesh_vertices_returns_none_when_missing(self):
        aff = Affordance()
        assert aff.mesh_vertices is None

    def test_mesh_vertices_raises_on_wrong_type(self):
        aff = Affordance(geometry={"mesh_vertices": [1, 2, 3]})
        with pytest.raises(TypeError, match="must be a torch.Tensor"):
            _ = aff.mesh_vertices

    def test_mesh_triangles_returns_tensor(self):
        triangles = torch.randint(0, 10, (5, 3))
        aff = Affordance(geometry={"mesh_triangles": triangles})
        assert torch.equal(aff.mesh_triangles, triangles)

    def test_mesh_triangles_returns_none_when_missing(self):
        aff = Affordance()
        assert aff.mesh_triangles is None

    def test_mesh_triangles_raises_on_wrong_type(self):
        aff = Affordance(geometry={"mesh_triangles": "bad"})
        with pytest.raises(TypeError, match="must be a torch.Tensor"):
            _ = aff.mesh_triangles

    def test_custom_config_get_set(self):
        aff = Affordance()
        aff.set_custom_config("key_a", 42)
        assert aff.get_custom_config("key_a") == 42
        assert aff.get_custom_config("missing") is None
        assert aff.get_custom_config("missing", "default") == "default"

    def test_get_batch_size_returns_one(self):
        # Base Affordance always returns 1
        assert Affordance().get_batch_size() == 1


# ---------------------------------------------------------------------------
# InteractionPoints
# ---------------------------------------------------------------------------


class TestInteractionPoints:
    """Tests for InteractionPoints affordance."""

    def test_default_points_shape(self):
        ip = InteractionPoints()
        assert ip.points.shape == (1, 3)

    def test_get_batch_size_matches_points(self):
        points = torch.randn(5, 3)
        ip = InteractionPoints(points=points)
        assert ip.get_batch_size() == 5

    def test_get_points_by_type_returns_matching_subset(self):
        points = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        ip = InteractionPoints(points=points, point_types=["push", "poke", "push"])
        result = ip.get_points_by_type("push")
        assert result is not None
        assert result.shape == (2, 3)
        assert torch.equal(result[0], points[0])
        assert torch.equal(result[1], points[2])

    def test_get_points_by_type_returns_none_for_missing_type(self):
        ip = InteractionPoints(points=torch.zeros(2, 3), point_types=["push", "push"])
        assert ip.get_points_by_type("poke") is None

    def test_get_approach_direction_from_normals(self):
        normals = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        ip = InteractionPoints(points=torch.zeros(2, 3), normals=normals)
        # Approach is opposite of normal
        assert torch.equal(ip.get_approach_direction(0), torch.tensor([0.0, 0.0, -1.0]))
        assert torch.equal(ip.get_approach_direction(1), torch.tensor([-1.0, 0.0, 0.0]))

    def test_get_approach_direction_default_without_normals(self):
        ip = InteractionPoints(points=torch.zeros(1, 3))
        direction = ip.get_approach_direction(0)
        assert torch.equal(direction, torch.tensor([0.0, 0.0, 1.0]))


# ---------------------------------------------------------------------------
# ObjectSemantics
# ---------------------------------------------------------------------------


class TestObjectSemantics:
    """Tests for ObjectSemantics dataclass."""

    def test_post_init_binds_label_and_geometry(self):
        geometry = {"bounding_box": [0.1, 0.2, 0.3]}
        aff = Affordance()
        sem = ObjectSemantics(
            affordance=aff,
            geometry=geometry,
            label="mug",
        )
        assert sem.affordance.object_label == "mug"
        assert sem.affordance.geometry is geometry

    def test_default_optional_fields(self):
        sem = ObjectSemantics(
            affordance=Affordance(),
            geometry={},
        )
        assert sem.label == "none"
        assert sem.properties == {}
        assert sem.entity is None


# ---------------------------------------------------------------------------
# ActionCfg
# ---------------------------------------------------------------------------


class TestActionCfg:
    """Tests for ActionCfg defaults."""

    def test_default_values(self):
        cfg = ActionCfg()
        assert cfg.name == "default"
        assert cfg.control_part == "arm"
        assert cfg.interpolation_type == "linear"
        assert cfg.velocity_limit is None
        assert cfg.acceleration_limit is None
