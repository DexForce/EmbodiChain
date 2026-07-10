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
from unittest.mock import Mock

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
        assert not hasattr(aff, "geometry")

    def test_failed_valid_grasp_poses_are_batched_with_inf_costs(self):
        aff = AntipodalAffordance()
        generator = Mock()
        generator.device = torch.device("cpu")
        generator.get_valid_grasp_poses.return_value = (
            False,
            torch.eye(4),
            0.0,
            torch.zeros(1),
        )
        aff._generator = generator

        results = aff.get_valid_grasp_poses(torch.eye(4).unsqueeze(0))

        grasp_poses, costs = results[0]
        assert grasp_poses.shape == (1, 4, 4)
        assert costs.shape == (1,)
        assert torch.isinf(costs).all()

    def test_valid_grasp_poses_casts_approach_direction_to_generator_device(self):
        aff = AntipodalAffordance()
        generator = Mock()
        generator.device = torch.device("cpu")
        generator.get_valid_grasp_poses.return_value = (
            True,
            torch.eye(4).unsqueeze(0),
            0.0,
            torch.zeros(1),
        )
        aff._generator = generator

        aff.get_valid_grasp_poses(
            torch.eye(4).unsqueeze(0),
            approach_direction=torch.tensor([0, 0, -1], dtype=torch.int64),
        )

        _, approach_direction = generator.get_valid_grasp_poses.call_args.args
        assert approach_direction.dtype == torch.float32
        assert approach_direction.device == generator.device

    def test_valid_grasp_poses_forwards_approach_alignment_angle(self):
        aff = AntipodalAffordance()
        generator = Mock()
        generator.device = torch.device("cpu")
        generator.get_valid_grasp_poses.return_value = (
            True,
            torch.eye(4).unsqueeze(0),
            0.0,
            torch.zeros(1),
        )
        aff._generator = generator

        aff.get_valid_grasp_poses(
            torch.eye(4).unsqueeze(0),
            max_approach_alignment_angle=0.1,
        )

        assert generator.get_valid_grasp_poses.call_args.kwargs == {
            "max_approach_alignment_angle": 0.1
        }

    def test_best_grasp_poses_casts_approach_direction_to_generator_device(self):
        aff = AntipodalAffordance()
        generator = Mock()
        generator.device = torch.device("cpu")
        generator.get_grasp_poses.return_value = (True, torch.eye(4), 0.05)
        aff._generator = generator

        aff.get_best_grasp_poses(
            torch.eye(4).unsqueeze(0),
            approach_direction=torch.tensor([0, 0, -1], dtype=torch.int64),
        )

        _, approach_direction = generator.get_grasp_poses.call_args.args
        assert approach_direction.dtype == torch.float32
        assert approach_direction.device == generator.device

    def test_upright_bottle_side_grasp_bias_prefers_mid_body_side_grasp(self):
        mesh_vertices = torch.tensor(
            [
                [-0.05, -1.0, -0.05],
                [0.05, -1.0, 0.05],
                [0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        aff = AntipodalAffordance(mesh_vertices=mesh_vertices)
        aff.set_custom_config(
            "grasp_pose_bias",
            {
                "mode": "upright_bottle_side_grasp",
                "preferred_height_fraction": [0.35, 0.75],
                "prefer_side_grasp": True,
            },
        )
        low_pose = torch.eye(4)
        low_pose[:3, 3] = torch.tensor([0.0, -0.95, 0.0])
        mid_pose = torch.eye(4)
        mid_pose[:3, 3] = torch.tensor([0.0, 0.0, 0.0])
        generator = Mock()
        generator.device = torch.device("cpu")
        generator.get_valid_grasp_poses.return_value = (
            True,
            torch.stack([low_pose, mid_pose]),
            torch.tensor([0.03, 0.04]),
            torch.tensor([0.0, 0.8]),
        )
        aff._generator = generator

        is_success, grasp_xpos, open_length = aff.get_best_grasp_poses(
            torch.eye(4).unsqueeze(0)
        )

        assert bool(is_success.item()) is True
        assert torch.allclose(grasp_xpos[0, :3, 3], torch.tensor([0.0, 0.0, 0.0]))
        assert open_length.item() == pytest.approx(0.04)
        generator.get_grasp_poses.assert_not_called()


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
