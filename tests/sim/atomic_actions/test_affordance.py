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
    AssembleAffordance,
    InteractionPoints,
    PressButtonAffordance,
    PullPushAffordance,
    TurnAffordance,
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

    def test_articulation_link_resolves_mesh_and_live_pose(self):
        vertices = torch.randn(8, 3)
        triangles = torch.randint(0, 8, (5, 3))
        link_pose = torch.eye(4).repeat(2, 1, 1)
        articulation = Mock()
        articulation.get_link_vert_face.return_value = (vertices, triangles)
        articulation.get_link_pose.return_value = link_pose

        aff = AntipodalAffordance(
            articulation=articulation,
            link_name="knob",
        )

        articulation.get_link_vert_face.assert_called_once_with("knob")
        assert aff.is_articulation
        assert aff.mesh_vertices is vertices
        assert aff.mesh_triangles is triangles
        assert aff.get_articulation_link_pose() is link_pose
        articulation.get_link_pose.assert_called_once_with("knob", to_matrix=True)

    def test_articulation_and_link_name_must_be_provided_together(self):
        with pytest.raises(ValueError, match="must be provided together"):
            AntipodalAffordance(articulation=Mock())
        with pytest.raises(ValueError, match="must be provided together"):
            AntipodalAffordance(link_name="knob")

    def test_articulation_geometry_is_mutually_exclusive_with_mesh_input(self):
        articulation = Mock()
        with pytest.raises(ValueError, match="either articulation"):
            AntipodalAffordance(
                articulation=articulation,
                link_name="knob",
                mesh_vertices=torch.zeros(1, 3),
            )

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

        approach_direction = generator.get_valid_grasp_poses.call_args.kwargs[
            "approach_direction"
        ]
        assert approach_direction.dtype == torch.float32
        assert approach_direction.device == generator.device

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


class TestTurnAffordance:
    def test_builds_grasp_pose_from_mesh_center_and_axes(self):
        vertices = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [2.0, 2.0, 2.0],
                [1.0, 1.0, 1.0],
            ]
        )
        triangles = torch.tensor([[0, 1, 2]])
        link_pose = torch.eye(4).repeat(2, 1, 1)
        link_pose[:, :3, :3] = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        link_pose[:, :3, 3] = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        articulation = Mock()
        articulation.get_link_vert_face.return_value = (vertices, triangles)
        articulation.get_link_pose.return_value = link_pose
        affordance = TurnAffordance(
            articulation=articulation,
            link_name="knob",
            turn_axis=torch.tensor([1.0, 0.0, 0.0]),
        )

        grasp_pose = affordance.get_grasp_pose()

        assert torch.allclose(
            grasp_pose[:, :3, 3],
            torch.matmul(link_pose[:, :3, :3], vertices.mean(dim=0))
            + link_pose[:, :3, 3],
        )
        assert torch.allclose(
            grasp_pose[:, :3, 1],
            torch.tensor([0.0, 0.0, 1.0]).expand(2, -1),
        )
        assert torch.allclose(
            grasp_pose[:, :3, 2],
            torch.tensor([0.0, 1.0, 0.0]).expand(2, -1),
        )
        articulation.get_link_vert_face.assert_called_once_with("knob")
        articulation.get_link_pose.assert_called_once_with("knob", to_matrix=True)

    def test_rejects_turn_axis_parallel_to_world_up(self):
        articulation = Mock()
        articulation.get_link_vert_face.return_value = (
            torch.ones(3, 3),
            torch.tensor([[0, 1, 2]]),
        )
        articulation.get_link_pose.return_value = torch.eye(4).unsqueeze(0)
        affordance = TurnAffordance(
            articulation=articulation,
            link_name="knob",
            turn_axis=torch.tensor([0.0, 0.0, 1.0]),
        )

        with pytest.raises(ValueError, match="parallel"):
            affordance.get_grasp_pose()


class TestPullPushAffordance:
    def test_uses_articulation_antipodal_sampling_with_batched_directions(self):
        vertices = torch.tensor(
            [
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        triangles = torch.tensor([[0, 1, 2]])
        link_pose = torch.eye(4).repeat(2, 1, 1)
        link_pose[:, :3, 3] = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        articulation = Mock()
        articulation.get_link_vert_face.return_value = (vertices, triangles)
        articulation.get_link_pose.return_value = link_pose
        affordance = PullPushAffordance(
            articulation=articulation,
            link_name="handle",
            translation_axis=torch.tensor([0.0, -1.0, 0.0]),
        )
        generator = Mock()
        generator.device = torch.device("cpu")
        first_grasp = torch.eye(4)
        first_grasp[:3, 3] = torch.tensor([1.0, 2.0, 3.0])
        second_grasp = torch.eye(4)
        second_grasp[:3, 3] = torch.tensor([4.0, 5.0, 6.0])
        generator.get_grasp_poses.side_effect = (
            (True, first_grasp, 0.03),
            (True, second_grasp, 0.04),
        )
        affordance._generator = generator
        approach_directions = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])

        success, grasp_poses, open_lengths = affordance.get_best_grasp_poses(
            link_pose,
            approach_direction=approach_directions,
        )

        assert isinstance(affordance, AntipodalAffordance)
        assert affordance.is_articulation
        assert success.tolist() == [True, True]
        assert torch.allclose(grasp_poses, torch.stack([first_grasp, second_grasp]))
        assert torch.allclose(open_lengths, torch.tensor([0.03, 0.04]))
        articulation.get_link_vert_face.assert_called_once_with("handle")
        assert torch.equal(
            generator.get_grasp_poses.call_args_list[0].args[1],
            approach_directions[0],
        )
        assert torch.equal(
            generator.get_grasp_poses.call_args_list[1].args[1],
            approach_directions[1],
        )

    def test_requires_articulation_backed_antipodal_geometry(self):
        with pytest.raises(ValueError, match="requires articulation and link_name"):
            PullPushAffordance(
                mesh_vertices=torch.zeros(3, 3),
                mesh_triangles=torch.tensor([[0, 1, 2]]),
            )

    @pytest.mark.parametrize(
        "translation_axis",
        (
            torch.zeros(3),
            torch.tensor([float("nan"), 0.0, 0.0]),
            torch.zeros(2),
        ),
    )
    def test_rejects_invalid_translation_axis(self, translation_axis):
        articulation = Mock()
        articulation.get_link_vert_face.return_value = (
            torch.ones(3, 3),
            torch.tensor([[0, 1, 2]]),
        )

        with pytest.raises(ValueError, match="translation_axis"):
            PullPushAffordance(
                articulation=articulation,
                link_name="handle",
                translation_axis=translation_axis,
            )


class TestPressButtonAffordance:
    def test_builds_press_pose_on_upstream_mesh_surface(self):
        vertices = torch.tensor(
            [
                [-2.0, -1.0, -1.0],
                [-2.0, 1.0, 1.0],
                [2.0, -1.0, 1.0],
                [2.0, 1.0, -1.0],
            ]
        )
        triangles = torch.tensor([[0, 1, 2], [1, 2, 3]])
        link_pose = torch.eye(4).repeat(2, 1, 1)
        link_pose[:, :3, 3] = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        articulation = Mock()
        articulation.get_link_vert_face.return_value = (vertices, triangles)
        articulation.get_link_pose.return_value = link_pose
        affordance = PressButtonAffordance(
            articulation=articulation,
            link_name="button",
            press_axis=torch.tensor([1.0, 0.0, 0.0]),
        )

        press_pose = affordance.get_press_pose()

        expected_surface = link_pose[:, :3, 3] + torch.tensor([-2.0, 0.0, 0.0])
        assert torch.allclose(press_pose[:, :3, 3], expected_surface)
        assert torch.allclose(
            press_pose[:, :3, 2],
            torch.tensor([1.0, 0.0, 0.0]).expand(2, -1),
        )
        articulation.get_link_vert_face.assert_called_once_with("button")
        articulation.get_link_pose.assert_called_once_with("button", to_matrix=True)

    def test_rejects_zero_press_axis(self):
        articulation = Mock()
        articulation.get_link_vert_face.return_value = (
            torch.ones(3, 3),
            torch.tensor([[0, 1, 2]]),
        )

        with pytest.raises(ValueError, match="press_axis must be non-zero"):
            PressButtonAffordance(
                articulation=articulation,
                link_name="button",
                press_axis=torch.zeros(3),
            )


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


class TestAssembleAffordance:
    def _rel_pose(self) -> torch.Tensor:
        pose = torch.eye(4)
        pose[:3, :3] = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
        )
        pose[2, 3] = 0.1
        return pose

    def test_default_fields(self):
        aff = AssembleAffordance()
        assert aff.base_object_label == ""
        assert aff.assemble_object_label == ""
        assert aff.base_object_entity is None
        assert aff.assemble_object_entity is None
        assert torch.equal(aff.assemble_to_base_pose, torch.eye(4))

    def test_get_assemble_object_pose_single_base_pose(self):
        aff = AssembleAffordance(assemble_to_base_pose=self._rel_pose())
        base_pose = torch.eye(4)
        base_pose[:3, 3] = torch.tensor([1.0, 2.0, 3.0])
        result = aff.get_assemble_object_pose(base_pose)
        assert result.shape == (1, 4, 4)
        assert torch.allclose(result[0], base_pose @ self._rel_pose())

    def test_get_assemble_object_pose_broadcasts_across_envs(self):
        n_envs = 3
        aff = AssembleAffordance(assemble_to_base_pose=self._rel_pose())
        base_pose = torch.eye(4).unsqueeze(0).repeat(n_envs, 1, 1)
        base_pose[:, 0, 3] = torch.arange(n_envs, dtype=torch.float32)
        result = aff.get_assemble_object_pose(base_pose)
        assert result.shape == (n_envs, 4, 4)
        expected = torch.bmm(
            base_pose, self._rel_pose().unsqueeze(0).repeat(n_envs, 1, 1)
        )
        assert torch.allclose(result, expected)

    def test_get_assemble_object_pose_broadcasts_batched_relative_pose(self):
        n_envs = 2
        rel = self._rel_pose().unsqueeze(0).repeat(n_envs, 1, 1)
        aff = AssembleAffordance(assemble_to_base_pose=rel)
        base_pose = torch.eye(4).unsqueeze(0).repeat(n_envs, 1, 1)
        base_pose[:, 2, 3] = 0.5
        result = aff.get_assemble_object_pose(base_pose)
        assert result.shape == (n_envs, 4, 4)
        assert torch.allclose(result, torch.bmm(base_pose, rel))
