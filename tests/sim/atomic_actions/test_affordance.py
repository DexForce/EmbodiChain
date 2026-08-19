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

from unittest.mock import Mock

import pytest
import torch

from embodichain.lab.sim.atomic_actions.affordance import (
    Affordance,
    AntipodalAffordance,
    AssembleAffordance,
    InteractionPoints,
    PressAffordance,
    SlideAffordance,
    TwistAffordance,
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

    def test_valid_grasp_poses_applies_object_aware_cost_callback(self):
        aff = AntipodalAffordance()
        generator = Mock()
        generator.device = torch.device("cpu")
        object_pose = torch.eye(4)
        object_pose[0, 3] = 0.25
        grasp_poses = torch.eye(4).repeat(2, 1, 1)
        costs = torch.tensor([0.2, 0.4])

        def get_valid_grasp_poses(**kwargs):
            adjusted = kwargs["pose_cost_fn"](grasp_poses, costs)
            return True, grasp_poses, 0.0, adjusted

        generator.get_valid_grasp_poses.side_effect = get_valid_grasp_poses
        aff._generator = generator

        results = aff.get_valid_grasp_poses(
            object_pose.unsqueeze(0),
            grasp_cost_fn=lambda obj, _grasps, current: current + obj[0, 3],
        )

        assert torch.allclose(results[0][1], torch.tensor([0.45, 0.65]))


class TestTwistAffordance:
    def test_requires_explicit_grasp_position_and_axis_origin(self):
        with pytest.raises(TypeError, match="grasp_position"):
            TwistAffordance()  # type: ignore[call-arg]

    @pytest.mark.parametrize(
        "twist_axis",
        (
            torch.tensor([1.0, 0.0, 1.0]),
            torch.tensor([0.0, 0.0, 1.0]),
            torch.tensor([0.0, 0.0, -1.0]),
        ),
    )
    def test_builds_right_handed_orthonormal_grasp_frame(self, twist_axis):
        link_pose = torch.eye(4).repeat(2, 1, 1)
        link_pose[:, :3, 3] = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        affordance = TwistAffordance(
            grasp_position=(0.25, -0.5, 0.75),
            axis_origin=(0.1, 0.2, 0.3),
            twist_axis=twist_axis,
        )

        grasp_pose = affordance.get_grasp_pose(link_pose)
        rotation = grasp_pose[:, :3, :3]

        assert torch.allclose(
            grasp_pose[:, :3, 3],
            link_pose[:, :3, 3] + torch.tensor([0.25, -0.5, 0.75]).expand(2, -1),
        )
        assert torch.allclose(
            torch.matmul(rotation.transpose(1, 2), rotation),
            torch.eye(3).expand(2, -1, -1),
            atol=1.0e-6,
        )
        assert torch.allclose(torch.linalg.det(rotation), torch.ones(2), atol=1.0e-6)


class TestSlideAffordance:
    def test_uses_local_antipodal_mesh_with_batched_directions(self):
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
        affordance = SlideAffordance(
            mesh_vertices=vertices,
            mesh_triangles=triangles,
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
        assert success.tolist() == [True, True]
        assert torch.allclose(grasp_poses, torch.stack([first_grasp, second_grasp]))
        assert torch.allclose(open_lengths, torch.tensor([0.03, 0.04]))
        assert torch.equal(
            generator.get_grasp_poses.call_args_list[0].args[1],
            approach_directions[0],
        )
        assert torch.equal(
            generator.get_grasp_poses.call_args_list[1].args[1],
            approach_directions[1],
        )

    def test_requires_local_antipodal_geometry(self):
        with pytest.raises(TypeError, match="mesh_vertices"):
            SlideAffordance()

    @pytest.mark.parametrize(
        "translation_axis",
        (
            torch.zeros(3),
            torch.tensor([float("nan"), 0.0, 0.0]),
            torch.zeros(2),
        ),
    )
    def test_rejects_invalid_translation_axis(self, translation_axis):
        with pytest.raises(ValueError, match="translation_axis"):
            SlideAffordance(
                mesh_vertices=torch.ones(3, 3),
                mesh_triangles=torch.tensor([[0, 1, 2]]),
                translation_axis=translation_axis,
            )


class TestPressAffordance:
    def test_requires_explicit_surface_press_position(self):
        with pytest.raises(TypeError, match="press_position"):
            PressAffordance()  # type: ignore[call-arg]

    @pytest.mark.parametrize(
        "press_axis",
        (
            torch.tensor([1.0, 0.0, 1.0]),
            torch.tensor([0.0, 0.0, 1.0]),
            torch.tensor([0.0, 0.0, -1.0]),
        ),
    )
    def test_builds_right_handed_orthonormal_press_frame(self, press_axis):
        link_pose = torch.eye(4).repeat(2, 1, 1)
        link_pose[:, :3, 3] = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        affordance = PressAffordance(
            press_axis=press_axis,
            press_position=(0.25, -0.5, 0.75),
        )

        press_pose = affordance.get_press_pose(link_pose)
        rotation = press_pose[:, :3, :3]

        assert torch.allclose(
            press_pose[:, :3, 3],
            link_pose[:, :3, 3] + torch.tensor([0.25, -0.5, 0.75]).expand(2, -1),
        )
        assert torch.allclose(
            torch.matmul(rotation.transpose(1, 2), rotation),
            torch.eye(3).expand(2, -1, -1),
            atol=1.0e-6,
        )
        assert torch.allclose(torch.linalg.det(rotation), torch.ones(2), atol=1.0e-6)

    def test_rejects_zero_press_axis(self):
        with pytest.raises(ValueError, match="press_axis must be non-zero"):
            PressAffordance(
                press_axis=torch.zeros(3),
                press_position=(0.0, 0.0, 0.0),
            )

    def test_uses_configured_press_position(self):
        object_pose = torch.eye(4).repeat(2, 1, 1)
        object_pose[:, :3, 3] = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        affordance = PressAffordance(
            press_axis=torch.tensor([1.0, 0.0, 0.0]),
            press_position=(0.25, -0.5, 0.75),
        )

        press_pose = affordance.get_press_pose(object_pose)

        assert torch.allclose(
            press_pose[:, :3, 3],
            object_pose[:, :3, 3] + torch.tensor([0.25, -0.5, 0.75]).expand(2, -1),
        )

    def test_per_call_press_position_overrides_affordance_position(self):
        affordance = PressAffordance(
            press_axis=torch.tensor([1.0, 0.0, 0.0]),
            press_position=(1.0, 1.0, 1.0),
        )

        press_pose = affordance.get_press_pose(
            torch.eye(4).unsqueeze(0),
            press_position=(0.1, 0.2, 0.3),
        )

        assert torch.allclose(
            press_pose[0, :3, 3],
            torch.tensor([0.1, 0.2, 0.3]),
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
        num_envs = 3
        aff = AssembleAffordance(assemble_to_base_pose=self._rel_pose())
        base_pose = torch.eye(4).unsqueeze(0).repeat(num_envs, 1, 1)
        base_pose[:, 0, 3] = torch.arange(num_envs, dtype=torch.float32)
        result = aff.get_assemble_object_pose(base_pose)
        assert result.shape == (num_envs, 4, 4)
        expected = torch.bmm(
            base_pose, self._rel_pose().unsqueeze(0).repeat(num_envs, 1, 1)
        )
        assert torch.allclose(result, expected)

    def test_get_assemble_object_pose_broadcasts_batched_relative_pose(self):
        num_envs = 2
        rel = self._rel_pose().unsqueeze(0).repeat(num_envs, 1, 1)
        aff = AssembleAffordance(assemble_to_base_pose=rel)
        base_pose = torch.eye(4).unsqueeze(0).repeat(num_envs, 1, 1)
        base_pose[:, 2, 3] = 0.5
        result = aff.get_assemble_object_pose(base_pose)
        assert result.shape == (num_envs, 4, 4)
        assert torch.allclose(result, torch.bmm(base_pose, rel))

    def test_get_assemble_object_pose_rejects_relative_batch_mismatch(self):
        aff = AssembleAffordance(assemble_to_base_pose=torch.eye(4).repeat(3, 1, 1))
        base_pose = torch.eye(4).repeat(2, 1, 1)

        with pytest.raises(ValueError, match="batch size must match"):
            aff.get_assemble_object_pose(base_pose)

    def test_get_assemble_object_pose_rejects_invalid_base_shape(self):
        aff = AssembleAffordance()

        with pytest.raises(ValueError, match="base_pose must have shape"):
            aff.get_assemble_object_pose(torch.eye(4).repeat(2, 1, 1, 1))
