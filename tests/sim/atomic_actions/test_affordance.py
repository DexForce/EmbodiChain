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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from embodichain.lab.sim.atomic_actions.affordance import (
    Affordance,
    AntipodalAffordance,
    AssembleAffordance,
    AxisAlignAffordance,
    InteractionPoints,
    OpenDoorAffordance,
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
    @staticmethod
    def _long_box_mesh() -> tuple[torch.Tensor, torch.Tensor]:
        vertices = torch.tensor(
            [
                [-0.1, -0.1, -1.0],
                [0.1, -0.1, -1.0],
                [0.1, 0.1, -1.0],
                [-0.1, 0.1, -1.0],
                [-0.1, -0.1, 1.0],
                [0.1, -0.1, 1.0],
                [0.1, 0.1, 1.0],
                [-0.1, 0.1, 1.0],
            ],
            dtype=torch.float32,
        )
        triangles = torch.tensor(
            [
                [0, 1, 2],
                [0, 2, 3],
                [4, 6, 5],
                [4, 7, 6],
                [0, 4, 5],
                [0, 5, 1],
                [1, 5, 6],
                [1, 6, 2],
                [2, 6, 7],
                [2, 7, 3],
                [3, 7, 4],
                [3, 4, 0],
            ],
            dtype=torch.long,
        )
        return vertices, triangles

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

    def test_owns_no_generator_configuration_or_runtime(self):
        affordance = AntipodalAffordance()
        assert not hasattr(affordance, "generator_cfg")
        assert not hasattr(affordance, "gripper_collision_cfg")
        assert not hasattr(affordance, "force_reannotate")
        assert not hasattr(affordance, "_generator")

    def test_requires_mesh_fields_together(self):
        with pytest.raises(ValueError, match="provided together"):
            AntipodalAffordance(mesh_vertices=torch.zeros(3, 3))

    def test_surface_svd_uses_at_most_1000_points_in_current_pose(self):
        vertices, triangles = self._long_box_mesh()
        affordance = AntipodalAffordance(
            mesh_vertices=vertices,
            mesh_triangles=triangles,
        )
        first_points = affordance.sample_surface_points(max_points=1000)
        second_points = affordance.sample_surface_points(max_points=1000)
        poses = torch.eye(4).repeat(2, 1, 1)
        poses[1, :3, :3] = torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
        )

        axes = affordance.get_object_longest_axis(poses, max_points=1000)

        assert first_points.shape == (1000, 3)
        assert torch.equal(first_points, second_points)
        assert torch.abs(axes[0, 2]) > 0.99
        assert torch.abs(axes[1, 0]) > 0.99


class TestAxisAlignAffordance:
    def test_extends_antipodal_affordance_with_owned_internal_axis(self):
        internal_axis = torch.tensor([1.0, 0.0, 0.0])

        affordance = AxisAlignAffordance(internal_axis=internal_axis)
        internal_axis[0] = 0.0

        assert isinstance(affordance, AntipodalAffordance)
        assert torch.equal(affordance.internal_axis, torch.tensor([1.0, 0.0, 0.0]))

    @pytest.mark.parametrize(
        "internal_axis",
        (
            torch.zeros(3),
            torch.tensor([float("nan"), 0.0, 0.0]),
            torch.zeros(2),
        ),
    )
    def test_rejects_invalid_internal_axis(self, internal_axis):
        with pytest.raises(ValueError, match="internal_axis"):
            AxisAlignAffordance(internal_axis=internal_axis)


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
    def test_uses_only_local_antipodal_mesh_and_translation_axis(self):
        vertices = torch.tensor(
            [
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        triangles = torch.tensor([[0, 1, 2]])
        affordance = SlideAffordance(
            mesh_vertices=vertices,
            mesh_triangles=triangles,
            translation_axis=torch.tensor([0.0, -1.0, 0.0]),
        )
        assert isinstance(affordance, AntipodalAffordance)
        assert affordance.mesh_vertices is vertices
        assert affordance.mesh_triangles is triangles
        assert torch.equal(
            affordance.translation_axis,
            torch.tensor([0.0, -1.0, 0.0]),
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


class TestOpenDoorAffordance:
    @staticmethod
    def _articulation_with_parent_hinge() -> Mock:
        fixed_joint = SimpleNamespace(
            name="door_to_door_handle_fixed",
            parent_link_name="door",
            child_link_name="door_handle",
            joint_type=SimpleNamespace(name="FIXED"),
            axis=torch.zeros(3).numpy(),
            origin_pose=torch.eye(4).numpy(),
            lower_limit=0.0,
            upper_limit=0.0,
        )
        hinge_joint = SimpleNamespace(
            name="door_hinge",
            parent_link_name="body",
            child_link_name="door",
            joint_type=SimpleNamespace(name="REVOLUTE"),
            axis=torch.tensor([0.0, 0.0, 1.0]).numpy(),
            origin_pose=torch.eye(4).numpy(),
            lower_limit=0.0,
            upper_limit=2.0,
        )
        entity = Mock()
        entity.get_joint_names.return_value = [fixed_joint.name, hinge_joint.name]
        entity.get_joint_info.side_effect = {
            fixed_joint.name: fixed_joint,
            hinge_joint.name: hinge_joint,
        }.get
        articulation = Mock()
        articulation.link_names = ["body", "door", "door_handle"]
        articulation._entities = [entity]
        body_pose = torch.eye(4).unsqueeze(0)
        handle_pose = torch.eye(4).unsqueeze(0)
        handle_pose[:, 0, 3] = 1.0
        poses = {
            "body": body_pose,
            "door_handle": handle_pose,
        }
        articulation.get_link_pose.side_effect = lambda link_name, **_: poses[link_name]
        articulation.get_link_vert_face.return_value = (
            torch.tensor(
                [
                    [0.9, -0.1, 0.0],
                    [1.1, -0.1, 0.0],
                    [1.0, 0.1, 0.0],
                ]
            ),
            torch.tensor([[0, 1, 2]]),
        )
        return articulation

    def test_resolves_first_parent_revolute_joint_from_handle_link(self):
        articulation = self._articulation_with_parent_hinge()

        affordance = OpenDoorAffordance.from_articulation(
            articulation,
            "door_handle",
        )

        assert affordance.joint_name == "door_hinge"
        assert torch.allclose(
            affordance.rotation_axis,
            torch.tensor([0.0, 0.0, 1.0]),
        )
        assert affordance.axis_origin == pytest.approx((-1.0, 0.0, 0.0))
        assert affordance.joint_limits == pytest.approx((0.0, 2.0))
        assert affordance.opening_direction == 1

    def test_accepts_affordance_owned_negative_opening_direction(self):
        articulation = self._articulation_with_parent_hinge()

        affordance = OpenDoorAffordance.from_articulation(
            articulation,
            "door_handle",
            opening_direction=-1,
        )

        assert affordance.opening_direction == -1

    def test_rejects_handle_without_parent_revolute_joint(self):
        articulation = self._articulation_with_parent_hinge()
        articulation._entities[0].get_joint_names.return_value = [
            "door_to_door_handle_fixed"
        ]

        with pytest.raises(ValueError, match="No parent revolute joint"):
            OpenDoorAffordance.from_articulation(articulation, "door_handle")


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
