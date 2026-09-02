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

from embodichain.lab.sim.objects import ArticulationJointKinematics

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
from embodichain.lab.sim.atomic_actions.core import ObjectSemantics

POINT_CLOUD_CENTER = torch.tensor([2.0, -3.0, 4.0])
PRISMATIC_JOINT_AXIS = torch.tensor([2.0, 2.0, 1.0])
REVOLUTE_JOINT_AXIS = torch.tensor([-1.0, 2.0, 2.0])
PRISMATIC_JOINT_AXIS_UNIT = PRISMATIC_JOINT_AXIS / torch.linalg.vector_norm(
    PRISMATIC_JOINT_AXIS
)
REVOLUTE_JOINT_AXIS_UNIT = REVOLUTE_JOINT_AXIS / torch.linalg.vector_norm(
    REVOLUTE_JOINT_AXIS
)
REVOLUTE_AXIS_ORIGIN = torch.tensor([0.75, -0.5, 0.25])
TARGET_POINT_OFFSETS = torch.tensor(
    [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ]
)
TARGET_LINK_POINT_CLOUD_KEY = "target_link_point_cloud"
ARTICULATION_POINT_CLOUD_KEY = "articulation_point_cloud"
TARGET_LINK_PRISMATIC_JOINT_AXIS_KEY = "target_link_prismatic_joint_axis"
TARGET_LINK_REVOLUTE_JOINT_AXIS_KEY = "target_link_revolute_joint_axis"
TARGET_LINK_REVOLUTE_AXIS_ORIGIN_KEY = "target_link_revolute_axis_origin"


def _axis_geometry(neighbor_offset: tuple[float, float, float]) -> dict[str, object]:
    """Build target-local clouds with one neighbor and one distant outlier."""
    target_points = POINT_CLOUD_CENTER + TARGET_POINT_OFFSETS
    neighbor = POINT_CLOUD_CENTER + torch.tensor(neighbor_offset)
    distant_outlier = POINT_CLOUD_CENTER + torch.tensor([8.0, 8.0, 8.0])
    return {
        TARGET_LINK_POINT_CLOUD_KEY: target_points.clone(),
        ARTICULATION_POINT_CLOUD_KEY: torch.cat(
            (
                target_points,
                neighbor.unsqueeze(0),
                distant_outlier.unsqueeze(0),
            ),
            dim=0,
        ),
        TARGET_LINK_PRISMATIC_JOINT_AXIS_KEY: PRISMATIC_JOINT_AXIS.clone(),
        TARGET_LINK_REVOLUTE_JOINT_AXIS_KEY: REVOLUTE_JOINT_AXIS.clone(),
        TARGET_LINK_REVOLUTE_AXIS_ORIGIN_KEY: REVOLUTE_AXIS_ORIGIN.clone(),
    }


def _joint_axis_metadata(kind: str) -> tuple[str, torch.Tensor]:
    """Return the geometry key and raw parent-joint axis for an affordance."""
    if kind in ("slide", "press"):
        return TARGET_LINK_PRISMATIC_JOINT_AXIS_KEY, PRISMATIC_JOINT_AXIS
    if kind == "twist":
        return TARGET_LINK_REVOLUTE_JOINT_AXIS_KEY, REVOLUTE_JOINT_AXIS
    raise ValueError(f"Unsupported affordance kind: {kind!r}.")


def _axis_affordance(
    kind: str,
    *,
    fallback_axis: torch.Tensor,
    press_position: tuple[float, float, float] | None = (0.0, 0.0, 0.0),
) -> tuple[Affordance, str]:
    """Construct one axis-bearing affordance with a legacy fallback axis."""
    if kind == "slide":
        return (
            SlideAffordance(
                mesh_vertices=torch.tensor(
                    [
                        [-0.1, -0.1, 0.0],
                        [0.1, -0.1, 0.0],
                        [0.0, 0.1, 0.0],
                    ]
                ),
                mesh_triangles=torch.tensor([[0, 1, 2]]),
                translation_axis=fallback_axis,
            ),
            "translation_axis",
        )
    if kind == "press":
        return (
            PressAffordance(
                press_axis=fallback_axis,
                press_position=press_position,
            ),
            "press_axis",
        )
    if kind == "twist":
        return (
            TwistAffordance(
                grasp_position=(0.0, 0.0, 0.0),
                axis_origin=(0.0, 0.0, 0.0),
                twist_axis=fallback_axis,
            ),
            "twist_axis",
        )
    raise ValueError(f"Unsupported affordance kind: {kind!r}.")


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


class TestArticulationGeometryAxisInference:
    @pytest.mark.parametrize(
        ("kind", "axis_field"),
        (
            ("slide", "translation_axis"),
            ("press", "press_axis"),
            ("twist", "twist_axis"),
        ),
    )
    @pytest.mark.parametrize("direction", (1.0, -1.0), ids=("positive", "negative"))
    def test_uses_parent_joint_axis_and_neighborhood_only_selects_sign(
        self,
        kind: str,
        axis_field: str,
        direction: float,
    ) -> None:
        _, raw_joint_axis = _joint_axis_metadata(kind)
        normalized_joint_axis = raw_joint_axis / torch.linalg.vector_norm(
            raw_joint_axis
        )
        neighbor_offset = normalized_joint_axis * (1.5 * direction)
        affordance, actual_axis_field = _axis_affordance(
            kind,
            fallback_axis=torch.tensor([1.0, 1.0, 1.0]),
        )

        ObjectSemantics(
            affordance=affordance,
            geometry=_axis_geometry(tuple(float(value) for value in neighbor_offset)),
            entity_id=f"{kind}-target",
        )

        assert actual_axis_field == axis_field
        actual_axis = getattr(affordance, axis_field)
        assert torch.allclose(
            actual_axis,
            normalized_joint_axis * direction,
            atol=1.0e-6,
        )
        assert torch.linalg.vector_norm(actual_axis).item() == pytest.approx(1.0)
        assert torch.count_nonzero(actual_axis).item() == 3

    @pytest.mark.parametrize("kind", ("slide", "press", "twist"))
    def test_empty_geometry_preserves_explicit_fallback_axis(self, kind: str) -> None:
        fallback_axis = torch.tensor([-1.0, 0.0, 0.0])
        affordance, axis_field = _axis_affordance(
            kind,
            fallback_axis=fallback_axis,
        )

        ObjectSemantics(
            affordance=affordance,
            geometry={},
            entity_id=f"rigid-{kind}-target",
        )

        assert torch.equal(getattr(affordance, axis_field), fallback_axis)

    @pytest.mark.parametrize(
        ("kind", "unrelated_axis_key", "unrelated_axis"),
        (
            (
                "slide",
                TARGET_LINK_REVOLUTE_JOINT_AXIS_KEY,
                REVOLUTE_JOINT_AXIS,
            ),
            (
                "press",
                TARGET_LINK_REVOLUTE_JOINT_AXIS_KEY,
                REVOLUTE_JOINT_AXIS,
            ),
            (
                "twist",
                TARGET_LINK_PRISMATIC_JOINT_AXIS_KEY,
                PRISMATIC_JOINT_AXIS,
            ),
        ),
    )
    def test_unrelated_joint_axis_metadata_preserves_fallback_axis(
        self,
        kind: str,
        unrelated_axis_key: str,
        unrelated_axis: torch.Tensor,
    ) -> None:
        fallback_axis = torch.tensor([-1.0, 0.5, 0.25])
        affordance, axis_field = _axis_affordance(
            kind,
            fallback_axis=fallback_axis,
        )

        ObjectSemantics(
            affordance=affordance,
            geometry={unrelated_axis_key: unrelated_axis.clone()},
            entity_id=f"unrelated-{kind}-joint-axis-target",
        )

        assert torch.equal(getattr(affordance, axis_field), fallback_axis)

    def test_twist_origin_metadata_alone_preserves_axis_fallback(self) -> None:
        fallback_axis = torch.tensor([-1.0, 0.5, 0.25])
        affordance = TwistAffordance(
            grasp_position=(0.0, 0.0, 0.0),
            axis_origin=(9.0, 8.0, 7.0),
            twist_axis=fallback_axis,
        )

        ObjectSemantics(
            affordance=affordance,
            geometry={
                TARGET_LINK_REVOLUTE_AXIS_ORIGIN_KEY: REVOLUTE_AXIS_ORIGIN.clone()
            },
            entity_id="twist-origin-only-target",
        )

        assert torch.equal(affordance.twist_axis, fallback_axis)
        assert affordance.axis_origin == pytest.approx(
            tuple(float(value) for value in REVOLUTE_AXIS_ORIGIN)
        )

    @pytest.mark.parametrize(
        "present_fields",
        (
            ("target",),
            ("articulation",),
            ("axis",),
            ("target", "articulation"),
            ("target", "axis"),
            ("articulation", "axis"),
        ),
    )
    @pytest.mark.parametrize("kind", ("slide", "press", "twist"))
    def test_incomplete_joint_axis_inference_metadata_is_rejected(
        self,
        kind: str,
        present_fields: tuple[str, ...],
    ) -> None:
        axis_key, raw_joint_axis = _joint_axis_metadata(kind)
        complete_geometry = _axis_geometry((1.0, 1.0, 0.5))
        geometry: dict[str, object] = {}
        if "target" in present_fields:
            geometry[TARGET_LINK_POINT_CLOUD_KEY] = complete_geometry[
                TARGET_LINK_POINT_CLOUD_KEY
            ]
        if "articulation" in present_fields:
            geometry[ARTICULATION_POINT_CLOUD_KEY] = complete_geometry[
                ARTICULATION_POINT_CLOUD_KEY
            ]
        if "axis" in present_fields:
            geometry[axis_key] = raw_joint_axis.clone()
        affordance, _ = _axis_affordance(
            kind,
            fallback_axis=torch.tensor([1.0, 0.0, 0.0]),
        )

        with pytest.raises(ValueError, match=axis_key):
            ObjectSemantics(
                affordance=affordance,
                geometry=geometry,
                entity_id=f"incomplete-{kind}-geometry-target",
            )

    @pytest.mark.parametrize("kind", ("slide", "press", "twist"))
    @pytest.mark.parametrize(
        ("joint_axis", "exception_type"),
        (
            ((1.0, 0.0, 0.0), TypeError),
            (torch.tensor([1, 0, 0]), ValueError),
            (torch.tensor([1.0, 0.0]), ValueError),
            (torch.tensor([float("nan"), 0.0, 0.0]), ValueError),
            (torch.zeros(3), ValueError),
        ),
    )
    def test_invalid_parent_joint_axis_metadata_is_rejected(
        self,
        kind: str,
        joint_axis: object,
        exception_type: type[Exception],
    ) -> None:
        axis_key, _ = _joint_axis_metadata(kind)
        geometry = _axis_geometry((1.0, 1.0, 0.5))
        geometry[axis_key] = joint_axis
        affordance, _ = _axis_affordance(
            kind,
            fallback_axis=torch.tensor([1.0, 0.0, 0.0]),
        )

        with pytest.raises(exception_type, match=axis_key):
            ObjectSemantics(
                affordance=affordance,
                geometry=geometry,
                entity_id=f"invalid-{kind}-joint-axis-target",
            )

    def test_degenerate_target_point_cloud_is_rejected(self) -> None:
        geometry = _axis_geometry((0.5, -1.0, -1.0))
        geometry[TARGET_LINK_POINT_CLOUD_KEY] = POINT_CLOUD_CENTER.repeat(3, 1)
        geometry[ARTICULATION_POINT_CLOUD_KEY] = POINT_CLOUD_CENTER.unsqueeze(0)
        affordance, _ = _axis_affordance(
            "twist",
            fallback_axis=torch.tensor([1.0, 0.0, 0.0]),
        )

        with pytest.raises(ValueError, match="degenerate target-link"):
            ObjectSemantics(
                affordance=affordance,
                geometry=geometry,
                entity_id="degenerate-geometry-target",
            )

    @pytest.mark.parametrize(
        ("kind", "perpendicular_offset"),
        (
            ("slide", (1.0, -1.0, 0.0)),
            ("press", (1.0, -1.0, 0.0)),
            ("twist", (2.0, 1.0, 0.0)),
        ),
    )
    def test_neighborhood_offset_perpendicular_to_joint_axis_is_ambiguous(
        self,
        kind: str,
        perpendicular_offset: tuple[float, float, float],
    ) -> None:
        affordance, _ = _axis_affordance(
            kind,
            fallback_axis=torch.tensor([1.0, 0.0, 0.0]),
        )

        with pytest.raises(ValueError, match="direction is ambiguous"):
            ObjectSemantics(
                affordance=affordance,
                geometry=_axis_geometry(perpendicular_offset),
                entity_id=f"ambiguous-{kind}-geometry-target",
            )

    def test_press_without_position_uses_outer_surface_opposite_inferred_axis(
        self,
    ) -> None:
        affordance = PressAffordance(
            press_axis=torch.tensor([1.0, 0.0, 0.0]),
            press_position=None,
        )

        ObjectSemantics(
            affordance=affordance,
            geometry=_axis_geometry(
                tuple(float(value) for value in -1.5 * PRISMATIC_JOINT_AXIS_UNIT)
            ),
            entity_id="automatic-press-target",
        )

        assert torch.allclose(
            affordance.press_axis,
            -PRISMATIC_JOINT_AXIS_UNIT,
            atol=1.0e-6,
        )
        expected_surface_center = POINT_CLOUD_CENTER + torch.tensor([0.5, 0.5, 0.0])
        assert affordance.press_position == pytest.approx(
            tuple(float(value) for value in expected_surface_center)
        )

    def test_twist_geometry_uses_revolute_joint_axis_origin(self) -> None:
        affordance = TwistAffordance(
            grasp_position=(0.0, 0.0, 0.0),
            axis_origin=(9.0, 8.0, 7.0),
            twist_axis=torch.tensor([1.0, 0.0, 0.0]),
        )

        ObjectSemantics(
            affordance=affordance,
            geometry=_axis_geometry(
                tuple(float(value) for value in -1.5 * REVOLUTE_JOINT_AXIS_UNIT)
            ),
            entity_id="automatic-twist-target",
        )

        assert torch.allclose(
            affordance.twist_axis,
            -REVOLUTE_JOINT_AXIS_UNIT,
            atol=1.0e-6,
        )
        assert affordance.axis_origin == pytest.approx(
            tuple(float(value) for value in REVOLUTE_AXIS_ORIGIN)
        )
        assert not torch.allclose(
            torch.tensor(affordance.require_axis_origin()),
            POINT_CLOUD_CENTER,
        )

    def test_twist_complete_clouds_without_joint_origin_preserve_fallback(
        self,
    ) -> None:
        geometry = _axis_geometry(
            tuple(float(value) for value in -1.5 * REVOLUTE_JOINT_AXIS_UNIT)
        )
        geometry.pop(TARGET_LINK_REVOLUTE_AXIS_ORIGIN_KEY)
        affordance = TwistAffordance(
            grasp_position=(0.0, 0.0, 0.0),
            axis_origin=(9.0, 8.0, 7.0),
            twist_axis=torch.tensor([1.0, 0.0, 0.0]),
        )

        ObjectSemantics(
            affordance=affordance,
            geometry=geometry,
            entity_id="automatic-twist-target-with-origin-fallback",
        )

        assert torch.allclose(
            affordance.twist_axis,
            -REVOLUTE_JOINT_AXIS_UNIT,
            atol=1.0e-6,
        )
        assert affordance.axis_origin == pytest.approx((9.0, 8.0, 7.0))

    def test_twist_complete_clouds_without_joint_origin_do_not_use_centroid(
        self,
    ) -> None:
        geometry = _axis_geometry(
            tuple(float(value) for value in -1.5 * REVOLUTE_JOINT_AXIS_UNIT)
        )
        geometry.pop(TARGET_LINK_REVOLUTE_AXIS_ORIGIN_KEY)
        affordance = TwistAffordance(
            grasp_position=(0.0, 0.0, 0.0),
            twist_axis=torch.tensor([1.0, 0.0, 0.0]),
        )

        ObjectSemantics(
            affordance=affordance,
            geometry=geometry,
            entity_id="automatic-twist-target-without-joint-origin",
        )

        assert torch.allclose(
            affordance.twist_axis,
            -REVOLUTE_JOINT_AXIS_UNIT,
            atol=1.0e-6,
        )
        assert affordance.axis_origin is None
        with pytest.raises(
            ValueError,
            match="target_link_revolute_axis_origin",
        ):
            affordance.require_axis_origin()

    @pytest.mark.parametrize(
        ("axis_origin", "exception_type"),
        (
            ((0.0, 0.0, 0.0), TypeError),
            (torch.tensor([0, 0, 0]), ValueError),
            (torch.tensor([0.0, 0.0]), ValueError),
            (torch.tensor([0.0, float("nan"), 0.0]), ValueError),
        ),
    )
    def test_twist_rejects_invalid_revolute_joint_axis_origin_metadata(
        self,
        axis_origin: object,
        exception_type: type[Exception],
    ) -> None:
        geometry = _axis_geometry((0.25, -0.125, -1.5))
        geometry[TARGET_LINK_REVOLUTE_AXIS_ORIGIN_KEY] = axis_origin
        affordance = TwistAffordance(
            grasp_position=(0.0, 0.0, 0.0),
            axis_origin=(9.0, 8.0, 7.0),
        )

        with pytest.raises(
            exception_type,
            match="target_link_revolute_axis_origin",
        ):
            ObjectSemantics(
                affordance=affordance,
                geometry=geometry,
                entity_id="twist-target-with-invalid-joint-origin",
            )

    def test_twist_without_point_cloud_geometry_preserves_axis_origin_fallback(
        self,
    ) -> None:
        affordance = TwistAffordance(
            grasp_position=(0.0, 0.0, 0.0),
            axis_origin=(9.0, 8.0, 7.0),
            twist_axis=torch.tensor([1.0, 0.0, 0.0]),
        )

        ObjectSemantics(
            affordance=affordance,
            geometry={},
            entity_id="rigid-twist-target",
        )

        assert torch.equal(
            affordance.twist_axis,
            torch.tensor([1.0, 0.0, 0.0]),
        )
        assert affordance.axis_origin == pytest.approx((9.0, 8.0, 7.0))


class TestTwistAffordance:
    def test_requires_explicit_grasp_position(self):
        with pytest.raises(TypeError, match="grasp_position"):
            TwistAffordance()  # type: ignore[call-arg]

    def test_requires_axis_origin_before_use_without_point_cloud_geometry(self):
        affordance = TwistAffordance(grasp_position=(0.0, 0.0, 0.0))

        with pytest.raises(ValueError, match="provided explicitly or resolved"):
            affordance.require_axis_origin()

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
    def _joint(
        name: str,
        joint_type: str,
        parent_link_name: str,
        child_link_name: str,
    ) -> ArticulationJointKinematics:
        return ArticulationJointKinematics(
            name=name,
            joint_type=joint_type,
            parent_link_name=parent_link_name,
            child_link_name=child_link_name,
            axis=(
                torch.tensor([0.0, 0.0, 1.0])
                if joint_type == "revolute"
                else torch.zeros(3)
            ),
            origin_pose=torch.eye(4),
            joint_limits=(0.0, 2.0) if joint_type != "fixed" else (0.0, 0.0),
        )

    @staticmethod
    def _articulation_with_parent_hinge() -> Mock:
        fixed_joint = TestOpenDoorAffordance._joint(
            "door_to_door_handle_fixed",
            "fixed",
            "door",
            "door_handle",
        )
        hinge_joint = TestOpenDoorAffordance._joint(
            "door_hinge",
            "revolute",
            "body",
            "door",
        )
        articulation = Mock(
            spec=[
                "get_parent_joint_chain",
                "get_link_pose",
                "get_link_vert_face",
            ]
        )
        articulation.get_parent_joint_chain.return_value = (
            fixed_joint,
            hinge_joint,
        )
        articulation.fixed_joint = fixed_joint
        articulation.hinge_joint = hinge_joint
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
        articulation.get_parent_joint_chain.assert_called_once_with("door_handle")

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
        articulation.get_parent_joint_chain.return_value = (articulation.fixed_joint,)

        with pytest.raises(ValueError, match="No active parent joint"):
            OpenDoorAffordance.from_articulation(articulation, "door_handle")

    def test_requires_explicit_hinge_across_prismatic_ancestor(self):
        articulation = self._articulation_with_parent_hinge()
        prismatic = self._joint(
            "handle_slide",
            "prismatic",
            "door",
            "door_handle",
        )
        articulation.get_parent_joint_chain.return_value = (
            prismatic,
            articulation.hinge_joint,
        )

        with pytest.raises(ValueError, match="Ambiguous active ancestors"):
            OpenDoorAffordance.from_articulation(articulation, "door_handle")

        affordance = OpenDoorAffordance.from_articulation(
            articulation,
            "door_handle",
            hinge_joint_name="door_hinge",
        )
        assert affordance.joint_name == "door_hinge"

    def test_requires_explicit_hinge_when_handle_has_revolute_latch(self):
        articulation = self._articulation_with_parent_hinge()
        latch = self._joint(
            "handle_latch",
            "revolute",
            "door",
            "door_handle",
        )
        articulation.get_parent_joint_chain.return_value = (
            latch,
            articulation.hinge_joint,
        )

        with pytest.raises(ValueError, match="Ambiguous active ancestors"):
            OpenDoorAffordance.from_articulation(articulation, "door_handle")

    def test_rejects_explicit_non_revolute_hinge(self):
        articulation = self._articulation_with_parent_hinge()
        prismatic = self._joint(
            "handle_slide",
            "prismatic",
            "door",
            "door_handle",
        )
        articulation.get_parent_joint_chain.return_value = (
            prismatic,
            articulation.hinge_joint,
        )

        with pytest.raises(ValueError, match="must be revolute"):
            OpenDoorAffordance.from_articulation(
                articulation,
                "door_handle",
                hinge_joint_name="handle_slide",
            )


class TestPressAffordance:
    def test_requires_surface_position_before_pose_without_point_cloud_geometry(self):
        affordance = PressAffordance()

        with pytest.raises(ValueError, match="provided explicitly or resolved"):
            affordance.get_press_pose(torch.eye(4).unsqueeze(0))

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
