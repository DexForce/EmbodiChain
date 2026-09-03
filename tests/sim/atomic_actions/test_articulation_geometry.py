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
import torch

import embodichain.lab.sim.atomic_actions.articulation_geometry as articulation_geometry_module
from embodichain.lab.sim.atomic_actions.articulation_geometry import (
    ArticulationAffordanceGeometry,
    sample_initial_articulation_geometry,
)
from embodichain.lab.sim.objects import ArticulationJointKinematics

POINT_CLOUD_TOLERANCE = 1.0e-6
NON_TARGET_ARTICULATION_POINT_CLOUD_KEY = "non_target_articulation_point_cloud"


class _ArticulationGeometryProvider:
    """Minimal structural provider for initial articulation geometry sampling."""

    def __init__(
        self,
        *,
        initial_qpos: tuple[float, ...],
        backend_joint_names: tuple[str, ...],
        link_meshes: dict[str, tuple[torch.Tensor, torch.Tensor]],
        link_poses: torch.Tensor,
        body_scale: tuple[float, float, float],
        parent_joint_chain: tuple[object, ...],
    ) -> None:
        self.device = torch.device("cpu")
        self.initial_qpos = initial_qpos
        self.initial_qpos_joint_names = backend_joint_names
        self.body_scale = body_scale
        self.link_names = list(link_meshes)
        self._link_meshes = link_meshes
        self._link_poses = link_poses
        self._parent_joint_chain = parent_joint_chain
        self.fk_calls: list[tuple[torch.Tensor, list[str], tuple[str, ...]]] = []

    def compute_fk(
        self,
        qpos: torch.Tensor,
        *,
        link_names: list[str],
        qpos_joint_names: tuple[str, ...],
    ) -> torch.Tensor:
        """Return configured initial link poses and record the FK request."""
        self.fk_calls.append((qpos.clone(), list(link_names), tuple(qpos_joint_names)))
        return self._link_poses.clone()

    def get_link_vert_face(
        self,
        link_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one configured link-local triangle mesh."""
        return self._link_meshes[link_name]

    def get_parent_joint_chain(
        self,
        link_name: str,
    ) -> tuple[object, ...]:
        """Return the configured target-to-root parent-joint chain."""
        assert link_name in self.link_names
        return self._parent_joint_chain


def _make_point_cloud_articulation(
    *,
    initial_qpos: tuple[float, ...] = (0.25,),
    backend_joint_names: tuple[str, ...] = ("joint",),
    link_meshes: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
    link_poses: torch.Tensor | None = None,
    body_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    parent_joint_chain: tuple[object, ...] = (),
) -> tuple[
    _ArticulationGeometryProvider,
    list[tuple[torch.Tensor, list[str], tuple[str, ...]]],
]:
    """Build a pure-Python structural provider for the geometry adapter."""
    if link_meshes is None:
        link_meshes = {
            "target": (
                torch.tensor(
                    ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
                    dtype=torch.float32,
                ),
                torch.tensor(((0, 1, 2),), dtype=torch.long),
            )
        }
    if link_poses is None:
        link_poses = torch.eye(4, dtype=torch.float32).repeat(
            1,
            len(link_meshes),
            1,
            1,
        )
    articulation = _ArticulationGeometryProvider(
        initial_qpos=initial_qpos,
        backend_joint_names=backend_joint_names,
        link_meshes=link_meshes,
        link_poses=link_poses,
        body_scale=body_scale,
        parent_joint_chain=parent_joint_chain,
    )
    return articulation, articulation.fk_calls


@pytest.mark.no_sim
class TestInitialArticulationGeometrySampling:
    """Pure CPU coverage for the Atomic Action articulation geometry adapter."""

    def test_passes_initial_joint_state_and_names_to_provider(self):
        articulation, fk_calls = _make_point_cloud_articulation(
            initial_qpos=(2.0, 1.0),
            backend_joint_names=("joint_b", "joint_a"),
        )

        sample_initial_articulation_geometry(
            articulation,
            "target",
            initial_qpos=articulation.initial_qpos,
            initial_qpos_joint_names=articulation.initial_qpos_joint_names,
            body_scale=articulation.body_scale,
            articulation_point_count=5,
            target_point_count=3,
        )

        assert len(fk_calls) == 1
        fk_qpos, fk_link_names, fk_qpos_joint_names = fk_calls[0]
        assert torch.equal(fk_qpos, torch.tensor(((2.0, 1.0),)))
        assert fk_link_names == ["target"]
        assert fk_qpos_joint_names == ("joint_b", "joint_a")

    def test_target_only_geometry_round_trips_empty_non_target_cloud(self):
        articulation, _ = _make_point_cloud_articulation()

        geometry = sample_initial_articulation_geometry(
            articulation,
            "target",
            initial_qpos=articulation.initial_qpos,
            initial_qpos_joint_names=articulation.initial_qpos_joint_names,
            body_scale=articulation.body_scale,
            articulation_point_count=13,
            target_point_count=7,
        )

        non_target_points = geometry.non_target_articulation_point_cloud
        assert non_target_points is not None
        assert non_target_points.shape == (0, 3)
        assert non_target_points.dtype == geometry.articulation_point_cloud.dtype
        assert non_target_points.device == geometry.articulation_point_cloud.device

        first_object_geometry = geometry.to_object_geometry()
        second_object_geometry = geometry.to_object_geometry()
        assert set(first_object_geometry) == {
            "target_link_point_cloud",
            "articulation_point_cloud",
            NON_TARGET_ARTICULATION_POINT_CLOUD_KEY,
        }
        assert torch.equal(
            first_object_geometry[NON_TARGET_ARTICULATION_POINT_CLOUD_KEY],
            non_target_points,
        )
        assert (
            first_object_geometry[NON_TARGET_ARTICULATION_POINT_CLOUD_KEY]
            is not non_target_points
        )
        assert (
            first_object_geometry[NON_TARGET_ARTICULATION_POINT_CLOUD_KEY]
            is not second_object_geometry[NON_TARGET_ARTICULATION_POINT_CLOUD_KEY]
        )

    def test_typed_geometry_omits_unknown_non_target_provenance(self):
        target_points = torch.tensor(((0.0, 0.0, 0.0),), dtype=torch.float32)
        articulation_points = target_points.clone()

        geometry = ArticulationAffordanceGeometry(
            target_link_point_cloud=target_points,
            articulation_point_cloud=articulation_points,
        )

        assert geometry.non_target_articulation_point_cloud is None
        assert (
            NON_TARGET_ARTICULATION_POINT_CLOUD_KEY not in geometry.to_object_geometry()
        )

    def test_uses_nearest_revolute_ancestor_after_fixed_descendants(self):
        triangle = torch.tensor(((0, 1, 2),), dtype=torch.long)
        link_meshes = {
            link_name: (
                torch.tensor(
                    ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
                    dtype=torch.float32,
                ),
                triangle,
            )
            for link_name in ("base", "near_parent", "moving", "target")
        }
        fixed_joint = ArticulationJointKinematics(
            name="target_fixed",
            joint_type="fixed",
            parent_link_name="moving",
            child_link_name="target",
            origin_pose=torch.eye(4),
            axis=torch.zeros(3),
        )
        near_origin_pose = torch.eye(4)
        near_origin_pose[:3, 3] = torch.tensor((1.0, 2.0, 3.0))
        near_joint = ArticulationJointKinematics(
            name="near_hinge",
            joint_type="revolute",
            parent_link_name="near_parent",
            child_link_name="moving",
            origin_pose=near_origin_pose,
            axis=torch.tensor((0.0, 0.0, 1.0)),
        )
        far_origin_pose = torch.eye(4)
        far_origin_pose[:3, 3] = torch.tensor((9.0, 9.0, 9.0))
        far_joint = ArticulationJointKinematics(
            name="far_hinge",
            joint_type="revolute",
            parent_link_name="base",
            child_link_name="near_parent",
            origin_pose=far_origin_pose,
            axis=torch.tensor((1.0, 0.0, 0.0)),
        )
        articulation, _ = _make_point_cloud_articulation(
            link_meshes=link_meshes,
            parent_joint_chain=(fixed_joint, near_joint, far_joint),
        )

        geometry = sample_initial_articulation_geometry(
            articulation,
            "target",
            initial_qpos=articulation.initial_qpos,
            initial_qpos_joint_names=articulation.initial_qpos_joint_names,
            body_scale=articulation.body_scale,
            articulation_point_count=16,
            target_point_count=8,
        )

        assert torch.equal(
            geometry.revolute_axis_origin,
            near_origin_pose[:3, 3],
        )
        assert torch.equal(
            geometry.revolute_joint_axis,
            near_joint.axis,
        )

    def test_uses_nearest_prismatic_axis_after_fixed_descendant(self):
        triangle = torch.tensor(((0, 1, 2),), dtype=torch.long)
        link_meshes = {
            link_name: (
                torch.tensor(
                    ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
                    dtype=torch.float32,
                ),
                triangle,
            )
            for link_name in ("base", "near_parent", "moving", "target")
        }
        fixed_joint = ArticulationJointKinematics(
            name="target_fixed",
            joint_type="fixed",
            parent_link_name="moving",
            child_link_name="target",
            origin_pose=torch.eye(4),
            axis=torch.zeros(3),
        )
        near_origin_pose = torch.eye(4)
        near_origin_pose[:3, :3] = torch.tensor(
            ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        )
        near_joint = ArticulationJointKinematics(
            name="near_slider",
            joint_type="prismatic",
            parent_link_name="near_parent",
            child_link_name="moving",
            origin_pose=near_origin_pose,
            axis=torch.tensor((4.0, 0.0, 0.0)),
        )
        far_joint = ArticulationJointKinematics(
            name="far_slider",
            joint_type="prismatic",
            parent_link_name="base",
            child_link_name="near_parent",
            origin_pose=torch.eye(4),
            axis=torch.tensor((0.0, 0.0, 2.0)),
        )
        articulation, _ = _make_point_cloud_articulation(
            link_meshes=link_meshes,
            parent_joint_chain=(fixed_joint, near_joint, far_joint),
        )

        geometry = sample_initial_articulation_geometry(
            articulation,
            "target",
            initial_qpos=articulation.initial_qpos,
            initial_qpos_joint_names=articulation.initial_qpos_joint_names,
            body_scale=articulation.body_scale,
            articulation_point_count=16,
            target_point_count=8,
        )

        assert torch.allclose(
            geometry.prismatic_joint_axis,
            torch.tensor((0.0, 1.0, 0.0)),
            atol=POINT_CLOUD_TOLERANCE,
        )
        assert geometry.revolute_joint_axis is None
        assert geometry.revolute_axis_origin is None

    def test_collects_both_nearest_joint_types_from_one_parent_chain(self):
        triangle = torch.tensor(((0, 1, 2),), dtype=torch.long)
        link_meshes = {
            link_name: (
                torch.tensor(
                    ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
                    dtype=torch.float32,
                ),
                triangle,
            )
            for link_name in ("root", "base", "near_parent", "hinge_child", "target")
        }
        near_prismatic = ArticulationJointKinematics(
            name="near_slider",
            joint_type="prismatic",
            parent_link_name="hinge_child",
            child_link_name="target",
            origin_pose=torch.eye(4),
            axis=torch.tensor((4.0, 0.0, 0.0)),
        )
        near_revolute_origin = torch.eye(4)
        near_revolute_origin[:3, 3] = torch.tensor((1.0, 2.0, 3.0))
        near_revolute = ArticulationJointKinematics(
            name="near_hinge",
            joint_type="revolute",
            parent_link_name="near_parent",
            child_link_name="hinge_child",
            origin_pose=near_revolute_origin,
            axis=torch.tensor((0.0, 5.0, 0.0)),
        )
        far_prismatic = ArticulationJointKinematics(
            name="far_slider",
            joint_type="prismatic",
            parent_link_name="base",
            child_link_name="near_parent",
            origin_pose=torch.eye(4),
            axis=torch.tensor((0.0, 0.0, 2.0)),
        )
        far_revolute_origin = torch.eye(4)
        far_revolute_origin[:3, 3] = torch.tensor((9.0, 9.0, 9.0))
        far_revolute = ArticulationJointKinematics(
            name="far_hinge",
            joint_type="revolute",
            parent_link_name="root",
            child_link_name="base",
            origin_pose=far_revolute_origin,
            axis=torch.tensor((0.0, 0.0, 7.0)),
        )
        articulation, _ = _make_point_cloud_articulation(
            link_meshes=link_meshes,
            parent_joint_chain=(
                near_prismatic,
                near_revolute,
                far_prismatic,
                far_revolute,
            ),
        )

        geometry = sample_initial_articulation_geometry(
            articulation,
            "target",
            initial_qpos=articulation.initial_qpos,
            initial_qpos_joint_names=articulation.initial_qpos_joint_names,
            body_scale=articulation.body_scale,
            articulation_point_count=16,
            target_point_count=8,
        )

        assert torch.equal(
            geometry.prismatic_joint_axis,
            torch.tensor((1.0, 0.0, 0.0)),
        )
        assert torch.equal(
            geometry.revolute_joint_axis,
            torch.tensor((0.0, 1.0, 0.0)),
        )
        assert torch.equal(
            geometry.revolute_axis_origin,
            near_revolute_origin[:3, 3],
        )

    def test_transforms_revolute_origin_from_rotated_parent_to_target_frame(self):
        triangle = torch.tensor(((0, 1, 2),), dtype=torch.long)
        link_meshes = {
            link_name: (
                torch.tensor(
                    ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
                    dtype=torch.float32,
                ),
                triangle,
            )
            for link_name in ("parent", "target")
        }
        root_from_parent = torch.eye(4)
        root_from_parent[:3, :3] = torch.tensor(
            ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        )
        root_from_parent[:3, 3] = torch.tensor((3.0, 4.0, 5.0))
        root_from_target = torch.eye(4)
        root_from_target[:3, :3] = torch.tensor(
            ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0))
        )
        root_from_target[:3, 3] = torch.tensor((0.5, -1.0, 2.0))
        joint_origin_pose = torch.eye(4)
        joint_origin_pose[:3, :3] = torch.tensor(
            ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0))
        )
        joint_origin_pose[:3, 3] = torch.tensor((1.0, 2.0, 3.0))
        link_poses = torch.stack((root_from_parent, root_from_target)).unsqueeze(0)
        joint = ArticulationJointKinematics(
            name="target_hinge",
            joint_type="revolute",
            parent_link_name="parent",
            child_link_name="target",
            origin_pose=joint_origin_pose,
            axis=torch.tensor((0.0, 4.0, 0.0)),
        )
        articulation, _ = _make_point_cloud_articulation(
            link_meshes=link_meshes,
            link_poses=link_poses,
            parent_joint_chain=(joint,),
        )

        geometry = sample_initial_articulation_geometry(
            articulation,
            "target",
            initial_qpos=articulation.initial_qpos,
            initial_qpos_joint_names=articulation.initial_qpos_joint_names,
            body_scale=articulation.body_scale,
            articulation_point_count=16,
            target_point_count=8,
        )

        target_from_joint = (
            torch.linalg.inv(root_from_target) @ root_from_parent @ joint_origin_pose
        )
        assert torch.allclose(
            geometry.revolute_axis_origin,
            target_from_joint[:3, 3],
            atol=POINT_CLOUD_TOLERANCE,
        )
        expected_axis = target_from_joint[:3, :3] @ joint.axis
        expected_axis = expected_axis / torch.linalg.vector_norm(expected_axis)
        assert torch.allclose(
            geometry.revolute_joint_axis,
            expected_axis,
            atol=POINT_CLOUD_TOLERANCE,
        )
        assert torch.linalg.vector_norm(
            geometry.revolute_joint_axis
        ).item() == pytest.approx(1.0)

    def test_omits_revolute_origin_when_parent_chain_has_only_fixed_joints(self):
        fixed_joint = ArticulationJointKinematics(
            name="target_fixed",
            joint_type="fixed",
            parent_link_name="parent",
            child_link_name="target",
            origin_pose=torch.eye(4),
            axis=torch.zeros(3),
        )
        articulation, _ = _make_point_cloud_articulation(
            parent_joint_chain=(fixed_joint,),
        )

        geometry = sample_initial_articulation_geometry(
            articulation,
            "target",
            initial_qpos=articulation.initial_qpos,
            initial_qpos_joint_names=articulation.initial_qpos_joint_names,
            body_scale=articulation.body_scale,
            articulation_point_count=5,
            target_point_count=3,
        )

        assert isinstance(geometry, ArticulationAffordanceGeometry)
        first_object_geometry = geometry.to_object_geometry()
        second_object_geometry = geometry.to_object_geometry()
        assert first_object_geometry is not second_object_geometry
        assert set(first_object_geometry) == {
            "target_link_point_cloud",
            "articulation_point_cloud",
            NON_TARGET_ARTICULATION_POINT_CLOUD_KEY,
        }
        assert geometry.prismatic_joint_axis is None
        assert geometry.revolute_joint_axis is None
        assert geometry.revolute_axis_origin is None

    @pytest.mark.parametrize("joint_type", ("prismatic", "revolute"))
    def test_rejects_joint_axis_metadata_with_unknown_parent_link(
        self,
        joint_type: str,
    ):
        joint = ArticulationJointKinematics(
            name="target_joint",
            joint_type=joint_type,
            parent_link_name="missing_parent",
            child_link_name="target",
            origin_pose=torch.eye(4),
            axis=torch.tensor((0.0, 0.0, 1.0)),
        )
        articulation, _ = _make_point_cloud_articulation(
            parent_joint_chain=(joint,),
        )

        with pytest.raises(ValueError, match="parent link.*not an articulation link"):
            sample_initial_articulation_geometry(
                articulation,
                "target",
                initial_qpos=articulation.initial_qpos,
                initial_qpos_joint_names=articulation.initial_qpos_joint_names,
                body_scale=articulation.body_scale,
                articulation_point_count=5,
                target_point_count=3,
            )

    @pytest.mark.parametrize("joint_type", ("prismatic", "revolute"))
    def test_rejects_joint_axis_metadata_with_invalid_origin_pose(
        self,
        joint_type: str,
    ):
        joint = SimpleNamespace(
            name="target_joint",
            joint_type=joint_type,
            parent_link_name="target",
            origin_pose=torch.full((4, 4), float("nan")),
            axis=torch.tensor((0.0, 0.0, 1.0)),
        )
        articulation, _ = _make_point_cloud_articulation(
            parent_joint_chain=(joint,),
        )

        with pytest.raises(ValueError, match="origin pose must be finite"):
            sample_initial_articulation_geometry(
                articulation,
                "target",
                initial_qpos=articulation.initial_qpos,
                initial_qpos_joint_names=articulation.initial_qpos_joint_names,
                body_scale=articulation.body_scale,
                articulation_point_count=5,
                target_point_count=3,
            )

    @pytest.mark.parametrize(
        ("joint_type", "axis"),
        (
            ("prismatic", torch.zeros(3)),
            ("revolute", torch.tensor((float("nan"), 0.0, 1.0))),
            ("prismatic", torch.zeros(2)),
        ),
    )
    def test_rejects_invalid_joint_axes(
        self,
        joint_type: str,
        axis: torch.Tensor,
    ):
        joint = SimpleNamespace(
            name="target_joint",
            joint_type=joint_type,
            parent_link_name="target",
            origin_pose=torch.eye(4),
            axis=axis,
        )
        articulation, _ = _make_point_cloud_articulation(
            parent_joint_chain=(joint,),
        )

        with pytest.raises(ValueError, match="axis must be finite and nonzero"):
            sample_initial_articulation_geometry(
                articulation,
                "target",
                initial_qpos=articulation.initial_qpos,
                initial_qpos_joint_names=articulation.initial_qpos_joint_names,
                body_scale=articulation.body_scale,
                articulation_point_count=5,
                target_point_count=3,
            )

    def test_non_target_cloud_excludes_target_link_in_target_initial_frame(self):
        triangle = torch.tensor(((0, 1, 2),), dtype=torch.long)
        link_meshes = {
            "body": (
                torch.tensor(
                    ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
                    dtype=torch.float32,
                ),
                triangle,
            ),
            "target": (
                torch.tensor(
                    ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
                    dtype=torch.float32,
                ),
                triangle,
            ),
        }
        initial_link_poses = torch.eye(4, dtype=torch.float32).repeat(1, 2, 1, 1)
        initial_link_poses[0, 0, :3, 3] = torch.tensor((4.0, 0.0, 0.0))
        initial_link_poses[0, 1, :3, :3] = torch.tensor(
            ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        )
        initial_link_poses[0, 1, :3, 3] = torch.tensor((1.0, 2.0, 0.0))
        articulation, _ = _make_point_cloud_articulation(
            link_meshes=link_meshes,
            link_poses=initial_link_poses,
        )

        geometry = sample_initial_articulation_geometry(
            articulation,
            "target",
            initial_qpos=articulation.initial_qpos,
            initial_qpos_joint_names=articulation.initial_qpos_joint_names,
            body_scale=articulation.body_scale,
            articulation_point_count=256,
            target_point_count=32,
        )

        assert set(geometry.to_object_geometry()) == {
            "target_link_point_cloud",
            "articulation_point_cloud",
            NON_TARGET_ARTICULATION_POINT_CLOUD_KEY,
        }
        target_points = geometry.target_link_point_cloud
        assert target_points.shape == (32, 3)
        assert torch.allclose(
            target_points[:, 2],
            torch.zeros(32),
            atol=POINT_CLOUD_TOLERANCE,
        )
        assert bool((target_points[:, :2] >= -POINT_CLOUD_TOLERANCE).all())
        assert bool(
            (
                target_points[:, 0] + target_points[:, 1] <= 1.0 + POINT_CLOUD_TOLERANCE
            ).all()
        )

        articulation_points = geometry.articulation_point_cloud
        assert articulation_points.shape == (256, 3)
        body_mask = articulation_points[:, 0] < -0.5
        assert bool(body_mask.any())
        assert bool((~body_mask).any())
        body_points = articulation_points[body_mask]
        assert bool(
            (
                (body_points[:, 0] >= -2.0 - POINT_CLOUD_TOLERANCE)
                & (body_points[:, 0] <= -1.0 + POINT_CLOUD_TOLERANCE)
                & (body_points[:, 1] >= -4.0 - POINT_CLOUD_TOLERANCE)
                & (body_points[:, 1] <= -3.0 + POINT_CLOUD_TOLERANCE)
            ).all()
        )
        sampled_target_points = articulation_points[~body_mask]
        assert bool((sampled_target_points[:, :2] >= -POINT_CLOUD_TOLERANCE).all())
        assert bool(
            (
                sampled_target_points[:, 0] + sampled_target_points[:, 1]
                <= 1.0 + POINT_CLOUD_TOLERANCE
            ).all()
        )

        non_target_points = geometry.non_target_articulation_point_cloud
        assert non_target_points is not None
        assert non_target_points.shape == (256, 3)
        assert bool((non_target_points[:, 0] < -0.5).all())
        assert bool(
            (
                (non_target_points[:, 0] >= -2.0 - POINT_CLOUD_TOLERANCE)
                & (non_target_points[:, 0] <= -1.0 + POINT_CLOUD_TOLERANCE)
                & (non_target_points[:, 1] >= -4.0 - POINT_CLOUD_TOLERANCE)
                & (non_target_points[:, 1] <= -3.0 + POINT_CLOUD_TOLERANCE)
            ).all()
        )
        assert torch.allclose(
            non_target_points[:, 2],
            torch.zeros(256),
            atol=POINT_CLOUD_TOLERANCE,
        )
        exported_non_target_points = geometry.to_object_geometry()[
            NON_TARGET_ARTICULATION_POINT_CLOUD_KEY
        ]
        assert exported_non_target_points is not non_target_points
        exported_non_target_points.zero_()
        assert not torch.equal(exported_non_target_points, non_target_points)

    def test_sampling_consumes_open3d_rng_without_resetting_it(self):
        import open3d as o3d

        articulation, _ = _make_point_cloud_articulation()
        o3d.utility.random.seed(7)

        first = sample_initial_articulation_geometry(
            articulation,
            "target",
            initial_qpos=articulation.initial_qpos,
            initial_qpos_joint_names=articulation.initial_qpos_joint_names,
            body_scale=articulation.body_scale,
            articulation_point_count=23,
            target_point_count=17,
        )
        second = sample_initial_articulation_geometry(
            articulation,
            "target",
            initial_qpos=articulation.initial_qpos,
            initial_qpos_joint_names=articulation.initial_qpos_joint_names,
            body_scale=articulation.body_scale,
            articulation_point_count=23,
            target_point_count=17,
        )

        assert not torch.equal(
            first.target_link_point_cloud,
            second.target_link_point_cloud,
        )
        assert not torch.equal(
            first.articulation_point_cloud,
            second.articulation_point_cloud,
        )

    def test_surface_sampling_preserves_torch_dtype_and_owns_its_data(self):
        vertices = torch.tensor(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            dtype=torch.float64,
        )
        triangles = torch.tensor(((0, 1, 2),), dtype=torch.long)
        original_vertices = vertices.clone()

        points = articulation_geometry_module._sample_mesh_surface_points(
            vertices, triangles, 11
        )

        assert points.device == vertices.device
        assert points.dtype == vertices.dtype
        assert points.shape == (11, 3)
        points.zero_()
        assert torch.equal(vertices, original_vertices)

    @pytest.mark.parametrize(
        "triangles",
        (
            torch.empty((0, 3), dtype=torch.long),
            torch.tensor(((0, 1, 2),), dtype=torch.long),
        ),
    )
    def test_surface_sampling_falls_back_to_vertices_without_valid_faces(
        self,
        triangles: torch.Tensor,
    ):
        vertices = torch.tensor(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
            dtype=torch.float32,
        )

        points = articulation_geometry_module._sample_mesh_surface_points(
            vertices, triangles, 7
        )

        assert points.shape == (7, 3)
        assert set(points[:, 0].tolist()) <= {0.0, 1.0, 2.0}

    @pytest.mark.parametrize(
        "target_triangles",
        (
            pytest.param(
                torch.empty((0, 3), dtype=torch.long),
                id="no-faces",
            ),
            pytest.param(
                torch.tensor(((0, 1, 2),), dtype=torch.long),
                id="degenerate-faces",
            ),
        ),
    )
    def test_rejects_mixed_mesh_when_target_has_no_valid_triangle_surface(
        self,
        target_triangles: torch.Tensor,
    ) -> None:
        valid_triangle = torch.tensor(((0, 1, 2),), dtype=torch.long)
        articulation, _ = _make_point_cloud_articulation(
            link_meshes={
                "body": (
                    torch.tensor(
                        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
                        dtype=torch.float32,
                    ),
                    valid_triangle,
                ),
                "target": (
                    torch.tensor(
                        ((10.0, 0.0, 0.0), (11.0, 0.0, 0.0), (12.0, 0.0, 0.0)),
                        dtype=torch.float32,
                    ),
                    target_triangles,
                ),
            }
        )

        with pytest.raises(
            ValueError,
            match="Link 'target' must contain at least one non-degenerate triangle",
        ):
            sample_initial_articulation_geometry(
                articulation,
                "target",
                initial_qpos=articulation.initial_qpos,
                initial_qpos_joint_names=articulation.initial_qpos_joint_names,
                body_scale=articulation.body_scale,
                articulation_point_count=32,
                target_point_count=8,
            )

    def test_merged_surface_sampling_is_weighted_by_face_area(self):
        triangle = torch.tensor(((0, 1, 2),), dtype=torch.long)
        articulation, _ = _make_point_cloud_articulation(
            link_meshes={
                "small": (
                    torch.tensor(
                        ((-10.0, 0.0, 0.0), (-9.0, 0.0, 0.0), (-10.0, 1.0, 0.0)),
                        dtype=torch.float32,
                    ),
                    triangle,
                ),
                "target": (
                    torch.tensor(
                        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0)),
                        dtype=torch.float32,
                    ),
                    triangle,
                ),
            }
        )

        geometry = sample_initial_articulation_geometry(
            articulation,
            "target",
            initial_qpos=articulation.initial_qpos,
            initial_qpos_joint_names=articulation.initial_qpos_joint_names,
            body_scale=articulation.body_scale,
            articulation_point_count=5_000,
            target_point_count=3,
        )

        articulation_points = geometry.articulation_point_cloud
        small_face_fraction = float((articulation_points[:, 0] < -9.0).float().mean())
        assert small_face_fraction == pytest.approx(0.2, abs=0.03)

    def test_non_target_merged_sampling_is_weighted_by_face_area(self):
        triangle = torch.tensor(((0, 1, 2),), dtype=torch.long)
        articulation, _ = _make_point_cloud_articulation(
            link_meshes={
                "small": (
                    torch.tensor(
                        ((-10.0, 0.0, 0.0), (-9.0, 0.0, 0.0), (-10.0, 1.0, 0.0)),
                        dtype=torch.float32,
                    ),
                    triangle,
                ),
                "large": (
                    torch.tensor(
                        ((-20.0, 0.0, 0.0), (-18.0, 0.0, 0.0), (-20.0, 2.0, 0.0)),
                        dtype=torch.float32,
                    ),
                    triangle,
                ),
                "target": (
                    torch.tensor(
                        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
                        dtype=torch.float32,
                    ),
                    triangle,
                ),
            }
        )

        geometry = sample_initial_articulation_geometry(
            articulation,
            "target",
            initial_qpos=articulation.initial_qpos,
            initial_qpos_joint_names=articulation.initial_qpos_joint_names,
            body_scale=articulation.body_scale,
            articulation_point_count=5_000,
            target_point_count=3,
        )

        non_target_points = geometry.non_target_articulation_point_cloud
        assert non_target_points is not None
        small_face_fraction = float((non_target_points[:, 0] > -15.0).float().mean())
        assert small_face_fraction == pytest.approx(0.2, abs=0.03)

    @pytest.mark.parametrize(
        ("target_link_name", "error_type", "message"),
        (
            (None, TypeError, "target_link_name must be a string"),
            (" ", ValueError, "target_link_name must be non-empty"),
            ("missing", ValueError, "Unknown articulation link"),
        ),
    )
    def test_rejects_invalid_target_link_names(
        self,
        target_link_name: object,
        error_type: type[Exception],
        message: str,
    ):
        articulation, _ = _make_point_cloud_articulation()

        with pytest.raises(error_type, match=message):
            sample_initial_articulation_geometry(
                articulation,
                target_link_name,  # type: ignore[arg-type]
                initial_qpos=articulation.initial_qpos,
                initial_qpos_joint_names=articulation.initial_qpos_joint_names,
                body_scale=articulation.body_scale,
            )

    @pytest.mark.parametrize(
        ("field_name", "value", "error_type", "message"),
        (
            (
                "articulation_point_count",
                True,
                TypeError,
                "articulation_point_count must be an integer",
            ),
            (
                "target_point_count",
                0,
                ValueError,
                "target_point_count must be positive",
            ),
        ),
    )
    def test_rejects_invalid_point_counts(
        self,
        field_name: str,
        value: object,
        error_type: type[Exception],
        message: str,
    ):
        articulation, _ = _make_point_cloud_articulation()

        with pytest.raises(error_type, match=message):
            sample_initial_articulation_geometry(
                articulation,
                "target",
                initial_qpos=articulation.initial_qpos,
                initial_qpos_joint_names=articulation.initial_qpos_joint_names,
                body_scale=articulation.body_scale,
                **{field_name: value},  # type: ignore[arg-type]
            )

    def test_rejects_non_unit_body_scale(self):
        articulation, _ = _make_point_cloud_articulation(
            body_scale=(1.0, 2.0, 1.0),
        )

        with pytest.raises(ValueError, match="requires unit body_scale"):
            sample_initial_articulation_geometry(
                articulation,
                "target",
                initial_qpos=articulation.initial_qpos,
                initial_qpos_joint_names=articulation.initial_qpos_joint_names,
                body_scale=articulation.body_scale,
            )

    def test_rejects_initial_joint_name_count_mismatch(self):
        articulation, _ = _make_point_cloud_articulation(
            initial_qpos=(0.1, 0.2),
            backend_joint_names=("joint",),
        )

        with pytest.raises(ValueError, match="matching its joint-name sequence"):
            sample_initial_articulation_geometry(
                articulation,
                "target",
                initial_qpos=articulation.initial_qpos,
                initial_qpos_joint_names=articulation.initial_qpos_joint_names,
                body_scale=articulation.body_scale,
            )

    @pytest.mark.parametrize(
        "initial_qpos",
        (
            (),
            (float("nan"),),
        ),
    )
    def test_rejects_invalid_initial_qpos(self, initial_qpos: tuple[float, ...]):
        articulation, _ = _make_point_cloud_articulation(
            initial_qpos=initial_qpos,
        )

        with pytest.raises(ValueError, match="finite vector matching"):
            sample_initial_articulation_geometry(
                articulation,
                "target",
                initial_qpos=articulation.initial_qpos,
                initial_qpos_joint_names=articulation.initial_qpos_joint_names,
                body_scale=articulation.body_scale,
            )

    def test_rejects_invalid_mesh_indices(self):
        articulation, _ = _make_point_cloud_articulation(
            link_meshes={
                "target": (
                    torch.zeros((3, 3), dtype=torch.float32),
                    torch.tensor(((0, 1, 3),), dtype=torch.long),
                )
            }
        )

        with pytest.raises(ValueError, match="triangles reference invalid vertices"):
            sample_initial_articulation_geometry(
                articulation,
                "target",
                initial_qpos=articulation.initial_qpos,
                initial_qpos_joint_names=articulation.initial_qpos_joint_names,
                body_scale=articulation.body_scale,
            )

    @pytest.mark.parametrize(
        "triangles",
        (
            torch.tensor(((0.0, 1.0, 2.0),), dtype=torch.float32),
            torch.tensor(((False, True, True),), dtype=torch.bool),
        ),
    )
    def test_rejects_non_integer_mesh_indices(self, triangles: torch.Tensor):
        articulation, _ = _make_point_cloud_articulation(
            link_meshes={
                "target": (
                    torch.zeros((3, 3), dtype=torch.float32),
                    triangles,
                )
            }
        )

        with pytest.raises(ValueError, match="triangles must use integer indices"):
            sample_initial_articulation_geometry(
                articulation,
                "target",
                initial_qpos=articulation.initial_qpos,
                initial_qpos_joint_names=articulation.initial_qpos_joint_names,
                body_scale=articulation.body_scale,
            )
