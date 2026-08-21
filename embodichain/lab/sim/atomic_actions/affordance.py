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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from embodichain.toolkits.graspkit.pg_grasp import (
    GraspGenerator,
    GraspGeneratorCfg,
)
from embodichain.toolkits.graspkit.pg_grasp.gripper_collision_checker import (
    GripperCollisionCfg,
)
from embodichain.utils import logger

if TYPE_CHECKING:
    from embodichain.lab.sim.common import BatchEntity


@dataclass
class Affordance:
    """Base class for affordance data.

    Represents an object's interaction possibilities. Subclasses carry whatever
    typed fields they need (mesh tensors, interaction points, etc.); the base
    class only carries an object label and a free-form custom_config dict.
    """

    object_label: str = ""
    """Label of the object this affordance belongs to."""

    custom_config: dict[str, Any] = field(default_factory=dict)
    """User-defined configuration payload."""

    def set_custom_config(self, key: str, value: Any) -> None:
        """Set a custom affordance configuration value."""
        self.custom_config[key] = value

    def get_custom_config(self, key: str, default: Any = None) -> Any:
        """Get a custom affordance configuration value."""
        return self.custom_config.get(key, default)

    def get_batch_size(self) -> int:
        """Return the batch size of this affordance data."""
        return 1


@dataclass
class AntipodalAffordance(Affordance):
    """Antipodal grasp affordance for parallel-jaw grippers.

    The affordance owns only target-local triangle-mesh data. Simulator entity
    handles and live poses belong to scene grounding, not semantic geometry.
    """

    mesh_vertices: torch.Tensor | None = None
    """Object mesh vertices, shape [N, 3]."""

    mesh_triangles: torch.Tensor | None = None
    """Object mesh triangle indices, shape [M, 3]."""

    generator_cfg: GraspGeneratorCfg | None = None
    """Optional grasp-generator configuration."""

    gripper_collision_cfg: GripperCollisionCfg | None = None
    """Optional gripper-collision configuration."""

    force_reannotate: bool = False
    """If True, recompute the grasp annotation on each access."""

    _generator: GraspGenerator | None = field(default=None, init=False, repr=False)

    MAX_SURFACE_POINT_COUNT: ClassVar[int] = 1000
    """Maximum point-cloud size used for geometry-distribution analysis."""

    def _init_generator(self) -> None:
        if self.mesh_vertices is None or self.mesh_triangles is None:
            logger.log_error(
                "mesh_vertices and mesh_triangles must be provided to initialize "
                "AntipodalAffordance.",
                ValueError,
            )
        self._generator = GraspGenerator(
            vertices=self.mesh_vertices,
            triangles=self.mesh_triangles,
            cfg=self.generator_cfg,
            gripper_collision_cfg=self.gripper_collision_cfg,
        )
        if self.force_reannotate or self._generator._hit_point_pairs is None:
            self._generator.annotate()

    def _resolve_approach_direction(
        self, approach_direction: torch.Tensor
    ) -> torch.Tensor:
        """Move the approach direction to the grasp generator device."""
        return approach_direction.to(
            device=self._generator.device,
            dtype=torch.float32,
        )

    def get_valid_grasp_poses(
        self,
        obj_poses: torch.Tensor,
        approach_direction: torch.Tensor = torch.tensor(
            [0, 0, -1], dtype=torch.float32
        ),
        obj_longest_axis: torch.Tensor | None = None,
        is_positive_part: bool | torch.Tensor = True,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Return valid grasps, optionally restricted to one end of an axis.

        Args:
            obj_poses: Batched object poses with shape ``(B, 4, 4)``.
            approach_direction: Shared world-frame approach direction.
            obj_longest_axis: Optional shared ``(3,)`` or batched ``(B, 3)``
                world-frame object axis. ``None`` keeps the default center mode
                and does not partition grasp candidates.
            is_positive_part: Select the positive or negative projected half of
                ``obj_longest_axis``. May be one bool or a ``(B,)`` bool tensor.

        Returns:
            Per-object ``(grasp_poses, costs)`` tuples.
        """
        if self._generator is None:
            self._init_generator()
        approach_direction = self._resolve_approach_direction(approach_direction)
        axes = self._resolve_grasp_region_axes(obj_poses, obj_longest_axis)
        positive_parts = self._resolve_positive_parts(obj_poses, is_positive_part)
        results = []
        for i, obj_pose in enumerate(obj_poses):
            is_success, grasp_poses, _, costs = self._generator.get_valid_grasp_poses(
                object_pose=obj_pose,
                approach_direction=approach_direction,
                obj_longest_axis=None if axes is None else axes[i],
                is_positive_part=bool(positive_parts[i].item()),
            )
            if grasp_poses.shape == (4, 4):
                grasp_poses = grasp_poses.unsqueeze(0)
            if costs.dim() == 0:
                costs = costs.unsqueeze(0)
            if not is_success:
                logger.log_warning(
                    f"Failed to find valid grasp poses for {i}-th object."
                )
                costs = torch.full(
                    (grasp_poses.shape[0],),
                    torch.inf,
                    dtype=torch.float32,
                    device=grasp_poses.device,
                )
            results.append((grasp_poses, costs))
        return results

    def sample_surface_points(self, max_points: int = 1000) -> torch.Tensor:
        """Deterministically sample at most 1000 target-local surface points.

        Triangle faces are selected proportionally to area and barycentric
        coordinates use deterministic low-discrepancy sequences. Degenerate or
        triangle-free meshes fall back to an evenly subsampled vertex cloud.

        Args:
            max_points: Requested point cap in ``[1, 1000]``.

        Returns:
            Target-local surface points with shape ``(N, 3)``.

        Raises:
            ValueError: If mesh geometry or ``max_points`` is invalid.
        """
        if not isinstance(max_points, int) or isinstance(max_points, bool):
            raise TypeError("max_points must be an integer.")
        if not 1 <= max_points <= self.MAX_SURFACE_POINT_COUNT:
            raise ValueError(
                f"max_points must be between 1 and {self.MAX_SURFACE_POINT_COUNT}."
            )
        if self.mesh_vertices is None:
            raise ValueError("AntipodalAffordance requires mesh_vertices.")
        vertices = self.mesh_vertices.to(dtype=torch.float32)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] == 0:
            raise ValueError("mesh_vertices must have shape (N, 3) with N > 0.")

        triangles = self.mesh_triangles
        if triangles is None or triangles.numel() == 0:
            return self._evenly_subsample_points(vertices, max_points)
        triangles = triangles.to(device=vertices.device, dtype=torch.long)
        if triangles.ndim != 2 or triangles.shape[1] != 3:
            raise ValueError("mesh_triangles must have shape (M, 3).")
        if triangles.min() < 0 or triangles.max() >= vertices.shape[0]:
            raise ValueError("mesh_triangles contains an invalid vertex index.")

        face_vertices = vertices[triangles]
        face_areas = 0.5 * torch.linalg.vector_norm(
            torch.cross(
                face_vertices[:, 1] - face_vertices[:, 0],
                face_vertices[:, 2] - face_vertices[:, 0],
                dim=1,
            ),
            dim=1,
        )
        valid_faces = face_areas > torch.finfo(vertices.dtype).eps
        if not valid_faces.any():
            return self._evenly_subsample_points(vertices, max_points)
        face_vertices = face_vertices[valid_faces]
        face_areas = face_areas[valid_faces]

        sample_index = torch.arange(
            max_points, device=vertices.device, dtype=vertices.dtype
        )
        area_quantiles = (sample_index + 0.5) / max_points
        cumulative_area = torch.cumsum(face_areas / face_areas.sum(), dim=0)
        face_indices = torch.searchsorted(cumulative_area, area_quantiles).clamp_max(
            face_vertices.shape[0] - 1
        )
        sampled_faces = face_vertices[face_indices]

        barycentric_u = torch.frac((sample_index + 0.5) * 0.7548776662466927)
        barycentric_v = torch.frac((sample_index + 0.5) * 0.5698402909980532)
        sqrt_u = torch.sqrt(barycentric_u)
        weights = torch.stack(
            (
                1.0 - sqrt_u,
                sqrt_u * (1.0 - barycentric_v),
                sqrt_u * barycentric_v,
            ),
            dim=1,
        )
        return torch.sum(sampled_faces * weights.unsqueeze(2), dim=1)

    def get_object_longest_axis(
        self,
        obj_poses: torch.Tensor,
        *,
        max_points: int = 1000,
    ) -> torch.Tensor:
        """Find the widest surface-point distribution axis in world space.

        Args:
            obj_poses: Current object poses with shape ``(B, 4, 4)``.
            max_points: Surface point cap, never greater than 1000.

        Returns:
            Normalized first right-singular vectors with shape ``(B, 3)``.

        Raises:
            ValueError: If poses or sampled geometry are invalid or degenerate.
        """
        if obj_poses.ndim != 3 or obj_poses.shape[1:] != (4, 4):
            raise ValueError("obj_poses must have shape (B, 4, 4).")
        points = self.sample_surface_points(max_points=max_points).to(
            device=obj_poses.device,
            dtype=torch.float32,
        )
        poses = obj_poses.to(dtype=torch.float32)
        world_points = (
            torch.matmul(points.unsqueeze(0), poses[:, :3, :3].transpose(1, 2))
            + poses[:, None, :3, 3]
        )
        centered = world_points - world_points.mean(dim=1, keepdim=True)
        if torch.any(torch.linalg.vector_norm(centered, dim=2).amax(dim=1) <= 1.0e-8):
            raise ValueError("Object surface point distribution is degenerate.")
        _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
        if torch.any(singular_values[:, 0] <= 1.0e-8):
            raise ValueError("Object surface point distribution has no principal axis.")
        return torch.nn.functional.normalize(vh[:, 0, :], dim=1)

    @staticmethod
    def _evenly_subsample_points(
        points: torch.Tensor,
        max_points: int,
    ) -> torch.Tensor:
        """Return an evenly spaced deterministic subset of ``points``."""
        if points.shape[0] <= max_points:
            return points.clone()
        indices = (
            torch.linspace(
                0,
                points.shape[0] - 1,
                max_points,
                device=points.device,
            )
            .round()
            .to(torch.long)
        )
        return points[indices]

    def _resolve_grasp_region_axes(
        self,
        obj_poses: torch.Tensor,
        obj_longest_axis: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Validate and broadcast optional world-frame grasp-region axes."""
        if obj_longest_axis is None:
            return None
        axes = torch.as_tensor(
            obj_longest_axis,
            dtype=torch.float32,
            device=self._generator.device,
        )
        if axes.shape == (3,):
            axes = axes.unsqueeze(0).expand(obj_poses.shape[0], -1)
        if axes.shape != (obj_poses.shape[0], 3):
            raise ValueError(
                "obj_longest_axis must have shape (3,) or "
                f"({obj_poses.shape[0]}, 3)."
            )
        if not torch.isfinite(axes).all() or torch.any(
            torch.linalg.vector_norm(axes, dim=1) <= 1.0e-8
        ):
            raise ValueError("obj_longest_axis must be finite and non-zero.")
        return torch.nn.functional.normalize(axes, dim=1)

    def _resolve_positive_parts(
        self,
        obj_poses: torch.Tensor,
        is_positive_part: bool | torch.Tensor,
    ) -> torch.Tensor:
        """Validate and broadcast positive/negative grasp-region selections."""
        if isinstance(is_positive_part, bool):
            return torch.full(
                (obj_poses.shape[0],),
                is_positive_part,
                dtype=torch.bool,
                device=self._generator.device,
            )
        positive_parts = torch.as_tensor(
            is_positive_part,
            device=self._generator.device,
        )
        if positive_parts.dtype != torch.bool or positive_parts.shape != (
            obj_poses.shape[0],
        ):
            raise ValueError(
                "is_positive_part must be a bool or a bool tensor with shape "
                f"({obj_poses.shape[0]},)."
            )
        return positive_parts

    def get_dual_arm_valid_grasp_poses(
        self,
        obj_poses: torch.Tensor,
        left_to_right_arm_direction: torch.Tensor,
        approach_direction: torch.Tensor = torch.tensor(
            [0, 0, -1], dtype=torch.float32
        ),
        middle_empty_ratio: float = 0.4,
    ) -> list[dict | None]:
        if self._generator is None:
            self._init_generator()
        approach_direction = self._resolve_approach_direction(approach_direction)
        results = []
        for i, obj_pose in enumerate(obj_poses):
            result = self._generator.get_dual_arm_valid_grasp_poses(
                object_pose=obj_pose,
                approach_direction=approach_direction,
                left_to_right_arm_direction=left_to_right_arm_direction,
                middle_empty_ratio=middle_empty_ratio,
            )
            results.append(result)
        return results

    def get_best_grasp_poses(
        self,
        obj_poses: torch.Tensor,
        approach_direction: torch.Tensor = torch.tensor(
            [0, 0, -1], dtype=torch.float32
        ),
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the best antipodal grasp for each object pose.

        Args:
            obj_poses: Batched object poses with shape ``(B, 4, 4)``.
            approach_direction: One shared ``(3,)`` world-frame direction or
                per-object directions with shape ``(B, 3)``.

        Returns:
            A success mask, best grasp poses, and gripper opening lengths with
            batch dimension ``B``.

        Raises:
            ValueError: If ``approach_direction`` has an incompatible shape.
        """
        if self._generator is None:
            self._init_generator()
        approach_direction = self._resolve_approach_direction(approach_direction)
        if approach_direction.shape == (3,):
            approach_directions = approach_direction.unsqueeze(0).expand(
                obj_poses.shape[0], -1
            )
        elif approach_direction.shape == (obj_poses.shape[0], 3):
            approach_directions = approach_direction
        else:
            raise ValueError(
                "approach_direction must have shape (3,) or "
                f"({obj_poses.shape[0]}, 3), got "
                f"{tuple(approach_direction.shape)}."
            )
        grasp_xpos_list: list[torch.Tensor] = []
        is_success_list: list[bool] = []
        open_length_list: list[float] = []
        for i, obj_pose in enumerate(obj_poses):
            is_success, grasp_xpos, open_length = self._generator.get_grasp_poses(
                obj_pose, approach_directions[i]
            )
            if is_success:
                grasp_xpos_list.append(grasp_xpos.unsqueeze(0))
            else:
                logger.log_warning(f"No valid grasp pose found for {i}-th object.")
                grasp_xpos_list.append(
                    torch.eye(
                        4, dtype=torch.float32, device=self._generator.device
                    ).unsqueeze(0)
                )
            is_success_list.append(is_success)
            open_length_list.append(open_length)
        is_success_t = torch.tensor(
            is_success_list, dtype=torch.bool, device=self._generator.device
        )
        grasp_xpos = torch.concatenate(grasp_xpos_list, dim=0)
        open_length_t = torch.tensor(
            open_length_list, dtype=torch.float32, device=self._generator.device
        )
        return is_success_t, grasp_xpos, open_length_t


@dataclass
class AxisAlignAffordance(AntipodalAffordance):
    """Antipodal grasp affordance with an object-local alignment axis."""

    internal_axis: torch.Tensor = field(
        default_factory=lambda: torch.tensor([0.0, 0.0, 1.0])
    )
    """Axis expressed in the target object's local frame."""

    def __post_init__(self) -> None:
        if (
            not isinstance(self.internal_axis, torch.Tensor)
            or self.internal_axis.shape != (3,)
            or not torch.isfinite(self.internal_axis).all()
        ):
            raise ValueError(
                "AxisAlignAffordance.internal_axis must be a finite (3,) tensor."
            )
        if torch.linalg.vector_norm(self.internal_axis) <= 1.0e-6:
            raise ValueError("AxisAlignAffordance.internal_axis must be non-zero.")
        self.internal_axis = self.internal_axis.clone()


@dataclass
class TwistAffordance(Affordance):
    """Target-local grasp point and rotation-axis geometry for twisting."""

    grasp_position: tuple[float, float, float] = field(kw_only=True)
    """Explicit target-local center of the gripper contact region."""

    axis_origin: tuple[float, float, float] = field(kw_only=True)
    """Explicit point on the rotation axis in the target-local frame."""

    twist_axis: torch.Tensor = field(
        default_factory=lambda: torch.tensor([0.0, 1.0, 0.0])
    )
    """Twist axis expressed in the target object's local frame."""

    joint_name: str | None = None
    """Optional stable articulation-joint name associated with the axis."""

    joint_limits: tuple[float, float] | None = None
    """Optional lower and upper angular limits in radians."""

    def __post_init__(self) -> None:
        if (
            not isinstance(self.twist_axis, torch.Tensor)
            or self.twist_axis.shape != (3,)
            or not torch.isfinite(self.twist_axis).all()
        ):
            raise ValueError("TwistAffordance.twist_axis must be a finite (3,) tensor.")
        if torch.linalg.vector_norm(self.twist_axis) <= 1.0e-6:
            raise ValueError("TwistAffordance.twist_axis must be non-zero.")
        self.twist_axis = self.twist_axis.clone()
        self.grasp_position = _validate_local_point(
            self.grasp_position, "TwistAffordance.grasp_position"
        )
        self.axis_origin = _validate_local_point(
            self.axis_origin, "TwistAffordance.axis_origin"
        )
        _validate_joint_metadata(self.joint_name, self.joint_limits)

    def get_grasp_pose(self, target_pose: torch.Tensor) -> torch.Tensor:
        """Construct a deterministic world grasp pose from local geometry.

        The pose z-axis follows :attr:`twist_axis`. The remaining axes are
        formed with an adaptive reference so the result is always in SO(3).

        Returns:
            Batched world-frame grasp poses with shape ``(B, 4, 4)``.

        Raises:
            ValueError: If ``target_pose`` is not a batched pose tensor.
        """
        if target_pose.dim() != 3 or target_pose.shape[1:] != (4, 4):
            raise ValueError("Target pose must have shape (B, 4, 4).")
        target_pose = target_pose.to(dtype=torch.float32)
        device = target_pose.device
        twist_axis = self.twist_axis.to(device=device, dtype=torch.float32)
        twist_axis = twist_axis / torch.linalg.vector_norm(twist_axis)

        z_axis = torch.matmul(target_pose[:, :3, :3], twist_axis)
        z_axis = torch.nn.functional.normalize(z_axis, dim=1)
        x_axis, y_axis = _orthogonal_xy_from_z(z_axis)

        grasp_pose = torch.eye(4, dtype=torch.float32, device=device).repeat(
            target_pose.shape[0], 1, 1
        )
        grasp_pose[:, :3, 0] = x_axis
        grasp_pose[:, :3, 1] = y_axis
        grasp_pose[:, :3, 2] = z_axis
        local_grasp = torch.tensor(
            self.grasp_position, dtype=torch.float32, device=device
        )
        grasp_pose[:, :3, 3] = (
            torch.matmul(target_pose[:, :3, :3], local_grasp) + target_pose[:, :3, 3]
        )
        return grasp_pose


@dataclass
class SlideAffordance(AntipodalAffordance):
    """Target-local antipodal grasp and translation-axis geometry.

    The positive translation-axis direction denotes approaching and pushing
    the articulated part closed. Pulling moves in the opposite direction.
    The mesh describes the actual graspable contact surface. The target pose is
    supplied separately by :class:`~.goals.SceneEntityPose` or a pose snapshot.
    """

    mesh_vertices: torch.Tensor = field(kw_only=True)
    """Target-local vertices for the graspable contact surface."""

    mesh_triangles: torch.Tensor = field(kw_only=True)
    """Triangle indices for the graspable contact surface."""

    translation_axis: torch.Tensor = field(
        default_factory=lambda: torch.tensor([0.0, 1.0, 0.0])
    )
    """Approach and push/close direction in the articulation-link frame."""

    joint_name: str | None = None
    """Optional stable prismatic-joint name associated with the link."""

    joint_limits: tuple[float, float] | None = None
    """Optional lower and upper translation limits in metres."""

    def __post_init__(self) -> None:
        if self.mesh_vertices.dim() != 2 or self.mesh_vertices.shape[1] != 3:
            raise ValueError("SlideAffordance.mesh_vertices must have shape (N, 3).")
        if (
            self.mesh_vertices.shape[0] == 0
            or not torch.isfinite(self.mesh_vertices).all()
        ):
            raise ValueError(
                "SlideAffordance.mesh_vertices must be finite and non-empty."
            )
        if self.mesh_triangles.dim() != 2 or self.mesh_triangles.shape[1] != 3:
            raise ValueError("SlideAffordance.mesh_triangles must have shape (M, 3).")
        if (
            not isinstance(self.translation_axis, torch.Tensor)
            or self.translation_axis.shape != (3,)
            or not torch.isfinite(self.translation_axis).all()
        ):
            raise ValueError(
                "SlideAffordance.translation_axis must be a finite (3,) tensor."
            )
        if torch.linalg.vector_norm(self.translation_axis) <= 1.0e-6:
            raise ValueError("SlideAffordance.translation_axis must be non-zero.")
        self.translation_axis = self.translation_axis.clone()
        _validate_joint_metadata(self.joint_name, self.joint_limits)


@dataclass
class PressAffordance(Affordance):
    """Explicit target-local contact point and pressing direction."""

    press_axis: torch.Tensor = field(
        default_factory=lambda: torch.tensor([0.0, 0.0, 1.0])
    )
    """Press direction expressed in the target object's local frame."""

    press_position: tuple[float, float, float] = field(kw_only=True)
    """Explicit local-frame point on the pressable contact surface."""

    def __post_init__(self) -> None:
        if (
            not isinstance(self.press_axis, torch.Tensor)
            or self.press_axis.shape != (3,)
            or not torch.isfinite(self.press_axis).all()
        ):
            raise ValueError("PressAffordance.press_axis must be a finite (3,) tensor.")
        if torch.linalg.vector_norm(self.press_axis) <= 1.0e-6:
            raise ValueError("PressAffordance.press_axis must be non-zero.")
        self.press_axis = self.press_axis.clone()
        self.press_position = _validate_local_point(
            self.press_position, "PressAffordance.press_position"
        )

    def get_press_pose(
        self,
        target_pose: torch.Tensor,
        press_position: tuple[float, float, float] | None = None,
    ) -> torch.Tensor:
        """Construct a press pose at the configured surface point.

        The end-effector z-axis follows :attr:`press_axis` in world space. An
        adaptive reference produces an orthonormal, right-handed frame.

        Args:
            target_pose: Current target world pose with shape ``(B, 4, 4)``.
            press_position: Optional per-call exact local-frame press position.
                It overrides :attr:`press_position`.

        Returns:
            Batched world-frame press poses with shape ``(B, 4, 4)``.

        Raises:
            ValueError: If an input has an invalid shape or value.
        """
        if target_pose.dim() != 3 or target_pose.shape[1:] != (4, 4):
            raise ValueError("Target pose must have shape (B, 4, 4).")
        target_pose = target_pose.to(dtype=torch.float32)
        device = target_pose.device
        press_axis = self.press_axis.to(device=device, dtype=torch.float32)
        press_axis = press_axis / torch.linalg.vector_norm(press_axis)
        configured_position = self._validate_press_position(
            press_position,
            field_name="press_position",
        )
        configured_position = (
            self.press_position if configured_position is None else configured_position
        )
        local_press_position = torch.tensor(
            configured_position,
            dtype=torch.float32,
            device=device,
        )

        z_axis = torch.matmul(target_pose[:, :3, :3], press_axis)
        z_axis = torch.nn.functional.normalize(z_axis, dim=1)
        x_axis, y_axis = _orthogonal_xy_from_z(z_axis)

        press_pose = torch.eye(4, dtype=torch.float32, device=device).repeat(
            target_pose.shape[0], 1, 1
        )
        press_pose[:, :3, 0] = x_axis
        press_pose[:, :3, 1] = y_axis
        press_pose[:, :3, 2] = z_axis
        press_pose[:, :3, 3] = (
            torch.matmul(target_pose[:, :3, :3], local_press_position)
            + target_pose[:, :3, 3]
        )
        return press_pose

    @staticmethod
    def _validate_press_position(
        value: tuple[float, float, float] | None,
        *,
        field_name: str,
    ) -> tuple[float, float, float] | None:
        """Validate and normalize an optional local-frame press position."""
        if value is None:
            return None
        position = torch.as_tensor(value, dtype=torch.float32)
        if position.shape != (3,) or not torch.isfinite(position).all():
            raise ValueError(f"{field_name} must be a finite (x, y, z) tuple.")
        return tuple(float(component) for component in position)


def _orthogonal_xy_from_z(z_axis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Complete normalized z axes into right-handed orthonormal frames."""
    basis = torch.eye(3, dtype=z_axis.dtype, device=z_axis.device)
    reference_indices = torch.argmin(torch.abs(z_axis), dim=1)
    reference = basis[reference_indices]
    y_axis = torch.nn.functional.normalize(
        torch.linalg.cross(reference, z_axis, dim=1), dim=1
    )
    x_axis = torch.nn.functional.normalize(
        torch.linalg.cross(y_axis, z_axis, dim=1), dim=1
    )
    return x_axis, y_axis


def _validate_local_point(
    value: tuple[float, float, float], field_name: str
) -> tuple[float, float, float]:
    """Validate and normalize one target-local 3D point."""
    point = torch.as_tensor(value, dtype=torch.float32)
    if point.shape != (3,) or not torch.isfinite(point).all():
        raise ValueError(f"{field_name} must be a finite (x, y, z) tuple.")
    return tuple(float(component) for component in point)


def _validate_joint_metadata(
    joint_name: str | None,
    joint_limits: tuple[float, float] | None,
) -> None:
    """Validate optional articulation joint metadata."""
    if joint_name is not None and (
        not isinstance(joint_name, str) or not joint_name.strip()
    ):
        raise ValueError("joint_name must be a non-empty string when provided.")
    if joint_limits is None:
        return
    limits = torch.as_tensor(joint_limits, dtype=torch.float32)
    if (
        limits.shape != (2,)
        or not torch.isfinite(limits).all()
        or limits[0] > limits[1]
    ):
        raise ValueError("joint_limits must be finite and ordered (lower, upper).")


@dataclass
class InteractionPoints(Affordance):
    """Batch of 3D interaction points on an object surface."""

    points: torch.Tensor = field(default_factory=lambda: torch.zeros(1, 3))
    """Batch of 3D interaction points with shape [B, 3]."""

    normals: torch.Tensor | None = None
    """Optional surface normals at each interaction point with shape [B, 3]."""

    point_types: list[str] = field(default_factory=list)
    """Optional labels for each point's interaction type."""

    def get_points_by_type(self, point_type: str) -> torch.Tensor | None:
        """Get points by their interaction type."""
        if point_type in self.point_types:
            indices = [i for i, t in enumerate(self.point_types) if t == point_type]
            return self.points[indices]
        return None

    def get_batch_size(self) -> int:
        """Return the number of interaction points in this affordance."""
        return self.points.shape[0]

    def get_approach_direction(self, point_idx: int) -> torch.Tensor:
        """Get recommended approach direction for a given point."""
        if self.normals is not None:
            return -self.normals[point_idx]
        return torch.tensor(
            [0, 0, 1], dtype=self.points.dtype, device=self.points.device
        )


@dataclass
class AssembleAffordance(Affordance):
    """Affordance describing how an assemble object fits onto a base object.

    The affordance stores the relative assembly relation. Canonical planning
    supplies the base object's snapshot pose through ``AssembleGoal.base_pose``;
    :attr:`base_object_entity` is retained only as a deprecated direct-core
    fallback when that goal field is omitted. The assemble object's target pose
    is ``base_pose @ assemble_to_base_pose``.
    """

    base_object_label: str = ""
    """Label of the base object the assemble object is placed onto."""

    base_object_entity: BatchEntity | None = None
    """Legacy live base entity used only when ``AssembleGoal.base_pose`` is absent."""

    assemble_object_label: str = ""
    """Label of the assemble object that is picked up and placed."""

    assemble_object_entity: BatchEntity | None = None
    """Optional simulation entity for the assemble object (reference/logging)."""

    assemble_to_base_pose: torch.Tensor = field(
        default_factory=lambda: torch.eye(4, dtype=torch.float32)
    )
    """Pose of the assemble object relative to the base object frame, shape
    ``(4, 4)`` or ``(num_envs, 4, 4)``."""

    def get_assemble_object_pose(self, base_pose: torch.Tensor) -> torch.Tensor:
        """Return the assemble-object target pose for a given base-object pose.

        The assemble object is placed at ``base_pose @ assemble_to_base_pose``.

        Args:
            base_pose: Base-object pose with shape ``(4, 4)`` or ``(num_envs, 4, 4)``.

        Returns:
            Assemble-object target pose with shape ``(num_envs, 4, 4)``.

        Raises:
            TypeError: If either pose value is not a tensor.
            ValueError: If either pose has an unsupported shape or batch size.
        """
        if not isinstance(base_pose, torch.Tensor):
            raise TypeError("base_pose must be a torch.Tensor.")
        base_pose = base_pose.to(dtype=torch.float32)
        if base_pose.shape == (4, 4):
            base_pose = base_pose.unsqueeze(0)
        elif (
            base_pose.dim() != 3
            or base_pose.shape[0] == 0
            or base_pose.shape[-2:] != (4, 4)
        ):
            raise ValueError("base_pose must have shape (4, 4) or (num_envs, 4, 4).")
        num_envs = base_pose.shape[0]
        if not isinstance(self.assemble_to_base_pose, torch.Tensor):
            raise TypeError("assemble_to_base_pose must be a torch.Tensor.")
        rel = self.assemble_to_base_pose.to(
            device=base_pose.device, dtype=torch.float32
        )
        if rel.shape == (4, 4):
            rel = rel.unsqueeze(0).repeat(num_envs, 1, 1)
        elif rel.dim() != 3 or rel.shape[-2:] != (4, 4) or rel.shape[0] == 0:
            raise ValueError(
                "assemble_to_base_pose must have shape (4, 4), (1, 4, 4), "
                "or (num_envs, 4, 4)."
            )
        elif rel.shape[0] == 1:
            rel = rel.repeat(num_envs, 1, 1)
        elif rel.shape[0] != num_envs:
            raise ValueError(
                "assemble_to_base_pose batch size must match base_pose batch size."
            )
        return torch.bmm(base_pose, rel)


__all__ = [
    "Affordance",
    "AntipodalAffordance",
    "AxisAlignAffordance",
    "SlideAffordance",
    "PressAffordance",
    "TwistAffordance",
    "InteractionPoints",
    "AssembleAffordance",
]
