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

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

import torch

_TARGET_LINK_POINT_CLOUD_KEY = "target_link_point_cloud"
_ARTICULATION_POINT_CLOUD_KEY = "articulation_point_cloud"
_TARGET_LINK_PRISMATIC_JOINT_AXIS_KEY = "target_link_prismatic_joint_axis"
_TARGET_LINK_REVOLUTE_JOINT_AXIS_KEY = "target_link_revolute_joint_axis"
_TARGET_LINK_REVOLUTE_AXIS_ORIGIN_KEY = "target_link_revolute_axis_origin"


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

    def resolve_from_object_geometry(self, geometry: Mapping[str, Any]) -> None:
        """Resolve geometry-derived fields after object semantics are assembled.

        Subclasses may override this hook when their derived semantic values
        require metadata owned by ``ObjectSemantics.geometry``.

        Args:
            geometry: Non-affordance object geometry metadata.
        """
        del geometry


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

    MAX_SURFACE_POINT_COUNT: ClassVar[int] = 1000
    """Maximum point-cloud size used for geometry-distribution analysis."""

    def __post_init__(self) -> None:
        """Validate optional target-local geometry without owning a generator."""
        if self.mesh_vertices is None and self.mesh_triangles is None:
            return
        if self.mesh_vertices is None or self.mesh_triangles is None:
            raise ValueError(
                "mesh_vertices and mesh_triangles must be provided together."
            )
        if (
            not isinstance(self.mesh_vertices, torch.Tensor)
            or not self.mesh_vertices.is_floating_point()
            or self.mesh_vertices.dim() != 2
            or self.mesh_vertices.shape[1] != 3
            or self.mesh_vertices.shape[0] == 0
            or not bool(torch.isfinite(self.mesh_vertices).all().item())
        ):
            raise ValueError(
                "AntipodalAffordance.mesh_vertices must be a non-empty finite "
                "floating tensor with shape (N, 3)."
            )
        if (
            not isinstance(self.mesh_triangles, torch.Tensor)
            or self.mesh_triangles.dtype == torch.bool
            or self.mesh_triangles.is_floating_point()
            or self.mesh_triangles.dim() != 2
            or self.mesh_triangles.shape[1] != 3
            or self.mesh_triangles.shape[0] == 0
        ):
            raise ValueError(
                "AntipodalAffordance.mesh_triangles must be a non-empty integer "
                "tensor with shape (M, 3)."
            )
        if self.mesh_vertices.device != self.mesh_triangles.device:
            raise ValueError("Antipodal affordance mesh tensors must share a device.")
        if (
            bool((self.mesh_triangles < 0).any().item())
            or int(self.mesh_triangles.max().item()) >= self.mesh_vertices.shape[0]
        ):
            raise ValueError(
                "AntipodalAffordance.mesh_triangles reference invalid vertices."
            )

    def sample_surface_points(self, max_points: int = 1000) -> torch.Tensor:
        """Deterministically sample target-local mesh-surface points.

        Args:
            max_points: Requested point cap in ``[1, 1000]``.

        Returns:
            Target-local surface points with shape ``(N, 3)``.
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
        triangles = self.mesh_triangles
        if triangles is None or triangles.numel() == 0:
            return self._evenly_subsample_points(vertices, max_points)
        triangles = triangles.to(device=vertices.device, dtype=torch.long)

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
        """Return the widest surface-point distribution axis in world space."""
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


@dataclass
class AxisAlignAffordance(AntipodalAffordance):
    """Antipodal grasp affordance with an object-local alignment axis."""

    internal_axis: torch.Tensor = field(
        default_factory=lambda: torch.tensor([0.0, 0.0, 1.0])
    )
    """Axis expressed in the target object's local frame."""

    def __post_init__(self) -> None:
        AntipodalAffordance.__post_init__(self)
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
    """Target-local grasp point and parent-joint rotation geometry."""

    grasp_position: tuple[float, float, float] = field(kw_only=True)
    """Explicit target-local center of the gripper contact region."""

    axis_origin: tuple[float, float, float] | None = field(
        default=None,
        kw_only=True,
    )
    """Fallback axis point, overridden by revolute-joint origin metadata."""

    twist_axis: torch.Tensor = field(
        default_factory=lambda: torch.tensor([0.0, 1.0, 0.0])
    )
    """Parent revolute-joint axis, signed toward articulation geometry."""

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
        if self.axis_origin is not None:
            self.axis_origin = _validate_local_point(
                self.axis_origin, "TwistAffordance.axis_origin"
            )
        _validate_joint_metadata(self.joint_name, self.joint_limits)

    def resolve_from_object_geometry(self, geometry: Mapping[str, Any]) -> None:
        """Resolve the target-local revolute axis, sign, and joint origin."""
        resolved = _infer_articulation_neighborhood_axis(
            geometry,
            field_name="TwistAffordance.twist_axis",
            axis_key=_TARGET_LINK_REVOLUTE_JOINT_AXIS_KEY,
        )
        if resolved is not None:
            self.twist_axis = resolved[0]

        resolved_origin = _resolve_geometry_local_point(
            geometry,
            key=_TARGET_LINK_REVOLUTE_AXIS_ORIGIN_KEY,
        )
        if resolved_origin is not None:
            self.axis_origin = resolved_origin

    def require_axis_origin(self) -> tuple[float, float, float]:
        """Return the explicit or geometry-derived local rotation-axis point.

        Returns:
            Resolved target-local rotation-axis origin.

        Raises:
            ValueError: If neither a fallback nor articulation geometry supplied
                the rotation-axis origin.
        """
        if self.axis_origin is None:
            raise ValueError(
                "TwistAffordance.axis_origin must be provided explicitly or "
                "resolved from geometry['target_link_revolute_axis_origin']; "
                "the target link's revolute joint origin is missing."
            )
        return self.axis_origin

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
    """Target-local antipodal grasp and parent-joint translation geometry.

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
    """Parent prismatic-joint axis, signed toward articulation geometry."""

    joint_name: str | None = None
    """Optional stable prismatic-joint name associated with the link."""

    joint_limits: tuple[float, float] | None = None
    """Optional lower and upper translation limits in metres."""

    def __post_init__(self) -> None:
        AntipodalAffordance.__post_init__(self)
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

    def resolve_from_object_geometry(self, geometry: Mapping[str, Any]) -> None:
        """Resolve the target-local prismatic axis and neighborhood sign."""
        resolved = _infer_articulation_neighborhood_axis(
            geometry,
            field_name="SlideAffordance.translation_axis",
            axis_key=_TARGET_LINK_PRISMATIC_JOINT_AXIS_KEY,
        )
        if resolved is not None:
            self.translation_axis = resolved[0]


@dataclass
class OpenDoorAffordance(AntipodalAffordance):
    """Target-local handle geometry and a resolved hinge axis.

    Use :meth:`from_articulation` to start from a graspable handle link. The
    factory consumes the articulation's public parent-joint chain, skips only
    fixed joints, and automatically accepts one unambiguous active revolute
    ancestor. Ambiguous mechanisms require an explicit hinge joint name. The
    resulting affordance owns no simulator entity or live pose.
    """

    mesh_vertices: torch.Tensor = field(kw_only=True)
    """Handle-local vertices for the graspable contact surface."""

    mesh_triangles: torch.Tensor = field(kw_only=True)
    """Triangle indices for the graspable contact surface."""

    rotation_axis: torch.Tensor = field(kw_only=True)
    """Resolved hinge axis expressed in the handle-link frame."""

    axis_origin: tuple[float, float, float] = field(kw_only=True)
    """Resolved point on the hinge axis in the handle-link frame."""

    joint_name: str = field(kw_only=True)
    """Stable name of the resolved parent revolute joint."""

    joint_limits: tuple[float, float] | None = None
    """Optional lower and upper hinge limits in radians."""

    opening_direction: int = 1
    """Joint-coordinate direction from the closed limit toward the open limit."""

    def __post_init__(self) -> None:
        AntipodalAffordance.__post_init__(self)
        if (
            not isinstance(self.rotation_axis, torch.Tensor)
            or self.rotation_axis.shape != (3,)
            or not torch.isfinite(self.rotation_axis).all()
        ):
            raise ValueError(
                "OpenDoorAffordance.rotation_axis must be a finite (3,) tensor."
            )
        if torch.linalg.vector_norm(self.rotation_axis) <= 1.0e-6:
            raise ValueError("OpenDoorAffordance.rotation_axis must be non-zero.")
        self.rotation_axis = self.rotation_axis.clone()
        self.axis_origin = _validate_local_point(
            self.axis_origin, "OpenDoorAffordance.axis_origin"
        )
        _validate_joint_metadata(self.joint_name, self.joint_limits)
        if type(self.opening_direction) is not int or self.opening_direction not in (
            -1,
            1,
        ):
            raise ValueError("opening_direction must be either -1 or 1.")

    @classmethod
    def from_articulation(
        cls,
        articulation: Articulation,
        link_name: str,
        *,
        hinge_joint_name: str | None = None,
        opening_direction: int = 1,
    ) -> OpenDoorAffordance:
        """Build handle semantics from a parent revolute joint.

        Automatic resolution skips only fixed joints. It succeeds when the
        handle chain contains exactly one active ancestor and that joint is
        revolute. A chain with multiple active ancestors can represent a latch,
        handle joint, or another mechanism and therefore requires an explicit
        ``hinge_joint_name``. Hinge geometry is converted from the joint frame
        into the requested handle-link frame using current public link poses.

        Args:
            articulation: Articulation containing the graspable handle link.
            link_name: Graspable handle link from which to start the traversal.
            hinge_joint_name: Optional explicit revolute ancestor. Required
                when more than one active ancestor makes automatic resolution
                ambiguous.
            opening_direction: Joint-coordinate direction from the closed legal
                endpoint toward the open endpoint. Defaults to increasing qpos.

        Returns:
            Pure target-local handle and hinge semantics.

        Raises:
            TypeError: If a supplied name is not a string.
            ValueError: If the link or explicit joint is unknown, automatic
                resolution is ambiguous, the selected joint is not revolute,
                or the resolved joint geometry is invalid.
        """
        if type(link_name) is not str:
            raise TypeError("link_name must be a string.")
        if not link_name or link_name != link_name.strip():
            raise ValueError("link_name must be non-empty.")
        if hinge_joint_name is not None and type(hinge_joint_name) is not str:
            raise TypeError("hinge_joint_name must be a string or None.")
        if hinge_joint_name is not None and (
            not hinge_joint_name or hinge_joint_name != hinge_joint_name.strip()
        ):
            raise ValueError("hinge_joint_name must be non-empty when provided.")

        parent_chain = articulation.get_parent_joint_chain(link_name)
        if not parent_chain:
            raise ValueError(
                f"Link {link_name!r} has no parent joint and cannot resolve a hinge."
            )

        if hinge_joint_name is not None:
            hinge = next(
                (joint for joint in parent_chain if joint.name == hinge_joint_name),
                None,
            )
            if hinge is None:
                available = [joint.name for joint in parent_chain]
                raise ValueError(
                    f"Joint {hinge_joint_name!r} is not an ancestor of link "
                    f"{link_name!r}. Available ancestors: {available}."
                )
        else:
            active_ancestors = tuple(
                joint for joint in parent_chain if joint.joint_type != "fixed"
            )
            if not active_ancestors:
                raise ValueError(
                    f"No active parent joint found for link {link_name!r}."
                )
            if len(active_ancestors) != 1:
                active_names = [joint.name for joint in active_ancestors]
                raise ValueError(
                    f"Ambiguous active ancestors {active_names} for link "
                    f"{link_name!r}; provide hinge_joint_name explicitly."
                )
            hinge = active_ancestors[0]
        if hinge.joint_type != "revolute":
            raise ValueError(
                f"Selected hinge joint {hinge.name!r} must be revolute, got "
                f"{hinge.joint_type!r}."
            )
        if hinge.joint_limits is None:
            raise ValueError(
                f"Selected hinge joint {hinge.name!r} must expose position limits."
            )

        handle_pose = articulation.get_link_pose(
            link_name, env_ids=[0], to_matrix=True
        )[0].to(dtype=torch.float32)
        parent_pose = articulation.get_link_pose(
            hinge.parent_link_name, env_ids=[0], to_matrix=True
        )[0].to(device=handle_pose.device, dtype=torch.float32)
        joint_origin = hinge.origin_pose.to(
            device=handle_pose.device,
            dtype=torch.float32,
        )
        joint_axis = hinge.axis.to(
            device=handle_pose.device,
            dtype=torch.float32,
        )
        if joint_origin.shape != (4, 4):
            raise ValueError(
                "Resolved revolute joint origin_pose must have shape (4, 4)."
            )
        if joint_axis.shape != (3,) or not torch.isfinite(joint_axis).all():
            raise ValueError(
                "Resolved revolute joint axis must be a finite (3,) vector."
            )
        if torch.linalg.vector_norm(joint_axis) <= 1.0e-6:
            raise ValueError("Resolved revolute joint axis must be non-zero.")

        joint_pose_world = torch.matmul(parent_pose, joint_origin)
        handle_rotation_world = handle_pose[:3, :3]
        rotation_axis_world = torch.matmul(
            joint_pose_world[:3, :3],
            joint_axis / torch.linalg.vector_norm(joint_axis),
        )
        rotation_axis_local = torch.matmul(
            handle_rotation_world.transpose(0, 1), rotation_axis_world
        )
        axis_origin_local = torch.matmul(
            handle_rotation_world.transpose(0, 1),
            joint_pose_world[:3, 3] - handle_pose[:3, 3],
        )
        vertices, triangles = articulation.get_link_vert_face(link_name)
        lower_limit, upper_limit = hinge.joint_limits
        return cls(
            mesh_vertices=vertices,
            mesh_triangles=triangles,
            rotation_axis=rotation_axis_local,
            axis_origin=tuple(float(value) for value in axis_origin_local),
            joint_name=hinge.name,
            joint_limits=(lower_limit, upper_limit),
            opening_direction=opening_direction,
        )


@dataclass
class PressAffordance(Affordance):
    """Target-local contact point and parent-joint pressing geometry."""

    press_axis: torch.Tensor = field(
        default_factory=lambda: torch.tensor([0.0, 0.0, 1.0])
    )
    """Parent prismatic-joint axis, signed toward articulation geometry."""

    press_position: tuple[float, float, float] | None = field(
        default=None,
        kw_only=True,
    )
    """Local contact point; inferred from articulation geometry when omitted."""

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
        if self.press_position is not None:
            self.press_position = _validate_local_point(
                self.press_position, "PressAffordance.press_position"
            )

    def resolve_from_object_geometry(self, geometry: Mapping[str, Any]) -> None:
        """Resolve the target-local prismatic axis, sign, and contact point."""
        resolved = _infer_articulation_neighborhood_axis(
            geometry,
            field_name="PressAffordance.press_axis",
            axis_key=_TARGET_LINK_PRISMATIC_JOINT_AXIS_KEY,
        )
        if resolved is None:
            return
        self.press_axis, target_points = resolved
        if self.press_position is None:
            self.press_position = _outer_surface_center(
                target_points,
                self.press_axis,
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
        if configured_position is None:
            raise ValueError(
                "PressAffordance.press_position must be provided explicitly or "
                "resolved from articulation joint geometry."
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


def _infer_articulation_neighborhood_axis(
    geometry: Mapping[str, Any],
    *,
    field_name: str,
    axis_key: str,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Resolve one signed joint axis from target-centered local geometry.

    The target-link point-cloud center defines a spherical neighborhood in the
    complete articulation cloud. Its radius is twice the target cloud's maximum
    distance from that center. The neighborhood-center offset disambiguates the
    sign of the normalized joint axis supplied in target-link coordinates.

    Args:
        geometry: Object geometry containing joint-axis and point-cloud entries.
        field_name: Affordance field name used in validation errors.
        axis_key: Geometry key containing the target-local joint axis.

    Returns:
        ``(axis, target_points)`` when joint-axis metadata is present, otherwise
        ``None`` so legacy and non-articulation affordances retain their axis.

    Raises:
        TypeError: If geometry, a point cloud, or the joint axis has an invalid
            type.
        ValueError: If geometry metadata is incomplete, malformed,
            geometrically degenerate, or directionally ambiguous.
    """
    if not isinstance(geometry, Mapping):
        raise TypeError("geometry must be a mapping.")
    has_target = _TARGET_LINK_POINT_CLOUD_KEY in geometry
    has_articulation = _ARTICULATION_POINT_CLOUD_KEY in geometry
    has_axis = axis_key in geometry
    if not has_target and not has_articulation and not has_axis:
        return None
    if not has_target or not has_articulation or not has_axis:
        raise ValueError(
            f"{field_name} inference requires "
            f"{_TARGET_LINK_POINT_CLOUD_KEY!r}, "
            f"{_ARTICULATION_POINT_CLOUD_KEY!r}, and {axis_key!r}."
        )

    target_points = _validate_local_point_cloud(
        geometry[_TARGET_LINK_POINT_CLOUD_KEY],
        field_name=f"geometry[{_TARGET_LINK_POINT_CLOUD_KEY!r}]",
    )
    articulation_points = _validate_local_point_cloud(
        geometry[_ARTICULATION_POINT_CLOUD_KEY],
        field_name=f"geometry[{_ARTICULATION_POINT_CLOUD_KEY!r}]",
    )
    if target_points.device != articulation_points.device:
        raise ValueError(
            "Articulation and target-link point clouds must share a device."
        )
    joint_axis = _validate_geometry_axis(
        geometry[axis_key],
        field_name=f"geometry[{axis_key!r}]",
    )
    if joint_axis.device != target_points.device:
        raise ValueError("Joint axis and point clouds must share a device.")
    target_points = target_points.to(dtype=torch.float32)
    articulation_points = articulation_points.to(dtype=torch.float32)
    joint_axis = joint_axis.to(dtype=torch.float32)
    joint_axis = joint_axis / torch.linalg.vector_norm(joint_axis)

    target_center = target_points.mean(dim=0)
    target_distances = torch.linalg.vector_norm(
        target_points - target_center,
        dim=1,
    )
    target_radius = target_distances.max()
    if float(target_radius.item()) <= 1.0e-8:
        raise ValueError(
            f"{field_name} cannot be inferred from a degenerate target-link "
            "point cloud."
        )

    neighborhood_radius = target_radius * 2.0
    neighborhood_mask = (
        torch.linalg.vector_norm(
            articulation_points - target_center,
            dim=1,
        )
        <= neighborhood_radius
    )
    if not bool(neighborhood_mask.any().item()):
        raise ValueError(f"{field_name} point-cloud neighborhood is empty.")
    neighborhood_center = articulation_points[neighborhood_mask].mean(dim=0)
    center_offset = neighborhood_center - target_center
    direction_score = torch.dot(center_offset, joint_axis)
    offset_tolerance = max(1.0e-8, float(target_radius.item()) * 1.0e-6)
    if abs(float(direction_score.item())) <= offset_tolerance:
        raise ValueError(
            f"{field_name} direction is ambiguous because the articulation "
            "neighborhood-center offset is orthogonal to the joint axis."
        )

    if float(direction_score.item()) < 0.0:
        joint_axis = -joint_axis
    return joint_axis, target_points


def _validate_local_point_cloud(value: Any, *, field_name: str) -> torch.Tensor:
    """Validate one non-empty finite floating point cloud."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{field_name} must be a torch.Tensor.")
    if (
        not value.is_floating_point()
        or value.dim() != 2
        or value.shape[1:] != (3,)
        or value.shape[0] == 0
        or not bool(torch.isfinite(value).all().item())
    ):
        raise ValueError(
            f"{field_name} must be a non-empty finite floating tensor with "
            "shape (N, 3)."
        )
    return value


def _validate_geometry_axis(value: Any, *, field_name: str) -> torch.Tensor:
    """Validate one finite, non-zero floating geometry axis."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{field_name} must be a torch.Tensor.")
    if (
        not value.is_floating_point()
        or value.shape != (3,)
        or not bool(torch.isfinite(value).all().item())
    ):
        raise ValueError(
            f"{field_name} must be a finite floating tensor with shape (3,)."
        )
    if float(torch.linalg.vector_norm(value).item()) <= 1.0e-6:
        raise ValueError(f"{field_name} must be non-zero.")
    return value


def _resolve_geometry_local_point(
    geometry: Mapping[str, Any],
    *,
    key: str,
) -> tuple[float, float, float] | None:
    """Resolve an optional finite target-local point tensor from geometry."""
    if key not in geometry:
        return None
    value = geometry[key]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"geometry[{key!r}] must be a torch.Tensor.")
    if (
        not value.is_floating_point()
        or value.shape != (3,)
        or not bool(torch.isfinite(value).all().item())
    ):
        raise ValueError(
            f"geometry[{key!r}] must be a finite floating tensor with shape (3,)."
        )
    return tuple(float(component) for component in value)


def _outer_surface_center(
    target_points: torch.Tensor,
    inward_axis: torch.Tensor,
) -> tuple[float, float, float]:
    """Return the sampled outer-surface center opposite an inward axis."""
    axis = inward_axis.to(device=target_points.device, dtype=torch.float32)
    projections = torch.matmul(target_points, axis)
    minimum = projections.min()
    radius = torch.linalg.vector_norm(
        target_points - target_points.mean(dim=0),
        dim=1,
    ).max()
    tolerance = max(1.0e-5, float(radius.item()) * 1.0e-4)
    surface_points = target_points[projections <= minimum + tolerance]
    center = surface_points.mean(dim=0)
    return tuple(float(component) for component in center)


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

    The affordance stores the relative assembly relation. Planning supplies the
    base object's snapshot pose through ``AssembleGoal.base_pose``. The assemble
    object's target pose is ``base_pose @ assemble_to_base_pose``.
    """

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
    "OpenDoorAffordance",
    "SlideAffordance",
    "PressAffordance",
    "TwistAffordance",
    "InteractionPoints",
    "AssembleAffordance",
]
