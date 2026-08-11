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

import torch
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any, TYPE_CHECKING

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
    """Antipodal grasp affordance for parallel-jaw grippers."""

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
        object_part: str = "center",
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if self._generator is None:
            self._init_generator()
        approach_direction = self._resolve_approach_direction(approach_direction)
        results = []
        for i, obj_pose in enumerate(obj_poses):
            is_success, grasp_poses, _, costs = self._generator.get_valid_grasp_poses(
                object_pose=obj_pose,
                approach_direction=approach_direction,
                object_part=object_part,
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
        if self._generator is None:
            self._init_generator()
        approach_direction = self._resolve_approach_direction(approach_direction)
        grasp_xpos_list: list[torch.Tensor] = []
        is_success_list: list[bool] = []
        open_length_list: list[float] = []
        for i, obj_pose in enumerate(obj_poses):
            is_success, grasp_xpos, open_length = self._generator.get_grasp_poses(
                obj_pose, approach_direction
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


def _owned_se3_offset(value: torch.Tensor, *, field_name: str) -> torch.Tensor:
    """Validate and own one affordance-local homogeneous transform."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{field_name} must be a torch.Tensor.")
    if value.shape != (4, 4):
        raise ValueError(f"{field_name} must have shape (4, 4).")
    if not value.is_floating_point():
        raise TypeError(f"{field_name} must use a floating-point dtype.")
    if not torch.isfinite(value).all():
        raise ValueError(f"{field_name} must contain only finite values.")
    checked = value.to(dtype=torch.float64)
    bottom = checked.new_tensor((0.0, 0.0, 0.0, 1.0))
    if not torch.allclose(checked[3], bottom, atol=1.0e-6, rtol=0.0):
        raise ValueError(f"{field_name} must have bottom row [0, 0, 0, 1].")
    rotation = checked[:3, :3]
    if not torch.allclose(
        rotation.T @ rotation,
        torch.eye(3, dtype=checked.dtype, device=checked.device),
        atol=1.0e-6,
        rtol=0.0,
    ) or not torch.isclose(
        torch.linalg.det(rotation),
        checked.new_tensor(1.0),
        atol=1.0e-6,
        rtol=0.0,
    ):
        raise ValueError(f"{field_name} must contain a proper SE(3) rotation.")
    return value.clone()


def _finite_scalar(value: float, *, field_name: str) -> float:
    """Return one finite non-boolean scalar as a float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a finite scalar.")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite.")
    return normalized


@dataclass(frozen=True, slots=True)
class ArticulationOperationTarget:
    """Named joint target and handle-relative operation displacement.

    ``displacement`` is deliberately explicit: it is the full signed handle
    stroke from the live source joint position captured during semantic
    grounding to ``target_position``. Recovery replans scale this stroke by
    the remaining live joint progress.
    """

    target_position: float
    """Absolute desired articulation joint position."""

    displacement: float
    """Signed operation displacement from the currently observed handle pose."""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "target_position",
            _finite_scalar(
                self.target_position,
                field_name="ArticulationOperationTarget.target_position",
            ),
        )
        object.__setattr__(
            self,
            "displacement",
            _finite_scalar(
                self.displacement,
                field_name="ArticulationOperationTarget.displacement",
            ),
        )

    def snapshot(self) -> ArticulationOperationTarget:
        """Return an independently constructed immutable target."""
        return ArticulationOperationTarget(self.target_position, self.displacement)


@dataclass(eq=False)
class ArticulationOperationAffordance(Affordance):
    """Declarative handle geometry for one articulated joint operation.

    The four offsets are expressed in the live handle frame. During semantic
    grounding the approach and contact poses are ``handle @ offset``. The
    operation and retract poses additionally insert a local translation of
    ``operation_axis * displacement * position_scale`` before their offsets.
    This keeps task code free of pose-matrix construction; the semantic
    compiler copies the geometry into a late-bound atomic goal.
    """

    joint_id: str = ""
    """Canonical joint identifier written to the atomic goal and effect."""

    approach_offset: torch.Tensor = field(
        default_factory=lambda: torch.eye(4, dtype=torch.float32)
    )
    contact_offset: torch.Tensor = field(
        default_factory=lambda: torch.eye(4, dtype=torch.float32)
    )
    operation_offset: torch.Tensor = field(
        default_factory=lambda: torch.eye(4, dtype=torch.float32)
    )
    retract_offset: torch.Tensor = field(
        default_factory=lambda: torch.eye(4, dtype=torch.float32)
    )
    operation_axis: torch.Tensor = field(
        default_factory=lambda: torch.tensor((1.0, 0.0, 0.0), dtype=torch.float32)
    )
    """Unit operation direction expressed in the observed handle frame."""

    position_scale: float = 1.0
    """Positive conversion from declared displacement units to pose metres."""

    semantic_targets: Mapping[str, ArticulationOperationTarget] = field(
        default_factory=dict
    )
    """Optional stable target names mapped to position/displacement pairs."""

    def __post_init__(self) -> None:
        if (
            type(self.joint_id) is not str
            or not self.joint_id
            or self.joint_id != self.joint_id.strip()
        ):
            raise ValueError(
                "ArticulationOperationAffordance.joint_id must be a non-empty "
                "canonical identifier."
            )
        for field_name in (
            "approach_offset",
            "contact_offset",
            "operation_offset",
            "retract_offset",
        ):
            setattr(
                self,
                field_name,
                _owned_se3_offset(
                    getattr(self, field_name),
                    field_name=f"ArticulationOperationAffordance.{field_name}",
                ),
            )
        axis = self.operation_axis
        if not isinstance(axis, torch.Tensor):
            raise TypeError(
                "ArticulationOperationAffordance.operation_axis must be a tensor."
            )
        if axis.shape != (3,) or not axis.is_floating_point():
            raise ValueError(
                "ArticulationOperationAffordance.operation_axis must be a "
                "floating tensor with shape (3,)."
            )
        if not torch.isfinite(axis).all():
            raise ValueError(
                "ArticulationOperationAffordance.operation_axis must be finite."
            )
        norm = torch.linalg.vector_norm(axis)
        if float(norm) <= torch.finfo(axis.dtype).eps:
            raise ValueError(
                "ArticulationOperationAffordance.operation_axis must be non-zero."
            )
        self.operation_axis = (axis / norm).clone()
        self.position_scale = _finite_scalar(
            self.position_scale,
            field_name="ArticulationOperationAffordance.position_scale",
        )
        if self.position_scale <= 0.0:
            raise ValueError(
                "ArticulationOperationAffordance.position_scale must be positive."
            )
        if not isinstance(self.semantic_targets, Mapping):
            raise TypeError(
                "ArticulationOperationAffordance.semantic_targets must be a mapping."
            )
        targets: dict[str, ArticulationOperationTarget] = {}
        for target_id, target in self.semantic_targets.items():
            if (
                type(target_id) is not str
                or not target_id
                or target_id != target_id.strip()
            ):
                raise ValueError(
                    "Articulation operation target IDs must be non-empty canonical "
                    "identifiers."
                )
            if type(target) is not ArticulationOperationTarget:
                raise TypeError(
                    "semantic_targets values must be exact "
                    "ArticulationOperationTarget values."
                )
            targets[target_id] = target.snapshot()
        self.semantic_targets = MappingProxyType(targets)

    def resolve_target(self, target_id: str) -> ArticulationOperationTarget:
        """Return an owned named target or raise with deterministic candidates."""
        if type(target_id) is not str or not target_id:
            raise ValueError("target_id must be a non-empty string.")
        try:
            target = self.semantic_targets[target_id]
        except KeyError as exc:
            raise KeyError(
                f"Unknown articulation target {target_id!r}; available targets are "
                f"{sorted(self.semantic_targets)}."
            ) from exc
        return target.snapshot()

    def __deepcopy__(self, memo: dict[int, object]) -> ArticulationOperationAffordance:
        """Copy immutable configuration despite ``MappingProxyType`` storage."""
        existing = memo.get(id(self))
        if existing is not None:
            assert isinstance(existing, ArticulationOperationAffordance)
            return existing
        copied = ArticulationOperationAffordance(
            object_label=self.object_label,
            custom_config=deepcopy(self.custom_config, memo),
            joint_id=self.joint_id,
            approach_offset=self.approach_offset,
            contact_offset=self.contact_offset,
            operation_offset=self.operation_offset,
            retract_offset=self.retract_offset,
            operation_axis=self.operation_axis,
            position_scale=self.position_scale,
            semantic_targets={
                target_id: target.snapshot()
                for target_id, target in self.semantic_targets.items()
            },
        )
        memo[id(self)] = copied
        return copied

    def ground_poses(
        self,
        handle_pose: torch.Tensor,
        *,
        displacement: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Ground four end-effector poses from a fresh handle observation.

        Args:
            handle_pose: Live handle pose with shape ``(4, 4)`` or ``(B, 4, 4)``.
            displacement: Signed displacement from this observed handle pose.

        Returns:
            Approach, contact, operation, and retract pose batches.
        """
        if not isinstance(handle_pose, torch.Tensor):
            raise TypeError("handle_pose must be a torch.Tensor.")
        if handle_pose.shape == (4, 4):
            handles = handle_pose.unsqueeze(0)
        elif (
            handle_pose.dim() == 3
            and handle_pose.shape[0] > 0
            and handle_pose.shape[-2:] == (4, 4)
        ):
            handles = handle_pose
        else:
            raise ValueError("handle_pose must have shape (4, 4) or (B, 4, 4).")
        if not handle_pose.is_floating_point() or not torch.isfinite(handle_pose).all():
            raise ValueError("handle_pose must be a finite floating tensor.")
        displacement = _finite_scalar(displacement, field_name="displacement")
        offsets = tuple(
            getattr(self, field_name)
            .to(
                device=handles.device,
                dtype=handles.dtype,
            )
            .unsqueeze(0)
            .expand(handles.shape[0], -1, -1)
            for field_name in (
                "approach_offset",
                "contact_offset",
                "operation_offset",
                "retract_offset",
            )
        )
        translation = (
            torch.eye(
                4,
                dtype=handles.dtype,
                device=handles.device,
            )
            .unsqueeze(0)
            .repeat(handles.shape[0], 1, 1)
        )
        translation[:, :3, 3] = self.operation_axis.to(
            device=handles.device,
            dtype=handles.dtype,
        ) * (displacement * self.position_scale)
        approach = torch.bmm(handles, offsets[0])
        contact = torch.bmm(handles, offsets[1])
        moved_handle = torch.bmm(handles, translation)
        operation = torch.bmm(moved_handle, offsets[2])
        retract = torch.bmm(moved_handle, offsets[3])
        return tuple(pose.clone() for pose in (approach, contact, operation, retract))


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
    ``(4, 4)`` or ``(n_envs, 4, 4)``."""

    def get_assemble_object_pose(self, base_pose: torch.Tensor) -> torch.Tensor:
        """Return the assemble-object target pose for a given base-object pose.

        The assemble object is placed at ``base_pose @ assemble_to_base_pose``.

        Args:
            base_pose: Base-object pose with shape ``(4, 4)`` or ``(n_envs, 4, 4)``.

        Returns:
            Assemble-object target pose with shape ``(n_envs, 4, 4)``.

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
            raise ValueError("base_pose must have shape (4, 4) or (n_envs, 4, 4).")
        n_envs = base_pose.shape[0]
        if not isinstance(self.assemble_to_base_pose, torch.Tensor):
            raise TypeError("assemble_to_base_pose must be a torch.Tensor.")
        rel = self.assemble_to_base_pose.to(
            device=base_pose.device, dtype=torch.float32
        )
        if rel.shape == (4, 4):
            rel = rel.unsqueeze(0).repeat(n_envs, 1, 1)
        elif rel.dim() != 3 or rel.shape[-2:] != (4, 4) or rel.shape[0] == 0:
            raise ValueError(
                "assemble_to_base_pose must have shape (4, 4), (1, 4, 4), "
                "or (n_envs, 4, 4)."
            )
        elif rel.shape[0] == 1:
            rel = rel.repeat(n_envs, 1, 1)
        elif rel.shape[0] != n_envs:
            raise ValueError(
                "assemble_to_base_pose batch size must match base_pose batch size."
            )
        return torch.bmm(base_pose, rel)


__all__ = [
    "Affordance",
    "AntipodalAffordance",
    "ArticulationOperationAffordance",
    "ArticulationOperationTarget",
    "InteractionPoints",
    "AssembleAffordance",
]
