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

"""Immutable target-local grasp constraints without Task Program identities."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from collections.abc import Mapping
import hashlib
import math
import torch
from embodichain.compute.geometry.gripper import full_object_opening_mask
from embodichain.utils.math import pose_inv
from embodichain.toolkits.graspkit.pose_generator import ParallelJawGraspPoseGenerator
from embodichain.toolkits.graspkit.hand_envelope import NominalHandEnvelope

__all__ = [
    "GraspGeometryConstraint",
    "grasp_geometry_digest",
    "ConstrainedGraspPoseGenerator",
]


def grasp_geometry_digest(vertices: torch.Tensor, triangles: torch.Tensor) -> str:
    """Bind a rule to immutable local geometry using the source digest convention.

    Args:
        vertices: Source object-local mesh vertices.
        triangles: Corresponding source topology indices.

    Returns:
        SHA-256 over float32 vertices/int64 faces, shapes and ordered bytes.
    """
    digest = hashlib.sha256()
    for value, dtype in ((vertices, torch.float32), (triangles, torch.int64)):
        data = value.detach().to(device="cpu", dtype=dtype).contiguous()
        digest.update(str(tuple(data.shape)).encode("ascii"))
        digest.update(data.numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class GraspGeometryConstraint:
    """Object-local region and release-plane operands, not skill/recovery policy."""

    geometry_digest: str
    local_axis: tuple[float, float, float]
    midpoint: float
    upper_half: bool
    release_pose: tuple[float, ...] | None = None
    plane_z: float | None = None
    margin: float = 0.0
    hand_envelope: NominalHandEnvelope | None = None
    model_id: str | None = None

    def __post_init__(self) -> None:
        if (
            type(self.geometry_digest) is not str
            or len(self.geometry_digest) != 64
            or any(c not in "0123456789abcdef" for c in self.geometry_digest)
        ):
            raise ValueError("Grasp constraint requires its source geometry digest.")
        if type(self.upper_half) is not bool:
            raise TypeError("upper_half must be bool.")
        if (
            type(self.local_axis) is not tuple
            or len(self.local_axis) != 3
            or any(
                type(x) not in (int, float) or not math.isfinite(x)
                for x in self.local_axis
            )
            or not any(self.local_axis)
        ):
            raise ValueError("Grasp local axis must be finite and nonzero.")
        for name in ("midpoint", "margin"):
            if type(getattr(self, name)) not in (int, float) or not math.isfinite(
                getattr(self, name)
            ):
                raise ValueError(f"{name} must be finite.")
        if self.margin < 0:
            raise ValueError("Grasp margin must be nonnegative.")
        if self.release_pose is not None:
            if (
                type(self.release_pose) is not tuple
                or len(self.release_pose) != 16
                or any(
                    type(x) not in (int, float) or not math.isfinite(x)
                    for x in self.release_pose
                )
            ):
                raise ValueError("Release pose must contain sixteen finite values.")
            if (
                type(self.plane_z) not in (int, float)
                or not math.isfinite(self.plane_z)
                or type(self.hand_envelope) is not NominalHandEnvelope
                or type(self.model_id) is not str
                or not self.model_id
            ):
                raise ValueError(
                    "Release plane requires explicit finite plane and nominal hand identity."
                )
        elif any(
            v is not None for v in (self.plane_z, self.hand_envelope, self.model_id)
        ):
            raise ValueError(
                "Release envelope fields require an explicit release pose."
            )
        if not self.upper_half and self.release_pose is None:
            raise ValueError(
                "A grasp constraint must declare a region or release plane."
            )

    def accepts(
        self,
        poses: torch.Tensor,
        object_pose: torch.Tensor,
        model: Mapping[str, object],
    ) -> torch.Tensor:
        """Apply original region/release arithmetic to candidate world poses.

        Args:
            poses: Ordered candidate gripper transforms, shape ``(K,4,4)``.
            object_pose: One observed object transform.
            model: Calibrated provider model with exact ID and opening width.

        Returns:
            Candidate mask; the independent opening filter is applied separately.
        """
        relative = pose_inv(object_pose) @ poses
        accepted = torch.isfinite(poses).all(-1).all(-1)
        if self.upper_half:
            accepted &= (
                relative[:, :3, 3] @ poses.new_tensor(self.local_axis)
            ) > self.midpoint
        if self.release_pose is not None:
            if model["model_id"] != self.model_id:
                raise ValueError(
                    "Grasp release envelope model differs from its declaration."
                )
            reserve = self.hand_envelope
            final = poses.new_tensor(self.release_pose).reshape(4, 4) @ relative
            opening = float(model["max_opening_width"])
            corners = poses.new_tensor(
                [
                    [side * opening + dx, dy, dz]
                    for side in (-1.0, 1.0)
                    for dx in (
                        -0.5 * reserve.finger_thickness,
                        0.5 * reserve.finger_thickness,
                    )
                    for dy in (-0.5 * reserve.finger_width, 0.5 * reserve.finger_width)
                    for dz in (
                        -0.5 * reserve.finger_length,
                        0.5 * reserve.finger_length,
                    )
                ]
            )
            world = corners @ final[:, :3, :3].transpose(-1, -2) + final[:, None, :3, 3]
            accepted &= world[..., 2].amin(-1) > self.plane_z + self.margin
        return accepted


class ConstrainedGraspPoseGenerator(ParallelJawGraspPoseGenerator):
    """Immutable per-call filtered view retaining source order and failure padding."""

    def __init__(
        self,
        delegate: ParallelJawGraspPoseGenerator,
        constraint: GraspGeometryConstraint,
    ) -> None:
        if (
            not isinstance(delegate, ParallelJawGraspPoseGenerator)
            or type(constraint) is not GraspGeometryConstraint
        ):
            raise TypeError(
                "Constrained sampling requires a parallel-jaw provider and typed constraint."
            )
        super().__init__(**delegate.gripper_data)
        self._delegate, self.constraint = delegate, constraint

    @property
    def paired_query_key(self) -> object:
        """Preserve the identity of the provider owning both paired roles."""
        return self._delegate.paired_query_key

    def _geometry(self, vertices, triangles):
        if (
            grasp_geometry_digest(vertices, triangles)
            != self.constraint.geometry_digest
        ):
            raise ValueError("Constrained Pick requires matching immutable geometry.")

    def accepts_grasp_poses(
        self,
        poses: torch.Tensor,
        object_poses: torch.Tensor,
        mesh_vertices: torch.Tensor,
        mesh_triangles: torch.Tensor,
    ) -> torch.Tensor:
        """Recheck exact alternatives against the original opening/region/plane.

        Args:
            poses: One world gripper pose per environment.
            object_poses: Matching observed object poses.
            mesh_vertices: Source local geometry bound by the constraint.
            mesh_triangles: Source topology bound by the constraint.

        Returns:
            Per-environment admissibility without changing any input.
        """
        if (
            poses.ndim != 3
            or poses.shape[1:] != (4, 4)
            or object_poses.shape != poses.shape
        ):
            raise ValueError("Explicit grasp filter requires matching (B,4,4) poses.")
        self._geometry(mesh_vertices, mesh_triangles)
        accepted = torch.isfinite(poses).all(-1).all(-1)
        for row in range(len(poses)):
            accepted[row] &= full_object_opening_mask(
                mesh_vertices,
                poses[row : row + 1],
                object_poses[row].to(poses),
                self.max_opening_width,
            )[0]
            accepted[row] &= self.constraint.accepts(
                poses[row : row + 1], object_poses[row].to(poses), self.gripper_data
            )[0]
        return accepted

    def get_valid_grasp_poses(
        self,
        *,
        mesh_vertices: torch.Tensor,
        mesh_triangles: torch.Tensor,
        obj_poses: torch.Tensor,
        approach_direction: torch.Tensor,
        obj_longest_axis: torch.Tensor | None = None,
        is_positive_part: bool | torch.Tensor = True,
        axis_region_selector: Any = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Filter ordinary proposals after delegate ranking without adding samples.

        Args:
            mesh_vertices: Source local vertices.
            mesh_triangles: Matching source topology.
            obj_poses: Current observed world object poses.
            approach_direction: Shared or row-specific direction.
            obj_longest_axis: Must be omitted for constrained center selection.
            is_positive_part: Existing provider selection flag, passed unchanged.
            axis_region_selector: Optional existing selector, passed unchanged.

        Returns:
            Ordered accepted rows or one finite source padding pose at infinite cost.
        """
        self._geometry(mesh_vertices, mesh_triangles)
        if obj_longest_axis is not None:
            raise ValueError(
                "Constrained Pick requires matching center-sampled geometry."
            )
        rows = self._delegate.get_valid_grasp_poses(
            mesh_vertices=mesh_vertices,
            mesh_triangles=mesh_triangles,
            obj_poses=obj_poses,
            approach_direction=approach_direction,
            obj_longest_axis=obj_longest_axis,
            is_positive_part=is_positive_part,
            **(
                {"axis_region_selector": axis_region_selector}
                if axis_region_selector is not None
                else {}
            ),
        )
        if obj_poses.shape != (len(rows), 4, 4):
            raise ValueError("Grasp filter rows must match the observed object poses.")
        filtered = []
        for row, (poses, costs) in enumerate(rows):
            if (
                poses.ndim != 3
                or poses.shape[1:] != (4, 4)
                or costs.shape != poses.shape[:1]
            ):
                raise ValueError(
                    "Grasp proposals must have matching pose and cost rows."
                )
            costs = costs.to(poses.device)
            accepted = torch.isfinite(costs)
            accepted &= full_object_opening_mask(
                mesh_vertices,
                poses,
                obj_poses[row].to(poses),
                self.max_opening_width,
            )
            accepted &= self.constraint.accepts(
                poses, obj_poses[row].to(poses), self.gripper_data
            )
            if accepted.any():
                filtered.append((poses[accepted].clone(), costs[accepted].clone()))
            else:
                padding = (
                    poses[:1] if poses.shape[0] else obj_poses[row : row + 1].to(poses)
                )
                filtered.append((padding.clone(), poses.new_full((1,), torch.inf)))
        return filtered

    def get_best_grasp_poses(
        self, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward the independent provider protocol without changing its operands."""
        return self._delegate.get_best_grasp_poses(**kwargs)

    def get_grasp_candidates(self, **kwargs: Any) -> Any:
        """Forward the independent articulation metadata protocol."""
        return self._delegate.get_grasp_candidates(**kwargs)

    def get_dual_arm_valid_grasp_poses(self, **kwargs: Any) -> Any:
        """Forward the independent paired-grasp protocol."""
        return self._delegate.get_dual_arm_valid_grasp_poses(**kwargs)
