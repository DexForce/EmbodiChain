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

"""Inert pose targets shared by configured Semantic Call lowerers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TypeAlias
import torch

from embodichain.lab.sim.atomic_skills import SceneEntityPose
from embodichain.lab.task_program.semantics import SemanticPose

__all__ = [
    "AbsolutePoseTarget",
    "SceneEntityTarget",
    "ConfiguredPoseTarget",
    "resolve_pose_target",
]


def _identifier(value: object, *, field_name: str) -> str:
    """Validate one exact non-empty identifier."""
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(
            f"{field_name} must be a non-empty string without outer whitespace."
        )
    return value


def _pose(value: tuple[float, ...]) -> tuple[float, ...]:
    """Validate and own one flattened SE(3) transform."""
    if type(value) is not tuple or len(value) != 16:
        raise TypeError("relative_pose must be an exact 16-value tuple.")
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float)) for item in value
    ):
        raise TypeError("relative_pose must contain sixteen real numbers.")
    normalized = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in normalized):
        raise ValueError("relative_pose must contain only finite values.")
    pose = torch.tensor(normalized, dtype=torch.float64).reshape(4, 4)
    if not torch.allclose(
        pose[3],
        torch.tensor((0.0, 0.0, 0.0, 1.0), dtype=torch.float64),
        atol=1.0e-6,
        rtol=0.0,
    ):
        raise ValueError("relative_pose must have bottom row [0, 0, 0, 1].")
    rotation = pose[:3, :3]
    if not torch.allclose(
        rotation.T @ rotation,
        torch.eye(3, dtype=torch.float64),
        atol=1.0e-6,
        rtol=0.0,
    ) or not torch.isclose(
        torch.linalg.det(rotation),
        torch.tensor(1.0, dtype=torch.float64),
        atol=1.0e-6,
        rtol=0.0,
    ):
        raise ValueError("relative_pose must contain a proper SE(3) rotation.")
    return normalized


def _point(
    value: tuple[float, float, float],
    *,
    field_name: str,
) -> tuple[float, float, float]:
    """Validate one finite local-frame point."""
    if type(value) is not tuple or len(value) != 3:
        raise TypeError(f"{field_name} must be an exact three-value tuple.")
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float)) for item in value
    ):
        raise TypeError(f"{field_name} must contain three real numbers.")
    normalized = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in normalized):
        raise ValueError(f"{field_name} must contain three finite values.")
    return normalized


@dataclass(frozen=True, slots=True)
class SceneEntityTarget:
    """Inert scene-relative target stored in a sealed integration declaration.

    Args:
        entity_id: Canonical reference entity whose observed pose supplies the frame.
        relative_pose: Optional flattened SE(3) offset in that entity's frame.
        world_orientation: Optional world-frame rotation overriding target orientation.
    """

    entity_id: str
    relative_pose: tuple[float, ...] | None = None
    world_orientation: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        _identifier(self.entity_id, field_name="entity_id")
        if self.relative_pose is not None:
            object.__setattr__(self, "relative_pose", _pose(self.relative_pose))
        if self.world_orientation is not None:
            if len(self.world_orientation) != 9 or not all(
                math.isfinite(x) for x in self.world_orientation
            ):
                raise ValueError("world_orientation requires nine finite values.")
            matrix = torch.eye(4, dtype=torch.float64)
            matrix[:3, :3] = torch.tensor(
                self.world_orientation, dtype=torch.float64
            ).reshape(3, 3)
            _pose(tuple(matrix.reshape(-1).tolist()))
            object.__setattr__(
                self,
                "world_orientation",
                tuple(float(x) for x in self.world_orientation),
            )

    def snapshot(self) -> SceneEntityTarget:
        """Return the immutable declaration, which owns no runtime tensors."""
        return self


@dataclass(frozen=True, slots=True)
class AbsolutePoseTarget:
    """Tensor-free absolute pose stored in a fingerprinted declaration.

    Args:
        position: World-frame XYZ position in metres.
        quaternion_xyzw: Finite nonzero orientation, normalized during construction.
    """

    position: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "position", _point(self.position, field_name="position")
        )
        quaternion = self.quaternion_xyzw
        if (
            type(quaternion) is not tuple
            or len(quaternion) != 4
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in quaternion
            )
        ):
            raise ValueError("quaternion_xyzw must contain four finite real values.")
        norm = math.hypot(*quaternion)
        if norm <= 1.0e-8:
            raise ValueError("quaternion_xyzw must be non-zero.")
        object.__setattr__(
            self, "quaternion_xyzw", tuple(value / norm for value in quaternion)
        )

    def snapshot(self) -> AbsolutePoseTarget:
        """Return the deeply immutable target declaration."""
        return self

    def to_matrix(self) -> torch.Tensor:
        """Create an independently owned runtime pose."""
        return SemanticPose(self.position, self.quaternion_xyzw).to_matrix()


def resolve_pose_target(
    value: AbsolutePoseTarget | SceneEntityTarget,
) -> torch.Tensor | SceneEntityPose:
    """Convert an inert declaration without observing or choosing a target.

    Args:
        value: Exact immutable absolute or scene-relative target declaration.

    Returns:
        An independently owned pose tensor or a late-bound SceneEntityPose. No
        simulator, Robot, FK, IK or planning service is consulted.
    """
    if type(value) is AbsolutePoseTarget:
        return value.to_matrix()
    if type(value) is SceneEntityTarget:
        return SceneEntityPose(
            value.entity_id,
            world_orientation=(
                None
                if value.world_orientation is None
                else torch.tensor(value.world_orientation, dtype=torch.float32).reshape(
                    3, 3
                )
            ),
            relative_pose=(
                None
                if value.relative_pose is None
                else torch.tensor(value.relative_pose, dtype=torch.float32).reshape(
                    4, 4
                )
            ),
        )
    raise TypeError(
        "Configured targets must be AbsolutePoseTarget or SceneEntityTarget."
    )


ConfiguredPoseTarget: TypeAlias = AbsolutePoseTarget | SceneEntityTarget
