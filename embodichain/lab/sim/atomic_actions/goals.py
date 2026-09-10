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

"""Goal contracts shared by atomic actions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .core import ObjectSemantics
    from .state import PlanningContext


@dataclass(frozen=True, slots=True, eq=False)
class SceneEntityPose:
    """Late-bound pose derived from a versioned scene entity.

    The semantic request remains stable while each call to
    :meth:`AtomicAction.plan` resolves the latest scene pose. This is the
    bridge used by an execution session to replan moving goals.
    """

    entity_id: str
    """Stable scene entity identifier."""

    relative_pose: torch.Tensor | None = None
    """Optional transform applied as ``entity_pose @ relative_pose``."""

    minimum_confidence: float = 0.0
    """Minimum accepted perception confidence."""

    world_displacement: torch.Tensor | None = None
    """Optional world-frame translation applied after ``relative_pose``."""

    world_orientation: torch.Tensor | None = None
    """Optional world-frame orientation applied before ``relative_pose``."""

    def __post_init__(self) -> None:
        if not isinstance(self.entity_id, str) or not self.entity_id.strip():
            raise ValueError("entity_id must be a non-empty string.")
        if self.relative_pose is not None:
            validate_pose_tensor(
                self.relative_pose,
                "relative_pose",
                allow_waypoints=False,
            )
            object.__setattr__(self, "relative_pose", self.relative_pose.clone())
        if self.world_displacement is not None:
            displacement = self.world_displacement
            if not isinstance(displacement, torch.Tensor):
                raise TypeError("world_displacement must be a torch.Tensor or None.")
            if displacement.dim() not in (1, 2) or displacement.shape[-1] != 3:
                raise ValueError(
                    "world_displacement must have shape (3,) or (num_envs, 3)."
                )
            if displacement.dim() == 2 and displacement.shape[0] == 0:
                raise ValueError("world_displacement batches must not be empty.")
            if not torch.isfinite(displacement).all():
                raise ValueError("world_displacement must contain finite values.")
            object.__setattr__(self, "world_displacement", displacement.clone())
        if self.world_orientation is not None:
            orientation = self.world_orientation
            if not isinstance(orientation, torch.Tensor):
                raise TypeError("world_orientation must be a torch.Tensor or None.")
            if orientation.dim() not in (2, 3) or orientation.shape[-2:] != (3, 3):
                raise ValueError(
                    "world_orientation must have shape (3, 3) or " "(num_envs, 3, 3)."
                )
            if orientation.dim() == 3 and orientation.shape[0] == 0:
                raise ValueError("world_orientation batches must not be empty.")
            if not torch.isfinite(orientation).all():
                raise ValueError("world_orientation must contain finite values.")
            object.__setattr__(self, "world_orientation", orientation.clone())
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be in [0, 1].")

    def snapshot(self) -> SceneEntityPose:
        """Return an independently owned late-bound pose value.

        Returns:
            Exact scene reference with an owned relative-pose tensor.
        """
        return SceneEntityPose(
            self.entity_id,
            relative_pose=self.relative_pose,
            minimum_confidence=self.minimum_confidence,
            world_displacement=self.world_displacement,
            world_orientation=self.world_orientation,
        )


PoseGoalValue = torch.Tensor | SceneEntityPose
"""Explicit pose tensor or a pose resolved from the latest scene snapshot."""


def validate_pose_tensor(
    value: torch.Tensor,
    name: str,
    *,
    allow_waypoints: bool,
) -> None:
    """Validate the environment-independent part of a pose goal.

    Args:
        value: Pose tensor to validate.
        name: Field name used in validation errors.
        allow_waypoints: Whether a batched waypoint dimension is accepted.

    Raises:
        TypeError: If ``value`` is not a tensor.
        ValueError: If the tensor shape is not a supported pose shape.
    """
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}.")
    valid_dims = {2, 3, 4} if allow_waypoints else {2, 3}
    if value.dim() not in valid_dims or value.shape[-2:] != (4, 4):
        supported = "(4, 4), (num_envs, 4, 4)"
        if allow_waypoints:
            supported += ", or (num_envs, n_waypoint, 4, 4)"
        raise ValueError(
            f"{name} must have shape {supported}, got {tuple(value.shape)}."
        )


def validate_pose_goal(
    value: PoseGoalValue,
    name: str,
    *,
    allow_waypoints: bool,
) -> None:
    """Validate an explicit or late-bound pose goal."""
    if isinstance(value, SceneEntityPose):
        return
    validate_pose_tensor(value, name, allow_waypoints=allow_waypoints)


def resolve_pose_goal(
    value: PoseGoalValue,
    context: PlanningContext,
    *,
    name: str,
) -> torch.Tensor:
    """Resolve a pose goal against a planning context.

    Args:
        value: Explicit tensor or scene-entity reference.
        context: Latest observed planning context.
        name: Field name used in validation errors.

    Returns:
        Explicit pose tensor. Scene references always return shape ``(B, 4, 4)``.
    """
    if isinstance(value, torch.Tensor):
        return value
    try:
        entity = context.scene.entities[value.entity_id]
    except KeyError as exc:
        raise KeyError(
            f"{name} references unknown scene entity {value.entity_id!r}."
        ) from exc
    if entity.confidence < value.minimum_confidence:
        raise ValueError(
            f"Scene entity {value.entity_id!r} confidence {entity.confidence} is "
            f"below {value.minimum_confidence}."
        )
    pose = entity.pose.to(device=context.robot.qpos.device, dtype=torch.float32)
    if pose.shape == (4, 4):
        pose = pose.unsqueeze(0).expand(context.batch_size, -1, -1)
    elif pose.shape != (context.batch_size, 4, 4):
        raise ValueError(
            f"Scene entity {value.entity_id!r} pose must match planning batch size."
        )
    if value.world_orientation is not None:
        orientation = value.world_orientation.to(device=pose.device, dtype=pose.dtype)
        if orientation.shape == (3, 3):
            orientation = orientation.unsqueeze(0).expand(context.batch_size, -1, -1)
        elif orientation.shape != (context.batch_size, 3, 3):
            raise ValueError(
                f"{name}.world_orientation must match planning batch size."
            )
        pose = pose.clone()
        pose[:, :3, :3] = orientation
    if value.relative_pose is None:
        resolved = pose.clone()
    else:
        relative = value.relative_pose.to(device=pose.device, dtype=pose.dtype)
        if relative.shape == (4, 4):
            relative = relative.unsqueeze(0).expand(context.batch_size, -1, -1)
        elif relative.shape != (context.batch_size, 4, 4):
            raise ValueError(f"{name}.relative_pose must match planning batch size.")
        resolved = torch.bmm(pose, relative)
    if value.world_displacement is None:
        return resolved
    displacement = value.world_displacement.to(device=pose.device, dtype=pose.dtype)
    if displacement.shape == (3,):
        displacement = displacement.unsqueeze(0).expand(context.batch_size, -1)
    elif displacement.shape != (context.batch_size, 3):
        raise ValueError(f"{name}.world_displacement must match planning batch size.")
    resolved[:, :3, 3] += displacement
    return resolved


def _resolve_object_pose(
    semantics: ObjectSemantics,
    context: PlanningContext,
    *,
    name: str = "object",
) -> torch.Tensor:
    """Resolve an object's pose from the current scene snapshot."""
    from .core import ObjectSemantics

    if not isinstance(semantics, ObjectSemantics):
        raise TypeError("semantics must be an ObjectSemantics instance.")
    return resolve_pose_goal(
        SceneEntityPose(semantics.entity_id),
        context,
        name=name,
    )


def collect_scene_dependencies(value: Any) -> tuple[str, ...]:
    """Collect stable scene entity identifiers referenced by a goal value."""
    from .core import ObjectSemantics

    found: set[str] = set()

    def visit(item: Any) -> None:
        if isinstance(item, SceneEntityPose):
            found.add(item.entity_id)
        elif isinstance(item, ObjectSemantics):
            return
        elif is_dataclass(item) and not isinstance(item, type):
            for data_field in fields(item):
                visit(getattr(item, data_field.name))
        elif isinstance(item, Mapping):
            for key, nested in item.items():
                visit(key)
                visit(nested)
        elif isinstance(item, Sequence) and not isinstance(
            item, (str, bytes, torch.Tensor)
        ):
            for nested in item:
                visit(nested)

    visit(value)
    return tuple(sorted(found))


@dataclass(frozen=True, slots=True, eq=False)
class ObjectActionGoal:
    """Shared semantic-object goal contract for object-centric skills."""

    semantics: ObjectSemantics
    """Semantic and geometric description of the object."""

    def __post_init__(self) -> None:
        from .core import ObjectSemantics

        if not isinstance(self.semantics, ObjectSemantics):
            raise TypeError("semantics must be an ObjectSemantics instance.")


__all__ = [
    "ObjectActionGoal",
    "PoseGoalValue",
    "SceneEntityPose",
    "collect_scene_dependencies",
    "resolve_pose_goal",
    "validate_pose_goal",
    "validate_pose_tensor",
]
