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

"""Observed robot state, symbolic task state, and scene snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .core import ObjectSemantics


def _resolve_runtime_device(device: torch.device | str) -> torch.device:
    """Resolve an indexless CUDA device to the active concrete GPU index."""
    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return resolved


def _validate_pose(value: torch.Tensor, name: str) -> int | None:
    """Validate a homogeneous transform and return its explicit batch size."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.shape == (4, 4):
        return None
    if value.dim() != 3 or value.shape[-2:] != (4, 4) or value.shape[0] == 0:
        raise ValueError(
            f"{name} must have shape (4, 4) or (num_envs, 4, 4), "
            f"got {tuple(value.shape)}."
        )
    return int(value.shape[0])


def _normalize_mask(
    value: torch.Tensor | None,
    *,
    batch_size: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    """Return an owned boolean mask with shape ``(batch_size,)``."""
    if value is None:
        return torch.ones(batch_size, dtype=torch.bool, device=device)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor or None.")
    if value.dtype != torch.bool:
        raise TypeError(f"{name} must have dtype torch.bool, got {value.dtype}.")
    if value.shape != (batch_size,):
        raise ValueError(
            f"{name} must have shape ({batch_size},), got {tuple(value.shape)}."
        )
    return value.to(device=device).clone()


def _broadcast_pose(
    value: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    """Resolve an optionally batched pose to the task-state batch."""
    pose_batch_size = _validate_pose(value, name)
    if value.device != device:
        raise ValueError(f"{name} must use task-state device {device}.")
    if pose_batch_size is None:
        return value.unsqueeze(0).expand(batch_size, -1, -1).clone()
    if pose_batch_size != batch_size:
        raise ValueError(
            f"{name} batch size must be {batch_size}, got {pose_batch_size}."
        )
    return value.clone()


@dataclass(frozen=True, slots=True, eq=False)
class HeldObjectState:
    """Observed or projected relation between an object and one manipulator."""

    semantics: ObjectSemantics
    """Semantics of the held object."""

    object_to_eef: torch.Tensor
    """Object-to-end-effector transform."""

    grasp_xpos: torch.Tensor
    """End-effector grasp pose."""

    env_mask: torch.Tensor | None = None
    """Environments in which the relation is active."""

    def __post_init__(self) -> None:
        from .core import ObjectSemantics

        if not isinstance(self.semantics, ObjectSemantics):
            raise TypeError("semantics must be an ObjectSemantics instance.")
        object_batch = _validate_pose(self.object_to_eef, "object_to_eef")
        grasp_batch = _validate_pose(self.grasp_xpos, "grasp_xpos")
        explicit_batches = {
            size for size in (object_batch, grasp_batch) if size is not None
        }
        if len(explicit_batches) > 1:
            raise ValueError("Held-object poses must use the same batch size.")
        if self.object_to_eef.device != self.grasp_xpos.device:
            raise ValueError("Held-object poses must use the same device.")
        if self.env_mask is not None:
            mask_batch = int(self.env_mask.shape[0]) if self.env_mask.dim() == 1 else -1
            batch_size = next(iter(explicit_batches), mask_batch)
            object.__setattr__(
                self,
                "env_mask",
                _normalize_mask(
                    self.env_mask,
                    batch_size=batch_size,
                    device=self.object_to_eef.device,
                    name="env_mask",
                ),
            )


def _normalize_held(
    value: HeldObjectState,
    *,
    batch_size: int,
    device: torch.device,
) -> HeldObjectState:
    """Normalize a held-object relation to one task-state batch."""
    return HeldObjectState(
        semantics=value.semantics,
        object_to_eef=_broadcast_pose(
            value.object_to_eef,
            batch_size=batch_size,
            device=device,
            name="HeldObjectState.object_to_eef",
        ),
        grasp_xpos=_broadcast_pose(
            value.grasp_xpos,
            batch_size=batch_size,
            device=device,
            name="HeldObjectState.grasp_xpos",
        ),
        env_mask=_normalize_mask(
            value.env_mask,
            batch_size=batch_size,
            device=device,
            name="HeldObjectState.env_mask",
        ),
    )


@dataclass(frozen=True, slots=True, eq=False)
class TaskState:
    """Symbolic task state, separate from measured robot state."""

    batch_size: int
    """Number of vectorized environments represented by the state."""

    device: torch.device | str
    """Device used by per-environment masks and relation tensors."""

    held_objects: Mapping[str, HeldObjectState] = field(default_factory=dict)
    """Single-manipulator held-object relations keyed by control resource."""

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("TaskState.batch_size must be greater than zero.")
        device = _resolve_runtime_device(self.device)
        normalized_held: dict[str, HeldObjectState] = {}
        for resource, value in self.held_objects.items():
            if not isinstance(resource, str) or not resource:
                raise TypeError("held_objects keys must be non-empty strings.")
            if not isinstance(value, HeldObjectState):
                raise TypeError("held_objects values must be HeldObjectState objects.")
            normalized_held[resource] = _normalize_held(
                value, batch_size=self.batch_size, device=device
            )

        object.__setattr__(self, "device", device)
        object.__setattr__(self, "held_objects", MappingProxyType(normalized_held))

    @classmethod
    def empty(
        cls,
        batch_size: int,
        device: torch.device | str,
    ) -> TaskState:
        """Create an empty symbolic state.

        Args:
            batch_size: Number of represented environments.
            device: Tensor device used by the state.

        Returns:
            Empty task state with explicit batch metadata.
        """
        return cls(batch_size=batch_size, device=device)

    def get_held_object(self, resource: str) -> HeldObjectState | None:
        """Return the object held by ``resource``, if any."""
        return self.held_objects.get(resource)


@dataclass(frozen=True, slots=True, eq=False)
class RobotObservation:
    """Measured robot state used as the start of planning or replanning."""

    timestamp: float
    qpos: torch.Tensor
    qvel: torch.Tensor
    qeffort: torch.Tensor | None = None
    root_pose: torch.Tensor | None = None
    root_twist: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.timestamp < 0.0:
            raise ValueError("RobotObservation.timestamp must be non-negative.")
        if not isinstance(self.qpos, torch.Tensor) or self.qpos.dim() != 2:
            raise ValueError(
                "RobotObservation.qpos must have shape (num_envs, robot_dof)."
            )
        if self.qpos.shape[0] == 0 or self.qpos.shape[1] == 0:
            raise ValueError("RobotObservation.qpos dimensions must be non-zero.")
        if not isinstance(self.qvel, torch.Tensor):
            raise TypeError("RobotObservation.qvel must be a torch.Tensor.")
        if self.qvel.shape != self.qpos.shape:
            raise ValueError("RobotObservation.qvel must match qpos shape.")
        if self.qvel.device != self.qpos.device:
            raise ValueError("RobotObservation.qpos and qvel must share a device.")
        if self.qeffort is not None:
            if self.qeffort.shape != self.qpos.shape:
                raise ValueError("RobotObservation.qeffort must match qpos shape.")
            if self.qeffort.device != self.qpos.device:
                raise ValueError("RobotObservation.qeffort must share the qpos device.")
        object.__setattr__(self, "qpos", self.qpos.clone())
        object.__setattr__(self, "qvel", self.qvel.clone())
        if self.qeffort is not None:
            object.__setattr__(self, "qeffort", self.qeffort.clone())
        if self.root_pose is not None:
            object.__setattr__(self, "root_pose", self.root_pose.clone())
        if self.root_twist is not None:
            object.__setattr__(self, "root_twist", self.root_twist.clone())

    @property
    def batch_size(self) -> int:
        """Number of represented vectorized environments."""
        return int(self.qpos.shape[0])

    @property
    def robot_dof(self) -> int:
        """Number of robot joint-position columns."""
        return int(self.qpos.shape[1])

    def with_qpos(self, qpos: torch.Tensor) -> RobotObservation:
        """Create a projected observation with a new position and zero velocity.

        Args:
            qpos: Projected joint positions with the same shape as this observation.

        Returns:
            New observation suitable for compiling the next action.
        """
        return RobotObservation(
            timestamp=self.timestamp,
            qpos=qpos,
            qvel=torch.zeros_like(qpos),
            qeffort=self.qeffort,
            root_pose=self.root_pose,
            root_twist=self.root_twist,
        )


@dataclass(frozen=True, slots=True, eq=False)
class EntityState:
    """Scene entity state addressable by a stable entity identifier."""

    pose: torch.Tensor
    confidence: float = 1.0

    def __post_init__(self) -> None:
        _validate_pose(self.pose, "EntityState.pose")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("EntityState.confidence must be in [0, 1].")
        object.__setattr__(self, "pose", self.pose.clone())


@dataclass(frozen=True, slots=True, eq=False)
class SceneSnapshot:
    """Versioned scene state used to ground dynamic goals and obstacles."""

    timestamp: float
    version: int
    entities: Mapping[str, EntityState] = field(default_factory=dict)
    collision_world_revision: int | tuple[int, ...] = 0
    """Global or per-environment collision-world revision."""

    collision_entity_ids: tuple[str, ...] = ()
    """Entity IDs whose poses update a planner's dynamic collision world."""

    def __post_init__(self) -> None:
        if self.timestamp < 0.0:
            raise ValueError("SceneSnapshot.timestamp must be non-negative.")
        if self.version < 0:
            raise ValueError("SceneSnapshot.version must be non-negative.")
        revision = self.collision_world_revision
        if isinstance(revision, bool):
            raise TypeError("collision_world_revision must contain integers.")
        if isinstance(revision, int):
            if revision < 0:
                raise ValueError(
                    "collision_world_revision must contain non-negative values."
                )
        else:
            if not isinstance(revision, tuple) or not revision:
                raise TypeError(
                    "collision_world_revision must be an integer or a non-empty "
                    "tuple of integers."
                )
            if any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in revision
            ):
                raise TypeError("collision_world_revision must contain integers.")
            if any(value < 0 for value in revision):
                raise ValueError(
                    "collision_world_revision must contain non-negative values."
                )
        normalized: dict[str, EntityState] = {}
        for entity_id, state in self.entities.items():
            if not isinstance(entity_id, str) or not entity_id:
                raise ValueError("Scene entity identifiers must be non-empty strings.")
            if not isinstance(state, EntityState):
                raise TypeError(
                    "SceneSnapshot entities must contain EntityState values."
                )
            normalized[entity_id] = state
        collision_entity_ids = tuple(self.collision_entity_ids)
        if len(set(collision_entity_ids)) != len(collision_entity_ids) or not all(
            isinstance(entity_id, str) and entity_id
            for entity_id in collision_entity_ids
        ):
            raise ValueError(
                "collision_entity_ids must contain unique non-empty entity IDs."
            )
        missing = set(collision_entity_ids).difference(normalized)
        if missing:
            raise ValueError(
                "collision_entity_ids reference missing scene entities: "
                f"{sorted(missing)}."
            )
        object.__setattr__(self, "entities", MappingProxyType(normalized))
        object.__setattr__(self, "collision_entity_ids", collision_entity_ids)

    def collision_world_revisions(self, batch_size: int) -> tuple[int, ...]:
        """Expand the collision revision to one value per environment.

        Args:
            batch_size: Number of environments represented by the planning context.

        Returns:
            Per-environment monotonic revision tuple.

        Raises:
            ValueError: If an explicit revision tuple does not match the batch.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        revision = self.collision_world_revision
        if isinstance(revision, int):
            return (revision,) * batch_size
        if len(revision) == 1:
            return revision * batch_size
        if len(revision) != batch_size:
            raise ValueError(
                "collision_world_revision must be global or have one value per "
                f"environment; got {len(revision)} values for batch {batch_size}."
            )
        return revision

    def collision_obstacle_poses(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Mapping[str, torch.Tensor]:
        """Return collision obstacle poses in planning batch order.

        Args:
            batch_size: Number of planning environments.
            device: Planner tensor device.
            dtype: Planner tensor dtype.

        Returns:
            Mapping from configured collision entity ID to ``(B, 4, 4)`` pose.
        """
        poses: dict[str, torch.Tensor] = {}
        for entity_id in self.collision_entity_ids:
            pose = self.entities[entity_id].pose.to(device=device, dtype=dtype)
            if pose.shape == (4, 4):
                pose = pose.unsqueeze(0).expand(batch_size, -1, -1)
            elif pose.shape != (batch_size, 4, 4):
                raise ValueError(
                    f"Collision entity {entity_id!r} pose must match planning "
                    f"batch size {batch_size}."
                )
            poses[entity_id] = pose.clone()
        return MappingProxyType(poses)

    @classmethod
    def empty(cls) -> SceneSnapshot:
        """Create an empty initial scene snapshot."""
        return cls(timestamp=0.0, version=0)


@dataclass(frozen=True, slots=True, eq=False)
class PlanningContext:
    """Complete side-effect-free input to :meth:`AtomicAction.plan`."""

    robot: RobotObservation
    task: TaskState
    scene: SceneSnapshot
    env_ids: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.robot, RobotObservation):
            raise TypeError("robot must be a RobotObservation.")
        if not isinstance(self.task, TaskState):
            raise TypeError("task must be a TaskState.")
        if not isinstance(self.scene, SceneSnapshot):
            raise TypeError("scene must be a SceneSnapshot.")
        if self.task.batch_size != self.robot.batch_size:
            raise ValueError("TaskState and RobotObservation batch sizes must match.")
        if self.task.device != self.robot.qpos.device:
            raise ValueError("TaskState and RobotObservation must share a device.")
        self.scene.collision_world_revisions(self.robot.batch_size)
        for entity_id, state in self.scene.entities.items():
            if state.pose.dim() == 3 and state.pose.shape[0] != self.robot.batch_size:
                raise ValueError(
                    f"Scene entity {entity_id!r} pose batch must match the "
                    "planning context."
                )
        if not isinstance(self.env_ids, torch.Tensor):
            raise TypeError("env_ids must be a torch.Tensor.")
        if self.env_ids.dtype != torch.long:
            raise TypeError("env_ids must have dtype torch.long.")
        if self.env_ids.shape != (self.robot.batch_size,):
            raise ValueError(
                "env_ids must identify every row in the planning batch; expected "
                f"shape ({self.robot.batch_size},), got {tuple(self.env_ids.shape)}."
            )
        if self.env_ids.device != self.robot.qpos.device:
            raise ValueError("env_ids and robot tensors must share a device.")
        if torch.unique(self.env_ids).numel() != self.env_ids.numel():
            raise ValueError("env_ids must be unique.")
        object.__setattr__(self, "env_ids", self.env_ids.clone())

    @property
    def batch_size(self) -> int:
        """Number of environments in this planning request."""
        return self.robot.batch_size

    @property
    def last_qpos(self) -> torch.Tensor:
        """Measured joint positions used as the planning start state."""
        return self.robot.qpos

    @property
    def held_objects(self) -> Mapping[str, HeldObjectState]:
        """Single-resource held-object relations."""
        return self.task.held_objects

    def get_held_object(self, resource: str) -> HeldObjectState | None:
        """Return the object held by ``resource``, if any."""
        return self.task.get_held_object(resource)

    def project(
        self,
        *,
        qpos: torch.Tensor,
        task: TaskState,
    ) -> PlanningContext:
        """Create the hypothetical context used to compile a following action.

        Args:
            qpos: Projected terminal joint positions.
            task: Task state after applying expected effects.

        Returns:
            New context. No measured state or simulator state is mutated.
        """
        return PlanningContext(
            robot=self.robot.with_qpos(qpos),
            task=task,
            scene=self.scene,
            env_ids=self.env_ids,
        )


__all__ = [
    "EntityState",
    "HeldObjectState",
    "PlanningContext",
    "RobotObservation",
    "SceneSnapshot",
    "TaskState",
]
