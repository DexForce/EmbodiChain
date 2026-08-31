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

"""Action Engine execution state at the atomic-planning boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import torch

from embodichain.lab.sim.atomic_actions import (
    HeldObjectState,
    SceneSnapshot,
    TaskState,
)

__all__ = ["ExecutionState"]


@dataclass(frozen=True, slots=True, eq=False)
class _CollisionOverrideSceneSnapshot(SceneSnapshot):
    """Keep semantic entity poses live while overriding collision poses."""

    collision_pose_overrides: Mapping[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        SceneSnapshot.__post_init__(self)
        normalized: dict[str, torch.Tensor] = {}
        for entity_id, pose in self.collision_pose_overrides.items():
            if entity_id not in self.collision_entity_ids:
                raise ValueError(
                    "Collision pose overrides must reference collision entities."
                )
            if (
                not isinstance(pose, torch.Tensor)
                or not pose.is_floating_point()
                or pose.dim() not in (2, 3)
                or pose.shape[-2:] != (4, 4)
                or not bool(torch.isfinite(pose).all().item())
            ):
                raise ValueError(
                    "Collision pose overrides must be finite floating tensors "
                    "with shape (4, 4) or (B, 4, 4)."
                )
            normalized[entity_id] = pose.detach().clone()
        object.__setattr__(
            self,
            "collision_pose_overrides",
            MappingProxyType(normalized),
        )

    def collision_obstacle_poses(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Mapping[str, torch.Tensor]:
        """Return planner poses with intentional-contact rows parked."""
        poses = dict(
            SceneSnapshot.collision_obstacle_poses(
                self,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )
        )
        for entity_id, override in self.collision_pose_overrides.items():
            pose = override.to(device=device, dtype=dtype)
            if pose.shape == (4, 4):
                pose = pose.unsqueeze(0).expand(batch_size, -1, -1)
            elif pose.shape != (batch_size, 4, 4):
                raise ValueError(
                    f"Collision override {entity_id!r} must match planning "
                    f"batch size {batch_size}."
                )
            poses[entity_id] = pose.clone()
        return MappingProxyType(poses)


@dataclass(slots=True, eq=False)
class ExecutionState:
    """Projected task state paired with the next full-robot planning seed.

    The simulation atomic-action package deliberately no longer exposes the
    legacy ``WorldState`` compatibility object. Action Engine keeps this narrow
    orchestration state locally and converts it to immutable ``TaskState`` and
    ``PlanningContext`` values immediately before invoking the shared planner.
    """

    last_qpos: torch.Tensor
    held_objects: dict[str, HeldObjectState] = field(default_factory=dict)

    def get_held_object(self, control_part: str) -> HeldObjectState | None:
        """Return the held-object relation for one control part."""
        return self.held_objects.get(control_part)

    def with_updates(
        self,
        *,
        last_qpos: torch.Tensor | None = None,
        held_objects: Mapping[str, HeldObjectState] | None = None,
    ) -> ExecutionState:
        """Return a detached successor state."""
        return ExecutionState(
            last_qpos=self.last_qpos if last_qpos is None else last_qpos,
            held_objects=dict(
                self.held_objects if held_objects is None else held_objects
            ),
        )

    def to_task_state(self) -> TaskState:
        """Convert this state to the shared immutable symbolic task contract."""
        return TaskState(
            batch_size=int(self.last_qpos.shape[0]),
            device=self.last_qpos.device,
            held_objects=self.held_objects,
        )

    @classmethod
    def from_task_state(
        cls,
        task: TaskState,
        *,
        last_qpos: torch.Tensor,
    ) -> ExecutionState:
        """Build an orchestration state from a committed or projected task state."""
        return cls(
            last_qpos=last_qpos,
            held_objects=dict(task.held_objects),
        )
