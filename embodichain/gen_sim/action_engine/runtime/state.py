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
from typing import Mapping

import torch

from embodichain.lab.sim.atomic_actions import (
    HeldObjectState,
    TaskState,
)

__all__ = ["ExecutionState"]


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
