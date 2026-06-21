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
from typing import Dict, Iterable, Optional, Tuple, Type

from embodichain.utils import logger

from .core import (
    ActionResult,
    AtomicAction,
    Target,
    WorldState,
)

# =============================================================================
# Global action registry (kept for third-party extensions)
# =============================================================================


_global_action_registry: Dict[str, Type[AtomicAction]] = {}


def register_action(name: str, action_class: Type[AtomicAction]) -> None:
    """Register a custom AtomicAction subclass globally under ``name``."""
    _global_action_registry[name] = action_class


def unregister_action(name: str) -> None:
    """Remove a previously-registered action class. No-op if absent."""
    _global_action_registry.pop(name, None)


def get_registered_actions() -> Dict[str, Type[AtomicAction]]:
    """Return a copy of the global action-class registry."""
    return _global_action_registry.copy()


# =============================================================================
# AtomicActionEngine
# =============================================================================


class AtomicActionEngine:
    """Sequences typed atomic actions while threading WorldState through them."""

    def __init__(self, motion_generator) -> None:
        self.motion_generator = motion_generator
        self.robot = motion_generator.robot
        self.device = motion_generator.device
        self._actions: Dict[str, AtomicAction] = {}

    @property
    def actions(self) -> Dict[str, AtomicAction]:
        """Registered actions keyed by name (read-only copy)."""
        return dict(self._actions)

    def register(self, action: AtomicAction, *, name: Optional[str] = None) -> None:
        """Register an action instance under ``name`` or its ``cfg.name``."""
        key = name if name is not None else action.cfg.name
        self._actions[key] = action

    def run(
        self,
        steps: Iterable[Tuple[str, Target]],
        state: Optional[WorldState] = None,
    ) -> Tuple[bool, torch.Tensor, WorldState]:
        """Run a sequence of named actions, threading WorldState through.

        Args:
            steps: Iterable of ``(action_name, typed_target)`` pairs.
            state: Initial world state. If None, seeded from ``robot.get_qpos()``.

        Returns:
            ``(success, concatenated_full_dof_trajectory, final_state)``.

            On failure ``success`` is False and the trajectory is the
            concatenation of steps that completed before the failure; the
            failing step contributes no waypoints. ``final_state`` is the
            state going into the failed step.
        """
        steps_list = list(steps)
        if state is None:
            state = WorldState(last_qpos=self.robot.get_qpos())

        full_traj = torch.empty(
            (state.last_qpos.shape[0], 0, self.robot.dof),
            dtype=torch.float32,
            device=self.device,
        )

        for name, target in steps_list:
            if name not in self._actions:
                logger.log_error(f"No action registered under name '{name}'", KeyError)
            action = self._actions[name]
            if not isinstance(target, action.TargetType):
                logger.log_error(
                    f"Action '{name}' expects target of type "
                    f"{action.TargetType.__name__}, got {type(target).__name__}",
                    TypeError,
                )
            result: ActionResult = action.execute(target, state)
            if not result.success:
                return False, full_traj, state
            full_traj = torch.cat([full_traj, result.trajectory], dim=1)
            state = result.next_state

        return True, full_traj, state


__all__ = [
    "AtomicActionEngine",
    "get_registered_actions",
    "register_action",
    "unregister_action",
]
