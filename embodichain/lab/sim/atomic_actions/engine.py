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
from typing import Iterable, TYPE_CHECKING

from embodichain.utils import logger

from .core import (
    ActionResult,
    AtomicAction,
    Target,
    WorldState,
)

if TYPE_CHECKING:
    from embodichain.lab.sim.planners import MotionGenerator


# =============================================================================
# Global action registry (kept for third-party extensions)
# =============================================================================


_global_action_registry: dict[str, type[AtomicAction]] = {}


def _target_type_name(target_type: type | tuple[type, ...]) -> str:
    """Return a readable name for one accepted target type or a tuple of them."""
    if isinstance(target_type, tuple):
        return " | ".join(t.__name__ for t in target_type)
    return target_type.__name__


def register_action(name: str, action_class: type[AtomicAction]) -> None:
    """Register a custom AtomicAction subclass globally under ``name``."""
    _global_action_registry[name] = action_class


def unregister_action(name: str) -> None:
    """Remove a previously-registered action class. No-op if absent."""
    _global_action_registry.pop(name, None)


def get_registered_actions() -> dict[str, type[AtomicAction]]:
    """Return a copy of the global action-class registry."""
    return _global_action_registry.copy()


# =============================================================================
# AtomicActionEngine
# =============================================================================


class AtomicActionEngine:
    """Sequences typed atomic actions while threading WorldState through them."""

    def __init__(self, motion_generator: MotionGenerator) -> None:
        self.motion_generator = motion_generator
        self.robot = motion_generator.robot
        self.device = motion_generator.device
        self._actions: dict[str, AtomicAction] = {}

    @property
    def actions(self) -> dict[str, AtomicAction]:
        """Registered actions keyed by name (read-only copy)."""
        return dict(self._actions)

    def register(self, action: AtomicAction, *, name: str | None = None) -> None:
        """Register an action instance under ``name`` or its ``cfg.name``."""
        key = name if name is not None else action.cfg.name
        self._actions[key] = action

    def run(
        self,
        steps: Iterable[tuple[str, Target]],
        state: WorldState | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, WorldState]:
        """Run a sequence of named actions, threading WorldState through.

        Args:
            steps: Iterable of ``(action_name, typed_target)`` pairs.
            state: Initial world state. If None, seeded from ``robot.get_qpos()``.

        Returns:
            ``(success, concatenated_full_dof_trajectory, final_state)``.

            ``success`` is a ``(B,)`` boolean tensor indicating which
            environments completed every step. Failed environments hold their
            last successful joint position in both ``full_traj`` and
            ``final_state.last_qpos`` for the remainder of the sequence.

            An empty ``steps`` iterable is a successful no-op returning an
            empty trajectory and the seed state.
        """
        if state is None:
            state = WorldState(last_qpos=self.robot.get_qpos().clone())

        b = state.last_qpos.shape[0]
        full_traj = torch.empty(
            (b, 0, self.robot.dof),
            dtype=torch.float32,
            device=self.device,
        )
        alive = torch.ones(b, dtype=torch.bool, device=self.device)

        for name, target in steps:
            if name not in self._actions:
                logger.log_error(f"No action registered under name '{name}'", KeyError)
            action = self._actions[name]
            if not isinstance(target, action.TargetType):
                logger.log_error(
                    f"Action '{name}' expects target of type "
                    f"{_target_type_name(action.TargetType)}, got {type(target).__name__}",
                    TypeError,
                )
            if not alive.any():
                # All envs dead: fill held rows for this step.
                held = state.last_qpos.unsqueeze(1).repeat(1, 1, 1)
                full_traj = torch.cat([full_traj, held], dim=1)
                continue
            prev_last_qpos = state.last_qpos.clone()
            result: ActionResult = action.execute(target, state)
            step_success = (
                result.success
                if isinstance(result.success, torch.Tensor)
                else torch.tensor(bool(result.success), device=self.device)
            )
            step_success = step_success.to(self.device)
            alive = alive & step_success
            # Failed envs freeze at their last successful qpos for this step's trajectory.
            traj = result.trajectory
            held_rows = prev_last_qpos.unsqueeze(1).repeat(1, traj.shape[1], 1)
            traj = torch.where(alive[:, None, None], traj, held_rows)
            full_traj = torch.cat([full_traj, traj], dim=1)
            state = result.next_state
            state.last_qpos = torch.where(
                alive[:, None], state.last_qpos, prev_last_qpos
            )

        return alive, full_traj, state


__all__ = [
    "AtomicActionEngine",
    "get_registered_actions",
    "register_action",
    "unregister_action",
]
