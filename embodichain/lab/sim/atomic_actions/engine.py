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
    ActionTarget,
    ActionResult,
    AtomicAction,
    WorldState,
    _resolve_runtime_device,
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
        self.device = _resolve_runtime_device(motion_generator.device)
        self._actions: dict[str, AtomicAction] = {}

    @property
    def actions(self) -> dict[str, AtomicAction]:
        """Registered actions keyed by name (read-only copy)."""
        return dict(self._actions)

    def register(self, action: AtomicAction, *, name: str | None = None) -> None:
        """Register an action instance under ``name`` or its ``cfg.name``."""
        declared_target_type = getattr(action, "TargetType", None)
        target_types = (
            declared_target_type
            if isinstance(declared_target_type, tuple)
            else (declared_target_type,)
        )
        if not target_types or not all(
            isinstance(target_type, type) and issubclass(target_type, ActionTarget)
            for target_type in target_types
        ):
            logger.log_error(
                "AtomicAction.TargetType must contain ActionTarget subclasses.",
                TypeError,
            )
        key = name if name is not None else action.cfg.name
        self._actions[key] = action

    def run(
        self,
        steps: Iterable[tuple[str, ActionTarget]],
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

        if state.robot_dof != self.robot.dof:
            raise ValueError(
                "Initial WorldState DoF must match the engine robot, "
                f"got {state.robot_dof} and {self.robot.dof}."
            )
        robot_batch_size = int(self.robot.get_qpos().shape[0])
        if state.batch_size != robot_batch_size:
            raise ValueError(
                "Initial WorldState batch size must match the engine robot, "
                f"got {state.batch_size} and {robot_batch_size}."
            )
        if state.last_qpos.device != self.device:
            raise ValueError(
                "Initial WorldState and AtomicActionEngine must use the same device."
            )

        b = state.batch_size
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
                # All envs dead: no further motion to plan.
                break
            prev_last_qpos = state.last_qpos.clone()
            result: ActionResult = action.execute(target, state)
            if result.trajectory.shape[0] != b:
                raise ValueError(
                    f"Action '{name}' returned batch {result.trajectory.shape[0]}, "
                    f"but the engine state batch is {b}."
                )
            if result.trajectory.shape[2] != self.robot.dof:
                raise ValueError(
                    f"Action '{name}' returned {result.trajectory.shape[2]} DoF, "
                    f"but the engine robot has {self.robot.dof}."
                )
            if result.trajectory.device != self.device:
                raise ValueError(
                    f"Action '{name}' returned a trajectory on "
                    f"{result.trajectory.device}, expected {self.device}."
                )
            step_success = result.success.to(self.device)
            alive = alive & step_success
            # Failed envs freeze at their last successful qpos for this step's trajectory.
            traj = result.trajectory
            held_rows = prev_last_qpos.unsqueeze(1).repeat(1, traj.shape[1], 1)
            traj = torch.where(alive[:, None, None], traj, held_rows)
            full_traj = torch.cat([full_traj, traj], dim=1)
            state = state.masked_merge(result.next_state, alive)

        return alive, full_traj, state


__all__ = [
    "AtomicActionEngine",
    "get_registered_actions",
    "register_action",
    "unregister_action",
]
