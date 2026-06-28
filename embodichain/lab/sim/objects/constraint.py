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

"""Rigid constraint wrapper binding two RigidObjects across arenas."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from embodichain.lab.sim.cfg import RigidConstraintCfg
    from embodichain.lab.sim.objects.rigid_object import RigidObject


@dataclass
class RigidConstraint:
    """Batch of fixed constraints linking two :class:`RigidObject` instances.

    Each entry binds ``rigid_object_a._entities[i]`` to
    ``rigid_object_b._entities[i]`` within arena ``i`` via a dexsim
    ``FixedConstraint``. The ``constraint_handles`` list has length
    ``num_envs`` with ``None`` wherever the constraint is not active in that
    arena, so arena index always equals list index.

    Args:
        cfg: The constraint configuration.
        constraint_handles: Per-arena dexsim constraint handles (None where inactive).
        rigid_object_a: The first RigidObject.
        rigid_object_b: The second RigidObject.
        device: The torch device.
    """

    cfg: RigidConstraintCfg
    constraint_handles: list[Any] = field(default_factory=list)
    rigid_object_a: RigidObject = None
    rigid_object_b: RigidObject = None
    device: torch.device = field(default_factory=torch.device("cpu"))

    @property
    def num_envs(self) -> int:
        """Number of arenas covered by this constraint."""
        return len(self.constraint_handles)

    def get_name(self, env_id: int) -> str:
        """Get the per-arena constraint name.

        For single-env constraints, returns the base name. For multi-env
        constraints, returns ``f"{base}_{env_id}"``.

        Args:
            env_id: The arena index.

        Returns:
            The constraint name registered in that arena.
        """
        if self.num_envs <= 1:
            return self.cfg.name
        return f"{self.cfg.name}_{env_id}"

    def _active_env_ids(self, env_ids: Sequence[int] | None) -> list[int]:
        """Resolve the requested env_ids, skipping handles that are None."""
        if env_ids is None:
            env_ids = range(self.num_envs)
        return [i for i in env_ids if self.constraint_handles[i] is not None]

    def get_relative_transform(
        self, env_ids: Sequence[int] | None = None
    ) -> list[np.ndarray]:
        """Get the relative transform of B in A for each active env.

        Args:
            env_ids: Subset of arenas. None -> all arenas. Inactive (None)
                handles are skipped.

        Returns:
            A list of 4x4 numpy arrays, one per active env.
        """
        results = []
        for i in self._active_env_ids(env_ids):
            results.append(self.constraint_handles[i].get_relative_transform())
        return results

    def get_local_pose(
        self, actor_index: int, env_ids: Sequence[int] | None = None
    ) -> list[np.ndarray]:
        """Get the local pose of the constraint frame for the given actor.

        Args:
            actor_index: 0 for object A, 1 for object B.
            env_ids: Subset of arenas. None -> all. Inactive handles skipped.

        Returns:
            A list of 4x4 numpy arrays, one per active env.
        """
        results = []
        for i in self._active_env_ids(env_ids):
            results.append(self.constraint_handles[i].get_local_pose(actor_index))
        return results

    def is_valid(self, env_ids: Sequence[int] | None = None) -> list[bool]:
        """Check validity of each active constraint handle.

        Args:
            env_ids: Subset of arenas. None -> all. Inactive handles skipped.

        Returns:
            A list of bools, one per active env.
        """
        return [
            self.constraint_handles[i].is_valid() for i in self._active_env_ids(env_ids)
        ]

    def destroy(
        self,
        env_ids: Sequence[int] | None = None,
        arena_resolver: Callable[[int], Any] | None = None,
    ) -> None:
        """Remove this constraint from the specified arenas.

        Args:
            env_ids: Subset of arenas to clear. None -> all active arenas.
            arena_resolver: Callable returning the arena for a given env index.
                Required to actually remove constraints from dexsim.
        """
        for i in self._active_env_ids(env_ids):
            if arena_resolver is not None:
                arena = arena_resolver(i)
                arena.remove_constraint(self.get_name(i))
            self.constraint_handles[i] = None
