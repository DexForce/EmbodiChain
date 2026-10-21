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

"""Terrain assignment and distance curriculum matching Isaac G1 Rough."""

from __future__ import annotations

import torch

__all__ = ["ReferenceTerrainCurriculum"]


class ReferenceTerrainCurriculum:
    """Assign terrain columns and advance rows at episode resets.

    Args:
        origins: Shared terrain origins, with shape (rows, columns, 3).
        num_envs: Number of independent robot instances.
        max_init_level: Maximum initial row, inclusive.
        terrain_length: Length of one terrain tile along X, in meters.
        episode_length_s: Time limit used by the commanded-distance rule.
        generator: Random generator for initial and overflow row assignment.
    """

    def __init__(
        self,
        origins: torch.Tensor,
        num_envs: int,
        max_init_level: int,
        terrain_length: float,
        episode_length_s: float,
        generator: torch.Generator,
    ) -> None:
        self.origins = origins
        self.generator = generator
        self.terrain_length = terrain_length
        self.episode_length_s = episode_length_s
        self.max_level, columns = origins.shape[:2]
        self.levels = torch.randint(
            0,
            min(max_init_level, self.max_level - 1) + 1,
            (num_envs,),
            device=origins.device,
            generator=generator,
        )
        self.types = torch.div(
            torch.arange(num_envs, device=origins.device),
            num_envs / columns,
            rounding_mode="floor",
        ).long()
        self.env_origins = origins[self.levels, self.types].clone()

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return terrain assignments and the shared task generator state."""
        return {
            "levels": self.levels.detach().cpu().clone(),
            "types": self.types.detach().cpu().clone(),
            "origins": self.origins.detach().cpu().clone(),
            "generator": self.generator.get_state().cpu().clone(),
        }

    def validate_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Check saved assignments without changing the current curriculum.

        Args:
            state: Terrain assignments, map origins and task RNG state.

        Raises:
            ValueError: The terrain map, instance count or indices differ.
        """
        levels = state["levels"].to(self.levels.device)
        types = state["types"].to(self.types.device)
        if levels.shape != self.levels.shape or types.shape != self.types.shape:
            raise ValueError("Terrain curriculum instance count differs")
        if not torch.equal(state["origins"].to(self.origins), self.origins):
            raise ValueError("Terrain curriculum map differs")
        if levels.dtype != torch.long or types.dtype != torch.long:
            raise ValueError("Terrain curriculum indices must be int64")
        if bool(((levels < 0) | (levels >= self.max_level)).any()):
            raise ValueError("Terrain curriculum level is out of range")
        if bool(((types < 0) | (types >= self.origins.shape[1])).any()):
            raise ValueError("Terrain curriculum type is out of range")
        torch.Generator(device=self.generator.device).set_state(
            state["generator"].cpu()
        )

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Restore assignments after checking the terrain map and instance count.

        Args:
            state: Terrain assignments, map origins and task RNG state.
        """
        self.validate_state_dict(state)
        levels = state["levels"].to(self.levels.device)
        types = state["types"].to(self.types.device)
        self.generator.set_state(state["generator"].cpu())
        self.levels.copy_(levels)
        self.types.copy_(types)
        self.env_origins.copy_(self.origins[levels, types])

    def update(
        self,
        env_ids: torch.Tensor,
        root_positions: torch.Tensor,
        commands: torch.Tensor,
    ) -> None:
        """Use pre-reset positions and commands to update selected rows.

        Args:
            env_ids: Instances whose episodes are being reset.
            root_positions: Positions for all instances, in terrain coordinates.
            commands: Current linear X/Y and yaw velocity commands for all instances.
        """
        distance = torch.linalg.vector_norm(
            root_positions[env_ids, :2] - self.env_origins[env_ids, :2], dim=1
        )
        move_up = distance > self.terrain_length / 2
        move_down = distance < (
            torch.linalg.vector_norm(commands[env_ids, :2], dim=1)
            * self.episode_length_s
            * 0.5
        )
        move_down &= ~move_up
        levels = self.levels[env_ids] + move_up.long() - move_down.long()
        random_levels = torch.randint(
            self.max_level,
            levels.shape,
            device=levels.device,
            generator=self.generator,
        )
        self.levels[env_ids] = torch.where(
            levels >= self.max_level, random_levels, levels.clamp_min(0)
        )
        self.env_origins[env_ids] = self.origins[
            self.levels[env_ids], self.types[env_ids]
        ]
