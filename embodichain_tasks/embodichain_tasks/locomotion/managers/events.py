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

from typing import TYPE_CHECKING, Sequence

import torch

from embodichain.lab.gym.envs.managers import Functor, FunctorCfg
from embodichain.lab.gym.envs.managers.cfg import SceneEntityCfg

if TYPE_CHECKING:
    from embodichain.lab.gym.envs import EmbodiedEnv

__all__ = ["push_articulation_by_setting_velocity"]


class push_articulation_by_setting_velocity(Functor):
    """Apply independently scheduled root-velocity disturbances."""

    def __init__(self, cfg: FunctorCfg, env: EmbodiedEnv) -> None:
        """Initialize the manager term from the environment and configuration.

        Args:
            cfg: Manager configuration containing the term parameters.
            env: Environment providing the task state and robot.
        """
        super().__init__(cfg, env)
        entity_cfg: SceneEntityCfg = cfg.params["entity_cfg"]
        self._asset = env.sim.get_robot(entity_cfg.uid)
        if self._asset is None:
            self._asset = env.sim.get_articulation(entity_cfg.uid)
        if self._asset is None:
            raise ValueError(f"Articulation '{entity_cfg.uid}' was not found.")
        self._interval_range_s = tuple(cfg.params["interval_range_s"])
        velocity_range = cfg.params["velocity_range"]
        self._velocity_lower = torch.tensor(
            [
                velocity_range[name][0]
                for name in ("x", "y", "z", "roll", "pitch", "yaw")
            ],
            dtype=torch.float32,
            device=env.device,
        )
        self._velocity_upper = torch.tensor(
            [
                velocity_range[name][1]
                for name in ("x", "y", "z", "roll", "pitch", "yaw")
            ],
            dtype=torch.float32,
            device=env.device,
        )
        self._steps_remaining = torch.empty(
            env.num_envs, dtype=torch.long, device=env.device
        )
        self.reset()

    def _resample_interval(self, env_ids: torch.Tensor) -> None:
        seconds = torch.empty(len(env_ids), device=self._env.device).uniform_(
            float(self._interval_range_s[0]),
            float(self._interval_range_s[1]),
        )
        self._steps_remaining[env_ids] = (
            torch.ceil(seconds / self._env.step_dt).to(torch.long).clamp_min(1)
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Resample disturbance timers for selected environments.

        Args:
            env_ids: Selected environment rows; None selects all rows when supported.
        """
        ids = (
            torch.arange(self.num_envs, device=self._env.device)
            if env_ids is None
            else torch.as_tensor(env_ids, dtype=torch.long, device=self._env.device)
        )
        self._resample_interval(ids)

    def __call__(
        self,
        env: EmbodiedEnv,
        env_ids: torch.Tensor | None,
        entity_cfg: SceneEntityCfg,
        interval_range_s: Sequence[float],
        velocity_range: dict[str, Sequence[float]],
    ) -> None:
        """Advance timers and apply due disturbances.

        Args:
            env: Environment containing the articulation.
            env_ids: Environment rows advanced by the event manager.
            entity_cfg: Articulation selected by this functor.
            interval_range_s: Minimum and maximum interval in seconds.
            velocity_range: Per-axis disturbance ranges.
        """
        del entity_cfg, interval_range_s, velocity_range
        ids = (
            torch.arange(self.num_envs, device=env.device)
            if env_ids is None
            else env_ids.to(device=env.device, dtype=torch.long)
        )
        self._steps_remaining[ids] -= 1
        due_ids = ids[self._steps_remaining[ids] <= 0]
        if len(due_ids) == 0:
            return
        random = torch.rand((len(due_ids), 6), device=env.device)
        disturbance = self._velocity_lower + random * (
            self._velocity_upper - self._velocity_lower
        )
        velocity = self._asset.body_data.root_vel[due_ids] + disturbance
        self._asset.set_root_velocity(velocity, env_ids=due_ids)
        self._resample_interval(due_ids)
