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

"""Unitree H1_2 flat-ground velocity task."""

from __future__ import annotations

import torch

from embodichain.lab.gym.utils.registration import register_env

from ._embodichain import EmbodiChainVelocityEnv
from .contracts._reward_terms import corrupt_actor_observation
from .contracts.h1_2.config import load_config
from .contracts.h1_2.mdp import (
    H12State,
    build_observations,
    compute_rewards,
    compute_termination,
)

__all__ = ["UnitreeH12FlatEnv"]

_CONFIG = load_config()


@register_env(
    "UnitreeH1_2FlatRL-v1",
    max_episode_steps=_CONFIG.max_episode_steps,
    override=True,
    supports_rl=True,
)
class UnitreeH12FlatEnv(EmbodiChainVelocityEnv):
    """Track planar velocity commands with the 27-DOF H1_2 model."""

    velocity_task_config = _CONFIG
    state_type = H12State
    build_observations_fn = staticmethod(build_observations)
    compute_rewards_fn = staticmethod(compute_rewards)
    compute_termination_fn = staticmethod(compute_termination)
    foot_link_names = ("left_ankle_roll_link", "right_ankle_roll_link")
    foot_offsets = ((0.04, 0.0, -0.04), (0.04, 0.0, -0.04))
    orientation_link_name = "torso_link"
    imu_offset = (-0.04452, -0.01891, 0.27756)

    @staticmethod
    def corrupt_actor_fn(
        actor: torch.Tensor, generator: torch.Generator
    ) -> torch.Tensor:
        """Apply the task-defined actor observation noise.

        Args:
            actor: Actor observation tensor to corrupt in place.
            generator: Random generator used to sample the task noise or delays.

        Returns:
            The actor observation tensor after adding the configured noise.
        """
        return corrupt_actor_observation(
            _CONFIG.data, _CONFIG.action_dim, actor, generator
        )
