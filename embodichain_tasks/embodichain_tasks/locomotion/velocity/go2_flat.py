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

"""Unitree Go2 flat-ground velocity task."""

from __future__ import annotations

import torch

from embodichain.lab.gym.utils.registration import register_env
from embodichain.learning.rl.policy_evaluation.camera import PolicyViewerCameraCfg

from ._embodichain import EmbodiChainVelocityEnv
from .contracts._reward_terms import corrupt_actor_observation
from .contracts.go2.config import load_config
from .contracts.go2.mdp import (
    Go2State,
    build_observations,
    compute_rewards,
    compute_termination,
)

__all__ = ["UnitreeGo2FlatEnv"]

_CONFIG = load_config()


@register_env(
    "UnitreeGo2FlatRL-v1",
    max_episode_steps=_CONFIG.max_episode_steps,
    override=True,
    supports_rl=True,
)
class UnitreeGo2FlatEnv(EmbodiChainVelocityEnv):
    """Track planar velocity commands with the 12-DOF Go2 model."""

    policy_viewer_camera_cfg = PolicyViewerCameraCfg(
        eye_offset=(-1.2, -1.0, 0.65), target_height=0.28
    )
    velocity_task_config = _CONFIG
    state_type = Go2State
    build_observations_fn = staticmethod(build_observations)
    compute_rewards_fn = staticmethod(compute_rewards)
    compute_termination_fn = staticmethod(compute_termination)
    foot_link_names = ("FR_calf", "FL_calf", "RR_calf", "RL_calf")
    foot_offsets = ((0.0, 0.0, -0.213),) * 4
    illegal_contact_link_names = (
        "base_link",
        "FR_hip",
        "FR_thigh",
        "FL_hip",
        "FL_thigh",
        "RR_hip",
        "RR_thigh",
        "RL_hip",
        "RL_thigh",
    )
    orientation_link_name = "base_link"

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
