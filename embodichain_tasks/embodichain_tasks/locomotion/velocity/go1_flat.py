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

"""Unitree Go1 flat-ground velocity task."""

from __future__ import annotations

import torch

from embodichain.lab.gym.utils.registration import register_env
from embodichain.learning.rl.policy_evaluation.camera import PolicyViewerCameraCfg

from ._embodichain import EmbodiChainVelocityEnv
from .contracts.go1.config import load_config
from .contracts.go1.mdp import (
    Go1State,
    build_observations,
    compute_rewards,
    compute_termination,
    corrupt_actor_observation,
)

__all__ = ["UnitreeGo1FlatEnv"]

_CONFIG = load_config()


@register_env(
    "UnitreeGo1FlatRL-v1",
    max_episode_steps=_CONFIG.max_episode_steps,
    override=True,
    supports_rl=True,
)
class UnitreeGo1FlatEnv(EmbodiChainVelocityEnv):
    """Track planar velocity commands with the 12-DOF Go1 model."""

    policy_viewer_camera_cfg = PolicyViewerCameraCfg(
        eye_offset=(-1.15, -0.95, 0.6), target_height=0.25
    )
    velocity_task_config = _CONFIG
    state_type = Go1State
    build_observations_fn = staticmethod(build_observations)
    compute_rewards_fn = staticmethod(compute_rewards)
    compute_termination_fn = staticmethod(compute_termination)
    foot_link_names = ("FR_foot", "FL_foot", "RR_foot", "RL_foot")
    foot_offsets = ((0.0, 0.0, 0.0),) * 4
    illegal_contact_link_names = (
        "trunk",
        "FR_hip",
        "FR_thigh",
        "FR_calf",
        "FL_hip",
        "FL_thigh",
        "FL_calf",
        "RR_hip",
        "RR_thigh",
        "RR_calf",
        "RL_hip",
        "RL_thigh",
        "RL_calf",
    )
    orientation_link_name = "trunk"

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
        return corrupt_actor_observation(_CONFIG, actor, generator)
