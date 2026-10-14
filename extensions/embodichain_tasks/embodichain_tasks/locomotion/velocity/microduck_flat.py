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

"""Pollen Robotics MicroDuck flat-ground velocity task."""

from __future__ import annotations

import torch

from embodichain.lab.gym.utils.registration import register_env
from embodichain.learning.rl.policy_evaluation.camera import PolicyViewerCameraCfg

from ._embodichain import EmbodiChainVelocityEnv
from .contracts._reward_terms import corrupt_actor_observation
from .contracts.microduck.config import load_config
from .contracts.microduck.mdp import (
    MicroDuckState,
    build_observations,
    compute_rewards,
    compute_termination,
)

__all__ = ["MicroDuckFlatEnv"]

_CONFIG = load_config()


@register_env(
    "MicroDuckFlatRL-v1",
    max_episode_steps=_CONFIG.max_episode_steps,
    override=True,
    supports_rl=True,
)
class MicroDuckFlatEnv(EmbodiChainVelocityEnv):
    """Track planar velocity commands with the 14-DOF MicroDuck model."""

    policy_viewer_camera_cfg = PolicyViewerCameraCfg(
        eye_offset=(-0.48, -0.4, 0.26), target_height=0.14
    )
    velocity_task_config = _CONFIG
    state_type = MicroDuckState
    build_observations_fn = staticmethod(build_observations)
    compute_rewards_fn = staticmethod(compute_rewards)
    compute_termination_fn = staticmethod(compute_termination)
    foot_link_names = ("ankle_left", "ankle_right")
    foot_offsets = ((0.0, -0.0238146, -0.0140852), (0.0, -0.0238146, -0.0140852))
    # Every tracked link except the two feet: ground contact on any of these
    # (trunk, shins, neck/head, jaw) means the robot is dragging its body,
    # which the task treats as a fall. Without this the Newton backend learns
    # to scoot on the jaw and shins with the feet in the air.
    illegal_contact_link_names = (
        "trunk_base",
        "yaw2roll",
        "hip_l",
        "upper_leg_left",
        "leg",
        "neck",
        "neck_pitch",
        "yaw_roll_motion",
        "jaw_soft",
        "bearing_roll",
        "hip_l_2",
        "upper_leg_right",
        "leg_2",
    )
    orientation_link_name = "trunk_base"

    @staticmethod
    def corrupt_actor_fn(
        actor: torch.Tensor, generator: torch.Generator
    ) -> torch.Tensor:
        return corrupt_actor_observation(
            _CONFIG.data, _CONFIG.action_dim, actor, generator
        )
