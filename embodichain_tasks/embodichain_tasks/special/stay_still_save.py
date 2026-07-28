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

"""Stay-still task environment for benchmarking LeRobot data saving.

This environment is intentionally cheap on the physics side - the robot holds
its initial configuration for the whole episode - so that the wall-clock cost
of a rollout is dominated by data saving rather than simulation. It is the
test fixture used by ``scripts/benchmark/data_pipeline/benchmark_lerobot_save.py``.

A camera sensor (``cam_high``) is attached so each frame carries a real RGB
image, which is what makes per-frame ``add_frame`` / PNG writing expensive and
exercises the async-saving optimizations.
"""

from __future__ import annotations

import torch

from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.utils.registration import register_env
from embodichain.utils import logger

__all__ = ["StayStillSaveEnv"]


@register_env("StayStillSave-v1", max_episode_steps=100)
class StayStillSaveEnv(EmbodiedEnv):
    """Robot holds still for 100 steps while a camera records.

    The demo action list is the initial joint configuration repeated for
    ``num_steps`` frames, so the robot does not move. Episode length is fixed
    at 100 steps via the ``register_env`` ``max_episode_steps`` argument and
    truncates automatically.

    The camera sensor, robot, scene and LeRobot dataset are configured through
    the gym config JSON (see
    ``embodichain_tasks/configs/gym/special/stay_still_save_ur10.json``).
    """

    def __init__(self, cfg: EmbodiedEnvCfg = None, **kwargs):
        if cfg is None:
            cfg = EmbodiedEnvCfg()
        super().__init__(cfg, **kwargs)

    def create_demo_action_list(self, *args, **kwargs):
        """Return 100 hold-still actions (initial qpos repeated).

        Returns:
            list[torch.Tensor]: One action per step, each shaped
            ``(num_envs, num_active_joints)``.
        """
        num_steps = 100

        # Initial pose, repeated for every step. The robot stays still.
        init_pose = self.robot.get_qpos()  # (num_envs, num_joints)

        action_list = [init_pose.clone() for _ in range(num_steps)]

        logger.log_info(
            f"Generated {len(action_list)} hold-still demo actions "
            f"(robot stationary)."
        )
        return action_list
