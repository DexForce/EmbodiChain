# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
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

import math
from typing import Dict, Any, Tuple

import torch

from embodichain.lab.gym.envs import RLEnv, EmbodiedEnvCfg
from embodichain.lab.gym.utils.registration import register_env
from embodichain.lab.sim.types import EnvObs


@register_env("ToppleStickRL", max_episode_steps=100, override=True)
class ToppleStickEnv(RLEnv):
    """Topple a tall stick using a robot arm."""

    def __init__(self, cfg: EmbodiedEnvCfg | None = None, **kwargs):
        if cfg is None:
            cfg = EmbodiedEnvCfg()
        super().__init__(cfg, **kwargs)

        self.success_threshold = getattr(self, "success_threshold", 60.0)

    def _compute_stick_tilt(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute tilt angle of the stick relative to world +Z axis.

        Returns:
            Tuple of (tilt_angle_rad, tilt_angle_deg), shape (num_envs,)
        """
        stick = self.sim.get_rigid_object("stick")
        stick_pose = stick.get_local_pose(to_matrix=True)
        rot = stick_pose[:, :3, :3]

        up_vec = rot[:, :3, 2]
        world_up = torch.tensor(
            [0.0, 0.0, 1.0], device=self.device, dtype=up_vec.dtype
        ).view(1, 3)

        cos_theta = (up_vec * world_up).sum(-1)
        cos_theta = torch.clamp(cos_theta, -1.0 + 1e-6, 1.0 - 1e-6)

        tilt_rad = torch.acos(cos_theta)
        tilt_deg = tilt_rad * (180.0 / math.pi)
        return tilt_rad, tilt_deg

    def compute_task_state(
        self, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Compute success/fail flags and task metrics.

        Success: stick tilt angle greater than ``success_threshold`` (in degrees).
        Fail: always False for now (no explicit fail condition besides truncation).
        """
        _, tilt_deg = self._compute_stick_tilt()

        angle_threshold = float(self.success_threshold)
        is_success = tilt_deg > angle_threshold
        is_fail = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        metrics: Dict[str, Any] = {
            "tilt_deg": tilt_deg,
        }
        return is_success, is_fail, metrics

    def check_truncated(self, obs: EnvObs, info: Dict[str, Any]) -> torch.Tensor:
        """Truncate when timeout"""
        is_timeout = self._elapsed_steps >= self.episode_length

        return is_timeout

