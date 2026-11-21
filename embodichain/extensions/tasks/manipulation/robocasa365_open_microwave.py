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

"""Franka opening the hinged door from RoboCasa365 OpenMicrowave episode 86."""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any

import torch

from embodichain.lab.gym.envs import EmbodiedEnv
from embodichain.lab.gym.utils.registration import register_env

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Articulation

__all__ = ["RoboCasa365OpenMicrowaveEnv"]


@register_env("RoboCasa365OpenMicrowave-Franka-v1", max_episode_steps=300)
class RoboCasa365OpenMicrowaveEnv(EmbodiedEnv):
    """Evaluate door travel, independently of the gripper and microwave buttons."""

    @cached_property
    def _door(self) -> tuple[Articulation, int]:
        asset = self.sim.get_articulation("microwave")
        name = "microwave_right_group_1_microjoint"
        if name not in asset.joint_names:
            raise ValueError(
                f"Missing microwave joint {name!r}. Available: {asset.joint_names}."
            )
        return asset, asset.joint_names.index(name)

    def compute_task_state(
        self, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Measure opening from the closed upper limit towards the negative lower limit."""
        asset, index = self._door
        position = asset.get_qpos()[:, index]
        lower, upper = asset.get_qpos_limits(joint_ids=[index])[:, 0].unbind(-1)
        fraction = (upper - position) / (upper - lower)
        valid = torch.isfinite(fraction) & (upper > lower)
        return valid & (fraction >= 0.90), ~valid, {"door_open_fraction": fraction}

    def is_task_success(self, **kwargs: Any) -> torch.Tensor:
        """Return the measured predicate for demonstrations and external evaluation."""
        return self.compute_task_state(**kwargs)[0]
