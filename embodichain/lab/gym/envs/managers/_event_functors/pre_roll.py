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

"""Fixed-step physical scene preparation for expert demonstrations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv

__all__ = ["pre_roll_with_robot_home"]


def pre_roll_with_robot_home(
    env: EmbodiedEnv,
    env_ids: torch.Tensor | Sequence[int] | None,
    steps: int,
    *,
    restore_home: bool = False,
    zero_velocity_target: bool = True,
) -> None:
    """Advance a full World under configured robot home targets before recording.

    Args:
        env: Environment owning the physics clock and initial joint configuration.
        env_ids: All arena IDs, or None for a full reset.
        steps: Number of physics updates to run before initializing observations.
        restore_home: Restore measured joints to home at the end of pre-roll.
        zero_velocity_target: Write zero velocity targets before physics advances.
    """
    if env_ids is not None and len(env_ids) != env.num_envs:
        raise ValueError("Fixed-step scene pre-roll requires a full reset.")
    if type(steps) is not int or steps < 0:
        raise ValueError("Pre-roll steps must be a nonnegative integer.")
    if type(restore_home) is not bool or type(zero_velocity_target) is not bool:
        raise ValueError("Pre-roll robot options must be booleans.")
    robot = env.robot
    home = torch.as_tensor(
        robot.cfg.init_qpos, device=env.device, dtype=torch.float32
    ).expand(env.num_envs, -1)
    zero = torch.zeros_like(home)
    robot.set_qpos(home, target=True)
    if zero_velocity_target:
        robot.set_qvel(zero, target=True)
    for _ in range(steps):
        env.sim.update(step=1, render_final_step=False)
    if restore_home:
        robot.set_state(qpos=home, qvel=zero, target_qpos=home, target_qvel=zero)
