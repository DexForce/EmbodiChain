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

"""Fixed Isaac Lab 3.0 EA velocity task settings."""

from __future__ import annotations

from dataclasses import MISSING

from embodichain.utils import configclass

from ..contracts._config import load_task_data

__all__ = ["IsaacVelocityConfig", "load_config"]


@configclass
class IsaacVelocityConfig:
    """Policy dimensions, timing, model parameters and resolved reference terms."""

    data: dict = MISSING
    joint_names: tuple[str, ...] = MISSING
    default_joint_position: tuple[float, ...] = MISSING
    action_scale: tuple[float, ...] = MISSING
    action_dim: int = MISSING
    actor_observation_dim: int = MISSING
    critic_observation_dim: int = MISSING
    physics_dt: float = 0.005
    control_dt: float = 0.02
    max_episode_steps: int = 1000


def load_config(task: str) -> IsaacVelocityConfig:
    """Load one packaged reference definition, retaining policy joint order."""
    data = load_task_data(task)
    robot = data["robot"]
    dimension = len(robot["joint_names"])
    observation_dimension = 12 + 3 * dimension + (187 if dimension == 37 else 0)
    return IsaacVelocityConfig(
        data=data,
        joint_names=tuple(robot["joint_names"]),
        default_joint_position=tuple(robot["default_joint_position"]),
        action_scale=tuple(robot["action_scale"]),
        action_dim=dimension,
        actor_observation_dim=observation_dimension,
        critic_observation_dim=observation_dimension,
    )
