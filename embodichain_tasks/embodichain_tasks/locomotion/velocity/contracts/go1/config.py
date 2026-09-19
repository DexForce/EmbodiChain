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

"""Load the Unitree Go1 velocity task configuration."""

from __future__ import annotations

from dataclasses import MISSING

from embodichain.utils import configclass

from .._config import load_task_data

__all__ = ["Go1VelocityConfig", "load_config"]


@configclass
class Go1VelocityConfig:
    """Dimensions and timing used by the Go1 velocity task."""

    data: dict = MISSING
    joint_names: tuple[str, ...] = MISSING
    default_joint_position: tuple[float, ...] = MISSING
    action_scale: tuple[float, ...] = MISSING
    actor_observation_dim: int = MISSING
    critic_observation_dim: int = MISSING
    action_dim: int = MISSING
    physics_dt: float = MISSING
    control_dt: float = MISSING
    max_episode_steps: int = MISSING


def load_config() -> Go1VelocityConfig:
    """Load the official MJLab Go1 flat-velocity task settings.

    Returns:
        The robot task configuration loaded from the packaged task.json.
    """
    data = load_task_data("go1_flat")
    robot = data["robot"]
    joint_names = tuple(robot["joint_names"])
    default_position = tuple(float(value) for value in robot["default_joint_position"])
    action_scale = tuple(float(value) for value in robot["action_scale"])
    action_dim = len(joint_names)
    physics = data["physics"]
    control_dt = float(physics["control_dt"])
    if not len(joint_names) == len(default_position) == len(action_scale):
        raise ValueError("Go1 joint, default pose, and action dimensions differ")
    observations = data["observations"]
    return Go1VelocityConfig(
        data=data,
        joint_names=joint_names,
        default_joint_position=default_position,
        action_scale=action_scale,
        actor_observation_dim=int(observations["actor_dimension"]),
        critic_observation_dim=int(observations["critic_dimension"]),
        action_dim=action_dim,
        physics_dt=float(physics["physics_dt"]),
        control_dt=control_dt,
        max_episode_steps=round(float(physics["episode_length_s"]) / control_dt),
    )
