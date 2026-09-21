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

"""Load the Unitree G1 velocity task configuration."""

from __future__ import annotations

from dataclasses import MISSING

from embodichain.utils import configclass

from .._config import load_task_data

__all__ = ["G1VelocityConfig", "load_config"]


@configclass
class G1VelocityConfig:
    """Dimensions and timing used by the G1 velocity task."""

    data: dict = MISSING
    joint_names: tuple[str, ...] = MISSING
    default_joint_position: tuple[float, ...] = MISSING
    action_scale: tuple[float, ...] = MISSING
    actor_observation_dim: int = MISSING
    critic_observation_dim: int = MISSING
    action_dim: int = MISSING
    phase_period: float = MISSING
    physics_dt: float = MISSING
    control_dt: float = MISSING
    max_episode_steps: int = MISSING


def load_config() -> G1VelocityConfig:
    """Load G1 settings from the packaged task definition.

    Returns:
        The robot task configuration loaded from the packaged task.json.
    """
    data = load_task_data("g1_flat")
    robot = data["robot"]
    joint_names = tuple(robot["joint_names"])
    default_position = tuple(float(value) for value in robot["default_joint_position"])
    action_scale = tuple(float(value) for value in robot["action_scale"])
    foot_names = data["observations"]["critic"]["terms"]["foot_height"]["params"][
        "asset_cfg"
    ]["site_names"]
    actor_dim = int(data["observations"]["actor_dimension"])
    physics = data["physics"]
    control_dt = float(physics["control_dt"])
    if not len(joint_names) == len(default_position) == len(action_scale):
        raise ValueError("G1 joint, default pose, and action dimensions differ")
    return G1VelocityConfig(
        data=data,
        joint_names=joint_names,
        default_joint_position=default_position,
        action_scale=action_scale,
        actor_observation_dim=actor_dim,
        critic_observation_dim=actor_dim + 3 + 6 * len(foot_names),
        action_dim=len(joint_names),
        phase_period=float(
            data["observations"]["actor"]["terms"]["phase"]["params"]["period"]
        ),
        physics_dt=float(physics["physics_dt"]),
        control_dt=control_dt,
        max_episode_steps=round(float(physics["episode_length_s"]) / control_dt),
    )
