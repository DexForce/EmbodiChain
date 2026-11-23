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
"""Scene inputs and reset handling for the Franka RJ45 insertion cycle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

if TYPE_CHECKING:
    import torch

    from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv

__all__ = ["cable_points", "load_config", "settle_franka_rj45", "task_directory"]

_TASK_CONFIG = "tasks/manipulation/cable/franka_rj45_cycle"
_JOINT_NAMES = (
    *(f"fr3_joint{index}" for index in range(1, 8)),
    "fr3_finger_joint1",
    "fr3_finger_joint2",
)


def task_directory() -> Path:
    """Locate the installed or source-checkout RJ45 task definitions.

    Returns:
        Directory containing the environment and its scenario inputs.
    """
    from embodichain.extensions.tasks.configs import get_config_path

    return get_config_path(f"{_TASK_CONFIG}/env.yaml").parent


def load_config() -> dict[str, Any]:
    """Read the portable parameters and original physical acceptance thresholds.

    Returns:
        Independent mapping of the calibrated RJ45 scenario inputs.
    """
    return json.loads((task_directory() / "config.json").read_text())


def cable_points(config: dict[str, Any]) -> np.ndarray:
    """Read the canonical centreline without resampling the measured scene.

    Args:
        config: RJ45 scene parameters, including the expected segment count.

    Returns:
        Original centreline points in world metres.
    """
    scene = yaml.safe_load((task_directory() / "env.yaml").read_text())["simulation"]
    (cable,) = scene["cable_object"]
    points = np.asarray(cable["points"], dtype=np.float32)
    if points.shape != (config["cable_segments"] + 1, 3):
        raise ValueError("RJ45 scene does not have the configured number of segments.")
    return points


def settle_franka_rj45(env: EmbodiedEnv, env_ids: torch.Tensor | None) -> None:
    """Advance the original pre-roll before starting the program clock.

    Args:
        env: The configured RJ45 environment.
        env_ids: IDs of the environments being reset.
    """
    import torch

    from .rj45_evaluation import attach_rj45_monitor

    if env_ids is not None and len(env_ids) != env.num_envs:
        raise ValueError(
            "Franka RJ45 pre-roll advances the world and needs a full reset."
        )
    if tuple(env.robot.joint_names) != _JOINT_NAMES:
        raise ValueError(f"Unexpected Franka joint order {env.robot.joint_names!r}.")
    config = load_config()
    env.franka_rj45_config = config
    cable = env.sim.get_cable_object("cable")
    anchor_baseline = cable.data.segment_pose_w[0, 0, :3].cpu().numpy().copy()
    home = [*config["home_qpos"], *([config["open_half_gap_m"]] * 2)]
    qpos = torch.tensor(home, dtype=torch.float32, device=env.device).expand(
        env.num_envs, -1
    )
    env.robot.set_qpos(qpos, target=True)
    env.robot.set_qvel(torch.zeros_like(qpos), target=True)
    steps = round(config["pre_roll_s"] * config["control_hz"] * config["substeps"])
    for _ in range(steps):
        env.sim.update(step=1, render_final_step=False)
    attach_rj45_monitor(env, config, anchor_baseline=anchor_baseline)
