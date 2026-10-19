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

"""Demo-local physical environment and controller configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

TASK_CONFIG = Path(__file__).resolve().parent / "configs/task.yaml"


def load_task_config(
    device: str, action_mode: str = "joint", pose_ik: str = "solver"
) -> dict[str, Any]:
    from embodichain.utils.utility import load_config
    from embodichain.lab.gym.utils._component_composition import _resolve_gym_components

    config = _resolve_gym_components(
        load_config(TASK_CONFIG), base_dir=TASK_CONFIG.parent
    ).config
    config.update(
        device=device,
        gpu_id=int(device.split(":")[-1]) if device.startswith("cuda:") else 0,
        headless=True,
        renderer="no-render",
        enable_sensor=False,
        num_envs=1,
    )
    if action_mode == "pose":
        config["env"].setdefault("extensions", {})["pick_place_action_mode"] = "pose"
        config["env"]["actions"] = {
            "arm": {
                "contract": "eef_pose.relative@1",
                "params": {
                    "part_name": "arm",
                    "translation_scale": 0.02,
                    "rotation_scale": 0.1,
                    "ik_method": pose_ik,
                },
            },
            "eef": {
                "contract": "gripper.parallel@1",
                "params": {
                    "part_name": "eef",
                    "command_mode": "continuous",
                    "lower_command": {".*": 0.0},
                    "upper_command": {".*": 0.04},
                },
            },
        }
    return config
