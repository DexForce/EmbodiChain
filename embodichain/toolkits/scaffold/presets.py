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

from __future__ import annotations

import json
from typing import Any

from embodichain.toolkits.scaffold.spec import RobotPreset, TaskSpec

ROBOT_PRESETS: dict[RobotPreset, dict[str, Any]] = {
    "cobot_magic": {
        "robot": {
            "uid": "CobotMagic",
            "robot_type": "CobotMagic",
            "init_pos": [0.0, 0.0, 0.7775],
            "init_qpos": [
                -0.3,
                0.3,
                1.0,
                1.0,
                -1.2,
                -1.2,
                0.0,
                0.0,
                0.6,
                0.6,
                0.0,
                0.0,
                0.05,
                0.05,
                0.05,
                0.05,
            ],
        },
        "sensor": [
            {
                "sensor_type": "Camera",
                "uid": "cam_high",
                "width": 640,
                "height": 480,
                "intrinsics": [
                    488.1665344238281,
                    488.1665344238281,
                    322.7323303222656,
                    213.17434692382812,
                ],
                "extrinsics": {"eye": [1.0, 0.0, 2.0], "target": [0.0, 0.0, 1.0]},
            }
        ],
        "light": {
            "indirect": {
                "emission_light": {"color": [1.0, 1.0, 1.0], "intensity": 150}
            },
            "direct": [
                {
                    "uid": "light_1",
                    "light_type": "point",
                    "color": [1.0, 1.0, 1.0],
                    "intensity": 20.0,
                    "init_pos": [0.0, 0.0, 3.0],
                    "radius": 10.0,
                }
            ],
        },
    },
    "ur5_minimal": {
        "robot": {
            "uid": "UR5",
            "fpath": "UniversalRobots/UR5/UR5.urdf",
            "init_pos": [0.0, 0.0, 0.7775],
            "init_qpos": [
                1.57079,
                -1.57079,
                1.57079,
                -1.57079,
                -1.57079,
                -3.14159,
            ],
        },
        "sensor": [
            {
                "sensor_type": "Camera",
                "uid": "cam_high",
                "width": 640,
                "height": 480,
                "intrinsics": [488.1665344238281, 488.1665344238281, 320.0, 240.0],
                "extrinsics": {"eye": [1.0, 0.0, 3.0], "target": [0.0, 0.0, 1.0]},
            }
        ],
        "light": {
            "indirect": {
                "emission_light": {"color": [1.0, 1.0, 1.0], "intensity": 150}
            },
            "direct": [
                {
                    "uid": "light_1",
                    "light_type": "point",
                    "color": [1.0, 1.0, 1.0],
                    "intensity": 20.0,
                    "init_pos": [0.0, 0.0, 3.0],
                    "radius": 10.0,
                }
            ],
        },
    },
}


def build_gym_config(spec: TaskSpec) -> dict[str, Any]:
    """Build a minimal gym JSON config for the task."""
    preset = ROBOT_PRESETS[spec.robot_preset]
    env_block: dict[str, Any] = {"events": {}, "observations": {}}

    if spec.workflow == "demo":
        env_block["events"]["record_camera"] = {
            "func": "record_camera_data",
            "mode": "interval",
            "interval_step": 1,
            "params": {
                "name": "cam1",
                "resolution": [320, 240],
                "eye": [2.0, 0.0, 2.0],
                "target": [0.5, 0.0, 1.0],
            },
        }
    elif spec.workflow == "rl" and spec.reward_style == "json":
        env_block["events"] = {
            "randomize_object": {
                "func": "randomize_rigid_object_pose",
                "mode": "reset",
                "params": {
                    "entity_cfg": {"uid": "object"},
                    "position_range": [[-0.1, -0.1, 0.0], [0.1, 0.1, 0.0]],
                    "relative_position": True,
                },
            }
        }
        env_block["observations"] = {
            "robot_qpos": {
                "func": "normalize_robot_joint_data",
                "mode": "modify",
                "name": "robot/qpos",
                "params": {},
            }
        }
        env_block["rewards"] = {
            "task_reward": {
                "func": "success_reward",
                "mode": "add",
                "weight": 1.0,
                "params": {},
            }
        }
        env_block["actions"] = {
            "delta_qpos": {
                "func": "DeltaQposTerm",
                "params": {"scale": 0.1},
            }
        }
        env_block["extensions"] = {"success_threshold": 0.05}

    config: dict[str, Any] = {
        "id": spec.gym_id,
        "max_episodes": spec.max_episodes,
        "max_episode_steps": spec.max_episode_steps,
        "env": env_block,
        "robot": preset["robot"],
        "sensor": preset["sensor"],
        "light": preset["light"],
        "background": [],
        "rigid_object": [],
        "rigid_object_group": [],
        "articulation": [],
    }

    if spec.workflow == "rl":
        config["rigid_object"] = [
            {
                "uid": "object",
                "shape": {"type": "box", "size": [0.05, 0.05, 0.05]},
                "physical_attr": {
                    "mass": 0.1,
                    "static_friction": 0.5,
                    "dynamic_friction": 0.5,
                },
                "init_pos": [0.5, 0.0, 0.05],
            }
        ]

    return config


def gym_config_to_json(spec: TaskSpec, *, indent: int = 4) -> str:
    return json.dumps(build_gym_config(spec), indent=indent) + "\n"
