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

import os
import json
import pytest
import tempfile


def test_rl_training():
    """Test RL training pipeline by running a few iterations."""
    # Load the existing push_cube config
    train_config_path = "configs/agents/rl/push_cube/train_config.json"
    gym_config_path = "configs/agents/rl/push_cube/gym_config.json"

    with open(train_config_path, "r") as f:
        train_config = json.load(f)

    with open(gym_config_path, "r") as f:
        gym_config = json.load(f)

    # Add dataset configuration dynamically to gym_config
    gym_config["env"]["dataset"] = {
        "lerobot": {
            "func": "LeRobotRecorder",
            "mode": "save",
            "params": {
                "robot_meta": {
                    "robot_type": "UR10_DH_Gripper",
                    "control_freq": 25,
                    "arm_dofs": 6,
                    "observation": {"vision": {}, "states": ["qpos"]},
                },
                "instruction": {"lang": "push_cube_to_target"},
                "extra": {
                    "scene_type": "tabletop",
                    "task_description": "push_cube_rl_test",
                },
                "use_videos": False,
                "export_success_only": False,
            },
        }
    }

    # Create temporary gym config with dataset
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(gym_config, f)
        temp_gym_config_path = f.name

    # Create temporary train config with reduced iterations for testing
    test_train_config = train_config.copy()
    test_train_config["trainer"][
        "gym_config"
    ] = temp_gym_config_path  # Point to temp gym config
    test_train_config["trainer"]["iterations"] = 2  # Only 2 iterations for testing
    test_train_config["trainer"]["rollout_steps"] = 32  # Fewer rollout steps
    test_train_config["trainer"]["eval_freq"] = 1000000  # Disable eval
    test_train_config["trainer"]["save_freq"] = 1000000  # Disable save
    test_train_config["trainer"]["headless"] = True
    test_train_config["trainer"]["use_wandb"] = False
    test_train_config["trainer"]["num_envs"] = 2  # Only 2 parallel envs for testing

    # Save temporary train config
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(test_train_config, f)
        temp_train_config_path = f.name

    try:
        from embodichain.agents.rl.train import train_from_config

        train_from_config(temp_train_config_path)
        assert True

    finally:
        # Clean up temporary files
        if os.path.exists(temp_train_config_path):
            os.remove(temp_train_config_path)
        if os.path.exists(temp_gym_config_path):
            os.remove(temp_gym_config_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
