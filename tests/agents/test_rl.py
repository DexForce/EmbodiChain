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
    config_path = "configs/agents/rl/push_cube/train_config.json"

    with open(config_path, "r") as f:
        config = json.load(f)

    # Create temporary config with reduced iterations for testing
    test_config = config.copy()
    test_config["trainer"]["iterations"] = 2  # Only 2 iterations for testing
    test_config["trainer"]["rollout_steps"] = 32  # Fewer rollout steps
    test_config["trainer"]["eval_freq"] = 1000000  # Disable eval
    test_config["trainer"]["save_freq"] = 1000000  # Disable save
    test_config["trainer"]["headless"] = True
    test_config["trainer"]["use_wandb"] = False
    test_config["trainer"]["num_envs"] = 2  # Only 2 parallel envs for testing

    # Save temporary config
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(test_config, f)
        temp_config_path = f.name

    try:
        from embodichain.agents.rl.train import train_from_config

        train_from_config(temp_config_path)
        assert True

    finally:
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
