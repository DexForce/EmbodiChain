# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
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

"""Tests for gym_utils module."""

from __future__ import annotations

import pytest
import torch

from tensordict import TensorDict

from embodichain.lab.gym.utils.gym_utils import init_rollout_buffer_from_config


class TestInitRolloutBufferFromConfig:
    """Tests for init_rollout_buffer_from_config function."""

    def test_basic_rollout_buffer(self):
        """Test that basic rollout buffer is created correctly."""
        config = {
            "sensor": [],
            "env": {},
        }

        buffer = init_rollout_buffer_from_config(
            config=config,
            max_episode_steps=100,
            batch_size=4,
            state_dim=7,
            device="cpu",
        )

        assert isinstance(buffer, TensorDict)
        assert buffer.batch_size == torch.Size([4, 100])

        # Check obs structure
        assert "obs" in buffer
        assert "robot" in buffer["obs"]
        assert "qpos" in buffer["obs"]["robot"]
        assert "qvel" in buffer["obs"]["robot"]
        assert "qf" in buffer["obs"]["robot"]

        # Check shapes
        assert buffer["obs"]["robot"]["qpos"].shape == (4, 100, 7)
        assert buffer["obs"]["robot"]["qvel"].shape == (4, 100, 7)
        assert buffer["obs"]["robot"]["qf"].shape == (4, 100, 7)

        # Check actions and rewards
        assert buffer["actions"].shape == (4, 100, 7)
        assert buffer["rewards"].shape == (4, 100)

    def test_extra_observation_with_shape_tuple(self):
        """Test that extra observations with shape tuple are added correctly."""
        config = {
            "sensor": [],
            "env": {
                "observations": {
                    "extra_position": {
                        "mode": "add",
                        "extra": {"shape": (3,)},
                    }
                }
            },
        }

        buffer = init_rollout_buffer_from_config(
            config=config,
            max_episode_steps=100,
            batch_size=4,
            state_dim=7,
            device="cpu",
        )

        assert "extra_position" in buffer["obs"]
        assert buffer["obs"]["extra_position"].shape == (4, 100, 3)
        assert buffer["obs"]["extra_position"].dtype == torch.float32

    def test_extra_observation_with_shape_list(self):
        """Test that extra observations with shape list are added correctly."""
        config = {
            "sensor": [],
            "env": {
                "observations": {
                    "extra_pose": {
                        "mode": "add",
                        "extra": {"shape": [7]},
                    }
                }
            },
        }

        buffer = init_rollout_buffer_from_config(
            config=config,
            max_episode_steps=100,
            batch_size=4,
            state_dim=7,
            device="cpu",
        )

        assert "extra_pose" in buffer["obs"]
        assert buffer["obs"]["extra_pose"].shape == (4, 100, 7)

    def test_extra_observation_multidimensional_shape(self):
        """Test that extra observations with multi-dimensional shape work."""
        config = {
            "sensor": [],
            "env": {
                "observations": {
                    "extra_image": {
                        "mode": "add",
                        "extra": {"shape": (64, 64, 3)},
                    }
                }
            },
        }

        buffer = init_rollout_buffer_from_config(
            config=config,
            max_episode_steps=100,
            batch_size=4,
            state_dim=7,
            device="cpu",
        )

        assert "extra_image" in buffer["obs"]
        assert buffer["obs"]["extra_image"].shape == (4, 100, 64, 64, 3)

    def test_multiple_extra_observations(self):
        """Test that multiple extra observations are all added correctly."""
        config = {
            "sensor": [],
            "env": {
                "observations": {
                    "extra_pos": {
                        "mode": "add",
                        "extra": {"shape": [3]},
                    },
                    "extra_vel": {
                        "mode": "add",
                        "extra": {"shape": [6]},
                    },
                }
            },
        }

        buffer = init_rollout_buffer_from_config(
            config=config,
            max_episode_steps=50,
            batch_size=2,
            state_dim=7,
            device="cpu",
        )

        assert "extra_pos" in buffer["obs"]
        assert "extra_vel" in buffer["obs"]
        assert buffer["obs"]["extra_pos"].shape == (2, 50, 3)
        assert buffer["obs"]["extra_vel"].shape == (2, 50, 6)

    def test_modify_mode_observation_ignored(self):
        """Test that observations in 'modify' mode are not added as extra observations."""
        config = {
            "sensor": [],
            "env": {
                "observations": {
                    "modified_obs": {
                        "mode": "modify",
                        "extra": {"shape": [5]},
                    }
                }
            },
        }

        buffer = init_rollout_buffer_from_config(
            config=config,
            max_episode_steps=100,
            batch_size=4,
            state_dim=7,
            device="cpu",
        )

        # modified_obs should NOT be in the buffer since mode is 'modify'
        assert "modified_obs" not in buffer["obs"]

    def test_extra_observation_without_shape_ignored(self):
        """Test that extra observations without shape are ignored."""
        config = {
            "sensor": [],
            "env": {
                "observations": {
                    "obs_no_shape": {
                        "mode": "add",
                        "extra": {"other_key": "value"},
                    }
                }
            },
        }

        buffer = init_rollout_buffer_from_config(
            config=config,
            max_episode_steps=100,
            batch_size=4,
            state_dim=7,
            device="cpu",
        )

        # obs_no_shape should NOT be in the buffer since no shape is provided
        assert "obs_no_shape" not in buffer["obs"]

    def test_extra_observation_with_nested_name(self):
        """Test that extra observations with nested names (using '/') are handled."""
        config = {
            "sensor": [],
            "env": {
                "observations": {
                    "custom/group1/value": {
                        "mode": "add",
                        "extra": {"shape": [4]},
                    }
                }
            },
        }

        buffer = init_rollout_buffer_from_config(
            config=config,
            max_episode_steps=100,
            batch_size=4,
            state_dim=7,
            device="cpu",
        )

        # Nested name should be handled by assign_data_to_dict
        assert "custom" in buffer["obs"]
        assert "group1" in buffer["obs"]["custom"]
        assert "value" in buffer["obs"]["custom"]["group1"]
        assert buffer["obs"]["custom"]["group1"]["value"].shape == (4, 100, 4)

    def test_sensor_and_extra_obs_together(self):
        """Test that both sensors and extra observations work together."""
        config = {
            "sensor": [
                {
                    "uid": "camera",
                    "width": 320,
                    "height": 240,
                    "enable_mask": True,
                }
            ],
            "env": {
                "observations": {
                    "extra_vec": {
                        "mode": "add",
                        "extra": {"shape": [10]},
                    }
                }
            },
        }

        buffer = init_rollout_buffer_from_config(
            config=config,
            max_episode_steps=100,
            batch_size=4,
            state_dim=7,
            device="cpu",
        )

        # Check sensor is present
        assert "sensor" in buffer["obs"]
        assert "camera" in buffer["obs"]["sensor"]
        assert buffer["obs"]["sensor"]["camera"]["color"].shape == (4, 100, 240, 320, 4)
        assert buffer["obs"]["sensor"]["camera"]["mask"].shape == (4, 100, 240, 320)

        # Check extra obs is present
        assert "extra_vec" in buffer["obs"]
        assert buffer["obs"]["extra_vec"].shape == (4, 100, 10)

    def test_different_batch_sizes(self):
        """Test that batch_size correctly affects extra observations."""
        config = {
            "sensor": [],
            "env": {
                "observations": {
                    "extra_data": {
                        "mode": "add",
                        "extra": {"shape": [5]},
                    }
                }
            },
        }

        buffer = init_rollout_buffer_from_config(
            config=config,
            max_episode_steps=50,
            batch_size=8,
            state_dim=7,
            device="cpu",
        )

        assert buffer["obs"]["extra_data"].shape == (8, 50, 5)

    def test_different_max_episode_steps(self):
        """Test that max_episode_steps correctly affects extra observations."""
        config = {
            "sensor": [],
            "env": {
                "observations": {
                    "extra_data": {
                        "mode": "add",
                        "extra": {"shape": [2]},
                    }
                }
            },
        }

        buffer = init_rollout_buffer_from_config(
            config=config,
            max_episode_steps=200,
            batch_size=4,
            state_dim=7,
            device="cpu",
        )

        assert buffer["obs"]["extra_data"].shape == (4, 200, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
