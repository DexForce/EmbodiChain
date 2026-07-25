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

import argparse
from types import SimpleNamespace

import gymnasium.spaces
import numpy as np
import pytest
import torch

from tensordict import TensorDict

from embodichain.lab.gym.utils.gym_utils import (
    build_env_cfg_from_args,
    build_trajectory_buffer,
    config_to_cfg,
    DEFAULT_MANAGER_MODULES,
    load_trajectory,
    merge_args_with_gym_config,
    init_rollout_buffer_from_config,
)
from embodichain.utils.utility import load_config, save_config


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


def test_merge_args_with_gym_config_overrides_max_episodes():
    """Test that CLI max_episodes overrides the gym config value."""
    args = argparse.Namespace(
        num_envs=1,
        device="cpu",
        headless=False,
        renderer="auto",
        gpu_id=0,
        arena_space=5.0,
        max_episodes=12,
    )
    gym_config = {"max_episodes": 3, "id": "Dummy-v0"}

    merged_config = merge_args_with_gym_config(args, gym_config)

    assert merged_config["max_episodes"] == 12
    assert gym_config["max_episodes"] == 3


def test_merge_args_with_gym_config_keeps_default_max_episodes():
    """Test that gym config max_episodes is preserved when CLI omits it."""
    args = argparse.Namespace(
        num_envs=1,
        device="cpu",
        headless=False,
        renderer="auto",
        gpu_id=0,
        arena_space=5.0,
        max_episodes=None,
    )
    gym_config = {"max_episodes": 3, "id": "Dummy-v0"}

    merged_config = merge_args_with_gym_config(args, gym_config)

    assert merged_config["max_episodes"] == 3


def test_sensor_and_extra_obs_together():
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


def test_different_batch_sizes():
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


def test_different_max_episode_steps():
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


class TestConfigToCfgFromFile:
    def test_yaml_gym_config_parses_to_cfg(self, tmp_path):
        config = {
            "id": "EmbodiedEnv-v1",
            "max_episode_steps": 100,
            "env": {
                "events": {},
                "observations": {},
                "rewards": {},
            },
            "robot": {
                "uid": "TestRobot",
                "urdf_cfg": {
                    "components": [
                        {
                            "component_type": "arm",
                            "urdf_path": "UniversalRobots/UR5/UR5.urdf",
                        }
                    ]
                },
                "init_pos": [0.0, 0.0, 0.0],
                "init_rot": [0.0, 0.0, 0.0],
                "init_qpos": [0.0] * 6,
            },
        }

        config_path = tmp_path / "gym_config.yaml"
        save_config(config_path, config)

        loaded = load_config(config_path)
        cfg = config_to_cfg(loaded, manager_modules=DEFAULT_MANAGER_MODULES)

        assert cfg.max_episode_steps == 100
        assert cfg.robot.uid == "TestRobot"

    def test_json_dataset_save_failed_episodes_parses_from_top_level(self, tmp_path):
        config = {
            "id": "EmbodiedEnv-v1",
            "env": {
                "events": {},
                "observations": {},
                "rewards": {},
                "dataset": {
                    "lerobot": {
                        "func": "LeRobotRecorder",
                        "mode": "save",
                        "save_failed_episodes": True,
                        "params": {},
                    }
                },
            },
            "robot": {
                "uid": "TestRobot",
                "urdf_cfg": {
                    "components": [
                        {
                            "component_type": "arm",
                            "urdf_path": "UniversalRobots/UR5/UR5.urdf",
                        }
                    ]
                },
                "init_pos": [0.0, 0.0, 0.0],
                "init_rot": [0.0, 0.0, 0.0],
                "init_qpos": [0.0] * 6,
            },
        }

        config_path = tmp_path / "gym_config.json"
        save_config(config_path, config)

        loaded = load_config(config_path)
        cfg = config_to_cfg(loaded, manager_modules=DEFAULT_MANAGER_MODULES)

        assert cfg.dataset.lerobot.save_failed_episodes is True
        assert "save_failed_episodes" not in cfg.dataset.lerobot.params

    def test_build_env_cfg_applies_modifier_before_parsing(self, tmp_path):
        config = {
            "id": "EmbodiedEnv-v1",
            "max_episode_steps": 100,
            "env": {"events": {}, "observations": {}, "rewards": {}},
            "robot": {
                "uid": "TestRobot",
                "urdf_cfg": {
                    "components": [
                        {
                            "component_type": "arm",
                            "urdf_path": "UniversalRobots/UR5/UR5.urdf",
                        }
                    ]
                },
                "init_pos": [0.0, 0.0, 0.0],
                "init_rot": [0.0, 0.0, 0.0],
                "init_qpos": [0.0] * 6,
            },
        }
        config_path = tmp_path / "gym_config.yaml"
        save_config(config_path, config)
        args = argparse.Namespace(
            gym_config=str(config_path),
            num_envs=1,
            device="cpu",
            headless=True,
            renderer="rasterization",
            gpu_id=0,
            arena_space=2.0,
            max_episodes=None,
            filter_visual_rand=False,
            filter_dataset_saving=False,
            preview=False,
            action_config=None,
        )

        cfg, merged_config, _ = build_env_cfg_from_args(
            args,
            gym_config_modifier=lambda value: value.update(max_episode_steps=321),
        )

        assert merged_config["max_episode_steps"] == 321
        assert cfg.max_episode_steps == 321


class _StubRobot:
    def __init__(self, dof: int):
        self.dof = dof
        self.uid = "robot"


class _StubArticulation:
    def __init__(self, dof: int, uid: str):
        self.dof = dof
        self.uid = uid


class _StubRigidObject:
    def __init__(self, uid: str):
        self.uid = uid


def _stub_env(robot_dof=6, articulations=None, rigid_objects=None):
    return SimpleNamespace(
        robot=_StubRobot(robot_dof),
        sim=SimpleNamespace(
            _articulations={
                uid: _StubArticulation(d, uid)
                for uid, d in (articulations or {}).items()
            },
            _rigid_objects={
                uid: _StubRigidObject(uid) for uid in (rigid_objects or [])
            },
        ),
    )


def test_build_trajectory_buffer_shapes():
    env = _stub_env(robot_dof=6, articulations={"drawer": 2}, rigid_objects=["cube"])
    num_envs = 3
    action_space = gymnasium.spaces.Box(
        low=-1, high=1, shape=(num_envs, 6), dtype=np.float32
    )
    buf = build_trajectory_buffer(
        env, max_steps=10, num_envs=num_envs, device="cpu", action_space=action_space
    )
    assert tuple(buf.batch_size) == (num_envs, 10)
    assert tuple(buf["states"]["robot"]["root_pose"].shape) == (num_envs, 10, 7)
    assert tuple(buf["states"]["robot"]["qpos"].shape) == (num_envs, 10, 6)
    assert tuple(buf["states"]["articulations"]["drawer"]["qpos"].shape) == (
        num_envs,
        10,
        2,
    )
    assert tuple(buf["states"]["rigid_objects"]["cube"]["pose"].shape) == (
        num_envs,
        10,
        7,
    )
    assert tuple(buf["actions"].shape) == (num_envs, 10, 6)


def test_build_trajectory_buffer_uids_filter():
    env = _stub_env(
        robot_dof=6,
        articulations={"drawer": 2, "door": 1},
        rigid_objects=["cube", "ball"],
    )
    buf = build_trajectory_buffer(
        env, max_steps=5, num_envs=1, device="cpu", uids=["cube"]
    )
    assert "articulations" not in buf["states"].keys()  # drawer/door filtered out
    assert "rigid_objects" in buf["states"].keys()
    assert "cube" in buf["states"]["rigid_objects"].keys()
    assert "ball" not in buf["states"]["rigid_objects"].keys()


def test_load_trajectory_validates_and_returns_dict(tmp_path):
    data = {
        "states": TensorDict({"a": torch.zeros(1, 4)}, batch_size=[1, 4]),
        "actions": torch.zeros(1, 4, 3),
        "meta": {"num_steps": 4, "num_envs": 1},
    }
    p = tmp_path / "traj.pt"
    torch.save(data, p)
    loaded = load_trajectory(str(p))
    assert loaded["meta"]["num_steps"] == 4

    with pytest.raises(ValueError):
        load_trajectory({"states": torch.zeros(1)})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
