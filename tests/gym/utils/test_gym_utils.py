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
    add_env_launcher_args_to_parser,
    build_env_cfg_from_args,
    build_trajectory_buffer,
    config_to_cfg,
    DEFAULT_MANAGER_MODULES,
    load_trajectory,
    merge_args_with_gym_config,
    init_rollout_buffer_from_config,
)
from embodichain.lab.sim.robots import URRobotCfg
from embodichain.utils.utility import load_config, save_config


def test_env_launcher_args_include_physics():
    """Test that launcher args expose the physics backend config selector."""
    parser = argparse.ArgumentParser()
    add_env_launcher_args_to_parser(parser)

    default_args = parser.parse_args([])
    assert default_args.physics == "default"

    newton_args = parser.parse_args(["--physics", "newton"])
    assert newton_args.physics == "newton"


def test_merge_args_with_gym_config_includes_physics():
    """Test that CLI physics config overrides the gym config."""
    parser = argparse.ArgumentParser()
    add_env_launcher_args_to_parser(parser)
    args = parser.parse_args(["--physics", "newton"])

    merged_config = merge_args_with_gym_config(args, {})

    assert merged_config["physics"] == "newton"


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
        physics="default",
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
        physics="default",
        gpu_id=0,
        arena_space=5.0,
        max_episodes=None,
    )
    gym_config = {"max_episodes": 3, "id": "Dummy-v0"}

    merged_config = merge_args_with_gym_config(args, gym_config)

    assert merged_config["max_episodes"] == 3


def test_merge_args_with_gym_config_enables_headless_viser():
    """The --viser options are translated into simulation configuration."""
    args = argparse.Namespace(
        num_envs=4,
        device="cpu",
        headless=False,
        renderer=None,
        gpu_id=0,
        arena_space=5.0,
        max_episodes=None,
        viser=True,
        viser_host="0.0.0.0",
        viser_port=9000,
        viser_fps=12.5,
        viser_image_fps=1.5,
        viser_soft_body_fps=4.0,
        viser_env_ids=[1, 3],
    )

    merged_config = merge_args_with_gym_config(args, {"id": "Dummy-v0"})

    assert merged_config["headless"] is True
    assert merged_config["visualization"] == {
        "backend": "viser",
        "scene_fps": 12.5,
        "sensor_image_fps": 1.5,
        "soft_body_fps": 4.0,
        "env_ids": [1, 3],
        "allow_commands": True,
        "viser_server": {"host": "0.0.0.0", "port": 9000},
    }


def test_merge_args_with_gym_config_accepts_all_viser_environments():
    """The all selector is preserved as an unbounded environment selection."""
    args = argparse.Namespace(
        num_envs=1024,
        device="cpu",
        headless=False,
        renderer=None,
        gpu_id=0,
        arena_space=5.0,
        max_episodes=None,
        viser=True,
        viser_host="127.0.0.1",
        viser_port=8080,
        viser_fps=15.0,
        viser_image_fps=2.0,
        viser_soft_body_fps=5.0,
        viser_env_ids=["all"],
    )

    merged_config = merge_args_with_gym_config(args, {"id": "Dummy-v0"})

    assert merged_config["visualization"]["env_ids"] is None


def test_launcher_preserves_gym_renderer_when_cli_omits_override():
    """A required gym config supplies the renderer unless CLI overrides it."""
    parser = argparse.ArgumentParser()
    add_env_launcher_args_to_parser(parser, require_gym_config=True)

    args = parser.parse_args(["--gym_config", "gym_config.yaml"])
    gym_config = {"id": "Dummy-v0", "render_cfg": {"renderer": "rt"}}
    merged_config = merge_args_with_gym_config(args, gym_config)

    assert args.renderer is None
    assert "renderer" not in merged_config
    assert merged_config["render_cfg"]["renderer"] == "rt"


def test_env_launcher_includes_viser_arguments():
    """The common environment launcher registers Viser options by default."""
    parser = argparse.ArgumentParser()
    add_env_launcher_args_to_parser(parser)

    args = parser.parse_args(
        [
            "--viser",
            "--viser-host",
            "0.0.0.0",
            "--viser-port",
            "9000",
            "--viser-fps",
            "12.5",
            "--viser-image-fps",
            "1.5",
            "--viser-soft-body-fps",
            "4.0",
            "--viser-env-ids",
            "1",
            "3",
        ]
    )

    assert args.viser is True
    assert not hasattr(args, "viser_gizmo")
    assert args.viser_host == "0.0.0.0"
    assert args.viser_port == 9000
    assert args.viser_fps == 12.5
    assert args.viser_image_fps == 1.5
    assert args.viser_soft_body_fps == 4.0
    assert args.viser_env_ids == [1, 3]


def test_viser_launcher_flag_implies_headless_viser_commands():
    parser = argparse.ArgumentParser()
    add_env_launcher_args_to_parser(parser)

    args = parser.parse_args(["--viser"])
    merged_config = merge_args_with_gym_config(args, {"id": "Dummy-v0"})

    assert merged_config["headless"] is True
    assert merged_config["visualization"]["backend"] == "viser"
    assert merged_config["visualization"]["allow_commands"] is True


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
    def test_robot_class_type_preserves_ur_variant(self):
        config = {
            "id": "EmbodiedEnv-v1",
            "env": {},
            "robot": {
                "class_type": "URRobot",
                "robot_type": "ur5",
                "uid": "TestUR5",
            },
        }

        cfg = config_to_cfg(config, manager_modules=DEFAULT_MANAGER_MODULES)

        assert isinstance(cfg.robot, URRobotCfg)
        assert cfg.robot.robot_type == "ur5"
        assert cfg.robot.uid == "TestUR5"
        assert cfg.robot.solver_cfg["arm"].ur_type == "ur5"
        assert config["robot"] == {
            "class_type": "URRobot",
            "robot_type": "ur5",
            "uid": "TestUR5",
        }

    def test_yaml_gym_config_parses_to_cfg(self, tmp_path):
        config = {
            "id": "EmbodiedEnv-v1",
            "max_episode_steps": 100,
            "physics_config": {
                "gravity": [0.0, 0.0, -1.62],
                "bounce_threshold": 1.5,
                "enable_ccd": True,
                "length_tolerance": 0.02,
                "speed_tolerance": 0.1,
            },
            "render_cfg": {
                "renderer": "rt",
                "spp": 4,
                "tone_mapping_enabled": True,
                "tone_mapping_exposure": 1.25,
            },
            "visualization": {
                "backend": "viser",
                "scene_fps": 12.5,
                "viser_server": {
                    "host": "0.0.0.0",
                    "port": 9000,
                },
            },
            "env": {
                "sim_steps_per_control": 2,
                "target_control_frequency": 20.0,
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
        assert cfg.sim_steps_per_control == 2
        assert cfg.target_control_frequency == 20.0
        np.testing.assert_array_equal(
            cfg.sim_cfg.physics_config.gravity, [0.0, 0.0, -1.62]
        )
        assert cfg.sim_cfg.physics_config.bounce_threshold == 1.5
        assert cfg.sim_cfg.physics_config.enable_ccd is True
        assert cfg.sim_cfg.physics_config.length_tolerance == 0.02
        assert cfg.sim_cfg.physics_config.speed_tolerance == 0.1
        assert cfg.sim_cfg.render_cfg.renderer == "rt"
        assert cfg.sim_cfg.render_cfg.spp == 4
        assert cfg.sim_cfg.render_cfg.tone_mapping_enabled is True
        assert cfg.sim_cfg.render_cfg.tone_mapping_exposure == 1.25
        assert cfg.sim_cfg.visualization.backend == "viser"
        assert cfg.sim_cfg.visualization.scene_fps == 12.5
        assert cfg.sim_cfg.visualization.viser_server.host == "0.0.0.0"
        assert cfg.sim_cfg.visualization.viser_server.port == 9000

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
            "physics_config": {
                "gravity": [0.0, 0.0, -3.71],
                "enable_ccd": True,
            },
            "render_cfg": {
                "renderer": "rt",
                "spp": 8,
                "tone_mapping_enabled": True,
            },
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
            renderer="fast-rt",
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
        np.testing.assert_array_equal(
            cfg.sim_cfg.physics_config.gravity, [0.0, 0.0, -3.71]
        )
        assert cfg.sim_cfg.physics_config.enable_ccd is True
        assert cfg.sim_cfg.render_cfg.renderer == "fast-rt"
        assert cfg.sim_cfg.render_cfg.spp == 8
        assert cfg.sim_cfg.render_cfg.tone_mapping_enabled is True

    @pytest.mark.parametrize(
        ("replay_mode", "expected"),
        [("control", True), ("dynamic", False)],
    )
    def test_control_replay_disables_dataset_saving(
        self, tmp_path, replay_mode, expected
    ):
        config = {
            "id": "EmbodiedEnv-v1",
            "max_episode_steps": 100,
            "env": {
                "events": {},
                "observations": {},
                "rewards": {},
                "dataset": {
                    "lerobot": {
                        "func": "LeRobotRecorder",
                        "mode": "save",
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
            replay=True,
            replay_mode=replay_mode,
            action_config=None,
        )

        cfg, _, _ = build_env_cfg_from_args(args)

        assert cfg.filter_dataset_saving is expected


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
    assert tuple(buf["states"]["robot"]["qvel"].shape) == (num_envs, 10, 6)
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
    assert tuple(buf["states"]["rigid_objects"]["cube"]["lin_vel"].shape) == (
        num_envs,
        10,
        3,
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
        "meta": {"num_steps": 4, "num_envs": 1, "lengths": [4]},
    }
    p = tmp_path / "traj.pt"
    torch.save(data, p)
    loaded = load_trajectory(str(p))
    assert loaded["meta"]["num_steps"] == 4

    with pytest.raises(ValueError):
        load_trajectory({"states": torch.zeros(1)})


def test_load_trajectory_rejects_misaligned_state_action_steps():
    num_envs = 1
    num_steps = 2
    states = TensorDict(
        {"robot": {"qpos": torch.zeros(num_envs, num_steps, 3)}},
        batch_size=[num_envs, num_steps],
    )
    data = {
        "states": states,
        "actions": torch.zeros(num_envs, num_steps + 1, 3),
        "meta": {
            "num_steps": num_steps,
            "num_envs": num_envs,
            "lengths": [num_steps],
        },
    }

    with pytest.raises(ValueError, match="actions shape"):
        load_trajectory(data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
