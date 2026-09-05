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
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import gymnasium.spaces
import numpy as np
import pytest
import torch

from tensordict import TensorDict

from embodichain.lab.task_program.integrations import IntegrationFingerprintMismatch
from embodichain.lab.gym.utils.registration import get_env_spec
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
from embodichain.lab.sim.sensors import CameraCfg
from embodichain.utils.utility import load_config, save_config

_REPOSITORY_ROOT = Path(__file__).parents[3]
_CUBE_GYM_CONFIG_PATH = (
    _REPOSITORY_ROOT
    / "embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.ur5.yaml"
)
_CUBE_TASK_PROGRAM_DIR = _CUBE_GYM_CONFIG_PATH.parent / "task_program"
_CUBE_INTEGRATION_PATH = _CUBE_TASK_PROGRAM_DIR / "integration.yaml"
_CUBE_ENVIRONMENT_PATH = _CUBE_GYM_CONFIG_PATH.parent / "env.yaml"
_COMPONENT_ROOT = _REPOSITORY_ROOT / "embodichain_tasks/configs/components"
_CUBE_POLICY_PATH = _COMPONENT_ROOT / "execution_policies/trajectory_open_loop.yaml"
_CUBE_EMBODIMENT_PATH = _COMPONENT_ROOT / "embodiments/ur5_dh_pgi_140_80.yaml"
_TABLEWARE_CONFIG_ROOT = (
    _REPOSITORY_ROOT / "embodichain_tasks/configs/tasks/manipulation/tableware"
)
CUBE_ROBOT_PROFILE_ID = "ur5_dh_pgi_140_80"
CUBE_SCENE_REGISTRY_ID = "task_program_repeated_pick_place"


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
        assert buffer["segment_accepted"].shape == (4, 100)
        assert not buffer["segment_accepted"].any()
        assert (buffer["segment_attempt_id"] == -1).all()
        assert (buffer["continuity_id"] == -1).all()

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


def test_launcher_seed_overrides_gym_config() -> None:
    """The common launcher exposes an explicit task-environment seed override."""
    parser = argparse.ArgumentParser()
    add_env_launcher_args_to_parser(parser, require_gym_config=True)

    args = parser.parse_args(["--gym_config", "gym_config.yaml", "--seed", "1234"])
    merged_config = merge_args_with_gym_config(args, {"seed": 99})

    assert merged_config["seed"] == 1234


def test_launcher_preserves_config_seed_without_override() -> None:
    """Omitting ``--seed`` keeps the value declared by the task config."""
    parser = argparse.ArgumentParser()
    add_env_launcher_args_to_parser(parser, require_gym_config=True)

    args = parser.parse_args(["--gym_config", "gym_config.yaml"])
    merged_config = merge_args_with_gym_config(args, {"seed": 99})

    assert merged_config["seed"] == 99


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
    @staticmethod
    def _minimal_gym_config() -> dict[str, object]:
        """Return a minimal config that reaches the generic parser."""
        return {
            "id": "TaskProgramRepeatedPickPlace-v1",
            "environment": {"component": "env.yaml"},
            "task_program": {
                "program": "program.yaml",
                "integration": "integration.yaml",
                "execution_policy": "policy.yaml",
            },
            "embodiment": {"component": "embodiment.yaml"},
        }

    @staticmethod
    def _task_program_payload() -> dict[str, object]:
        """Return one minimal strict Task Program payload."""
        return {
            "program_id": "configured_pick",
            "targets": {},
            "program": {
                "kind": "invoke",
                "call": {"kind": "pick", "object": "cube"},
            },
        }

    @staticmethod
    def _environment_component_payload() -> dict[str, object]:
        """Return one reusable physical environment component."""
        simulation = load_config(_CUBE_ENVIRONMENT_PATH)["simulation"]
        return {
            "environment_id": "configured_pick",
            "max_episodes": 5,
            "max_episode_steps": 1200,
            "simulation": simulation,
            "env": {},
        }

    @staticmethod
    def _environment_component_gym_config() -> dict[str, object]:
        """Return one runnable deployment selecting a reusable environment."""
        return {
            "id": "TaskProgramRepeatedPickPlace-v1",
            "environment": {"component": "env.yaml"},
            "task_program": {
                "program": "program.yaml",
                "integration": "integration.yaml",
                "execution_policy": "policy.yaml",
            },
            "embodiment": {"component": "embodiment.yaml"},
        }

    @classmethod
    def _write_deployment(
        cls,
        directory: Path,
        *,
        include_program: bool = True,
        include_integration: bool = True,
    ) -> Path:
        """Write one test-owned component deployment."""
        directory.mkdir(parents=True, exist_ok=True)
        if include_program:
            save_config(directory / "program.yaml", cls._task_program_payload())
        if include_integration:
            integration = load_config(_CUBE_INTEGRATION_PATH)
            integration["program_id"] = "configured_pick"
            save_config(
                directory / "integration.yaml",
                integration,
            )
        save_config(directory / "policy.yaml", load_config(_CUBE_POLICY_PATH))
        save_config(
            directory / "embodiment.yaml",
            load_config(_CUBE_EMBODIMENT_PATH),
        )
        save_config(directory / "env.yaml", cls._environment_component_payload())
        return directory

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

    def test_handwritten_config_composes_embodiment_without_task_program(
        self,
        tmp_path,
    ) -> None:
        """Physical embodiment reuse is independent of Task Program."""
        save_config(
            tmp_path / "embodiment.yaml",
            {
                "embodiment_id": "test_ur5",
                "simulation": {
                    "class_type": "URRobot",
                    "robot_type": "ur5",
                    "uid": "TestUR5",
                },
                "sensor": [],
            },
        )
        config = {
            "id": "EmbodiedEnv-v1",
            "env": {},
            "embodiment": {
                "component": "embodiment.yaml",
                "overrides": {"uid": "OverriddenUR5"},
            },
        }
        original = deepcopy(config)

        cfg = config_to_cfg(
            config,
            manager_modules=DEFAULT_MANAGER_MODULES,
            source_path=tmp_path / "env.yaml",
        )

        assert isinstance(cfg.robot, URRobotCfg)
        assert cfg.robot.robot_type == "ur5"
        assert cfg.robot.uid == "OverriddenUR5"
        assert cfg.sensor == []
        assert cfg.task_program is None
        assert config == original

    def test_handwritten_config_composes_environment_without_task_program(
        self,
        tmp_path,
    ) -> None:
        """Physical environment reuse is independent of Task Program."""
        self._write_deployment(tmp_path)
        config = {
            "id": "EmbodiedEnv-v1",
            "environment": {"component": "env.yaml"},
            "embodiment": {"component": "embodiment.yaml"},
        }

        cfg = config_to_cfg(
            config,
            manager_modules=DEFAULT_MANAGER_MODULES,
            source_path=tmp_path / "task.handwritten.yaml",
        )

        assert cfg.max_episode_steps == 1200
        assert [rigid.uid for rigid in cfg.rigid_object] == ["cube"]
        assert cfg.task_program is None

    def test_removed_task_component_selector_fails_closed(self) -> None:
        """The former mixed-ownership component cannot be silently ignored."""
        config = {
            "id": "EmbodiedEnv-v1",
            "task": {"component": "task.yaml"},
            "env": {},
            "robot": {"uid": "TestRobot"},
        }

        with pytest.raises(ValueError, match="task.component has been removed"):
            config_to_cfg(config, manager_modules=DEFAULT_MANAGER_MODULES)

    @pytest.mark.parametrize("field_name", ("robot", "sensor"))
    def test_handwritten_config_rejects_mixed_embodiment_ownership(
        self,
        tmp_path,
        field_name: str,
    ) -> None:
        """Inline and component-owned physical fields cannot conflict."""
        save_config(
            tmp_path / "embodiment.yaml",
            {
                "embodiment_id": "test_robot",
                "simulation": {"uid": "TestRobot"},
                "sensor": [],
            },
        )
        config = {
            "id": "EmbodiedEnv-v1",
            "env": {},
            "embodiment": {"component": "embodiment.yaml"},
            field_name: {} if field_name == "robot" else [],
        }

        with pytest.raises(ValueError, match="embodiment.component owns"):
            config_to_cfg(
                config,
                manager_modules=DEFAULT_MANAGER_MODULES,
                source_path=tmp_path / "env.yaml",
            )

    def test_handwritten_config_composes_scene_without_task_program(
        self,
        tmp_path,
    ) -> None:
        """A handwritten task may also select a reusable physical scene."""
        save_config(
            tmp_path / "scene.yaml",
            {
                "scene_id": "empty_lit_scene",
                "simulation": {"light": {"direct": []}},
            },
        )
        config = {
            "id": "EmbodiedEnv-v1",
            "env": {},
            "robot": {"uid": "TestRobot"},
            "scene": {"component": "scene.yaml"},
        }

        cfg = config_to_cfg(
            config,
            manager_modules=DEFAULT_MANAGER_MODULES,
            source_path=tmp_path / "env.yaml",
        )

        assert cfg.robot.uid == "TestRobot"
        assert cfg.light.direct == []
        assert cfg.task_program is None

    def test_handwritten_config_rejects_mixed_scene_ownership(
        self,
        tmp_path,
    ) -> None:
        """A scene component cannot silently replace an inline scene field."""
        save_config(
            tmp_path / "scene.yaml",
            {
                "scene_id": "component_scene",
                "simulation": {"light": {"direct": []}},
            },
        )
        config = {
            "id": "EmbodiedEnv-v1",
            "env": {},
            "robot": {"uid": "TestRobot"},
            "scene": {"component": "scene.yaml"},
            "light": {"direct": []},
        }

        with pytest.raises(ValueError, match="scene.component fields"):
            config_to_cfg(
                config,
                manager_modules=DEFAULT_MANAGER_MODULES,
                source_path=tmp_path / "env.yaml",
            )

    def test_deployment_resolves_components_from_its_own_directory(
        self,
        tmp_path,
    ) -> None:
        """All task-deployment component paths share one relative base."""
        deployment_dir = self._write_deployment(tmp_path / "deployment")
        gym_dir = tmp_path / "gym"
        gym_dir.mkdir()
        config = self._environment_component_gym_config()
        config["environment"] = {"component": "../deployment/env.yaml"}
        task_program = config["task_program"]
        assert type(task_program) is dict
        for field_name in ("program", "integration", "execution_policy"):
            task_program[field_name] = f"../deployment/{task_program[field_name]}"
        config["embodiment"] = {"component": "../deployment/embodiment.yaml"}

        cfg = config_to_cfg(
            config,
            manager_modules=DEFAULT_MANAGER_MODULES,
            source_path=gym_dir / "task.ur5.yaml",
        )

        assert deployment_dir.is_dir()
        assert cfg.task_program.program_id == "configured_pick"
        assert cfg.task_program.integration.scene_registry == CUBE_SCENE_REGISTRY_ID
        assert cfg.max_episode_steps == 1200

    def test_launcher_returns_environment_component_runtime_values(
        self,
        tmp_path,
    ) -> None:
        """The launcher exposes environment-owned run controls after composition."""
        self._write_deployment(tmp_path)
        gym_path = tmp_path / "task.ur5.yaml"
        save_config(gym_path, self._environment_component_gym_config())
        args = argparse.Namespace(
            gym_config=str(gym_path),
            num_envs=1,
            device="cpu",
            headless=True,
            renderer=None,
            gpu_id=0,
            arena_space=2.0,
            max_episodes=None,
            filter_visual_rand=False,
            filter_dataset_saving=False,
            preview=False,
            action_config=None,
        )

        _, gym_config, _ = build_env_cfg_from_args(args)

        assert gym_config["max_episodes"] == 5
        assert gym_config["env"] == {}
        assert "environment" not in gym_config

    def test_environment_component_rejects_separate_scene(self, tmp_path) -> None:
        """A deployment cannot select two owners for its physical scene."""
        self._write_deployment(tmp_path)
        save_config(
            tmp_path / "scene.yaml",
            {
                "scene_id": "duplicate_scene",
                "simulation": {"light": {"direct": []}},
            },
        )
        config = self._environment_component_gym_config()
        config["scene"] = {"component": "scene.yaml"}

        with pytest.raises(ValueError, match="environment.component owns"):
            config_to_cfg(
                config,
                manager_modules=DEFAULT_MANAGER_MODULES,
                source_path=tmp_path / "task.ur5.yaml",
            )

    def test_scene_binding_rejects_unknown_physical_entity(self, tmp_path) -> None:
        """Semantic roots must bind an entity declared by the physical scene."""
        self._write_deployment(tmp_path)
        integration_path = tmp_path / "integration.yaml"
        integration = load_config(integration_path)
        integration["scene_binding"]["rigid_objects"][0][
            "simulation_uid"
        ] = "missing_cube"
        save_config(integration_path, integration)

        with pytest.raises(ValueError, match="missing_cube.*physical environment"):
            config_to_cfg(
                self._minimal_gym_config(),
                manager_modules=DEFAULT_MANAGER_MODULES,
                source_path=tmp_path / "task.ur5.yaml",
            )

    @pytest.mark.parametrize(
        "task_name",
        (
            "blocks_ranking_rgb",
            "blocks_ranking_size",
            "match_object_container",
            "place_object_drawer",
            "stack_blocks_two",
            "stack_cups",
        ),
    )
    def test_official_handwritten_config_uses_shared_embodiment(
        self,
        task_name: str,
    ) -> None:
        """CobotMagic handwritten demos share one exact physical component."""
        config_path = _TABLEWARE_CONFIG_ROOT / task_name / "env.json"
        config = load_config(config_path)

        assert config["embodiment"] == {
            "component": "../../../../components/embodiments/cobotmagic.yaml"
        }
        assert "robot" not in config
        assert "sensor" not in config

        cfg = config_to_cfg(
            config,
            manager_modules=DEFAULT_MANAGER_MODULES,
            source_path=config_path,
        )

        assert cfg.robot.uid == "CobotMagic"
        assert [sensor.uid for sensor in cfg.sensor] == [
            "cam_high",
            "cam_right_wrist",
            "cam_left_wrist",
        ]
        assert all(type(sensor) is CameraCfg for sensor in cfg.sensor)

    @pytest.mark.parametrize("field_name", ("robot", "sensor"))
    def test_configured_task_program_rejects_top_level_embodiment_fields(
        self,
        tmp_path,
        field_name: str,
    ) -> None:
        """A deployment obtains both robot and sensors from its embodiment."""
        self._write_deployment(tmp_path)
        config = self._minimal_gym_config()
        config[field_name] = {} if field_name == "robot" else []

        with pytest.raises(ValueError, match="embodiment.component"):
            config_to_cfg(
                config,
                manager_modules=DEFAULT_MANAGER_MODULES,
                source_path=tmp_path / "task.ur5.yaml",
            )

    def test_embodiment_requires_an_explicit_sensor_suite(self, tmp_path) -> None:
        """Even an empty sensor suite is an explicit embodiment-owned field."""
        self._write_deployment(tmp_path)
        embodiment_path = tmp_path / "embodiment.yaml"
        embodiment = load_config(embodiment_path)
        embodiment.pop("sensor")
        save_config(embodiment_path, embodiment)
        config = self._minimal_gym_config()

        with pytest.raises(ValueError, match="missing required fields.*sensor"):
            config_to_cfg(
                config,
                manager_modules=DEFAULT_MANAGER_MODULES,
                source_path=tmp_path / "env.yaml",
            )

    def test_task_program_requires_embodiment_skill_profile(self, tmp_path) -> None:
        """A Task Program embodiment must expose semantic capabilities."""
        self._write_deployment(tmp_path)
        component_path = tmp_path / "embodiment.yaml"
        component = load_config(component_path)
        component.pop("skill_profile")
        save_config(component_path, component)

        with pytest.raises(
            ValueError,
            match="embodiment component must declare skill_profile",
        ):
            config_to_cfg(
                self._minimal_gym_config(),
                manager_modules=DEFAULT_MANAGER_MODULES,
                source_path=tmp_path / "env.yaml",
            )

    def test_task_integration_requires_scene_binding(self, tmp_path) -> None:
        """Every task integration owns one semantic scene binding."""
        self._write_deployment(tmp_path)
        integration_path = tmp_path / "integration.yaml"
        integration = load_config(integration_path)
        integration.pop("scene_binding")
        save_config(integration_path, integration)

        with pytest.raises(
            ValueError,
            match="task integration is missing required fields.*scene_binding",
        ):
            config_to_cfg(
                self._minimal_gym_config(),
                manager_modules=DEFAULT_MANAGER_MODULES,
                source_path=tmp_path / "env.yaml",
            )

    def test_task_program_rejects_separate_scene_binding_component(
        self,
        tmp_path,
    ) -> None:
        """The removed three-path format fails instead of being ignored."""
        self._write_deployment(tmp_path)
        config = self._minimal_gym_config()
        task_program = config["task_program"]
        assert type(task_program) is dict
        task_program["scene_binding"] = "scene_binding.yaml"

        with pytest.raises(ValueError, match="unsupported fields.*scene_binding"):
            config_to_cfg(
                config,
                manager_modules=DEFAULT_MANAGER_MODULES,
                source_path=tmp_path / "env.yaml",
            )

    def test_environment_component_rejects_task_program_metadata(
        self,
        tmp_path,
    ) -> None:
        """Reusable environments cannot own Task Program selections."""
        self._write_deployment(tmp_path)
        environment_path = tmp_path / "env.yaml"
        environment = load_config(environment_path)
        integration = load_config(tmp_path / "integration.yaml")
        environment["task_program"] = integration["scene_binding"]
        save_config(environment_path, environment)

        with pytest.raises(ValueError, match="unsupported fields.*task_program"):
            config_to_cfg(
                self._minimal_gym_config(),
                manager_modules=DEFAULT_MANAGER_MODULES,
                source_path=tmp_path / "task.ur5.yaml",
            )

    def test_embodiment_rejects_removed_task_program_metadata_name(
        self,
        tmp_path,
    ) -> None:
        """The former ambiguous embodiment metadata key is not accepted."""
        self._write_deployment(tmp_path)
        component_path = tmp_path / "embodiment.yaml"
        component = load_config(component_path)
        component["task_program"] = component.pop("skill_profile")
        save_config(component_path, component)

        with pytest.raises(ValueError, match="unsupported fields.*task_program"):
            config_to_cfg(
                self._minimal_gym_config(),
                manager_modules=DEFAULT_MANAGER_MODULES,
                source_path=tmp_path / "env.yaml",
            )

    def test_environment_rejects_embodiment_owned_sensors(self, tmp_path) -> None:
        """Environments cannot silently add or replace embodiment sensors."""
        self._write_deployment(tmp_path)
        environment_path = tmp_path / "env.yaml"
        environment = load_config(environment_path)
        environment["simulation"]["sensor"] = []
        save_config(environment_path, environment)
        config = self._minimal_gym_config()

        with pytest.raises(ValueError, match="unsupported fields.*sensor"):
            config_to_cfg(
                config,
                manager_modules=DEFAULT_MANAGER_MODULES,
                source_path=tmp_path / "task.ur5.yaml",
            )

    def test_components_are_resolved_from_gym_config_source(
        self,
        tmp_path,
    ) -> None:
        """All component paths are relative to their Gym config file."""
        gym_dir = tmp_path / "gym" / "task"
        gym_dir.mkdir(parents=True)
        self._write_deployment(tmp_path / "deployment")
        gym_path = gym_dir / "gym_config.json"
        config = self._minimal_gym_config()
        task_program = config["task_program"]
        assert type(task_program) is dict
        for field_name in (
            "program",
            "integration",
            "execution_policy",
        ):
            task_program[field_name] = f"../../deployment/{task_program[field_name]}"
        config["environment"] = {"component": "../../deployment/env.yaml"}
        config["embodiment"] = {"component": "../../deployment/embodiment.yaml"}

        cfg = config_to_cfg(
            config,
            manager_modules=DEFAULT_MANAGER_MODULES,
            source_path=gym_path,
        )

        assert cfg.task_program.program_id == "configured_pick"
        assert cfg.task_program.integration.scene_registry == CUBE_SCENE_REGISTRY_ID
        assert get_env_spec(str(config["id"])).task_program_registration is not None

    def test_build_env_cfg_loads_source_relative_task_program(
        self,
        tmp_path,
    ) -> None:
        """The normal file launcher attaches the decoded program before init."""
        gym_dir = tmp_path / "gym"
        gym_dir.mkdir()
        self._write_deployment(tmp_path / "deployment")
        gym_path = gym_dir / "gym_config.json"
        config = self._minimal_gym_config()
        task_program = config["task_program"]
        assert type(task_program) is dict
        for field_name in (
            "program",
            "integration",
            "execution_policy",
        ):
            task_program[field_name] = f"../deployment/{task_program[field_name]}"
        config["environment"] = {"component": "../deployment/env.yaml"}
        config["embodiment"] = {"component": "../deployment/embodiment.yaml"}
        save_config(gym_path, config)
        args = argparse.Namespace(
            gym_config=str(gym_path),
            num_envs=1,
            device="cpu",
            headless=True,
            renderer=None,
            gpu_id=0,
            arena_space=2.0,
            max_episodes=None,
            filter_visual_rand=False,
            filter_dataset_saving=False,
            preview=False,
            action_config=None,
        )

        cfg, _, _ = build_env_cfg_from_args(args)

        assert cfg.task_program.program_id == "configured_pick"

    def test_cli_program_override_is_selected_and_loaded_once(
        self,
        tmp_path,
        monkeypatch,
    ) -> None:
        """The CLI override replaces the Gym path at the single loader boundary."""
        from embodichain.lab.task_program.language import loader

        gym_path = tmp_path / "gym_config.json"
        override_path = tmp_path / "override.yaml"
        save_config(override_path, self._task_program_payload())
        deployment_dir = self._write_deployment(
            tmp_path / "deployment",
            include_program=False,
        )
        (deployment_dir / "program.yaml").write_text(
            "this configured program must not be loaded\n",
            encoding="utf-8",
        )
        config = self._minimal_gym_config()
        task_program = config["task_program"]
        assert type(task_program) is dict
        for field_name in (
            "program",
            "integration",
            "execution_policy",
        ):
            task_program[field_name] = str(deployment_dir / task_program[field_name])
        config["environment"] = {"component": str(deployment_dir / "env.yaml")}
        config["embodiment"] = {"component": str(deployment_dir / "embodiment.yaml")}
        save_config(gym_path, config)
        args = argparse.Namespace(
            gym_config=str(gym_path),
            task_program=str(override_path),
            num_envs=1,
            device="cpu",
            headless=True,
            renderer=None,
            gpu_id=0,
            arena_space=2.0,
            max_episodes=None,
            filter_visual_rand=False,
            filter_dataset_saving=False,
            preview=False,
            action_config=None,
        )
        calls: list[str] = []
        original = loader.load_task_program

        def load_once(path, **kwargs):
            calls.append(str(path))
            return original(path, **kwargs)

        monkeypatch.setattr(loader, "load_task_program", load_once)

        cfg, _, _ = build_env_cfg_from_args(args)

        assert cfg.task_program.program_id == "configured_pick"
        assert calls == [str(override_path)]

    def test_registration_drift_fails_during_repeated_config_load(
        self,
        tmp_path,
        monkeypatch,
    ) -> None:
        """A repeated config load rejects drift in its registered integration."""
        from embodichain.lab.task_program.language import loader

        deployment_dir = self._write_deployment(tmp_path / "deployment")
        program_path = deployment_dir / "program.yaml"
        config = self._minimal_gym_config()
        task_program = config["task_program"]
        assert type(task_program) is dict
        for field_name in (
            "program",
            "integration",
            "execution_policy",
        ):
            task_program[field_name] = str(deployment_dir / task_program[field_name])
        config["environment"] = {"component": str(deployment_dir / "env.yaml")}
        config["embodiment"] = {"component": str(deployment_dir / "embodiment.yaml")}
        config_to_cfg(config, manager_modules=DEFAULT_MANAGER_MODULES)
        registration = get_env_spec(str(config["id"])).task_program_registration
        assert registration is not None
        binding = registration.scene_binding.rigid_objects[0]
        original_semantic_type = binding.semantic_type
        object.__setattr__(binding, "semantic_type", "changed_cube")
        loader_calls: list[str] = []
        original_load = loader.load_task_program

        def tracked_load(path, **kwargs):
            loader_calls.append(str(path))
            return original_load(path, **kwargs)

        monkeypatch.setattr(loader, "load_task_program", tracked_load)

        try:
            with pytest.raises(IntegrationFingerprintMismatch, match="changed"):
                config_to_cfg(config, manager_modules=DEFAULT_MANAGER_MODULES)
        finally:
            object.__setattr__(binding, "semantic_type", original_semantic_type)

        assert loader_calls == [str(program_path)]

    def test_config_to_cfg_uses_cwd_without_source_path(
        self,
        tmp_path,
        monkeypatch,
    ) -> None:
        """Dictionary-only callers retain explicit current-directory semantics."""
        self._write_deployment(tmp_path)
        monkeypatch.chdir(tmp_path)
        config = self._minimal_gym_config()

        cfg = config_to_cfg(config, manager_modules=DEFAULT_MANAGER_MODULES)

        assert cfg.task_program.program_id == "configured_pick"

    @pytest.mark.parametrize("value", [None, True, 1, {}, "", " program.yaml"])
    def test_component_path_rejects_ambiguous_values(
        self,
        tmp_path,
        value,
    ) -> None:
        """A component path rejects coercion, null, and outer whitespace."""
        self._write_deployment(tmp_path)
        config = self._minimal_gym_config()
        task_program = config["task_program"]
        assert type(task_program) is dict
        task_program["program"] = value

        with pytest.raises((TypeError, ValueError), match="task_program.program"):
            config_to_cfg(
                config,
                manager_modules=DEFAULT_MANAGER_MODULES,
                source_path=tmp_path / "env.yaml",
            )

    @pytest.mark.parametrize(
        "removed_field",
        [
            "task_program_dir",
            "task_program_path",
            "task_program_integration_path",
        ],
    )
    def test_removed_task_program_path_fields_fail_closed(
        self,
        removed_field: str,
    ) -> None:
        """Removed bundle and two-path formats fail instead of being ignored."""
        config = self._minimal_gym_config()
        config[removed_field] = "removed.yaml"

        with pytest.raises(ValueError, match="component mapping"):
            config_to_cfg(config, manager_modules=DEFAULT_MANAGER_MODULES)

    @pytest.mark.parametrize(
        ("field_name", "filename"),
        (
            ("program", "program.yaml"),
            ("integration", "integration.yaml"),
            ("execution_policy", "policy.yaml"),
        ),
    )
    def test_missing_component_fails_before_environment_init(
        self,
        tmp_path,
        field_name: str,
        filename: str,
    ) -> None:
        """Every selected component must be an existing regular file."""
        deployment = self._write_deployment(tmp_path)
        (deployment / filename).unlink()
        config = self._minimal_gym_config()

        with pytest.raises(FileNotFoundError, match=f"task_program.{field_name}"):
            config_to_cfg(
                config,
                manager_modules=DEFAULT_MANAGER_MODULES,
                source_path=tmp_path / "env.yaml",
            )

    def test_integration_component_requires_yaml_extension(
        self,
        tmp_path,
    ) -> None:
        """Task integrations use one explicit YAML component format."""
        self._write_deployment(tmp_path)
        save_config(
            tmp_path / "integration.json",
            load_config(_CUBE_INTEGRATION_PATH),
        )
        config = self._minimal_gym_config()
        task_program = config["task_program"]
        assert type(task_program) is dict
        task_program["integration"] = "integration.json"

        with pytest.raises(ValueError, match="task_program.integration"):
            config_to_cfg(
                config,
                manager_modules=DEFAULT_MANAGER_MODULES,
                source_path=tmp_path / "env.yaml",
            )

    def test_task_integration_rejects_removed_version_field(
        self,
        tmp_path,
    ) -> None:
        """Component declarations contain no compatibility version field."""
        self._write_deployment(tmp_path)
        integration = load_config(tmp_path / "integration.yaml")
        integration["version"] = 1
        save_config(tmp_path / "integration.yaml", integration)
        config = self._minimal_gym_config()

        with pytest.raises(ValueError, match="unsupported fields.*version"):
            config_to_cfg(
                config,
                manager_modules=DEFAULT_MANAGER_MODULES,
                source_path=tmp_path / "env.yaml",
            )

    @pytest.mark.parametrize("suffix", ["yaml", "json"])
    @pytest.mark.parametrize("enabled", [None, False, True])
    def test_gym_config_preserves_entity_gizmo_startup_preference(
        self, tmp_path: Path, suffix: str, enabled: bool | None
    ) -> None:
        """Task deployments default to native interaction and can opt out."""
        config = {
            "id": "EmbodiedEnv-v1",
            "env": {},
            "robot": {"uid": "TestRobot"},
        }
        if enabled is not None:
            config["enable_entity_gizmo"] = enabled
        config_path = tmp_path / f"gym_config.{suffix}"
        save_config(config_path, config)

        cfg = config_to_cfg(
            load_config(config_path), manager_modules=DEFAULT_MANAGER_MODULES
        )

        assert cfg.sim_cfg.enable_entity_gizmo is (enabled is not False)

    @pytest.mark.parametrize("suffix", ["yaml", "json"])
    @pytest.mark.parametrize(
        "settings", [None, {}, {"ik_solver": "embodichain"}, {"ik_start_enabled": True}]
    )
    def test_gym_config_parses_automatic_robot_gizmo_settings(
        self, tmp_path: Path, suffix: str, settings: dict | None
    ) -> None:
        """Deployments may disable automatic IK controls or select their solver."""
        path = tmp_path / f"gym_config.{suffix}"
        save_config(
            path,
            {
                "id": "EmbodiedEnv-v1",
                "env": {},
                "robot": {"uid": "robot"},
                "robot_ik_gizmo": settings,
            },
        )
        cfg = config_to_cfg(load_config(path), manager_modules=DEFAULT_MANAGER_MODULES)
        if settings is None:
            assert cfg.sim_cfg.robot_ik_gizmo is None
        else:
            assert cfg.sim_cfg.robot_ik_gizmo.ik_solver == settings.get(
                "ik_solver", "dexsim"
            )
            assert cfg.sim_cfg.robot_ik_gizmo.ik_start_enabled is settings.get(
                "ik_start_enabled", False
            )

    def test_yaml_gym_config_parses_to_cfg(self, tmp_path):
        config = {
            "id": "EmbodiedEnv-v1",
            "seed": 2026,
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
                "events": {
                    "global_light": {
                        "func": "randomize_emission_light",
                        "mode": "interval",
                        "interval_step": 7,
                        "is_global": True,
                        "params": {"intensity_range": [0.1, 0.9]},
                    }
                },
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
        assert cfg.seed == 2026
        assert cfg.events.global_light.interval_step == 7
        assert cfg.events.global_light.is_global is True
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
