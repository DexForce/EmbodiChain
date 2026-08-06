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

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv
from embodichain.lab.gym.envs.managers.cfg import DatasetFunctorCfg
from embodichain.lab.gym.envs.managers.dataset_manager import DatasetManager
from embodichain.lab.gym.envs.managers.record import record_camera_data
from embodichain.lab.gym.utils.profiler import EnvProfiler


class DatasetManagerStub:
    """Minimal dataset manager used to exercise episode selection."""

    available_modes = ["save"]

    def __init__(self, save_failed_episodes: bool) -> None:
        self.save_failed_episodes = save_failed_episodes
        self.saved_env_ids: torch.Tensor | None = None

    def apply(self, mode: str, env_ids: torch.Tensor) -> None:
        assert mode == "save"
        self.saved_env_ids = env_ids.clone()


def make_env_for_episode_selection(
    *, save_failed_episodes: bool, successful_env_ids: list[int]
) -> tuple[SimpleNamespace, DatasetManagerStub]:
    num_envs = 3
    success_status = torch.zeros(num_envs, dtype=torch.bool)
    success_status[successful_env_ids] = True
    manager = DatasetManagerStub(save_failed_episodes)
    env = SimpleNamespace(
        num_envs=num_envs,
        dataset_manager=manager,
        episode_success_status=success_status,
        _task_success=torch.zeros(num_envs, dtype=torch.bool),
        # _initialize_episode opens profiler sections around each manager call;
        # a disabled EnvProfiler is a no-op so the stub can exercise the save
        # logic without a real BaseEnv (which would need a sim).
        _profiler=EnvProfiler(None, torch.device("cpu")),
        cfg=SimpleNamespace(
            events=None,
            observations=None,
            rewards=None,
            dataset=None,
        ),
        event_manager=None,
        observation_manager=None,
        reward_manager=None,
        rollout_buffer=None,
    )
    return env, manager


def test_save_failed_episodes_reads_typed_functor_config() -> None:
    def recorder(env, env_ids) -> None:
        pass

    manager = DatasetManager.__new__(DatasetManager)
    manager._mode_functor_cfgs = {
        "save": [
            DatasetFunctorCfg(
                func=recorder,
                save_failed_episodes=True,
            )
        ]
    }

    assert manager.save_failed_episodes is True


def test_save_failed_episodes_defaults_to_false() -> None:
    def recorder(env, env_ids) -> None:
        pass

    functor_cfg = DatasetFunctorCfg(func=recorder)

    assert functor_cfg.save_failed_episodes is False


def test_apply_forwards_only_functor_params() -> None:
    calls = []

    def recorder(env, env_ids, *, use_videos: bool) -> None:
        calls.append((env, env_ids, use_videos))

    env = object()
    env_ids = torch.tensor([0])
    manager = DatasetManager.__new__(DatasetManager)
    manager._env = env
    manager._mode_functor_names = {"save": ["recorder"]}
    manager._mode_functor_cfgs = {
        "save": [
            DatasetFunctorCfg(
                func=recorder,
                save_failed_episodes=True,
                params={"use_videos": True},
            )
        ]
    }

    manager.apply(mode="save", env_ids=env_ids)

    assert calls == [(env, env_ids, True)]


def test_initialize_episode_limits_successes_to_reset_envs() -> None:
    env, manager = make_env_for_episode_selection(
        save_failed_episodes=False,
        successful_env_ids=[0, 1],
    )

    EmbodiedEnv._initialize_episode(env, env_ids=[1, 2])

    assert torch.equal(manager.saved_env_ids, torch.tensor([1]))


def test_initialize_episode_saves_failed_reset_envs_when_enabled() -> None:
    env, manager = make_env_for_episode_selection(
        save_failed_episodes=True,
        successful_env_ids=[1],
    )

    EmbodiedEnv._initialize_episode(env, env_ids=[1, 2])

    assert torch.equal(manager.saved_env_ids, torch.tensor([1, 2]))


def test_discard_reset_does_not_auto_save_trajectory() -> None:
    """save_data=False clears trajectory state without writing a file."""
    env, _ = make_env_for_episode_selection(
        save_failed_episodes=False,
        successful_env_ids=[],
    )
    env._traj_buffer = object()
    env._traj_steps = torch.tensor([3, 2, 1])
    env.cfg.trajectory_auto_save = True
    env._save_trajectory_for_env = MagicMock()

    EmbodiedEnv._initialize_episode(env, env_ids=[0], save_data=False)

    env._save_trajectory_for_env.assert_not_called()
    assert env._traj_steps.tolist() == [0, 2, 1]


def test_discard_reset_clears_camera_frames_without_saving() -> None:
    """save_data=False drops video frames through the recorder abort hook."""
    env, _ = make_env_for_episode_selection(
        save_failed_episodes=False,
        successful_env_ids=[],
    )
    recorder = record_camera_data.__new__(record_camera_data)
    recorder._frames = [object()]
    recorder.save_and_clear = MagicMock()
    env.cfg.events = object()
    env.event_manager = SimpleNamespace(
        _mode_functor_cfgs={"interval": [SimpleNamespace(func=recorder)]},
        available_modes=[],
    )

    EmbodiedEnv._initialize_episode(env, env_ids=[0], save_data=False)

    recorder.save_and_clear.assert_not_called()
    assert recorder._frames == []
