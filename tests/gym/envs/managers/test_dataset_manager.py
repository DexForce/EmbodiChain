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
from unittest.mock import Mock

import torch

from embodichain.lab.gym.envs import EmbodiedEnv
from embodichain.lab.gym.envs.managers.dataset_manager import DatasetManager


def test_apply_does_not_forward_save_failed_episode_policy() -> None:
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
            SimpleNamespace(
                func=recorder,
                params={"save_failed_episodes": True, "use_videos": True},
            )
        ]
    }

    manager.apply(mode="save", env_ids=env_ids)

    assert calls == [(env, env_ids, True)]


def _make_episode_env(current_rollout_step: int) -> SimpleNamespace:
    dataset_manager = Mock()
    dataset_manager.available_modes = ["save"]
    dataset_manager.save_failed_episodes = True
    return SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        current_rollout_step=current_rollout_step,
        dataset_manager=dataset_manager,
        cfg=SimpleNamespace(
            events=None,
            observations=None,
            rewards=None,
            dataset=None,
        ),
        event_manager=None,
        observation_manager=None,
        reward_manager=None,
        rollout_buffer={},
        _rollout_buffer_mode="expert",
        episode_success_status=torch.tensor([False]),
        _task_success=torch.tensor([False]),
    )


def test_initialize_episode_skips_empty_failed_episode() -> None:
    env = _make_episode_env(current_rollout_step=0)

    EmbodiedEnv._initialize_episode(env, env_ids=[0])

    env.dataset_manager.apply.assert_not_called()


def test_initialize_episode_saves_recorded_failed_episode_when_enabled() -> None:
    env = _make_episode_env(current_rollout_step=1)

    EmbodiedEnv._initialize_episode(env, env_ids=[0])

    env.dataset_manager.apply.assert_called_once()
    call = env.dataset_manager.apply.call_args
    assert call.kwargs["mode"] == "save"
    assert call.kwargs["env_ids"].tolist() == [0]
