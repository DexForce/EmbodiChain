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

import torch

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
