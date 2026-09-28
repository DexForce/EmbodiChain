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

"""interval_step==1 快路径：functor 收到显式全量 ID 张量（非 None）。"""

from __future__ import annotations

import torch

from embodichain.lab.gym.envs.managers.event_manager import EventManager


class _RecordingFunctor:
    """记录 env_ids 的哑 functor。"""

    def __init__(self):
        self.calls = []

    def __call__(self, env, env_ids, **kwargs):
        self.calls.append(None if env_ids is None else env_ids.clone())


def test_interval_one_fastpath_passes_explicit_all_row_ids(monkeypatch):
    """interval_step==1 且非 global：functor 收到全量 ID 张量（非 None）。"""
    manager = EventManager.__new__(EventManager)
    manager._env = object()
    manager._seed = None

    num_envs = 4096
    counter = torch.zeros(num_envs, dtype=torch.long, device="cpu")
    manager._interval_functor_step_count = [counter]

    class _Cfg:
        is_global = False
        interval_step = 1

    manager._mode_functor_names = {"interval": ["push_robot"]}
    manager._mode_functor_cfgs = {"interval": [_Cfg()]}

    # 拦截 _call_event_functor，记录传入的 env_ids
    received = []
    monkeypatch.setattr(
        manager,
        "_call_event_functor",
        lambda mode, name, cfg, env, ids: received.append(ids),
    )

    manager.apply(mode="interval")
    # 第二次调用必须复用缓存的全行 ID 张量,而不是每步重新 arange
    manager.apply(mode="interval")

    assert len(received) == 2
    ids = received[0]
    assert ids is not None, "快路径传了 None——functor 的 len(env_ids) 会崩"
    assert ids.shape[0] == num_envs
    assert ids.dtype == torch.long
    assert torch.equal(ids, torch.arange(num_envs, dtype=torch.long))
    assert received[1] is ids, "两次 apply 应复用 manager 缓存的同一 ID 张量"
