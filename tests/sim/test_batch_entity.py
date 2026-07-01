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

from embodichain.lab.sim.common import BatchEntity


class _BatchEntityForTest(BatchEntity):
    def __init__(self) -> None:
        self.reset_calls = 0
        cfg = SimpleNamespace(uid="test_entity")
        super().__init__(
            cfg=cfg,
            entities=[object()],
            device=torch.device("cpu"),
        )

    def set_local_pose(self, pose, env_ids=None) -> None:
        pass

    def get_local_pose(self, to_matrix: bool = False) -> torch.Tensor:
        return torch.empty(0)

    def reset(self, env_ids=None) -> None:
        self.reset_calls += 1


def test_batch_entity_does_not_reset_in_constructor() -> None:
    entity = _BatchEntityForTest()

    assert entity.reset_calls == 0


def test_batch_entity_reset_is_explicit() -> None:
    entity = _BatchEntityForTest()
    entity.reset()

    assert entity.reset_calls == 1
