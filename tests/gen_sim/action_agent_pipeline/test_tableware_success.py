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

import pytest
import torch

from embodichain.gen_sim.action_agent_pipeline.env_adapters.tableware.success import (
    evaluate_configured_success,
)


class _FakeSim:
    def get_rigid_object(self, uid: str):
        return None


class _FakeEnv:
    num_envs = 1
    device = torch.device("cpu")
    sim = _FakeSim()


def test_success_unknown_rigid_object_uid_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="Unknown rigid object uid: 'missing'"):
        evaluate_configured_success(
            _FakeEnv(),
            {
                "type": "object_xy_near",
                "object": "missing",
                "target_xy": [0.0, 0.0],
            },
        )
