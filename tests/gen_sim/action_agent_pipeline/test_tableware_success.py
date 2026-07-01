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
    def __init__(self, objects=None) -> None:
        self.objects = objects or {}

    def get_rigid_object(self, uid: str):
        return self.objects.get(uid)


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


class _FakeObject:
    def __init__(self, position) -> None:
        self.position = torch.as_tensor(position, dtype=torch.float32).reshape(1, 3)

    def get_local_pose(self, to_matrix: bool = True):
        pose = torch.eye(4, dtype=torch.float32).reshape(1, 4, 4)
        pose[:, :3, 3] = self.position
        return pose


def test_object_lifted_requires_initial_height() -> None:
    env = _FakeEnv()
    env.sim = _FakeSim({"apple": _FakeObject([0.0, 0.0, 0.2])})
    env.obj_info = {}

    with pytest.raises(ValueError, match="requires an initial height"):
        evaluate_configured_success(
            env,
            {
                "type": "object_lifted",
                "object": "apple",
                "min_height": 0.1,
            },
        )


def test_object_lifted_uses_recorded_initial_height() -> None:
    env = _FakeEnv()
    env.sim = _FakeSim({"apple": _FakeObject([0.0, 0.0, 0.25])})
    env.obj_info = {"apple": {"height": torch.tensor(0.1)}}

    success = evaluate_configured_success(
        env,
        {
            "type": "object_lifted",
            "object": "apple",
            "min_height": 0.1,
        },
    )

    assert success.tolist() == [True]
