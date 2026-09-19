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

import pytest
import torch

from embodichain.lab.gym.envs import EmbodiedEnv


def _env(explicit_ids: list[int] | None = None) -> EmbodiedEnv:
    env = object.__new__(EmbodiedEnv)
    env.cfg = SimpleNamespace(
        control_parts=["legs"] if explicit_ids is None else [],
        active_joint_ids=[] if explicit_ids is None else explicit_ids,
    )
    env.robot = SimpleNamespace(
        control_parts={"legs": [0, 2]},
        active_joint_ids=[0, 1, 2],
        dof=3,
        get_joint_ids=lambda **kwargs: [0, 2],
        build_pk_serial_chain=Mock(),
        body_data=SimpleNamespace(qpos_limits=torch.tensor([[[-1.0, 1.0]] * 3])),
    )
    return env


@pytest.fixture(autouse=True)
def _clean_class_joint_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(EmbodiedEnv, "active_joint_ids", [])


def test_second_environment_preserves_first_joint_ids() -> None:
    first, second = _env(), _env()
    first._setup_robot()
    second._setup_robot()
    assert first.active_joint_ids == second.active_joint_ids == [0, 2]
    assert first.active_joint_ids is not second.active_joint_ids
    assert first.single_action_space.shape == second.single_action_space.shape == (2,)


def test_repeated_setup_does_not_duplicate_joints() -> None:
    env = _env()
    env._setup_robot()
    env._setup_robot()
    assert env.active_joint_ids == [0, 2]
    assert env.single_action_space.shape == (2,)


def test_explicit_joint_ids_are_owned_by_environment() -> None:
    configured = [0, 2]
    env = _env(configured)
    env._setup_robot()
    configured.append(1)
    assert env.active_joint_ids == [0, 2]
