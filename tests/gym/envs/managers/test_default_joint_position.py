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
import pytest
import torch
from embodichain.lab.gym.envs.managers import ActionManager, ActionTermCfg
from embodichain.lab.gym.envs.managers.actions import DefaultJointPositionTerm


def make_manager():
    robot = SimpleNamespace(
        joint_names=["unused", "a", "b"],
        cfg=SimpleNamespace(init_qpos=[9.0, 0.2, -0.4]),
    )
    env = SimpleNamespace(
        num_envs=2, device=torch.device("cpu"), active_joint_ids=[1, 2], robot=robot
    )
    cfg = ActionTermCfg(
        func=DefaultJointPositionTerm,
        params=dict(joint_names=["a", "b"], scale=[0.5, 2.0], clip=1.0),
    )
    return ActionManager({"joints": cfg}, env)


def test_joint_mapping_works_without_task_specific_attributes():
    manager = make_manager()
    term = manager.get_term("joints")
    term.position_bias[0] = torch.tensor([0.1, 0.2])
    target = term.process_action(torch.tensor([[2.0, -2.0], [0.0, 0.0]]))
    torch.testing.assert_close(target, torch.tensor([[0.6, -2.6], [0.2, -0.4]]))
    assert torch.equal(term.action, torch.tensor([[1.0, -1.0], [0.0, 0.0]]))


def test_manager_selective_reset_clears_action_history_only_in_selected_rows():
    manager = make_manager()
    term = manager.get_term("joints")
    term.position_bias.fill_(0.1)
    term.process_action(torch.ones(2, 2))
    term.process_action(torch.full((2, 2), 0.5))
    manager.reset(env_ids=[0])
    assert torch.equal(term.action, torch.tensor([[0.0, 0.0], [0.5, 0.5]]))
    assert torch.equal(term.previous_action, torch.tensor([[0.0, 0.0], [1.0, 1.0]]))
    assert torch.equal(term.position_bias, torch.full((2, 2), 0.1))
