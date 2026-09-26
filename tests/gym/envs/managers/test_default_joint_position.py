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

"""Tests for default-offset locomotion joint actions."""

from __future__ import annotations

import torch

from embodichain.lab.gym.envs.managers import (
    ActionManager,
    ActionTermCfg,
    resolve_action_contract,
)
from embodichain.lab.gym.envs.managers.actions import DefaultJointPositionAction

from action_test_utils import make_action_env


def make_manager() -> ActionManager:
    """Build a two-joint default-position manager."""
    env = make_action_env(
        num_envs=2,
        joint_names=("unused", "a", "b"),
        parts={"policy": (1, 2)},
    )
    env.robot.cfg.init_qpos = [9.0, 0.2, -0.4]
    cfg = ActionTermCfg(
        func=DefaultJointPositionAction,
        params={
            "joint_names": ["a", "b"],
            "scale": [0.5, 2.0],
            "clip": 1.0,
        },
    )
    return ActionManager({"joint_position": cfg}, env)


def test_joint_mapping_works_without_task_specific_attributes() -> None:
    """Default offset, scale, clipping, and bias remain generic."""
    manager = make_manager()
    term = manager.get_term("joint_position")
    term.position_bias[0] = torch.tensor([0.1, 0.2])

    manager.process_action(torch.tensor([[2.0, -2.0], [0.0, 0.0]]))

    torch.testing.assert_close(
        term.processed_actions,
        torch.tensor([[0.6, -2.6], [0.2, -0.4]]),
    )
    torch.testing.assert_close(
        term.raw_actions,
        torch.tensor([[1.0, -1.0], [0.0, 0.0]]),
    )


def test_manager_selective_reset_clears_action_history_only_in_selected_rows() -> None:
    """Locomotion history resets selected rows without clearing encoder bias."""
    manager = make_manager()
    term = manager.get_term("joint_position")
    term.position_bias.fill_(0.1)
    manager.process_action(torch.ones(2, 2))
    manager.process_action(torch.full((2, 2), 0.5))

    manager.reset(env_ids=[0])

    torch.testing.assert_close(
        term.raw_actions,
        torch.tensor([[0.0, 0.0], [0.5, 0.5]]),
    )
    torch.testing.assert_close(
        term.previous_raw_actions,
        torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
    )
    torch.testing.assert_close(term.position_bias, torch.full((2, 2), 0.1))


def test_default_position_contract_resolves_and_binds() -> None:
    """The stable contract identifies the current implementation and term."""
    manager = make_manager()

    assert (
        resolve_action_contract("joint_position.default_offset@1")
        is DefaultJointPositionAction
    )
    assert manager.get_term_by_contract(
        "joint_position.default_offset@1"
    ) is manager.get_term("joint_position")
