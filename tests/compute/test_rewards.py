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

"""Reward boundary and chunk partition contracts."""

from __future__ import annotations

import pytest
import torch

from embodichain.compute.rewards import aggregate_reward_sequence, potential_shaping


def test_shaping_telescopes_across_skill_switch_and_chunk_boundaries():
    gamma = 0.9
    phi = torch.tensor([0.1, 0.4, 0.6, 0.8, 0.9], dtype=torch.float64)
    rewards = potential_shaping(
        phi[:-1], phi[1:], torch.zeros(4, dtype=torch.bool), discount=gamma
    )
    flags = torch.zeros((1, 4), dtype=torch.bool)
    total, continuation, lengths = aggregate_reward_sequence(
        rewards[None], ~flags, flags, flags, discount=gamma
    )
    assert total.item() == pytest.approx(gamma**4 * 0.9 - 0.1)
    assert continuation.item() == pytest.approx(gamma**4)
    assert lengths.item() == 4
    parts = []
    for start in (0, 2):
        part_flags = flags[:, :2]
        parts.append(
            aggregate_reward_sequence(
                rewards[None, start : start + 2],
                ~part_flags,
                part_flags,
                part_flags,
                discount=gamma,
            )[0]
        )
    torch.testing.assert_close(total, parts[0] + gamma**2 * parts[1])


def test_termination_and_truncation_close_chunk_but_bootstrap_differently():
    rewards = torch.tensor([[1.0, 2.0, float("nan")], [1.0, 2.0, 100.0]])
    terminal = torch.tensor([[False, True, False], [False, False, False]])
    truncated = torch.tensor([[False, False, False], [False, True, False]])
    total, continuation, lengths = aggregate_reward_sequence(
        rewards, torch.ones_like(terminal), terminal, truncated, discount=0.5
    )
    torch.testing.assert_close(total, torch.tensor([2.0, 2.0]))
    torch.testing.assert_close(continuation, torch.tensor([0.0, 0.25]))
    assert lengths.tolist() == [2, 2]
    shaped = potential_shaping(
        torch.tensor([0.4, 0.4]),
        torch.tensor([0.8, 0.8]),
        torch.tensor([True, False]),
        discount=0.5,
    )
    torch.testing.assert_close(shaped, torch.tensor([-0.4, 0.0]))


def test_missing_slot_and_empty_chunk_never_consume_later_rewards():
    reward = torch.tensor([[1.0, float("nan"), 999.0]])
    valid = torch.tensor([[True, False, True]])
    flags = torch.zeros_like(valid)
    total, continuation, lengths = aggregate_reward_sequence(
        reward, valid, flags, flags, discount=0.5
    )
    assert (total.item(), continuation.item(), lengths.item()) == (1.0, 0.5, 1)
    total, continuation, lengths = aggregate_reward_sequence(
        reward[:, :0], valid[:, :0], flags[:, :0], flags[:, :0], discount=0.5
    )
    assert (total.item(), continuation.item(), lengths.item()) == (0.0, 1.0, 0)


@pytest.mark.parametrize("discount", [-1, 2, float("nan"), True])
def test_invalid_discount_rejected(discount):
    with pytest.raises(ValueError, match="discount"):
        potential_shaping(
            torch.zeros(1),
            torch.zeros(1),
            torch.zeros(1, dtype=torch.bool),
            discount=discount,
        )


def test_chunk_rejects_executed_prefix_crossing_episode_identity():
    reward = torch.ones(1, 3)
    flags = torch.zeros(1, 3, dtype=torch.bool)
    with pytest.raises(ValueError, match="cross episodes"):
        aggregate_reward_sequence(
            reward,
            ~flags,
            flags,
            flags,
            discount=0.9,
            episode_ids=torch.tensor([[0, 1, 1]]),
        )
    terminal = torch.tensor([[True, False, False]])
    total, _, length = aggregate_reward_sequence(
        reward,
        ~flags,
        terminal,
        flags,
        discount=0.9,
        episode_ids=torch.tensor([[0, 1, 1]]),
    )
    assert total.item() == 1 and length.item() == 1
