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

"""Owned ring storage, episode boundaries and exact sampling recovery."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from embodichain.data_pipeline.datasets.transition_replay import TransitionReplay


def _append(replay: TransitionReplay, step: int) -> None:
    replay.append(
        torch.full((2, 3), float(step)),
        torch.full((2, 1), float(step)),
        torch.full((2, 3), float(step + 1)),
        terminated=torch.tensor([step == 3, False]),
        truncated=torch.tensor([False, step == 4]),
    )


def test_wraparound_windows_never_expose_following_episodes():
    replay = TransitionReplay(4, 2, 3, 1, seed=7)
    for step in range(6):
        _append(replay, step)
    windows = replay.sample_windows(128, 3)
    valid = windows["valid"]
    assert windows["state"].min() >= 2 and windows["state"].max() <= 5
    assert valid[:, 0].all()
    assert (~valid).any()
    assert torch.equal(
        windows["episode_id"], windows["episode_id"][:, :1].expand_as(valid)
    )
    for batch in range(len(valid)):
        count = int(valid[batch].sum())
        assert valid[batch, :count].all() and not valid[batch, count:].any()
        torch.testing.assert_close(
            windows["state"][batch, count:],
            windows["state"][batch, count - 1].expand(3 - count, -1),
        )
        torch.testing.assert_close(
            windows["next_state"][batch, :count], windows["state"][batch, :count] + 1
        )


def test_append_and_samples_own_storage_without_graphs():
    replay = TransitionReplay(2, 1, 3, 1)
    state = torch.ones(1, 3, requires_grad=True)
    replay.append(state, torch.ones(1, 1), state * 2)
    sampled = replay.sample_windows(1, 1)
    assert (
        not sampled["state"].requires_grad and not sampled["next_state"].requires_grad
    )
    with torch.no_grad():
        state.fill_(50)
    replay.append(state, torch.zeros(1, 1), state)
    replay.append(state, torch.zeros(1, 1), state)
    torch.testing.assert_close(sampled["state"], torch.ones(1, 1, 3))


def test_history_padding_respects_episode_and_oldest_retained_boundary():
    replay = TransitionReplay(4, 2, 3, 1, seed=9)
    for step in range(6):
        _append(replay, step)
    starts = replay.sample_starts(128, 5)
    valid = starts["history_valid"]
    assert valid[:, -1].all()
    assert not (valid[:, :-1] & ~valid[:, 1:]).any()
    assert torch.equal(
        starts["history"]["episode_id"],
        starts["start"]["episode_id"][:, None].expand_as(valid),
    )
    assert starts["history"]["state"].min() >= 2
    torch.testing.assert_close(
        starts["history"]["state"][:, -1], starts["start"]["state"]
    )


def test_snapshot_restores_content_boundaries_and_rng_before_mutation():
    replay = TransitionReplay(4, 2, 3, 1, seed=10)
    for step in range(6):
        _append(replay, step)
    state = replay.state_dict()
    expected = replay.sample_windows(32, 3)
    restored = TransitionReplay(4, 2, 3, 1, seed=99)
    restored.load_state_dict(state)
    actual = restored.sample_windows(32, 3)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
    invalid = deepcopy(state)
    invalid["data"]["action"] = torch.zeros(4, 2, 2)
    before = restored.state_dict()
    with pytest.raises(ValueError, match="layout"):
        restored.load_state_dict(invalid)
    after = restored.state_dict()
    for key in before["data"]:
        torch.testing.assert_close(
            before["data"][key], after["data"][key], atol=0, rtol=0
        )
    assert before["cursor"] == after["cursor"]


def test_memory_budget_and_invalid_sampling_fail_without_waits():
    replay = TransitionReplay(4, 2, 3, 1)
    with pytest.raises(ValueError, match="max_bytes"):
        TransitionReplay(4, 2, 3, 1, max_bytes=replay.allocated_bytes - 1)
    with pytest.raises(ValueError, match="retained"):
        replay.sample_windows(1, 1)
    with pytest.raises(ValueError, match="nonempty"):
        replay.sample_starts(1, 2)
