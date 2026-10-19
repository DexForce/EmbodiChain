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

from collections import Counter

import numpy as np
import pytest
import torch

from examples.ods.simple_bc_demo.usage import CompactFrameSet, UsageTracker


def _event(run: int, episode: int, *, timestamp: int = 1_000_000_000) -> dict:
    return {
        "kind": "published",
        "run_id": run,
        "episode_id": episode,
        "generation": 1,
        "monotonic_ns": timestamp,
        "valid_frame_indices": list(range(700)),
        "eligible_anchor_indices": list(range(700)),
        "valid_frames": 700,
        "eligible_anchors": 700,
    }


def _assert_counts(storage: dict, expected: Counter) -> None:
    actual = {
        (*key, index): int(value)
        for key, values in storage.items()
        for index, value in enumerate(values)
        if value
    }
    assert actual == dict(expected)
    assert all(values.dtype == np.int64 for values in storage.values())


@pytest.mark.parametrize("seed", range(6))
def test_all_four_counters_match_explicit_frame_oracle_with_sparse_multirun_ids(
    seed, monkeypatch
):
    tracker = UsageTracker(episode_capacity_hint=256)
    rng = np.random.default_rng(seed)
    sources = [
        (run, episode)
        for run in [3, 10**12]
        for episode in [0, 1, 255, 256, 513, 10**9]
    ]
    tracker.lifecycle([_event(*key) for key in sources])
    monkeypatch.setattr(
        "examples.ods.simple_bc_demo.usage.time.monotonic_ns",
        lambda: 2_000_000_000,
    )
    expected = {
        name: Counter() for name in ("sampled", "anchors", "targets", "history")
    }
    for iteration in range(16):
        ids = torch.full((37, 9, 3), -1, dtype=torch.int64)
        for row in range(len(ids)):
            run, episode = sources[int(rng.integers(len(sources)))]
            start = int(rng.integers(650))
            length = int(rng.integers(2, 10))
            ids[row, :length, 0] = run
            ids[row, :length, 1] = episode
            ids[row, :length, 2] = torch.arange(start, start + length)
        valid = ids[..., 0] >= 0
        loss = valid.clone()
        loss[:, 0] = False
        loss &= torch.from_numpy(rng.random(loss.shape) > 0.2)
        batch = {"frame_ids": ids, "valid": valid, "loss_mask": loss}
        tracker.sample(batch)
        expected["sampled"].update(map(tuple, ids[:, 1].tolist()))
        if iteration % 3:  # Sampling alone must not count as optimizer completion.
            selected = tracker.complete(batch)
            assert torch.equal(selected, ids[loss])
            expected["anchors"].update(map(tuple, ids[:, 1].tolist()))
            expected["targets"].update(map(tuple, ids[loss].tolist()))
            expected["history"].update(map(tuple, ids[:, :2][valid[:, :2]].tolist()))
    tracker.lifecycle(
        [{**_event(*sources[0]), "kind": "evicted", "monotonic_ns": 3_000_000_000}]
    )
    for name in expected:
        _assert_counts(getattr(tracker, name), expected[name])
    summary = tracker.summary()
    assert summary["completed_chunks"] == sum(expected["anchors"].values())
    assert summary["supervised_targets"] == sum(expected["targets"].values())
    assert sum(tracker.age_histogram.values()) == sum(expected["sampled"].values())
    assert len(CompactFrameSet(tracker.targets, accumulate=False)) == len(
        expected["targets"]
    )
    assert summary["first_use_latency_seconds_p50_p95"] == [1, 1]


def test_counter_growth_preserves_preexisting_views_and_new_slots():
    tracker = UsageTracker(episode_capacity_hint=256)
    expected = Counter()
    for episode, frame in [
        (1, 0),
        (2, 2),
        (3, 63),
        (4, 64),
        (5, 300),
        (1, 599),
        (255, 699),
        (256, 0),
    ]:
        key = (4, episode, frame)
        batch = {"frame_ids": torch.tensor([key, key, (4, 1, 0)])}
        tracker.complete(batch)
        expected.update(map(tuple, batch["frame_ids"].tolist()))
        _assert_counts(tracker.targets, expected)
        _assert_counts(tracker.anchors, expected)
    shared = CompactFrameSet(tracker.targets, accumulate=False)
    tracker.complete({"frame_ids": torch.tensor([[4, 1, 698]])})
    expected[(4, 1, 698)] += 1
    assert len(shared) == len(expected)


def test_sparse_run_and_episode_ids_allocate_only_observed_rows():
    tracker = UsageTracker(episode_capacity_hint=256)
    ids = torch.tensor([[2**62, 1, 0], [2**62, 10**12, 699], [2**62, 2**63 - 1, 5]])
    tracker.complete({"frame_ids": ids})
    _assert_counts(tracker.targets, Counter(map(tuple, ids.tolist())))
    # Each sparse source receives one compact row, not 256 rows or max-ID rows.
    counters = tracker.targets._blocks
    assert sum(block.values.shape[0] for block in counters.values()) == 3
    assert (
        sum(block.values.nbytes + block.slots.nbytes for block in counters.values())
        < 20_000
    )


def test_noncontiguous_ids_and_empty_targets_preserve_exact_accounting():
    tracker = UsageTracker(episode_capacity_hint=256)
    ids = torch.tensor([[[4, 1, 0], [4, 1, 1]], [[4, 2, 0], [4, 2, 1]]])
    ids = ids.transpose(0, 1)
    batch = {
        "frame_ids": ids,
        "valid": torch.ones(2, 2, dtype=torch.bool),
        "loss_mask": torch.zeros(2, 2, dtype=torch.bool),
    }
    tracker.complete(batch)
    assert tracker.targets == {}
    _assert_counts(tracker.anchors, Counter(map(tuple, ids[:, 1].tolist())))
    _assert_counts(tracker.history, Counter(map(tuple, ids.reshape(-1, 3).tolist())))


def test_padding_rejection_does_not_mutate_counter_blocks():
    tracker = UsageTracker(episode_capacity_hint=256)
    with pytest.raises(ValueError, match="Padding"):
        tracker.sample({"frame_ids": torch.tensor([[4, 1, 0], [-1, -1, -1]])})
    assert tracker.sampled == {}
    assert tracker.sampled._blocks == {}


@pytest.mark.parametrize(
    "capacity, expected",
    [
        (None, "dict"),
        (16, "dict"),
        (32, "dict"),
        (63, "dict"),
        (64, "blocked"),
        (128, "blocked"),
        (256, "blocked"),
    ],
)
def test_capacity_hint_selects_one_fixed_backend(capacity, expected):
    tracker = UsageTracker(episode_capacity_hint=capacity)
    assert tracker.counter_backend == expected
    tracker.lifecycle([_event(1, episode) for episode in range(100)])
    tracker.complete({"frame_ids": torch.tensor([[1, 99, 7]])})
    assert tracker.counter_backend == expected
    if expected == "dict":
        assert type(tracker.targets) is dict
        assert type(tracker.sampled) is dict
    assert tracker.summary()["supervised_targets"] == 1


@pytest.mark.parametrize("capacity", [0, -1, True, 32.0, "32"])
def test_invalid_capacity_hint_rejected(capacity):
    with pytest.raises(ValueError, match="episode_capacity_hint"):
        UsageTracker(episode_capacity_hint=capacity)


def test_backend_choice_preserves_all_summary_fields(monkeypatch):
    small = UsageTracker(episode_capacity_hint=32)
    large = UsageTracker(episode_capacity_hint=256)
    events = [_event(1, 0), _event(1, 255), _event(1, 10**12)]
    batch = {
        "frame_ids": torch.tensor([[1, 0, 1], [1, 255, 2], [1, 10**12, 3], [1, 255, 2]])
    }
    monkeypatch.setattr(
        "examples.ods.simple_bc_demo.usage.time.monotonic_ns",
        lambda: 2_000_000_000,
    )
    for tracker in (small, large):
        tracker.lifecycle(events)
        tracker.sample(batch)
        tracker.complete(batch)
    assert small.summary() == large.summary()
