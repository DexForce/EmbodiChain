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

import numpy as np
import pytest
import torch

from examples.ods.simple_bc_demo.usage import (
    AnchorBudget,
    CompactFrameSet,
    UsageTracker,
)


def publication(episode: int, frames: list[int], generation: int = 1) -> dict:
    return {
        "kind": "published",
        "run_id": 3,
        "episode_id": episode,
        "generation": generation,
        "eligible_anchors": len(frames),
        "valid_frames": len(frames),
        "valid_frame_indices": frames,
        "eligible_anchor_indices": frames,
        "monotonic_ns": 20_000_000_000,
    }


def test_credit_earned_once_debited_only_on_completion_and_capped() -> None:
    budget = AnchorBudget(2, 4, 4)
    event = publication(1, [0, 1, 2, 3])
    budget.publish(event)
    assert budget.credits == 8
    budget.publish(event)
    assert budget.credits == 8
    budget.complete(4, 0.5)
    assert budget.credits == 4
    budget.publish(publication(2, [0, 1, 2, 3], 2))
    assert budget.credits == 8
    assert budget.discarded_credits == 4
    budget.complete(4, 0.5)
    budget.complete(4, 0.5)
    assert not budget.can_update
    with pytest.raises(ValueError, match="credits"):
        budget.complete(4, 0.5)


def test_first_publication_lead_time_and_bootstrap_prior() -> None:
    budget = AnchorBudget(32, 4, 100, prior_seconds=30)
    assert budget.lead_seconds == 37.5
    budget.request(2, 12)
    budget.request(2, 18)  # Coalescing preserves actual first request time.
    budget.publish(publication(1, [0], 2))
    assert budget.lead_seconds == 10
    budget.publish(publication(2, [0], 2))
    assert budget.summary()["first_publication_latencies_seconds"] == [8]
    budget.complete(4, 0.5)
    assert budget.should_refill()


def test_overlapping_chunks_masks_history_and_unused_eviction() -> None:
    tracker = UsageTracker()
    tracker.lifecycle([publication(1, [0, 1, 2, 3, 4]), publication(2, [0, 2])])
    ids = torch.tensor(
        [
            [[3, 1, 0], [3, 1, 1], [3, 1, 2], [3, 1, 3]],
            [[3, 1, 1], [3, 1, 2], [3, 1, 3], [-1, -1, -1]],
        ]
    )
    batch = {
        "frame_ids": ids,
        "valid": (ids[..., 0] >= 0),
        "loss_mask": torch.tensor(
            [[False, True, True, True], [False, True, True, False]]
        ),
    }
    tracker.sample(batch)
    assert tracker.summary()["completed_chunks"] == 0
    tracker.complete(batch)
    tracker.lifecycle([{**publication(2, [0, 2]), "kind": "evicted"}])
    summary = tracker.summary()
    assert summary["completed_chunks"] == 2
    assert summary["supervised_targets"] == 5
    first = summary["episodes"][0]
    assert first["completed_anchor_counts"] == [0, 1, 1, 0, 0]
    assert first["supervised_frame_counts"] == [0, 1, 2, 2, 0]
    assert first["history_frame_counts"] == [1, 2, 1, 0, 0]
    assert summary["evicted_unused_at_run_end_fraction"] == 1


def test_new_publication_never_reuses_retired_identity() -> None:
    tracker = UsageTracker()
    event = publication(1, [0])
    tracker.lifecycle([event, {**event, "kind": "evicted"}])
    with pytest.raises(ValueError, match="Repeated"):
        tracker.lifecycle([event])
    tracker.lifecycle([publication(2, [0], 2)])
    assert len(tracker.episodes) == 2


def test_compact_counts_match_reference_across_runs_sparse_frames_and_repeats() -> None:
    generator = torch.Generator().manual_seed(90)
    ids = torch.stack(
        [
            torch.randint(0, 3, (4096,), generator=generator),
            torch.randint(0, 8, (4096,), generator=generator),
            torch.randint(0, 600, (4096,), generator=generator),
        ],
        -1,
    )
    expected = set(map(tuple, ids.tolist()))
    counts = CompactFrameSet()
    counts.add_tensor(ids[::2])
    counts.add_tensor(ids[1::2])
    assert counts == expected
    assert len(counts) == len(expected)
    assert sum(int(values.sum()) for values in counts.counts.values()) == len(ids)
    counts.add_tensor(ids)
    assert len(counts) == len(expected)
    assert sum(int(values.sum()) for values in counts.counts.values()) == 2 * len(ids)


def test_invalid_masked_identity_rejected_without_padding_alias() -> None:
    counts = CompactFrameSet()
    with pytest.raises(ValueError, match="Padding"):
        counts.add_tensor(torch.tensor([[3, 1, -1]]))
    assert len(counts) == 0


@pytest.mark.parametrize("reuse", [0, -1, float("nan"), float("inf")])
def test_invalid_credit_configuration(reuse: float) -> None:
    with pytest.raises(ValueError):
        AnchorBudget(reuse, 4, 10)


def test_anchor_denominator_excludes_valid_but_ineligible_frames() -> None:
    tracker = UsageTracker()
    event = publication(1, [0, 2, 3])
    event["eligible_anchor_indices"] = [0, 3]
    event["eligible_anchors"] = 2
    tracker.lifecycle([event])
    batch = {"frame_ids": torch.tensor([[3, 1, 0], [3, 1, 3]])}
    tracker.sample(batch)
    tracker.complete(batch)
    row = tracker.summary()["episodes"][0]
    assert row["completed_anchor_coverage"] == 1
    assert row["completed_anchor_counts"] == [1, 1]
    assert row["supervised_frame_counts"] == [1, 0, 1]


def test_unpublished_source_cannot_silently_disappear_from_summary() -> None:
    tracker = UsageTracker()
    tracker.sample({"frame_ids": torch.tensor([[3, 9, 1]])})
    with pytest.raises(ValueError, match="lifecycle publication"):
        tracker.summary()


def test_sampling_rng_isolated_reproducible_and_restorable() -> None:
    from examples.ods.simple_bc_demo.usage import SamplingRNG

    torch.manual_seed(81)
    global_state = torch.get_rng_state().clone()
    stream = SamplingRNG(12)
    with stream.scope():
        first = torch.randint(0, 1000, (16,))
    assert torch.equal(torch.get_rng_state(), global_state)
    saved = stream.state.clone()
    torch.rand(100)  # Learner CPU noise cannot change the sampler sequence.
    global_state = torch.get_rng_state().clone()
    with stream.scope():
        expected = torch.randint(0, 1000, (16,))
    stream.state = saved
    with stream.scope():
        actual = torch.randint(0, 1000, (16,))
    assert torch.equal(actual, expected)
    assert not torch.equal(first, expected)
    assert torch.equal(torch.get_rng_state(), global_state)


def test_sampling_rng_restores_global_state_after_sampling_failure() -> None:
    from examples.ods.simple_bc_demo.usage import SamplingRNG

    state = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="deliberate"):
        with SamplingRNG(9).scope():
            torch.rand(8)
            raise RuntimeError("deliberate")
    assert torch.equal(torch.get_rng_state(), state)


def test_age_uses_publication_clock_and_counts_sampled_anchors(monkeypatch) -> None:
    tracker = UsageTracker()
    event = publication(1, [0, 1])
    tracker.lifecycle([event])
    monkeypatch.setattr(
        "examples.ods.simple_bc_demo.usage.time.monotonic_ns",
        lambda: 25_050_000_000,
    )
    tracker.sample({"frame_ids": torch.tensor([[3, 1, 0], [3, 1, 1]])})
    summary = tracker.summary()
    assert summary["first_use_latency_seconds_p50_p95"] == pytest.approx([5.05, 5.05])
    assert summary["sample_age_seconds_p50_p95_upper_bins"] == pytest.approx([5.1, 5.1])
    assert sum(tracker.age_histogram.values()) == 2


def test_age_handles_publication_arriving_between_drain_and_sample(monkeypatch) -> None:
    tracker = UsageTracker()
    monkeypatch.setattr(
        "examples.ods.simple_bc_demo.usage.time.monotonic_ns",
        lambda: 21_000_000_000,
    )
    tracker.sample({"frame_ids": torch.tensor([[3, 1, 0]])})
    tracker.lifecycle([publication(1, [0])])
    assert not tracker._pending_ages
    assert tracker.summary()["first_use_latency_seconds_p50_p95"] == [1, 1]


def test_sparse_episode_ids_use_bounded_fallback_and_view_does_not_double_count() -> (
    None
):
    counts = CompactFrameSet()
    ids = torch.tensor([[7, 1, 0], [7, 10**9, 599], [7, 10**9, 599]])
    counts.add_tensor(ids)
    assert counts == {(7, 1, 0), (7, 10**9, 599)}
    assert sum(len(values) for values in counts.counts.values()) <= 1200
    view = CompactFrameSet(counts.counts, accumulate=False)
    view.add_tensor(ids)
    assert len(view) == 2
    assert sum(int(values.sum()) for values in counts.counts.values()) == 3
    with pytest.raises(ValueError, match="externally"):
        CompactFrameSet(accumulate=False)


def test_completed_target_ids_are_returned_for_exact_tracker_reuse() -> None:
    tracker = UsageTracker()
    tracker.lifecycle([publication(1, [0, 1, 2])])
    ids = torch.tensor([[[3, 1, 0], [3, 1, 1], [3, 1, 2], [-1, -1, -1]]])
    batch = {
        "frame_ids": ids,
        "valid": ids[..., 0] >= 0,
        "loss_mask": torch.tensor([[False, True, True, False]]),
    }
    selected = tracker.complete(batch)
    assert torch.equal(selected, torch.tensor([[3, 1, 1], [3, 1, 2]]))
    view = CompactFrameSet(tracker.targets, accumulate=False)
    view.add_tensor(selected)
    assert len(view) == 2
    assert tracker.summary()["supervised_targets"] == 2


def test_deferred_panel_cap_keeps_last_three_generations_and_all_fixed_panels() -> None:
    from pathlib import Path
    from examples.ods.simple_bc_demo.usage import (
        retained_deferred_checkpoints,
    )

    generations = [Path(f"generation_{i}.pt") for i in range(1, 7)]
    initial, fixed, latest = Path("initial.pt"), Path("step_1000.pt"), Path("latest.pt")
    panels = [(initial, "initial"), (generations[0], "g1"), (fixed, "fixed")]
    panels += [(path, str(path)) for path in generations[1:]]
    panels.append((latest, "latest"))
    selected = retained_deferred_checkpoints(panels, generations)
    assert [path for path, _ in selected] == [initial, fixed, *generations[-3:], latest]
    assert len(panels) == 9  # Selection does not mutate the saved-file inventory.
