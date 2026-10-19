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

import copy
import os
import time
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict
from embodichain.data_pipeline.datasets import (
    TensorDataSource,
    SequenceSpecCfg,
)

from examples.data_pipeline._ods_pick_place_benchmark import (
    EfficiencyTracker,
    _content_manifest,
    _validate_tensors,
    resource_snapshot,
)

_SCHEMA = {
    "observation": {"names": ["joint", "height"]},
    "action": {"names": ["request"]},
}


def _corpus() -> TensorDataSource:
    valid = torch.tensor([[True, False, False], [True, True, True], [False] * 3])
    ids = torch.stack(
        (
            torch.full((3, 3), 11),
            torch.arange(3)[:, None].expand(3, 3),
            torch.arange(3)[None].expand(3, 3),
        ),
        -1,
    )
    tensors = {
        "observations": torch.arange(18).float().reshape(3, 3, 2),
        "actions": torch.arange(9).float().reshape(3, 3, 1),
        "valid": valid,
        "frame_ids": ids,
        "generations": torch.arange(1, 4)[:, None].expand(3, 3),
    }
    return TensorDataSource(
        TensorDict(
            {
                ("obs" if key == "observations" else key): value
                for key, value in tensors.items()
            },
            batch_size=tensors["valid"].shape,
        ),
        schema=_SCHEMA,
        source_id="test-corpus",
        metadata={"schema": _SCHEMA, **_content_manifest(tensors, _SCHEMA)},
    )


def test_context_preserves_source_boundaries_and_masks_padding() -> None:
    corpus = _corpus()
    batch = corpus.sample_context(
        256, SequenceSpecCfg(2, 5), generator=torch.Generator().manual_seed(18)
    )
    assert batch["obs"].shape == (256, 6, 2)
    assert batch["valid"][:, 1].all()
    assert not batch["loss_mask"][:, 0].any()
    assert (batch["frame_ids"][~batch["valid"]] == -1).all()
    assert (batch["obs"][~batch["valid"]] == 0).all()
    for ids, valid in zip(batch["frame_ids"], batch["valid"]):
        assert len(ids[valid, :2].unique(dim=0)) == 1
        assert (ids[valid, 2].diff() == 1).all()
    batch["obs"].zero_()
    assert corpus.data["obs"].count_nonzero() > 0


def test_context_generation_filter_never_crosses_a_validity_gap() -> None:
    corpus = _corpus()
    corpus.data["valid"][1] = torch.tensor([True, False, True])
    batch = corpus.sample_context(64, SequenceSpecCfg(2, 5, generations=(2,)))
    assert batch["valid"].sum().item() == 64
    assert (batch["generations"][batch["valid"]] == 2).all()
    with pytest.raises(RuntimeError, match="No valid"):
        corpus.sample_context(1, SequenceSpecCfg(2, 5, generations=(9,)))


def test_corpus_schema_validation_rejects_misaligned_or_nonfinite_supervision() -> None:
    corpus = _corpus()
    values = {
        key: corpus.data["obs" if key == "observations" else key].clone()
        for key in ("observations", "actions", "valid", "frame_ids", "generations")
    }
    _validate_tensors(values, _SCHEMA)
    values["actions"] = values["actions"].double()
    with pytest.raises(ValueError):
        _validate_tensors(values, _SCHEMA)
    values["actions"] = values["actions"].float()
    values["observations"][0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        _validate_tensors(values, _SCHEMA)


def test_content_identity_is_independent_of_engine_run_id() -> None:
    corpus = _corpus()
    tensors = {
        key: corpus.data["obs" if key == "observations" else key].clone()
        for key in ("observations", "actions", "valid", "frame_ids", "generations")
    }
    first = _content_manifest(tensors, _SCHEMA)
    tensors["frame_ids"][..., 0] += 80
    assert first["content_id"] == _content_manifest(tensors, _SCHEMA)["content_id"]


def test_t80_uses_second_consecutive_dev_result_arrival_and_checkpoint_counts() -> None:
    tracker = EfficiencyTracker(20, allocated_gpu_ids=["physical0", "physical0"])
    tracker.record_update(1, torch.tensor([[1, 1, 0], [1, 1, 1]]))
    tracker.record_checkpoint("first")
    tracker.record_evaluation(3, 0.8, checkpoint="first")
    tracker.record_update(4, torch.tensor([[1, 1, 0], [1, 1, 2]]))
    tracker.record_checkpoint("second")
    tracker.record_update(
        5, torch.tensor([[1, 1, 3]])
    )  # Work after the evaluated weights.
    tracker.record_evaluation(7, 0.9, checkpoint="second")
    metrics = tracker.finish(10, "updates")
    assert metrics["efficiency/reached_80"] is True
    assert metrics["efficiency/t80_dev_seconds"] == 7
    assert metrics["efficiency/t80_dev_checkpoint"] == "second"
    assert metrics["learning/targets_to_80"] == 4
    assert metrics["learning/unique_frames_to_80"] == 3
    assert metrics["cost/gpu_hours_to_80"] == pytest.approx(7 / 3600)


@pytest.mark.parametrize("middle", [0.79, None])
def test_failed_or_subthreshold_evaluation_breaks_consecutive_attainment(
    middle: float | None,
) -> None:
    tracker = EfficiencyTracker(10)
    tracker.record_evaluation(1, 0.9, checkpoint="a")
    tracker.record_evaluation(2, middle, checkpoint="b")
    tracker.record_evaluation(3, 0.9, checkpoint="c")
    metrics = tracker.finish(5, "failure")
    assert metrics["efficiency/reached_80"] is False
    assert metrics["efficiency/t80_dev_seconds"] is None
    assert metrics["efficiency/t80_dev_right_censored"] is True
    assert metrics["efficiency/censor_seconds"] == 5


def test_skipped_scheduled_checkpoint_does_not_confirm_t80() -> None:
    tracker = EfficiencyTracker(10)
    tracker.record_evaluation(1, 0.9, checkpoint="a", scheduled_index=0)
    tracker.record_evaluation(2, 0.9, checkpoint="c", scheduled_index=2)
    assert tracker.finish(3, "timeout")["efficiency/reached_80"] is False


def test_auc_uses_left_step_best_available_quality_without_lookahead() -> None:
    tracker = EfficiencyTracker(10)
    tracker.record_evaluation(2, 0.6, checkpoint="a")
    tracker.record_evaluation(4, 0.2, checkpoint="b")
    metrics = tracker.finish(5, "updates")
    # Zero for [0,2), then best=.6 through H=10, including after early completion.
    assert metrics["efficiency/auc_success_time"] == pytest.approx(0.6 * 8 / 10)


def test_development_results_after_horizon_cannot_attain_target() -> None:
    tracker = EfficiencyTracker(5)
    tracker.record_evaluation(4, 0.8, checkpoint="a")
    tracker.record_evaluation(6, 0.9, checkpoint="b")
    metrics = tracker.finish(7, "wall_time")
    assert metrics["efficiency/reached_80"] is False
    assert metrics["efficiency/censor_seconds"] == 5
    assert metrics["efficiency/auc_success_time"] == pytest.approx(0.8 / 5)


def test_valid_target_and_unique_frame_counts_exclude_masked_padding() -> None:
    tracker = EfficiencyTracker(10)
    ids = torch.tensor([[[1, 2, 0], [1, 2, 1], [-1, -1, -1]]])
    mask = torch.tensor([[True, True, False]])
    tracker.record_update(1, ids, mask)
    tracker.record_update(2, ids, mask)
    metrics = tracker.finish(3, "updates")
    assert metrics["learning/valid_targets"] == 4
    assert metrics["learning/unique_frames"] == 2
    assert metrics["data/targets_per_unique_frame"] == 2


def test_wait_intervals_are_unioned_and_separated_from_sampling_and_evaluation() -> (
    None
):
    tracker = EfficiencyTracker(10)
    tracker.record_interval("training_loop", 0, 8)
    tracker.record_interval("training_loop", 4, 10)
    tracker.record_interval("evaluation", 4, 6)
    tracker.record_interval("data_wait", 1, 3)
    tracker.record_interval("data_wait", 2, 5)
    tracker.record_interval("batch_service", 6, 6.2)
    metrics = tracker.finish(10, "updates")
    assert metrics["pipeline/training_loop_seconds"] == 8  # Union 10 less evaluation 2.
    assert metrics["pipeline/exposed_data_wait_seconds"] == 3  # [1,4), less evaluation.
    assert metrics["pipeline/data_wait_fraction"] == pytest.approx(3 / 8)
    assert metrics["pipeline/batch_latency_ms_p50_p95"] == pytest.approx([200, 200])


def test_failure_before_any_result_is_censored_and_unsupported_metrics_are_null() -> (
    None
):
    metrics = EfficiencyTracker(10).finish(2, "producer_failure")
    assert metrics["efficiency/t80_dev_seconds"] is None
    assert metrics["efficiency/censor_seconds"] == 2
    assert metrics["efficiency/auc_success_time"] == 0
    assert metrics["pipeline/data_wait_fraction"] is None
    assert metrics["cost/host_pss_sampled_peak_bytes"] is None
    assert metrics["cost/gpu_peak_bytes"] is None
    assert metrics["pipeline/producer_learner_overlap_fraction"] is None


def test_tracker_rejects_time_reversal_without_counting_the_update() -> None:
    tracker = EfficiencyTracker(10)
    ids = torch.tensor([[1, 2, 0]])
    tracker.record_update(2, ids)
    with pytest.raises(ValueError, match="monotonic"):
        tracker.record_update(1, ids)
    assert tracker.finish(3, "updates")["learning/valid_targets"] == 1


def test_resource_snapshot_deduplicates_pids_and_unknown_pss_remains_null() -> None:
    pid = os.getpid()
    snapshot = resource_snapshot([pid, pid])
    assert list(snapshot["processes"]) == [str(pid)]
    assert snapshot["processes"][str(pid)]["cpu_core_seconds"] >= 0
    # This PID exceeds Linux's PID range and cannot denote a live child.
    incomplete = resource_snapshot([pid, 2**31 - 1])
    assert incomplete["host_pss_bytes"] is None
    assert incomplete["missing_pids"] == [2**31 - 1]


def test_resource_counters_subtract_existing_cpu_baseline_and_keep_child_work() -> None:
    tracker = EfficiencyTracker(10)
    old_process = time.time() - 100  # Parent existed before the benchmark clock.
    new_process = time.time() + 1  # Explicitly belongs to this benchmark allocation.
    for parent_cpu, child_cpu, pss in ((100, 2, 500), (105, 4, 450)):
        tracker.record_resources(
            {
                "host_pss_bytes": pss,
                "processes": {
                    "1": {"create_time": old_process, "cpu_core_seconds": parent_cpu},
                    "2": {"create_time": new_process, "cpu_core_seconds": child_cpu},
                },
            }
        )
    metrics = tracker.finish(5, "updates")
    assert (
        metrics["cost/cpu_core_seconds_observed"] == 9
    )  # Parent delta 5 plus child total 4.
    assert metrics["cost/host_pss_sampled_peak_bytes"] == 500


@pytest.mark.parametrize(
    "left,right,expected",
    [
        ([], [(0, 1)], []),
        ([(0, 0), (2, 2)], [(0, 3)], []),
        ([(0, 10), (2, 3)], [(4, 5), (5, 7)], [(4, 7)]),
        ([(6, 8), (0, 2)], [(1, 7)], [(1, 2), (6, 7)]),
        ([(0, 1)], [(1, 2)], []),
        ([(0.25, 1.5), (3.0, 4.0)], [(1.0, 3.5)], [(1.0, 1.5), (3.0, 3.5)]),
    ],
)
def test_interval_intersection_handles_single_pass_iterables(left, right, expected):
    from examples.data_pipeline._ods_pick_place_benchmark import _intersection

    assert _intersection(iter(left), iter(right)) == expected


def test_interval_intersection_matches_discrete_coverage_oracle():
    import random
    from examples.data_pipeline._ods_pick_place_benchmark import _intersection

    rng = random.Random(27)
    for _ in range(100):
        left = [tuple(sorted(rng.choices(range(30), k=2))) for _ in range(12)]
        right = [tuple(sorted(rng.choices(range(30), k=2))) for _ in range(8)]
        covered_left = {i for start, end in left for i in range(start, end)}
        covered_right = {i for start, end in right for i in range(start, end)}
        actual = _intersection(left, right)
        assert {
            i for start, end in actual for i in range(start, end)
        } == covered_left & covered_right
        assert all(start < end for start, end in actual)
        assert all(first[1] <= second[0] for first, second in zip(actual, actual[1:]))


def test_checkpoint_save_events_identify_weights_and_clock():
    tracker = EfficiencyTracker(100)
    ids = torch.tensor([[1, 2, 3]], dtype=torch.int64)
    tracker.record_update(2.0, ids)
    tracker.record_checkpoint("frozen.pt", policy_step=1, elapsed=2.5)
    event = tracker.timeline[-1]
    assert event == {
        "kind": "checkpoint_saved",
        "seconds": 2.5,
        "checkpoint": "frozen.pt",
        "policy_step": 1,
    }
    assert tracker._checkpoints["frozen.pt"] == (1, 1)
    with pytest.raises(ValueError, match="nondecreasing"):
        tracker.record_checkpoint("old-clock.pt", policy_step=1, elapsed=1.0)
    assert "old-clock.pt" not in tracker._checkpoints
    with pytest.raises(ValueError, match="requires"):
        tracker.record_checkpoint("unknown-step.pt", elapsed=3.0)
