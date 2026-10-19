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
import json
import os
import time
from pathlib import Path

import pytest
import torch

from examples.ods.simple_bc_demo.data import (
    EfficiencyTracker,
    export_corpus,
    load_corpus,
    resource_snapshot,
)

_SCHEMA = {
    "observation": {"names": ["measured_joint", "cube_height"]},
    "action": {"names": ["arm_request"]},
    "version": 1,
}


def _export(path: Path, *, run_id: int = 11) -> dict:
    # Two unequal-length episodes plus one empty padded row.
    valid = torch.tensor([[True, False, False], [True, True, True], [False] * 3])
    observations = torch.arange(18, dtype=torch.float32).reshape(3, 3, 2)
    actions = torch.arange(9, dtype=torch.float32).reshape(3, 3, 1)
    ids = torch.zeros(3, 3, 3, dtype=torch.int64)
    for row in range(3):
        ids[row, :, 0] = run_id
        ids[row, :, 1] = row
        ids[row, :, 2] = torch.arange(3)
    generations = torch.tensor([[1] * 3, [2] * 3, [3] * 3])
    return export_corpus(
        path,
        observations,
        actions,
        valid,
        ids,
        schema=_SCHEMA,
        generations=generations,
        provenance={"teacher": "qualified-v1"},
    )


def test_cpu_mmap_corpus_preserves_schema_masks_and_frame_identity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "corpus.pt"
    manifest = _export(path)
    corpus = load_corpus(path, expected_schema=_SCHEMA)
    assert corpus.observations.device.type == "cpu"
    assert corpus.manifest["schema"] == _SCHEMA
    assert corpus.valid.sum() == 4  # One frame in episode zero, three in episode one.
    assert corpus.frame_ids[1, 2].tolist() == [11, 1, 2]
    assert corpus.observations[~corpus.valid].count_nonzero() == 0
    assert manifest["tensor_file_bytes"] == path.stat().st_size
    with pytest.raises(FileExistsError):
        _export(path)


def test_offline_context_samples_short_episodes_and_excludes_padded_targets(
    tmp_path: Path,
) -> None:
    path = tmp_path / "corpus.pt"
    _export(path)
    corpus = load_corpus(path, expected_schema=_SCHEMA)
    batch = corpus.sample_context(256, 2, 5)
    assert batch["observations"].shape == (256, 6, 2)
    assert batch["valid"][:, 1].all()
    assert not batch["loss_mask"][:, 0].any()
    assert (batch["frame_ids"][~batch["valid"]] == -1).all()
    assert (batch["observations"][~batch["valid"]] == 0).all()
    short = batch["frame_ids"][:, 1, 1] == 0
    assert short.any()
    assert (batch["loss_mask"][short].sum(1) == 1).all()
    assert not batch["valid"][short, 0].any()


def test_corpus_content_identity_ignores_engine_run_id(tmp_path: Path) -> None:
    first = _export(tmp_path / "first.pt", run_id=11)
    second = _export(tmp_path / "second.pt", run_id=99)
    assert first["content_id"] == second["content_id"]
    assert (
        first["tensors"]["frame_ids"]["sha256"]
        != second["tensors"]["frame_ids"]["sha256"]
    )


def test_corpus_rejects_changed_observation_action_schema(tmp_path: Path) -> None:
    path = tmp_path / "corpus.pt"
    _export(path)
    incompatible = copy.deepcopy(_SCHEMA)
    incompatible["action"]["names"] = ["different_control_contract"]
    with pytest.raises(ValueError, match="schema mismatch"):
        load_corpus(path, expected_schema=incompatible)


def test_corpus_rejects_corrupt_tensor_content(tmp_path: Path) -> None:
    path = tmp_path / "corpus.pt"
    _export(path)
    tensors = torch.load(path, weights_only=True)
    tensors["actions"][1, 1] += 1
    torch.save(tensors, path)
    with pytest.raises(ValueError, match="hash mismatch"):
        load_corpus(path, expected_schema=_SCHEMA)


def test_corpus_rejects_corrupt_manifest_hash(tmp_path: Path) -> None:
    path = tmp_path / "corpus.pt"
    _export(path)
    manifest_path = Path(str(path) + ".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_sha256"] = "invalid"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_corpus(path, expected_schema=_SCHEMA)


def test_corpus_rejects_incorrect_within_episode_frame_ids(tmp_path: Path) -> None:
    path = tmp_path / "corpus.pt"
    _export(path)
    tensors = torch.load(path, weights_only=True)
    tensors["frame_ids"][1, 2, 2] = 0
    with pytest.raises(ValueError, match="within-episode steps"):
        export_corpus(
            tmp_path / "bad.pt",
            tensors["observations"],
            tensors["actions"],
            tensors["valid"],
            tensors["frame_ids"],
            schema=_SCHEMA,
        )
    assert not (tmp_path / "bad.pt").exists()


def test_frame_sampler_is_episode_uniform_and_never_selects_padding(
    tmp_path: Path,
) -> None:
    path = tmp_path / "corpus.pt"
    _export(path)
    corpus = load_corpus(path, expected_schema=_SCHEMA)
    sample = corpus.sample_frames(4000, generator=torch.Generator().manual_seed(5))
    assert sample["valid"].all()
    assert (sample["frame_ids"][:, 0] == 11).all()
    # Equal episode probability despite the threefold difference in length.
    assert (sample["frame_ids"][:, 1] == 0).float().mean().item() == pytest.approx(
        0.5, abs=0.03
    )
    assert (sample["frame_ids"][sample["frame_ids"][:, 1] == 0, 2] == 0).all()
    sample["observations"].fill_(-100)
    assert corpus.observations[corpus.valid].min() >= 0


def test_sampler_filters_generations_and_explicit_episode_subsets(
    tmp_path: Path,
) -> None:
    path = tmp_path / "corpus.pt"
    _export(path)
    corpus = load_corpus(path, expected_schema=_SCHEMA)
    generated = corpus.sample_frames(20, generations=[2])
    assert (generated["frame_ids"][:, 1] == 1).all()
    assert (generated["generations"] == 2).all()
    selected = corpus.sample_frames(20, episode_subset=[0])
    assert (selected["frame_ids"][:, 1] == 0).all()
    with pytest.raises(ValueError, match="No valid frames"):
        corpus.sample_frames(1, episode_subset=[2])


def test_explicit_frame_manifest_replays_order_and_repetitions(tmp_path: Path) -> None:
    path = tmp_path / "corpus.pt"
    _export(path)
    corpus = load_corpus(path, expected_schema=_SCHEMA)
    identities = torch.tensor([[11, 1, 2], [11, 0, 0], [11, 1, 2]])
    sample = corpus.sample_frame_ids(identities)
    torch.testing.assert_close(sample["frame_ids"], identities)
    torch.testing.assert_close(sample["actions"][:, 0], torch.tensor([5.0, 0.0, 5.0]))
    with pytest.raises(ValueError, match="Unknown or invalid"):
        corpus.sample_frame_ids(torch.tensor([[11, 0, 1]]))


def test_history_sampler_is_episode_uniform_with_exact_target_counts(
    tmp_path: Path,
) -> None:
    path = tmp_path / "corpus.pt"
    _export(path)
    corpus = load_corpus(path, expected_schema=_SCHEMA)
    sample = corpus.sample_histories(
        4000,
        supervised_steps=1,
        start_probability=0,
        generator=torch.Generator().manual_seed(5),
    )
    episode_ids = sample["frame_ids"][:, 0, 1]
    assert (episode_ids == 0).float().mean().item() == pytest.approx(0.5, abs=0.03)
    assert (sample["loss_mask"].sum(-1) == 1).all()
    assert not (sample["loss_mask"] & ~sample["valid"]).any()
    # A full prefix begins at episode step zero and never crosses another row.
    for row in range(len(episode_ids)):
        identities = sample["frame_ids"][row, sample["valid"][row]]
        assert (identities[:, 1] == episode_ids[row]).all()
        torch.testing.assert_close(identities[:, 2], torch.arange(len(identities)))


def test_history_sampler_marks_padding_and_preserves_independent_copies(
    tmp_path: Path,
) -> None:
    path = tmp_path / "corpus.pt"
    _export(path)
    corpus = load_corpus(path, expected_schema=_SCHEMA)
    sample = corpus.sample_histories(
        32,
        supervised_steps=1,
        start_probability=0,
        generator=torch.Generator().manual_seed(7),
    )
    padding = ~sample["valid"]
    assert padding.any()
    assert sample["observations"][padding].count_nonzero() == 0
    assert sample["actions"][padding].count_nonzero() == 0
    assert (sample["frame_ids"][padding] == -1).all()
    assert (sample["generations"][padding] == -1).all()
    assert not sample["loss_mask"][padding].any()
    assert set(sample) == {
        "observations",
        "actions",
        "valid",
        "frame_ids",
        "generations",
        "loss_mask",
    }
    sample["observations"].fill_(-100)
    assert corpus.observations[corpus.valid].min() >= 0


def test_history_filters_keep_all_prefixes_and_targets_in_selected_generations(
    tmp_path: Path,
) -> None:
    path = tmp_path / "four-episodes.pt"
    # Four accepted episodes split across G1/G2, with five frames per episode.
    valid = torch.ones(4, 5, dtype=torch.bool)
    ids = torch.zeros(4, 5, 3, dtype=torch.int64)
    ids[..., 0] = 11
    ids[..., 1] = torch.arange(4)[:, None]
    ids[..., 2] = torch.arange(5)[None]
    generations = torch.tensor([1, 1, 2, 2])[:, None].expand(4, 5)
    export_corpus(
        path,
        torch.zeros(4, 5, 2),
        torch.zeros(4, 5, 1),
        valid,
        ids,
        schema=_SCHEMA,
        generations=generations,
    )
    corpus = load_corpus(path, expected_schema=_SCHEMA)
    sample = corpus.sample_histories(
        400,
        supervised_steps=2,
        generations=[1],
        generator=torch.Generator().manual_seed(5),
    )
    assert sample["loss_mask"].sum() == 800  # 400 sampled rows, two targets each.
    assert (sample["loss_mask"].sum(-1) == 2).all()
    assert (sample["generations"][sample["valid"]] == 1).all()
    assert (sample["frame_ids"][sample["valid"]][:, 1] < 2).all()
    assert (sample["frame_ids"][:, 0, 1] == 0).float().mean().item() == pytest.approx(
        0.5, abs=0.06
    )
    selected = corpus.sample_histories(
        32, supervised_steps=2, episode_subset=[1, 2], generations=[1]
    )
    assert (selected["frame_ids"][selected["valid"]][:, 1] == 1).all()
    with pytest.raises(RuntimeError, match="No valid offline history window"):
        corpus.sample_histories(
            1, supervised_steps=2, episode_subset=[0, 1], generations=[2]
        )


def test_history_sampler_can_supervise_first_actions_without_a_prefix(
    tmp_path: Path,
) -> None:
    path = tmp_path / "corpus.pt"
    _export(path)
    corpus = load_corpus(path, expected_schema=_SCHEMA)
    sample = corpus.sample_histories(
        32,
        supervised_steps=2,
        start_probability=1,
        generator=torch.Generator().manual_seed(2),
    )
    assert sample["valid"].shape == (32, 2)
    assert sample["valid"].all()
    assert sample["loss_mask"].all()
    assert (
        sample["frame_ids"][..., 1] == 1
    ).all()  # Only the length-three episode qualifies.
    assert (sample["frame_ids"][:, 0, 2] == 0).all()


def test_history_sampler_stops_at_validity_holes_and_uses_only_intact_windows(
    tmp_path: Path,
) -> None:
    path = tmp_path / "holes.pt"
    # Two isolated valid frames precede a hole; only the final two-frame run qualifies.
    valid = torch.tensor([[True, False, True, False, True, True]])
    ids = torch.stack(
        (torch.full((6,), 11), torch.zeros(6, dtype=torch.int64), torch.arange(6)), -1
    )[None]
    observations = torch.arange(12, dtype=torch.float32).reshape(1, 6, 2)
    actions = torch.arange(6, dtype=torch.float32).reshape(1, 6, 1)
    export_corpus(path, observations, actions, valid, ids, schema=_SCHEMA)
    sample = load_corpus(path, expected_schema=_SCHEMA).sample_histories(
        8, supervised_steps=2, generator=torch.Generator().manual_seed(2)
    )
    assert sample["valid"].shape == (8, 2)
    assert sample["loss_mask"].all()
    assert (sample["frame_ids"][..., 2] == torch.tensor([4, 5])).all()
    assert "generations" not in sample
    histories = load_corpus(path, expected_schema=_SCHEMA).sample_histories(
        64,
        supervised_steps=1,
        start_probability=0,
        generator=torch.Generator().manual_seed(3),
    )
    # When step five is supervised, its complete uninterrupted prefix is step four.
    ends_at_five = (
        histories["frame_ids"][..., 2].masked_fill(~histories["loss_mask"], -1).amax(-1)
        == 5
    )
    assert ends_at_five.any()
    assert (histories["frame_ids"][ends_at_five, :2, 2] == torch.tensor([4, 5])).all()


def test_history_sampler_rejects_corpora_without_an_intact_window(
    tmp_path: Path,
) -> None:
    path = tmp_path / "corpus.pt"
    _export(path)
    corpus = load_corpus(path, expected_schema=_SCHEMA)
    corpus.valid[:, 1] = False  # Each remaining contiguous run contains one frame.
    with pytest.raises(RuntimeError, match="No valid offline history window"):
        corpus.sample_histories(1, supervised_steps=2)


def test_corpus_does_not_silently_accept_unrepresented_continuity_fields(
    tmp_path: Path,
) -> None:
    path = tmp_path / "corpus.pt"
    _export(path)
    tensors = torch.load(path, weights_only=True)
    tensors["continuity_id"] = torch.zeros_like(tensors["valid"], dtype=torch.int64)
    torch.save(tensors, path)
    with pytest.raises(ValueError, match="four numeric tensor fields"):
        load_corpus(path, expected_schema=_SCHEMA)


@pytest.mark.parametrize(
    "batch_size,steps,probability",
    [(0, 1, 0), (1, 0, 0), (1, 4, 0), (1, 1, -0.1), (1, 1, 1.1)],
)
def test_history_sampler_rejects_invalid_counts_and_probabilities(
    tmp_path: Path, batch_size: int, steps: int, probability: float
) -> None:
    path = tmp_path / "corpus.pt"
    _export(path)
    corpus = load_corpus(path, expected_schema=_SCHEMA)
    with pytest.raises(ValueError, match="Invalid history"):
        corpus.sample_histories(batch_size, steps, probability)


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
    from examples.ods.simple_bc_demo.data import _intersection

    assert _intersection(iter(left), iter(right)) == expected


def test_interval_intersection_matches_discrete_coverage_oracle():
    import random
    from examples.ods.simple_bc_demo.data import _intersection

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
