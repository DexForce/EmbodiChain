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

"""Receipt and frozen-domain semantics for cross-task scene acceptance reports."""

from __future__ import annotations
from types import SimpleNamespace
import numpy as np
import torch
from scripts.benchmark.expansion.scene_first import (
    STRATEGIES,
    _aggregate,
    _confirmed_episodes,
    _strategy_order,
    _task_pose_cells,
    _xy_cell,
)


def test_each_strategy_runs_once_at_each_position_and_after_each_peer():
    strategies = list(STRATEGIES)
    orders = [_strategy_order(strategies, i) for i in range(4)]
    assert all(set(column) == set(strategies) for column in zip(*orders))
    assert len({pair for order in orders for pair in zip(order, order[1:])}) == 12


def test_cross_seed_coverage_counts_each_measured_cell_once():
    run = dict(
        case="stack",
        strategy="bidirectional",
        executed=2,
        physically_accepted=2,
        committed=2,
        preparation_rejected=1,
        budget={"candidates": 3},
        wall_s=30.0,
        peak_rss_mib=2000,
        scene_joint_declared_cells=256,
        scene_marginal_declared_cells=32,
        error=None,
    )
    shared = ["a", 0, 0]
    rows = [
        dict(
            run,
            seed=seed,
            coverage_cells={
                "scene_joint": [[shared, ["b", seed, 0]]],
                "scene_marginal": [shared, ["b", seed, 0]],
                "task": [["a", "initial", 0, 0], ["a", "placed", seed, 0]],
            },
        )
        for seed in (1, 2)
    ]
    (result,) = _aggregate(rows)
    assert result["coverage"] == {"scene_joint": 2, "scene_marginal": 3, "task": 3}
    assert (result["committed"], result["raw_proposals"]) == (4, 6)


def test_staged_numerical_file_without_receipt_has_no_coverage_credit(tmp_path):
    folder = tmp_path / "episodes" / "staged"
    folder.mkdir(parents=True)
    (folder / "episode.npz").write_bytes(b"not committed")
    assert _confirmed_episodes(tmp_path) == ()


def test_confirmed_readback_uses_verified_receipt_storage(monkeypatch, tmp_path):
    from unittest.mock import Mock
    import embodichain.lab.expansion.persistence as persistence

    episode = object()
    load = Mock(return_value=episode)
    monkeypatch.setattr(persistence, "load_expansion_episode", load)
    monkeypatch.setattr(
        persistence.ArtifactEpisodeSink,
        "inspect",
        lambda self: (
            SimpleNamespace(confirmed=True, storage_id="confirmed.npz"),
            SimpleNamespace(confirmed=False, storage_id="pending.npz"),
        ),
    )
    assert _confirmed_episodes(tmp_path) == (episode,)
    load.assert_called_once_with("confirmed.npz")


def test_xy_bins_are_fixed_and_reject_outside_domain():
    bounds = [[-0.1, 0.1], [-0.1, 0.1], [0, 0]]
    assert _xy_cell([-0.1, -0.1, 0], [0, 0, 0], bounds) == (0, 0)
    assert _xy_cell([0.1, 0.1, 0], [0, 0, 0], bounds) == (3, 3)
    assert _xy_cell([0.11, 0, 0], [0, 0, 0], bounds) is None
    assert _xy_cell([0, 0, 0], [0, 0, 0], [[0, 0], [0, 0], [0, 0]]) == (0, 0)


def test_both_arms_count_but_fixed_goal_tracking_noise_adds_no_diversity():
    nominal = {
        "block_1": [0.625, -0.15, 0.86],
        "block_2": [0.675, 0, 0.86],
        "block_3": [0.725, 0.15, 0.86],
    }
    cfg = {
        "scene": {
            uid: {"translation_m": [[-0.025, 0.025], [-0.03, 0.03], [0, 0]]}
            for uid in ("block_1", "block_3")
        }
    }
    episode = SimpleNamespace(
        observations={
            "rigid/block_1/state": torch.tensor(
                [[0.62, -0.155, 0.86], [0.59, -0.005, 0.86]]
            ),
            "rigid/block_3/state": torch.tensor(
                [[0.72, 0.145, 0.86], [0.76, 0.005, 0.86]]
            ),
        }
    )
    cells = _task_pose_cells(episode, "dual_arm_sort", cfg, nominal)
    assert cells == {
        ("block_1", "initial", 1, 1),
        ("block_3", "initial", 1, 1),
        ("block_1", "placed-fixed"),
        ("block_3", "placed-fixed"),
    }
    for value in episode.observations.values():
        value[-1, :2] += 0.01
    assert _task_pose_cells(episode, "dual_arm_sort", cfg, nominal) == cells


def test_relative_placement_offsets_use_the_reference_sampling_domain():
    nominal = {
        "block_1": [0.625, -0.15, 0.86],
        "block_2": [0.675, 0, 0.86],
        "block_3": [0.725, 0.15, 0.86],
    }
    bounds = [[-0.025, 0.025], [-0.03, 0.03], [0, 0]]
    cfg = {"scene": {uid: {"translation_m": bounds} for uid in nominal}}
    episode = SimpleNamespace(
        observations={
            "rigid/block_1/state": torch.tensor(
                [nominal["block_1"], [0.575, -0.02, 0.86]]
            ),
            "rigid/block_3/state": torch.tensor(
                [nominal["block_3"], [0.775, 0.02, 0.86]]
            ),
        }
    )
    cells = _task_pose_cells(episode, "dual_arm_sort", cfg, nominal)
    assert ("block_1", "placed", 0, 0) in cells
    assert ("block_3", "placed", 3, 3) in cells
