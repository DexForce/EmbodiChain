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

"""CPU contracts for ODS rate, memory and selection-budget configuration."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from examples.data_pipeline import _ods_pick_place_collection as collection
from examples.data_pipeline.ods_pick_place_showcase import _parse_args


def _snapshot(generation: int) -> dict:
    return {
        "observations": torch.full((2, 3, 2), float(generation)),
        "actions": torch.zeros(2, 3, 2),
        "valid": torch.ones(2, 3, dtype=torch.bool),
        "frame_ids": torch.tensor(
            [[[7, (generation - 1) * 2 + r, t] for t in range(3)] for r in range(2)]
        ),
        "generations": torch.full((2, 3), generation),
    }


def test_direct_collection_overlaps_next_generation_and_preserves_row_order(
    tmp_path, monkeypatch
):
    calls = []
    rows = []
    engine = SimpleNamespace(run_id=7, completed=1)
    engine.get_stats = lambda: {"completed_generation": engine.completed}

    class Writer:
        def __init__(self, *a, **kw):
            self.stats = {"rows": 4}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def submit(self, row, progress):
            calls.append("submit")
            progress()
            rows.append(row["frame_ids"][0, 1].item())

        def finalize(self, progress):
            calls.append("finalize")
            progress()

    def refill():
        calls.append("refill")
        return 2

    def progress():
        engine.completed = 2

    monkeypatch.setattr(collection, "DirectLeRobotWriter", Writer)
    times = iter(range(100))
    report = collection.collect_lerobot(
        engine,
        _snapshot(1),
        tmp_path / "dataset",
        schema={},
        episodes=4,
        snapshot=lambda e: _snapshot(2),
        request_refill=refill,
        progress=progress,
        clock=lambda: float(next(times)),
        record_interval=lambda *args: None,
    )
    assert calls.index("refill") < calls.index("submit")
    assert rows == [0, 1, 2, 3]
    assert report["intermediate_pt_file"] is False
    assert not list(tmp_path.rglob("*.pt"))


def test_cold_offline_episode_budget_is_independent_of_training_rounds(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "showcase",
            "--policy",
            "chunk-pose",
            "--mode",
            "offline-full",
            "--offline-storage",
            "lerobot",
            "--offline-episodes",
            "256",
            "--producer-envs",
            "8",
            "--rounds",
            "1",
            "--updates-per-round",
            "2500",
            "--defer-evaluation",
            "--deferred-evaluation-schedule",
            "updates",
        ],
    )
    args = _parse_args()
    assert args.offline_episodes == args.max_generated_episodes == 256
    assert args.rounds * args.updates_per_round == 2500
    assert args.deferred_evaluation_schedule == "updates"


@pytest.mark.parametrize(
    "flags",
    [
        ["--offline-episodes", "31"],
        ["--offline-episodes", "64", "--max-generated-episodes", "256"],
        ["--offline-sample-budget", "4"],
        ["--offline-sample-budget", "5", "--offline-sample-hard-limit", "4"],
        ["--offline-storage", "lerobot", "--export-data", "bad.pt"],
    ],
)
def test_invalid_collection_budget_or_transport_is_rejected(monkeypatch, flags):
    monkeypatch.setattr(
        sys,
        "argv",
        ["showcase", "--policy", "chunk-pose", "--mode", "offline-full", *flags],
    )
    with pytest.raises(SystemExit) as error:
        _parse_args()
    assert error.value.code == 2


def test_generation_wait_has_a_deadline_without_global_wall_budget(
    tmp_path, monkeypatch
):
    engine = SimpleNamespace(run_id=7)
    engine.get_stats = lambda: {"completed_generation": 1}
    closed = []

    class Writer:
        def __init__(self, *args, **kwargs):
            self.stats = {"worker_joined": True}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            closed.append(True)

        def submit(self, row, progress):
            progress()

    monkeypatch.setattr(collection, "DirectLeRobotWriter", Writer)
    ticks = iter([0.0, 1.0, 2.0, 3.0])
    monkeypatch.setattr(collection.time, "monotonic", lambda: next(ticks))
    with pytest.raises(TimeoutError, match="deadline"):
        collection.collect_lerobot(
            engine,
            _snapshot(1),
            tmp_path / "dataset",
            schema={},
            episodes=4,
            snapshot=lambda e: _snapshot(2),
            request_refill=lambda: 2,
            progress=lambda: None,
            clock=lambda: 0.0,
            record_interval=lambda *args: None,
            generation_timeout=0.1,
        )
    assert closed == [True]


def test_unjoined_writer_cannot_be_mistaken_for_normal_budget_completion(
    tmp_path, monkeypatch
):
    class BudgetExpired(Exception):
        pass

    class Writer:
        def __init__(self, *args, **kwargs):
            self.stats = {"worker_joined": False}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def progress():
        raise BudgetExpired()

    monkeypatch.setattr(collection, "DirectLeRobotWriter", Writer)
    with pytest.raises(RuntimeError, match="did not join") as error:
        collection.collect_lerobot(
            SimpleNamespace(run_id=7),
            _snapshot(1),
            tmp_path / "dataset",
            schema={},
            episodes=2,
            snapshot=lambda e: _snapshot(2),
            request_refill=lambda: 2,
            progress=progress,
            clock=lambda: 0.0,
            record_interval=lambda *args: None,
        )
    assert isinstance(error.value.__cause__, BudgetExpired)


def test_protected_offline_resume_cannot_reset_existing_source_quotas(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "showcase",
            "--policy",
            "chunk-pose",
            "--mode",
            "offline-full",
            "--offline-sample-budget",
            "8",
            "--offline-sample-hard-limit",
            "8",
            "--resume",
            "old.pt",
        ],
    )
    with pytest.raises(SystemExit) as error:
        _parse_args()
    assert error.value.code == 2
