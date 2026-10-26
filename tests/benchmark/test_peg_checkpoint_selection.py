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
"""Guard balanced model selection against incomplete or misleading evidence."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest
import torch

from scripts.benchmark.assembly.select_peg import (
    _discover,
    _select_best,
    _verified_success,
)


def test_selection_prefers_score_then_earlier_update() -> None:
    rows = [
        {"status": "completed", "success_rate": 0.8, "num_updates": 100, "sha256": "a"},
        {"status": "completed", "success_rate": 0.9, "num_updates": 300, "sha256": "b"},
        {"status": "completed", "success_rate": 0.9, "num_updates": 200, "sha256": "c"},
    ]
    assert _select_best(rows)["num_updates"] == 200


@pytest.mark.parametrize("status,score", [("failed", 1.0), ("completed", float("nan"))])
def test_unverified_candidate_cannot_silently_disappear(
    status: str, score: float
) -> None:
    with pytest.raises(ValueError):
        _select_best(
            [{"status": status, "success_rate": score, "num_updates": 1, "sha256": "x"}]
        )


def test_selection_checks_success_against_raw_episodes(tmp_path: Path) -> None:
    path = tmp_path / "episodes.jsonl"
    path.write_text(
        "\n".join(json.dumps({"success": value}) for value in [1, 0, 1, 0]) + "\n"
    )
    assert _verified_success(path, 4, 0.5) == (0.5, 2)
    with pytest.raises(ValueError, match="disagrees"):
        _verified_success(path, 4, 1.0)
    with pytest.raises(ValueError, match="one binary success"):
        _verified_success(path, 3, 0.5)


def test_duplicate_update_requires_identical_policy_state(tmp_path: Path) -> None:
    folder = tmp_path / "checkpoints"
    folder.mkdir()
    checkpoint = {
        "num_updates": 50,
        "global_step": 100,
        "policy": {"weight": torch.tensor([1.0])},
    }
    torch.save(checkpoint, folder / "first.pt")
    torch.save(checkpoint, folder / "second.pt")
    assert len(_discover(tmp_path)) == 1
    checkpoint["policy"]["weight"] = torch.tensor([2.0])
    torch.save(checkpoint, folder / "second.pt")
    with pytest.raises(ValueError, match="Different policy states"):
        _discover(tmp_path)


def test_report_binds_selection_to_tested_weights_and_independent_seed(
    tmp_path: Path,
) -> None:
    from scripts.benchmark.assembly.rl_report import _selection

    run = tmp_path.resolve()
    weights = run / "policy.pt"
    weights.write_bytes(b"frozen policy identity")
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    row = {
        "status": "completed",
        "success_rate": 0.8,
        "num_updates": 50,
        "global_step": 100,
        "sha256": digest,
        "path": str(weights),
    }
    document = {
        "schema": "embodichain-peg-balanced-selection/v1",
        "status": "completed",
        "run": str(run),
        "protocol": {"episode_allocation": "balanced", "seed": 10000},
        "declared_candidates": [row],
        "candidates": [row],
        "selected_checkpoint": row,
    }
    path = run / "selection.json"
    path.write_text(json.dumps(document))
    assessment = {"checkpoint": {"sha256": digest}, "evaluation_reset_seed": 20000}
    assert _selection(path, run, assessment)["selected_checkpoint"]["sha256"] == digest
    with pytest.raises(ValueError, match="different validation-selected checkpoint"):
        _selection(path, run, {**assessment, "checkpoint": {"sha256": "other weights"}})
    with pytest.raises(ValueError, match="seed must differ"):
        _selection(path, run, {**assessment, "evaluation_reset_seed": 10000})
