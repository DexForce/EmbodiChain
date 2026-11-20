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

"""Grasp metric contracts for fixed attempt budgets and physical retention."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import numpy as np
import torch
from embodichain_benchmarks.grasp_pose_generator.metrics import summarize
from embodichain.lab.sim.grasp_validation import GraspValidator, _result


def _trace(length=361):
    row = np.zeros((length, 19))
    row[:, 3:12] = np.eye(3).reshape(9)
    row[:, 12:15] = [0.09, 1, 1]
    return row


def test_rotating_retained_target_and_directional_drop():
    row = _trace()
    row[320:, 3:12] = np.diag([-1, -1, 1]).reshape(9)
    kept = _result(row, "x", False)
    assert kept["success"] and kept["disturbance"]["rotation_deg"] == 180
    row[335:, 13:15] = 0
    lost = _result(row, "y", False)
    assert lost["hold"]["success"] and not lost["success"]
    assert np.isclose(lost["disturbance"]["drop_s"], 0.35)
    assert _result(row, "x", True)["status"] == "missing"


def test_ranking_missing_evidence_and_zero_padding():
    rows = []
    for rank in (1, 2, 3):
        for direction in ("base", "x", "y"):
            trace = _trace()
            if rank != 3:
                trace[:, 13:15] = 0
            rows.append(
                dict(_result(trace, direction, False), rank=rank, condition=direction)
            )
    case = dict(
        count=3,
        trials=rows,
        object="cube",
        family="boxes",
        object_holdout=True,
        gripper="hand",
        algorithm="test",
    )

    def scores():
        return summarize([case], ["hand"])["folds"]["hand"]["both_new"]["algorithms"][
            "test"
        ]["metrics"]

    assert scores() == {
        "Success@1": 0,
        "Hit@5": 1,
        "Hit@10": 1,
        "Precision@10": 0.1,
        "Robust@1": 0,
    }
    rows[0].update(_result(_trace(), "base", False))
    rows[1].update(_result(_trace(), "x", False))
    assert scores()["Success@1"] == 1 and scores()["Robust@1"] == 0
    rows.pop()
    assert not summarize([case], ["hand"])["complete"]


def test_macro_weighting_and_paired_intervals():
    records = []
    good = [
        dict(_result(_trace(), d, False), rank=1, condition=d)
        for d in ("base", "x", "y")
    ]
    for method in ("first", "second"):
        for object_id, placements in (("easy", 3), ("hard", 1)):
            for seed in range(placements):
                rows = good if object_id == "easy" else []
                records.append(
                    dict(
                        object=object_id,
                        family=object_id,
                        gripper="hand",
                        object_holdout=True,
                        algorithm=method,
                        seed=seed,
                        count=int(bool(rows)),
                        trials=rows,
                    )
                )
    results = summarize(records, ["hand"])["folds"]["hand"]["both_new"]
    assert results["algorithms"]["first"]["metrics"]["Success@1"] == 0.5
    assert results["paired_difference_ci95"]["first minus second"]["Success@1"] == [
        0.0,
        0.0,
    ]
    assert not summarize(records[:-1], ["hand"])["complete"]


def test_dexuni_contact_evidence_uses_distance_only_for_unmeasured_rows():
    validator = GraspValidator.__new__(GraspValidator)
    validator._geometry_contacts = True
    validator.target_id = torch.tensor([[1]])
    validator.pad_ids = torch.tensor([[2, 3]])
    validator.support_id = torch.tensor([[4]])
    data = dict(
        user_ids=torch.tensor([[[1, 2], [3, 1], [1, 4]]]),
        is_valid=torch.ones(1, 3, dtype=torch.bool),
        impulse_valid=torch.tensor([[True, False, False]]),
        impulse=torch.tensor([[0.0, 0.0, 0.0]]),
        distance=torch.tensor([[-0.001, 0.0, 0.001]]),
    )
    validator.sensor = SimpleNamespace(
        update=lambda: None,
        get_data=lambda: data,
        get_latest_contact_loss_mask=lambda: torch.tensor([False]),
    )
    fingers, support, lost = validator._contacts()
    assert fingers.tolist() == [[False, True]]
    assert not support.item() and not lost.item()
    data["distance"][0, 1] = float("nan")
    assert validator._contacts()[2].item()


@pytest.mark.gpu
@pytest.mark.subprocess_sim
@pytest.mark.parametrize("physics", ["default", "newton"])
def test_gpu_cli_isolates_missed_and_held_grasps(tmp_path: Path, physics: str) -> None:
    """Two misses and a centred grasp share a native nine-arena batch."""
    fixtures = files("embodichain_benchmarks.grasp_pose_generator")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "embodichain",
            "benchmark",
            "grasp-pose-generator",
            "validate",
            "--physics",
            physics,
            "--case",
            str(fixtures / "block.yaml"),
            "--poses",
            str(fixtures / "candidates.jsonl"),
            "--num-envs",
            "9",
            "--output",
            str(tmp_path),
        ],
        check=True,
        timeout=120,
    )
    (checkpoint,) = tmp_path.glob("*.json")
    case = json.loads(checkpoint.read_text())
    assert case["protocol"]["physics"] == physics
    assert case["protocol"]["solver"] == (
        "dexuni" if physics == "newton" else "default"
    )
    assert case["finished"]
    trials = {(r["rank"], r["condition"]): r for r in case["trials"]}
    assert len(trials) == 9
    for direction in ("base", "x", "y"):
        assert not trials[1, direction]["success"]
        assert not trials[2, direction]["success"]
        assert trials[3, direction]["success"]
    for direction in ("x", "y"):
        trial = trials[3, direction]
        assert np.isfinite(trial["peak_acceleration_m_s2"])
        assert (
            1.5
            < trial["peak_axis_acceleration_m_s2"]
            <= trial["peak_acceleration_m_s2"]
        )
        if physics == "default":
            assert trial["peak_acceleration_m_s2"] < 2.5
