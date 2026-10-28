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

"""Protect paired held-out simulation report aggregation."""

from __future__ import annotations

import json

import pytest

from examples.learning.lewm._evaluation import merge_sim_reports


def test_merge_requires_exact_seed_coverage_and_matching_pairs(tmp_path) -> None:
    def row(seed, success, distance):
        return {
            "seed": seed,
            "success": success,
            "distance": distance,
            "planning_seconds": [0.5],
            "cycle_seconds": [0.6],
            "initial_cube": [seed, 0],
            "initial_qpos": [1],
            "initial_goal": [2],
        }

    paths = []
    for index in range(2):
        seed = 300_000 + index
        report = {
            "split": "test",
            "weights_sha256": "same",
            "cem": {"horizon": 5},
            "step_dt": 0.04,
            "success_threshold": 0.1,
            "device": "cuda:0",
            "gpu": "test",
            "torch": "test",
            "gpu_peak_bytes": 1,
            "start_index": index,
            "episodes": 1,
            "records": {
                "lewm": [row(seed, index == 0, 0.12)],
                "random": [row(seed, False, 0.14)],
            },
        }
        path = tmp_path / f"{index}.json"
        path.write_text(json.dumps(report))
        paths.append(str(path))
    merged = merge_sim_reports(paths, 2)
    assert merged["passed"]
    assert merged["summary"]["lewm"]["success_rate"] == 0.5
    assert merged["summary"]["random"]["mean_final_distance"] == pytest.approx(0.14)
    with pytest.raises(ValueError, match="each expected seed"):
        merge_sim_reports([paths[0], paths[0]], 2)
    report = json.loads((tmp_path / "1.json").read_text())
    report["records"]["random"][0]["initial_goal"] = [3]
    (tmp_path / "1.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="different initial states"):
        merge_sim_reports(paths, 2)
