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

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("newton")

from scripts.benchmark.scene.feasibility_collision import (
    _agreement,
    _markdown,
    _qualification_summary,
)

pytestmark = pytest.mark.no_sim


def test_incomplete_graph_qualification_is_unranked_and_errors_visible(
    tmp_path: Path,
) -> None:
    def complete(fixture: str, algorithm: str) -> dict:
        return {
            "fixture": fixture,
            "algorithm": algorithm,
            "scope": "full_corpus_qualification",
            "success_rate": 1.0,
            "matched": 2,
            "total": 2,
            "false_negative": 0,
            "false_positive": 0,
            "ambiguous": 0,
            "outside_detection_band": 0,
        }

    qualifications = [
        complete("primitive", "fcl_cpu"),
        complete("robot", "fcl_cpu"),
        complete("primitive", "newton_cuda_graph"),
        {
            "fixture": "robot",
            "scope": "cuda_graph_qualification_error",
            "reason": "contact buffer saturated",
        },
    ]
    summary = {r["algorithm"]: r for r in _qualification_summary([], qualifications)}
    assert summary["fcl_cpu"]["success_rate"] == 1.0
    graph = summary["newton_cuda_graph"]
    assert graph["success_rate"] is None
    assert graph["coverage"] == "1/2" and graph["status"] == "incomplete"
    output = tmp_path / "report.md"
    _markdown(output, [], {}, qualifications)
    text = output.read_text()
    assert "contact buffer saturated" in text and "| incomplete |" in text


def test_nonfinite_or_boundary_gaps_cannot_count_as_agreement() -> None:
    metrics = _agreement(np.array([False, True, True]), np.array([np.nan, np.inf, 0.0]))
    assert metrics["matched"] == 1 and metrics["total"] == 3
    assert metrics["invalid_outputs"] == 1 and metrics["ambiguous"] == 1
