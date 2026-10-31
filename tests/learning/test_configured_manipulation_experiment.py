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

"""Prevent short or stale experiments from claiming formal qualification."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import runpy

import pytest


@pytest.fixture
def experiment(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    module = runpy.run_path(
        str(root / "examples/learning/configured_manipulation/compare_push_cube.py")
    )
    monkeypatch.setitem(
        module["_validate_result"].__globals__, "_fingerprint", lambda *args: "expected"
    )
    return module


def _result(variant="configured", seed=42):
    cases = [
        {"case_seed": 20000 + case, "initial": {"cube": [0, 0, 0]}, "success": True}
        for case in range(3)
    ]
    return {
        "fingerprint": "expected",
        "variant": variant,
        "seed": seed,
        "iterations": 2,
        "transitions": 131072,
        "untrained_cases": deepcopy(cases),
        "final_cases": deepcopy(cases),
        "success_rate": 1.0,
        "untrained_success_rate": 0.0,
    }


@pytest.mark.parametrize("change", ["budget", "cases", "fingerprint"])
def test_stale_experiment_cannot_be_reused(experiment, change):
    result = _result()
    if change == "budget":
        result["iterations"] = 1
    elif change == "cases":
        result["final_cases"].pop()
    else:
        result["fingerprint"] = "different"
    with pytest.raises(ValueError):
        experiment["_validate_result"](result, "configured", 42, 2, 3)


def test_smoke_summary_reports_actual_budget_without_formal_acceptance(
    experiment, tmp_path
):
    for variant in ("legacy", "configured"):
        for seed in (42, 43, 44):
            directory = tmp_path / f"{variant}_{seed}"
            directory.mkdir()
            (directory / "result.json").write_text(json.dumps(_result(variant, seed)))
    experiment["_summarize"](tmp_path, iterations=2, cases=3)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["formal_protocol"] is False
    assert summary["accepted"] is False
    assert summary["cases_per_checkpoint"] == 3
    assert "1024 rollout steps × 2 updates" in (tmp_path / "report.md").read_text()
    with pytest.raises(ValueError):
        experiment["_summarize"](tmp_path)
