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
"""CPU checks for S02 reporting and measurement boundaries."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

TECH = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, TECH / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load("s02_runner", "physics/experiments/dexuni/run_s02.py")


def test_s02_report_keeps_failure_and_excludes_media(tmp_path: Path) -> None:
    attempts = [
        {
            "platform": "embodichain",
            "mode": "measurement",
            "success": True,
            "metrics": {
                "step_ms": np.float32(20),
                "grasp_compression_pct": np.float32(35),
                "place_center_error_mm": 0.5,
            },
        },
        {
            "platform": "embodichain",
            "mode": "measurement",
            "success": False,
            "error": "worker timeout",
        },
        {
            "platform": "embodichain",
            "mode": "media",
            "success": True,
            "metrics": {"step_ms": 9999, "grasp_compression_pct": 9999},
        },
    ]
    runner._report(tmp_path, attempts)
    report = (tmp_path / "REPORT.md").read_text()
    assert "| 1/2 |" in report
    assert "35.000 ± 0.000" in report
    assert "9999" not in report
    assert sum(line.startswith("| ---") for line in report.splitlines()) == 3
    saved = json.loads((tmp_path / "attempts.json").read_text())
    assert saved[1]["error"] == "worker timeout"
    assert isinstance(saved[0]["metrics"]["grasp_compression_pct"], float)


def test_empty_s02_report_has_explicit_missing_samples(tmp_path: Path) -> None:
    runner._report(tmp_path, [])
    report = (tmp_path / "REPORT.md").read_text()
    assert report.count("| 0/5 |") == 2
    assert "nan" not in report.lower()
    assert "N/A" in report


def test_s02_warmup_rule_does_not_include_graph_capture() -> None:
    measurements = np.r_[np.full(30, 5.0), np.full(40, 0.04)]
    assert abs(runner.summarize_steps(measurements)["step_ms"] - 40) < 1e-9


checker = _load("s02_checker", "physics/experiments/dexuni/check_s02_adapter.py")


def _states(shift: float) -> dict:
    return {
        "joint_names": np.array(["joint"]),
        "phase": np.repeat(np.arange(10), 3),
        "particles": np.full((30, 2, 3), shift),
        "width": np.full(30, 0.04),
    }


def test_adapter_check_accepts_submillimetre_motion_difference() -> None:
    checker.compare_states(_states(0.0), _states(0.0002))


def test_adapter_check_rejects_material_motion_mismatch() -> None:
    with pytest.raises(AssertionError, match="Adapter differs"):
        checker.compare_states(_states(0.0), _states(0.004))
