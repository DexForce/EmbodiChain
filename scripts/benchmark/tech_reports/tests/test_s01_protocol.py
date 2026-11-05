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
"""CPU checks for S01 measurement/reporting boundaries; no simulator imports."""

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


runner = _load("s01_runner", "physics/experiments/dexuni/run_s01.py")
paths = _load("s01_paths", "physics/experiments/dexuni/paths.py")
media = _load("s01_media", "reporting/dexuni/build_s01_media.py")


def test_startup_is_excluded_without_selecting_favorable_tail() -> None:
    # Very slow startup cannot contaminate the declared timing scope.
    samples = np.r_[np.full(30, 8.0), 0.010, 0.020, 0.030]
    result = runner.summarize_steps(samples)
    assert result["step_ms"] == pytest.approx(20)
    assert result["startup_steps_s"] == pytest.approx(240)


@pytest.mark.parametrize(
    "samples", [np.ones(30), np.zeros(31), np.r_[np.ones(30), np.nan]]
)
def test_invalid_or_empty_measurement_window_is_rejected(samples) -> None:
    with pytest.raises(ValueError):
        runner.summarize_steps(samples)


def test_sources_and_symlinks_into_legacy_are_protected(tmp_path: Path) -> None:
    for owner in ("assets", "physics", "reporting", "core"):
        with pytest.raises(ValueError):
            paths.output_dir("s01c", TECH / owner / "measurements")
    legacy = TECH.parents[2] / "dexuni_benchmark"
    alias = tmp_path / "alias"
    alias.symlink_to(legacy, target_is_directory=True)
    with pytest.raises(ValueError):
        paths.output_dir("s01c", alias / "measurements")
    assert (
        paths.output_dir("s01c")
        == TECH.parents[2] / "outputs/benchmarks/tech_reports/dexuni/development/s01c"
    )


def test_failure_remains_visible_and_media_is_not_a_timing_sample(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "task": "s01c",
            "platform": "embodichain",
            "mode": "measurement",
            "success": True,
            "metrics": {
                "step_ms": np.float32(20),
                "hold_compression_pct": np.float32(5),
            },
        },
        {
            "task": "s01c",
            "platform": "embodichain",
            "mode": "measurement",
            "success": False,
            "error": "OOM",
        },
        {
            "task": "s01c",
            "platform": "embodichain",
            "mode": "media",
            "success": True,
            "metrics": {"step_ms": 9999},
        },
    ]
    runner._report(tmp_path, rows)
    report = (tmp_path / "REPORT.md").read_text()
    assert "20.000 ± 0.000" in report
    assert "5.000 ± 0.000" in report
    assert "| 1/2 |" in report
    assert "9999" not in report
    saved = json.loads((tmp_path / "attempts.json").read_text())
    assert saved[1]["error"] == "OOM"
    assert isinstance(saved[0]["metrics"]["hold_compression_pct"], float)
    assert sum(line.startswith("| ---") for line in report.splitlines()) == 3


def test_video_time_mapping_stops_after_source_ends() -> None:
    assert media.frame_index(0.0, 30, 3) == 0
    assert media.frame_index(2 / 30, 30, 3) == 2
    assert media.frame_index(3 / 30, 30, 3) is None
