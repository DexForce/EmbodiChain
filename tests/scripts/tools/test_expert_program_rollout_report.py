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

import pytest

from scripts.tools.expert_program_rollout_report import (
    DEFAULT_REPORT_PATH,
    REPOSITORY_ROOT,
    build_task_size_metrics,
    main,
    render_report,
)

EXPECTED_CURRENT_COUNTS = {
    # Each tuple is (raw LF bytes, raw file bytes) for the explicit task pair.
    "Cube": (166, 5_645),
    "Drawer": (348, 12_600),
}

EXPECTED_SOURCE_PATHS = {
    "Cube": (
        "embodichain_tasks/embodichain_tasks/expert_program/" "repeated_pick_place.py",
        "embodichain_tasks/configs/expert_program/repeated_pick_place.yaml",
    ),
    "Drawer": (
        "embodichain_tasks/embodichain_tasks/expert_program/open_drawer.py",
        "embodichain_tasks/configs/expert_program/open_drawer.yaml",
    ),
}


def test_current_counts_use_only_the_four_declared_sources() -> None:
    metrics = build_task_size_metrics(REPOSITORY_ROOT)

    actual = {
        metric.task: (
            metric.current_lines,
            metric.current_bytes,
            tuple(source.path for source in metric.sources),
        )
        for metric in metrics
    }
    expected = {
        task: (*EXPECTED_CURRENT_COUNTS[task], EXPECTED_SOURCE_PATHS[task])
        for task in EXPECTED_CURRENT_COUNTS
    }
    assert actual == expected


def test_render_is_deterministic() -> None:
    metrics = build_task_size_metrics(REPOSITORY_ROOT)

    first = render_report(metrics)
    second = render_report(metrics)

    assert first == second


def test_render_rejects_empty_metric_snapshot() -> None:
    with pytest.raises(ValueError, match="at least one task snapshot"):
        render_report(())


def test_checked_in_report_matches_deterministic_render() -> None:
    expected = render_report(build_task_size_metrics(REPOSITORY_ROOT))

    assert DEFAULT_REPORT_PATH.read_text(encoding="utf-8") == expected


def test_check_mode_accepts_current_report() -> None:
    assert main(["--check"]) == 0


def test_check_mode_rejects_stale_report(tmp_path) -> None:
    stale_report = tmp_path / "expert_program_rollout_report.md"
    stale_report.write_text("stale\n", encoding="utf-8")

    assert main(["--check", "--output", str(stale_report)]) == 1
