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

"""Tests that the atomic-action measurement layer follows the standard.

The criteria live in ``scripts/benchmark/atomic_action/BENCHMARK_STANDARD.md``.
These tests read that document and check the shipped code against it, so a
stage list, a failure reason, or a tolerance cannot drift from the text that
defines it. The rest covers the stage bookkeeping the benchmarks rely on.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from scripts.benchmark.atomic_action.common import (
    FAILURE_TAXONOMY,
    PHYSICAL_DROP_MARGIN_M,
    PRIMITIVE_POSITION_TOLERANCE_M,
    PRIMITIVE_ROTATION_TOLERANCE_RAD,
    SKILL_STAGES,
    StageLadder,
    TASK_POSITION_TOLERANCE_M,
    TASK_ROTATION_TOLERANCE_RAD,
    build_stage_leaderboard,
    dropped_below_support,
    hand_is_released,
)

STANDARD_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "benchmark"
    / "atomic_action"
    / "BENCHMARK_STANDARD.md"
)
CAMEL_BOUNDARY = re.compile(r"(?<!^)(?=[A-Z])")


def _standard_text() -> str:
    """Return the measurement standard's text."""
    return STANDARD_PATH.read_text(encoding="utf-8")


def _documented_stages() -> dict[str, tuple[str, ...]]:
    """Return the skill-to-stages table from section 1 of the standard."""
    stages: dict[str, tuple[str, ...]] = {}
    for line in _standard_text().splitlines():
        columns = [column.strip() for column in line.strip().strip("|").split("|")]
        if len(columns) != 4 or not columns[0].startswith("`"):
            continue
        if columns[1] not in ("`PRIMITIVE_*`", "`TASK_*`"):
            continue
        skill = CAMEL_BOUNDARY.sub("_", columns[0].strip("`")).lower()
        stages[skill] = tuple(name.strip().strip("`") for name in columns[2].split("→"))
    return stages


def _ladder(*recorded: tuple[str, bool]) -> StageLadder:
    """Return a pick_up ladder with the given stage outcomes recorded."""
    ladder = StageLadder(stages=SKILL_STAGES["pick_up"])
    for stage, passed in recorded:
        ladder.record(stage, passed, "" if passed else "task_goal_miss")
    return ladder


def test_skill_stages_match_the_standard() -> None:
    """Every skill's stages are exactly the ones section 1 lists, in order."""
    documented = _documented_stages()
    assert documented, "No skill rows parsed from the standard."
    assert SKILL_STAGES == documented


def test_failure_taxonomy_matches_the_standard() -> None:
    """The failure reasons are exactly the task-level subset section 1 names."""
    reasons = re.search(
        r"Reasons are the task-level subset\n(.*?)\n\n", _standard_text(), re.S
    )
    assert reasons is not None
    documented = set(re.findall(r"`([a-z_]+)`", reasons.group(1)))
    assert set(FAILURE_TAXONOMY) == documented


def test_tolerances_match_the_standard() -> None:
    """The tolerance constants carry the values section 0 publishes."""
    text = _standard_text()
    for name, value in (
        ("PRIMITIVE_POSITION_TOLERANCE_M", PRIMITIVE_POSITION_TOLERANCE_M),
        ("PRIMITIVE_ROTATION_TOLERANCE_RAD", PRIMITIVE_ROTATION_TOLERANCE_RAD),
        ("TASK_POSITION_TOLERANCE_M", TASK_POSITION_TOLERANCE_M),
        ("TASK_ROTATION_TOLERANCE_RAD", TASK_ROTATION_TOLERANCE_RAD),
        ("PHYSICAL_DROP_MARGIN_M", PHYSICAL_DROP_MARGIN_M),
    ):
        match = re.search(rf"^{name}\s*=\s*([0-9.]+)", text, re.M)
        assert match is not None, f"{name} is not published in the standard."
        assert float(match.group(1)) == pytest.approx(value)


def test_a_stage_after_a_failure_is_not_reached() -> None:
    """Recording past the first failure is ignored, so no later stage counts."""
    ladder = _ladder(("grasped", True), ("lifted", False))
    ladder.record("held", True)

    assert ladder.passed == {"grasped": True, "lifted": False}
    assert ladder.failure_stage == "lifted"
    assert ladder.failure_reason == "task_goal_miss"
    assert not ladder.success


def test_a_closing_stage_fails_the_case() -> None:
    """A skill that lifts an object and then drops it did not pick it up."""
    ladder = _ladder(("grasped", True), ("lifted", True), ("held", False))

    assert not ladder.success
    assert ladder.failure_stage == "held"


def test_stages_must_be_recorded_in_order() -> None:
    """A stage recorded out of its declared order is a programming error."""
    ladder = StageLadder(stages=SKILL_STAGES["pick_up"])
    with pytest.raises(ValueError, match="out of order"):
        ladder.record("lifted", True)


def test_a_failure_reason_must_come_from_the_taxonomy() -> None:
    """A reason outside the taxonomy is rejected rather than reported."""
    ladder = StageLadder(stages=SKILL_STAGES["pick_up"])
    with pytest.raises(ValueError, match="Unknown failure reason"):
        ladder.record("grasped", False, "controller_tracking_failure")


def test_an_unreached_stage_reports_not_applicable() -> None:
    """An unreached stage is N/A, never zero, in the per-case row."""
    fields = _ladder(("grasped", False)).as_row_fields()

    assert fields["grasped"] == "0.000000"
    assert fields["lifted"] == "N/A"
    assert fields["held"] == "N/A"
    assert fields["failure_reason"] == "task_goal_miss"


def test_diagnostics_never_decide_the_outcome() -> None:
    """Plan validity and tracking error are carried, and change nothing."""
    ladder = _ladder(("grasped", True), ("lifted", True), ("held", True))
    ladder.motion_valid = False
    ladder.motion_detail = "joint_limit_violation"
    ladder.max_tracking_error_rad = 1.5

    assert ladder.success
    assert ladder.as_row_fields()["motion_valid"] == "0.000000"
    assert ladder.as_row_fields()["max_tracking_error_rad"] == "1.5000"


def test_stage_rates_divide_by_the_cases_that_reached_the_stage() -> None:
    """stage_success_rate[i] = passed(i) / reached(i), success = passed(last)."""
    results = [
        {"ladder": _ladder(("grasped", True), ("lifted", True), ("held", True))},
        {"ladder": _ladder(("grasped", True), ("lifted", False))},
        {"ladder": _ladder(("grasped", False))},
    ]

    row = build_stage_leaderboard("pick_up", results)[0]

    assert row["grasped_rate"] == "66.67%"
    assert row["lifted_rate"] == "50.00%"
    assert row["held_rate"] == "100.00%"
    assert row["success_rate"] == "33.33%"
    assert row["evaluated_cases"] == 3


def test_a_stage_no_case_reached_is_not_applicable() -> None:
    """A rate with an empty denominator is reported as N/A, not as zero."""
    row = build_stage_leaderboard("pick_up", [{"ladder": _ladder(("grasped", False))}])[
        0
    ]

    assert row["lifted_rate"] == "N/A"
    assert row["success_rate"] == "0.00%"


def test_a_drop_is_measured_against_the_lowest_point_of_the_replay() -> None:
    """An object dropped mid-motion counts even if it rolls back to the target."""
    support = 0.50

    assert dropped_below_support(support - PHYSICAL_DROP_MARGIN_M - 0.01, support)
    assert not dropped_below_support(support - PHYSICAL_DROP_MARGIN_M + 0.01, support)
    assert not dropped_below_support(float("nan"), support)


def test_release_compares_the_two_commands_the_skill_used() -> None:
    """The hand is released when it ends nearer its open than its closed pose."""
    import torch

    class _Robot:
        def __init__(self, achieved: list[float]) -> None:
            self._achieved = torch.tensor([achieved])

        def get_qpos(self, name: str, target: bool) -> torch.Tensor:
            del name, target
            return self._achieved

    open_qpos = torch.tensor([0.0, 0.0])
    close_qpos = torch.tensor([0.04, 0.04])

    assert hand_is_released(_Robot([0.005, 0.005]), "hand", open_qpos, close_qpos)
    assert not hand_is_released(_Robot([0.038, 0.038]), "hand", open_qpos, close_qpos)
