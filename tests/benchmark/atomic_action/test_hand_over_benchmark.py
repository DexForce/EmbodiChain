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

"""Regression tests for HandOver's post-release placement measurement."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from scripts.benchmark.atomic_action import hand_over_benchmark as benchmark
from scripts.benchmark.atomic_action.common import SKILL_STAGES, StageLadder


@pytest.mark.parametrize(
    ("xy_error", "height_error", "expected"),
    [
        # Observed can/pencil settling must not consume the horizontal budget.
        (0.02, -0.041777, True),
        (0.02, -0.065871, True),
        (0.03, -0.10, True),
        (0.03, 0.03, True),
        (0.0, 0.0, True),
        # The same resting offsets above the target must not count as placed.
        (0.02, 0.041777, False),
        (0.02, 0.065871, False),
        (0.0, 0.10, False),
        (0.0, 0.031, False),
        (0.031, 0.0, False),
        (0.031, -0.065871, False),
        (0.0, -0.101, False),
        (float("nan"), 0.0, False),
        (0.0, float("nan"), False),
        (float("inf"), 0.0, False),
        (0.0, float("inf"), False),
        (0.0, float("-inf"), False),
        (-0.01, 0.0, False),
    ],
)
def test_delivery_uses_independent_horizontal_and_height_budgets(
    xy_error: float, height_error: float, expected: bool
) -> None:
    """Settling passes, while lateral misses and excessive height still fail."""
    assert benchmark._delivery_goal_reached(xy_error, height_error) is expected


def test_report_keeps_3d_distance_as_a_diagnostic() -> None:
    """A passing placement can have a 3D error above the old 3 cm threshold."""
    xy_error, height_error = 0.02, -0.065871
    distance = math.hypot(xy_error, height_error)
    ladder = StageLadder(stages=SKILL_STAGES["hand_over"])
    for stage in ladder.stages[:-1]:
        ladder.record(stage, True, "task_goal_miss")
    ladder.record(
        "placed",
        benchmark._delivery_goal_reached(xy_error, height_error),
        "task_goal_miss",
    )
    result = benchmark._case_result(
        benchmark.HAND_OVER_CASES["horizontal_pencil"],
        0,
        ladder,
        0.1,
        None,
        SimpleNamespace(
            initial_position=0.23,
            settled_position=distance,
            max_tracking_error_rad=0.0,
        ),
        0.534129,
        False,
        "",
        final_xy_error=xy_error,
        final_height_error=height_error,
    )
    _, rows = benchmark._build_rows([result])

    assert result["success"]
    assert result["final_delivery_distance_m"] > 0.03
    assert rows[0]["placed"] == "1.000000"
    assert float(rows[0]["final_delivery_xy_error_m"]) == pytest.approx(xy_error)
    assert float(rows[0]["final_delivery_z_error_m"]) == pytest.approx(
        height_error, abs=0.00005
    )


def test_failed_planning_reports_unmeasured_delivery_errors() -> None:
    """Planning failures have no placement measurements to populate."""
    ladder = StageLadder(stages=SKILL_STAGES["hand_over"])
    ladder.fail("grasped", "planner_reported_failure")
    result = benchmark._case_result(
        benchmark.HAND_OVER_CASES["vertical_can"],
        0,
        ladder,
        0.1,
        None,
        None,
        None,
        None,
        "",
    )
    _, rows = benchmark._build_rows([result])

    assert not result["success"]
    assert result["final_delivery_xy_error_m"] is None
    assert result["final_delivery_z_error_m"] is None
    assert rows[0]["placed"] == "N/A"
