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

"""Verify the declarative acceptance graph of the FR3 connector task."""

from __future__ import annotations

from copy import deepcopy

import yaml

from embodichain.extensions.tasks.configs import get_config_path
from embodichain.lab.evaluation._episode_reduction import (
    _evaluate_scalar_checks,
    _reduce_measurements,
)


def test_rj45_acceptance_requires_simultaneous_press_and_final_placement() -> None:
    path = get_config_path(
        "tasks/manipulation/cable/franka_rj45_cycle/task_program/evaluation.yaml"
    )
    evaluation = yaml.safe_load(path.read_text())
    graph = evaluation["graph"]
    check_names = graph["rules"]["rj45_success"]["refs"]
    assert len(check_names) == 17
    reductions = evaluation["measurements"]["reductions"]
    reference = {
        "tip_z_m": 0.0055,
        "tool_axis_z_abs": 1.0,
        "depth_m": 0.0105,
        "lateral_m": 0.0002,
        "orientation_error_deg": 1.0,
        "latch_angle_deg": 0.0,
        "latch_angle_abs_deg": 0.0,
        "finger_latch_contact": False,
        "latch_socket_contact": True,
        "latch_limit_violation_rad": 0.01,
        "return_error_m": 0.0002,
        "return_orientation_error_deg": 1.0,
        "home_error_rad": 0.00001,
        "finger_gap_m": 0.079,
        "anchor_error_m": 0.000001,
        "attachment_error_m": 0.0000005,
        "penetration_m": 0.0007,
        "dropped_contacts": 0,
        "finite": True,
        "press_samples": [],
    }
    rows = [dict(reference, frame=frame) for frame in range(2401)]
    for row in rows[360:541]:
        row["tip_z_m"] = 0.15
    rows[1470]["finger_latch_contact"] = True
    rows[1470]["press_samples"] = [{"depth_m": 0.0105, "latch_angle_deg": -10.0}]
    rows[1650]["depth_m"] = -0.03

    passed = _evaluate_scalar_checks(graph, _reduce_measurements(rows, reductions))
    assert passed["rj45_success"]

    asynchronous = deepcopy(rows)
    asynchronous[1470]["press_samples"] = [
        {"depth_m": 0.0105, "latch_angle_deg": 0.0},
        {"depth_m": -0.03, "latch_angle_deg": -10.0},
    ]
    asynchronous[-1]["return_error_m"] = 0.024
    failed = _evaluate_scalar_checks(
        graph, _reduce_measurements(asynchronous, reductions)
    )
    assert {name for name in check_names if not failed[name]} == {
        "physical_latch_press_before_withdrawal",
        "placed_on_floor",
    }
