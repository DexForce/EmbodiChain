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

"""Verify acceptance for the standard Franka RJ45 insertion Task Program."""

from __future__ import annotations

import numpy as np
import yaml

from embodichain.extensions.tasks.manipulation.cable.rj45_measurements import (
    RJ45_CHECKS,
    evaluate_rj45,
)
from embodichain.extensions.tasks.manipulation.cable.rj45_scene import (
    load_config,
    task_directory,
)


def test_rj45_evaluation_covers_all_published_physical_checks() -> None:
    component = task_directory() / "task_program/evaluation.yaml"
    evaluation = yaml.safe_load(component.read_text())
    rules = evaluation["graph"]["rules"]
    assert rules["rj45_success"]["refs"] == list(RJ45_CHECKS)
    assert evaluation["final_acceptance"] == ["rj45_success"]
    for check in RJ45_CHECKS:
        assert rules[check]["bindings"]["value"] == f"cables/cable/rj45_{check}"


def test_rj45_acceptance_rejects_missing_contact_and_final_placement_slip() -> None:
    config = load_config()
    tcp_pose = np.eye(4)
    tcp_pose[2, 1] = 1
    reference = {
        "tip_m": [0.48, -0.14, 0.0055],
        "actual_tcp_pose": tcp_pose,
        "depth_m": 0.0105,
        "lateral_m": 0.0002,
        "orientation_error_deg": 1.0,
        "latch_angle_deg": 0.0,
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
    }
    rows = [
        dict(reference, time_s=0.0),
        dict(reference, time_s=8.0, tip_m=[0.48, -0.14, 0.15]),
        dict(reference, time_s=18.5),
        dict(reference, time_s=24.5, finger_latch_contact=True, latch_angle_deg=-10.0),
        dict(reference, time_s=27.5, depth_m=-0.03),
        dict(reference, time_s=40.0),
    ]
    assert evaluate_rj45(rows, config)["checks"] == dict.fromkeys(RJ45_CHECKS, True)

    rows[1]["finger_latch_contact"] = None
    rows[-1]["return_error_m"] = 0.024
    report = evaluate_rj45(rows, config)
    assert not report["passed"]
    assert {name for name, status in report["checks"].items() if not status} == {
        "insertion_without_finger_latch_press",
        "placed_on_floor",
    }
