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

"""Measured pouring acceptance from sampled rigid poses, without fluid claims."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from embodichain.lab.task_program.integrations.simulation.asset_pouring import (
    PouringAcceptanceMonitor,
)

BOTTLE_MOUTH = (0.0, 0.0, 0.20)
CUP_MOUTH = (0.0, 0.0, 0.10)
BOTTLE_RADIUS = 0.01
CUP_RADIUS = 0.04


def _pose(position=(0, 0, 0), *, angle=0.0, yaw=0.0):
    result = np.eye(4)
    c, s = math.cos(angle), math.sin(angle)
    cy, sy = math.cos(yaw), math.sin(yaw)
    result[:3, :3] = np.array(((cy, -sy, 0), (sy, cy, 0), (0, 0, 1))) @ np.array(
        ((c, 0, s), (0, 1, 0), (-s, 0, c))
    )
    result[:3, 3] = position
    return result


def _pour_pose(cup, *, tilt=math.radians(60), radial=0.0, height=0.05):
    bottle = _pose(angle=tilt)
    desired_mouth = (
        cup[:3, :3] @ (np.array(CUP_MOUTH) + np.array((radial, 0, height))) + cup[:3, 3]
    )
    bottle[:3, 3] = desired_mouth - bottle[:3, :3] @ BOTTLE_MOUTH
    return bottle


def _monitor(**kwargs):
    options = dict(
        bottle_mouth_xyz=BOTTLE_MOUTH,
        cup_mouth_xyz=CUP_MOUTH,
        bottle_mouth_radius_m=BOTTLE_RADIUS,
        cup_opening_radius_m=CUP_RADIUS,
    )
    options.update(kwargs)
    return PouringAcceptanceMonitor(**options)


def _start(**kwargs):
    monitor = _monitor(**kwargs)
    bottle = _pose((-0.3, 0, 0.1))
    cup = _pose()
    monitor.reset(bottle, cup)
    return monitor, bottle, cup


def _observe_valid(monitor, cup):
    for _ in range(2):
        monitor.observe(_pour_pose(cup), cup, in_pour_segment=True)


def _statuses(result):
    return {check.check_id: check.status for check in result.checks}


def test_valid_simultaneous_trace_and_return_pass_without_claiming_liquid_transfer():
    monitor, bottle, cup = _start()
    _observe_valid(monitor, cup)
    result = monitor.validate_final(bottle, cup)
    assert result.accepted
    metadata = monitor.to_metadata()
    assert metadata["qualifying_sample_count"] == 2
    assert metadata["fluid_transfer_measured"] is False
    assert metadata["max_pour_tilt_rad"] == pytest.approx(math.radians(60))
    json.dumps(metadata, allow_nan=False)


def test_angle_and_alignment_at_different_samples_cannot_combine_into_success():
    monitor, bottle, cup = _start()
    for _ in range(2):
        monitor.observe(_pour_pose(cup, radial=0.06), cup, in_pour_segment=True)
        monitor.observe(_pour_pose(cup, tilt=0), cup, in_pour_segment=True)
    assert not monitor.validate_final(bottle, cup).accepted
    assert monitor.to_metadata()["max_pour_tilt_rad"] > math.radians(50)
    assert monitor.to_metadata()["min_pour_radial_distance_m"] == pytest.approx(0)
    assert monitor.to_metadata()["qualifying_sample_count"] == 0


@pytest.mark.parametrize(
    "radial,height", [(0.06, 0.05), (0.035, 0.05), (0, -0.01), (0, 0.005), (0, 0.30)]
)
def test_wrong_aperture_or_height_rejects_tilted_trace(radial, height):
    monitor, bottle, cup = _start()
    for _ in range(2):
        monitor.observe(
            _pour_pose(cup, radial=radial, height=height), cup, in_pour_segment=True
        )
    assert (
        _statuses(monitor.validate_final(bottle, cup))["pour.simultaneous_alignment"]
        == "failed"
    )


def test_no_samples_or_missing_reset_never_accepts():
    monitor = _monitor()
    bottle = _pose((-0.3, 0, 0.1))
    cup = _pose()
    assert not monitor.validate_final(bottle, cup).accepted
    with pytest.raises(RuntimeError, match="reset"):
        monitor.observe(bottle, cup, in_pour_segment=True)
    monitor.reset(bottle, cup)
    result = monitor.validate_final(bottle, cup)
    assert not result.accepted
    assert _statuses(result)["pour.samples_valid"] == "unavailable"


def test_samples_outside_pour_segment_do_not_count_and_one_sample_is_insufficient():
    monitor, bottle, cup = _start()
    for _ in range(3):
        monitor.observe(_pour_pose(cup), cup, in_pour_segment=False)
    assert not monitor.validate_final(bottle, cup).accepted
    monitor.observe(_pour_pose(cup), cup, in_pour_segment=True)
    assert not monitor.validate_final(bottle, cup).accepted
    assert monitor.to_metadata()["qualifying_sample_count"] == 1


def test_cup_displacement_excursion_cannot_be_hidden_by_returning_it():
    monitor, bottle, cup = _start()
    _observe_valid(monitor, cup)
    moved = _pose((0.006, 0, 0))
    monitor.observe(bottle, moved, in_pour_segment=False)
    result = monitor.validate_final(bottle, cup)
    assert not result.accepted
    assert _statuses(result)["pour.cup_stable"] == "failed"
    assert monitor.to_metadata()["max_cup_displacement_m"] == pytest.approx(0.006)


def test_cup_tilt_excursion_cannot_be_hidden_by_returning_it():
    monitor, bottle, cup = _start()
    _observe_valid(monitor, cup)
    monitor.observe(bottle, _pose(angle=math.radians(20)), in_pour_segment=False)
    assert _statuses(monitor.validate_final(bottle, cup))["pour.cup_stable"] == "failed"


def test_final_state_requires_return_upright_and_target_stability():
    monitor, bottle, cup = _start()
    _observe_valid(monitor, cup)
    displaced = bottle.copy()
    displaced[0, 3] += 0.03
    assert (
        _statuses(monitor.validate_final(displaced, cup))["pour.bottle_returned"]
        == "failed"
    )
    tilted = _pose(bottle[:3, 3], angle=math.radians(11))
    assert (
        _statuses(monitor.validate_final(tilted, cup))["pour.bottle_upright"]
        == "failed"
    )
    assert (
        _statuses(monitor.validate_final(bottle, _pose((0.006, 0, 0))))[
            "pour.cup_stable"
        ]
        == "failed"
    )


def test_mouth_alignment_uses_full_asset_and_cup_transforms():
    monitor = _monitor()
    bottle = _pose((0.2, 0.7, 0.2))
    cup = _pose((0.8, 0.5, 0.1), yaw=math.pi / 2)
    monitor.reset(bottle, cup)
    _observe_valid(monitor, cup)
    assert monitor.validate_final(bottle, cup).accepted


def test_reset_owns_poses_clears_prior_evidence_and_exports_detached_metadata():
    monitor, bottle, cup = _start()
    _observe_valid(monitor, cup)
    original = bottle.copy()
    bottle[0, 3] = 99
    assert monitor.validate_final(original, cup).accepted
    metadata = monitor.to_metadata()
    metadata["initial_bottle_pose"][0][3] = 42
    assert monitor.to_metadata()["initial_bottle_pose"][0][3] == pytest.approx(-0.3)
    monitor.reset(original, cup)
    assert not monitor.validate_final(original, cup).accepted
    assert monitor.to_metadata()["qualifying_sample_count"] == 0


@pytest.mark.parametrize(
    "invalid",
    [
        np.diag([2.0, 1.0, 1.0, 1.0]),
        np.diag([-1.0, 1.0, 1.0, 1.0]),
        np.full((4, 4), np.nan),
        np.eye(3),
    ],
)
def test_malformed_pose_poisoning_cannot_be_caught_and_ignored(invalid):
    monitor, bottle, cup = _start()
    _observe_valid(monitor, cup)
    with pytest.raises(ValueError):
        monitor.observe(invalid, cup, in_pour_segment=True)
    assert (
        _statuses(monitor.validate_final(bottle, cup))["pour.samples_valid"] == "failed"
    )


def test_invalid_reset_removes_previous_qualifying_trace():
    monitor, bottle, cup = _start()
    _observe_valid(monitor, cup)
    with pytest.raises(ValueError):
        monitor.reset(np.full((4, 4), np.nan), cup)
    assert not monitor.validate_final(bottle, cup).accepted


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cup_opening_radius_m": 0},
        {"bottle_mouth_radius_m": 0},
        {"bottle_mouth_radius_m": 0.1},
        {"min_mouth_height_m": 0},
        {"max_mouth_height_m": 0.01},
        {"min_tilt_rad": float("nan")},
        {"upright_tolerance_rad": math.pi},
        {"min_pour_samples": True},
        {"min_pour_samples": 1},
        {"bottle_mouth_xyz": (0, 0, float("inf"))},
    ],
)
def test_invalid_geometry_and_thresholds_rejected(kwargs):
    with pytest.raises(ValueError):
        _monitor(**kwargs)


def test_invalid_segment_marker_and_final_pose_poison_valid_trace():
    monitor, bottle, cup = _start()
    _observe_valid(monitor, cup)
    with pytest.raises(TypeError, match="in_pour_segment"):
        monitor.observe(bottle, cup, in_pour_segment="pour")
    assert not monitor.validate_final(bottle, cup).accepted
    monitor.reset(bottle, cup)
    _observe_valid(monitor, cup)
    with pytest.raises(ValueError):
        monitor.validate_final(np.full((4, 4), np.inf), cup)
    assert not monitor.validate_final(bottle, cup).accepted
