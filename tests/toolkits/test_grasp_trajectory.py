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

"""Numerical contracts for smooth grasp/assembly waypoint timing."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.tools.grasp_assemble import _trajectory
from scripts.tools.grasp_assemble._trajectory import time_parameterize

LIMITS = {
    "control_dt": 0.01,
    "velocity_limit": 0.7,
    "acceleration_limit": 0.9,
    "jerk_limit": 3.0,
}
PATH = np.array([[0, 0], [0.1, 0.4], [0.3, 0.6], [0.8, 0.2]])


def test_exact_endpoints_and_fixed_command_grid() -> None:
    result = time_parameterize(PATH, **LIMITS)
    np.testing.assert_array_equal(result["positions"][[0, -1]], PATH[[0, -1]])
    np.testing.assert_array_equal(result["velocities"][[0, -1]], np.zeros((2, 2)))
    np.testing.assert_array_equal(result["accelerations"][[0, -1]], np.zeros((2, 2)))
    assert result["dt"][0] == 0
    np.testing.assert_array_equal(result["dt"][1:], LIMITS["control_dt"])
    assert result["dt"].sum() == pytest.approx(result["duration"])
    assert result["duration"] >= result["toppra_duration"] > 0


def test_derivative_limits_include_jerk_after_duration_dilation() -> None:
    result = time_parameterize(PATH, **LIMITS)
    for values, limit, peak in [
        ("velocities", "velocity_limit", "peak_velocity"),
        ("accelerations", "acceleration_limit", "peak_acceleration"),
        ("jerks", "jerk_limit", "peak_jerk"),
    ]:
        assert np.abs(result[values]).max() <= LIMITS[limit]
        assert result["validation"][peak] <= LIMITS[limit]
    assert result["validation"]["limits_satisfied"]
    assert result["validation"]["verification_samples"] > len(result["positions"])


def test_analytical_derivatives_match_sampled_motion() -> None:
    result = time_parameterize(PATH, **LIMITS)
    # Acceleration remains continuous across the path/clock knots, but jerk
    # can jump there. A central velocity difference therefore has an error
    # bounded by jerk_limit * control_dt / 2, including intervals across knots.
    numerical_v = np.gradient(result["positions"], LIMITS["control_dt"], axis=0)
    numerical_a = np.gradient(result["velocities"], LIMITS["control_dt"], axis=0)
    np.testing.assert_allclose(numerical_v[2:-2], result["velocities"][2:-2], atol=1e-4)
    np.testing.assert_allclose(
        numerical_a[2:-2],
        result["accelerations"][2:-2],
        atol=LIMITS["jerk_limit"] * LIMITS["control_dt"] / 2,
    )


def test_smoothed_clock_is_monotone_and_has_rest_endpoints() -> None:
    times = np.array([0.0, 0.1, 0.100001, 0.3, 0.5, 0.8, 1.0])
    speeds = np.array([0.0, 0.3, 0.30001, 0.1, 0.4, 0.2, 0.0])
    clock = _trajectory._smooth_clock(times, speeds, spacing=0.05)
    sample_times = np.linspace(times[0], times[-1], 2001)
    assert np.all(np.diff(clock(sample_times)) >= -1e-14)
    np.testing.assert_allclose(clock(times[[0, -1]]), [0.0, 1.0], atol=1e-14)
    np.testing.assert_allclose(clock(times[[0, -1]], 1), 0.0, atol=1e-14)
    np.testing.assert_allclose(clock(times[[0, -1]], 2), 0.0, atol=1e-14)
    for order in (0, 1, 2):
        np.testing.assert_allclose(
            clock(clock.x[1:-1] - 1e-10, order),
            clock(clock.x[1:-1] + 1e-10, order),
            atol=1e-7,
        )


def test_invalid_clock_fit_does_not_discard_other_spacings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _trajectory._smooth_clock
    calls = []

    def reject_first_fit(*args):
        calls.append(args)
        if len(calls) == 1:
            raise ValueError("coarse clock fit has no forward motion")
        return original(*args)

    monkeypatch.setattr(_trajectory, "_smooth_clock", reject_first_fit)
    result = time_parameterize(PATH, **LIMITS)
    assert len(calls) > 1
    assert result["validation"]["limits_satisfied"]
    assert result["validation"]["toppra_calls"] == 1


def test_consecutive_duplicate_waypoints_do_not_change_motion() -> None:
    expected = time_parameterize(PATH, **LIMITS)
    actual = time_parameterize(np.repeat(PATH, 3, axis=0), **LIMITS)
    np.testing.assert_array_equal(actual["positions"], expected["positions"])
    assert actual["duration"] == expected["duration"]
    np.testing.assert_array_equal(
        actual["waypoint_indices"], np.repeat(expected["waypoint_indices"], 3)
    )
    np.testing.assert_array_equal(
        actual["waypoint_times"], np.repeat(expected["waypoint_times"], 3)
    )


def test_waypoint_passages_map_to_nearest_control_samples() -> None:
    result = time_parameterize(PATH, **LIMITS)
    indices = result["waypoint_indices"]
    assert indices.shape == (len(PATH),)
    assert indices[0] == 0
    assert indices[-1] == len(result["positions"]) - 1
    assert np.all(np.diff(indices) >= 0)
    np.testing.assert_allclose(
        indices * LIMITS["control_dt"],
        result["waypoint_times"],
        atol=LIMITS["control_dt"] / 2 + 1e-12,
    )
    # The fixed command grid need not hit a waypoint exactly. Its error is
    # bounded by the velocity limit times the nearest-sample time offset.
    np.testing.assert_allclose(
        result["positions"][indices],
        PATH,
        atol=LIMITS["velocity_limit"] * LIMITS["control_dt"] / 2 + 1e-12,
    )


def test_continuous_path_calls_toppra_once_and_does_not_stop_at_waypoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    original = _trajectory._toppra_parameterization

    def record_call(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(_trajectory, "_toppra_parameterization", record_call)
    progress = np.linspace(0.0, 1.0, 5)
    path = np.column_stack((progress, progress * 0.5))
    result = time_parameterize(path, **LIMITS)
    assert len(calls) == 1
    assert len(calls[0][1]) == len(path)
    passage_velocities = result["velocities"][result["waypoint_indices"][1:-1]]
    assert np.all(np.linalg.norm(passage_velocities, axis=1) > 0.1)
    assert result["validation"]["limits_satisfied"]


@pytest.mark.parametrize("path", [np.array([[0.4, 0.2]]), np.ones((5, 2))])
def test_stationary_path_has_valid_hold_timing(path: np.ndarray) -> None:
    result = time_parameterize(path, **LIMITS)
    np.testing.assert_array_equal(result["positions"], np.repeat(path[:1], 2, axis=0))
    assert result["duration"] == LIMITS["control_dt"]
    assert result["toppra_duration"] == 0
    assert not result["velocities"].any()
    assert not result["accelerations"].any()
    assert not result["jerks"].any()
    assert result["waypoint_indices"].shape == (len(path),)
    assert result["waypoint_indices"][0] == 0
    assert np.all(np.diff(result["waypoint_indices"]) >= 0)
    if len(path) > 1:
        assert result["waypoint_indices"][-1] == len(result["positions"]) - 1
        assert result["waypoint_times"][-1] == result["duration"]


def test_additional_slowdown_preserves_limits_and_endpoints() -> None:
    fast = time_parameterize(PATH, **LIMITS)
    slow = time_parameterize(PATH, duration_scale=2.0, **LIMITS)
    assert slow["duration"] >= fast["duration"] * 2 - LIMITS["control_dt"] - 1e-12
    np.testing.assert_array_equal(slow["positions"][[0, -1]], PATH[[0, -1]])
    assert slow["validation"]["peak_jerk"] < fast["validation"]["peak_jerk"] / 7.9


@pytest.mark.parametrize(
    "path", [np.array([]), np.zeros((0, 2)), np.zeros((2, 0)), [[0], [np.nan]]]
)
def test_invalid_joint_paths_fail_before_planning(path: np.ndarray) -> None:
    with pytest.raises(ValueError, match="joint_path"):
        time_parameterize(path, **LIMITS)


@pytest.mark.parametrize(
    "setting,value",
    [
        ("control_dt", 0),
        ("control_dt", True),
        ("velocity_limit", -1),
        ("acceleration_limit", float("nan")),
        ("jerk_limit", float("inf")),
        ("jerk_limit", "3"),
        ("duration_scale", 0.5),
    ],
)
def test_invalid_timing_settings_rejected(setting: str, value: float) -> None:
    with pytest.raises(ValueError, match=setting):
        time_parameterize(PATH, **(LIMITS | {setting: value}))


def test_near_duplicate_solver_grid_points_preserve_all_geometric_waypoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import toppra

    # Chord lengths lie close to tenths, but not bit-identically on TOPPRA's
    # uniform constraint grid. Previously cumulative timestamps repeated.
    path = np.linspace(0.0, 0.6, 15)[:, None] * np.array(
        [1.0, 0.3, -0.4, 0.2, 0.5, 0.6]
    )
    path = path[2:-2]
    solver_grids, spline_inputs = [], []
    original_solver = toppra.algorithm.TOPPRA
    original_spline = toppra.SplineInterpolator

    def solver(*args, **kwargs):
        solver_grids.append(kwargs["gridpoints"].copy())
        return original_solver(*args, **kwargs)

    def spline(knots, values, *args, **kwargs):
        spline_inputs.append((knots.copy(), values.copy()))
        return original_spline(knots, values, *args, **kwargs)

    monkeypatch.setattr(toppra.algorithm, "TOPPRA", solver)
    monkeypatch.setattr(toppra, "SplineInterpolator", spline)
    result = time_parameterize(
        path,
        control_dt=0.01,
        velocity_limit=0.6,
        acceleration_limit=1.2,
        jerk_limit=6.0,
    )
    assert len(solver_grids) == 1
    assert np.min(np.diff(solver_grids[0])) > 1e-12
    assert len(spline_inputs[0][0]) == len(path)
    np.testing.assert_allclose(spline_inputs[0][1], path, atol=1e-14)
    assert len(result["waypoint_times"]) == len(path)
    assert np.all(np.diff(result["waypoint_times"]) > 0)
    assert np.all(
        np.linalg.norm(result["velocities"][result["waypoint_indices"][1:-1]], axis=1)
        > 0.01
    )
    assert result["validation"]["limits_satisfied"]
    np.testing.assert_allclose(
        result["positions"][result["waypoint_indices"]], path, atol=0.6 * 0.01 / 2
    )
