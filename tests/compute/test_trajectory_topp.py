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

"""Contracts for differentiable time-optimal path parameterization."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from embodichain.compute import trajectory
from embodichain.compute.trajectory.topp import _evaluate_spline, _spline_moments

FR3_VELOCITY = torch.tensor(
    [2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26], dtype=torch.float64
)
FR3_ACCELERATION = 10.0


def _random_path(seed: int, count: int = 10, joints: int = 7) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    steps = torch.randn(count, joints, generator=generator, dtype=torch.float64) * 0.12
    return torch.cumsum(steps, dim=0)


def _grid_derivatives(waypoints: np.ndarray, gridpoints: int):
    import toppra as ta

    path = ta.SplineInterpolator(np.linspace(0.0, 1.0, len(waypoints)), waypoints)
    grid = np.linspace(0.0, 1.0, gridpoints)
    return (
        path,
        grid,
        torch.tensor(path(grid, 1))[None],
        torch.tensor(path(grid, 2))[None],
    )


@pytest.mark.parametrize("count", [2, 3, 4, 7])
def test_spline_matches_scipy_not_a_knot(count: int) -> None:
    from scipy.interpolate import CubicSpline

    knots = _random_path(count, count).numpy()
    s = np.linspace(0.0, 1.0, 301)
    reference = CubicSpline(np.linspace(0.0, 1.0, count), knots, bc_type="not-a-knot")
    values = torch.tensor(knots)[None]
    q, dq, ddq = _evaluate_spline(
        values, _spline_moments(values), torch.tensor(s)[None]
    )
    for derivative, ours in enumerate((q, dq, ddq)):
        np.testing.assert_allclose(
            ours[0].numpy(), reference(s, derivative), atol=1e-10
        )


@pytest.mark.parametrize("scheme", ["interpolation", "collocation"])
@pytest.mark.parametrize("held", [0, 4])
def test_parameterization_matches_toppra(scheme: str, held: int) -> None:
    # A held tail puts a near-stationary stretch into the path, which is where
    # omitting the "cannot be forced below zero speed" bound used to diverge.
    import toppra as ta
    import toppra.constraint as constraint

    waypoints = _random_path(1).numpy()
    if held:
        waypoints = np.concatenate([waypoints, np.repeat(waypoints[-1:], held, 0)])
    path, grid, dq, ddq = _grid_derivatives(waypoints, 301)
    velocity = FR3_VELOCITY.numpy()
    discretization = {
        "interpolation": constraint.DiscretizationType.Interpolation,
        "collocation": constraint.DiscretizationType.Collocation,
    }[scheme]
    solver = ta.algorithm.TOPPRA(
        [
            constraint.JointVelocityConstraint(np.stack([-velocity, velocity], 1)),
            constraint.JointAccelerationConstraint(
                np.array([[-FR3_ACCELERATION, FR3_ACCELERATION]] * 7),
                discretization_scheme=discretization,
            ),
        ],
        path,
        gridpoints=grid,
        parametrizer="ParametrizeConstAccel",
    )
    _, sdot, _ = solver.compute_parameterization(0, 0)
    reference_duration = float(np.sum(2 * np.diff(grid) / (sdot[:-1] + sdot[1:])))

    profile = trajectory.parameterize_time_optimal(
        dq,
        ddq,
        torch.tensor(grid),
        FR3_VELOCITY,
        FR3_ACCELERATION,
        discretization=scheme,
        backend="torch",
    )
    # toppra's LP tolerance leaves ~1e-7 where the exact speed is zero, which
    # the square root amplifies; away from those points the match is ~1e-7.
    assert float(profile.duration) == pytest.approx(reference_duration, rel=1e-4)
    np.testing.assert_allclose(
        profile.sdot_sq[0].numpy(), sdot**2, atol=1e-5 * float((sdot**2).max())
    )


def test_straight_line_is_bang_bang() -> None:
    # Every joint travels 0.5 rad; acceleration binds before velocity, so the
    # optimum is a triangular profile lasting 2 * sqrt(d / a).
    waypoints = torch.stack([torch.zeros(7), torch.full((7,), 0.5)])[None].double()
    result = trajectory.retime_time_optimal(
        waypoints,
        FR3_VELOCITY,
        FR3_ACCELERATION,
        sample_interval=0.005,
        backend="torch",
    )
    assert float(result.duration) == pytest.approx(
        2 * math.sqrt(0.5 / FR3_ACCELERATION), rel=1e-4
    )
    torch.testing.assert_close(result.positions[0, [0, -1]], waypoints[0])
    assert float(result.accelerations.abs().max()) == pytest.approx(
        FR3_ACCELERATION, rel=1e-3
    )


def test_resampled_output_respects_limits_on_the_default_grid() -> None:
    # Limits are exact at grid points; the default density keeps the overshoot
    # between them to about one percent.
    waypoints = torch.stack([_random_path(seed) for seed in range(4)])
    result = trajectory.retime_time_optimal(
        waypoints,
        FR3_VELOCITY,
        FR3_ACCELERATION,
        sample_interval=0.002,
        backend="torch",
    )
    assert bool(result.success.all())
    assert float((result.velocities.abs() / FR3_VELOCITY).max()) <= 1.0 + 1e-6
    assert float(result.accelerations.abs().max()) <= FR3_ACCELERATION * 1.02


@pytest.mark.parametrize("backend", ["torch", "warp"])
def test_gradients_match_finite_differences(backend: str) -> None:
    waypoints = torch.stack(
        [_random_path(seed, count=6, joints=3) for seed in range(2)]
    )

    def loss(values: torch.Tensor) -> tuple[torch.Tensor, int]:
        result = trajectory.retime_time_optimal(
            values, 2.0, 8.0, sample_interval=0.01, gridpoints=200, backend=backend
        )
        value = (
            result.duration.sum()
            + result.positions.square().mean()
            + 0.1 * result.velocities.square().mean()
            + 0.01 * result.accelerations.square().mean()
        )
        return value, result.positions.shape[1]

    leaf = waypoints.clone().requires_grad_(True)
    value, samples = loss(leaf)
    (gradient,) = torch.autograd.grad(value, leaf)
    assert bool(torch.isfinite(gradient).all())
    step = 1e-7
    with torch.no_grad():
        for row, knot, joint in ((0, 2, 0), (1, 4, 2), (0, 5, 1), (1, 1, 1)):
            offset = torch.zeros_like(waypoints)
            offset[row, knot, joint] = step
            (plus, n_plus), (minus, n_minus) = loss(waypoints + offset), loss(
                waypoints - offset
            )
            assert n_plus == n_minus == samples
            numeric = (plus - minus) / (2 * step)
            assert float(gradient[row, knot, joint]) == pytest.approx(
                float(numeric), rel=1e-4, abs=1e-8
            )


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]
)
@pytest.mark.parametrize("scheme", ["interpolation", "collocation"])
def test_warp_matches_torch_forward_and_backward(device: str, scheme: str) -> None:
    waypoints = torch.stack(
        [_random_path(seed, count=8, joints=3) for seed in range(3)]
    )
    grid = torch.linspace(0.0, 1.0, 121, dtype=torch.float64)
    _, dq, ddq = _evaluate_spline(
        waypoints, _spline_moments(waypoints), grid.expand(3, -1)
    )
    outputs = {}
    for backend in ("torch", "warp"):
        velocity = dq.to(device).clone().requires_grad_(True)
        acceleration = ddq.to(device).clone().requires_grad_(True)
        profile = trajectory.parameterize_time_optimal(
            velocity,
            acceleration,
            grid.to(device),
            2.0,
            8.0,
            discretization=scheme,
            backend=backend,
        )
        (
            profile.duration.sum()
            + 0.1 * profile.sdot_sq.sum()
            + 0.01 * profile.sddot.sum()
        ).backward()
        outputs[backend] = (profile, velocity.grad, acceleration.grad)
    (reference, ref_dq, ref_ddq), (warp, warp_dq, warp_ddq) = (
        outputs["torch"],
        outputs["warp"],
    )
    torch.testing.assert_close(warp.duration, reference.duration)
    torch.testing.assert_close(warp.sdot_sq, reference.sdot_sq)
    torch.testing.assert_close(warp_dq, ref_dq)
    torch.testing.assert_close(warp_ddq, ref_ddq)


def test_holding_at_the_goal_does_not_change_the_motion() -> None:
    # A converged environment that holds its pose inside a batched rollout
    # repeats its final sample; geometrically that is one point, not a segment.
    path = _random_path(3, count=8)
    held = torch.cat([path, path[-1:].expand(4, -1)])
    results = [
        trajectory.retime_time_optimal(
            values[None],
            FR3_VELOCITY,
            FR3_ACCELERATION,
            sample_interval=0.01,
            gridpoints=400,
            backend="torch",
        )
        for values in (path, held)
    ]
    torch.testing.assert_close(results[0].duration, results[1].duration)
    torch.testing.assert_close(results[0].positions, results[1].positions)


def test_stationary_and_short_rows_pad_with_a_held_pose() -> None:
    long_path = _random_path(4, count=8)
    short_path = long_path[:3].clone()
    short_path = torch.cat([short_path, short_path[-1:].expand(5, -1)])
    still = long_path[:1].expand(8, -1)
    result = trajectory.retime_time_optimal(
        torch.stack([long_path, short_path, still]),
        FR3_VELOCITY,
        FR3_ACCELERATION,
        sample_interval=0.01,
        gridpoints=400,
        backend="torch",
    )
    assert bool(result.success.all())
    assert float(result.duration[2]) == 0.0
    assert bool((result.dt[2] == 0).all())
    torch.testing.assert_close(
        result.positions[2], still[:1].expand_as(result.positions[2])
    )
    padded = torch.nonzero(result.dt[1] == 0).flatten()[1:]
    assert padded.numel() > 0
    assert float(result.velocities[1, padded].abs().max()) == 0.0
    assert float(result.accelerations[1, padded].abs().max()) == 0.0
    torch.testing.assert_close(result.positions[1, -1], short_path[-1])


def test_tiny_moves_get_a_real_duration() -> None:
    waypoints = torch.stack([torch.zeros(7), torch.full((7,), 1e-4)])[None].double()
    result = trajectory.retime_time_optimal(
        waypoints, FR3_VELOCITY, FR3_ACCELERATION, sample_interval=0.01, backend="torch"
    )
    assert float(result.duration) > 0.0


def test_a_move_below_the_duplicate_tolerance_still_ends_at_the_goal() -> None:
    start = torch.zeros(7, dtype=torch.float64)
    goal = torch.full((7,), 5e-7, dtype=torch.float64)
    result = trajectory.retime_time_optimal(
        torch.stack([start, goal])[None],
        FR3_VELOCITY,
        FR3_ACCELERATION,
        sample_interval=0.01,
        backend="torch",
    )
    assert float(result.duration) == 0.0
    torch.testing.assert_close(result.positions[0, 0], start)
    torch.testing.assert_close(result.positions[0, -1], goal)


@pytest.mark.parametrize("backend", ["torch", "warp"])
def test_a_tiny_path_derivative_still_bounds_the_speed(backend: str) -> None:
    # dq/ds far below the degeneracy threshold, against a velocity limit just
    # as small: dropping the bound would let the joint run far over it.
    grid = torch.linspace(0.0, 1.0, 51)
    path_velocity = torch.zeros(1, 51, 3)
    path_velocity[..., 0] = 1e-8
    profile = trajectory.parameterize_time_optimal(
        path_velocity, torch.zeros_like(path_velocity), grid, 1e-5, 1e3, backend=backend
    )
    joint_speed = profile.sdot_sq.clamp_min(0).sqrt() * 1e-8
    assert float(joint_speed.max()) <= 1e-5 * (1 + 1e-4)


@pytest.mark.parametrize("backend", ["torch", "warp"])
def test_samples_start_and_end_exactly_at_rest(backend: str) -> None:
    result = trajectory.retime_time_optimal(
        _random_path(7)[None],
        FR3_VELOCITY,
        FR3_ACCELERATION,
        sample_interval=0.01,
        gridpoints=300,
        backend=backend,
    )
    last = int((result.dt[0] > 0).sum())
    assert float(result.velocities[0, 0].abs().max()) == 0.0
    assert float(result.velocities[0, last].abs().max()) == 0.0


def test_more_joints_than_the_warp_kernels_support() -> None:
    waypoints = torch.cumsum(torch.randn(1, 5, 20, dtype=torch.float64) * 0.1, dim=1)
    result = trajectory.retime_time_optimal(
        waypoints, 2.0, 10.0, sample_interval=0.01, gridpoints=200, backend="auto"
    )
    assert bool(result.success.all())
    with pytest.raises(ValueError, match="at most 16 joints"):
        trajectory.retime_time_optimal(
            waypoints, 2.0, 10.0, sample_interval=0.01, gridpoints=200, backend="warp"
        )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(sample_interval=0.0), "sample_interval"),
        (dict(sample_interval=0.01, discretization="midpoint"), "discretization"),
        (dict(sample_interval=0.01, backend="numpy"), "backend"),
        (
            dict(sample_interval=0.01, duplicate_tolerance=float("nan")),
            "duplicate_tolerance",
        ),
        (
            dict(sample_interval=0.01, duplicate_tolerance=float("inf")),
            "duplicate_tolerance",
        ),
        (dict(sample_interval=0.01, duplicate_tolerance=-1.0), "duplicate_tolerance"),
    ],
)
def test_invalid_arguments_are_rejected(kwargs: dict, message: str) -> None:
    waypoints = _random_path(5, count=4)[None]
    with pytest.raises(ValueError, match=message):
        trajectory.retime_time_optimal(
            waypoints, FR3_VELOCITY, FR3_ACCELERATION, **kwargs
        )


def test_non_positive_limits_are_rejected() -> None:
    waypoints = _random_path(6, count=4)[None]
    with pytest.raises(ValueError, match="velocity_limits"):
        trajectory.retime_time_optimal(
            waypoints, -FR3_VELOCITY, FR3_ACCELERATION, sample_interval=0.01
        )
