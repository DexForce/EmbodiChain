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

"""Numerical and device contracts for the in-tree TOPPRA implementations."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from embodichain.compute.trajectory._toppra import _NumpyToppra
from embodichain.compute.trajectory._toppra_warp import _retime_toppra_warp


@pytest.mark.parametrize("count", [2, 3, 4, 7, 20])
def test_warp_matches_numpy_not_a_knot_geometry(count: int) -> None:
    points = np.random.default_rng(count).normal(size=(3, count, 3))
    vlim = np.array([[-0.4, 1.0], [-1.2, 0.7], [-0.8, 1.4]])
    alim = np.array([[-1.2, 2.0], [-2.1, 0.8], [-0.6, 1.1]])
    result = _retime_toppra_warp(torch.tensor(points), vlim, alim, sample_count=128)
    assert result["success"].all()
    for row in range(len(points)):
        reference = _NumpyToppra(points[row], vlim, alim)
        duration = result["dt"][row].sum().item()
        assert duration == pytest.approx(reference.duration, rel=1e-10)
        expected = reference.sample(np.linspace(0, reference.duration, 128))
        for key, array in zip(("positions", "velocities", "accelerations"), expected):
            np.testing.assert_allclose(result[key][row], array, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("backend", ["numpy", "warp"])
def test_straight_path_matches_analytic_trapezoid(backend: str) -> None:
    # Distance 2, vmax 1, amax 2 => 0.5 + 1.5 + 0.5 = 2.5 seconds.
    points = np.array([[0.0], [2.0]])
    if backend == "numpy":
        trajectory = _NumpyToppra(points, 1.0, 2.0, grid_size=401)
        duration = trajectory.duration
    else:
        result = _retime_toppra_warp(
            torch.tensor(points[None]), 1.0, 2.0, grid_size=401, sample_count=128
        )
        assert result["success"].all()
        duration = result["dt"].sum().item()
    assert duration == pytest.approx(2.5, rel=1e-5)


@pytest.mark.parametrize("seed", [3, 12, 421])
def test_continuous_asymmetric_limits_and_interval_dynamics(seed: int) -> None:
    points = np.random.default_rng(seed).normal(size=(7, 3))
    vlim = np.array([[-0.4, 1.0], [-1.2, 0.7], [-0.8, 1.4]])
    alim = np.array([[-1.2, 2.0], [-2.1, 0.8], [-0.6, 1.1]])
    trajectory = _NumpyToppra(points, vlim, alim)
    # Probe inside every timing interval, rather than only a uniform clock.
    times = (
        trajectory.times[:-1, None]
        + np.diff(trajectory.times)[:, None] * np.linspace(0.001, 0.999, 31)
    ).ravel()
    _, velocity, acceleration = trajectory.sample(times)
    assert np.all(velocity >= vlim[:, 0] - 1e-9)
    assert np.all(velocity <= vlim[:, 1] + 1e-9)
    assert np.all(acceleration >= alim[:, 0] - 1e-9)
    assert np.all(acceleration <= alim[:, 1] + 1e-9)
    np.testing.assert_allclose(
        np.diff(trajectory.speeds**2),
        2 * np.diff(trajectory.grid) * trajectory.accelerations,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        0.5
        * (trajectory.speeds[:-1] + trajectory.speeds[1:])
        * np.diff(trajectory.times),
        np.diff(trajectory.grid),
        atol=1e-12,
    )


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_mixed_batch_duplicates_stationary_failure_and_time_padding(
    device: str,
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    points = torch.tensor(
        [
            [[0.0, 2.0], [0.0, 2.0], [1.0, 2.0], [1.0, 2.0]],
            [[3.0, 4.0], [3.0, 4.0], [3.0, 4.0], [3.0, 4.0]],
            [[0.0, 0.0], [float("nan"), 0.0], [1.0, 0.0], [1.0, 0.0]],
            [[0.0, 2.0], [0.0, 2.0], [0.1, 2.0], [0.1, 2.0]],
        ],
        dtype=torch.float64,
        device=device,
    )
    result = _retime_toppra_warp(points, 1.0, 2.0, sample_dt=0.03)
    assert result["success"].tolist() == [True, True, False, True]
    assert torch.equal(
        result["positions"][1], points[1, :1].expand_as(result["positions"][1])
    )
    assert result["dt"][1:3].count_nonzero() == 0
    assert result["positions"][2].count_nonzero() == 0
    for row in (0, 3):
        valid = 1 + int((result["dt"][row] > 0).sum())
        assert result["dt"][row].max() <= 0.03
        torch.testing.assert_close(result["positions"][row, 0], points[row, 0])
        torch.testing.assert_close(
            result["positions"][row, valid - 1 :],
            points[row, -1:].expand_as(result["positions"][row, valid - 1 :]),
        )
        assert result["velocities"][row, [0, valid - 1]].count_nonzero() == 0
        assert result["velocities"][row, valid:].count_nonzero() == 0
        assert result["accelerations"][row, valid:].count_nonzero() == 0
        reference = _NumpyToppra(points[row].cpu().numpy(), 1.0, 2.0)
        assert result["dt"][row].sum().item() == pytest.approx(reference.duration)


def test_one_sided_and_locked_joint_limits_isolate_infeasible_rows() -> None:
    points = torch.tensor(
        [[[0.0, 2.0], [1.0, 2.0]], [[0.0, 2.0], [-1.0, 2.0]], [[0.0, 2.0], [1.0, 3.0]]],
        dtype=torch.float64,
    )
    result = _retime_toppra_warp(
        points, [[0.0, 1.0], [0.0, 0.0]], [[-2.0, 1.0], [0.0, 0.0]], sample_count=128
    )
    assert result["success"].tolist() == [True, False, False]
    assert result["velocities"][0, :, 0].min() >= 0
    assert result["velocities"][0, :, 0].max() <= 1 + 1e-9
    assert result["accelerations"][0, :, 0].min() >= -2 - 1e-9
    assert result["accelerations"][0, :, 0].max() <= 1 + 1e-9


@pytest.mark.parametrize("scale", [1e-4, 1.0, 1e4])
def test_time_scaling_and_tiny_real_moves(scale: float) -> None:
    points = np.array([[0.0], [1e-7]])
    result = _retime_toppra_warp(
        torch.tensor(points[None]), scale, scale**2, sample_count=128
    )
    assert result["success"].all()
    assert result["dt"].sum() > 0
    assert result["positions"][0, -1, 0] == points[-1, 0]
    reference = _NumpyToppra(points, 1.0, 1.0)
    assert result["dt"].sum().item() * scale == pytest.approx(
        reference.duration, rel=1e-9
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_count": 1},
        {"sample_count": 2.5},
        {"sample_dt": 0.0},
        {"sample_dt": float("nan")},
        {"sample_count": 10, "grid_size": 2},
    ],
)
def test_invalid_sampling_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        _retime_toppra_warp(torch.zeros(1, 2, 1), 1.0, 2.0, **kwargs)


@pytest.mark.gpu
def test_cuda_matches_reference_on_nondefault_stream_without_host_waypoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    points = np.random.default_rng(47).normal(size=(8, 6, 6))
    expected = [_NumpyToppra(row, 1.0, 2.0) for row in points]
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        tensor = torch.tensor(points, device="cuda")
        with monkeypatch.context() as patch:

            def forbidden(*args, **kwargs):
                raise AssertionError("CUDA retiming must not copy waypoints to CPU")

            patch.setattr(torch.Tensor, "cpu", forbidden)
            result = _retime_toppra_warp(tensor, 1.0, 2.0, sample_count=128)
        # Queue consumption on the same stream before any explicit synchronize.
        duration = result["dt"].sum(dim=1)
    stream.synchronize()
    assert result["success"].all()
    assert all(value.is_cuda for value in result.values())
    for row, reference in enumerate(expected):
        assert duration[row].item() == pytest.approx(reference.duration, rel=1e-10)
        values = reference.sample(np.linspace(0, reference.duration, 128))
        for key, value in zip(("positions", "velocities", "accelerations"), values):
            np.testing.assert_allclose(
                result[key][row].cpu(), value, atol=1e-8, rtol=1e-8
            )


@pytest.mark.parametrize(
    ("points", "kept"),
    [
        ([[0.0], [2e-7], [0.5]], [0, 2]),
        ([[0.3], [0.3], [0.5], [0.5]], [0, 2, 3]),
        ([[0.0], [6e-7], [1.2e-6], [0.5]], [0, 2, 3]),
    ],
)
def test_legacy_waypoint_cleanup_preserves_spline_geometry(
    points: list[list[float]], kept: list[int]
) -> None:
    from scipy.interpolate import CubicSpline

    waypoints = np.array(points)
    expected_points = waypoints[kept]
    expected_path = CubicSpline(np.linspace(0.0, 1.0, len(kept)), expected_points)
    reference = _NumpyToppra(waypoints, 1.0, 2.0)
    phase = np.linspace(0.0, 1.0, 1001)
    np.testing.assert_allclose(reference.path(phase), expected_path(phase), atol=1e-12)
    result = _retime_toppra_warp(
        torch.tensor(waypoints[None]), 1.0, 2.0, sample_count=128
    )
    assert result["success"].all()
    assert result["dt"].sum().item() == pytest.approx(reference.duration, rel=1e-10)
    expected = reference.sample(np.linspace(0.0, reference.duration, 128))
    for key, values in zip(("positions", "velocities", "accelerations"), expected):
        np.testing.assert_allclose(result[key][0], values, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_float32_cleanup_matches_original_threshold_arithmetic(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    # float32(1e-6) is just below the double-precision threshold. Filtering must
    # happen with the input arithmetic, before promotion for the spline solve.
    points = np.array([[0.0], [1e-6], [0.5]], dtype=np.float32)
    reference = _NumpyToppra(points, 1.0, 2.0)
    # The original NumPy worker follows the installed NumPy scalar-promotion
    # rules, which differ between 1.x and 2.x at this exact boundary.
    expected_points = points if np.float32(1e-6) >= 1e-6 else points[[0, 2]]
    np.testing.assert_array_equal(reference.points, expected_points)
    result = _retime_toppra_warp(
        torch.from_numpy(points[None]).to(device), 1.0, 2.0, sample_count=128
    )
    assert result["success"].all()
    expected = reference.sample(np.linspace(0.0, reference.duration, 128))
    for key, value in zip(("positions", "velocities", "accelerations"), expected):
        np.testing.assert_allclose(result[key][0].cpu(), value, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("count", [2, 3, 4, 7])
def test_waypoint_autograd_matches_finite_differences(count: int) -> None:
    # Exercise linear, parabolic and general cubic paths, all output Jacobians,
    # asymmetric bounds, and both velocity and acceleration active constraints.
    generator = torch.Generator().manual_seed(123 + count)
    points = torch.randn(1, count, 2, generator=generator, dtype=torch.float64)
    points.requires_grad_()

    def evaluate(q: torch.Tensor) -> tuple[torch.Tensor, ...]:
        result = _retime_toppra_warp(
            q,
            [[-0.7, 1.1], [-0.9, 0.8]],
            [[-1.3, 2.0], [-1.6, 1.1]],
            sample_count=9,
            grid_size=25,
        )
        assert result["success"].all()
        return tuple(
            result[key] for key in ("positions", "velocities", "accelerations", "dt")
        )

    tracked = evaluate(points)
    with torch.no_grad():
        untracked = evaluate(points)
    for actual, expected in zip(tracked, untracked):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual.requires_grad and not expected.requires_grad
    assert torch.autograd.gradcheck(
        evaluate, (points,), eps=1e-6, atol=2e-4, rtol=2e-3, fast_mode=True
    )


def test_duration_gradient_matches_acceleration_limited_scaling() -> None:
    distance = 0.73
    points = torch.tensor(
        [[[0.0], [distance]]], dtype=torch.float64, requires_grad=True
    )
    result = _retime_toppra_warp(points, 100.0, [[-1.7, 2.3]], sample_count=17)
    duration = result["dt"].sum()
    # With inactive velocity bounds, T scales as sqrt(distance), including
    # discretization. This probes the full solver rather than interpolation.
    gradient = torch.autograd.grad(duration, points)[0]
    derivative = duration.detach() / (2 * distance)
    torch.testing.assert_close(
        gradient[0, :, 0], torch.stack((-derivative, derivative))
    )


def test_autograd_cleanup_holds_and_failures_keep_gradients_isolated() -> None:
    points = torch.tensor(
        [
            [[0.0, 0.0], [0.0, 0.0], [0.4, 0.2], [1.0, -0.1]],
            [[2.0, 3.0], [2.0, 3.0], [2.0, 3.0], [2.0, 3.0]],
            [[0.0, 0.0], [float("nan"), 0.0], [0.4, 0.2], [1.0, 0.0]],
            # A finite but infeasible path with one-sided velocity bounds.
            [[0.0, 0.0], [-0.3, 0.0], [-0.6, 0.0], [-1.0, 0.0]],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    result = _retime_toppra_warp(points, [[0.0, 1.0], [-1.0, 1.0]], 2.0, sample_count=9)
    assert result["success"].tolist() == [True, True, False, False]
    sum(
        result[key].sum() for key in ("positions", "velocities", "accelerations", "dt")
    ).backward()
    assert torch.isfinite(points.grad).all()
    assert points.grad[0, 1].count_nonzero() == 0  # Removed duplicate.
    assert points.grad[2:].count_nonzero() == 0
    torch.testing.assert_close(
        points.grad[1, 0], torch.full((2,), 9.0, dtype=points.dtype)
    )
    assert points.grad[1, 1:].count_nonzero() == 0


def test_autograd_repeated_backward_and_independent_forwards() -> None:
    points = torch.tensor(
        [[[0.1], [0.7], [1.3]]], dtype=torch.float64, requires_grad=True
    )
    first = _retime_toppra_warp(points, 1.0, 2.0, sample_count=9)
    second = _retime_toppra_warp(points, 1.0, 2.0, sample_count=9)
    seed = torch.linspace(0.2, 1.0, 9, dtype=points.dtype).view(1, 9, 1)
    before = seed.clone()
    one = torch.autograd.grad(first["positions"], points, seed, retain_graph=True)[0]
    two = torch.autograd.grad(first["positions"], points, seed, retain_graph=True)[0]
    both = torch.autograd.grad(
        (first["positions"], second["positions"]), points, (seed, seed)
    )[0]
    torch.testing.assert_close(seed, before)
    torch.testing.assert_close(one, two)
    torch.testing.assert_close(both, 2 * one)


@pytest.mark.parametrize("sampling", [{"sample_count": 9}, {"sample_dt": 0.23}])
@pytest.mark.parametrize(
    ("velocity", "acceleration"),
    [
        (0.9, 1.7),
        ([0.9, 1.1], [1.7, 1.3]),
        ([[-0.7, 1.1], [-0.9, 0.8]], [[-1.3, 2.0], [-1.6, 1.1]]),
    ],
)
def test_waypoint_and_limit_gradients_match_both_sampling_modes(
    sampling: dict,
    velocity: object,
    acceleration: object,
) -> None:
    generator = torch.Generator().manual_seed(127)
    points = torch.randn(
        1, 4, 2, generator=generator, dtype=torch.float64, requires_grad=True
    )
    vlim = torch.tensor(velocity, dtype=torch.float64, requires_grad=True)
    alim = torch.tensor(acceleration, dtype=torch.float64, requires_grad=True)

    def evaluate(
        q: torch.Tensor, v: torch.Tensor, a: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        result = _retime_toppra_warp(q, v, a, grid_size=25, **sampling)
        assert result["success"].all()
        return tuple(
            result[key] for key in ("positions", "velocities", "accelerations", "dt")
        )

    tracked = evaluate(points, vlim, alim)
    with torch.no_grad():
        untracked = evaluate(points, vlim, alim)
    for actual, expected in zip(tracked, untracked):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual.requires_grad and not expected.requires_grad
    assert torch.autograd.gradcheck(
        evaluate,
        (points, vlim, alim),
        eps=1e-6,
        atol=2e-4,
        rtol=2e-3,
        fast_mode=True,
    )


def test_time_sampling_gradient_preserves_padding_and_shared_limits() -> None:
    points = torch.tensor(
        [[[0.0], [0.23]], [[0.0], [0.73]], [[2.0], [2.0]], [[0.0], [float("nan")]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    acceleration = torch.tensor(1.7, dtype=torch.float64, requires_grad=True)
    result = _retime_toppra_warp(points, 100.0, acceleration, sample_dt=0.17)
    assert result["success"].tolist() == [True, True, True, False]
    durations = result["dt"].sum(dim=1)
    dq, da = torch.autograd.grad(
        durations.sum(), (points, acceleration), retain_graph=True
    )
    # Acceleration-limited duration scales as sqrt(distance / acceleration).
    expected_q = durations[:2].detach() / (2 * points[:2, -1, 0].detach())
    torch.testing.assert_close(dq[:2, -1, 0], expected_q)
    torch.testing.assert_close(dq[:2, 0, 0], -expected_q)
    torch.testing.assert_close(
        da, -durations.sum().detach() / (2 * acceleration.detach())
    )
    assert dq[2:].count_nonzero() == 0
    assert torch.isfinite(dq).all()
    short_length = 1 + int((result["dt"][0] > 0).sum())
    assert short_length < result["positions"].shape[1]
    padding = result["positions"][0, short_length:].sum()
    padding_grad, limit_grad = torch.autograd.grad(padding, (points, acceleration))
    expected = torch.zeros_like(points)
    expected[0, -1, 0] = result["positions"].shape[1] - short_length
    torch.testing.assert_close(padding_grad, expected)
    assert limit_grad == 0


@pytest.mark.gpu
@pytest.mark.parametrize("sampling", [{"sample_count": 9}, {"sample_dt": 0.23}])
def test_cuda_autograd_matches_cpu_and_finite_differences_on_current_stream(
    monkeypatch: pytest.MonkeyPatch,
    sampling: dict,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    generator = torch.Generator().manual_seed(130)
    inputs = (
        torch.randn(1, 7, 2, generator=generator, dtype=torch.float64),
        torch.tensor([[-0.7, 1.1], [-0.9, 0.8]], dtype=torch.float64),
        torch.tensor([[-1.3, 2.0], [-1.6, 1.1]], dtype=torch.float64),
    )
    directions = tuple(
        torch.randn(value.shape, generator=generator, dtype=value.dtype)
        for value in inputs
    )

    def loss(q: torch.Tensor, v: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        out = _retime_toppra_warp(q, v, a, grid_size=25, **sampling)
        return sum(
            out[key].square().sum()
            for key in ("positions", "velocities", "accelerations", "dt")
        )

    cpu = tuple(value.clone().requires_grad_() for value in inputs)
    expected = torch.autograd.grad(loss(*cpu), cpu)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        gpu = tuple(value.to("cuda").requires_grad_() for value in inputs)
        perturbations = tuple(value.to("cuda") for value in directions)
        with monkeypatch.context() as patch:

            def forbidden(*args, **kwargs):
                raise AssertionError(
                    "Autograd must not copy waypoints or motion limits to CPU"
                )

            patch.setattr(torch.Tensor, "cpu", forbidden)
            gradients = torch.autograd.grad(loss(*gpu), gpu)
            eps = 1e-6
            with torch.no_grad():
                plus = tuple(
                    value + eps * direction
                    for value, direction in zip(gpu, perturbations)
                )
                minus = tuple(
                    value - eps * direction
                    for value, direction in zip(gpu, perturbations)
                )
                numerical = (loss(*plus) - loss(*minus)) / (2 * eps)
            analytical = sum(
                (grad * direction).sum()
                for grad, direction in zip(gradients, perturbations)
            )
    stream.synchronize()
    for actual, reference in zip(gradients, expected):
        torch.testing.assert_close(actual.cpu(), reference, atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(analytical, numerical, atol=2e-4, rtol=2e-3)


def test_autograd_batches_noncontiguous_inputs_without_cross_row_gradients() -> None:
    generator = torch.Generator().manual_seed(901)
    base = torch.randn(
        2, 3, 5, generator=generator, dtype=torch.float64, requires_grad=True
    )
    points = base.transpose(1, 2)
    assert not points.is_contiguous()
    batched = _retime_toppra_warp(points, 1.0, 2.0, sample_count=11)
    batched_gradient = torch.autograd.grad(batched["dt"][0].sum(), base)[0]
    separate = points[0:1].detach().requires_grad_()
    single = _retime_toppra_warp(separate, 1.0, 2.0, sample_count=11)
    expected = torch.autograd.grad(single["dt"].sum(), separate)[0]
    torch.testing.assert_close(batched_gradient[0], expected[0].T)
    assert batched_gradient[1].count_nonzero() == 0
