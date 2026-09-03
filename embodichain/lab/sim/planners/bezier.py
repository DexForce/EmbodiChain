# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ----------------------------------------------------------------------------
"""Differentiable quadratic and quintic Bézier path utilities.

The functions in this module describe geometric paths in an arbitrary Euclidean
vector space (for example ``D=DOF`` joint vectors or ``D=3`` Cartesian
positions). They deliberately do not assign time to a path; callers can
compose the returned arc-length parameter with a trapezoidal or Double-S time
law. Rotations require a manifold-aware interpolation such as SLERP and must
not be interpolated as raw matrix or quaternion coordinates here.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "BezierPath",
    "bezier_derivative",
    "bezier_evaluate",
    "sample_bezier_path",
]


class BezierPath:
    """Euclidean quadratic or quintic Bézier path value object."""

    def __init__(self, control_points: torch.Tensor) -> None:
        _validate_control_points(control_points)
        self._control_points = control_points

    @property
    def control_points(self) -> torch.Tensor:
        """Return the control points without copying the underlying tensor."""
        return self._control_points

    @property
    def degree(self) -> int:
        """Return the polynomial degree (2 or 5)."""
        return int(self._control_points.shape[-2] - 1)

    @property
    def dimension(self) -> int:
        """Return the Euclidean dimension of each path point."""
        return int(self._control_points.shape[-1])

    @property
    def length(self) -> torch.Tensor:
        """Return a dense-sampling approximation of geometric path length."""
        _, cumulative = self.sample(1025, arc_length=False)
        return cumulative[..., -1]

    def evaluate(self, parameter: torch.Tensor) -> torch.Tensor:
        """Evaluate the path at normalized parameter values."""
        return bezier_evaluate(self._control_points, parameter)

    def derivative(self, parameter: torch.Tensor, order: int = 1) -> torch.Tensor:
        """Evaluate a derivative with respect to normalized path parameter."""
        return bezier_derivative(self._control_points, parameter, order=order)

    def tangent(self, parameter: torch.Tensor) -> torch.Tensor:
        """Return the first geometric derivative, matching ``PathBase`` semantics."""
        return self.derivative(parameter, order=1)

    def curvature(self, parameter: torch.Tensor) -> torch.Tensor:
        """Return the second geometric derivative, matching ``PathBase`` semantics."""
        return self.derivative(parameter, order=2)

    def arc_tangent(self, parameter: torch.Tensor) -> torch.Tensor:
        """Return the unit tangent with respect to geometric arc length."""
        first = self.tangent(parameter)
        speed = torch.linalg.vector_norm(first, dim=-1, keepdim=True)
        if bool((speed <= torch.finfo(first.dtype).eps).any().item()):
            raise ValueError("Arc-length tangent is undefined at a stationary point.")
        return first / speed

    def arc_curvature(self, parameter: torch.Tensor) -> torch.Tensor:
        """Return the second derivative with respect to geometric arc length."""
        first = self.tangent(parameter)
        second = self.curvature(parameter)
        speed = torch.linalg.vector_norm(first, dim=-1, keepdim=True)
        if bool((speed <= torch.finfo(first.dtype).eps).any().item()):
            raise ValueError("Arc-length curvature is undefined at a stationary point.")
        speed_derivative = (first * second).sum(dim=-1, keepdim=True) / speed
        return second / speed.square() - first * speed_derivative / speed.pow(3)

    def sample(
        self, sample_count: int, *, arc_length: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample points and their cumulative geometric lengths."""
        return sample_bezier_path(
            self._control_points, sample_count, arc_length=arc_length
        )


def _validate_control_points(
    control_points: torch.Tensor, *, allow_any_degree: bool = False
) -> int:
    if not isinstance(control_points, torch.Tensor):
        raise TypeError("control_points must be a torch.Tensor.")
    if control_points.ndim < 2 or (
        not allow_any_degree and control_points.shape[-2] not in (3, 6)
    ):
        raise ValueError("control_points must have shape (..., 3|6, D).")
    if not control_points.is_floating_point():
        raise TypeError("control_points must use a floating-point dtype.")
    if not bool(torch.isfinite(control_points).all().item()):
        raise ValueError("control_points must contain only finite values.")
    return int(control_points.shape[-2] - 1)


def bezier_evaluate(
    control_points: torch.Tensor, parameter: torch.Tensor
) -> torch.Tensor:
    """Evaluate quadratic or quintic Bézier control points at ``parameter``.

    ``control_points`` has shape ``(..., degree + 1, D)`` and ``parameter``
    broadcasts against its leading dimensions. Values are clamped to ``[0, 1]``.
    """
    degree = _validate_control_points(control_points)
    parameter = torch.as_tensor(
        parameter, dtype=control_points.dtype, device=control_points.device
    )
    t = parameter.clamp(0.0, 1.0)
    if not bool(torch.isfinite(t).all().item()):
        raise ValueError("parameter must contain only finite values.")
    # A batched path commonly receives ``(B, N)`` parameters while control
    # points are ``(B, K, D)``. Insert the sample axis so each batch is paired
    # with its own parameter row instead of relying on ambiguous broadcasting.
    leading_rank = control_points.ndim - 2
    if t.ndim == leading_rank + 1 and leading_rank > 0:
        if t.shape[:leading_rank] == control_points.shape[:-2]:
            control_points = control_points.unsqueeze(-3)
    result = torch.zeros(
        torch.broadcast_shapes(t.shape, control_points.shape[:-2])
        + (control_points.shape[-1],),
        dtype=control_points.dtype,
        device=control_points.device,
    )
    for index in range(degree + 1):
        coefficient = (
            math.comb(degree, index) * (1.0 - t) ** (degree - index) * t**index
        )
        result = result + coefficient[..., None] * control_points[..., index, :]
    return result


def bezier_derivative(
    control_points: torch.Tensor, parameter: torch.Tensor, order: int = 1
) -> torch.Tensor:
    """Evaluate a Bézier derivative with respect to its normalized parameter."""
    degree = _validate_control_points(control_points)
    if (
        isinstance(order, bool)
        or not isinstance(order, int)
        or order < 0
        or order > degree
    ):
        raise ValueError(f"order must be an integer in [0, {degree}].")
    if order == 0:
        return bezier_evaluate(control_points, parameter)
    differentiated = control_points
    for current_degree in range(degree, degree - order, -1):
        differentiated = current_degree * (
            differentiated[..., 1:, :] - differentiated[..., :-1, :]
        )
    _validate_control_points(differentiated, allow_any_degree=True)
    return _evaluate_control_points(differentiated, parameter)


def _evaluate_control_points(
    control_points: torch.Tensor, parameter: torch.Tensor
) -> torch.Tensor:
    """Evaluate arbitrary-degree control points for internal derivatives."""
    degree = _validate_control_points(control_points, allow_any_degree=True)
    parameter = torch.as_tensor(
        parameter, dtype=control_points.dtype, device=control_points.device
    )
    t = parameter.clamp(0.0, 1.0)
    if not bool(torch.isfinite(t).all().item()):
        raise ValueError("parameter must contain only finite values.")
    leading_rank = control_points.ndim - 2
    if t.ndim == leading_rank + 1 and leading_rank > 0:
        if t.shape[:leading_rank] == control_points.shape[:-2]:
            control_points = control_points.unsqueeze(-3)
    leading = torch.broadcast_shapes(t.shape, control_points.shape[:-2])
    result = torch.zeros(
        leading + (control_points.shape[-1],),
        dtype=control_points.dtype,
        device=control_points.device,
    )
    for index in range(degree + 1):
        coefficient = (
            math.comb(degree, index) * (1.0 - t) ** (degree - index) * t**index
        )
        result = result + coefficient[..., None] * control_points[..., index, :]
    return result


def sample_bezier_path(
    control_points: torch.Tensor, sample_count: int, *, arc_length: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a Bézier path and return ``(points, cumulative_length)``.

    Arc-length sampling uses a dense monotonic lookup table and linear inverse
    interpolation. It is deterministic, differentiable with respect to the
    control points away from lookup-bin boundaries, and suitable as input to a
    separate scalar time-parameterization stage.
    """
    _validate_control_points(control_points)
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count < 2
    ):
        raise ValueError("sample_count must be an integer of at least 2.")
    table_count = max(1025, sample_count * 32)
    table_t = torch.linspace(
        0.0, 1.0, table_count, dtype=control_points.dtype, device=control_points.device
    )
    table_points = bezier_evaluate(control_points.unsqueeze(-3), table_t)
    distances = torch.linalg.vector_norm(
        table_points[..., 1:, :] - table_points[..., :-1, :], dim=-1
    )
    cumulative = torch.cat(
        (torch.zeros_like(distances[..., :1]), distances.cumsum(dim=-1)), dim=-1
    )
    if not arc_length:
        parameters = torch.linspace(
            0.0,
            1.0,
            sample_count,
            dtype=control_points.dtype,
            device=control_points.device,
        )
        points = bezier_evaluate(control_points.unsqueeze(-3), parameters)
        sampled_distances = torch.linalg.vector_norm(
            points[..., 1:, :] - points[..., :-1, :], dim=-1
        )
        sampled_cumulative = torch.cat(
            (
                torch.zeros_like(sampled_distances[..., :1]),
                sampled_distances.cumsum(dim=-1),
            ),
            dim=-1,
        )
        return points, sampled_cumulative
    targets = torch.linspace(
        0.0, 1.0, sample_count, dtype=control_points.dtype, device=control_points.device
    )
    targets = targets * cumulative[..., -1:]
    indices = torch.searchsorted(cumulative, targets, right=True).clamp(
        1, table_count - 1
    )
    lower = cumulative.gather(-1, indices - 1)
    upper = cumulative.gather(-1, indices)
    alpha = (
        (targets - lower)
        / (upper - lower).clamp_min(torch.finfo(control_points.dtype).eps)
    ).clamp(0.0, 1.0)
    parameters = table_t[indices - 1] + alpha * (
        table_t[indices] - table_t[indices - 1]
    )
    points = bezier_evaluate(control_points.unsqueeze(-3), parameters)
    return points, targets
