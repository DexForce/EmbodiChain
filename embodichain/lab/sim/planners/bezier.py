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
    "bezier_arc_length",
    "bezier_derivative",
    "bezier_evaluate",
    "sample_bezier_path",
]


def quintic_blend_control_points(
    previous: torch.Tensor,
    corner: torch.Tensor,
    following: torch.Tensor,
    blend_tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct HolisticMotion-compatible quintic corner control points.

    Returns the six control points and the analytic path-parameter length of
    the blend. Straight portions run from ``previous`` to the first control
    point and from the last control point to ``following``.
    """
    if previous.shape != corner.shape or corner.shape != following.shape:
        raise ValueError("Corner points must have identical shapes.")
    if previous.ndim != 1 or not previous.is_floating_point():
        raise ValueError("Corner points must be one-dimensional floating tensors.")
    if not math.isfinite(blend_tolerance) or blend_tolerance <= 0.0:
        raise ValueError("blend_tolerance must be finite and greater than zero.")
    incoming = corner - previous
    outgoing = following - corner
    incoming_length = torch.linalg.vector_norm(incoming)
    outgoing_length = torch.linalg.vector_norm(outgoing)
    epsilon = torch.finfo(previous.dtype).eps
    if bool((incoming_length <= epsilon).item() or (outgoing_length <= epsilon).item()):
        raise ValueError("A blend corner requires two non-degenerate edges.")
    incoming_tangent = incoming / incoming_length
    outgoing_tangent = outgoing / outgoing_length
    tangent_difference = torch.linalg.vector_norm(outgoing_tangent - incoming_tangent)
    if bool((tangent_difference <= epsilon).item()):
        raise ValueError("A collinear same-direction corner does not require blending.")
    distance = torch.minimum(
        previous.new_tensor(4.0 * blend_tolerance) / tangent_difference,
        torch.minimum(incoming_length / 2.0, outgoing_length / 2.0),
    )
    start = corner - distance * incoming_tangent
    end = corner + distance * outgoing_tangent
    tangent_sum = incoming_tangent + outgoing_tangent
    chord = end - start
    a = 256.0 - 49.0 * tangent_sum.square().sum()
    b = 420.0 * torch.dot(chord, tangent_sum)
    c = -900.0 * chord.square().sum()
    discriminant = b.square() - 4.0 * a * c
    if bool((discriminant < 0.0).item()):
        raise ValueError("Blend corner does not admit valid quintic controls.")
    root = (-b + torch.sqrt(discriminant)) / (2.0 * a)
    if not bool(torch.isfinite(root).item()) or bool((root <= epsilon).item()):
        raise ValueError("Blend corner is degenerate or reverses direction.")
    controls = torch.stack(
        (
            start,
            start + 0.2 * root * incoming_tangent,
            start + 0.4 * root * incoming_tangent,
            end - 0.4 * root * outgoing_tangent,
            end - 0.2 * root * outgoing_tangent,
            end,
        )
    )
    return controls, root


def quintic_blend_segments(
    waypoints: torch.Tensor, blend_tolerance: float
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Build ordered linear/quintic segments for a waypoint path.

    Linear segments contain two control points and quintic segments contain
    six. The returned length tensor follows the same order and uses the path
    parameter lengths produced by HolisticMotion.
    """
    if waypoints.ndim != 2 or waypoints.shape[0] < 2:
        raise ValueError("waypoints must have shape (K, D) with K >= 2.")
    if not waypoints.is_floating_point() or not bool(
        torch.isfinite(waypoints).all().item()
    ):
        raise ValueError("waypoints must be a finite floating-point tensor.")
    if not math.isfinite(blend_tolerance) or blend_tolerance < 0.0:
        raise ValueError("blend_tolerance must be finite and non-negative.")
    edge_active = (
        torch.linalg.vector_norm(waypoints[1:] - waypoints[:-1], dim=-1)
        > torch.finfo(waypoints.dtype).eps
    )
    keep = torch.cat(
        (torch.ones(1, dtype=torch.bool, device=waypoints.device), edge_active)
    )
    waypoints = waypoints[keep]
    if waypoints.shape[0] < 2:
        raise ValueError("waypoints must contain at least two distinct positions.")
    segments: list[torch.Tensor] = []
    lengths: list[torch.Tensor] = []
    current = waypoints[0]
    epsilon = torch.finfo(waypoints.dtype).eps
    for index in range(1, waypoints.shape[0] - 1):
        previous, corner, following = waypoints[index - 1 : index + 2]
        incoming = corner - previous
        outgoing = following - corner
        incoming_norm = torch.linalg.vector_norm(incoming)
        outgoing_norm = torch.linalg.vector_norm(outgoing)
        if bool((incoming_norm <= epsilon).item() or (outgoing_norm <= epsilon).item()):
            continue
        direction_change = torch.linalg.vector_norm(
            outgoing / outgoing_norm - incoming / incoming_norm
        )
        if blend_tolerance == 0.0 or bool((direction_change <= epsilon).item()):
            line_length = torch.linalg.vector_norm(corner - current)
            if bool((line_length > epsilon).item()):
                segments.append(torch.stack((current, corner)))
                lengths.append(line_length)
            current = corner
            continue
        controls, curve_length = quintic_blend_control_points(
            previous, corner, following, blend_tolerance
        )
        line_length = torch.linalg.vector_norm(controls[0] - current)
        if bool((line_length > epsilon).item()):
            segments.append(torch.stack((current, controls[0])))
            lengths.append(line_length)
        segments.append(controls)
        lengths.append(curve_length)
        current = controls[-1]
    final_length = torch.linalg.vector_norm(waypoints[-1] - current)
    if bool((final_length > epsilon).item()):
        segments.append(torch.stack((current, waypoints[-1])))
        lengths.append(final_length)
    if not segments:
        raise ValueError("waypoints must contain at least two distinct positions.")
    return segments, torch.stack(lengths)


def evaluate_quintic_blend_path(
    waypoints: torch.Tensor,
    blend_tolerance: float,
    distance: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate position and first two derivatives by global path distance."""
    segments, lengths = quintic_blend_segments(waypoints, blend_tolerance)
    cumulative = lengths.cumsum(dim=0)
    distance = torch.as_tensor(distance, dtype=waypoints.dtype, device=waypoints.device)
    if not bool(torch.isfinite(distance).all().item()):
        raise ValueError("distance must contain only finite values.")
    clamped = distance.clamp(0.0, cumulative[-1])
    indices = torch.searchsorted(
        cumulative.contiguous(), clamped.contiguous(), right=True
    )
    indices = indices.clamp_max(len(segments) - 1)
    starts = torch.cat((torch.zeros_like(cumulative[:1]), cumulative[:-1]))
    local = clamped - starts[indices]
    parameter = (local / lengths[indices]).clamp(0.0, 1.0)
    shape = distance.shape + (waypoints.shape[1],)
    position = torch.empty(shape, dtype=waypoints.dtype, device=waypoints.device)
    tangent = torch.empty_like(position)
    curvature = torch.empty_like(position)
    for index, segment in enumerate(segments):
        mask = indices == index
        if not bool(mask.any().item()):
            continue
        inverse_length = lengths[index].reciprocal()
        if segment.shape[0] == 2:
            delta = segment[1] - segment[0]
            position[mask] = segment[0] + parameter[mask, None] * delta
            tangent[mask] = delta * inverse_length
            curvature[mask] = 0.0
        else:
            position[mask] = bezier_evaluate(segment, parameter[mask])
            tangent[mask] = (
                bezier_derivative(segment, parameter[mask], 1) * inverse_length
            )
            curvature[mask] = (
                bezier_derivative(segment, parameter[mask], 2) * inverse_length.square()
            )
    return position, tangent, curvature


def compose_quintic_blend_state(
    waypoints: torch.Tensor,
    blend_tolerance: float,
    distance: torch.Tensor,
    path_velocity: torch.Tensor,
    path_acceleration: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compose joint states from a blended path and scalar time law.

    The scalar inputs must have identical shapes and represent ``s``,
    ``ds/dt`` and ``d²s/dt²`` respectively.
    """
    distance = torch.as_tensor(distance, dtype=waypoints.dtype, device=waypoints.device)
    path_velocity = torch.as_tensor(
        path_velocity, dtype=waypoints.dtype, device=waypoints.device
    )
    path_acceleration = torch.as_tensor(
        path_acceleration, dtype=waypoints.dtype, device=waypoints.device
    )
    if (
        distance.shape != path_velocity.shape
        or distance.shape != path_acceleration.shape
    ):
        raise ValueError(
            "distance, path_velocity and path_acceleration must share a shape."
        )
    if not bool(torch.isfinite(path_velocity).all().item()) or not bool(
        torch.isfinite(path_acceleration).all().item()
    ):
        raise ValueError("Path derivatives must contain only finite values.")
    position, tangent, curvature = evaluate_quintic_blend_path(
        waypoints, blend_tolerance, distance
    )
    velocity = tangent * path_velocity[..., None]
    acceleration = (
        curvature * path_velocity[..., None].square()
        + tangent * path_acceleration[..., None]
    )
    return position, velocity, acceleration


def compose_quintic_blend_jerk(
    waypoints: torch.Tensor,
    blend_tolerance: float,
    distance: torch.Tensor,
    path_velocity: torch.Tensor,
    path_acceleration: torch.Tensor,
    path_jerk: torch.Tensor,
) -> torch.Tensor:
    """Compose joint jerk using the full third-order chain rule."""
    distance = torch.as_tensor(distance, dtype=waypoints.dtype, device=waypoints.device)
    values = [
        torch.as_tensor(value, dtype=waypoints.dtype, device=waypoints.device)
        for value in (path_velocity, path_acceleration, path_jerk)
    ]
    if any(value.shape != distance.shape for value in values):
        raise ValueError("All scalar time-law inputs must share the distance shape.")
    if not bool(torch.isfinite(distance).all().item()) or any(
        not bool(torch.isfinite(value).all().item()) for value in values
    ):
        raise ValueError("Scalar time-law inputs must contain only finite values.")
    _, lengths = quintic_blend_segments(waypoints, blend_tolerance)
    cumulative = lengths.cumsum(dim=0)
    clamped = distance.clamp(0.0, cumulative[-1])
    indices = torch.searchsorted(
        cumulative.contiguous(), clamped.contiguous(), right=True
    )
    indices = indices.clamp_max(len(lengths) - 1)
    starts = torch.cat((torch.zeros_like(cumulative[:1]), cumulative[:-1]))
    parameter = ((clamped - starts[indices]) / lengths[indices]).clamp(0.0, 1.0)
    segments, _ = quintic_blend_segments(waypoints, blend_tolerance)
    _, tangent, curvature = evaluate_quintic_blend_path(
        waypoints, blend_tolerance, distance
    )
    third = torch.zeros_like(tangent)
    for index, segment in enumerate(segments):
        mask = indices == index
        if segment.shape[0] == 6 and bool(mask.any().item()):
            third[mask] = bezier_derivative(segment, parameter[mask], 3) / lengths[
                index
            ].pow(3)
    velocity, acceleration, jerk = values
    return (
        third * velocity[..., None].pow(3)
        + 3.0 * curvature * velocity[..., None] * acceleration[..., None]
        + tangent * jerk[..., None]
    )


def project_quintic_blend_limits(
    waypoints: torch.Tensor,
    blend_tolerance: float,
    velocity_limit: torch.Tensor,
    acceleration_limit: torch.Tensor,
    *,
    samples: int = 2049,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Conservatively project joint limits onto global path distance.

    Curvature and tangential bounds are projected independently. The planner
    checks their composed peak and applies the minimal uniform time stretch.
    """
    if velocity_limit.shape != (waypoints.shape[1],) or acceleration_limit.shape != (
        waypoints.shape[1],
    ):
        raise ValueError("Limits must have shape (D,) matching waypoint dimension.")
    if bool(
        (velocity_limit <= 0).any().item() or (acceleration_limit <= 0).any().item()
    ):
        raise ValueError("Limits must be positive.")
    _, lengths = quintic_blend_segments(waypoints, blend_tolerance)
    distance = torch.linspace(
        0.0,
        lengths.sum().item(),
        samples,
        dtype=waypoints.dtype,
        device=waypoints.device,
    )
    _, tangent, curvature = evaluate_quintic_blend_path(
        waypoints, blend_tolerance, distance
    )
    epsilon = torch.finfo(waypoints.dtype).eps
    tangent_peak = tangent.abs().amax(dim=0)
    curvature_peak = curvature.abs().amax(dim=0)
    active_tangent = tangent_peak > epsilon
    path_velocity = (
        velocity_limit[active_tangent] / tangent_peak[active_tangent]
    ).amin()
    curvature_active = curvature_peak > epsilon
    if bool(curvature_active.any().item()):
        curvature_velocity = torch.sqrt(
            (acceleration_limit[curvature_active] / curvature_peak[curvature_active])
        ).amin()
        path_velocity = torch.minimum(path_velocity, curvature_velocity)
    path_acceleration = (
        acceleration_limit[active_tangent] / tangent_peak[active_tangent]
    ).amin()
    return path_velocity, path_acceleration


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
        """Return geometric length using 32-point Gauss-Legendre quadrature."""
        return bezier_arc_length(self._control_points)

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


def bezier_arc_length(control_points: torch.Tensor) -> torch.Tensor:
    """Integrate the Euclidean Bézier speed over ``[0, 1]``.

    A fixed 32-point Gauss-Legendre rule is deterministic, differentiable with
    respect to the control points, and substantially more accurate than a
    polyline-length estimate for curved quadratic and quintic paths.
    """
    _validate_control_points(control_points)
    nodes = control_points.new_tensor(
        [
            0.0483076656877383,
            0.1444719615827965,
            0.2392873622521371,
            0.3318686022821277,
            0.4213512761306353,
            0.5068999089322294,
            0.5877157572407623,
            0.6630442669302152,
            0.7321821187402897,
            0.7944837959679424,
            0.8493676137325700,
            0.8963211557660521,
            0.9349060759377397,
            0.9647622555875064,
            0.9856115115452684,
            0.9972638618494816,
        ]
    )
    weights = control_points.new_tensor(
        [
            0.0965400885147278,
            0.0956387200792749,
            0.0938443990808046,
            0.0911738786957639,
            0.0876520930044038,
            0.0833119242269468,
            0.0781938957870703,
            0.0723457941088485,
            0.0658222227763618,
            0.0586840934785355,
            0.0509980592623762,
            0.0428358980222267,
            0.0342738629130214,
            0.0253920653092621,
            0.0162743947309057,
            0.0070186100094701,
        ]
    )
    parameters = torch.cat(((1.0 - nodes) * 0.5, (1.0 + nodes) * 0.5))
    quadrature_weights = torch.cat((weights, weights)) * 0.5
    derivative = bezier_derivative(control_points.unsqueeze(-3), parameters)
    speed = torch.linalg.vector_norm(derivative, dim=-1)
    return (speed * quadrature_weights).sum(dim=-1)


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
