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

from __future__ import annotations

import pytest
import torch

from embodichain.lab.sim.planners.bezier import (
    BezierPath,
    bezier_arc_length,
    compose_quintic_blend_state,
    compose_quintic_blend_jerk,
    bezier_derivative,
    bezier_evaluate,
    evaluate_quintic_blend_path,
    project_quintic_blend_limits,
    quintic_blend_control_points,
    quintic_blend_segments,
    sample_bezier_path,
)


def test_quadratic_bezier_arc_length_matches_closed_form() -> None:
    control_points = torch.tensor(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float64
    )
    expected = 1.0 + torch.asinh(torch.tensor(1.0, dtype=torch.float64)) / (2.0**0.5)

    assert bezier_arc_length(control_points).item() == pytest.approx(
        expected.item(), abs=1e-14
    )
    assert BezierPath(control_points).length.item() == pytest.approx(
        expected.item(), abs=1e-14
    )


def test_quintic_corner_controls_and_path_length_match_binding() -> None:
    previous = torch.tensor([0.0, 0.0], dtype=torch.float64)
    corner = torch.tensor([1.0, 0.0], dtype=torch.float64)
    following = torch.tensor([1.0, 1.0], dtype=torch.float64)
    controls, blend_length = quintic_blend_control_points(
        previous, corner, following, 0.1
    )
    trim = 0.282842712474619

    assert torch.allclose(
        controls[0], torch.tensor([1.0 - trim, 0.0], dtype=torch.float64)
    )
    assert torch.allclose(controls[-1], torch.tensor([1.0, trim], dtype=torch.float64))
    total_length = 2.0 * (1.0 - trim) + blend_length.item()
    assert total_length == pytest.approx(1.897644073535952, abs=1e-12)


def test_multi_waypoint_blend_segments_match_binding_length() -> None:
    waypoints = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [2.0, 1.0]],
        dtype=torch.float64,
    )
    segments, lengths = quintic_blend_segments(waypoints, 0.1)

    assert [segment.shape[0] for segment in segments] == [2, 6, 2, 6, 2]
    assert torch.equal(segments[0][0], waypoints[0])
    assert torch.equal(segments[-1][-1], waypoints[-1])
    assert lengths.sum().item() == pytest.approx(2.7952881470719038, abs=1e-12)


def test_blend_degenerate_waypoints_match_binding_behavior() -> None:
    duplicate = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
        dtype=torch.float64,
    )
    _, lengths = quintic_blend_segments(duplicate, 0.1)
    assert lengths.sum().item() == pytest.approx(1.897644073535952, abs=1e-12)

    collinear = torch.tensor([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]], dtype=torch.float64)
    _, lengths = quintic_blend_segments(collinear, 0.1)
    assert lengths.sum().item() == pytest.approx(1.0)

    reverse = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
    with pytest.raises(ValueError, match="degenerate|reverses"):
        quintic_blend_segments(reverse, 0.1)

    with pytest.raises(ValueError, match="distinct"):
        quintic_blend_segments(torch.zeros((3, 2), dtype=torch.float64), 0.1)


def test_blended_path_distance_derivatives_are_analytic() -> None:
    waypoints = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
    midpoint = torch.tensor(1.897644073535952 / 2.0, dtype=torch.float64)
    position, tangent, curvature = evaluate_quintic_blend_path(waypoints, 0.1, midpoint)
    step = 1e-5
    before = evaluate_quintic_blend_path(waypoints, 0.1, midpoint - step)[0]
    after = evaluate_quintic_blend_path(waypoints, 0.1, midpoint + step)[0]

    assert torch.allclose(
        position,
        torch.tensor([0.9309738779010016, 0.06902612209899879], dtype=torch.float64),
        atol=1e-12,
    )
    assert torch.allclose(tangent, (after - before) / (2.0 * step), atol=1e-8)
    assert torch.isfinite(curvature).all()
    clamped = evaluate_quintic_blend_path(waypoints, 0.1, torch.tensor([-1.0, 5.0]))[0]
    assert torch.equal(clamped, waypoints[[0, -1]])


def test_blended_path_composes_scalar_time_law_by_chain_rule() -> None:
    waypoints = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
    distance = torch.tensor(1.897644073535952 / 2.0, dtype=torch.float64)
    position, velocity, acceleration = compose_quintic_blend_state(
        waypoints,
        0.1,
        distance,
        torch.tensor(0.3, dtype=torch.float64),
        torch.tensor(0.4, dtype=torch.float64),
    )
    tangent = evaluate_quintic_blend_path(waypoints, 0.1, distance)[1]
    assert torch.allclose(velocity, tangent * 0.3)
    assert torch.isfinite(acceleration).all()
    stopped = compose_quintic_blend_state(
        waypoints, 0.1, distance, torch.tensor(0.0), torch.tensor(0.0)
    )
    assert torch.equal(stopped[0], position)
    assert torch.count_nonzero(stopped[1]) == 0
    assert torch.count_nonzero(stopped[2]) == 0


def test_blended_path_third_order_chain_rule_matches_difference() -> None:
    waypoints = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
    distance = torch.tensor(1.897644073535952 / 2.0, dtype=torch.float64)
    speed = torch.tensor(0.3, dtype=torch.float64)
    jerk = compose_quintic_blend_jerk(
        waypoints, 0.1, distance, speed, torch.tensor(0.0), torch.tensor(0.0)
    )
    step = 1e-5
    before = compose_quintic_blend_state(
        waypoints, 0.1, distance - speed * step, speed, torch.tensor(0.0)
    )[2]
    after = compose_quintic_blend_state(
        waypoints, 0.1, distance + speed * step, speed, torch.tensor(0.0)
    )[2]
    assert torch.allclose(jerk, (after - before) / (2.0 * step), atol=1e-7)


def test_blended_path_limit_projection_bounds_independent_terms() -> None:
    waypoints = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
    joint_velocity = torch.tensor([0.8, 0.6], dtype=torch.float64)
    joint_acceleration = torch.tensor([1.0, 0.7], dtype=torch.float64)
    path_velocity, path_acceleration = project_quintic_blend_limits(
        waypoints, 0.1, joint_velocity, joint_acceleration
    )
    distance = torch.linspace(0.0, 1.897644073535952, 2049, dtype=torch.float64)
    _, velocity, curvature_acceleration = compose_quintic_blend_state(
        waypoints,
        0.1,
        distance,
        path_velocity.expand_as(distance),
        torch.zeros_like(distance),
    )
    _, _, tangent_acceleration = compose_quintic_blend_state(
        waypoints,
        0.1,
        distance,
        torch.zeros_like(distance),
        path_acceleration.expand_as(distance),
    )
    assert torch.all(velocity.abs().amax(dim=0) <= joint_velocity + 1e-12)
    assert torch.all(
        curvature_acceleration.abs().amax(dim=0) <= joint_acceleration + 1e-12
    )
    assert torch.all(
        tangent_acceleration.abs().amax(dim=0) <= joint_acceleration + 1e-12
    )


def test_bezier_path_encapsulates_geometry_without_changing_tensor_shape() -> None:
    control_points = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float64)
    path = BezierPath(control_points)

    assert path.degree == 2
    assert path.dimension == 1
    assert torch.allclose(
        path.evaluate(torch.tensor([0.5])),
        torch.tensor([[1.0]], dtype=control_points.dtype),
    )
    assert path.control_points is control_points
    parameter = torch.tensor([0.5], dtype=torch.float64)
    assert torch.allclose(path.tangent(parameter), path.derivative(parameter, 1))
    assert torch.allclose(path.curvature(parameter), path.derivative(parameter, 2))
    assert torch.allclose(
        path.arc_tangent(parameter), torch.tensor([[1.0]], dtype=torch.float64)
    )
    assert torch.allclose(
        path.arc_curvature(parameter), torch.zeros((1, 1), dtype=torch.float64)
    )


def test_quintic_bezier_reaches_endpoints_and_endpoint_derivatives() -> None:
    control_points = torch.tensor(
        [[0.0, 0.0], [0.2, 0.0], [0.8, 0.2], [1.2, 0.8], [1.8, 1.0], [2.0, 1.0]],
        dtype=torch.float64,
    )

    positions = bezier_evaluate(control_points, torch.tensor([0.0, 1.0]))
    velocities = bezier_derivative(control_points, torch.tensor([0.0, 1.0]))

    assert torch.allclose(positions, control_points[[0, -1]])
    assert torch.allclose(
        velocities,
        5.0
        * torch.stack(
            (
                control_points[1] - control_points[0],
                control_points[-1] - control_points[-2],
            )
        ),
    )


def test_quadratic_bezier_matches_reference_second_order_curve() -> None:
    control_points = torch.tensor([[0.0], [2.0], [4.0]], dtype=torch.float64)

    points = bezier_evaluate(control_points, torch.tensor([0.25, 0.75]))
    curvature = bezier_derivative(control_points, torch.tensor([0.25, 0.75]), order=2)

    assert torch.allclose(points[:, 0], torch.tensor([1.0, 3.0], dtype=torch.float64))
    assert torch.allclose(curvature[:, 0], torch.full((2,), 0.0, dtype=torch.float64))


def test_batched_parameters_are_evaluated_per_path() -> None:
    first = torch.tensor([[0.0], [1.0], [2.0]])
    second = torch.tensor([[10.0], [11.0], [12.0]])
    control_points = torch.stack((first, second))
    parameters = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

    result = bezier_evaluate(control_points, parameters)

    assert result.shape == (2, 2, 1)
    assert torch.equal(result[..., 0], torch.tensor([[0.0, 2.0], [12.0, 10.0]]))


def test_arc_length_sampling_is_nearly_uniform() -> None:
    control_points = torch.tensor(
        [[0.0, 0.0], [0.0, 2.0], [3.0, 2.0]],
        dtype=torch.float64,
    )

    points, cumulative_length = sample_bezier_path(control_points, 21)
    segment_lengths = torch.linalg.vector_norm(points[1:] - points[:-1], dim=-1)

    assert torch.equal(points[[0, -1]], control_points[[0, -1]])
    assert torch.all(torch.diff(cumulative_length) >= 0.0)
    assert float(segment_lengths.max() - segment_lengths.min()) < 0.01


def test_batched_arc_length_sampling_keeps_paths_independent() -> None:
    control_points = torch.tensor(
        [
            [[0.0], [0.0], [0.0]],
            [[0.0], [2.0], [4.0]],
        ],
        dtype=torch.float64,
    )

    points, lengths = sample_bezier_path(control_points, 9)

    assert points.shape == (2, 9, 1)
    assert lengths.shape == (2, 9)
    assert torch.allclose(points[:, 0, :], control_points[:, 0, :])
    assert torch.allclose(points[:, -1, :], control_points[:, -1, :])
    assert lengths[0, -1].item() == pytest.approx(0.0)
    assert lengths[1, -1].item() == pytest.approx(4.0, abs=1e-3)


def test_arc_length_inverse_matches_nonuniform_quadratic_closed_form() -> None:
    controls = torch.tensor([[0.0], [0.1], [1.0]], dtype=torch.float64)
    path = BezierPath(controls)
    distance = torch.tensor([-1.0, 0.0, 0.1, 0.5, 1.0, 2.0], dtype=torch.float64)
    # This straight curve has x(u) = 0.2*u + 0.8*u**2, so distance and
    # polynomial parameter differ even though geometric curvature is zero.
    expected = (-0.2 + torch.sqrt(0.04 + 3.2 * distance.clamp(0.0, 1.0))) / 1.6

    parameter = path.parameter_at_arc_length(distance)

    torch.testing.assert_close(parameter, expected, atol=1e-7, rtol=0.0)
    torch.testing.assert_close(
        path.evaluate(parameter)[..., 0], distance.clamp(0.0, 1.0), atol=2e-8, rtol=0.0
    )
    scalar = path.parameter_at_arc_length(distance[3])
    assert scalar.ndim == 0
    torch.testing.assert_close(scalar, parameter[3])


def test_arc_length_inverse_handles_batched_and_stationary_paths() -> None:
    controls = torch.tensor(
        [[[0.0], [0.1], [1.0]], [[0.0], [0.2], [2.0]], [[3.0], [3.0], [3.0]]],
        dtype=torch.float64,
    )
    path = BezierPath(controls)
    distance = torch.tensor([0.0, 0.5, 1.0, 2.0], dtype=torch.float64)
    parameter = path.parameter_at_arc_length(distance)
    expected = torch.minimum(distance[None], torch.tensor([[1.0], [2.0], [0.0]]))

    torch.testing.assert_close(
        path.evaluate(parameter)[..., 0],
        expected + controls[:, :1, 0],
        atol=4e-8,
        rtol=0.0,
    )
    assert torch.count_nonzero(parameter[2]) == 0
    torch.testing.assert_close(
        path.parameter_at_arc_length(expected), parameter, atol=1e-12, rtol=0.0
    )


@pytest.mark.parametrize("distance", [float("nan"), float("inf"), -float("inf")])
def test_arc_length_inverse_rejects_nonfinite_distance(distance: float) -> None:
    path = BezierPath(torch.tensor([[0.0], [0.5], [1.0]]))
    with pytest.raises(ValueError, match="distance must contain only finite"):
        path.parameter_at_arc_length(torch.tensor(distance))


@pytest.mark.parametrize("table_count", [True, 1, 1.5])
def test_arc_length_inverse_rejects_invalid_table_count(table_count: object) -> None:
    path = BezierPath(torch.tensor([[0.0], [0.5], [1.0]]))
    with pytest.raises(
        ValueError, match="table_count must be an integer of at least 2"
    ):
        path.parameter_at_arc_length(torch.tensor(0.5), table_count=table_count)  # type: ignore[arg-type]


@pytest.mark.parametrize("sample_count", [True, 1, 1.5])
def test_sample_count_must_be_an_integer_of_at_least_two(sample_count: object) -> None:
    control_points = torch.zeros((3, 2), dtype=torch.float32)

    with pytest.raises(ValueError, match="at least 2"):
        sample_bezier_path(control_points, sample_count)  # type: ignore[arg-type]


def test_public_evaluation_rejects_unsupported_degree_and_nonfinite_parameter() -> None:
    with pytest.raises(ValueError, match=r"3\|6"):
        bezier_evaluate(torch.zeros((5, 2)), torch.tensor(0.5))

    with pytest.raises(ValueError, match="finite"):
        bezier_evaluate(torch.zeros((3, 2)), torch.tensor(float("nan")))

    with pytest.raises(ValueError, match="finite"):
        bezier_evaluate(torch.tensor([[0.0], [float("inf")], [1.0]]), torch.tensor(0.5))


def test_arc_derivatives_reject_stationary_curve() -> None:
    path = BezierPath(torch.ones((3, 2), dtype=torch.float64))
    with pytest.raises(ValueError, match="stationary"):
        path.arc_tangent(torch.tensor(0.5, dtype=torch.float64))
    with pytest.raises(ValueError, match="stationary"):
        path.arc_curvature(torch.tensor(0.5, dtype=torch.float64))


def test_blend_jerk_rejects_nonfinite_time_law() -> None:
    waypoints = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
    finite = torch.tensor([0.0, 0.5], dtype=torch.float64)
    nonfinite = torch.tensor([0.0, float("nan")], dtype=torch.float64)

    with pytest.raises(ValueError, match="finite"):
        compose_quintic_blend_jerk(waypoints, 0.1, finite, finite, finite, nonfinite)
