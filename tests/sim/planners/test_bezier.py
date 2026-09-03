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
    bezier_derivative,
    bezier_evaluate,
    sample_bezier_path,
)


def test_bezier_path_encapsulates_geometry_without_changing_tensor_shape() -> None:
    control_points = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float64)
    path = BezierPath(control_points)

    assert path.degree == 2
    assert path.dimension == 1
    assert torch.allclose(path.evaluate(torch.tensor([0.5])), torch.tensor([[1.0]]))
    assert path.control_points is control_points
    parameter = torch.tensor([0.5], dtype=torch.float64)
    assert torch.allclose(path.tangent(parameter), path.derivative(parameter, 1))
    assert torch.allclose(path.curvature(parameter), path.derivative(parameter, 2))


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
        5.0 * torch.stack((control_points[1] - control_points[0], control_points[-1] - control_points[-2])),
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


@pytest.mark.parametrize("sample_count", [True, 1, 1.5])
def test_sample_count_must_be_an_integer_of_at_least_two(sample_count: object) -> None:
    control_points = torch.zeros((3, 2), dtype=torch.float32)

    with pytest.raises(ValueError, match="at least 2"):
        sample_bezier_path(control_points, sample_count)  # type: ignore[arg-type]


def test_public_evaluation_rejects_unsupported_degree_and_nonfinite_parameter() -> None:
    with pytest.raises(ValueError, match="3\|6"):
        bezier_evaluate(torch.zeros((5, 2)), torch.tensor(0.5))

    with pytest.raises(ValueError, match="finite"):
        bezier_evaluate(torch.zeros((3, 2)), torch.tensor(float("nan")))

    with pytest.raises(ValueError, match="finite"):
        bezier_evaluate(torch.tensor([[0.0], [float("inf")], [1.0]]), torch.tensor(0.5))
