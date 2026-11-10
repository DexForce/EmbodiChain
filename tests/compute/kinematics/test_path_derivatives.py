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

from embodichain.compute.kinematics._path_derivatives import (
    joint_derivatives_from_path_time_law,
    nearest_equivalent_joint_solution,
)


def test_path_time_law_produces_exact_joint_derivatives_for_identity_model() -> None:
    sample_count = 101
    parameter = torch.linspace(0.0, torch.pi, sample_count, dtype=torch.float64)
    path_position = torch.linspace(
        0.0, 0.1, sample_count, dtype=torch.float64
    ).unsqueeze(0)
    path_velocity = torch.sin(parameter).unsqueeze(0)
    path_acceleration = torch.cos(parameter).unsqueeze(0)
    jacobians = (
        torch.eye(6, dtype=torch.float64)
        .reshape(1, 1, 6, 6)
        .expand(1, sample_count, -1, -1)
    )
    tangent = torch.tensor([[0.0, 0.0, -1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)

    velocity, acceleration = joint_derivatives_from_path_time_law(
        jacobians,
        path_position,
        path_velocity,
        path_acceleration,
        tangent,
    )

    assert torch.allclose(velocity[..., 2], -path_velocity, atol=1e-7)
    assert torch.allclose(acceleration[..., 2], -path_acceleration, atol=1e-7)
    assert torch.count_nonzero(velocity[..., [0, 1, 3, 4, 5]]) == 0


def test_path_time_law_includes_jacobian_curvature_acceleration() -> None:
    sample_count = 51
    path_position = torch.linspace(
        0.0, 1.0, sample_count, dtype=torch.float64
    ).unsqueeze(0)
    path_velocity = torch.full_like(path_position, 0.5)
    path_acceleration = torch.zeros_like(path_position)
    jacobians = (
        torch.eye(6, dtype=torch.float64)
        .reshape(1, 1, 6, 6)
        .repeat(1, sample_count, 1, 1)
    )
    jacobians[0, :, 0, 0] = 1.0 + path_position[0]
    tangent = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64)

    velocity, acceleration = joint_derivatives_from_path_time_law(
        jacobians,
        path_position,
        path_velocity,
        path_acceleration,
        tangent,
    )

    expected_q_s = 1.0 / (1.0 + path_position)
    expected_q_ss = -1.0 / (1.0 + path_position).square()
    assert torch.allclose(velocity[..., 0], expected_q_s * 0.5, atol=1e-12)
    assert torch.allclose(acceleration[..., 0], expected_q_ss * 0.25, atol=1e-12)


def test_nearest_equivalent_ik_solution_removes_angle_wrap() -> None:
    seed = torch.tensor([[3.13]])
    wrapped_solution = torch.tensor([[-3.13]])
    limits = torch.tensor([[[-2.0 * torch.pi, 2.0 * torch.pi]]])

    continuous = nearest_equivalent_joint_solution(wrapped_solution, seed, limits)

    assert (continuous - seed).abs().item() < 0.03


def test_equivalent_solution_outside_limits_preserves_original() -> None:
    solution = torch.tensor([[-3.13]])
    result = nearest_equivalent_joint_solution(
        solution, torch.tensor([[3.13]]), torch.tensor([[[-torch.pi, torch.pi]]])
    )
    torch.testing.assert_close(result, solution)


@pytest.mark.parametrize(
    "parameter",
    [
        torch.tensor([[0.0, 0.0, 1.0]]),
        torch.tensor([[0.0, 1.0, 0.5]]),
    ],
)
def test_nonincreasing_path_parameter_is_rejected(parameter: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        joint_derivatives_from_path_time_law(
            torch.eye(6).repeat(1, 3, 1, 1),
            parameter,
            torch.ones_like(parameter),
            torch.zeros_like(parameter),
            torch.ones(1, 6),
        )
