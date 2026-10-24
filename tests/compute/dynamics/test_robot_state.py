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

"""Independent rotation and local-state integration checks."""

from __future__ import annotations

import math

import torch

from embodichain.compute.dynamics.robot_state import (
    integrate_robot_state,
    robot_state_error,
    robot_state_features,
    robot_state_increment,
    rotation_exp,
    rotation_log,
)


def _state(batch: int = 2) -> torch.Tensor:
    result = torch.zeros(batch, 22, dtype=torch.float64)
    result[:, 3:12] = torch.eye(3, dtype=torch.float64).flatten()
    return result


def test_so3_known_axis_roundtrip_and_zero_gradients():
    vectors = torch.tensor(
        [[0.0, 0, 0], [0, 0, math.pi / 2], [0.1, -0.2, 0.3]],
        dtype=torch.float64,
        requires_grad=True,
    )
    matrix = rotation_exp(vectors)
    expected_z = torch.tensor([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=torch.float64)
    torch.testing.assert_close(matrix[1], expected_z, atol=1e-12, rtol=0)
    torch.testing.assert_close(rotation_log(matrix), vectors, atol=1e-10, rtol=1e-10)
    assert torch.autograd.gradcheck(
        lambda value: rotation_log(rotation_exp(value)), (vectors,), eps=1e-6, atol=1e-5
    )
    at_pi = rotation_log(
        rotation_exp(torch.tensor([[math.pi, 0, 0]], dtype=torch.float64))
    )
    torch.testing.assert_close(
        at_pi.abs(),
        torch.tensor([[math.pi, 0, 0]], dtype=torch.float64),
        atol=1e-8,
        rtol=0,
    )
    near_pi = torch.tensor([[1.0, -2.0, 3.0]], dtype=torch.float64)
    near_pi = near_pi / near_pi.norm() * (math.pi - 1e-3)
    torch.testing.assert_close(
        rotation_log(rotation_exp(near_pi)), near_pi, atol=1e-9, rtol=1e-9
    )


def test_body_frame_increment_uses_current_rotation():
    state = _state()
    state[:, 3:12] = rotation_exp(
        torch.tensor([[0.0, 0, math.pi / 2]], dtype=torch.float64)
    ).flatten(-2)
    increment = torch.zeros(2, 16, dtype=torch.float64)
    increment[:, 0] = 0.2
    increment[:, 6] = 0.4
    increment[:, 12:] = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    predicted = integrate_robot_state(state, increment)
    torch.testing.assert_close(
        predicted[:, :3],
        torch.tensor([[0.0, 0.2, 0]], dtype=torch.float64).expand(2, -1),
        atol=1e-12,
        rtol=0,
    )
    torch.testing.assert_close(
        predicted[:, 12:15],
        torch.tensor([[0.0, 0.4, 0]], dtype=torch.float64).expand(2, -1),
        atol=1e-12,
        rtol=0,
    )
    torch.testing.assert_close(robot_state_increment(state, predicted), increment)
    torch.testing.assert_close(
        robot_state_error(predicted, predicted),
        torch.zeros_like(increment),
        atol=1e-12,
        rtol=0,
    )


def test_features_are_invariant_to_world_xy_translation_and_yaw():
    state = _state()
    state[:, 12:18] = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=torch.float64)
    rotation = rotation_exp(torch.tensor([0.0, 0, 0.7], dtype=torch.float64))
    transformed = state.clone()
    transformed[:, :2] += 15
    transformed[:, 3:12] = (rotation @ state[:, 3:12].reshape(-1, 3, 3)).flatten(-2)
    transformed[:, 12:15] = (rotation @ state[:, 12:15].unsqueeze(-1)).squeeze(-1)
    transformed[:, 15:18] = (rotation @ state[:, 15:18].unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(
        robot_state_features(state), robot_state_features(transformed)
    )
