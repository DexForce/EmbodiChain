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

"""Numerical gradient contracts for the shared IK computation."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from embodichain.compute.kinematics import (
    DifferentiableIKCfg,
    SerialChainKinematics,
    damped_least_squares_step,
    pose_error,
    solve_dls_ik,
)


def model_and_tcp():
    urdf = (
        Path(__file__).resolve().parents[2] / "fixtures/kinematics/seven_dof.urdf"
    ).read_text()
    model = SerialChainKinematics.from_urdf(urdf, "link6", dtype=torch.float64)
    angle = torch.tensor(
        [[0.0, -0.2, 0.1], [0.2, 0.0, -0.3], [-0.1, 0.3, 0.0]], dtype=torch.float64
    )
    tcp = torch.eye(4, dtype=torch.float64)
    tcp[:3, :3] = torch.matrix_exp(angle)
    tcp[:3, 3] = torch.tensor([0.1, -0.05, 0.08], dtype=torch.float64)
    return model, tcp


def se3(delta):
    x, y, z = delta[..., 3:].unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(
        *delta.shape[:-1], 3, 3
    )
    top = torch.cat((torch.matrix_exp(skew), delta[..., :3, None]), -1)
    bottom = delta.new_tensor([0, 0, 0, 1]).expand(*delta.shape[:-1], 1, 4)
    return torch.cat((top, bottom), -2)


def test_dls_gradcheck_and_singular_jacobian():
    torch.manual_seed(19)
    error = torch.randn(2, 6, dtype=torch.float64, requires_grad=True)
    jac = torch.randn(2, 6, 7, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda e, j: damped_least_squares_step(e, j, 0.05), (error, jac), fast_mode=True
    )
    output = damped_least_squares_step(error, jac * 0, 0.05)
    grads = torch.autograd.grad(output.sum(), (error, jac))
    assert all(torch.isfinite(g).all() for g in grads)
    torch.testing.assert_close(output, torch.zeros_like(output))


@pytest.mark.parametrize("damping", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_damping(damping):
    with pytest.raises(ValueError):
        damped_least_squares_step(torch.zeros(6), torch.zeros(6, 7), damping)


@pytest.mark.parametrize("angle", [0.0, 1e-9, 0.3, 3.141592653589793 - 1e-4])
def test_rotation_residual_gradient(angle):
    delta = torch.tensor(
        [[0.1, 0.2, -0.1, angle, 0.0, 0.0]], dtype=torch.float64, requires_grad=True
    )
    identity = torch.eye(4, dtype=torch.float64)

    def residual(d):
        p, r = pose_error(identity, se3(d))
        return torch.cat((p, r), -1)

    assert torch.autograd.gradcheck(residual, (delta,), eps=1e-6, atol=1e-4, rtol=1e-3)
    torch.testing.assert_close(residual(delta), delta)


def test_tcp_jacobian_forward_and_backward():
    model, tcp = model_and_tcp()
    q = torch.linspace(-0.2, 0.3, 7, dtype=torch.float64)[None].requires_grad_()
    evaluate = lambda joints: model.evaluate(joints, tcp)
    assert torch.autograd.gradcheck(
        lambda joints: evaluate(joints)[0], (q,), fast_mode=True
    )
    assert torch.autograd.gradcheck(
        lambda joints: evaluate(joints)[1], (q,), fast_mode=True
    )
    reference = evaluate(q)[1]
    epsilon = 1e-6
    columns = []
    for j in range(7):
        change = torch.zeros_like(q)
        change[:, j] = epsilon
        minus, plus = evaluate(q - change)[0], evaluate(q + change)[0]
        p, r = pose_error(minus, plus)
        columns.append(torch.cat((p, r), -1) / (2 * epsilon))
    torch.testing.assert_close(
        reference, torch.stack(columns, -1), atol=1e-5, rtol=1e-4
    )


def test_complete_ik_target_and_seed_gradcheck():
    model, tcp = model_and_tcp()
    evaluate = lambda joints: model.evaluate(joints, tcp)
    q = torch.linspace(-0.2, 0.3, 7, dtype=torch.float64)[None].requires_grad_()
    goal = evaluate(q.detach() + 0.04)[0]
    delta = torch.full((1, 6), 0.01, dtype=torch.float64, requires_grad=True)
    cfg = DifferentiableIKCfg(max_iterations=4)

    def solve(d, seed):
        return solve_dls_ik(
            se3(d) @ goal, seed, evaluate, model.chain.low, model.chain.high, cfg
        ).qpos

    assert torch.autograd.gradcheck(
        solve, (delta, q), eps=1e-6, atol=1e-4, rtol=1e-3, fast_mode=True
    )


def test_configuration_rejects_invalid_iterations():
    for value in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            DifferentiableIKCfg(max_iterations=value)
