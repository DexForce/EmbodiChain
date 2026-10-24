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

"""Numerical checks of the native FK autograd boundary."""

from __future__ import annotations

import pytest
import torch
import warp as wp

newton = pytest.importorskip("newton")

from embodichain.lab.sim.physics.solvers._kinematics import differentiable_fk


def _model():
    builder = newton.ModelBuilder()
    body = builder.add_link(
        mass=1.0,
        com=wp.vec3(0.0, 0.0, -0.5),
        inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    )
    joint = builder.add_joint_revolute(
        parent=-1, child=body, axis=wp.vec3(0.0, 1.0, 0.0)
    )
    builder.add_articulation([joint])
    return builder.finalize(device="cpu")


def test_fk_pose_and_velocity_gradients_match_finite_difference():
    model = _model()
    q = torch.tensor([0.4], requires_grad=True)
    qd = torch.tensor([0.7], requires_grad=True)
    pose, velocity = differentiable_fk(model, q, qd)
    pose_weights = torch.arange(1.0, 8.0).view(1, 7)
    velocity_weights = torch.arange(1.0, 7.0).view(1, 6)
    loss = (pose * pose_weights).sum() + (velocity * velocity_weights).sum()
    gradients = torch.autograd.grad(loss, (q, qd), retain_graph=True)
    repeated = torch.autograd.grad(loss, (q, qd))
    assert all(torch.allclose(a, b) for a, b in zip(gradients, repeated))

    def evaluate(position, speed):
        a, b = differentiable_fk(model, position, speed)
        return (a * pose_weights).sum() + (b * velocity_weights).sum()

    epsilon = 1.0e-3
    expected_q = (
        evaluate(q.detach() + epsilon, qd.detach())
        - evaluate(q.detach() - epsilon, qd.detach())
    ) / (2 * epsilon)
    expected_qd = (
        evaluate(q.detach(), qd.detach() + epsilon)
        - evaluate(q.detach(), qd.detach() - epsilon)
    ) / (2 * epsilon)
    assert torch.allclose(gradients[0], expected_q.reshape(1), atol=2.0e-3, rtol=2.0e-3)
    assert torch.allclose(
        gradients[1], expected_qd.reshape(1), atol=2.0e-3, rtol=2.0e-3
    )


def test_later_fk_does_not_overwrite_retained_outputs_or_graph():
    model = _model()
    q = torch.tensor([0.2], requires_grad=True)
    qd = torch.tensor([0.3], requires_grad=True)
    first, _ = differentiable_fk(model, q, qd)
    saved = first.detach().clone()
    differentiable_fk(model, q + 1.0, qd + 2.0)
    assert torch.equal(first, saved)
    first.sum().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()


def test_fk_rejects_unmatched_coordinate_layout():
    model = _model()
    with pytest.raises(ValueError, match="flattened"):
        differentiable_fk(model, torch.zeros(1, 1), torch.zeros(1))
    with pytest.raises(TypeError, match="float32"):
        differentiable_fk(model, torch.zeros(1, dtype=torch.float64), torch.zeros(1))
