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

"""FK-only network gradients and independent forward-call graphs."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from embodichain.compute.kinematics import SerialChainKinematics, pose_error
from embodichain.learning.layers import DifferentiableFKLayer


def make_fk(device: str = "cpu") -> DifferentiableFKLayer:
    xml = (
        Path(__file__).resolve().parents[2] / "fixtures/kinematics/seven_dof.urdf"
    ).read_text()
    model = SerialChainKinematics.from_urdf(
        xml, "link6", dtype=torch.float64, device=device
    )
    tcp = torch.eye(4, dtype=torch.float64, device=device)
    tcp[:3, 3] = tcp.new_tensor([0.1, -0.05, 0.08])
    return DifferentiableFKLayer(model, tcp).eval()


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_fk_pose_loss_reaches_network_without_explicit_jacobian(monkeypatch, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    fk = make_fk(device)
    predictor = torch.nn.Linear(2, fk.dof, dtype=torch.float64, device=device)
    with torch.no_grad():
        predictor.weight.copy_(
            torch.linspace(-0.1, 0.2, 14, device=device).reshape(7, 2)
        )
        predictor.bias.fill_(0.3)
        # FK must evaluate joints even when they are outside configured limits.
        fk.lower.fill_(-0.1)
        fk.upper.fill_(0.1)
    observation = torch.tensor(
        [[-0.2, 0.3], [0.4, -0.1]], dtype=torch.float64, device=device
    )
    with torch.no_grad():
        joints = predictor(observation)
        reference, _ = fk.evaluate(joints)
        target = fk(joints + 0.04)

    def unexpected_jacobian(*args, **kwargs):
        raise AssertionError("FK-only calls must not construct a geometric Jacobian")

    monkeypatch.setattr(fk.model.chain, "jacobian_tensor", unexpected_jacobian)
    network = torch.nn.Sequential(predictor, fk)
    pose = network(observation)
    torch.testing.assert_close(pose, reference)
    position, rotation = pose_error(pose, target)
    loss = (position.square().sum(-1) + rotation.square().sum(-1)).mean()
    loss.backward()
    for parameter in predictor.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all() and parameter.grad.norm() > 0
    assert not list(fk.parameters())


def test_fk_repeated_calls_preserve_outputs_and_batch_gradients():
    fk = make_fk()
    q = torch.linspace(-0.2, 0.3, 14, dtype=torch.float64).reshape(2, 7)
    q.requires_grad_()
    original = q.detach().clone()
    first = fk(q)
    snapshot = first.detach().clone()
    second = fk(q + 0.02)
    other_grad = torch.autograd.grad(first[0].sum(), q, retain_graph=True)[0]
    torch.testing.assert_close(other_grad[1], torch.zeros_like(q[1]))
    actual = torch.autograd.grad(first.sum() + second.sum(), q)[0]
    expected = torch.autograd.grad(
        fk.evaluate(q)[0].sum() + fk.evaluate(q + 0.02)[0].sum(), q
    )[0]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(first, snapshot)
    torch.testing.assert_close(q, original)
