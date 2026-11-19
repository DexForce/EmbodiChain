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

"""Analytic gravity feedforward without an environment or simulator."""

from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch

from embodichain.compute.kinematics import URDFGravityCompensation

_URDF = """<robot name="pendulum"><link name="base"/><link name="load">
<inertial><origin xyz="0.4 0 0"/><mass value="2"/>
<inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial></link>
<joint name="axis" type="continuous"><parent link="base"/><child link="load"/>
<axis xyz="0 0 1"/></joint></robot>"""


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "mode",
    [nullcontext, torch.no_grad, torch.inference_mode],
    ids=["grad", "no_grad", "inference"],
)
def test_gravity_torques_match_pendulum_potential_derivative(
    dtype: torch.dtype,
    mode,
) -> None:
    with mode():
        gravity = torch.tensor([0, -9.81, 0], dtype=dtype)
        model = URDFGravityCompensation(_URDF, gravity=gravity, dtype=dtype)
        q = torch.tensor(
            [[0.0], [torch.pi / 2], [torch.pi]], dtype=dtype, requires_grad=True
        )
        actual = model.torques(q)
    # U = m*g*l*sin(q), so holding torque is dU/dq = m*g*l*cos(q).
    expected = 2 * 9.81 * 0.4 * torch.cos(q)
    assert model.joint_names == ("axis",)
    assert not actual.requires_grad and q.grad is None
    torch.testing.assert_close(actual, expected.detach())


def test_gravity_empty_batch_and_massless_body_produce_owned_zero_torques() -> None:
    model = URDFGravityCompensation(_URDF.replace('mass value="2"', 'mass value="0"'))
    assert model.torques(torch.empty(0, 1)).shape == (0, 1)
    q = torch.tensor([[0.4], [-0.7]])
    torch.testing.assert_close(model.torques(q), torch.zeros_like(q))


def test_gravity_requires_encoder_coordinates_in_declared_model_layout() -> None:
    model = URDFGravityCompensation(_URDF)
    with pytest.raises(ValueError, match="shape"):
        model.torques(torch.zeros(2, 2))
    with pytest.raises(ValueError, match="dtype/device"):
        model.torques(torch.zeros(2, 1, dtype=torch.float64))
