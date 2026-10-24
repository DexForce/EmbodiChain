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
from embodichain.compute.dynamics import reference_task_values


def test_cartpole_observation_and_official_cost_definition() -> None:
    q = torch.tensor([[1.0, 0.2], [-0.7, 2 * torch.pi + 0.2]], dtype=torch.double)
    qd = torch.tensor([[0.3, -0.4], [0.1, 0.2]], dtype=torch.double)
    obs, reward, failed = reference_task_values("cartpole", q, qd)
    torch.testing.assert_close(obs[:, 1], torch.full((2,), 0.2, dtype=torch.double))
    expected = -(
        torch.full((2,), 0.2**2, dtype=torch.double)
        + 0.05 * q[:, 0] ** 2
        + 0.1 * (qd[:, 0] ** 2 + qd[:, 1] ** 2)
    )
    torch.testing.assert_close(reward, expected)
    assert obs.shape == (2, 4) and not failed.any()
    assert torch.autograd.gradcheck(
        lambda a, b: reference_task_values("cartpole", a, b)[:2],
        (q.requires_grad_(), qd.requires_grad_()),
    )


def test_ant_com_velocity_forward_reward_and_translation_invariance() -> None:
    q = torch.zeros(2, 15, dtype=torch.double)
    q[:, 2] = 1.0
    angle = torch.tensor([0.2, -0.4], dtype=torch.double)
    q[:, 5] = torch.sin(angle / 2)
    q[:, 6] = torch.cos(angle / 2)
    qd = torch.zeros(2, 14, dtype=torch.double)
    qd[:, 1] = torch.tensor([0.3, 0.7], dtype=torch.double)
    obs, reward, failed = reference_task_values("ant", q, qd)
    torch.testing.assert_close(reward, qd[:, 1] + 0.1 + torch.cos(angle))
    moved = q.clone()
    moved[:, :2] += torch.tensor([100.0, -100.0], dtype=torch.double)
    changed = reference_task_values("ant", moved, qd)
    torch.testing.assert_close(changed[0], obs)
    torch.testing.assert_close(changed[1], reward)
    assert obs.shape == (2, 27) and not failed.any()


def test_reference_failures_are_explicit_and_shapes_checked() -> None:
    q = torch.zeros(2, 15)
    q[:, 2] = torch.tensor([0.2, 1.0])
    q[0, 6] = 1.0
    _, _, failed = reference_task_values("ant", q, torch.zeros(2, 14))
    assert failed.tolist() == [True, True]  # low torso / invalid quaternion
    _, _, failed = reference_task_values(
        "cartpole", torch.zeros(1, 2), torch.tensor([[float("nan"), 0.0]])
    )
    assert failed.item()
    with pytest.raises(ValueError, match="dimensions"):
        reference_task_values("ant", torch.zeros(2, 2), torch.zeros(2, 2))
