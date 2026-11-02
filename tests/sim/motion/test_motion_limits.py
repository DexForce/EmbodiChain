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

"""Positive request resolution independent of neural planners or models."""

from __future__ import annotations

from types import SimpleNamespace
import pytest
import torch
from embodichain.lab.sim.motion.limits import parse_motion_limits, resolve_motion_limits


@pytest.mark.no_sim
@pytest.mark.parametrize(
    "limit_request",
    [
        {"velocity": 1.0},
        {"velocity": 0.0, "acceleration": 2.0},
        {"velocity": -1.0, "acceleration": 2.0},
        {"velocity": float("inf"), "acceleration": 2.0},
        {"velocity": [[1.0, 1.0]], "acceleration": 2.0},
        {"velocity": [True, 1.0], "acceleration": 2.0},
    ],
)
def test_positive_request_contract_rejects_partial_frozen_signed_or_batched_limits(
    limit_request,
):

    with pytest.raises(ValueError):
        parse_motion_limits(limit_request, dof=2)


@pytest.mark.no_sim
def test_request_resolution_preserves_control_order_and_reliable_acceleration_caps():

    robot = SimpleNamespace(
        joint_names=["a", "b"],
        control_parts={"arm": ["b", "a"]},
        get_joint_ids=lambda name: [1, 0],
        get_qvel_limits=lambda joint_ids: torch.tensor([[0.5, 2.0], [0.4, 1.5]]),
    )
    r = resolve_motion_limits(
        robot,
        {"velocity": [3.0, 1.0], "acceleration": 4.0},
        dof=2,
        control_part="arm",
        batch_size=2,
        robot_acceleration_limits=torch.tensor([3.0, 2.0]),
    )
    torch.testing.assert_close(r.velocity, torch.tensor([1.5, 0.4]))
    torch.testing.assert_close(r.acceleration, torch.tensor([3.0, 2.0]))
    assert r.metadata()["requested_velocity"] == [3.0, 1.0]
