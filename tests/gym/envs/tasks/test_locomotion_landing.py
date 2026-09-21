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

from types import SimpleNamespace
import pytest
import torch
from embodichain.lab.sim.sensors.contact_history import ContactHistory
from embodichain_tasks.locomotion.velocity.contracts.anymal_c.config import load_config
from embodichain_tasks.locomotion.velocity.contracts.anymal_c.mdp import _reward_terms


def test_anymal_landing_reward_consumes_completed_flight_duration():
    config = load_config()
    history = ContactHistory(
        torch.tensor([[10, 11, 12, 13]]), counterpart_ids=torch.tensor([[0]])
    )
    data = dict(
        user_ids=torch.tensor([[[0, 10]]]),
        is_valid=torch.tensor([[False]]),
        normal=torch.tensor([[[0.0, 0.0, 1.0]]]),
        friction=torch.zeros(1, 1, 3),
        impulse=torch.ones(1, 1),
    )
    for _ in range(3):
        history.begin_control_step()
        history.update(data, 0.1)
    history.begin_control_step()
    data["is_valid"].fill_(True)
    history.update(data, 0.1)
    state = SimpleNamespace(
        command=torch.tensor([[1.0, 0.0, 0.0]]),
        base_lin_vel_b=torch.zeros(1, 3),
        base_ang_vel_b=torch.zeros(1, 3),
        illegal_contact_force_by_body=torch.zeros(1, 5),
        joint_pos=torch.zeros(1, 12),
        joint_vel=torch.zeros(1, 12),
        joint_acc=torch.zeros(1, 12),
        joint_torque=torch.zeros(1, 12),
        action=torch.zeros(1, 12),
        last_action=torch.zeros(1, 12),
        projected_gravity_b=torch.zeros(1, 3),
        foot_air_time=history.current_air_time,
        last_foot_air_time=history.last_air_time,
        first_foot_contact=history.first_contact,
    )
    reward = _reward_terms(config, state)["feet_air_time"].item()
    assert reward == pytest.approx(
        0.4 - config.data["rewards"]["feet_air_time"]["threshold"]
    )
    assert state.foot_air_time[0, 0] == 0
