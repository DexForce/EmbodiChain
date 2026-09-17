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
import torch
from embodichain.lab.gym.envs.managers import EventManager, EventCfg, SceneEntityCfg
from embodichain.lab.gym.envs.managers.randomization.physics import (
    push_articulation_by_setting_velocity,
)


def make_manager():
    velocity = torch.zeros(3, 6)
    asset = SimpleNamespace(body_data=SimpleNamespace(root_vel=velocity))

    def set_velocity(value, env_ids):
        velocity[env_ids] = value

    asset.set_root_velocity = set_velocity
    env = SimpleNamespace(
        num_envs=3,
        device=torch.device("cpu"),
        step_dt=0.1,
        cfg=SimpleNamespace(seed=42),
        sim=SimpleNamespace(asset_uids=["robot"], get_robot=lambda uid: asset),
    )
    cfg = EventCfg(
        func=push_articulation_by_setting_velocity,
        mode="interval",
        interval_step=1,
        params=dict(
            entity_cfg=SceneEntityCfg(uid="robot"),
            interval_range_s=[0.1, 0.3],
            velocity_range={
                axis: [-0.2, 0.2] for axis in ("x", "y", "z", "roll", "pitch", "yaw")
            },
        ),
    )
    return EventManager({"push": cfg}, env), velocity


def test_disturbance_seed_replays_root_velocity_sequence():
    manager, velocity = make_manager()
    results = []
    for _ in range(2):
        manager.set_seed(19)
        manager.reset()
        velocity.zero_()
        for _ in range(8):
            manager.apply(mode="interval")
        results.append(velocity.clone())
    assert results[0].abs().sum() > 0
    assert torch.equal(*results)


def test_selective_reset_keeps_other_disturbance_timers_and_velocities():
    manager, velocity = make_manager()
    manager.apply(mode="interval")
    term = manager.get_functor_cfg("push").func
    before = term._steps_remaining[[0, 2]].clone()
    counter = manager._interval_functor_step_count[0][[0, 2]].clone()
    state = velocity.clone()
    manager.reset(env_ids=[1])
    assert torch.equal(term._steps_remaining[[0, 2]], before)
    assert torch.equal(manager._interval_functor_step_count[0][[0, 2]], counter)
    assert torch.equal(velocity, state)
