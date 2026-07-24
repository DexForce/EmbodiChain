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

import gc

import gymnasium as gym
import numpy as np
import pytest
import torch

from embodichain.data import get_data_path
from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.utils.registration import register_env
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import JointDrivePropertiesCfg, RigidObjectCfg, RobotCfg
from embodichain.lab.sim.shapes import CubeCfg


@register_env("ReplayTest-v1", max_episode_steps=100, override=True)
class ReplayTestEnv(EmbodiedEnv):
    """UR10 + a dynamic rigid cube, used for record/replay integration tests."""

    def __init__(
        self,
        record_trajectory: bool = True,
        num_envs: int = 2,
        device: str = "cpu",
        **kwargs,
    ):
        cfg = EmbodiedEnvCfg()
        cfg.num_envs = num_envs
        cfg.max_episode_steps = 100
        cfg.sim_cfg = SimulationManagerCfg(headless=True, sim_device=device)
        cfg.robot = RobotCfg(
            uid="UR10",
            fpath=get_data_path("UniversalRobots/UR10/UR10.urdf"),
            init_pos=(0.0, 0.0, 1.0),
            drive_pros=JointDrivePropertiesCfg(drive_type="force"),
        )
        cfg.rigid_object = [
            RigidObjectCfg(
                uid="cube",
                shape=CubeCfg(size=[0.03, 0.03, 0.03]),
                init_pos=(0.0, 0.0, 0.5),
                body_type="dynamic",
            )
        ]
        cfg.record_trajectory = record_trajectory
        cfg.init_rollout_buffer = True
        super().__init__(cfg, **kwargs)


def _drive(env, num_steps: int = 5) -> list:
    """Step env with a smooth sinusoidal action list; return the actions."""
    init_qpos = env.robot.get_qpos()
    actions = []
    for i in range(num_steps):
        t = i / max(num_steps - 1, 1)
        offset = torch.zeros_like(init_qpos)
        offset[:, 0] = torch.sin(torch.tensor(t * 2.0 * np.pi)) * 0.2
        actions.append(init_qpos + offset)
    for a in actions:
        env.step(a)
    return actions


def test_record_trajectory_populates_states():
    env = ReplayTestEnv(record_trajectory=True, num_envs=2, device="cpu")
    try:
        env.reset()
        _drive(env, num_steps=5)
        assert "states" in env.rollout_buffer.keys()
        assert env.current_rollout_step == 5
        states = env.rollout_buffer["states"]
        assert tuple(states["robot"]["qpos"].shape) == (2, 100, 6)
        assert tuple(states["rigid_objects"]["cube"]["pose"].shape) == (2, 100, 7)
        # The last recorded step must reflect the actual robot qpos after driving.
        recorded = states["robot"]["qpos"][:, env.current_rollout_step - 1]
        actual = env.robot.get_qpos()
        assert torch.allclose(recorded, actual, atol=1e-5)
    finally:
        env.close()
        SimulationManager.flush_cleanup_queue()
        gc.collect()
