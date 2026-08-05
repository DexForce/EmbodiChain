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

"""End-to-end check that the profiler records sections through a real
BaseEnv step/reset loop (enabled path)."""

from __future__ import annotations

import gc

import gymnasium as gym
import numpy as np
import pytest
import torch

from embodichain.data import get_data_path
from embodichain.lab.gym.envs import BaseEnv, EnvCfg
from embodichain.lab.gym.utils.profiler import EnvProfilerCfg
from embodichain.lab.gym.utils.registration import register_env
from embodichain.lab.sim import SimulationManagerCfg
from embodichain.lab.sim.cfg import RobotCfg
from embodichain.lab.sim.objects import Robot


@register_env("ProfilerProbe-v1", max_episode_steps=100, override=True)
class _ProfilerProbeEnv(BaseEnv):
    """Minimal BaseEnv with profiling enabled, for integration testing."""

    def __init__(self, device: str = "cpu", **kwargs):
        cfg = EnvCfg(
            sim_cfg=SimulationManagerCfg(
                headless=True, arena_space=2.0, sim_device=device
            ),
            num_envs=2,
            profiler=EnvProfilerCfg(enable_time=True, warmup_steps=0),
        )
        super().__init__(cfg=cfg, **kwargs)

    def _setup_robot(self, **kwargs) -> Robot:
        file_path = get_data_path("UniversalRobots/UR10/UR10.urdf")
        robot: Robot = self.sim.add_robot(
            cfg=RobotCfg(uid="UR10", fpath=file_path, init_pos=(0, 0, 1))
        )
        qpos_limits = robot.body_data.qpos_limits[0].cpu().numpy()
        self.single_action_space = gym.spaces.Box(
            low=qpos_limits[:, 0], high=qpos_limits[:, 1], dtype=np.float32
        )
        return robot

    def _step_action(self, action):
        self.robot.set_qpos(qpos=action)
        return action


@pytest.mark.skipif(
    not get_data_path("UniversalRobots/UR10/UR10.urdf"),
    reason="UR10 asset not available",
)
class TestProfilerIntegration:
    def test_records_step_reset_sections(self):
        env = _ProfilerProbeEnv(device="cpu")
        try:
            assert env._profiler is env.sim.profiler
            obs, info = env.reset()
            action = torch.as_tensor(
                env.action_space.sample(),
                dtype=torch.float32,
                device=env.device,
            )
            obs, reward, done, truncated, info = env.step(action)

            stats = env._profiler._stats

            # step pipeline
            assert "step" in stats and stats["step"].n == 1
            assert "step.preprocess_action" in stats
            assert "step.step_action" in stats
            assert "step.sim_update" in stats
            assert "step.update_sim_state" in stats
            assert "step.get_obs" in stats
            assert "step.get_obs.proprio" in stats
            assert "step.get_obs.sensor" in stats
            assert "step.get_obs.sensor.render_camera_group" in stats
            assert "step.get_obs.sensor.sensor_fetch" in stats
            assert "step.get_obs.extend" in stats
            assert "step.reward" in stats
            assert "step.hook_after" in stats

            # reset pipeline
            assert "reset" in stats and stats["reset"].n == 1
            assert "reset.is_task_success" in stats
            assert "reset.reset_objects_state" in stats
            assert "reset.initialize_episode" in stats
            assert "reset.get_obs" in stats

            # report() runs without error and reflects recorded sections
            data = env._profiler.report()
            assert "step.sim_update" in data["sections"]
            assert data["sections"]["step"]["calls"] == 1
        finally:
            env.close()
            import embodichain.lab.sim as om

            om.SimulationManager.flush_cleanup_queue()
            gc.collect()
