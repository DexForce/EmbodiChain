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

"""Live Gym reset and exhaustion with a small synthetic source recording."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import torch
from tensordict import TensorDict
from embodichain.lab.gym.envs.fluid_replay import FluidReplayEnv
from embodichain.lab.gym.utils.gym_utils import config_to_cfg
from embodichain.lab.sim import SimulationManager
from embodichain.lab.sim.cfg import FluidSpawnCfg, FluidRenderCfg, RenderCfg
from embodichain.utils.utility import load_config

pytestmark = [pytest.mark.gpu, pytest.mark.subprocess_sim]


class _FluidReplayRuntime:
    def setup_method(self):
        from dexsim.engine.newton_physics.solver_runtime import ensure_solver_available

        if not getattr(ensure_solver_available("sph"), "supports_world_reset", False):
            pytest.skip("requires world-isolated SPH build")
        task = (
            Path(__file__).resolve().parents[3]
            / "embodichain_tasks/configs/tasks/manipulation/tableware/pour_cup/task.franka.fluid.yaml"
        )
        cfg = config_to_cfg(load_config(task), source_path=task)
        cfg.sim_cfg.render_cfg = RenderCfg(renderer="no-render")
        cfg.sim_cfg.headless = True
        cfg.enable_sensor = False
        cfg.events.record_camera = None
        cfg.sim_steps_per_control = 1
        cfg.sim_cfg.physics_dt = 0.001
        solver = cfg.sim_cfg.physics_cfg.solver_cfg
        solver.update(
            volume_map_boundary_enabled=False,
            sdf_boundary_enabled=False,
            particle_support_radius=0.0225,
        )
        for obj in cfg.background:
            obj.shape.collision.sdf_resolution = 16
        fluid = cfg.fluid_object[0]
        fluid.particle_spacing = 0.01
        fluid.particle_radius = 0.003
        fluid.spawn = FluidSpawnCfg(
            kind="box", size=(0.02, 0.02, 0.02), reference_uid="cup"
        )
        fluid.render = FluidRenderCfg(mode="none")
        root = torch.tensor([0, 0, 0, 0, 0, 0, 1.0]).repeat(1, 2, 1)
        cup = torch.tensor([-0.75, 0, 0.552, 0, 0, 0, 1.0]).repeat(1, 2, 1)
        qpos = torch.tensor(cfg.robot.init_qpos).reshape(1, 1, -1).repeat(1, 2, 1)
        # Synthetic provenance exercises the replay contract, not source acceptance.
        source = {
            "meta": {
                "num_envs": 1,
                "num_steps": 2,
                "lengths": [2],
                "dt": 0.004,
                "rigid_object_uids": ["cup"],
                "fluid_source": {
                    "kind": "task_program",
                    "accepted": True,
                    "program_hash": "synthetic-test",
                },
            },
            "states": TensorDict(
                {
                    "robot": {"qpos": qpos, "root_pose": root},
                    "rigid_objects": {"cup": {"pose": cup}},
                },
                batch_size=[1, 2],
            ),
            "actions": torch.zeros(1, 2, qpos.shape[-1]),
        }
        self.env = FluidReplayEnv(cfg, trajectory=source, speed=1.0, hold_seconds=0.0)

    def teardown_method(self):
        if hasattr(self, "env"):
            self.env.close(exit_process=False)
            del self.env
        SimulationManager.flush_cleanup_queue()

    def test_partial_reset_preserves_other_world_then_finishes_independently(self):
        env = self.env
        env.reset(seed=0)
        action = torch.zeros(
            (env.num_envs, len(env.active_joint_ids)), device=env.device
        )
        env.step(action)
        env.step(action)
        before = env.sim.get_fluid_object("water").data.particle_state_w.clone()
        env.reset(
            options={
                "reset_ids": torch.tensor([0], device=env.device),
                "save_data": False,
            }
        )
        torch.testing.assert_close(
            env.sim.get_fluid_object("water").data.particle_state_w[1],
            before[1],
            rtol=0,
            atol=0,
        )
        np.testing.assert_allclose(env._replay_time, [0, 0.002])
        env.step(action)
        _, _, _, truncated, info = env.step(action)
        assert info["fluid_replay_finished"].tolist() == [False, True]
        assert truncated.tolist() == [False, True]
        env.step(action)
        env.step(action)
        with pytest.raises(RuntimeError, match="exhausted"):
            env.step(action)


def test_fluid_replay_reset_and_exhaustion_in_fresh_process():
    from dexsim.engine.newton_physics.solver_runtime import ensure_solver_available

    if not getattr(ensure_solver_available("sph"), "supports_world_reset", False):
        pytest.skip("requires world-isolated SPH build")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve())],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    worker = _FluidReplayRuntime()
    try:
        worker.setup_method()
        worker.test_partial_reset_preserves_other_world_then_finishes_independently()
    finally:
        worker.teardown_method()
