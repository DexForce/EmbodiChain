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

"""Exercise process renderer lifetime and late Trainer cameras on real DexSim."""

from __future__ import annotations

import gc
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.subprocess_sim
@pytest.mark.gpu
@pytest.mark.parametrize(
    "scenario,task",
    [
        ("no-render-first", "go1_flat"),
        ("camera-first", "go1_flat"),
        ("explicit-no-render", "g1_flat"),
    ]
    + [
        ("trainer-recording", task)
        for task in (
            "go1_flat",
            "go2_flat",
            "g1_flat",
            "h1_2_flat",
            "anymal_c_flat",
            "microduck_flat",
        )
    ],
)
def test_auto_renderer_process_lifecycle(
    scenario: str, task: str, tmp_path: Path
) -> None:
    report = tmp_path / "renderer.json"
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), scenario, task, str(report)],
        cwd=Path(__file__).resolve().parents[3],
        env={**os.environ, "EMBODICHAIN_SIM_EXIT_PROCESS": "0"},
        text=True,
        capture_output=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert report.is_file(), result.stdout + result.stderr
    result = json.loads(report.read_text())
    assert result["scenario"] == scenario
    assert result["task"] == task


def _run_scenario(scenario: str, task: str, report: Path) -> None:
    import torch
    from embodichain.lab.gym.envs.managers.cfg import EventCfg
    from embodichain.lab.gym.envs.managers.event_manager import EventManager
    from embodichain.lab.gym.envs.managers.record import record_camera_data
    from embodichain.lab.gym.utils.gym_utils import config_to_cfg, get_manager_modules
    from embodichain.lab.gym.utils.registration import build_env
    from embodichain.lab.sim import SimulationManager, cfg as sim_cfg
    from embodichain.lab.sim.sensors import CameraCfg
    from embodichain.learning.rl.runtime import _build_gym_environment
    from embodichain.utils.utility import load_config

    importlib.import_module(f"embodichain_tasks.locomotion.velocity.{task}")
    sim_cfg.DEFAULT_RENDERER = "auto"
    task_path = Path(
        f"embodichain_tasks/configs/tasks/locomotion/velocity/{task}/env.newton.yaml"
    )
    raw = load_config(task_path)
    agent = load_config(task_path.parent / "agents/ppo.newton.yaml")

    def configuration(camera: bool):
        config = config_to_cfg(raw, manager_modules=get_manager_modules())
        config.num_envs = 1
        config.sim_cfg.startup_summary = "off"
        if camera:
            config.sensor.append(CameraCfg(uid="camera", width=16, height=16))
        return config

    env = runtime = manager = None
    try:
        if scenario == "trainer-recording":
            event = EventCfg(
                func=record_camera_data,
                mode="interval",
                params={
                    "name": "trainer-camera",
                    "resolution": (16, 16),
                    "eye": (2, 2, 2),
                    "target": (0, 0, 0.3),
                    "save_path": str(report.parent / "videos"),
                },
            )
            runtime = _build_gym_environment(
                {"trainer": {"gym_config": str(task_path)}},
                simulation_device=torch.device("cuda:0"),
                num_envs=2,
                headless=True,
                renderer=None,
                gpu_id=0,
                trainer_events=(event,),
            )
            env = runtime.env.unwrapped
            assert env.sim.has_native_renderer
            # Trainer initializes the same EventManager after its runtime exists.
            manager = EventManager({"record": event}, env)
            # Include the late recording cameras when budgeting the real PPO
            # batch, rather than only checking one environment can initialize.
            import math

            native_env = env.sim.get_world().get_env()
            nodes_per_env = math.ceil(
                native_env.get_active_scene_node_count() / env.num_envs
            )
            required = nodes_per_env * agent["trainer"]["num_envs"]
            assert required <= native_env.get_scene_node_capacity(), (
                task,
                native_env.get_active_scene_node_count(),
                required,
                native_env.get_scene_node_capacity(),
            )
            assert agent["trainer"]["num_envs"] == 4096
            env.sim.get_sensor("trainer-camera").update()
            assert env.sim.get_sensor("trainer-camera").get_data()["color"].shape[
                -3:-1
            ] == (16, 16)
        else:
            config = configuration(scenario == "camera-first")
            if scenario == "explicit-no-render":
                config.sim_cfg.render_cfg.renderer = "no-render"
            env = build_env(raw["id"], base_env_cfg=config).unwrapped
            if scenario == "explicit-no-render":
                assert env.sim.get_world().get_env().get_scene_node_capacity() == 131071
                assert env.sim.sim_config.scene_node_capacity == 262144
                env.reset()
                env.step(torch.zeros(env.action_space.shape, device=env.device))
            if scenario == "camera-first":
                assert env.sim.has_native_renderer
                env.sim.get_sensor("camera").update()
                assert env.sim.get_sensor("camera").get_data()["color"].shape[
                    -3:-1
                ] == (16, 16)
            else:
                assert not env.sim.has_native_renderer
            initial_renderer = env.sim.sim_config.render_cfg.renderer
            env.close(exit_process=False)
            env = None
            gc.collect()
            SimulationManager.flush_cleanup_queue()
            assert not SimulationManager._instances
            if scenario == "no-render-first":
                with pytest.raises(RuntimeError, match="Start a new process"):
                    build_env(raw["id"], base_env_cfg=configuration(True))
                assert not SimulationManager._instances
            elif scenario == "camera-first":
                env = build_env(raw["id"], base_env_cfg=configuration(False)).unwrapped
                assert env.sim.has_native_renderer
                assert env.sim.sim_config.render_cfg.renderer == initial_renderer
        report.write_text(
            json.dumps({"scenario": scenario, "task": task}), encoding="utf-8"
        )
    finally:
        if manager is not None:
            manager.get_functor_cfg("record").func.finalize()
        if env is not None:
            env.close(exit_process=False)
        env = runtime = manager = None
        gc.collect()
        SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    _run_scenario(sys.argv[1], sys.argv[2], Path(sys.argv[3]))
