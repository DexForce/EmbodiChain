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

"""Validate runnable locomotion configurations without patching their fields."""

from __future__ import annotations

from pathlib import Path
import faulthandler
import gc
import importlib
import json
import os
import subprocess
import sys

import pytest

from embodichain.lab.gym.utils.gym_utils import config_to_cfg
from embodichain.lab.sim import cfg as sim_cfg
from embodichain.utils.utility import load_config

_TASKS = (
    "locomotion/velocity/g1_flat",
    "locomotion/velocity/h1_2_flat",
    "locomotion/velocity/go1_flat",
    "locomotion/velocity/go2_flat",
    "locomotion/velocity/anymal_c_flat",
    "locomotion/velocity/microduck_flat",
    "classic_control/humanoid",
)


@pytest.mark.no_sim
@pytest.mark.parametrize("task", _TASKS)
@pytest.mark.parametrize("backend", ("default", "newton"))
def test_packaged_deployment_and_agent_config_agree(
    task: str, backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real config decoding must work without smoke-script backend overrides."""
    monkeypatch.setattr(sim_cfg, "get_data_path", lambda value: value)
    directory = Path("embodichain_tasks/configs/tasks") / task
    suffix = "" if backend == "default" else ".newton"
    path = directory / f"env{suffix}.yaml"
    raw = load_config(str(path))
    agent = load_config(str(directory / f"agents/ppo{suffix}.yaml"))
    assert raw["physics"] == backend
    assert "physics_config_by_backend" not in raw
    cfg = config_to_cfg(
        raw,
        source_path=path,
        manager_modules=[
            f"embodichain_tasks.locomotion.managers.{module}"
            for module in ("observations", "rewards")
        ],
    )
    physics_type = (
        sim_cfg.DefaultPhysicsCfg if backend == "default" else sim_cfg.NewtonPhysicsCfg
    )
    assert isinstance(cfg.sim_cfg.physics_cfg, physics_type)
    assert cfg.sim_cfg.render_cfg.renderer in {
        "auto",
        "hybrid",
        "fast-rt",
        "rt",
        "no-render",
    }
    if task.startswith("locomotion/"):
        assert "renderer" not in raw
        assert "renderer" not in raw.get("render_cfg", {})
        assert "renderer" not in agent["trainer"]
        assert cfg.sim_cfg.render_cfg.renderer == "auto"
    else:
        assert agent["trainer"]["renderer"] == cfg.sim_cfg.render_cfg.renderer
    assert Path(agent["trainer"]["gym_config"]).resolve() == path.resolve()
    assert set(agent) == {"trainer", "policy", "algorithm"}
    assert agent["policy"]["initial_action_std"] > 0


@pytest.mark.subprocess_sim
@pytest.mark.gpu
@pytest.mark.parametrize(
    "task", [task for task in _TASKS if task.startswith("locomotion/")]
)
@pytest.mark.parametrize("backend", ("default", "newton"))
def test_locomotion_runtime_uses_native_no_render(task: str, backend: str) -> None:
    """Packaged PPO runtime reaches the real NoRender engine without overrides."""
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), task, backend],
        cwd=Path(__file__).resolve().parents[4],
        env={**os.environ, "EMBODICHAIN_SIM_EXIT_PROCESS": "0"},
        text=True,
        capture_output=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = next(
        line
        for line in result.stdout.splitlines()
        if line.startswith("NORENDER_CHECK ")
    )
    print(report)


def _verify_native_no_render(task: str, backend: str) -> None:
    import torch
    from dexsim.types import Renderer
    from embodichain.lab.sim import SimulationManager
    from embodichain.learning.rl.runtime import _build_gym_environment

    importlib.import_module(
        f"embodichain_tasks.locomotion.velocity.{task.rsplit('/', 1)[-1]}"
    )
    directory = Path("embodichain_tasks/configs/tasks") / task
    suffix = "" if backend == "default" else ".newton"
    agent_path = directory / f"agents/ppo{suffix}.yaml"
    agent = load_config(agent_path)
    assert "renderer" not in agent["trainer"]
    sim_cfg.DEFAULT_RENDERER = "auto"
    runtime = env = sim = observation = None
    try:
        runtime = _build_gym_environment(
            agent,
            simulation_device=torch.device(agent["trainer"]["device"]),
            num_envs=1,
            headless=agent["trainer"]["headless"],
            renderer=agent["trainer"].get("renderer"),
            gpu_id=0,
            seed=agent["trainer"]["seed"],
            config_dir=agent_path.parent,
        )
        env = runtime.env.unwrapped
        sim = env.sim
        assert sim._requested_renderer == "auto"
        assert sim.sim_config.render_cfg.renderer == "no-render"
        assert sim.get_world().get_renderer_type() == Renderer.NORENDER
        assert not sim.get_world().is_window_initialized()
        assert not sim.has_native_renderer
        assert sim._sensors and all(
            sensor.cfg.sensor_type == "ContactSensor"
            for sensor in sim._sensors.values()
        )
        observation, _ = env.reset()
        for _ in range(2):
            observation, reward, _, _, _ = env.step(
                torch.zeros(env.action_space.shape, device=env.device)
            )
            assert torch.isfinite(reward).all()
            assert torch.isfinite(observation["policy"]).all()
        env.reset()
        print(
            "NORENDER_CHECK "
            + json.dumps(
                {
                    "task": task,
                    "physics": sim.physics.name,
                    "requested_renderer": sim._requested_renderer,
                    "native_renderer": sim.get_world().get_renderer_type().name,
                    "contact_sensors": len(sim._sensors),
                    "steps": 2,
                    "reset": "passed",
                }
            ),
            flush=True,
        )
    finally:
        if env is not None:
            env.close(exit_process=False)
        runtime = env = sim = observation = None
        gc.collect()
        SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    faulthandler.enable()
    faulthandler.dump_traceback_later(150)
    _verify_native_no_render(sys.argv[1], sys.argv[2])
