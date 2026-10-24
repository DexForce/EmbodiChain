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

"""Registered neural tasks, file-owned configs and actual APG routing."""

from __future__ import annotations
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from embodichain.lab.gym.utils.gym_utils import config_to_cfg
from embodichain.lab.gym.utils.registration import REGISTERED_ENVS, build_env
from embodichain.learning.rl.env import (
    get_registered_learning_env_names,
    build_learning_env,
)
from embodichain.utils.utility import load_config

_ROOT = Path(__file__).resolve().parents[4]
_CONFIGS = _ROOT / "extensions/embodichain_tasks/configs/tasks/classic_control"


@pytest.mark.parametrize("robot,title", [("cartpole", "Cartpole"), ("ant", "Ant")])
def test_registered_task_and_yaml_ownership(robot: str, title: str) -> None:
    importlib.import_module(f"embodichain_tasks.classic_control.{robot}")
    assert f"{title}Dynamics-v1" in REGISTERED_ENVS
    assert f"{title.lower()}dynamics-v1" in get_registered_learning_env_names()
    path = _CONFIGS / robot / "env.nerd.yaml"
    config = load_config(path)
    parsed = config_to_cfg(config, source_path=path)
    assert parsed.sim_steps_per_control == 1
    assert parsed.robot is None
    assert parsed.sim_cfg.enable_ground_plane is (robot == "ant")
    assert parsed.sim_cfg.physics_cfg.requires_grad
    assert parsed.sim_cfg.physics_cfg.solver_cfg["solver_type"] == "nerd"
    agent = load_config(_CONFIGS / robot / "agents/apg.nerd.yaml")
    assert agent["trainer"]["learning_env"]["name"] == f"{title}Dynamics-v1"
    assert "gym_config" not in agent["trainer"]
    assert agent["algorithm"]["name"] == "apg"


def _check_task_and_trainer(tmp_path: Path, robot: str, title: str) -> None:
    importlib.import_module(f"embodichain_tasks.classic_control.{robot}")
    from embodichain.lab.sim.physics.solvers.collection import DynamicsSimulation
    from embodichain.learning.rl.train import train_from_config

    helper_path = _ROOT / "tests/sim/physics/solvers/test_nerd_solver.py"
    spec = importlib.util.spec_from_file_location("_nerd_export_helper", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    with DynamicsSimulation(
        robot,
        num_envs=2,
        device="cpu",
        num_substeps=1,
    ) as teacher:
        checkpoint = helper._small_export(
            teacher.sim.spawn_result, tmp_path / "model", robot
        )
    deployment_path = _CONFIGS / robot / "env.nerd.yaml"
    deployment = load_config(deployment_path)
    deployment.update(num_envs=2, max_episode_steps=4, renderer="no-render")
    deployment["physics_config"]["solver_cfg"]["options"]["checkpoint"] = str(
        checkpoint
    )
    env = build_env(
        deployment["id"], config_to_cfg(deployment, source_path=deployment_path)
    )
    try:
        obs1, _ = env.reset(seed=9)
        saved = obs1.detach().clone()
        obs2, _ = env.reset(seed=9)
        torch.testing.assert_close(saved, obs2)
        width = 1 if robot == "cartpole" else 8
        observation_width = 4 if robot == "cartpole" else 27
        assert obs2.shape == (2, observation_width)
        action = torch.full((2, width), 0.1, requires_grad=True)
        obs, reward, terminated, truncated, info = env.step(action)
        assert reward.shape == (2,) and obs.shape == (2, observation_width)
        (reward.sum() + 0.001 * obs.square().sum()).backward()
        assert (
            action.grad is not None
            and torch.isfinite(action.grad).all()
            and action.grad.abs().sum() > 0
        )
        assert not env.detach_state().requires_grad
        # Selected episode reset clears only selected history, uses a continuing
        # named RNG stream, and keeps the other row's physical state unchanged.
        before = env.functional_state
        unchanged = (
            before.physical["joint_q"][env._binding.q_indices[1]].detach().clone()
        )
        env.reset(options={"reset_ids": torch.tensor([0])})
        assert env.functional_state.valid_history_lengths.tolist() == [0, 1]
        torch.testing.assert_close(
            env.functional_state.physical["joint_q"][env._binding.q_indices[1]],
            unchanged,
        )
    finally:
        env.close()
    cfg = load_config(_CONFIGS / robot / "agents/apg.nerd.yaml")
    cfg["trainer"].update(
        device="cpu",
        num_envs=2,
        iterations=1,
        segment_length=2,
        update_horizon=2,
        enable_eval=False,
        use_wandb=False,
        save_frequency_updates=0,
    )
    cfg["trainer"]["learning_env"]["cfg"].update(
        checkpoint=str(checkpoint), max_episode_steps=4, random_reset=False
    )
    cfg["policy"]["actor"]["network_cfg"] = {
        "hidden_sizes": [8],
        "activation": "tanh",
        "last_activation": "tanh",
    }
    path = tmp_path / "apg.json"
    path.write_text(json.dumps(cfg))
    os.chdir(tmp_path)
    result = train_from_config(path)
    assert result is not None
    assert result["global_step"] == 4
    assert list((tmp_path / "outputs").rglob("*.pt"))


@pytest.mark.subprocess_sim
@pytest.mark.parametrize("robot,title", [("cartpole", "Cartpole"), ("ant", "Ant")])
def test_actual_registered_task_and_one_apg_update(
    tmp_path: Path, robot: str, title: str
) -> None:
    code = "import importlib.util,sys;from pathlib import Path;s=importlib.util.spec_from_file_location('neural_task_test',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m._check_task_and_trainer(Path(sys.argv[2]),sys.argv[3],sys.argv[4])"
    subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(Path(__file__).resolve()),
            str(tmp_path),
            robot,
            title,
        ],
        check=True,
        env={
            **os.environ,
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "PYTHONPATH": str(_ROOT),
            "EMBODICHAIN_SIM_EXIT_PROCESS": "0",
        },
    )
