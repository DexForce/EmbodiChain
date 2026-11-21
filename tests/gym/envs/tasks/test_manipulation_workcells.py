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

"""Check measured acceptance and native reset for the two manipulation fixtures."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import subprocess
import sys

import pytest
import torch

from embodichain.lab.gym.utils.gym_utils import config_to_cfg
from embodichain.lab.gym.utils.registration import make
from embodichain.utils.utility import load_config


@contextmanager
def _workcell(name: str, num_envs: int):
    import embodichain.extensions.tasks  # Register handwritten task environments.

    source = (
        Path(__file__).resolve().parents[4]
        / "embodichain/extensions/tasks/configs/tasks/manipulation"
        / name
        / "task.franka.yaml"
    )
    payload = load_config(source)
    payload.update(
        headless=True, device="cpu", renderer="no-render", enable_sensor=False
    )
    cfg = config_to_cfg(payload, source_path=source)
    cfg.num_envs = num_envs
    env = make(payload["id"], cfg=cfg)
    try:
        env.reset(seed=0, options={"save_data": False})
        yield env
    finally:
        env.close(exit_process=False)


def _check_microwave_door():
    with _workcell("robocasa365_open_microwave", num_envs=2) as env:
        fixture = env.sim.get_articulation("microwave")
        joint = fixture.joint_names.index("microwave_right_group_1_microjoint")
        assert not env.is_task_success().any()
        button = fixture.joint_names.index("microwave_right_group_1_Button001_joint")
        fixture.set_qpos(torch.full((2, 1), 0.003), joint_ids=[button], target=False)
        assert not env.is_task_success().any()
        # 80 and 85 degrees straddle RoboCasa's 81 degree opening threshold.
        positions = torch.deg2rad(torch.tensor([[-80.0], [-85.0]]))
        fixture.set_qpos(positions, joint_ids=[joint], target=False)
        assert env.is_task_success().tolist() == [False, True]
        # Use the closed stop for the negative rollout. Near-threshold doors
        # can continue moving during the following four physical substeps.
        fixture.set_qpos(
            torch.zeros(1, 1), joint_ids=[joint], env_ids=[0], target=False
        )
        action = torch.zeros(env.num_envs, env.action_space.shape[-1])
        _, reward, _, _, _ = env.step(action)
        assert reward.tolist() == [0.0, 1.0]
        env.reset(seed=0, options={"save_data": False})
        _, reward, _, _, info = env.step(action)
        assert not info["success"].any()
        assert not reward.any()


def _check_drawer_storage():
    from embodichain.lab.gym.envs.task_program._final_acceptance import (
        _measure_final_conditions,
    )

    with _workcell("calvin_drawer_storage", num_envs=2) as env:

        def rule():
            return _measure_final_conditions(env, env._final_acceptance_rules).satisfied

        assert rule().tolist() == [False, False]

        drawer = env.sim.get_articulation("drawer")
        tray = drawer.get_link_pose("drawer_link", to_matrix=True)
        # Place only arena 0's red block inside. Arena 1 holds the blue distractor.
        local_center = torch.tensor([0.0, 0.045, 0.080, 1.0])
        for row, uid in enumerate(("block_red", "block_blue")):
            block = env.sim.get_rigid_object(uid)
            pose = block.get_local_pose(to_matrix=True)[row : row + 1].clone()
            pose[0, :3, 3] = (tray[row] @ local_center)[:3]
            block.set_local_pose(pose, env_ids=[row])
        for _ in range(12):
            env._advance_physics()
        assert rule().tolist() == [True, False]
        # Physical containment alone cannot complete the four-segment program.
        assert not env.is_task_success().any()
        env.reset(seed=0, options={"save_data": False})
        assert not rule().any()


def _check_drawer_episode():
    from embodichain.lab.gym.envs.demo import execute_demo_episode

    with _workcell("calvin_drawer_storage", num_envs=1) as env:
        result = execute_demo_episode(env)
        assert result.completed and result.all_success, result.to_metadata()


def _check_microwave_episode():
    from embodichain.lab.gym.envs.demo import execute_demo_episode

    with _workcell("robocasa365_open_microwave", num_envs=1) as env:
        assert not env.is_task_success().any()
        result = execute_demo_episode(env)
        assert result.completed and result.all_success, result.to_metadata()
        assert bool(env.is_task_success()[0])


@pytest.mark.requires_sim
@pytest.mark.subprocess_sim
@pytest.mark.parametrize(
    "check",
    [
        "_check_microwave_door",
        "_check_drawer_storage",
        pytest.param("_check_drawer_episode", marks=pytest.mark.slow),
        pytest.param("_check_microwave_episode", marks=pytest.mark.slow),
    ],
)
def test_native_workcell(check):
    path = Path(__file__).resolve()
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy, sys; runpy.run_path(sys.argv[1])[sys.argv[2]]()",
            str(path),
            check,
        ],
        cwd=path.parents[4],
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert result.returncode == 0, result.stdout + result.stderr
