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

"""Regression tests for the PolaRiS DROID task compositions and measured rubrics."""

from pathlib import Path

import gymnasium as gym
import pytest
import torch
import yaml

from embodichain.extensions.tasks.manipulation.polaris._env import (
    PLACEMENTS,
    PolarisRubric,
)
from embodichain.lab.gym.envs import execute_demo_episode
from embodichain.lab.gym.utils.gym_utils import config_to_cfg
from embodichain.lab.gym.utils.registration import REGISTERED_ENVS

ROOT = (
    Path(__file__).resolve().parents[4]
    / "embodichain/extensions/tasks/configs/tasks/manipulation/polaris"
)


@pytest.mark.parametrize("scene", PLACEMENTS)
def test_official_deployment_composes_and_registers(scene):
    folder = ROOT / scene
    manifest = yaml.safe_load((folder / "catalog.yaml").read_text())
    assert manifest["task_key"] == f"polaris_{scene}"
    path = folder / manifest["deployments"]["droid"]["config"]
    deployment = yaml.safe_load(path.read_text())
    cfg = config_to_cfg(deployment, source_path=path)
    assert REGISTERED_ENVS[deployment["id"]].cls.scene == scene
    assert gym.spec(deployment["id"]).id == deployment["id"]
    names = {obj.uid for obj in cfg.rigid_object}
    assert names >= {
        name for obj, target, _ in PLACEMENTS[scene] for name in (obj, target)
    }
    assert cfg.target_control_frequency == 15
    expert = yaml.safe_load(
        (folder / manifest["deployments"]["expert"]["config"]).read_text()
    )
    spec = REGISTERED_ENVS[expert["id"]]
    assert spec.supports_handwritten_demo
    assert spec.cls is REGISTERED_ENVS[deployment["id"]].cls
    assert spec.default_kwargs["expert_initial_condition"] == 0
    for grasp in cfg.extensions["expert"]:
        assert {grasp["object"], grasp["target"]} <= names


def test_single_object_order_release_and_partial_reset():
    state = PolarisRubric("move_latte_cup", 2, torch.device("cpu"))
    pos = torch.zeros((2, 3))
    bounds = torch.tensor([[[0.0, 0.0, 0.05], [1.0, 1.0, 0.12]]]).repeat(2, 1, 1)
    positions = {"latte_art_cup": pos}
    boxes = {"latte_art_cup": bounds, "cutting_board": bounds.clone()}

    # A high, centered cup cannot score before the arm comes within 20 cm.
    state.update(pos + 0.3, torch.zeros(2), positions, boxes)
    assert not state.reached.any()
    boxes["cutting_board"][:, :, 0] += 0.21
    state.update(pos, torch.zeros(2), positions, boxes)
    assert state.reached.tolist() == [[True, True, False]] * 2
    boxes["cutting_board"] = bounds.clone()
    state.update(pos, torch.tensor([0.1, 0.09]), positions, boxes)
    assert state.reached.tolist() == [[True, True, False], [True, True, True]]
    boxes["cutting_board"][:, :, 0] += 2.0
    state.update(pos + 1.0, torch.ones(2), positions, boxes)
    assert state.reached[1].all()
    state.reset(torch.tensor([1]))
    assert state.reached.tolist() == [[True, True, False], [False, False, False]]


def test_kitchen_requires_both_placements_before_green_over_wood():
    state = PolarisRubric("block_stack_kitchen", 1, torch.device("cpu"))
    bounds = torch.tensor([[[0.0, 0.0, 0.04], [1.0, 1.0, 0.10]]])
    pos = torch.zeros((1, 3))
    positions = {"green_cube": pos, "wood_cube": pos + 1.0}
    boxes = {
        "green_cube": bounds,
        "wood_cube": bounds.clone(),
        "green_tray": bounds.clone(),
    }
    state.update(pos, torch.zeros(1), positions, boxes)
    assert state.reached.tolist() == [[True, True, True, False, False, False, False]]
    boxes["wood_cube"] = bounds + torch.tensor([0.55, 0.0, 0.0])
    boxes["green_tray"][:, 1, 0] = 2.0
    state.update(pos + 1.0, torch.zeros(1), positions, boxes)
    assert state.reached.tolist() == [[True, True, True, True, True, True, False]]
    boxes["wood_cube"] = bounds + torch.tensor([0.5, 0.0, 0.0])
    state.update(pos, torch.zeros(1), positions, boxes)
    assert state.reached.all()


@pytest.mark.requires_sim
@pytest.mark.gpu
def test_move_latte_cup_physical_score_and_partial_reset():
    """Exercise native cameras, measured criteria, selective reset and terminal info."""
    path = ROOT / "move_latte_cup/task.droid.yaml"
    deployment = yaml.safe_load(path.read_text())
    cfg = config_to_cfg(deployment, source_path=path)
    cfg.num_envs = 2
    cfg.init_rollout_buffer = False
    env = gym.make(deployment["id"], cfg=cfg)
    target = env.unwrapped
    try:
        obs, _ = env.reset(seed=42, options={"initial_condition": 0})
        joints = target.robot.get_qpos()[:, :7]
        hold_open = torch.cat((joints, -torch.ones_like(joints[:, :1])), dim=1)
        for _ in range(15):
            obs, _, _, _, _ = env.step(hold_open)
        for name in ("external_camera", "wrist_camera"):
            rgb = obs["sensor"][name]["color"][..., :3]
            assert rgb.shape == (2, 360, 640, 3)
            assert rgb[0].float().std() > 5

        cup = target.sim.get_rigid_object("latte_art_cup")
        board = target.sim.get_rigid_object("cutting_board")
        # Place the physical cup near the gripper and above the table. This
        # exercises the scorer without assuming an expert grasping policy.
        pose = cup.get_local_pose()[:1].clone()
        gripper = target.robot.get_link_pose("robotiq_arg2f_base_link")
        pose[0, :3] = gripper[0, :3] + pose.new_tensor([0, 0.17, 0.035])
        cup.set_local_pose(pose, env_ids=[0])
        _, _, terminated, _, info = env.step(hold_open)
        assert info["metrics"]["criteria"].tolist() == [
            [True, True, False],
            [False, False, False],
        ]
        assert not terminated.any()
        before_reset = cup.get_local_pose()[0].clone()
        _, info = env.reset(options={"reset_ids": [1], "initial_condition": 1})
        assert info["metrics"]["criteria"].tolist() == [
            [True, True, False],
            [False, False, False],
        ]
        assert torch.allclose(cup.get_local_pose()[0], before_reset, atol=1e-6)

        pose = cup.get_local_pose()[:1].clone()
        pose[0, :2] = board.get_local_pose()[0, :2]
        pose[0, 2] = 0.20
        cup.set_local_pose(pose, env_ids=[0])
        _, _, terminated, _, info = env.step(hold_open)
        assert terminated.tolist() == [True, False]
        assert info["success"].tolist() == [True, False]
        assert info["metrics"]["criteria"][0].all()
        assert info["metrics"]["progress"][0].item() == 1.0
        assert not target.is_task_success().any()
    finally:
        target.close(exit_process=False)


@pytest.mark.slow
@pytest.mark.requires_sim
@pytest.mark.gpu
def test_pan_clean_expert_scrubs_and_releases_sponge():
    """Run the real gripper through both scrubbing strokes before native success."""
    path = ROOT / "pan_clean/task.expert.yaml"
    deployment = yaml.safe_load(path.read_text())
    cfg = config_to_cfg(deployment, source_path=path)
    cfg.init_rollout_buffer = False
    env = gym.make(deployment["id"], cfg=cfg)
    try:
        env.reset(seed=42)
        target = env.unwrapped
        assert not target.is_task_success().any()
        result = execute_demo_episode(env)
        assert result.completed and result.all_success
        assert not any(result.truncated)
        names = [segment.name for segment in result.segments]
        assert names.index("sponge_lift") < names.index("sponge_scrub_out")
        assert names.index("sponge_scrub_out") < names.index("sponge_scrub_back")
        assert names.index("sponge_scrub_back") < names.index("sponge_release")
        assert target.rubric.reached[0].all()
    finally:
        env.unwrapped.close(exit_process=False)
