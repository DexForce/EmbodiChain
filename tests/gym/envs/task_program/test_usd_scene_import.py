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

"""A whole-scene import participates in real configured Task Program execution."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import gymnasium as gym
import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade
import pytest
import yaml

from embodichain.lab.gym.envs.demo import execute_demo_episode
from embodichain.lab.gym.utils.gym_utils import config_to_cfg
from embodichain.lab.sim import SimulationManager
from embodichain.utils.utility import load_config

pytestmark = [pytest.mark.subprocess_sim, pytest.mark.gpu]


def test_imported_scene_completes_verified_pick_place_and_resets(
    tmp_path: Path,
) -> None:
    env = dict(os.environ, EMBODICHAIN_SIM_EXIT_PROCESS="0")
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy,sys; from pathlib import Path; "
            "scope=runpy.run_path(sys.argv[1]); "
            "scope['_exercise_imported_scene_verified_pick_place'](Path(sys.argv[2]))",
            str(Path(__file__).resolve()),
            str(tmp_path),
        ],
        env=env,
        check=True,
        timeout=180,
    )


def _exercise_imported_scene_verified_pick_place(tmp_path: Path) -> None:
    source = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(source))
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1)
    material = UsdShade.Material.Define(stage, "/World/contact")
    contact = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    contact.CreateStaticFrictionAttr(0.99)
    contact.CreateDynamicFrictionAttr(0.97)
    for name, size, position in (
        ("table", (0.6, 1.0, 0.04), (0.6, 0.0, 0.73)),
        ("cube", (0.05, 0.05, 0.05), (0.45, 0.15, 0.777)),
    ):
        box = UsdGeom.Cube.Define(stage, f"/World/{name}")
        box.CreateSizeAttr(1)
        box.AddTranslateOp().Set(Gf.Vec3d(*position))
        box.AddScaleOp().Set(Gf.Vec3f(*size))
        UsdPhysics.CollisionAPI.Apply(box.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(box.GetPrim()).Bind(
            material, materialPurpose="physics"
        )
        if name == "cube":
            UsdPhysics.RigidBodyAPI.Apply(box.GetPrim())
            UsdPhysics.MassAPI.Apply(box.GetPrim()).CreateMassAttr(0.05)
    stage.GetRootLayer().Save()
    environment = {
        "environment_id": "usd_pick_place_test",
        "physics": "default",
        "max_episode_steps": 2400,
        "physics_config": {"enable_ccd": True},
        "simulation": {
            "usd_scene": {
                "name": "environment",
                "source": {"file_path": "scene.usda"},
                "asset_cfgs": {
                    "/World/cube": {"uid": "cube"},
                    "/World/table": {"uid": "support_surface"},
                },
            }
        },
        "env": {"sim_steps_per_control": 4},
    }
    (tmp_path / "env.yaml").write_text(yaml.safe_dump(environment))
    task_root = (
        Path(__file__).resolve().parents[4]
        / "extensions/embodichain_tasks/configs/tasks/manipulation/repeated_pick_place"
    )
    source_task = task_root / "task.dual_ur5_dh_pgi_140_80.yaml"
    config = load_config(source_task)
    config["id"] = "USDSceneVerifiedPickPlaceTest-v1"
    config["environment"] = {"component": str(tmp_path / "env.yaml")}
    config["enable_sensor"] = False
    program = load_config(task_root / "task_program/program.tabletop.yaml")
    program["program"]["repeat"]["count"] = 1
    program["program"]["repeat"]["body"][0]["validators"] = [
        {"kind": "object_lifted", "object": "cube", "minimum_lift": 0.04}
    ]
    (tmp_path / "program.yaml").write_text(yaml.safe_dump(program))
    config["task_program"]["program"] = str(tmp_path / "program.yaml")
    cfg = config_to_cfg(config, source_path=source_task)
    cfg.sim_cfg.headless = True
    cfg.sim_cfg.render_cfg.renderer = "no-render"
    cfg.sim_cfg.startup_summary = "off"
    env = None
    try:
        env = gym.make(config["id"], cfg=cfg)
        env.reset(seed=42)
        assert env.unwrapped.sim.get_robot_uid_list() == ["DualUR5"]
        assert set(env.unwrapped.sim.get_rigid_object_uid_list()) == {
            "cube",
            "support_surface",
        }
        result = execute_demo_episode(env)
        metadata = result.to_metadata()
        (tmp_path / "execution_result.json").write_text(
            json.dumps(metadata, allow_nan=False)
        )
        assert result.completed and result.all_success, result.terminal_reason
        assert len(metadata["segments"]) == 2
        for segment in metadata["segments"]:
            assert segment["metadata"]["validation"]["accepted_mask"] == [True]
        env.reset(options={"save_data": False})
        pose = env.unwrapped.sim.get_rigid_object("cube").get_local_pose()
        assert np.allclose(pose[0, :3].cpu().numpy(), [0.45, 0.15, 0.777], atol=0.004)
    finally:
        if env is not None:
            env.unwrapped.close(exit_process=False)
        env = None
        SimulationManager.flush_cleanup_queue()
