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
from importlib.util import find_spec
import math
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


@pytest.mark.parametrize(
    "input_kind",
    [
        "usd",
        pytest.param(
            "gaussian_zip",
            marks=pytest.mark.skipif(
                find_spec("embodichain.lab.sim.cfg.visuals") is None,
                reason="Requires Gaussian visual integration from PR #934",
            ),
        ),
    ],
)
def test_imported_scene_completes_verified_pick_place_and_resets(
    tmp_path: Path,
    input_kind: str,
) -> None:
    env = dict(os.environ, EMBODICHAIN_SIM_EXIT_PROCESS="0")
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy,sys; from pathlib import Path; "
            "scope=runpy.run_path(sys.argv[1]); "
            "scope['_exercise_imported_scene_verified_pick_place'](Path(sys.argv[2]),sys.argv[3])",
            str(Path(__file__).resolve()),
            str(tmp_path),
            input_kind,
        ],
        env=env,
        check=True,
        timeout=180,
    )


def _exercise_imported_scene_verified_pick_place(
    tmp_path: Path, input_kind: str = "usd"
) -> None:
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
    if input_kind == "gaussian_zip":
        fields = "x y z f_dc_0 f_dc_1 f_dc_2 opacity scale_0 scale_1 scale_2 rot_0 rot_1 rot_2 rot_3".split()
        values = (0, 0, 0, 1.4, -1.4, -1.4, 4, *([math.log(0.05)] * 3), 1, 0, 0, 0)
        (tmp_path / "capture.ply").write_text(
            "ply\nformat ascii 1.0\nelement vertex 1\n"
            + "".join(f"property float {name}\n" for name in fields)
            + "end_header\n"
            + " ".join(map(str, values))
            + "\n"
        )
        visual = UsdGeom.Xform.Define(stage, "/World/capture")
        visual.GetPrim().GetReferences().AddReference("./capture.ply")
        visual.AddTranslateOp().Set(Gf.Vec3d(1.0, 0.6, 0.8))
        proxy = UsdGeom.Cube.Define(stage, "/World/proxy")
        proxy.CreateSizeAttr(0.1)
        proxy.AddTranslateOp().Set(Gf.Vec3d(1.0, 0.6, 0.8))
        UsdPhysics.CollisionAPI.Apply(proxy.GetPrim())
    stage.GetRootLayer().Save()
    prefix = ""
    if input_kind == "gaussian_zip":
        from embodichain.gen_sim.scene_engine.pipeline.utils.scene_package import (
            _export_scene_package,
        )

        source = _export_scene_package(
            scene_usd_path=source, destination=tmp_path / "scene.zip"
        )
        prefix = "/Asset"
    environment = {
        "environment_id": "usd_pick_place_test",
        "physics": "default",
        "max_episode_steps": 2400,
        "physics_config": {"enable_ccd": True},
        "simulation": {
            "usd_scene": {
                "name": "environment",
                "source": {"file_path": source.name},
                "asset_cfgs": {
                    prefix + "/World/cube": {"uid": "cube"},
                    prefix + "/World/table": {"uid": "support_surface"},
                },
            }
        },
        "env": {"sim_steps_per_control": 4},
    }
    if input_kind == "gaussian_zip":
        environment["render_cfg"] = {
            "preset": "gaussian",
            "denoising": {"window": "off", "offscreen": "off"},
        }
        environment["simulation"]["usd_scene"]["asset_cfgs"].update(
            {
                prefix + "/World/capture": {"uid": "capture"},
                prefix + "/World/proxy": {"uid": "proxy"},
            }
        )
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
    if input_kind == "usd":
        cfg.sim_cfg.render_cfg.renderer = "no-render"
    cfg.sim_cfg.startup_summary = "off"
    env = camera = None
    try:
        env = gym.make(config["id"], cfg=cfg)
        env.reset(seed=42)
        assert env.unwrapped.sim.get_robot_uid_list() == ["DualUR5"]
        expected = {
            "cube",
            "support_surface",
        }
        if input_kind == "gaussian_zip":
            expected.add("proxy")
            from embodichain.lab.sim.sensors import CameraCfg

            camera = env.unwrapped.sim.add_sensor(
                CameraCfg(
                    uid="capture_view",
                    width=160,
                    height=120,
                    intrinsics=(120, 120, 80, 60),
                    enable_depth=True,
                    extrinsics=CameraCfg.ExtrinsicsCfg(
                        eye=(1.3, -1.0, 1.4), target=(0.55, 0, 0.9), up=(0, 0, 1)
                    ),
                )
            )
            env.unwrapped.sim.render_camera_group([camera.group_id])
            camera.update(fetch_only=True)
            assert camera.capture_metadata["valid"]
            assert camera.capture_metadata["output_contract"]["same_frame_identity"]
            resolved = Path(
                env.unwrapped.sim.get_rigid_object("cube").cfg.source.file_path
            )
            assert (resolved.parent / "capture.ply").is_file()
        assert set(env.unwrapped.sim.get_rigid_object_uid_list()) == expected
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
        camera = None
        if env is not None:
            env.unwrapped.close(exit_process=False)
        env = None
        SimulationManager.flush_cleanup_queue()
    if input_kind == "gaussian_zip":
        assert not resolved.exists()
