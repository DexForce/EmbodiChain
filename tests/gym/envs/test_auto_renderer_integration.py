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

"""Fresh-process Gym startup, stepping and capture with demand-based auto."""

from __future__ import annotations

import faulthandler
import gc
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.subprocess_sim, pytest.mark.gpu]


@pytest.mark.parametrize(
    ("backend", "with_camera"),
    [("default", False), ("newton", False), ("default", True)],
)
def test_auto_renderer_environment_lifecycle(backend, with_camera, tmp_path) -> None:
    urdf = tmp_path / "slider.urdf"
    urdf.write_text(
        """<robot name="slider">
        <link name="base">
          <inertial><mass value="1"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
          <visual><geometry><box size="0.1 0.1 0.1"/></geometry></visual>
          <collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision>
        </link>
        <link name="child">
          <inertial><mass value="1"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
          <visual><geometry><box size="0.1 0.1 0.1"/></geometry></visual>
          <collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision>
        </link>
        <joint name="slide" type="prismatic">
          <parent link="base"/><child link="child"/><origin xyz="0 0 0.2"/><axis xyz="1 0 0"/>
          <limit lower="-0.2" upper="0.2" effort="10" velocity="1"/>
        </joint></robot>""",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            backend,
            str(int(with_camera)),
            str(urdf),
        ],
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _run_environment(backend: str, with_camera: bool, urdf_path: str) -> None:
    import torch

    from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
    from embodichain.lab.gym.envs.managers.cfg import EventCfg
    from embodichain.lab.gym.envs.managers.randomization.visual import (
        randomize_visual_material,
    )
    from embodichain.lab.sim import (
        SimulationManager,
        SimulationManagerCfg,
        cfg as sim_cfg,
    )
    from embodichain.lab.sim.cfg import (
        DefaultPhysicsCfg,
        LightCfg,
        NewtonPhysicsCfg,
        RigidObjectCfg,
        RobotCfg,
    )
    from embodichain.lab.sim.sensors import CameraCfg
    from embodichain.lab.sim.shapes import CubeCfg

    sim_cfg.DEFAULT_RENDERER = "auto"
    physics = (
        NewtonPhysicsCfg(
            device="cuda:0", num_substeps=1, solver_cfg={"solver_type": "mujoco_warp"}
        )
        if backend == "newton"
        else DefaultPhysicsCfg(device="cpu")
    )
    config = EmbodiedEnvCfg(
        sim_cfg=SimulationManagerCfg(
            headless=True, physics_cfg=physics, startup_summary="off"
        ),
        robot=RobotCfg(uid="slider", fpath=urdf_path, init_pos=(0.0, 0.0, 0.5)),
        sensor=[CameraCfg(uid="camera", width=32, height=32)] if with_camera else [],
        background=[
            RigidObjectCfg(uid="background", shape=CubeCfg(), body_type="static")
        ],
        ignore_terminations=True,
    )
    if not with_camera:
        config.light.direct = [LightCfg(uid="light")]
        config.light.indirect = {"env_map": "must-not-be-downloaded.hdr"}
        config.events = {"visual": EventCfg(func=randomize_visual_material)}
    env = None
    try:
        env = EmbodiedEnv(config)
        assert env.sim.has_native_renderer is with_camera
        assert env.sim._requested_renderer == "auto"
        assert sim_cfg.DEFAULT_RENDERER == "auto"
        assert not env.sim.is_window_opened
        assert env.sim.get_rigid_object("background") is not None
        assert env.sim._spawn_scene.handles("default_plane")
        if not with_camera:
            assert env.sim.sim_config.render_cfg.renderer == "no-render"
            assert not env.sim._sensors and not env.sim._lights
            assert env.cfg.events["visual"] is None
        obs, _ = env.reset()
        for _ in range(2):
            obs, _, _, _, _ = env.step(torch.zeros((1, 1), device=env.device))
        assert torch.isfinite(obs["robot"]["qpos"]).all()
        if with_camera:
            image = obs["sensor"]["camera"]["color"]
            assert image.shape == (1, 32, 32, 4)
        env.reset()
        print(
            f"{backend}: auto renderer environment lifecycle passed; camera={with_camera}",
            flush=True,
        )
    finally:
        if env is not None:
            env.close(exit_process=False)
        env = None
        gc.collect()
        SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    faulthandler.enable()
    faulthandler.dump_traceback_later(120)
    _run_environment(sys.argv[1], bool(int(sys.argv[2])), sys.argv[3])
