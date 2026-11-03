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

"""Bounded subprocess scene for native WebRTC browser acceptance."""

from __future__ import annotations

import json
from pathlib import Path
import signal
import sys
import time

import numpy as np
import torch

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import (
    DenoisingCfg,
    RenderCfg,
    RigidObjectCfg,
    physics_cfg_for_backend,
)
from embodichain.lab.sim.sensors import CameraCfg
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.lab.visualization import DexSimWebCfg, VisualizationCfg


def main() -> None:
    port, backend, result_path = int(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    running = True

    def stop(*_: object) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            startup_summary="off",
            physics_cfg=physics_cfg_for_backend(backend),
            render_cfg=RenderCfg(
                renderer="hybrid",
                denoising=DenoisingCfg(window="optix", offscreen="optix"),
            ),
            visualization=VisualizationCfg(
                backend="dexsim-web",
                allow_commands=True,
                web=DexSimWebCfg(
                    port=port, width=640, height=360, eye=(2, -3, 2), target=(0, 0, 0.4)
                ),
            ),
        )
    )
    sensor = cube = None
    try:
        cube = sim.add_rigid_object(
            RigidObjectCfg(
                uid="cube",
                shape=CubeCfg(size=[0.4, 0.4, 0.4]),
                body_type="kinematic",
                init_pos=[0, 0, 0.4],
            )
        )
        sim.prepare()
        sensor = sim.add_sensor(
            CameraCfg(
                uid="task_camera",
                width=640,
                height=360,
                intrinsics=(300, 300, 320, 180),
                enable_color=True,
            )
        )
        sensor.look_at(
            eye=torch.tensor([[1.0, -2.0, 1.0]], device=sim.device),
            target=torch.tensor([[0.0, 0.0, 0.4]], device=sim.device),
        )
        sim.update(step=1)
        sim.render_camera_group([sensor.group_id])
        sensor.update(fetch_only=True)
        pixels = sensor._data_buffer["color"].clone()
        pose = sensor.get_local_pose(to_matrix=True).clone()
        intrinsic = sensor.get_intrinsics().clone()
        viewer = sim._web_viewer
        original_state = viewer.state

        def state() -> dict:
            return dict(
                original_state(),
                test_sim_step=sim._visualization_sim_step,
                test_sensor_pose=sensor.get_local_pose(to_matrix=True).tolist(),
                test_cube_pose=cube.get_local_pose(to_matrix=True).tolist(),
                test_intrinsic=np.asarray(viewer.camera.get_intrinsic()).tolist(),
                test_gizmo_count=viewer.gizmo.gizmo_count if viewer.gizmo else 0,
            )

        viewer.state = state
        assert not sim.is_window_opened and not sim.can_open_native_window()
        assert sensor.group_id != viewer.group.get_group_id()
        start = time.monotonic()
        while running and time.monotonic() - start < 120:
            sim.capture_visualization(force=False)
            time.sleep(0.005)
        sensor.update(fetch_only=True)
        assert torch.equal(sensor._data_buffer["color"], pixels)
        assert torch.equal(sensor.get_local_pose(to_matrix=True), pose)
        assert torch.equal(sensor.get_intrinsics(), intrinsic)
        assert sim._visualization_sim_step == 1
        result_path.write_text(
            json.dumps(
                {
                    "sensor_unchanged": True,
                    "physics_steps": 1,
                    "frames": viewer.frames,
                    "render_seconds": viewer.render_seconds,
                }
            )
        )
    finally:
        sim.destroy(exit_process=False)
        sensor = cube = None
        SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    main()
