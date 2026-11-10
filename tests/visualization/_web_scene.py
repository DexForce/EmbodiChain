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
import socket
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
from embodichain.lab.sim.sensors import CameraCfg, StereoCameraCfg
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.lab.visualization import DexSimWebCfg, VisualizationCfg


def main() -> None:
    port, backend, result_path = int(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    num_envs = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    renderer = sys.argv[6] if len(sys.argv) > 6 else "hybrid"
    running = True

    def stop(*_: object) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            web=True,
            num_envs=num_envs,
            physics_cfg=physics_cfg_for_backend(backend),
            render_cfg=RenderCfg(
                renderer=renderer,
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
        if len(sys.argv) > 5 and sys.argv[5] == "sensors":
            sim.prepare()
            depth = sim.add_sensor(
                CameraCfg(
                    uid="depth_only",
                    width=65,
                    height=49,
                    enable_color=False,
                    enable_depth=True,
                )
            )
            stereo = sim.add_sensor(
                StereoCameraCfg(uid="stereo", width=64, height=48, enable_color=True)
            )
            assert depth.num_instances == stereo.num_instances == num_envs
            assert not depth.cfg.enable_color and depth.cfg.enable_depth
            assert depth.cfg.width == 65 and depth.cfg.height == 49
            assert sim._visualization_sim_step == 0
            result_path.write_text(
                json.dumps(
                    {
                        "num_envs": num_envs,
                        "depth_size": [65, 49],
                        "stereo_instances": stereo.num_instances,
                    }
                )
            )
            return
        cube = sim.add_rigid_object(
            RigidObjectCfg(
                uid="cube",
                shape=CubeCfg(size=[0.4, 0.4, 0.4]),
                body_type="kinematic",
                init_pos=[0, 0, 0.4],
            )
        )
        sim.prepare()
        offset = sim.arena_offsets[0].detach().cpu().numpy()
        sim.viewer.set_look_at(
            np.array([2.0, -3.0, 2.0]) + offset, np.array([0.0, 0.0, 0.4]) + offset
        )
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
            eye=torch.tensor([[1.0, -2.0, 1.0]], device=sim.device).expand(
                num_envs, -1
            ),
            target=torch.tensor([[0.0, 0.0, 0.4]], device=sim.device).expand(
                num_envs, -1
            ),
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
                test_arena_offsets=sim.arena_offsets.tolist(),
                test_sensor_pose=sensor.get_local_pose(to_matrix=True).tolist(),
                test_cube_pose=cube.get_local_pose(to_matrix=True).tolist(),
                test_camera_pose=viewer.get_pose_matrix().tolist(),
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
        # Prove World teardown released the listener while this host is alive;
        # process exit would release its sockets even if native cleanup leaked.
        with socket.socket() as released:
            released.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            released.bind(("127.0.0.1", port))


if __name__ == "__main__":
    main()
