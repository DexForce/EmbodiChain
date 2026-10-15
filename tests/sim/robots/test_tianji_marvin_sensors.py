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

"""Live projection, metric depth and rigid mounting for Marvin wrist cameras."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from pathlib import Path

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import RenderCfg, RigidObjectCfg
from embodichain.lab.sim.material import VisualMaterialCfg
from embodichain.lab.sim.shapes import CubeCfg
from embodichain_robots import TianjiMarvinCfg


@pytest.mark.gpu
@pytest.mark.requires_sim
class TestTianjiMarvinSensors:
    def setup_method(self) -> None:
        self.sim = SimulationManager(
            SimulationManagerCfg(
                headless=True,
                device="cpu",
                num_envs=1,
                render_cfg=RenderCfg(renderer="hybrid"),
            )
        )
        self.cfg = TianjiMarvinCfg.from_dict(
            {"enable_sensor": True, "uid": "camera_test_marvin"}
        )
        self.robot = self.sim.add_robot(self.cfg)
        self.sim.prepare()

    def teardown_method(self) -> None:
        self.sim.destroy()
        SimulationManager.flush_cleanup_queue()

    def test_mount_motion_and_asymmetric_rgbd_projection(self, tmp_path: Path) -> None:
        cameras = [self.sim.get_sensor(cfg.uid) for cfg in self.cfg.sensors]
        initial = [camera.get_arena_pose(to_matrix=True).clone() for camera in cameras]
        for side in ("left", "right"):
            qpos = self.robot.get_qpos(name=f"{side}_arm").clone()
            qpos[:, 0] += 0.25 if side == "left" else -0.25
            self.robot.set_qpos(qpos, name=f"{side}_arm", target=False)
        self.sim.sync_render_state()
        for cfg, camera, before in zip(self.cfg.sensors, cameras, initial, strict=True):
            parent = self.robot.get_link_pose(
                link_name=cfg.extrinsics.parent, to_matrix=True
            )
            expected = parent @ cfg.extrinsics.transformation
            torch.testing.assert_close(
                camera.get_arena_pose(to_matrix=True), expected, atol=2e-5, rtol=1e-5
            )
            assert not torch.allclose(before, expected)
            assert camera.is_attached
            assert (
                camera.cfg.extrinsics.parent
                == f"camera_test_marvin/{cfg.extrinsics.parent}"
            )

        # Distinct quadrants, colors and distances expose flips, swaps and depth units.
        markers = [(-0.065, -0.040, 0.25), (0.055, 0.035, 0.32), (0.005, -0.025, 0.40)]
        colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        marker_size = 0.018
        for cfg, camera in zip(self.cfg.sensors, cameras, strict=True):
            pose = camera.get_arena_pose(to_matrix=True)[0].cpu().numpy()
            # Mounted render poses are OpenGL; project in optical +Z/+Y-down.
            pose = pose @ np.diag([1.0, -1.0, -1.0, 1.0])
            for index, (point, color) in enumerate(zip(markers, colors, strict=True)):
                position = pose[:3, :3] @ point + pose[:3, 3]
                # Preserve optical-axis alignment without mixing Euler conventions.
                marker_pose = pose.copy()
                marker_pose[:3, 3] = position
                self.sim.add_rigid_object(
                    RigidObjectCfg(
                        uid=f"{cfg.uid}_marker_{index}",
                        body_type="static",
                        init_local_pose=marker_pose,
                        shape=CubeCfg(
                            size=(marker_size,) * 3,
                            visual_material=VisualMaterialCfg(
                                uid=f"{cfg.uid}_marker_material_{index}",
                                base_color=[*color, 1.0],
                                emissive=list(color),
                                emissive_intensity=1.0,
                            ),
                        ),
                    )
                )
        self.sim.prepare()
        self.sim.sync_render_state()
        for cfg, camera in zip(self.cfg.sensors, cameras, strict=True):
            for _ in range(4):
                camera.update()
            data = camera.get_data()
            assert data["color"].shape == (1, 360, 640, 4)
            assert data["depth"].shape == (1, 360, 640)
            assert data["depth"].dtype == torch.float32
            rgb = data["color"][0, :, :, :3].cpu().numpy().astype(float)
            depth = data["depth"][0].cpu().numpy()
            # Retain diagnostics for renderer/projection regressions.
            np.savez_compressed(
                tmp_path / f"{cfg.uid}.npz", rgb=rgb.astype(np.uint8), depth=depth
            )
            fx, fy, cx, cy = cfg.intrinsics
            for index, (x, y, z) in enumerate(markers):
                u, v = round(fx * x / z + cx), round(fy * y / z + cy)
                patch = rgb[v - 2 : v + 3, u - 2 : u + 3].mean(axis=(0, 1))
                assert patch[index] > max(np.delete(patch, index)) + 40
                # Front face is perpendicular to the optical axis, in metres.
                assert float(
                    np.median(depth[v - 2 : v + 3, u - 2 : u + 3])
                ) == pytest.approx(z - marker_size / 2, abs=0.002)
