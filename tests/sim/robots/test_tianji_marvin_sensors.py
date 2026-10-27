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

"""Live projection, metric depth and mounting for Marvin and W1/Pika cameras."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from pathlib import Path

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import DenoisingCfg, RenderCfg, RigidObjectCfg
from embodichain.robot_registry import resolve_robot_cfg
from embodichain.lab.sim.material import VisualMaterialCfg
from embodichain.lab.sim.shapes import CubeCfg
from embodichain_robots import TianjiMarvinCfg
from embodichain.utils.utility import load_config


@pytest.mark.gpu
@pytest.mark.requires_sim
class TestTianjiMarvinSensors:
    def setup_method(self) -> None:
        self.sim = SimulationManager(
            SimulationManagerCfg(
                headless=True,
                device="cpu",
                num_envs=1,
                # This geometry test must not require optional NGX/DLSS support.
                render_cfg=RenderCfg(
                    renderer="hybrid",
                    denoising=DenoisingCfg(window="off", offscreen="off"),
                ),
            )
        )
        self.cfg = self._robot_cfg()
        self.robot = self.sim.add_robot(self.cfg)
        self.sim.prepare()

    def _robot_cfg(self):
        return TianjiMarvinCfg.from_dict(
            {"enable_sensor": True, "uid": "camera_test_marvin"}
        )

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
        if any(c.uid == "cam_high" for c in self.cfg.sensors):
            head = self.robot.get_qpos(name="head").clone()
            head[:, 0] += 0.15
            self.robot.set_qpos(head, name="head", target=False)
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
                == f"{self.cfg.uid}/{cfg.extrinsics.parent}"
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
            assert data["color"].shape == (1, cfg.height, cfg.width, 4)
            assert data["depth"].shape == (1, cfg.height, cfg.width)
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


class TestDexforceW1PikaSensors(TestTianjiMarvinSensors):
    def _robot_cfg(self):
        component = (
            Path(__file__).parents[3]
            / "extensions/embodichain_tasks/configs/components/embodiments"
            / "dexforce_w1_030_pika.yaml"
        )
        payload = load_config(component)["simulation"]
        payload.update(
            enable_sensor=True,
            uid="camera_test_w1_pika",
            init_pos=[0.0, 0.0, 0.0],
        )
        return resolve_robot_cfg(payload.pop("class_type"), payload)

    def test_ready_pose_uses_source_joint_order(self) -> None:
        """Native breadth-first joint ordering must preserve the designed TCPs."""
        for side, sign in (("left", 1), ("right", -1)):
            qpos = self.robot.get_qpos(name=f"{side}_arm")
            pose = self.robot.compute_fk(qpos=qpos, name=f"{side}_arm", to_matrix=True)
            torch.testing.assert_close(
                pose[0, :3, 3],
                # Accepted ready pose: forward forearms, level jaw-opening
                # axes and a slight downward approach, in the base frame.
                torch.tensor([0.52147, 0.26728 * sign, 1.06227]),
                atol=2e-4,
                rtol=0,
            )
            assert abs(float(pose[0, 2, 0])) < 0.01
            assert -0.20 < float(pose[0, 2, 2]) < -0.15
            flange = self.robot.get_link_pose(link_name=f"{side}_ee", to_matrix=True)
            native_tcp = flange @ torch.as_tensor(
                self.cfg.solver_cfg[f"{side}_arm"].tcp, dtype=flange.dtype
            )
            torch.testing.assert_close(pose, native_tcp, atol=2e-5, rtol=0)
            shoulder = self.robot.get_link_pose(
                link_name=f"shoulder_roll_{side[0]}_j2_link", to_matrix=True
            )
            elbow = self.robot.get_link_pose(
                link_name=f"elbow_yaw_{side[0]}_j4_link", to_matrix=True
            )
            assert float(shoulder[0, 2, 3] - elbow[0, 2, 3]) > 0.15
        head_pitch = np.deg2rad(40.0)
        torch.testing.assert_close(
            self.robot.get_qpos(name="head"),
            torch.tensor([[0.0, head_pitch]], dtype=torch.float32),
        )
        torso = self.robot.get_link_pose(link_name="upper_body_base", to_matrix=True)
        torch.testing.assert_close(torso[0, :3, 2], torch.tensor([0.0, 0.0, 1.0]))
        assert self.robot.dof == 24
        assert self.robot.active_dof == 22

        # The camera is rigidly forward-facing on the head. Only the physical
        # NECK2 rotation may aim it downward; a mount-time pitch hides mistakes.
        camera = self.sim.get_sensor("cam_high")
        torch.testing.assert_close(
            -camera.cfg.extrinsics.transformation[:3, 2],
            torch.tensor([1.0, 0.0, 0.0]),
            atol=1e-7,
            rtol=0,
        )
        for pitch in (0.0, head_pitch):
            head = torch.tensor([[0.0, pitch]], dtype=torch.float32)
            self.robot.set_qpos(head, name="head", target=False)
            self.sim.sync_render_state()
            parent = self.robot.get_link_pose(
                link_name="head_pitch_j2_link", to_matrix=True
            )
            actual = camera.get_arena_pose(to_matrix=True)
            torch.testing.assert_close(
                actual,
                parent @ camera.cfg.extrinsics.transformation,
                atol=2e-5,
                rtol=0,
            )
            torch.testing.assert_close(
                -actual[0, :3, 2],
                torch.tensor([np.cos(pitch), 0.0, -np.sin(pitch)], dtype=torch.float32),
                atol=2e-5,
                rtol=0,
            )
