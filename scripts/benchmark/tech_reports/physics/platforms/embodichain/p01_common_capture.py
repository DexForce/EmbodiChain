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

"""Live EmbodiChain capture using the shared P-01 camera and display geometry."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from embodichain.lab.sim.cfg import VisualObjectCfg
from embodichain.lab.sim.material import VisualMaterialCfg
from embodichain.lab.sim.sensors import CameraCfg
from embodichain.lab.sim.shapes import CubeCfg


class CommonCapture:
    def __init__(self, sim, case: str, output: Path, dt: float, config: Path) -> None:
        self.sim, self.case, self.dt = sim, case, dt
        self.cfg = json.loads(config.read_text())
        self.view = self.cfg[case]
        self.output = output / "native_camera"
        self.output.mkdir()
        self.frames = []
        self.stride = round(1 / (self.cfg["fps"] * dt))
        assert abs(self.stride * dt - 1 / self.cfg["fps"]) < 1e-10
        sim.set_ground_plane_visibility(False)
        for b in self.view["visual_boxes"]:
            sim.add_visual_object(
                VisualObjectCfg(
                    uid=f"display_{b['uid']}",
                    init_pos=tuple(b["pos"]),
                    shape=CubeCfg(
                        size=b["size"],
                        visual_material=VisualMaterialCfg(
                            uid=f"display_{b['material']}",
                            base_color=self.cfg["materials"][b["material"]],
                            roughness=0.55,
                        ),
                    ),
                )
            )
        self.camera = sim.add_sensor(
            CameraCfg(
                uid="common_camera",
                width=self.cfg["width"],
                height=self.cfg["height"],
                intrinsics=tuple(self.cfg["intrinsics"]),
                extrinsics=CameraCfg.ExtrinsicsCfg(
                    eye=tuple(self.view["eye"]),
                    target=tuple(self.view["target"]),
                    up=(0.0, 0.0, 1.0),
                ),
            )
        )

    def capture(self, step: int, state: np.ndarray) -> None:
        t = step * self.dt
        if step == 0 or step % self.stride or t > self.view["window_s"] + 1e-12:
            return
        eye, target = np.array(self.view["eye"]), np.array(self.view["target"])
        if self.case == "freefall":
            z = self.cfg["freefall_z0_m"] - 0.5 * self.cfg["gravity_m_s2"] * t * t
            eye[2] += z
            target[2] += z
        self.camera.look_at(
            torch.tensor([eye.tolist()], device=self.sim.device),
            torch.tensor([target.tolist()], device=self.sim.device),
        )
        self.sim.render_camera_group([self.camera.group_id])
        self.camera.update(fetch_only=True)
        rgb = self.camera.get_data()["color"][0].cpu().numpy()
        assert rgb.dtype == np.uint8 and np.ptp(rgb) > 8
        name = f"frame_{len(self.frames):04d}.png"
        Image.fromarray(rgb).save(self.output / name)
        self.frames.append(
            dict(
                file=name,
                step=step,
                time_s=t,
                state=state.tolist(),
                camera_eye=eye.tolist(),
                camera_target=target.tolist(),
                camera_world_opengl=self.camera.get_local_pose(to_matrix=True)[0]
                .cpu()
                .tolist(),
                intrinsics=self.camera.get_intrinsics()[0].cpu().tolist(),
            )
        )

    def finish(self) -> dict:
        meta = dict(
            source="EmbodiChain live Camera / hybrid renderer",
            common_config=self.cfg,
            playback_speed=1,
            physics_pose_writes_during_rollout=False,
            frames=self.frames,
        )
        (self.output / "capture.json").write_text(json.dumps(meta, indent=2) + "\n")
        return {k: v for k, v in meta.items() if k != "frames"}
