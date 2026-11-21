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

"""Native EmbodiChain camera frames from live P-02 simulation."""

from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from embodichain.lab.sim.cfg import VisualObjectCfg
from embodichain.lab.sim.material import VisualMaterialCfg
from embodichain.lab.sim.sensors import CameraCfg
from embodichain.lab.sim.shapes import CubeCfg
from scripts.benchmark.tech_reports.physics.experiments.p02_media_inputs import (
    view,
    marks,
    encode,
    freeze_capture_sources,
)


class Capture:
    """Record the common view, without advancing physics or changing body state."""

    def __init__(
        self,
        sim,
        cfg: dict,
        scene: dict,
        inputs: dict,
        initial: int,
        output: Path,
        dt: float,
    ) -> None:
        self.sim, self.cfg, self.scene, self.inputs, self.initial, self.dt = (
            sim,
            cfg,
            scene,
            inputs,
            initial,
            dt,
        )
        self.output = output / "native_camera"
        self.output.mkdir()
        self.frames = []
        freeze_capture_sources(output, Path(__file__))
        self.stride = round(1 / (cfg["capture"]["fps"] * dt))
        assert abs(self.stride * dt - 1 / cfg["capture"]["fps"]) < 1e-10
        for b in marks(cfg, scene):
            sim.add_visual_object(
                VisualObjectCfg(
                    uid=b["uid"],
                    init_pos=tuple(b["pos"]),
                    init_rot=tuple(b["rot"]),
                    shape=CubeCfg(
                        size=b["size"],
                        visual_material=VisualMaterialCfg(
                            uid="ruler_ink",
                            base_color=[0.22, 0.25, 0.28, 1],
                            roughness=0.6,
                        ),
                    ),
                )
            )
        eye, target, _ = view(cfg, scene, inputs, initial, 0)
        c = cfg["capture"]
        self.camera = sim.add_sensor(
            CameraCfg(
                uid="p02_camera",
                width=c["width"],
                height=c["height"],
                intrinsics=tuple(c["intrinsics"]),
                extrinsics=CameraCfg.ExtrinsicsCfg(
                    eye=tuple(eye), target=tuple(target), up=(0, 0, 1)
                ),
            )
        )

    def capture(self, step: int, state: np.ndarray) -> None:
        t = step * self.dt
        eye, target, window = view(self.cfg, self.scene, self.inputs, self.initial, t)
        if step == 0 or step % self.stride or t > window + 1e-12:
            return
        self.camera.look_at(
            torch.tensor(eye[None], dtype=torch.float32, device=self.sim.device),
            torch.tensor(target[None], dtype=torch.float32, device=self.sim.device),
        )
        self.sim.render_camera_group([self.camera.group_id])
        self.camera.update(fetch_only=True)
        rgb = self.camera.get_data()["color"][0].cpu().numpy()
        assert rgb.dtype == np.uint8 and np.ptp(rgb) > 8
        filename = f"frame_{len(self.frames):04d}.png"
        Image.fromarray(rgb).save(self.output / filename)
        self.frames.append(
            dict(
                file=filename,
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
        return encode(
            self.output,
            self.frames,
            self.cfg["capture"],
            "EmbodiChain live Camera / hybrid renderer",
        )
