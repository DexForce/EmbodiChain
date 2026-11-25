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

"""Native Isaac Lab RTX camera frames from live P-02 simulation."""

from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade
import isaaclab.sim as sim_utils
from isaaclab.sensors import Camera, CameraCfg
from isaaclab_physx.renderers import (
    IsaacRtxRendererCfg,
    IsaacRtxRendererGlobalSettingsCfg,
)
from scripts.benchmark.tech_reports.physics.experiments.p02_media_inputs import (
    view,
    marks,
    encode,
    freeze_capture_sources,
)


class Capture:
    """Record the common view without advancing physics during rendering."""

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
        stage = sim.stage
        colors = dict(
            support=cfg["capture"]["surface_color"],
            background_ground=cfg["capture"]["background_color"],
            ruler=[0.22, 0.25, 0.28],
        )
        for i in range(1 if scene["kind"] == "slope" else scene["layers"]):
            colors[f"block_{i:02d}"] = cfg["capture"]["body_colors"][i % 3]
        materials = {}
        for name, color in colors.items():
            mat = UsdShade.Material.Define(stage, f"/World/CaptureLooks/{name}")
            shader = UsdShade.Shader.Define(stage, f"/World/CaptureLooks/{name}/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(*color)
            )
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
            mat.CreateSurfaceOutput().ConnectToSource(
                shader.ConnectableAPI(), "surface"
            )
            materials[name] = mat
            if name != "ruler":
                UsdShade.MaterialBindingAPI.Apply(
                    stage.GetPrimAtPath(f"/World/{name}/shape")
                ).Bind(mat)
        for b in marks(cfg, scene):
            geom = UsdGeom.Cube.Define(stage, f'/World/{b["uid"]}')
            geom.CreateSizeAttr(1)
            geom.AddTranslateOp().Set(Gf.Vec3d(*b["pos"]))
            geom.AddRotateXYZOp().Set(Gf.Vec3f(*b["rot"]))
            geom.AddScaleOp().Set(Gf.Vec3f(*b["size"]))
            UsdShade.MaterialBindingAPI.Apply(geom.GetPrim()).Bind(materials["ruler"])
        dome = UsdLux.DomeLight.Define(stage, "/World/CaptureDome")
        dome.CreateIntensityAttr(450)
        light = UsdLux.DistantLight.Define(stage, "/World/CaptureKey")
        light.CreateIntensityAttr(2000)
        light.CreateAngleAttr(8)
        light.AddRotateXYZOp().Set(Gf.Vec3f(30, -25, -20))
        c = cfg["capture"]
        self.camera = Camera(
            CameraCfg(
                prim_path="/World/P02Camera",
                update_period=0,
                width=c["width"],
                height=c["height"],
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=c["intrinsics"][0] * 36 / c["width"],
                    horizontal_aperture=36,
                    clipping_range=(0.01, 1000),
                ),
                background_color=tuple(c["background_color"]),
                renderer_cfg=IsaacRtxRendererCfg(
                    global_settings=IsaacRtxRendererGlobalSettingsCfg(
                        antialiasing_mode="DLSS",
                        dlss_mode=2,
                        enable_dl_denoiser=True,
                        enable_dlssg=False,
                        enable_global_illumination=False,
                        enable_reflections=False,
                        enable_ambient_occlusion=False,
                        samples_per_pixel=4,
                    )
                ),
            )
        )

    def capture(self, step: int, state: np.ndarray) -> None:
        t = step * self.dt
        eye, target, window = view(self.cfg, self.scene, self.inputs, self.initial, t)
        if step == 0 or step % self.stride or t > window + 1e-12:
            return
        self.sim.forward()
        self.camera.set_world_poses_from_view([eye.tolist()], [target.tolist()])
        for _ in range(24 if not self.frames else 4):
            self.sim.render()
            self.camera.update(dt=0, force_recompute=True)
            rgb = self.camera.data.output["rgb"].torch[0, ..., :3].cpu().numpy().copy()
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
                camera_pos_readback=self.camera.data.pos_w.torch[0].cpu().tolist(),
                camera_quat_opengl_xyzw=self.camera.data.quat_w_opengl.torch[0]
                .cpu()
                .tolist(),
                intrinsics=self.camera.data.intrinsic_matrices.torch[0].cpu().tolist(),
            )
        )

    def finish(self) -> dict:
        return encode(
            self.output,
            self.frames,
            self.cfg["capture"],
            "Isaac Lab live Camera / IsaacRtxRenderer",
        )
