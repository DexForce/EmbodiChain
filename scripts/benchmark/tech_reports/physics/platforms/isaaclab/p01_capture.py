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

"""Capture native Isaac Lab RTX camera frames during a P-01 physics run."""

from __future__ import annotations

import json
import subprocess
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


class NativeCapture:
    """Render live states without writing body poses or advancing physics."""

    def __init__(
        self, sim, case: str, output: Path, dt: float, common_view: Path | None = None
    ) -> None:
        self.sim, self.case, self.dt = sim, case, dt
        self.common = json.loads(common_view.read_text()) if common_view else None
        self.output = output / "native_camera"
        self.output.mkdir()
        self.frames = []
        self.stride = round(1 / (40 * dt))
        assert abs(self.stride * dt - 0.025) < 1e-10
        self.window = 1.0 if case == "freefall" else 10.0
        stage = sim.stage
        materials = {}
        for name, color in {
            "body": (0.03, 0.26, 0.55),
            "metal": (0.19, 0.24, 0.30),
            "backdrop": (0.62, 0.66, 0.70),
            "ruler": (0.25, 0.31, 0.37),
        }.items():
            material = UsdShade.Material.Define(stage, f"/World/CaptureLooks/{name}")
            shader = UsdShade.Shader.Define(stage, f"/World/CaptureLooks/{name}/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(*color)
            )
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
            material.CreateSurfaceOutput().ConnectToSource(
                shader.ConnectableAPI(), "surface"
            )
            materials[name] = material

        def box(path: str, pos: tuple, size: tuple, material: str) -> None:
            geom = UsdGeom.Cube.Define(stage, path)
            geom.CreateSizeAttr(1.0)
            geom.AddTranslateOp().Set(Gf.Vec3d(*pos))
            geom.AddScaleOp().Set(Gf.Vec3f(*size))
            UsdShade.MaterialBindingAPI.Apply(geom.GetPrim()).Bind(materials[material])

        body_path = (
            "/World/Body/visual" if case == "freefall" else "/World/Pendulum/bob/visual"
        )
        UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(body_path)).Bind(
            materials["body"]
        )
        if case == "pendulum":
            # This cylinder is a visual child of the already mass-specified bob.
            rod = UsdGeom.Cylinder.Define(stage, "/World/Pendulum/bob/capture_rod")
            rod.CreateRadiusAttr(0.0025)
            rod.CreateHeightAttr(0.5)
            rod.CreateAxisAttr("Z")
            rod.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.25))
            UsdShade.MaterialBindingAPI.Apply(rod.GetPrim()).Bind(materials["metal"])
        if self.common:
            for b in self.common[case]["visual_boxes"]:
                box(
                    f"/World/Common_{b['uid']}",
                    tuple(b["pos"]),
                    tuple(b["size"]),
                    b["material"],
                )
        elif case == "pendulum":
            box(
                "/World/CaptureSupport", (0, 0.018, 1.025), (0.17, 0.05, 0.035), "metal"
            )
            box(
                "/World/CaptureBackdrop", (0, 0.13, 0.75), (1.5, 0.015, 1.2), "backdrop"
            )
            for i in range(11):
                z = 0.35 + 0.075 * i
                box(
                    f"/World/CaptureGrid/h{i}",
                    (0, 0.115, z),
                    (1.2, 0.002, 0.0008),
                    "ruler",
                )
            for i in range(13):
                x = -0.45 + 0.075 * i
                box(
                    f"/World/CaptureGrid/v{i}",
                    (x, 0.115, 0.725),
                    (0.0008, 0.002, 0.85),
                    "ruler",
                )
        else:
            box(
                "/World/CaptureBackdrop", (0, 0.18, -1.5), (1.2, 0.015, 6.5), "backdrop"
            )
            box("/World/CaptureRuler", (0.18, 0.15, -1.5), (0.005, 0.005, 6.2), "metal")
            for i in range(25):
                z = 1.5 - 0.25 * i
                box(
                    f"/World/CaptureTicks/t{i}",
                    (0.2, 0.145, z),
                    (0.06, 0.006, 0.004),
                    "metal",
                )
        dome = UsdLux.DomeLight.Define(stage, "/World/CaptureDome")
        dome.CreateIntensityAttr(450.0)
        light = UsdLux.DistantLight.Define(stage, "/World/CaptureKey")
        light.CreateIntensityAttr(2000.0)
        light.CreateAngleAttr(8.0)
        light.AddRotateXYZOp().Set(Gf.Vec3f(30, -25, -20))
        self.camera = Camera(
            CameraCfg(
                prim_path="/World/CaptureCamera",
                update_period=0.0,
                width=1280,
                height=720,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=(
                        self.common["intrinsics"][0] * 36.0 / self.common["width"]
                        if self.common
                        else 40.0
                    ),
                    horizontal_aperture=36.0,
                    clipping_range=(0.01, 1000.0),
                ),
                background_color=(0.9, 0.92, 0.94),
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
        """Write a native RGB frame at a known physical sample time."""
        t = step * self.dt
        if self.common:
            if step == 0 or step % self.stride or t > self.window + 1e-12:
                return
        elif step == 0 or (step - 1) % self.stride or t >= self.window:
            return
        if self.common:
            eye = np.array(self.common[self.case]["eye"], dtype=float)
            target = np.array(self.common[self.case]["target"], dtype=float)
            if self.case == "freefall":
                z = (
                    self.common["freefall_z0_m"]
                    - 0.5 * self.common["gravity_m_s2"] * t * t
                )
                eye[2] += z
                target[2] += z
            eye, target = eye.tolist(), target.tolist()
        elif self.case == "freefall":
            z = float(state[0])
            eye, target = (0.22, -0.85, z + 0.11), (0.03, 0, z)
        else:
            eye, target = (0.25, -1.7, 0.8), (0, 0, 0.76)
        self.sim.forward()
        self.camera.set_world_poses_from_view([eye], [target])
        # The first physics step publishes the initialized pose to Fabric.
        # Warm up rendering at that fixed time without advancing physics.
        for _ in range(24 if not self.frames else 4):
            self.sim.render()
            self.camera.update(dt=0.0, force_recompute=True)
            rgb = self.camera.data.output["rgb"].torch[0, ..., :3].cpu().numpy().copy()
        assert rgb.dtype == np.uint8 and np.ptp(rgb) > 8
        path = self.output / f"frame_{len(self.frames):04d}.png"
        Image.fromarray(rgb).save(path)
        self.frames.append(
            dict(
                file=path.name,
                step=step,
                time_s=t,
                state=state.tolist(),
                camera_eye=eye,
                camera_target=target,
                intrinsics=self.camera.data.intrinsic_matrices.torch[0].cpu().tolist(),
                camera_pos_readback=self.camera.data.pos_w.torch[0].cpu().tolist(),
                camera_quat_opengl_xyzw=self.camera.data.quat_w_opengl.torch[0]
                .cpu()
                .tolist(),
            )
        )

    def finish(self) -> dict:
        """Encode the live-camera frames, preserving unmodified PNGs."""
        count = len(self.frames)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                "40",
                "-i",
                str(self.output / "frame_%04d.png"),
                "-frames:v",
                str(count),
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(self.output / "native_1x.mp4"),
            ],
            check=True,
        )
        meta = dict(
            source="Isaac Lab Camera / IsaacRtxRenderer; captured during live PhysX stepping",
            width=1280,
            height=720,
            capture_fps=40,
            playback_speed=1,
            window_s=[self.frames[0]["time_s"], self.frames[-1]["time_s"]],
            initial_render_note="Capture starts after the first physical step to avoid a stale reset pose in Fabric.",
            last_frame_hold_s=0.025,
            video_duration_s=count / 40,
            camera_motion=(
                (
                    "shared analytic trajectory"
                    if self.common
                    else "follows falling body"
                )
                if self.case == "freefall"
                else "fixed"
            ),
            visual_only_additions="materials, lights, background and reference marks; pendulum also has a massless visual rod and support",
            physics_pose_writes_during_rollout=False,
            common_config=self.common,
            frames=self.frames,
        )
        (self.output / "capture.json").write_text(json.dumps(meta, indent=2) + "\n")
        return {k: v for k, v in meta.items() if k != "frames"}
