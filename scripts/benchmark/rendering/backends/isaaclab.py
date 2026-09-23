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

"""Isaac Lab adapter, imported only in its installed Python environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from ..workload import PilotCfg

__all__ = ["IsaacLabCamera"]


class IsaacLabCamera:
    """One procedural RGB camera, using Isaac Lab's installed public API."""

    def __init__(self, cfg: PilotCfg) -> None:
        from isaaclab.app import AppLauncher

        self.app = AppLauncher(headless=True, enable_cameras=True, device="cuda:0").app
        import math
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import Camera, CameraCfg
        from isaaclab_physx.renderers import IsaacRtxRendererCfg
        from ..workload import scene_spec

        self.torch = torch
        self.scene = scene_spec()
        self.sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(
                dt=1 / 240,
                device="cuda:0",
                render=sim_utils.RenderCfg(antialiasing_mode="Off", enable_dlssg=False),
            )
        )
        for box in self.scene["boxes"]:
            shape = sim_utils.CuboidCfg(
                size=tuple(box["size"]),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=tuple(box["color"]),
                    roughness=1.0,
                    metallic=0.0,
                ),
            )
            shape.func("/World/" + box["id"], shape, translation=tuple(box["position"]))
        light = sim_utils.DistantLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0))
        light.func("/World/Sun", light)
        fx = cfg.width / (
            2 * math.tan(math.radians(self.scene["horizontal_fov_deg"]) / 2)
        )
        camera_cfg = CameraCfg(
            prim_path="/World/Camera",
            update_period=0.0,
            width=cfg.width,
            height=cfg.height,
            data_types=["rgb"],
            renderer_cfg=IsaacRtxRendererCfg(),
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                horizontal_aperture=24.0 * cfg.width / fx,
                clipping_range=(self.scene["near_m"], self.scene["far_m"]),
            ),
        )
        self.camera = Camera(camera_cfg)
        self.sim.reset()
        self.set_probe_offset(0.0)
        self.metadata = {
            "renderer": "Isaac RTX",
            "antialiasing": "Off",
            "dlss_frame_generation": False,
            "light": {
                "type": "DistantLight",
                "intensity": 3000.0,
                "unit": "Isaac/USD native",
            },
            "physics_steps_in_measurement": 0,
            "source_camera_channels": 3,
            "intrinsic_matrix": self.camera.data.intrinsic_matrices.torch[0]
            .cpu()
            .tolist(),
        }

    def capture(self) -> np.ndarray:
        """Fetch a fresh camera render and complete the host transfer."""
        # RTX annotators read the last Kit frame. Pump Kit once before
        # fetching; camera.update alone would benchmark stale buffer reads.
        self.sim.render()
        self.camera.update(dt=1 / 60, force_recompute=True)
        # Isaac Lab 3 uses ProxyArray; the legacy dict converter leaves it
        # unchanged, so access its explicit Torch view before host readback.
        rgb = self.camera.data.output["rgb"].torch
        return rgb[0, ..., :3].contiguous().cpu().numpy().copy()

    def set_probe_offset(self, offset: float) -> None:
        """Move the camera eye in world x by offset [m], without stepping."""
        eye = list(self.scene["eye"])
        eye[0] += offset
        self.camera.set_world_poses_from_view(
            self.torch.tensor([eye], device=self.sim.device),
            self.torch.tensor([self.scene["target"]], device=self.sim.device),
        )
        self.sim.render_context.reset_transform_cadence()

    def close(self) -> None:
        """Shut down the isolated Isaac Sim application."""
        self.camera = None
        self.app.close()
