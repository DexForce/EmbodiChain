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

"""Isaac Lab adapter for the pure-rendering R-series suite."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from ..suite import RenderCaseCfg
    from ..workload import PilotCfg

__all__ = ["IsaacLabCamera"]


class IsaacLabCamera:
    """Own one Isaac Lab camera batch and expose completed captures."""

    def __init__(self, cfg: PilotCfg | RenderCaseCfg) -> None:
        from isaaclab.app import AppLauncher

        self.app = AppLauncher(headless=True, enable_cameras=True, device="cuda:0").app
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import Camera, CameraCfg
        from isaaclab_physx.renderers import IsaacRtxRendererCfg
        from ..suite import scene_spec

        self.cfg = cfg
        self.torch = torch
        self.sim_utils = sim_utils
        self.scene = scene_spec()
        self.num_envs = int(getattr(cfg, "num_envs", 1))
        self.cameras_per_env = int(getattr(cfg, "cameras_per_env", 1))
        self.total_cameras = self.num_envs * self.cameras_per_env
        self.modalities = tuple(getattr(cfg, "modalities", ("rgb",)))
        self.temporal_mode = str(getattr(cfg, "temporal_mode", "static"))
        self._frame_index = 0
        self._probe_offset = 0.0
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
        spawn = sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            horizontal_aperture=24.0 * cfg.width / fx,
            clipping_range=(self.scene["near_m"], self.scene["far_m"]),
        )
        for index in range(self.total_cameras):
            spawn.func(f"/World/Cameras/Camera_{index}", spawn)
        sim_utils.update_stage()
        self.cameras = []
        for index in range(self.total_cameras):
            camera_cfg = CameraCfg(
                prim_path=f"/World/Cameras/Camera_{index}",
                update_period=0.0,
                width=cfg.width,
                height=cfg.height,
                data_types=list(self.modalities),
                renderer_cfg=IsaacRtxRendererCfg(),
                spawn=spawn,
            )
            self.cameras.append(Camera(camera_cfg))
        self.sim.reset()
        self.set_probe_offset(0.0)
        self.metadata = {
            "renderer": "Isaac RTX",
            "antialiasing": "Off",
            "dlss_frame_generation": False,
            "num_envs": self.num_envs,
            "cameras_per_env": self.cameras_per_env,
            "modalities": list(self.modalities),
            "batch_semantics": "independent_usd_camera_views",
            "light": {
                "type": "DistantLight",
                "intensity": 3000.0,
                "unit": "Isaac/USD native",
            },
            "physics_steps_in_measurement": 0,
            "source_camera_channels": 3 if "rgb" in self.modalities else None,
            "intrinsic_matrix": self.cameras[0]
            .data.intrinsic_matrices.torch[0]
            .cpu()
            .tolist(),
        }

    def capture(self) -> np.ndarray:
        """Render once and return the first RGB image for camera-pilot compatibility."""
        packet = self.capture_packet("host_readback")
        if "rgb" not in packet.arrays:
            raise ValueError("camera-pilot compatibility requires RGB output")
        return packet.arrays["rgb"][0]

    def _render(self) -> None:
        """Advance Kit rendering and refresh the camera output generation."""
        if self.temporal_mode == "moving":
            self._apply_probe_offset(
                self._probe_offset + 0.02 * math.sin(self._frame_index * 0.15)
            )
        self.sim.render()
        for camera in self.cameras:
            camera.update(dt=1 / 60, force_recompute=True)
        self._frame_index += 1

    def _device_arrays(self) -> dict[str, object]:
        """Return native camera buffers in a common batch shape."""
        arrays = {}
        for modality in self.modalities:
            values = [camera.data.output[modality].torch for camera in self.cameras]
            value = self.torch.cat(values, dim=0)
            if modality == "depth" and value.ndim == 3:
                value = value.unsqueeze(-1)
            if modality == "rgb":
                value = value[..., :3]
            if modality == "normals":
                value = value[..., :3]
            arrays[modality] = value
        return arrays

    def capture_packet(self, delivery: str = "host_readback"):
        """Render and return a suite capture packet with transfer accounting."""
        from ..suite import CapturePacket

        self._render()
        device_arrays = self._device_arrays()
        self.torch.cuda.synchronize()
        if delivery == "render_only":
            return CapturePacket(
                arrays=device_arrays,
                render_calls=1,
                readback_calls=0,
                gpu_sync_calls=1,
                host_bytes=0,
                exposure_count=self.total_cameras,
                delivery=delivery,
            )
        host_arrays = {
            name: value.contiguous().cpu().numpy().copy()
            for name, value in device_arrays.items()
        }
        host_bytes = sum(value.nbytes for value in host_arrays.values())
        if delivery == "duplicate_readback":
            host_arrays = {name: value.copy() for name, value in host_arrays.items()}
            host_bytes *= 2
            readbacks = 2
        elif delivery == "host_readback":
            readbacks = 1
        else:
            raise ValueError(f"Unsupported delivery: {delivery}")
        return CapturePacket(
            arrays=host_arrays,
            render_calls=1,
            readback_calls=readbacks,
            gpu_sync_calls=1,
            host_bytes=host_bytes,
            exposure_count=self.total_cameras,
            delivery=delivery,
        )

    def capture_host(self):
        """Return one host packet for validation and sample artifacts."""
        return self.capture_packet("host_readback")

    def set_probe_offset(self, offset: float) -> None:
        """Move all tiled camera views in world x by ``offset`` metres."""
        self._apply_probe_offset(offset)
        self._probe_offset = offset

    def _apply_probe_offset(self, offset: float) -> None:
        """Apply an offset without changing the temporal probe baseline."""
        eye = self.torch.tensor(
            [
                [
                    self.scene["eye"][0] + offset,
                    self.scene["eye"][1],
                    self.scene["eye"][2],
                ]
            ]
            * self.total_cameras,
            device=self.sim.device,
        )
        target = self.torch.tensor(
            [self.scene["target"]] * self.total_cameras,
            device=self.sim.device,
        )
        for camera_index, camera in enumerate(self.cameras):
            camera_eye = eye.clone()
            camera_eye[:, 1] += camera_index * 0.04
            camera.set_world_poses_from_view(camera_eye, target)
        self.sim.render_context.reset_transform_cadence()

    def close(self) -> None:
        """Shut down the isolated Isaac Sim application."""
        self.cameras = []
        self.app.close()
