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

"""EmbodiChain adapter for the pure-rendering R-series suite."""

from __future__ import annotations

import gc
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from ..suite import RenderCaseCfg
    from ..workload import PilotCfg

__all__ = ["EmbodiChainCamera"]


def _install_dexsim_material_compat() -> bool:
    """Skip an optional native refresh call absent in older DexSim wheels.

    EmbodiChain's spawn adapter is shared with newer DexSim builds where
    ``RenderBody.refresh_materials`` exists. The installed 0.5 runtime used by
    this benchmark can materialize the same actor without that optional method.
    The shim is process-local and leaves newer runtimes unchanged.
    """
    try:
        from dexsim.spawn.adapters import common
    except ImportError:
        return False
    if getattr(common.refresh_render_body_materials, "_embodichain_compat", False):
        return True

    def refresh(render_body: object | None) -> None:
        if render_body is None:
            return
        method = getattr(render_body, "refresh_materials", None)
        if method is not None:
            method()

    refresh._embodichain_compat = True
    common.refresh_render_body_materials = refresh
    return True


class EmbodiChainCamera:
    """Own one native world and expose completed batched captures."""

    def __init__(self, cfg: PilotCfg | RenderCaseCfg) -> None:
        import torch
        from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
        from embodichain.lab.sim.cfg import DLSSCfg, LightCfg, RenderCfg, RigidObjectCfg
        from embodichain.lab.sim.material import VisualMaterialCfg
        from embodichain.lab.sim.sensors import CameraCfg
        from embodichain.lab.sim.shapes import CubeCfg
        from ..suite import scene_spec

        material_compat = _install_dexsim_material_compat()
        self.cfg = cfg
        self.torch = torch
        self.scene = scene_spec()
        self.num_envs = int(getattr(cfg, "num_envs", 1))
        self.cameras_per_env = int(getattr(cfg, "cameras_per_env", 1))
        self.modalities = tuple(getattr(cfg, "modalities", ("rgb",)))
        self.temporal_mode = str(getattr(cfg, "temporal_mode", "static"))
        self._frame_index = 0
        self._probe_offset = 0.0
        self.sim = SimulationManager(
            SimulationManagerCfg(
                headless=True,
                device="cuda:0",
                num_envs=self.num_envs,
                physics_dt=1 / 240,
                render_cfg=RenderCfg(
                    renderer="hybrid", dlss=DLSSCfg(dlss_enabled=False)
                ),
                enable_entity_gizmo=False,
                robot_ik_gizmo=None,
                startup_summary="off",
            )
        )
        self.sim.set_ground_plane_visibility(False)
        for box in self.scene["boxes"]:
            self.sim.add_rigid_object(
                RigidObjectCfg(
                    uid=box["id"],
                    body_type="static",
                    init_pos=tuple(box["position"]),
                    shape=CubeCfg(
                        size=box["size"],
                        visual_material=VisualMaterialCfg(
                            uid=box["id"] + "_material",
                            base_color=[*box["color"], 1.0],
                            roughness=1.0,
                            metallic=0.0,
                        ),
                    ),
                )
            )
        self.sim.add_light(
            LightCfg(
                uid="sun",
                light_type="direction",
                direction=(0.0, 0.0, -1.0),
                color=(1.0, 1.0, 1.0),
                intensity=3.0,
            )
        )
        self.sim.prepare()
        fx = cfg.width / (
            2 * math.tan(math.radians(self.scene["horizontal_fov_deg"]) / 2)
        )
        self.cameras = []
        for camera_index in range(self.cameras_per_env):
            eye = list(self.scene["eye"])
            eye[1] += camera_index * 0.04
            self.cameras.append(
                self.sim.add_sensor(
                    CameraCfg(
                        uid=f"camera_{camera_index}",
                        width=cfg.width,
                        height=cfg.height,
                        near=self.scene["near_m"],
                        far=self.scene["far_m"],
                        intrinsics=(fx, fx, cfg.width / 2, cfg.height / 2),
                        enable_color="rgb" in self.modalities,
                        enable_depth="depth" in self.modalities,
                        enable_normal="normals" in self.modalities,
                        extrinsics=CameraCfg.ExtrinsicsCfg(
                            eye=tuple(eye),
                            target=tuple(self.scene["target"]),
                        ),
                    )
                )
            )
        self.sim.prepare()
        self.metadata = {
            "renderer": "DexSim hybrid",
            "dlss": False,
            "num_envs": self.num_envs,
            "cameras_per_env": self.cameras_per_env,
            "modalities": list(self.modalities),
            "batch_semantics": "arena_camera_groups",
            "light": {
                "type": "direction",
                "direction": [0, 0, -1],
                "intensity": 3.0,
                "unit": "W/m2",
            },
            "environment_emission_intensity": 100.0,
            "physics_steps_in_measurement": 0,
            "source_camera_channels": 4 if "rgb" in self.modalities else None,
            "dexsim_material_refresh_compat": material_compat,
            "intrinsic_matrix": self.cameras[0].get_intrinsics()[0].tolist(),
        }

    def capture(self) -> np.ndarray:
        """Render once and return the first RGB image for camera-pilot compatibility."""
        packet = self.capture_packet("host_readback")
        if "rgb" not in packet.arrays:
            raise ValueError("camera-pilot compatibility requires RGB output")
        return packet.arrays["rgb"][0]

    def _render(self) -> None:
        """Issue one render for all configured camera groups."""
        if self.temporal_mode == "moving":
            self._apply_probe_offset(
                self._probe_offset + 0.02 * math.sin(self._frame_index * 0.15)
            )
        self.sim.render_camera_group([camera.group_id for camera in self.cameras])
        for camera in self.cameras:
            camera.update(fetch_only=True)
        self._frame_index += 1

    def _device_arrays(self) -> dict[str, object]:
        """Concatenate all camera groups into one batch per modality."""
        keys = {"rgb": "color", "depth": "depth", "normals": "normal"}
        arrays = {}
        for modality in self.modalities:
            tensors = [camera.get_data()[keys[modality]] for camera in self.cameras]
            value = self.torch.cat(tensors, dim=0)
            if modality == "rgb":
                value = value[..., :3]
            elif modality == "depth":
                value = value.unsqueeze(-1)
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
                exposure_count=self.num_envs * self.cameras_per_env,
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
            exposure_count=self.num_envs * self.cameras_per_env,
            delivery=delivery,
        )

    def capture_host(self):
        """Return one host packet for validation and sample artifacts."""
        return self.capture_packet("host_readback")

    def set_probe_offset(self, offset: float) -> None:
        """Move every camera eye in world x by ``offset`` metres."""
        self._apply_probe_offset(offset)
        self._probe_offset = offset

    def _apply_probe_offset(self, offset: float) -> None:
        """Apply an offset without changing the temporal probe baseline."""
        eye = list(self.scene["eye"])
        eye[0] += offset
        tensor = self.torch.tensor
        for camera_index, camera in enumerate(self.cameras):
            camera_eye = list(eye)
            camera_eye[1] += camera_index * 0.04
            camera.look_at(
                tensor([camera_eye] * self.num_envs, device=self.sim.device),
                tensor([self.scene["target"]] * self.num_envs, device=self.sim.device),
                tensor([[0.0, 0.0, 1.0]] * self.num_envs, device=self.sim.device),
            )

    def close(self) -> None:
        """Release camera owners before draining deferred world destruction."""
        from embodichain.lab.sim import SimulationManager

        self.cameras = []
        self.sim.destroy(exit_process=False)
        self.sim = None
        gc.collect()
        SimulationManager.flush_cleanup_queue()
