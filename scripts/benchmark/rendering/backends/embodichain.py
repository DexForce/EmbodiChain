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

"""EmbodiChain adapter for a render-only procedural camera pilot."""

from __future__ import annotations

from typing import TYPE_CHECKING
import gc

if TYPE_CHECKING:
    import numpy as np
    from ..workload import PilotCfg

__all__ = ["EmbodiChainCamera"]


class EmbodiChainCamera:
    """Own one native world and expose completed HWC RGB captures."""

    def __init__(self, cfg: PilotCfg) -> None:
        import torch
        from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
        from embodichain.lab.sim.cfg import RenderCfg, DLSSCfg, RigidObjectCfg, LightCfg
        from embodichain.lab.sim.shapes import CubeCfg
        from embodichain.lab.sim.material import VisualMaterialCfg
        from embodichain.lab.sim.sensors import CameraCfg
        from ..workload import scene_spec
        import math

        self.torch = torch
        self.scene = scene_spec()
        self.sim = SimulationManager(
            SimulationManagerCfg(
                headless=True,
                device="cuda:0",
                num_envs=1,
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
        self.camera = self.sim.add_sensor(
            CameraCfg(
                uid="camera",
                width=cfg.width,
                height=cfg.height,
                near=self.scene["near_m"],
                far=self.scene["far_m"],
                intrinsics=(fx, fx, cfg.width / 2, cfg.height / 2),
                extrinsics=CameraCfg.ExtrinsicsCfg(
                    eye=tuple(self.scene["eye"]),
                    target=tuple(self.scene["target"]),
                ),
            )
        )
        self.sim.prepare()
        self.metadata = {
            "renderer": "DexSim hybrid",
            "dlss": False,
            "light": {
                "type": "direction",
                "direction": [0, 0, -1],
                "intensity": 3.0,
                "unit": "W/m2",
            },
            "environment_emission_intensity": 100.0,
            "physics_steps_in_measurement": 0,
            "source_camera_channels": 4,
            "intrinsic_matrix": self.camera.get_intrinsics()[0].tolist(),
        }

    def capture(self) -> np.ndarray:
        """Render once and return completed RGB bytes on the host."""
        self.sim.render_camera_group([self.camera.group_id])
        self.camera.update(fetch_only=True)
        return (
            self.camera.get_data()["color"][0, ..., :3]
            .contiguous()
            .cpu()
            .numpy()
            .copy()
        )

    def set_probe_offset(self, offset: float) -> None:
        """Move the camera eye in world x by offset [m], without stepping."""
        eye = list(self.scene["eye"])
        eye[0] += offset
        tensor = self.torch.tensor
        self.camera.look_at(
            tensor([eye], device=self.sim.device),
            tensor([self.scene["target"]], device=self.sim.device),
            tensor([[0.0, 0.0, 1.0]], device=self.sim.device),
        )

    def close(self) -> None:
        """Release camera owners before draining deferred world destruction."""
        from embodichain.lab.sim import SimulationManager

        self.camera = None
        self.sim.destroy(exit_process=False)
        self.sim = None
        gc.collect()
        SimulationManager.flush_cleanup_queue()
