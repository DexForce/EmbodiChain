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

"""EmbodiChain consumer of the frozen shared rendering scene."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

from ..scenes import camera_intrinsics, load_scene, scene_fingerprint

__all__ = ["EmbodiChainScene", "material_display_color"]


def material_display_color(linear_rgb: list[float]) -> list[float]:
    """Encode canonical linear RGB for VisualMaterialCfg's display-sRGB API.

    Args:
        linear_rgb: Scene-linear RGB reflectance.

    Returns:
        Display-sRGB values whose native decoding recovers the input reflectance.
    """
    return [
        12.92 * value if value <= 0.0031308 else 1.055 * value ** (1 / 2.4) - 0.055
        for value in linear_rgb
    ]


class EmbodiChainScene:
    """Create one camera group per view over identical scene replicas.

    Args:
        scene_path: Frozen manifest and canonical geometry directory.
        width: Output width in pixels.
        height: Output height in pixels.
        gpu_id: Physical CUDA and graphics adapter index.
        num_envs: Number of identical arenas.
        modalities: Requested canonical output names.
    """

    def __init__(
        self,
        scene_path: Path,
        *,
        width: int,
        height: int,
        gpu_id: int,
        num_envs: int = 1,
        modalities: tuple[str, ...] = ("rgb", "depth", "seg"),
    ) -> None:
        import numpy as np
        import torch
        from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
        from embodichain.lab.sim.cfg import (
            DenoisingCfg,
            LightCfg,
            RenderCfg,
            RigidObjectCfg,
        )
        from embodichain.lab.sim.material import VisualMaterialCfg
        from embodichain.lab.sim.sensors import CameraCfg
        from embodichain.lab.sim.shapes import CubeCfg, MeshCfg

        self.scene = load_scene(scene_path)
        self.torch = torch
        self.device = f"cuda:{gpu_id}"
        self.modalities = modalities
        self.num_envs = num_envs
        self.objects = {}
        self.cameras = []
        self.sim = SimulationManager(
            SimulationManagerCfg(
                headless=True,
                width=width,
                height=height,
                device=self.device,
                gpu_id=gpu_id,
                num_envs=num_envs,
                enable_entity_gizmo=False,
                robot_ik_gizmo=None,
                startup_summary="off",
                dexsim_startup_info=False,
                render_cfg=RenderCfg(
                    renderer="hybrid",
                    spp=4,
                    denoising=DenoisingCfg(window="off", offscreen="nrd"),
                    tone_mapping_enabled=False,
                ),
            )
        )
        self.sim.set_ground_plane_visibility(False)
        light = self.scene["lighting"]
        self.sim.set_emission_light(
            color=[1.0, 1.0, 1.0], intensity=light["dexsim_environment_intensity"]
        )
        for obj in self.scene["objects"]:
            material = obj["material"]
            material_cfg = VisualMaterialCfg(
                uid=obj["uid"] + "_material",
                base_color=[
                    *material_display_color(material["base_color_linear"]),
                    1.0,
                ],
                roughness=material["roughness"],
                metallic=material["metallic"],
            )
            geometry = obj["geometry"]
            if geometry["kind"] == "box":
                shape = CubeCfg(size=geometry["size"], visual_material=material_cfg)
            else:
                with np.load(
                    scene_path.parent / geometry["path"], allow_pickle=False
                ) as arrays:
                    shape = MeshCfg(
                        vertices=arrays["vertices"].copy(),
                        triangles=arrays["triangles"].copy(),
                        normals=arrays["normals"].copy(),
                        visual_material=material_cfg,
                    )
            self.objects[obj["uid"]] = self.sim.add_rigid_object(
                RigidObjectCfg(
                    uid=obj["uid"],
                    body_type="static",
                    shape=shape,
                    init_local_pose=np.asarray(obj["world_from_object"]),
                )
            )
        self.sim.add_light(
            LightCfg(
                uid="benchmark_sun",
                light_type="direction",
                direction=tuple(light["direction"]),
                color=tuple(light["color"]),
                intensity=light["dexsim_directional_intensity"],
            )
        )
        self.sim.prepare()
        matrix = camera_intrinsics(self.scene, width, height)
        model = self.scene["camera_model"]
        for camera in self.scene["cameras"]:
            self.cameras.append(
                self.sim.add_sensor(
                    CameraCfg(
                        uid=camera["name"],
                        width=width,
                        height=height,
                        near=model["near_m"],
                        far=model["far_m"],
                        intrinsics=(
                            matrix[0][0],
                            matrix[1][1],
                            matrix[0][2],
                            matrix[1][2],
                        ),
                        enable_color="rgb" in modalities,
                        enable_depth="depth" in modalities,
                        enable_mask="seg" in modalities,
                        enable_normal="normals" in modalities,
                        extrinsics=CameraCfg.ExtrinsicsCfg(
                            eye=tuple(camera["eye"]),
                            target=tuple(camera["target"]),
                            up=tuple(camera["up"]),
                        ),
                    )
                )
            )
        self.sim.prepare()
        self._id_map = [
            (
                self.objects[obj["uid"]].get_user_ids().to(self.device),
                obj["instance_id"],
            )
            for obj in self.scene["objects"]
        ]
        self.metadata = {
            "backend": "embodichain",
            "renderer": "DexSim hybrid",
            "spp": 4,
            "denoising": "nrd",
            "tone_mapping": False,
            "device": self.device,
            "scene_sha256": scene_fingerprint(self.scene),
            "num_envs": num_envs,
            "cameras_per_env": len(self.cameras),
            "camera_names": [c["name"] for c in self.scene["cameras"]],
            "batch_order": "environment_then_camera",
            "intrinsics": [
                camera.get_intrinsics()[0].tolist() for camera in self.cameras
            ],
            "camera_poses": [
                camera.get_arena_pose(to_matrix=True)[0].cpu().tolist()
                for camera in self.cameras
            ],
            "depth_semantics": "optical_axis_metres; invalid_or_beyond_far_is_infinity",
            "lighting": light,
            "physics_steps": 0,
            "render_boundary": "render_camera_group followed by blocking host delivery",
            "quality_status": "not_qualified",
            "material_input_encoding": "linear reflectance encoded as display-sRGB for VisualMaterialCfg",
        }

    def capture(self) -> dict[str, Any]:
        """Return completed host arrays in environment-major camera order."""
        torch = self.torch
        self.sim.render_camera_group([camera.group_id for camera in self.cameras])
        for camera in self.cameras:
            camera.update(fetch_only=True)
        arrays = {}
        for modality in self.modalities:
            key = {
                "rgb": "color",
                "depth": "depth",
                "seg": "mask",
                "normals": "normal",
            }[modality]
            value = torch.stack(
                [camera.get_data()[key] for camera in self.cameras], dim=1
            )
            if modality == "rgb":
                value = value[..., :3]
            if modality == "depth":
                far = self.scene["camera_model"]["far_m"]
                value = torch.where((value > 0) & (value < far), value, torch.inf)
            if modality == "seg":
                mapped = torch.zeros_like(value, dtype=torch.int32)
                for raw_ids, canonical in self._id_map:
                    mapped[value == raw_ids.reshape(self.num_envs, 1, 1, 1)] = canonical
                value = mapped
            arrays[modality] = value.flatten(0, 1).contiguous().cpu().numpy()
        return arrays

    def close(self) -> None:
        """Release native child views before their world."""
        from embodichain.lab.sim import SimulationManager

        self.cameras.clear()
        self.objects.clear()
        self.sim.destroy(exit_process=False)
        self.sim = None
        gc.collect()
        SimulationManager.flush_cleanup_queue()
