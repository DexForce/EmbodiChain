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

"""L-02 execution using the public EmbodiChain scene and object-group APIs."""

from __future__ import annotations
from collections.abc import Callable
import numpy as np
import torch
import warp as wp
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import (
    NewtonPhysicsCfg,
    RenderCfg,
    DenoisingCfg,
    RigidObjectCfg,
    RigidObjectGroupCfg,
    UsdSourceCfg,
    LightCfg,
)
from embodichain.lab.sim.sensors import CameraCfg

__all__ = ["Simulation"]


class Simulation:
    def __init__(self, cfg: dict, mark: Callable[[str], None]) -> None:
        self.cfg = cfg
        self.sim = SimulationManager(
            SimulationManagerCfg(
                headless=True,
                num_envs=1,
                physics_cfg=NewtonPhysicsCfg(
                    device="cuda:0",
                    gravity=(0, 0, 0),
                    physics_dt=cfg["dt"],
                    num_substeps=1,
                    use_cuda_graph=True,
                    solver_cfg={
                        "solver_type": "dexuni",
                        "joint_mode": "dynamic",
                        "vbd_options": {
                            "iterations": cfg["dexuni_iterations"],
                            "rigid_enable_sleep": True,
                        },
                        "collision_options": {
                            "rigid_contact_max": max(4096, 2 * cfg["object_count"])
                        },
                    },
                ),
                render_cfg=RenderCfg(
                    renderer="hybrid",
                    spp=8 if cfg.get("record") else 1,
                    denoising=DenoisingCfg(
                        window="off", offscreen="optix" if cfg.get("record") else "off"
                    ),
                ),
            )
        )
        self.sim.set_ground_plane_visibility(False)
        self.sim.set_emission_light([1.0, 1.0, 1.0], 20.0 if cfg.get("record") else 1.0)
        mark("engine_initialized")
        self.group = self.sim.add_rigid_object_group(
            RigidObjectGroupCfg(
                uid="objects",
                rigid_objects={
                    f"object_{i:05d}": RigidObjectCfg(
                        uid=f"object_{i:05d}",
                        source=UsdSourceCfg(file_path=a["path"]),
                        asset_physics_mode="preserve",
                        init_pos=p,
                    )
                    for i, (a, p) in enumerate(zip(cfg["objects"], cfg["positions"]))
                },
            )
        )
        self.sim.add_light(
            LightCfg(
                uid="key", light_type="direction", direction=(-1, -1, -2), intensity=3.0
            )
        )
        c = cfg["camera"]
        self.camera = self.sim.add_sensor(
            CameraCfg(
                uid="camera",
                width=c["width"],
                height=c["height"],
                intrinsics=tuple(c["intrinsics"]),
                extrinsics=CameraCfg.ExtrinsicsCfg(
                    eye=tuple(c["eye"]), target=tuple(c["target"]), up=(0, 0, 1)
                ),
            )
        )
        mark("declared")
        self.sim.prepare()
        wp.synchronize()
        torch.cuda.synchronize()
        mark("prepared")
        self.meta = {
            "physics": "Newton / DexUni VBD 10",
            "renderer": (
                "hybrid, spp=8, OptiX, environment intensity 20 (media only)"
                if cfg.get("record")
                else "hybrid, spp=1, denoising off, environment intensity 1"
            ),
            "gravity": [0, 0, 0],
            "cache": "New process, USD parser no persistent stage cache, analytic colliders bypass mesh cooking; shader/kernel caches retained; native renderer asset cache not externally controlled",
        }

    def step(self) -> np.ndarray:
        self.sim.update(step=1, render_final_step=False)
        state = (
            torch.cat(
                (
                    self.group.get_local_pose(),
                    self.group.body_data.lin_vel,
                    self.group.body_data.ang_vel,
                ),
                dim=-1,
            )[0]
            .cpu()
            .numpy()
        )
        wp.synchronize()
        torch.cuda.synchronize()
        return state

    def render(self) -> np.ndarray:
        self.sim.render_camera_group([self.camera.group_id])
        self.camera.update(fetch_only=True)
        return self.camera.get_data()["color"][0].cpu().numpy()[..., :3]

    def audit(self) -> tuple[dict, dict]:
        from dexsim.engine.newton_physics.backend_registry import get_newton_backend

        b = get_newton_backend(self.sim._world)
        model = b.model
        arrays = {
            "mass": self.group.get_mass()[0].cpu().numpy(),
            "inertia": self.group.get_inertia()[0].cpu().numpy(),
            "pose": self.group.get_local_pose()[0].cpu().numpy(),
        }
        native = {}
        for name in [
            "shape_type",
            "shape_scale",
            "shape_flags",
            "shape_body",
            "shape_transform",
            "body_mass",
            "body_inertia",
        ]:
            v = getattr(model, name, None)
            native[name] = v.numpy().tolist() if hasattr(v, "numpy") else str(v)
        # Public group geometry inspection is after timing, all objects included.
        meshes = []
        materials = []
        for i in range(self.cfg["object_count"]):
            v = self.group.get_object_vertices(i)[0].cpu().numpy()
            f = self.group.get_object_triangles(i)[0].cpu().numpy()
            meshes.append(
                {
                    "vertices": len(v),
                    "triangles": len(f),
                    "extent": (v.max(0) - v.min(0)).tolist(),
                }
            )
        for entity in self.group._entities[0]:
            body = entity.get_render_body()
            mat = body.get_material(0)
            materials.append(
                dict(
                    id=mat.get_material_id(),
                    name=mat.get_name(),
                    color_linear=np.asarray(mat.get_base_color_linear()).tolist(),
                )
            )
        return arrays, dict(
            **self.meta,
            effective_sim_config=self.sim.sim_config.to_dict(),
            native=native,
            meshes=meshes,
            materials=materials,
            camera_world_opengl=self.camera.get_local_pose(to_matrix=True)[0]
            .cpu()
            .tolist(),
            intrinsics=self.camera.get_intrinsics()[0].cpu().tolist(),
        )

    def close(self) -> None:
        pass
