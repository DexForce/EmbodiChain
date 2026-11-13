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

"""L-02: one asset declaration replicated to num_envs native arenas."""

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
    """Use public scene replication; backend arrays are read only for audit."""

    def __init__(self, cfg: dict, mark: Callable[[str], None]) -> None:
        self.cfg = cfg
        self.sim = SimulationManager(
            SimulationManagerCfg(
                headless=True,
                num_envs=cfg["num_envs"],
                arena_space=cfg["env_spacing"],
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
                            "rigid_enable_sleep": cfg.get("rigid_enable_sleep", True),
                        },
                        "collision_options": {
                            "rigid_contact_max": max(4096, 2 * cfg["num_envs"])
                        },
                    },
                ),
                render_cfg=RenderCfg(
                    renderer="hybrid" if cfg["record"] else "no-render",
                    spp=8,
                    denoising=DenoisingCfg(
                        window="off", offscreen="optix" if cfg["record"] else "off"
                    ),
                ),
            )
        )
        mark("engine_initialized")
        self.group = self.sim.add_rigid_object_group(
            RigidObjectGroupCfg(
                uid="objects",
                rigid_objects={
                    "object": RigidObjectCfg(
                        uid="object",
                        source=UsdSourceCfg(file_path=cfg["objects"][0]["path"]),
                        asset_physics_mode="preserve",
                        init_pos=cfg["positions"][0],
                    )
                },
            )
        )
        mark("declared")
        self.sim.prepare()
        wp.synchronize()
        torch.cuda.synchronize()
        mark("prepared")
        self.camera = None

    def step(self) -> np.ndarray:
        self.sim.update(step=1, render_final_step=False)
        state = self.group.body_state.cpu().numpy()
        wp.synchronize()
        torch.cuda.synchronize()
        return state

    def audit(self) -> tuple[dict, dict]:
        from dexsim.engine.newton_physics.backend_registry import get_newton_backend

        backend = get_newton_backend(self.sim.get_world())
        model = backend.model
        sleeping = backend.solver.vbd_solver._rigid_sleep is not None
        assert sleeping == self.cfg.get("rigid_enable_sleep", True)
        arrays = {
            "mass": self.group.get_mass().cpu().numpy(),
            "inertia": self.group.get_inertia().cpu().numpy(),
            "pose": self.group.get_local_pose().cpu().numpy(),
            "env_origins": self.sim.arena_offsets.cpu().numpy(),
        }
        for name in [
            "shape_type",
            "shape_scale",
            "shape_body",
            "body_mass",
            "body_world",
            "shape_world",
            "body_q",
        ]:
            v = getattr(model, name, None)
            if hasattr(v, "numpy"):
                arrays[name] = v.numpy()
        return arrays, {
            "num_envs": self.sim.num_envs,
            "objects_per_env": self.group.num_objects,
            "instance_count": self.group.num_instances,
            "arena_names": [
                self.sim.get_env(i).get_name() for i in range(self.sim.num_envs)
            ],
            "replication": "SimulationManagerCfg.num_envs -> SpawnScene replicated arenas",
            "collision_isolation": "native arena world IDs; overlap probe saved for pilots",
            "effective_config": self.sim.sim_config.to_dict(),
            "physics": f"Newton / DexUni VBD {self.cfg['dexuni_iterations']}, sleep {sleeping}",
            "rigid_enable_sleep": sleeping,
            "body_count": model.body_count,
            "renderer": "hybrid (media)" if self.cfg["record"] else "no-render",
        }

    def isolation_probe(self) -> dict:
        before = self.step()
        # Put env 0 and env 1 at the same world location. With isolated arenas,
        # interpenetration must not generate cross-environment collision response.
        pose = self.group.get_local_pose()[0:1].clone()
        pose[0, 0, :3] += self.sim.arena_offsets[1] - self.sim.arena_offsets[0]
        pose[0, 0, 0] += 0.01
        self.group.set_local_pose(pose, env_ids=[0])
        after_write = self.group.body_state.cpu().numpy().copy()
        states = np.stack([self.step() for _ in range(20)])
        return {"before": before, "after_write": after_write, "states": states}

    def render(self) -> np.ndarray:
        if self.camera is None:
            self.sim.set_ground_plane_visibility(False)
            self.sim.set_emission_light([1.0, 1.0, 1.0], 20.0)
            self.sim.add_light(
                LightCfg(
                    uid="key",
                    light_type="direction",
                    direction=(-1, -1, -2),
                    intensity=3.0,
                )
            )
            c = self.cfg["camera"]
            center = self.sim.arena_offsets.cpu().numpy().mean(0)
            self.camera = self.sim.add_sensor(
                CameraCfg(
                    uid="camera",
                    width=c["width"],
                    height=c["height"],
                    intrinsics=tuple(c["intrinsics"]),
                    extrinsics=CameraCfg.ExtrinsicsCfg(
                        eye=tuple(np.asarray(c["eye"]) + center),
                        target=tuple(np.asarray(c["target"]) + center),
                        up=(0, 0, 1),
                    ),
                )
            )
        self.sim.render_camera_group([self.camera.group_id])
        self.camera.update(fetch_only=True)
        return self.camera.get_data()["color"][0].cpu().numpy()[..., :3]

    def camera_audit(self) -> dict:
        return {
            "camera_world_opengl": self.camera.get_local_pose(to_matrix=True)[0]
            .cpu()
            .tolist(),
            "intrinsics": self.camera.get_intrinsics()[0].cpu().tolist(),
            "camera_instances": self.camera.num_instances,
        }
