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

"""L-03: replicate one articulation through EmbodiChain public scene APIs."""

from __future__ import annotations

from collections.abc import Callable
import numpy as np
import torch
import warp as wp
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import (
    ArticulationCfg,
    ArticulationRootPropertiesCfg,
    JointDrivePropertiesCfg,
    NewtonPhysicsCfg,
    RenderCfg,
    DenoisingCfg,
    UsdSourceCfg,
    LightCfg,
)
from embodichain.lab.sim.sensors import CameraCfg

__all__ = ["Simulation"]


class Simulation:
    """Prepare a complete passive robot in every arena; audit native arrays read only."""

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
                        "vbd_options": {"iterations": 100, "rigid_enable_sleep": True},
                        "collision_options": {"rigid_contact_max": 4096},
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
        self.robot = self.sim.add_articulation(
            ArticulationCfg(
                uid="robot",
                source=UsdSourceCfg(file_path=cfg["usd_path"]),
                init_pos=cfg["initial_position"],
                asset_physics_mode="overlay",
                root_props=ArticulationRootPropertiesCfg(
                    fixed_base=cfg["fixed_base"], self_collision_enabled=False
                ),
                joint_drive_props=JointDrivePropertiesCfg(
                    drive_type="none",
                    target_mode="none",
                    stiffness=0.0,
                    damping=0.0,
                    friction=0.0,
                    armature=0.0,
                ),
            )
        )
        mark("declared")
        self.sim.prepare()
        wp.synchronize()
        torch.cuda.synchronize()
        mark("prepared")
        self.names = self.robot.joint_names
        self.links = self.robot.link_names
        q = torch.tensor(
            [[cfg["initial_qpos"][n] for n in self.names]], device=self.sim.device
        ).repeat(cfg["num_envs"], 1)
        self.robot.set_qpos(q, target=False)
        self.robot.set_qvel(torch.zeros_like(q), target=False)
        self.robot.set_local_pose(
            torch.tensor(
                [cfg["initial_position"] + [0, 0, 0, 1]], device=self.sim.device
            ).repeat(cfg["num_envs"], 1)
        )
        self.robot.set_root_velocity(
            torch.zeros((cfg["num_envs"], 6), device=self.sim.device)
        )
        self.initial = self.sample()
        self.camera = None

    def sample(self) -> dict[str, np.ndarray]:
        """Read every environment's joint, root and link state in local coordinates."""
        state = {
            "qpos": self.robot.get_qpos().cpu().numpy().copy(),
            "qvel": self.robot.get_qvel().cpu().numpy().copy(),
            "link_pose": torch.stack(
                [self.robot.get_link_pose(n) for n in self.links], dim=1
            )
            .cpu()
            .numpy()
            .copy(),
            "root_pose": self.robot.get_local_pose().cpu().numpy().copy(),
            "root_velocity": self.robot.body_data.root_vel.cpu().numpy().copy(),
        }
        wp.synchronize()
        torch.cuda.synchronize()
        return state

    def step(self) -> dict[str, np.ndarray]:
        """Complete one physics step and read all replicated states."""
        self.sim.update(step=1, render_final_step=False)
        return self.sample()

    def audit(self) -> tuple[dict, dict]:
        """Save source-independent native counts and per-environment physical properties."""
        from dexsim.engine.newton_physics.backend_registry import get_newton_backend

        backend = get_newton_backend(self.sim.get_world())
        model = backend.model
        arrays = {
            "mass": self.robot.get_mass().cpu().numpy(),
            "inertia_principal": self.robot.get_inertia().cpu().numpy(),
            "com_pose": self.robot.get_com_pose().cpu().numpy(),
            "joint_limits": self.robot.get_qpos_limits().cpu().numpy(),
            "env_origins": self.sim.arena_offsets.cpu().numpy(),
        }
        for key in (
            "body_world",
            "body_mass",
            "body_q",
            "shape_type",
            "shape_scale",
            "shape_body",
            "shape_world",
            "shape_flags",
            "shape_transform",
            "joint_type",
            "joint_parent",
            "joint_child",
        ):
            arrays["native_" + key] = getattr(model, key).numpy()
        return arrays, {
            "num_envs": self.sim.num_envs,
            "instance_count": self.robot.num_instances,
            "arena_names": [
                self.sim.get_env(i).get_name() for i in range(self.sim.num_envs)
            ],
            "joint_names": self.names,
            "link_names": self.links,
            "native_body_count": model.body_count,
            "native_joint_dof_count": model.joint_dof_count,
            "native_joint_labels": list(model.joint_label),
            "native_shape_labels": list(model.shape_label),
            "vbd_active": backend.solver.vbd_solver is not None,
            "effective_vbd_sleep": (
                backend.solver.vbd_solver._rigid_sleep is not None
                if backend.solver.vbd_solver is not None
                else None
            ),
            "articulation_solver": type(backend.solver.mujoco_solver).__name__,
            "config": self.sim.sim_config.to_dict(),
        }

    def close(self) -> None:
        """Queue destruction; the entry point flushes after child references unwind."""
        self.sim.destroy(exit_process=False)

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
