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

"""P-03 physical execution through EmbodiChain public simulation interfaces."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import (
    NewtonPhysicsCfg,
    RenderCfg,
    DenoisingCfg,
    RobotCfg,
    UsdSourceCfg,
    JointDrivePropertiesCfg,
    ArticulationRootPropertiesCfg,
    RigidObjectCfg,
    RigidBodyPhysicsCfg,
    RigidBodyMaterialCfg,
    NewtonCollisionPropertiesCfg,
)
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.lab.sim.material import VisualMaterialCfg

__all__ = ["Simulation"]


class Simulation:
    """One robot and one target per isolated environment."""

    def __init__(
        self, cfg: dict, inputs: dict, shape: str, n: int, record: bool, output: Path
    ) -> None:
        self.cfg, self.inputs, self.n, self.output = cfg, inputs, n, output
        self.device = "cuda:0"
        self.names = [f"panda_joint{i}" for i in range(1, 8)] + [
            "panda_finger_joint1",
            "panda_finger_joint2",
        ]
        self.sim = SimulationManager(
            SimulationManagerCfg(
                headless=True,
                num_envs=n,
                physics_cfg=NewtonPhysicsCfg(
                    device=self.device,
                    physics_dt=cfg["dt"],
                    num_substeps=1,
                    use_cuda_graph=True,
                    solver_cfg={
                        "solver_type": "dexuni",
                        "joint_mode": "dynamic",
                        "coupling": cfg.get("coupling", "one_way"),
                        "vbd_options": {
                            "iterations": cfg["dexuni_iterations"],
                            "rigid_enable_sleep": True,
                        },
                        "collision_options": {"rigid_contact_max": max(4096, n * 256)},
                    },
                ),
                render_cfg=RenderCfg(
                    renderer="hybrid" if record else "no-render",
                    spp=8,
                    denoising=DenoisingCfg(
                        window="off", offscreen="optix" if record else "off"
                    ),
                ),
            )
        )
        material = RigidBodyMaterialCfg(
            static_friction=cfg["mu"], dynamic_friction=cfg["mu"], restitution=0
        )
        collision = NewtonCollisionPropertiesCfg(
            margin=cfg["rest_offset_m"],
            gap=cfg["contact_offset_m"] - cfg["rest_offset_m"],
        )
        attrs = RigidBodyPhysicsCfg(material_props=material, collision_props=collision)
        self.robot = self.sim.add_robot(
            RobotCfg(
                uid="franka",
                source=UsdSourceCfg(file_path=cfg["asset"]),
                init_pos=cfg["robot_base"],
                init_qpos=inputs["q_target"][0, 0].tolist(),
                asset_physics_mode="overlay",
                attrs=attrs,
                root_props=ArticulationRootPropertiesCfg(
                    fixed_base=True, self_collision_enabled=False
                ),
                joint_drive_props=JointDrivePropertiesCfg(
                    drive_type="force",
                    stiffness={
                        "panda_joint[1-7]": cfg["arm_kp"],
                        "panda_finger_joint1": cfg["finger_kp"],
                        "panda_finger_joint2": 0.0,
                    },
                    damping={
                        "panda_joint[1-7]": cfg["arm_kd"],
                        "panda_finger_joint1": cfg["finger_kd"],
                        "panda_finger_joint2": 0.0,
                    },
                    max_effort={
                        "panda_joint[1-4]": 87.0,
                        "panda_joint[5-7]": 12.0,
                        "panda_finger_joint[1-2]": cfg["finger_force"],
                    },
                    max_velocity=dict(zip(self.names, cfg["joint_velocity_limits"])),
                    friction=0.0,
                    armature=0.0,
                ),
            )
        )
        self.table = self.sim.add_rigid_object(
            RigidObjectCfg(
                uid="table",
                body_type="static",
                init_pos=cfg["table_center"],
                shape=CubeCfg(
                    size=cfg["table_size"],
                    visual_material=VisualMaterialCfg(
                        uid="table_material", base_color=[0.35, 0.38, 0.42, 1]
                    ),
                ),
                attrs=attrs,
            )
        )
        self.target = self.sim.add_rigid_object(
            RigidObjectCfg(
                uid="target",
                source=UsdSourceCfg(
                    file_path=str(Path(cfg["shape_dir"]) / f"{shape}.usda")
                ),
                init_pos=[0.55, 0.5, 0.775],
                asset_physics_mode=cfg.get("target_asset_physics_mode", "preserve"),
                attrs=attrs,
            )
        )
        self.camera = None
        if record:
            from embodichain.lab.sim.sensors import CameraCfg

            c = cfg["capture"]
            self.camera = self.sim.add_sensor(
                CameraCfg(
                    uid="p03_camera",
                    width=c["width"],
                    height=c["height"],
                    intrinsics=tuple(c["intrinsics"]),
                    extrinsics=CameraCfg.ExtrinsicsCfg(
                        eye=tuple(c["eye"]), target=tuple(c["target"]), up=(0, 0, 1)
                    ),
                )
            )
        self.sim.prepare()
        assert self.sim.physics.solver_type == "dexuni"
        self.order = [self.names.index(name) for name in self.robot.joint_names]
        self.links = [
            "panda_fingertip_centered",
            "panda_leftfinger",
            "panda_rightfinger",
        ]
        self.meta = dict(
            solver="DexUni",
            effective_sim_config=self.sim.sim_config.to_dict(),
            joint_names=self.robot.joint_names,
            link_names=self.robot.link_names,
            robot_mass=self.robot.get_mass().cpu().tolist(),
            robot_inertia=self.robot.get_inertia().cpu().tolist(),
            robot_com=self.robot.get_com_pose().cpu().tolist(),
            drives=[x.cpu().tolist() for x in self.robot.get_joint_drive()],
            mimic=repr(self.robot._mimic_info),
            target_mass=self.target.get_mass().cpu().tolist(),
            target_inertia=self.target.get_inertia().cpu().tolist(),
            target_friction=self.target.get_friction().cpu().tolist(),
            contact_export="Not available through public DexUni ContactQuery; geometric estimates retained separately.",
        )
        self.force = torch.zeros((n, 3), device=self.device)
        # Read-only model audit; execution still uses the public facade above.
        if n <= 2:
            from dexsim.engine.newton_physics.backend_registry import get_newton_backend
            import json

            backend = get_newton_backend(self.sim._world)
            audit = {
                "features": repr(backend.solver.features),
                "ownership": repr(backend.solver.ownership),
            }
            for key in [
                "body_label",
                "shape_label",
                "shape_body",
                "shape_transform",
                "shape_scale",
                "shape_type",
                "shape_flags",
                "shape_collision_group",
                "shape_collision_filter_pairs",
                "shape_world",
                "shape_material_mu",
                "shape_margin",
                "shape_gap",
            ]:
                value = getattr(backend.model, key, None)
                if hasattr(value, "numpy"):
                    value = value.numpy().tolist()
                audit[key] = value
            mj = backend.mujoco_solver
            if mj is not None:
                for key in [
                    "geom_contype",
                    "geom_conaffinity",
                    "geom_bodyid",
                    "geom_size",
                    "geom_pos",
                    "geom_quat",
                    "geom_friction",
                ]:
                    value = getattr(mj.mj_model, key, None)
                    audit["mj_" + key] = (
                        value.tolist() if hasattr(value, "tolist") else value
                    )
            (output / "model-audit.json").write_text(
                json.dumps(audit, indent=2, default=str) + "\n"
            )
        self.env = None
        if cfg.get("execution_mode") == "env":
            from embodichain.lab.gym.envs.base_env import BaseEnv, EnvCfg
            from gymnasium.spaces import Box

            adapter = self
            self.last_states = torch.empty((4, n, 52), device=self.device)

            class GraspEnv(BaseEnv):
                """Benchmark task using the complete inherited BaseEnv.step."""

                def _setup_scene(env, **kwargs):
                    env.sim = adapter.sim

                def _setup_robot(env, **kwargs):
                    env.single_action_space = Box(
                        -np.inf, np.inf, (9,), dtype=np.float32
                    )
                    env.active_joint_ids = list(range(9))
                    return adapter.robot

                def _step_action(env, action):
                    adapter.command(action)
                    return action

                def _advance_physics(env):
                    for j in range(4):
                        adapter.step(adapter.pending_forces[j])
                        adapter.last_states[j] = adapter.state()

                def get_obs(env, **kwargs):
                    state = adapter.state()
                    return {"policy": torch.cat([state[:, :25], state[:, 39:]], dim=1)}

                def get_info(env, **kwargs):
                    info = super().get_info(**kwargs)
                    state = adapter.state()
                    info["finite"] = torch.isfinite(state).all(dim=1)
                    info["object_height"] = state[:, 41]
                    return info

            self.env = GraspEnv(
                EnvCfg(
                    num_envs=n,
                    sim_cfg=self.sim.sim_config,
                    sim_steps_per_control=4,
                    max_episode_steps=500,
                    seed=20261022,
                )
            )
            self.env._replay_no_auto_reset = True
            self.meta["env_step"] = (
                "Inherited EmbodiChain BaseEnv.step; four public physical substeps with sampling; no auto reset."
            )

    def reset(self, episode: int) -> None:
        if self.env is not None:
            self.env._elapsed_steps.zero_()
        self.sim.reset_objects_state()
        q = torch.tensor(
            self.inputs["q_target"][episode, 0, self.order],
            dtype=torch.float32,
            device=self.device,
        ).repeat(self.n, 1)
        self.robot.set_qpos(q, target=False)
        self.robot.set_qvel(torch.zeros_like(q), target=False)
        self.robot.set_qpos(q)
        pose = torch.tensor(
            self.inputs["object_initial_pose"][episode],
            dtype=torch.float32,
            device=self.device,
        ).repeat(self.n, 1)
        self.target.set_local_pose(pose)
        self.target.set_velocity(
            lin_vel=torch.zeros((self.n, 3), device=self.device),
            ang_vel=torch.zeros((self.n, 3), device=self.device),
        )

    def command(self, q: torch.Tensor) -> None:
        self.robot.set_qpos(q[:, self.order])

    def step(self, force: torch.Tensor) -> None:
        self.target.add_force_torque(force=force)
        self.sim.update(step=1, render_final_step=False)

    def control_step(self, q: torch.Tensor, forces: torch.Tensor) -> None:
        self.pending_forces = forces
        self.env.step(q)

    def state(self) -> torch.Tensor:
        return torch.cat(
            [
                self.robot.get_qpos(),
                self.robot.get_qvel(),
                *[self.robot.get_link_pose(x) for x in self.links],
                self.target.body_state,
            ],
            dim=1,
        )

    def render(self) -> tuple[np.ndarray, dict]:
        self.sim.render_camera_group([self.camera.group_id])
        self.camera.update(fetch_only=True)
        return self.camera.get_data()["color"][0].cpu().numpy(), dict(
            camera_world_opengl=self.camera.get_local_pose(to_matrix=True)[0]
            .cpu()
            .tolist(),
            intrinsics=self.camera.get_intrinsics()[0].cpu().tolist(),
        )

    def close(self) -> None:
        self.sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()
