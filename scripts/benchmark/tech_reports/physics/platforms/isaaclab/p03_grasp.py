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

"""P-03 GPU PhysX execution; import after Isaac AppLauncher initialization."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
import isaaclab.sim as su
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab_physx.physics import PhysxCfg

__all__ = ["Simulation"]


class Simulation:
    """Matching source robot, target geometry and physical command stream."""

    def __init__(
        self, cfg: dict, inputs: dict, shape: str, n: int, record: bool, output: Path
    ) -> None:
        self.cfg, self.inputs, self.n, self.output = cfg, inputs, n, output
        self.device = "cuda:0"
        self.names = [f"panda_joint{i}" for i in range(1, 8)] + [
            "panda_finger_joint1",
            "panda_finger_joint2",
        ]
        self.links = [
            "panda_fingertip_centered",
            "panda_leftfinger",
            "panda_rightfinger",
        ]
        physics = PhysxCfg(
            solver_type=1,
            enable_external_forces_every_iteration=True,
            min_position_iteration_count=cfg["physx_position_iterations"],
            max_position_iteration_count=cfg["physx_position_iterations"],
            min_velocity_iteration_count=cfg["physx_velocity_iterations"],
            max_velocity_iteration_count=cfg["physx_velocity_iterations"],
        )
        sim_cfg = su.SimulationCfg(
            dt=cfg["dt"],
            device=self.device,
            physics=physics,
            visualizer_cfgs=[],
            use_newton_actuators=False,
            log_dir=str(output / "sim_logs"),
        )
        material = su.RigidBodyMaterialCfg(
            static_friction=cfg["mu"],
            dynamic_friction=cfg["mu"],
            restitution=0,
            friction_combine_mode="average",
        )
        collision = su.CollisionPropertiesCfg(
            rest_offset=cfg["rest_offset_m"], contact_offset=cfg["contact_offset_m"]
        )
        limits = dict(zip(self.names, cfg["joint_velocity_limits"]))
        sc = InteractiveSceneCfg(num_envs=n, env_spacing=2.5)
        sc.robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=su.UsdFileCfg(
                usd_path=cfg["asset"],
                articulation_props=su.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False
                ),
                collision_props=collision,
                physics_material=material,
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=tuple(cfg["robot_base"]),
                joint_pos=dict(zip(self.names, inputs["q_target"][0, 0].tolist())),
            ),
            actuators={
                "arm": ImplicitActuatorCfg(
                    joint_names_expr=["panda_joint[1-7]"],
                    stiffness=cfg["arm_kp"],
                    damping=cfg["arm_kd"],
                    joint_effort_limit={
                        "panda_joint[1-4]": 87.0,
                        "panda_joint[5-7]": 12.0,
                    },
                    joint_velocity_limit={
                        k: v for k, v in limits.items() if "finger" not in k
                    },
                    armature=0.0,
                ),
                "leader": ImplicitActuatorCfg(
                    joint_names_expr=["panda_finger_joint1"],
                    stiffness=cfg["finger_kp"],
                    damping=cfg["finger_kd"],
                    joint_effort_limit=cfg["finger_force"],
                    joint_velocity_limit=limits["panda_finger_joint1"],
                ),
                "follower": ImplicitActuatorCfg(
                    joint_names_expr=["panda_finger_joint2"],
                    stiffness=0.0,
                    damping=0.0,
                    joint_effort_limit=cfg["finger_force"],
                    joint_velocity_limit=limits["panda_finger_joint2"],
                ),
            },
        )
        sc.table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            spawn=su.CuboidCfg(
                size=tuple(cfg["table_size"]),
                collision_props=collision,
                physics_material=material,
                visual_material=su.PreviewSurfaceCfg(diffuse_color=(0.35, 0.38, 0.42)),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(cfg["table_center"])),
        )
        sc.target = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Target",
            spawn=su.UsdFileCfg(
                usd_path=str(Path(cfg["shape_dir"]) / f"{shape}.usda"),
                collision_props=collision,
                physics_material=material,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.55, 0.5, 0.775)),
        )
        sc.ground = AssetBaseCfg(prim_path="/World/ground", spawn=su.GroundPlaneCfg())
        self.env = None
        if cfg.get("execution_mode") == "env":
            from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg

            adapter = self
            self.last_states = torch.empty((4, n, 52), device=self.device)

            class GraspEnv(DirectRLEnv):
                """Benchmark task retaining the complete DirectRLEnv.step loop."""

                def _pre_physics_step(env, actions):
                    adapter.command(actions)
                    adapter.substep = 0

                def _apply_action(env):
                    adapter.target.instantaneous_wrench_composer.set_forces_and_torques_index(
                        forces=adapter.pending_forces[adapter.substep, :, None, :],
                        is_global=True,
                    )

                def _get_observations(env):
                    state = adapter.state()
                    env.extras["finite"] = torch.isfinite(state).all(dim=1)
                    env.extras["object_height"] = state[:, 41]
                    return {"policy": torch.cat([state[:, :25], state[:, 39:]], dim=1)}

                def _get_rewards(env):
                    return torch.zeros(env.num_envs, device=env.device)

                def _get_dones(env):
                    return (
                        torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
                        env.episode_length_buf >= 500,
                    )

                def _reset_envs_from_buffer(env):
                    return None

            self.env = GraspEnv(
                DirectRLEnvCfg(
                    sim=sim_cfg,
                    scene=sc,
                    decimation=4,
                    episode_length_s=10.0,
                    action_space=9,
                    observation_space=38,
                    seed=20261022,
                )
            )
            self.env.render_enabled = False
            self.sim, self.scene = self.env.sim, self.env.scene
        else:
            self.sim = su.SimulationContext(sim_cfg)
            self.scene = InteractiveScene(sc)
        self.robot = self.scene["robot"]
        self.target = self.scene["target"]
        self.camera = None
        if record:
            from isaaclab.sensors import Camera, CameraCfg
            from isaaclab_physx.renderers import (
                IsaacRtxRendererCfg,
                IsaacRtxRendererGlobalSettingsCfg,
            )
            from pxr import Gf, UsdLux

            c = cfg["capture"]
            UsdLux.DomeLight.Define(self.sim.stage, "/World/Dome").CreateIntensityAttr(
                450
            )
            light = UsdLux.DistantLight.Define(self.sim.stage, "/World/Key")
            light.CreateIntensityAttr(2000)
            light.AddRotateXYZOp().Set(Gf.Vec3f(30, -25, -20))
            self.camera = Camera(
                CameraCfg(
                    prim_path="/World/P03Camera",
                    update_period=0,
                    width=c["width"],
                    height=c["height"],
                    data_types=["rgb"],
                    spawn=su.PinholeCameraCfg(
                        focal_length=c["intrinsics"][0] * 36 / c["width"],
                        horizontal_aperture=36,
                        clipping_range=(0.01, 1000),
                    ),
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
                            show_all_partitions_by_default=True,
                        )
                    ),
                )
            )
        if self.env is None:
            self.sim.reset()
            self.scene.update(cfg["dt"])
        self.origins = self.scene.env_origins
        if hasattr(self.origins, "torch"):
            self.origins = self.origins.torch
        self.order = [self.names.index(name) for name in self.robot.joint_names]
        self.link_ids = [self.robot.body_names.index(x) for x in self.links]
        self.meta = dict(
            solver="PhysX TGS",
            effective_sim_config=self.sim.cfg.to_dict(),
            joint_names=self.robot.joint_names,
            link_names=self.robot.body_names,
            robot_mass=self.robot.data.body_mass.torch.cpu().tolist(),
            robot_inertia=self.robot.data.body_inertia.torch.cpu().tolist(),
            robot_com=self.robot.data.body_com_pose_b.torch.cpu().tolist(),
            velocity_limits=self.robot.data.joint_vel_limits.torch.cpu().tolist(),
            target_mass=self.target.data.body_mass.torch.cpu().tolist(),
            target_inertia=self.target.data.body_inertia.torch.cpu().tolist(),
            target_material=self.target.root_view.get_material_properties()
            .numpy()
            .tolist(),
            contact_export="Contact forces not sampled by this runner; geometry is retained for independent event analysis.",
        )
        self.sim.stage.Export(str(output / "scene.usda"))
        self.frames = 0
        self.sampling_control = False
        if self.env is not None:
            assert not self.env._physics_handles_decimation
            update = self.scene.update

            def sample_after_update(dt: float) -> None:
                update(dt)
                if self.sampling_control:
                    self.last_states[self.substep] = self.state()
                    self.substep += 1

            self.scene.update = sample_after_update
            self.meta["env_step"] = (
                "Inherited IsaacLab DirectRLEnv.step; native four-step decimation; no auto reset."
            )

    def reset(self, episode: int) -> None:
        if self.env is not None:
            self.env.episode_length_buf.zero_()
        self.scene.reset()
        q = torch.tensor(
            self.inputs["q_target"][episode, 0, self.order],
            dtype=torch.float32,
            device=self.device,
        ).repeat(self.n, 1)
        self.robot.write_joint_position_to_sim_index(position=q)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q))
        self.robot.actuators.target_command.set_position_index(value=q)
        pose = torch.tensor(
            self.inputs["object_initial_pose"][episode],
            dtype=torch.float32,
            device=self.device,
        ).repeat(self.n, 1)
        pose[:, :3] += self.origins
        self.target.write_root_pose_to_sim_index(root_pose=pose)
        self.target.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros((self.n, 6), device=self.device)
        )
        self.scene.update(0)

    def command(self, q: torch.Tensor) -> None:
        self.robot.actuators.target_command.set_position_index(value=q[:, self.order])

    def step(self, force: torch.Tensor) -> None:
        self.target.instantaneous_wrench_composer.set_forces_and_torques_index(
            forces=force[:, None, :], is_global=True
        )
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(self.cfg["dt"])

    def control_step(self, q: torch.Tensor, forces: torch.Tensor) -> None:
        self.pending_forces = forces
        self.sampling_control = True
        self.env.step(q)
        self.sampling_control = False

    def state(self) -> torch.Tensor:
        poses = self.robot.data.body_link_pose_w.torch[:, self.link_ids].clone()
        poses[:, :, :3] -= self.origins[:, None, :]
        obj = self.target.data.root_state_w.torch.clone()
        obj[:, :3] -= self.origins
        return torch.cat(
            [
                self.robot.data.joint_pos.torch,
                self.robot.data.joint_vel.torch,
                poses.flatten(1),
                obj,
            ],
            dim=1,
        )

    def render(self) -> tuple[np.ndarray, dict]:
        c = self.cfg["capture"]
        self.sim.forward()
        origin = self.origins[0].cpu().numpy()
        self.camera.set_world_poses_from_view(
            [(np.array(c["eye"]) + origin).tolist()],
            [(np.array(c["target"]) + origin).tolist()],
        )
        for _ in range(24 if not self.frames else 4):
            self.sim.render()
            self.camera.update(dt=0, force_recompute=True)
        self.frames += 1
        return self.camera.data.output["rgb"].torch[
            0, ..., :3
        ].cpu().numpy().copy(), dict(
            camera_pos_readback=self.camera.data.pos_w.torch[0].cpu().tolist(),
            camera_quat_opengl_xyzw=self.camera.data.quat_w_opengl.torch[0]
            .cpu()
            .tolist(),
            intrinsics=self.camera.data.intrinsic_matrices.torch[0].cpu().tolist(),
        )

    def close(self) -> None:
        self.sim.stop()
