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

"""L-03: replicate one common USD articulation with Isaac Lab / PhysX."""

from __future__ import annotations

from collections.abc import Callable
import numpy as np
import torch
import warp as wp
import isaaclab.sim as su
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab_physx.physics import PhysxCfg

__all__ = ["Simulation"]


class Simulation:
    """Keep native cloning and collision filtering, and restore the complete initial state."""

    def __init__(self, cfg: dict, mark: Callable[[str], None]) -> None:
        self.cfg = cfg
        self.sim = su.SimulationContext(
            su.SimulationCfg(
                dt=cfg["dt"],
                gravity=(0, 0, 0),
                device="cuda:0",
                physics=PhysxCfg(
                    solver_type=1,
                    min_position_iteration_count=4,
                    max_position_iteration_count=4,
                    min_velocity_iteration_count=1,
                    max_velocity_iteration_count=1,
                ),
                visualizer_cfgs=[],
                use_newton_actuators=False,
                log_dir=cfg["output"] + "/sim_logs",
            )
        )
        mark("engine_initialized")
        scene_cfg = InteractiveSceneCfg(
            num_envs=cfg["num_envs"],
            env_spacing=cfg["env_spacing"],
            replicate_physics=True,
            filter_collisions=True,
        )
        scene_cfg.robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=su.UsdFileCfg(
                usd_path=cfg["usd_path"],
                articulation_props=su.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=tuple(cfg["initial_position"]), joint_pos=cfg["initial_qpos"]
            ),
            actuators={
                "passive": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    stiffness=0.0,
                    damping=0.0,
                    friction=0.0,
                    armature=0.0,
                )
            },
        )
        if cfg["record"]:
            scene_cfg.camera = self._camera_cfg()
        self.scene = InteractiveScene(scene_cfg)
        self.robot = self.scene["robot"]
        self.camera = self.scene["camera"] if cfg["record"] else None
        mark("declared")
        self.sim.reset()
        self.scene.update(0)
        wp.synchronize()
        torch.cuda.synchronize()
        mark("prepared")
        self.names = self.robot.joint_names
        self.links = self.robot.body_names
        q = torch.tensor(
            [[cfg["initial_qpos"][n] for n in self.names]], device=self.sim.device
        ).repeat(cfg["num_envs"], 1)
        self.robot.write_joint_position_to_sim_index(position=q)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q))
        pose = torch.tensor(
            [cfg["initial_position"] + [0, 0, 0, 1]], device=self.sim.device
        ).repeat(cfg["num_envs"], 1)
        pose[:, :3] += self.scene.env_origins
        self.robot.write_root_link_pose_to_sim_index(root_pose=pose)
        self.robot.write_root_link_velocity_to_sim_index(
            root_velocity=torch.zeros((cfg["num_envs"], 6), device=self.sim.device)
        )
        self.initial = self.sample()
        if cfg["record"]:
            c = cfg["camera"]
            origin = self.scene.env_origins.cpu().numpy().mean(0)
            self.camera.set_world_poses_from_view(
                np.repeat([np.asarray(c["eye"]) + origin], cfg["num_envs"], axis=0),
                np.repeat([np.asarray(c["target"]) + origin], cfg["num_envs"], axis=0),
            )

    def sample(self) -> dict[str, np.ndarray]:
        """Read every local-frame state; this Isaac 6.1 API already uses xyzw."""
        poses = self.robot.data.body_link_pose_w.torch.clone()
        poses[..., :3] -= self.scene.env_origins[:, None, :]
        root = self.robot.data.root_pose_w.torch.clone()
        root[:, :3] -= self.scene.env_origins
        state = {
            "qpos": self.robot.data.joint_pos.torch.cpu().numpy().copy(),
            "qvel": self.robot.data.joint_vel.torch.cpu().numpy().copy(),
            "link_pose": poses.cpu().numpy(),
            "root_pose": root.cpu().numpy(),
            "root_velocity": self.robot.data.root_link_vel_w.torch.cpu().numpy().copy(),
        }
        wp.synchronize()
        torch.cuda.synchronize()
        return state

    def step(self) -> dict[str, np.ndarray]:
        """Complete one physics step and all-environment readback."""
        self.sim.step(render=False)
        self.scene.update(self.cfg["dt"])
        return self.sample()

    def audit(self) -> tuple[dict, dict]:
        """Save physical properties and native articulation shape count."""
        arrays = {
            "mass": self.robot.data.body_mass.torch.cpu().numpy(),
            "inertia_matrix": self.robot.data.body_inertia.torch.cpu().numpy(),
            "com_pose": self.robot.data.body_com_pose_b.torch.cpu().numpy(),
            "joint_limits": self.robot.data.joint_pos_limits.torch.cpu().numpy(),
            "env_origins": self.scene.env_origins.cpu().numpy(),
        }
        return arrays, {
            "num_envs": self.scene.num_envs,
            "instance_count": self.robot.num_instances,
            "arena_names": self.scene.env_prim_paths,
            "joint_names": self.names,
            "link_names": self.links,
            "native_shapes_per_articulation": self.robot.root_view.max_shapes,
            "effective_fixed_base": self.robot.is_fixed_base,
            "config": self.sim.cfg.to_dict(),
            "physics": "GPU PhysX TGS 4/1",
            "native_reset_note": "Preparation includes internal physics warmup; root and joint state are restored once before the measured first step.",
        }

    def close(self) -> None:
        """Application shutdown is owned by the fresh-process entry point."""

    def _camera_cfg(self):
        from isaaclab.sensors import CameraCfg
        from isaaclab_physx.renderers import (
            IsaacRtxRendererCfg,
            IsaacRtxRendererGlobalSettingsCfg,
        )
        from pxr import Gf, UsdLux

        light = UsdLux.DistantLight.Define(self.sim.stage, "/World/Key")
        light.CreateIntensityAttr(2000)
        light.AddRotateXYZOp().Set(Gf.Vec3f(30, -25, -20))
        c = self.cfg["camera"]
        return CameraCfg(
            prim_path="{ENV_REGEX_NS}/L03Camera",
            update_period=0,
            width=c["width"],
            height=c["height"],
            data_types=["rgb"],
            spawn=su.PinholeCameraCfg(
                focal_length=c["intrinsics"][0] * 36 / c["width"],
                horizontal_aperture=36,
                clipping_range=(0.01, 100),
            ),
            renderer_cfg=IsaacRtxRendererCfg(
                enable_scene_partitioning=False,
                global_settings=IsaacRtxRendererGlobalSettingsCfg(
                    antialiasing_mode="DLSS",
                    dlss_mode=2,
                    enable_dlssg=False,
                    enable_dl_denoiser=True,
                    enable_global_illumination=False,
                    enable_reflections=False,
                    enable_ambient_occlusion=False,
                    samples_per_pixel=8,
                    show_all_partitions_by_default=True,
                    carb_settings={"/rtx/post/histogram/enabled": False},
                ),
            ),
        )

    def render(self) -> np.ndarray:
        self.sim.forward()
        self.sim.render()
        self.camera.update(dt=0, force_recompute=True)
        return self.camera.data.output["rgb"].torch[0, ..., :3].cpu().numpy()

    def camera_audit(self) -> dict:
        from isaaclab.utils.math import matrix_from_quat

        pose = torch.eye(4, device="cuda")
        pose[:3, 3] = self.camera.data.pos_w.torch[0]
        pose[:3, :3] = matrix_from_quat(self.camera.data.quat_w_opengl.torch)[0]
        return {
            "camera_world_opengl": pose.cpu().tolist(),
            "intrinsics": self.camera.data.intrinsic_matrices.torch[0].cpu().tolist(),
            "camera_instances": self.cfg["num_envs"],
        }
