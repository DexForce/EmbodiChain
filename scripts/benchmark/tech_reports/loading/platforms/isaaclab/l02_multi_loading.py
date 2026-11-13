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

"""L-02: InteractiveScene clones one USD rigid object into num_envs arenas."""

from __future__ import annotations

from collections.abc import Callable
import numpy as np
import torch
import warp as wp
import isaaclab.sim as su
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab_physx.physics import PhysxCfg

__all__ = ["Simulation"]


class Simulation:
    """Retain native physics replication and collision isolation."""

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
        sc = InteractiveSceneCfg(
            num_envs=cfg["num_envs"],
            env_spacing=cfg["env_spacing"],
            replicate_physics=True,
            filter_collisions=True,
        )
        sc.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            spawn=su.UsdFileCfg(usd_path=cfg["objects"][0]["path"]),
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(cfg["positions"][0])),
        )
        if cfg["record"]:
            sc.camera = self._camera_cfg()
        self.scene = InteractiveScene(sc)
        self.obj = self.scene["object"]
        self.camera = self.scene["camera"] if cfg["record"] else None
        mark("declared")
        self.sim.reset()
        self.scene.update(0)
        wp.synchronize()
        torch.cuda.synchronize()
        mark("prepared")
        if cfg["record"]:
            c = cfg["camera"]
            origin = self.scene.env_origins.cpu().numpy().mean(0)
            self.camera.set_world_poses_from_view(
                np.repeat([np.asarray(c["eye"]) + origin], cfg["num_envs"], axis=0),
                np.repeat([np.asarray(c["target"]) + origin], cfg["num_envs"], axis=0),
            )

    def step(self) -> np.ndarray:
        self.sim.step(render=False)
        self.scene.update(self.cfg["dt"])
        state = self.obj.data.root_state_w.torch.clone()
        state[:, :3] -= self.scene.env_origins
        # Canonical report state uses xyzw, not Isaac's wxyz.
        state[:, 3:7] = state[:, [4, 5, 6, 3]]
        wp.synchronize()
        torch.cuda.synchronize()
        return state[:, None, :].cpu().numpy()

    def audit(self) -> tuple[dict, dict]:
        pose = self.obj.data.root_state_w.torch[:, :7].clone()
        pose[:, :3] -= self.scene.env_origins
        pose[:, 3:7] = pose[:, [4, 5, 6, 3]]
        arrays = {
            "mass": self.obj.data.body_mass.torch.cpu().numpy(),
            "inertia": self.obj.data.body_inertia.torch.cpu().numpy(),
            "env_origins": self.scene.env_origins.cpu().numpy(),
            "pose": pose[:, None].cpu().numpy(),
        }
        from pxr import UsdGeom, UsdPhysics

        colliders = []
        for prim in self.sim.stage.Traverse():
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                colliders.append(
                    {
                        "path": str(prim.GetPath()),
                        "type": prim.GetTypeName(),
                        "size": (
                            UsdGeom.Cube(prim).GetSizeAttr().Get()
                            if prim.IsA(UsdGeom.Cube)
                            else UsdGeom.Sphere(prim).GetRadiusAttr().Get()
                        ),
                    }
                )
        return arrays, {
            "num_envs": self.scene.num_envs,
            "objects_per_env": 1,
            "instance_count": self.obj.num_instances,
            "colliders": colliders,
            "arena_names": self.scene.env_prim_paths,
            "replication": "InteractiveSceneCfg(num_envs, replicate_physics=True, filter_collisions=True)",
            "effective_config": self.sim.cfg.to_dict(),
            "physics": "GPU PhysX TGS 4/1",
            "renderer": "RTX (media)" if self.cfg["record"] else "disabled",
            "native_reset_note": "PhysxManager.reset includes internal simulation updates; retained in T_ready",
        }

    def isolation_probe(self) -> dict:
        before = self.step()
        pose = self.obj.data.root_pose_w.torch[1:2].clone()
        pose[0, 0] += 0.01
        self.obj.write_root_pose_to_sim(pose, env_ids=torch.tensor([0], device="cuda"))
        after_write = self.step()
        states = np.stack([self.step() for _ in range(20)])
        return {"before": before, "after_write": after_write, "states": states}

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
            prim_path="{ENV_REGEX_NS}/L02Camera",
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
