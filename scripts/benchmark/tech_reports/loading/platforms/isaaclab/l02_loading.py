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

"""L-02 execution using Isaac Lab rigid-object collections and native RTX camera."""

from __future__ import annotations
from collections.abc import Callable
import numpy as np
import torch
import warp as wp
import isaaclab.sim as su
from isaaclab.assets import (
    RigidObjectCollection,
    RigidObjectCollectionCfg,
    RigidObjectCfg,
)
from isaaclab.sensors import Camera, CameraCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.renderers import (
    IsaacRtxRendererCfg,
    IsaacRtxRendererGlobalSettingsCfg,
)
from pxr import Gf, UsdLux, UsdGeom, UsdPhysics, UsdShade

__all__ = ["Simulation"]


class Simulation:
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
        ground = UsdGeom.Plane.Define(self.sim.stage, "/World/Ground")
        ground.CreateAxisAttr("Z")
        ground.CreateVisibilityAttr("invisible")
        UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
        self.group = RigidObjectCollection(
            RigidObjectCollectionCfg(
                rigid_objects={
                    f"object_{i:05d}": RigidObjectCfg(
                        prim_path=f"/World/Objects/object_{i:05d}",
                        spawn=su.UsdFileCfg(usd_path=a["path"]),
                        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(p)),
                    )
                    for i, (a, p) in enumerate(zip(cfg["objects"], cfg["positions"]))
                }
            )
        )
        light = UsdLux.DistantLight.Define(self.sim.stage, "/World/Key")
        light.CreateIntensityAttr(2000)
        light.AddRotateXYZOp().Set(Gf.Vec3f(30, -25, -20))
        c = cfg["camera"]
        self.camera = Camera(
            CameraCfg(
                prim_path="/World/L02Camera",
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
                    global_settings=IsaacRtxRendererGlobalSettingsCfg(
                        antialiasing_mode="DLSS" if cfg.get("record") else "Off",
                        dlss_mode=2,
                        enable_dlssg=False,
                        enable_dl_denoiser=bool(cfg.get("record")),
                        enable_global_illumination=False,
                        enable_reflections=False,
                        enable_ambient_occlusion=False,
                        samples_per_pixel=8 if cfg.get("record") else 1,
                        show_all_partitions_by_default=True,
                        carb_settings={"/rtx/post/histogram/enabled": False},
                    )
                ),
            )
        )
        mark("declared")
        self.sim.reset()
        self.group.update(0)
        wp.synchronize()
        torch.cuda.synchronize()
        mark("prepared")
        self.camera.set_world_poses_from_view([c["eye"]], [c["target"]])
        self.meta = {
            "physics": "GPU PhysX TGS 4/1",
            "renderer": (
                "RTX Realtime, DLSS quality, samples=8, DL denoiser on, autoexposure off (media only)"
                if cfg.get("record")
                else "RTX Realtime, AA off, samples=1, DLSS/denoiser/autoexposure off"
            ),
            "gravity": [0, 0, 0],
            "native_reset_note": "Installed PhysxManager.reset performs two internal update_simulation calls before ready state; included in T_ready. No user warmup steps.",
            "cache": "New process, analytic colliders do not use cooked mesh cache; general shader/kernel and OS caches retained; renderer native scene cache not externally controlled",
        }

    def step(self) -> np.ndarray:
        self.sim.step(render=False)
        self.group.update(self.cfg["dt"])
        state = self.group.data.body_state_w.torch[0].cpu().numpy()
        wp.synchronize()
        torch.cuda.synchronize()
        return state

    def render(self) -> np.ndarray:
        self.sim.forward()
        self.sim.render()
        self.camera.update(dt=0, force_recompute=True)
        return self.camera.data.output["rgb"].torch[0, ..., :3].cpu().numpy()

    def audit(self) -> tuple[dict, dict]:
        arrays = {
            "mass": self.group.data.body_mass.torch.cpu().numpy().reshape(-1),
            "inertia": self.group.data.body_inertia.torch.cpu()
            .numpy()
            .reshape(-1, 3, 3),
            "pose": self.group.data.body_state_w.torch[0, :, :7].cpu().numpy(),
        }
        meshes = []
        materials = []
        collisions = []
        for prim in self.sim.stage.Traverse():
            if prim.IsA(UsdGeom.Mesh):
                m = UsdGeom.Mesh(prim)
                mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
                shader = UsdShade.Shader(mat.GetPrim().GetChild("Shader"))
                materials.append(
                    dict(
                        path=str(mat.GetPath()),
                        color_linear=list(shader.GetInput("diffuseColor").Get()),
                    )
                )
                points = np.asarray(m.GetPointsAttr().Get())
                meshes.append(
                    {
                        "vertices": len(points),
                        "triangles": len(m.GetFaceVertexCountsAttr().Get()),
                        "extent": (points.max(0) - points.min(0)).tolist(),
                    }
                )
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                collisions.append(
                    {"path": str(prim.GetPath()), "type": prim.GetTypeName()}
                )
        from isaaclab.utils.math import matrix_from_quat

        pose = torch.eye(4, device="cuda")
        pose[:3, 3] = self.camera.data.pos_w.torch[0]
        pose[:3, :3] = matrix_from_quat(self.camera.data.quat_w_opengl.torch)[0]
        return arrays, dict(
            **self.meta,
            effective_sim_config=self.sim.cfg.to_dict(),
            meshes=meshes,
            materials=materials,
            collisions=collisions,
            camera_world_opengl=pose.cpu().tolist(),
            intrinsics=self.camera.data.intrinsic_matrices.torch[0].cpu().tolist(),
        )

    def close(self) -> None:
        pass
