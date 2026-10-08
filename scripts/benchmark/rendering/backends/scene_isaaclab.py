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

"""Isaac Lab tiled-camera consumer of the frozen shared rendering scene."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..scenes import camera_intrinsics, camera_world_pose, load_scene, scene_fingerprint

__all__ = ["IsaacLabScene", "camera_prim_path"]


def camera_prim_path(env_id: int, camera_id: int) -> str:
    """Return a view with its own parent, as required by Camera's sensor count.

    Args:
        env_id: Zero-based scene replica index.
        camera_id: Zero-based camera index within the replica.

    Returns:
        USD camera path with one parent prim per logical camera exposure.
    """
    return f"/World/Envs/env_{env_id:06d}/Cameras/view_{camera_id:02d}/Camera"


class IsaacLabScene:
    """Author matching USD geometry and one vectorized camera render product.

    Args:
        scene_path: Frozen manifest and canonical geometry directory.
        width: Output width in pixels.
        height: Output height in pixels.
        gpu_id: Physical CUDA and graphics adapter index.
        num_envs: Number of identical scene replicas.
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
        from isaaclab.app import AppLauncher

        self.device = f"cuda:{gpu_id}"
        self.app = AppLauncher(
            headless=True,
            enable_cameras=True,
            device=self.device,
            multi_gpu=False,
            kit_args=f"--/renderer/activeGpu={gpu_id} --/renderer/multiGpu/enabled=false",
        ).app
        import numpy as np
        import torch
        from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import Camera, CameraCfg
        from isaaclab_physx.renderers import IsaacRtxRendererCfg

        self.torch = torch
        self.scene = load_scene(scene_path)
        self.modalities = modalities
        self.num_envs = num_envs
        self.sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(
                dt=1 / 240,
                device=self.device,
                render=sim_utils.RenderCfg(
                    antialiasing_mode="DLAA",
                    enable_dlssg=False,
                    enable_dl_denoiser=True,
                    samples_per_pixel=4,
                ),
            )
        )
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        self.object_paths = {}
        mesh_cache = {}
        eye_positions, orientations = [], []
        model = self.scene["camera_model"]
        k = camera_intrinsics(self.scene, width, height)
        spawn = sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            horizontal_aperture=24.0 * width / k[0][0],
            clipping_range=(model["near_m"], model["far_m"]),
        )
        for index, obj in enumerate(self.scene["objects"]):
            mat_path = f"/World/Materials/material_{index:04d}"
            material = UsdShade.Material.Define(stage, mat_path)
            shader = UsdShade.Shader.Define(stage, mat_path + "/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            properties = obj["material"]
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(*properties["base_color_linear"])
            )
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
                properties["roughness"]
            )
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(
                properties["metallic"]
            )
            material.CreateSurfaceOutput().ConnectToSource(
                shader.ConnectableAPI(), "surface"
            )
        for env_id in range(num_envs):
            offset = np.array([env_id * 10.0, 0.0, 0.0])
            prefix = f"/World/Envs/env_{env_id:06d}"
            UsdGeom.Xform.Define(stage, prefix)
            for index, obj in enumerate(self.scene["objects"]):
                path = prefix + "/Objects/" + obj["uid"]
                geometry = obj["geometry"]
                pose = np.asarray(obj["world_from_object"], dtype=np.float64).copy()
                pose[:3, 3] += offset
                if geometry["kind"] == "box":
                    shape = UsdGeom.Cube.Define(stage, path)
                    shape.CreateSizeAttr(1.0)
                    pose[:3, :3] = pose[:3, :3] @ np.diag(geometry["size"])
                else:
                    key = geometry["path"]
                    if key not in mesh_cache:
                        with np.load(
                            scene_path.parent / key, allow_pickle=False
                        ) as arrays:
                            mesh_cache[key] = {
                                key: arrays[key].copy()
                                for key in ("vertices", "triangles", "normals")
                            }
                    arrays = mesh_cache[key]
                    shape = UsdGeom.Mesh.Define(stage, path)
                    shape.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(arrays["vertices"]))
                    shape.CreateFaceVertexCountsAttr([3] * len(arrays["triangles"]))
                    shape.CreateFaceVertexIndicesAttr(
                        arrays["triangles"].reshape(-1).tolist()
                    )
                    shape.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(arrays["normals"]))
                    shape.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
                    shape.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
                UsdGeom.Xformable(shape).AddTransformOp().Set(
                    Gf.Matrix4d(*pose.T.reshape(-1).tolist())
                )
                UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(
                    UsdShade.Material.Get(
                        stage, f"/World/Materials/material_{index:04d}"
                    )
                )
                sim_utils.add_labels(
                    shape.GetPrim(), [obj["uid"]], instance_name="class"
                )
                self.object_paths[path] = obj["instance_id"]
            for camera_index, spec in enumerate(self.scene["cameras"]):
                camera_path = camera_prim_path(env_id, camera_index)
                view_path = camera_path.rsplit("/", 1)[0]
                UsdGeom.Xform.Define(stage, view_path)
                spawn.func(camera_path, spawn)
                eye, quaternion = camera_world_pose(spec)
                orientations.append(quaternion)
                eye_positions.append((np.asarray(eye) + offset).tolist())
        lighting = self.scene["lighting"]
        light = UsdLux.DistantLight.Define(stage, "/World/Sun")
        light.CreateIntensityAttr(lighting["isaac_directional_intensity"])
        light.CreateColorAttr(Gf.Vec3f(*lighting["color"]))
        light.CreateAngleAttr(0.0)
        rotation = Gf.Rotation(
            Gf.Vec3d(0, 0, -1), Gf.Vec3d(*lighting["direction"])
        ).GetQuat()
        UsdGeom.Xformable(light).AddOrientOp().Set(
            Gf.Quatf(rotation.GetReal(), Gf.Vec3f(rotation.GetImaginary()))
        )
        if lighting["isaac_dome_intensity"]:
            dome = UsdLux.DomeLight.Define(stage, "/World/Dome")
            dome.CreateIntensityAttr(lighting["isaac_dome_intensity"])
        sim_utils.update_stage()
        self.camera = Camera(
            CameraCfg(
                prim_path="/World/Envs/env_.*/Cameras/view_.*/Camera",
                update_period=0.0,
                update_latest_camera_pose=True,
                width=width,
                height=height,
                data_types=[
                    "instance_id_segmentation_fast" if value == "seg" else value
                    for value in modalities
                ],
                renderer_cfg=IsaacRtxRendererCfg(
                    colorize_instance_id_segmentation=False
                ),
                spawn=None,
            )
        )
        self.sim.reset()
        self.camera.set_world_poses(
            positions=torch.tensor(eye_positions, device=self.device),
            orientations=torch.tensor(orientations, device=self.device),
            convention="opengl",
        )
        self.sim.render_context.reset_transform_cadence()
        self.metadata = {
            "backend": "isaaclab",
            "renderer": "Isaac RTX tiled",
            "spp": 4,
            "antialiasing": "DLAA",
            "dl_denoiser": True,
            "dlss_frame_generation": False,
            "device": self.device,
            "scene_sha256": scene_fingerprint(self.scene),
            "num_envs": num_envs,
            "cameras_per_env": len(self.scene["cameras"]),
            "camera_names": [c["name"] for c in self.scene["cameras"]],
            "batch_order": "environment_then_camera",
            "camera_sensor_instances": 1,
            "intrinsics": self.camera.data.intrinsic_matrices.torch.cpu().tolist(),
            "camera_positions": self.camera.data.pos_w.torch.cpu().tolist(),
            "camera_quaternions_xyzw": self.camera.data.quat_w_opengl.torch.cpu().tolist(),
            "depth_semantics": "optical_axis_metres; invalid_or_beyond_far_is_infinity",
            "lighting": lighting,
            "physics_steps": 0,
            "render_boundary": "sim.render and vectorized camera update followed by blocking host delivery",
            "quality_status": "not_qualified",
        }

    def capture(self) -> dict[str, Any]:
        """Return completed host arrays from the single tiled render product."""
        self.sim.render()
        self.camera.update(dt=1 / 60, force_recompute=True)
        # Pose buffers are populated by camera updates, not by set_world_poses.
        self.metadata["camera_positions"] = self.camera.data.pos_w.torch.cpu().tolist()
        self.metadata["camera_quaternions_xyzw"] = (
            self.camera.data.quat_w_opengl.torch.cpu().tolist()
        )
        arrays = {}
        for modality in self.modalities:
            key = "instance_id_segmentation_fast" if modality == "seg" else modality
            value = self.camera.data.output[key].torch
            if modality == "rgb":
                value = value[..., :3]
            elif modality in {"depth", "seg"} and value.ndim == 4:
                value = value[..., 0]
            if modality == "depth":
                far = self.scene["camera_model"]["far_m"]
                value = self.torch.where(
                    (value > 0) & (value < far), value, self.torch.inf
                )
            arrays[modality] = value.contiguous().cpu().numpy()
        if "seg" in arrays:
            import numpy as np

            raw = arrays["seg"]
            mapped = np.zeros_like(raw, dtype=np.int32)
            # Replicator records the USD paths for its per-frame instance IDs.
            info = self.camera.data.info
            self.metadata["segmentation_info"] = str(info)
            mapping = (
                info.get("instance_id_segmentation_fast", {}).get("idToLabels", {})
                if isinstance(info, dict)
                else {}
            )
            for instance, label in mapping.items():
                path = label if isinstance(label, str) else label.get("primPath", "")
                for object_path, canonical in self.object_paths.items():
                    if path == object_path or path.startswith(object_path + "/"):
                        mapped[raw == int(instance)] = canonical
                        break
            arrays["seg"] = mapped
            self.metadata["segmentation_mapping_available"] = bool(mapping)
        return arrays

    def close(self) -> None:
        """Close the isolated Isaac Sim application."""
        self.camera = None
        self.app.close()
