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

"""Bounded ManiSkill/SAPIEN raster reference over the approved canonical scene.

Imports are simulator-free. Construction uses the pinned ManiSkill batched
camera path at both N=1 and N=16. PhysX initializes its pose buffers outside
measurement; capture never advances physics. These profiles are unqualified
raster references, not quality-equivalent ray-tracing comparisons.
"""

from __future__ import annotations

from copy import deepcopy
import ctypes
import hashlib
from importlib import metadata
import json
from pathlib import Path
from typing import Any, Mapping

from scripts.benchmark.core.artifacts import write_json
from scripts.benchmark.rendering.scenes import camera_intrinsics, load_scene

__all__ = ["ManiSkillSceneReference", "raster_profile", "runtime_evidence"]

_VERSIONS = {"mani-skill": "3.0.1", "sapien": "3.0.3"}
_MANISKILL_COMMIT = "62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3"
_PHYSX_VERSION = "105.1-physx-5.3.1.patch0"
_PHYSX_SHA256 = "4c582a16509a71faf5592fe9708586dfcc7ab61ae932eabc1ddd81f290818706"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def raster_profile(shader: str = "default") -> dict[str, Any]:
    """Return the frozen, deliberately unqualified raster reference settings.

    Args:
        shader: Pinned SAPIEN ``default`` or ``minimal`` raster shader.

    Returns:
        Portable settings; no host paths or quality qualification are included.
    """
    if shader not in {"default", "minimal"}:
        raise ValueError("Only default/minimal raster reference shaders are supported")
    return {
        "id": f"maniskill_raster_{shader}-v3",
        "shader": shader,
        "packages": dict(_VERSIONS),
        "maniskill_commit": _MANISKILL_COMMIT,
        "physx_version": _PHYSX_VERSION,
        "physx_library_sha256": _PHYSX_SHA256,
        "msaa": 1,
        "exposure": 1.0,
        "ambient_linear": [0.15, 0.15, 0.15],
        "directional_color_scale": 3.0,
        "material_specular": 0.5,
        "shadow": True,
        "shadow_map_size": 2048,
        "texture_delivery": ["Color"],
        "setup_method": "render_only_camera_pose_buffer_v1",
        "rgb_encoding": "native_gamma_2.2_approximation_not_IEC_sRGB",
        "rgb_quantization": "native_uint8_or_clamp_round_float_times_255",
        "quality_status": "not_qualified",
        "equal_quality_comparison": False,
    }


def runtime_evidence(library: Path) -> dict[str, Any]:
    """Check pinned package metadata and a downloaded PhysX library without loading it.

    Args:
        library: Explicit local library; this function never downloads anything.

    Returns:
        Version, source and checksum evidence suitable for freezing in a run.
    """
    versions = {name: metadata.version(name) for name in _VERSIONS}
    if versions != _VERSIONS:
        raise RuntimeError(f"Pinned ManiSkill/SAPIEN versions required: {versions}")
    library = Path(library).absolute()
    if not library.is_file():
        raise FileNotFoundError(f"Pre-download the pinned PhysX GPU library: {library}")
    digest = _sha256(library)
    if digest != _PHYSX_SHA256:
        raise ValueError("PhysX GPU library does not match the frozen release SHA256")
    distribution = metadata.distribution("mani-skill")
    source = json.loads(distribution.read_text("direct_url.json") or "{}")
    if (
        source.get("vcs_info", {}).get("commit_id") != _MANISKILL_COMMIT
        or source.get("url", "").removesuffix(".git")
        != "https://github.com/haosulab/ManiSkill"
    ):
        raise RuntimeError(
            "Install the frozen ManiSkill source commit, not an unverified version-only wheel"
        )
    source_files = {
        name: _sha256(Path(distribution.locate_file(name)))
        for name in (
            "mani_skill/envs/scene.py",
            "mani_skill/sensors/camera.py",
            "mani_skill/render/shaders.py",
        )
    }
    return {
        "packages": versions,
        "maniskill_source": source,
        "source_file_sha256": source_files,
        "library_path": str(library),
        "library_sha256": digest,
        "physx_version": _PHYSX_VERSION,
        "url": f"https://github.com/sapien-sim/physx-precompiled/releases/download/{_PHYSX_VERSION}/linux-so.zip",
        "license_provenance": {
            "spdx": "BSD-3-Clause",
            "url": f"https://raw.githubusercontent.com/sapien-sim/physx-precompiled/{_PHYSX_VERSION}/license.md",
            "sha256": "7ad36e924a4e83b4706a7f5bc89afe1a2be0eb8ddcb2abfd1d28147f86c41de1",
        },
        "automatic_downloads": False,
    }


def _camera_ros_matrix(view: Mapping[str, Any]) -> Any:
    """Map SAPIEN's forward/left/up camera axes to the canonical look-at view."""
    import numpy as np

    eye = np.asarray(view["eye"], dtype=np.float64)
    forward = np.asarray(view["target"], dtype=np.float64) - eye
    forward /= np.linalg.norm(forward)
    left = np.cross(np.asarray(view["up"], dtype=np.float64), forward)
    left /= np.linalg.norm(left)
    up = np.cross(forward, left)
    matrix = np.eye(4)
    matrix[:3, :3] = np.column_stack((forward, left, up))
    matrix[:3, 3] = eye
    if not np.isfinite(matrix).all():
        raise ValueError("Camera look-at axes are degenerate")
    return matrix


def _rgb_device(array: Any, torch: Any, shader: str) -> Any:
    """Pack native gamma-encoded RGBA as RGB8 on its existing device."""
    rgb = array[..., :3]
    if shader == "minimal":
        if array.dtype != torch.uint8:
            raise ValueError("The minimal Color target must be native uint8")
    else:
        if array.dtype != torch.float32:
            raise ValueError("The default Color target must be native float32")
        rgb = (rgb.clamp(0, 1) * 255).round().to(torch.uint8)
    return rgb.contiguous()


def _setup_camera_pose_buffer(scene: Any, torch: Any, device: Any) -> dict[str, Any]:
    """Initialize renderer input without applying collisionless bodies to physics.

    ManiSkill's general _setup applies every dynamic actor to PhysX and then
    synchronizes in articulation update, even with zero articulations. This
    static reference only needs the indexed CUDA pose buffer read by SAPIEN's
    render group. Applying/fetching simulation state serves no rendering role.
    """
    if scene.articulations:
        raise ValueError("The static raster reference cannot initialize articulations")
    actors = list(scene.actors.values())
    if not actors or any(actor.px_body_type != "kinematic" for actor in actors):
        raise ValueError("Only actual kinematic camera mounts are supported")
    try:
        scene.px.gpu_init()
        torch.cuda.synchronize(device)
    except RuntimeError as exc:
        raise RuntimeError("PhysX pose-buffer allocation/warm-start failed") from exc
    data = scene.px.cuda_rigid_body_data.torch()
    if data.device != device:
        raise RuntimeError("PhysX pose buffer is on a different CUDA device")
    scene.non_static_actors = actors
    for actor in actors:
        actor._body_data_index
        actor.set_pose(actor.initial_pose)
    data[:, 7:] = 0
    try:
        torch.cuda.synchronize(device)
    except RuntimeError as exc:
        raise RuntimeError("Render-only camera pose publication failed") from exc
    scene._gpu_sim_initialized = True
    scene._needs_fetch = False
    return {
        "method": "render_only_camera_pose_buffer_v1",
        "gpu_init_calls": 1,
        "setup_physics_warm_start_steps": 1,
        "rigid_pose_rows": int(data.shape[0]),
        "articulations": 0,
        "simulation_apply_calls": 0,
        "simulation_fetch_calls": 0,
        "articulation_update_calls": 0,
        "capture_physics_steps": 0,
    }


class ManiSkillSceneReference:
    """Static canonical geometry and one batched RGB camera per environment."""

    def __init__(
        self,
        scene_path: Path,
        *,
        num_envs: int,
        gpu_id: int,
        shader: str,
        physx_gpu_library: Path,
        construction_evidence: Path | None = None,
    ) -> None:
        """Construct pinned render resources, outside any performance window.

        Args:
            scene_path: Approved Franka canonical manifest and mesh directory.
            num_envs: Exactly one or sixteen independent scene replicas.
            gpu_id: CUDA/NVML index in PCI_BUS_ID order, with visibility unmasked.
            shader: ``default`` or ``minimal`` raster profile.
            physx_gpu_library: Previously downloaded, checksum-verified runtime.
            construction_evidence: Optional checkpoint path for device/setup
                evidence, retained even if native construction fails.
        """
        if type(num_envs) is not int or num_envs not in (1, 16):
            raise ValueError("Reference points are limited to N=1 and N=16")
        if type(gpu_id) is not int or gpu_id < 0:
            raise ValueError("GPU index must be a nonnegative integer")
        profile = raster_profile(shader)
        runtime = runtime_evidence(physx_gpu_library)
        source = load_scene(scene_path)
        if source["scene_id"] != "franka_workstation":
            raise ValueError("The reference uses the approved Franka workstation only")

        import numpy as np
        import torch
        import sapien
        import sapien.physx as physx
        from mani_skill.envs.scene import ManiSkillScene
        from mani_skill.envs.utils.system.backend import parse_sim_and_render_backend
        from mani_skill.render import ShaderConfig
        from mani_skill.sensors.camera import Camera, CameraConfig

        self.torch, self.sapien = torch, sapien
        self.gpu_id, self.num_envs, self.shader = gpu_id, num_envs, shader
        self.source = source
        self.scene_path = Path(scene_path)
        self.device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(self.device)
        self.stream = torch.cuda.current_stream(self.device)
        # Match SAPIEN's loader, but use the explicit verified path and never
        # invoke its network-enabled enable_gpu() wrapper.
        self._cuda_library = ctypes.CDLL("libcuda.so.1", ctypes.RTLD_GLOBAL)
        self._physx_library = ctypes.CDLL(runtime["library_path"], ctypes.RTLD_LOCAL)
        if physx.version() != _PHYSX_VERSION:
            raise RuntimeError(
                "SAPIEN's compiled PhysX version differs from the frozen library"
            )
        physx._enable_gpu()
        backend = parse_sim_and_render_backend(f"cuda:{gpu_id}", f"cuda:{gpu_id}")
        for device in (backend.sim_device, backend.render_device):
            if device is None or not device.is_cuda() or device.cuda_id != gpu_id:
                raise RuntimeError(
                    "SAPIEN silently selected a different or non-CUDA device"
                )
        px = physx.PhysxGpuSystem(device=backend.sim_device)
        px.gpu_set_cuda_stream(self.stream.cuda_stream)
        sub_scenes = []
        for index in range(num_envs):
            sub = sapien.Scene(
                systems=[px, sapien.render.RenderSystem(backend.render_device)]
            )
            px.set_scene_offset(sub, [4.0 * index, 0.0, 0.0])
            sub_scenes.append(sub)
        self.scene = ManiSkillScene(sub_scenes, device=self.device, backend=backend)
        self._entities = []
        for obj in source["objects"]:
            material = sapien.render.RenderMaterial(
                base_color=[*obj["material"]["base_color_linear"], 1.0],
                roughness=obj["material"]["roughness"],
                metallic=obj["material"]["metallic"],
                specular=profile["material_specular"],
            )
            geometry = obj["geometry"]
            if geometry["kind"] == "box":
                shape = sapien.render.RenderShapeBox(
                    np.asarray(geometry["size"]) / 2, material
                )
            else:
                with np.load(
                    self.scene_path.parent / geometry["path"], allow_pickle=False
                ) as mesh:
                    vertices = np.asarray(mesh["vertices"], dtype=np.float32)
                    shape = sapien.render.RenderShapeTriangleMesh(
                        vertices=vertices,
                        triangles=np.asarray(mesh["triangles"], dtype=np.uint32),
                        normals=np.asarray(mesh["normals"], dtype=np.float32),
                        uvs=np.zeros((len(vertices), 2), dtype=np.float32),
                        material=material,
                    )
            for index, sub in enumerate(sub_scenes):
                entity = sapien.Entity()
                entity.name = f"env{index}_{obj['uid']}"
                entity.pose = sapien.Pose(
                    np.asarray(obj["world_from_object"], dtype=np.float32)
                )
                body = sapien.render.RenderBodyComponent()
                body.attach(shape.clone())
                entity.add_component(body)
                sub.add_entity(entity)
                self._entities.append(entity)

        self.scene.set_ambient_light(profile["ambient_linear"])
        self.scene.add_directional_light(
            source["lighting"]["direction"],
            [
                value * profile["directional_color_scale"]
                for value in source["lighting"]["color"]
            ],
            shadow=profile["shadow"],
            shadow_map_size=profile["shadow_map_size"],
        )
        self.views = [
            deepcopy(
                next(
                    view
                    for view in source["cameras"]
                    if view["name"] == "external_front"
                )
            )
        ]
        builder = self.scene.create_actor_builder()
        builder.initial_pose = sapien.Pose(
            _camera_ros_matrix(self.views[0]).astype(np.float32)
        )
        self.camera_mount = builder.build_kinematic("reference_camera_mount")
        properties = torch.cuda.get_device_properties(self.device)
        construction = {
            "stage": "before_pose_buffer_initialization",
            "requested_cuda_id": gpu_id,
            "torch_device": {
                "cuda_id": torch.cuda.current_device(),
                "name": properties.name,
                "uuid": (
                    str(properties.uuid)
                    if getattr(properties, "uuid", None) is not None
                    else None
                ),
            },
            "physx_device": {"cuda_id": px.device.cuda_id, "pci": px.device.pci_string},
            "render_devices": [
                {
                    "cuda_id": sub.render_system.device.cuda_id,
                    "pci": sub.render_system.device.pci_string,
                }
                for sub in sub_scenes
            ],
            "cuda_stream": self.stream.cuda_stream,
            "articulations": len(self.scene.articulations),
            "kinematic_camera_mounts": len(self.camera_mount._objs),
            "hidden_geometry_added": False,
        }
        aligned = (
            construction["torch_device"]["cuda_id"] == gpu_id
            and construction["physx_device"]["cuda_id"] == gpu_id
            and all(
                item["cuda_id"] == gpu_id for item in construction["render_devices"]
            )
        )
        construction["device_alignment"] = "passed" if aligned else "failed"
        if construction_evidence is not None:
            write_json(construction_evidence, construction)
        if not aligned:
            raise RuntimeError(
                "Constructed Torch/PhysX/render systems are on different devices"
            )
        # gpu_init performs an internal setup warm-start. No canonical visual
        # has a physics body, and the kinematic camera pose is restored below.
        setup = _setup_camera_pose_buffer(self.scene, torch, self.device)
        construction.update(stage="pose_buffer_initialized", setup=setup)
        if construction_evidence is not None:
            write_json(construction_evidence, construction)
        shader_config = ShaderConfig(
            shader_pack=shader, texture_names={"Color": ["rgb"]}
        )
        sapien.render.set_msaa(profile["msaa"])
        self.camera = Camera(
            CameraConfig(
                uid="external_front",
                pose=sapien.Pose(),
                mount=self.camera_mount,
                width=128,
                height=128,
                intrinsic=np.asarray(camera_intrinsics(source, 128, 128)),
                near=source["camera_model"]["near_m"],
                far=source["camera_model"]["far_m"],
                shader_config=shader_config,
            ),
            self.scene,
        )
        self.scene.sensors = {"external_front": self.camera}
        for camera in self.camera.camera._render_cameras:
            if shader == "default":
                camera.set_property("exposure", profile["exposure"])
        self.scene._setup_gpu_rendering()
        # Set this BEFORE camera-group construction: SAPIEN copies the group
        # stream into each BatchedCamera's Vulkan->CUDA semaphore wait.
        self.scene.render_system_group.set_cuda_stream(self.stream.cuda_stream)
        self.scene.update_render(update_human_render_cameras=False)
        self.set_cameras(self.views)
        self.synchronize()
        shader_root = Path(sapien.__file__).parent / "vulkan_shader" / shader
        shader_files = ("gbuffer.frag", "composite0.frag")
        self.metadata = {
            "backend": "maniskill",
            "renderer": "SAPIEN Vulkan raster",
            "profile": profile,
            "runtime": runtime,
            "physics_steps_during_capture": 0,
            "physics_setup": "PhysX GPU pose buffers; one gpu_init internal warm-start call; canonical visuals have no physics bodies",
            "pose_buffer_setup": setup,
            "stock_asset_downloads": False,
            "canonical_geometry": "exact approved boxes and NPZ triangle meshes; static render components",
            "num_envs": num_envs,
            "cameras_per_env": 1,
            "batch_order": "environment_then_camera",
            "requested_output": [128, 128],
            "transferred_render_targets": ["Color"],
            "render_attachments": "native raster shader intermediates may include unrequested geometry/depth attachments; only Color is delivered",
            "shader_sources": {
                name: {
                    "path": str(shader_root / name),
                    "sha256": _sha256(shader_root / name),
                }
                for name in shader_files
            },
            "render_device": {
                "cuda_id": backend.render_device.cuda_id,
                "name": backend.render_device.name,
                "pci": backend.render_device.pci_string,
            },
            "torch_device": {
                "cuda_id": gpu_id,
                "name": properties.name,
                "uuid": (
                    str(properties.uuid)
                    if getattr(properties, "uuid", None) is not None
                    else None
                ),
            },
            "stream": {
                "cuda_stream": self.stream.cuda_stream,
                "policy": "same selected Torch stream for PhysX publication, render-group update, Vulkan semaphore wait and RGB packing",
            },
            "completion": "BatchedCamera Vulkan timeline semaphore wait queued on shared CUDA stream; selected device synchronization",
            "native_render_s": None,
            "internal_resolution": {
                "status": "not_verified",
                "reason": "output dimensions do not prove native workload equivalence",
            },
            "quality_status": "not_qualified",
            "quality_limitations": [
                "raster profile differs from formal ray-tracing profiles",
                "photometric calibration absent",
                "native gamma-2.2 encoding is not exact IEC sRGB",
                "depth and segmentation semantics are not assessed by RGB-only reference points",
            ],
            "licenses": {
                "mani-skill": "Apache-2.0",
                "sapien": "Apache-2.0",
                "assets": deepcopy(source.get("robot_source")),
                "physx": runtime["license_provenance"],
            },
        }
        self.refresh_metadata()

    def set_cameras(self, views: list[dict[str, Any]]) -> None:
        """Publish the one canonical view to every environment's GPU pose buffer.

        Args:
            views: One canonical eye/target/up view, shared by all environments.
        """
        if len(views) != 1:
            raise ValueError(
                "The reference requires exactly one camera per environment"
            )
        with self.torch.cuda.stream(self.stream):
            self.camera_mount.set_pose(
                self.sapien.Pose(_camera_ros_matrix(views[0]).astype("float32"))
            )
        self.views = deepcopy(views)

    def apply_state(self, state: Mapping[str, Any]) -> None:
        """Accept only the frozen static object poses and republish its camera.

        Args:
            state: Canonical frozen object matrices and named camera views.
        """
        expected = {
            obj["uid"]: obj["world_from_object"] for obj in self.source["objects"]
        }
        if state.get("objects") != expected:
            raise ValueError(
                "The raster reference does not support moving object state"
            )
        self.set_cameras([state["cameras"]["external_front"]])

    def render_device(self) -> Mapping[str, Any]:
        """Submit one fresh raster exposure and pack RGB8 on the selected GPU.

        Returns:
            RGB uint8 tensor shaped N x 128 x 128 x 3 on the selected device.
        """
        with self.torch.cuda.stream(self.stream):
            self.scene.update_render(update_human_render_cameras=False)
            self.camera.capture()
            arrays = self.camera.get_obs(
                rgb=True,
                depth=False,
                position=False,
                segmentation=False,
                normal=False,
                albedo=False,
                apply_texture_transforms=False,
            )
            rgba = arrays["Color"]
            if rgba.device != self.device:
                raise RuntimeError("Camera delivery is on the wrong CUDA device")
            return {"rgb": _rgb_device(rgba, self.torch, self.shader)}

    def synchronize(self) -> None:
        """Complete Vulkan interoperability and all work on the selected GPU."""
        self.torch.cuda.synchronize(self.device)

    def to_host(self, arrays: Mapping[str, Any]) -> Mapping[str, Any]:
        """Perform exactly one blocking D2H for the packed RGB batch.

        Args:
            arrays: Original device arrays returned by :meth:`render_device`.

        Returns:
            Completed NumPy storage, without an additional CPU copy.
        """
        with self.torch.cuda.stream(self.stream):
            return {
                name: value.detach().to(device="cpu", non_blocking=False).numpy()
                for name, value in arrays.items()
            }

    def refresh_metadata(self) -> None:
        """Record camera pose and intrinsics outside the measurement window."""
        import numpy as np

        actual = self.camera_mount.pose.raw_pose.detach().cpu().numpy()
        cameras = self.camera.camera._render_cameras
        expected_pose = _camera_ros_matrix(self.views[0])
        expected_intrinsics = np.asarray(camera_intrinsics(self.source, 128, 128))
        for index, camera in enumerate(cameras):
            actual_pose = self.sapien.Pose(
                actual[index, :3], actual[index, 3:]
            ).to_transformation_matrix()
            if not np.allclose(actual_pose, expected_pose, atol=1e-5):
                raise ValueError(f"Camera pose mismatch in environment {index}")
            if not np.allclose(
                camera.get_intrinsic_matrix(), expected_intrinsics, atol=1e-4
            ):
                raise ValueError(f"Camera intrinsics mismatch in environment {index}")
        self.metadata["cameras"] = [
            {
                "env_id": index,
                "camera_name": "external_front",
                "canonical_view": deepcopy(self.views[0]),
                "world_from_camera_ros": _camera_ros_matrix(self.views[0]).tolist(),
                "actual_mount_position_quaternion_wxyz": actual[index].tolist(),
                "actual_intrinsics": np.asarray(camera.get_intrinsic_matrix()).tolist(),
                "expected_intrinsics": camera_intrinsics(self.source, 128, 128),
                "pose_and_intrinsics_verified": True,
            }
            for index, camera in enumerate(cameras)
        ]

    def close(self) -> None:
        """Release the per-process scene after completing outstanding work."""
        self.synchronize()
        self.camera = self.camera_mount = None
        self._entities.clear()
        self.scene = None
