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

"""EmbodiChain consumer of the frozen shared rendering scene."""

from __future__ import annotations

import gc
import copy
import math
from pathlib import Path
from typing import Any

from ..scenes import camera_intrinsics, load_scene, scene_fingerprint

__all__ = ["EmbodiChainScene", "material_display_color"]


def _linear_u8_to_srgb_u8(value: Any, torch: Any) -> Any:
    """Encode native Hybrid/DLSS linear UNORM bytes on the existing device.

    This is a delivery conversion, not a photometric fit. It cannot recover
    precision already lost by the native linear-UNORM8 output packing.
    """
    linear = value.to(torch.float32) / 255.0
    encoded = torch.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * linear.pow(1.0 / 2.4) - 0.055,
    )
    return (encoded.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)


class _ArenaSlice:
    """Camera-owner view of a contiguous subset of the existing scene arenas."""

    def __init__(self, owner: Any, start: int, stop: int) -> None:
        self.owner, self.start, self.num_envs = owner, start, stop - start

    def get_world(self) -> Any:
        return self.owner.get_world()

    def get_env(self, index: int) -> Any:
        if not 0 <= index < self.num_envs:
            raise IndexError(index)
        return self.owner.get_env(self.start + index)


class _CameraBatch:
    """One logical view with bounded native multiview groups over all arenas.

    The native Vulkan render pass uses a 32-bit view mask. Chunking keeps the
    complete scene resident and submits every group once per measured batch;
    it does not reuse one environment's image or skip renderer work.
    """

    def __init__(
        self, camera_type: Any, config: Any, owner: Any, torch: Any, capacity: int
    ) -> None:
        self.torch = torch
        self.parts: list[tuple[int, int, Any]] = []
        self._data: dict[str, Any] = {}
        try:
            for start in range(0, owner.num_envs, capacity):
                stop = min(start + capacity, owner.num_envs)
                camera = camera_type(
                    config, device=owner.device, owner=_ArenaSlice(owner, start, stop)
                )
                self.parts.append((start, stop, camera))
        except BaseException:
            self.destroy()
            raise

    @property
    def group_ids(self) -> list[int]:
        return [camera.group_id for _, _, camera in self.parts]

    def _join(self, values: list[Any]) -> Any:
        return values[0] if len(values) == 1 else self.torch.cat(values, dim=0)

    def update(self, **kwargs: Any) -> None:
        for _, _, camera in self.parts:
            camera.update(**kwargs)
        data = [camera.get_data() for _, _, camera in self.parts]
        # TensorDict iteration yields batch members, whereas keys() names modalities.
        self._data = {
            key: self._join([part[key] for part in data]) for key in data[0].keys()
        }

    def get_data(self) -> dict[str, Any]:
        return self._data

    def get_intrinsics(self) -> Any:
        return self._join([camera.get_intrinsics() for _, _, camera in self.parts])

    def get_arena_pose(self, to_matrix: bool = False) -> Any:
        return self._join(
            [camera.get_arena_pose(to_matrix=to_matrix) for _, _, camera in self.parts]
        )

    def look_at(self, eye: Any, target: Any, up: Any) -> None:
        for start, stop, camera in self.parts:
            camera.look_at(eye[start:stop], target[start:stop], up[start:stop])

    def destroy(self) -> None:
        for _, _, camera in self.parts:
            camera.destroy()
        self.parts.clear()
        self._data.clear()


def _native_lighting(
    scene: dict[str, Any], profile: dict[str, Any] | None
) -> dict[str, Any]:
    """Resolve explicit native light units without changing the shared manifest."""
    lighting = scene["lighting"]
    values = {
        "directional_intensity": lighting["dexsim_directional_intensity"],
        "environment_intensity": lighting["dexsim_environment_intensity"],
        "environment_color": [1.0, 1.0, 1.0],
    }
    overrides = (profile or {}).get("lighting_controls", {}).get("embodichain", {})
    if set(overrides) - values.keys():
        raise ValueError("Unsupported native lighting control")
    values.update(copy.deepcopy(overrides))
    color = values["environment_color"]
    if not isinstance(color, (list, tuple)) or len(color) != 3:
        raise ValueError("Environment color must contain three linear RGB values")
    for value in [
        values["directional_intensity"],
        values["environment_intensity"],
        *color,
    ]:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError("Native light controls must be finite and nonnegative")
    values["environment_color"] = list(color)
    return values


def material_display_color(linear_rgb: list[float]) -> list[float]:
    """Encode canonical linear RGB for VisualMaterialCfg's display-sRGB API.

    Args:
        linear_rgb: Scene-linear RGB reflectance.

    Returns:
        Display-sRGB values whose native decoding recovers the input reflectance.
    """
    return [
        12.92 * value if value <= 0.0031308 else 1.055 * value ** (1 / 2.4) - 0.055
        for value in linear_rgb
    ]


class EmbodiChainScene:
    """Create one camera group per view over identical scene replicas.

    Args:
        scene_path: Frozen manifest and canonical geometry directory.
        width: Output width in pixels.
        height: Output height in pixels.
        gpu_id: Physical CUDA and graphics adapter index.
        num_envs: Number of identical arenas.
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
        profile: dict[str, Any] | None = None,
        camera_names: tuple[str, ...] | None = None,
        camera_specs: list[dict[str, Any]] | None = None,
        diagnostic_pose_updates: bool = True,
        enable_native_profiling: bool = False,
    ) -> None:
        import numpy as np
        import torch
        from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
        from embodichain.lab.sim.cfg import (
            DenoisingCfg,
            DLSSCfg,
            LightCfg,
            RenderCfg,
            RigidObjectCfg,
        )
        from embodichain.lab.sim.material import VisualMaterialCfg
        from embodichain.lab.sim.sensors import Camera, CameraCfg
        from embodichain.lab.sim.shapes import CubeCfg, MeshCfg
        from embodichain.utils import configclass

        postprocess_evidence: dict[str, Any] = {}

        @configclass
        class _DiagnosticRenderCfg(RenderCfg):
            """Enable native timestamps only in explicitly intrusive runs."""

            native_profiling: bool = False

            def apply_to_dexsim_config(self, world_config: Any) -> None:
                super().apply_to_dexsim_config(world_config)
                world_config.enable_render_profiler = self.native_profiling
                postprocess = world_config.postprocess_config
                for name in (
                    "auto_exposure_enabled",
                    "bloom_enabled",
                    "dof_enabled",
                    "ca_enabled",
                    "vignette_enabled",
                ):
                    setattr(postprocess, name, False)
                postprocess_evidence.update(
                    {
                        name: getattr(postprocess, name)
                        for name in dir(postprocess)
                        if not name.startswith("_")
                        and isinstance(
                            getattr(postprocess, name), (bool, int, float, str)
                        )
                    }
                )

        self.scene = load_scene(scene_path)
        self.source_scene_sha256 = scene_fingerprint(self.scene)
        if camera_specs is not None:
            self.scene["cameras"] = copy.deepcopy(camera_specs)
        elif camera_names is not None:
            by_name = {item["name"]: item for item in self.scene["cameras"]}
            self.scene["cameras"] = [by_name[name] for name in camera_names]
        self._diagnostic_pose_updates = diagnostic_pose_updates
        self.profile = copy.deepcopy(profile)
        lighting_controls = _native_lighting(self.scene, profile)
        settings = (profile or {}).get(
            "embodichain",
            {"renderer": "hybrid", "spp": 4, "denoising": "nrd", "tone_mapping": False},
        )
        if (
            settings["renderer"] == "fast-rt"
            and settings["denoising"] == "nrd"
            and "depth" in modalities
        ):
            raise ValueError(
                "FastRT NRD exports jittered range depth, not canonical optical depth; "
                "use the DLAA profile with its pixel-centre public AOV pass"
            )
        self._linear_rgb_source = (
            settings["renderer"] == "hybrid" and settings["denoising"] == "dlss"
        )
        group_capacity = settings.get("camera_group_capacity", 32)
        if type(group_capacity) is not int or not 1 <= group_capacity <= 32:
            raise ValueError("Native camera-group capacity must be between 1 and 32")
        self._render_calls = 0
        self._scene_frame_id = 0
        self._pending_frame = None
        self._state_publications = 0
        self._pending_native_flush = True
        self._state_flushes = 0
        self._object_poses = {
            obj["uid"]: obj["world_from_object"] for obj in self.scene["objects"]
        }
        self.torch = torch
        self.device = f"cuda:{gpu_id}"
        self.modalities = modalities
        self.num_envs = num_envs
        self.objects = {}
        self.cameras = []
        self.sim = SimulationManager(
            SimulationManagerCfg(
                headless=True,
                width=width,
                height=height,
                device=self.device,
                gpu_id=gpu_id,
                num_envs=num_envs,
                enable_entity_gizmo=False,
                robot_ik_gizmo=None,
                startup_summary="off",
                dexsim_startup_info=False,
                render_cfg=_DiagnosticRenderCfg(
                    renderer=settings["renderer"],
                    spp=settings["spp"],
                    denoising=DenoisingCfg(
                        window="off", offscreen=settings["denoising"]
                    ),
                    dlss=DLSSCfg(
                        dlss_quality=settings.get("dlss_quality", 5),
                        tiled_enabled=settings.get("dlss_tiled", True),
                        exposure_compensation=1.0,
                    ),
                    tone_mapping_enabled=settings["tone_mapping"],
                    tone_mapping_exposure=settings.get("tone_mapping_exposure", 1.0),
                    native_profiling=enable_native_profiling,
                ),
            )
        )
        self.sim.set_ground_plane_visibility(False)
        light = self.scene["lighting"]
        self.sim.set_emission_light(
            color=lighting_controls["environment_color"],
            intensity=lighting_controls["environment_intensity"],
        )
        for obj in self.scene["objects"]:
            material = obj["material"]
            material_cfg = VisualMaterialCfg(
                uid=obj["uid"] + "_material",
                base_color=[
                    *material_display_color(material["base_color_linear"]),
                    1.0,
                ],
                roughness=material["roughness"],
                metallic=material["metallic"],
            )
            geometry = obj["geometry"]
            if geometry["kind"] == "box":
                shape = CubeCfg(size=geometry["size"], visual_material=material_cfg)
            else:
                with np.load(
                    scene_path.parent / geometry["path"], allow_pickle=False
                ) as arrays:
                    shape = MeshCfg(
                        vertices=arrays["vertices"].copy(),
                        triangles=arrays["triangles"].copy(),
                        normals=arrays["normals"].copy(),
                        visual_material=material_cfg,
                    )
            self.objects[obj["uid"]] = self.sim.add_rigid_object(
                RigidObjectCfg(
                    uid=obj["uid"],
                    body_type="static",
                    shape=shape,
                    init_local_pose=np.asarray(obj["world_from_object"]),
                )
            )
        self.sim.add_light(
            LightCfg(
                uid="benchmark_sun",
                light_type="direction",
                direction=tuple(light["direction"]),
                color=tuple(light["color"]),
                intensity=lighting_controls["directional_intensity"],
            )
        )
        self.sim.prepare()
        # All geometry is driven by frozen transforms. Disable integration in
        # addition to never invoking update()/a positive-time world step.
        self.sim.enable_physics(False)
        matrix = camera_intrinsics(self.scene, width, height)
        model = self.scene["camera_model"]
        for camera in self.scene["cameras"]:
            self.cameras.append(
                _CameraBatch(
                    Camera,
                    CameraCfg(
                        uid=camera["name"],
                        width=width,
                        height=height,
                        near=model["near_m"],
                        far=model["far_m"],
                        intrinsics=(
                            matrix[0][0],
                            matrix[1][1],
                            matrix[0][2],
                            matrix[1][2],
                        ),
                        enable_color="rgb" in modalities,
                        enable_depth="depth" in modalities,
                        enable_mask="seg" in modalities,
                        enable_normal="normals" in modalities,
                        extrinsics=CameraCfg.ExtrinsicsCfg(
                            eye=tuple(camera["eye"]),
                            target=tuple(camera["target"]),
                            up=tuple(camera["up"]),
                        ),
                    ),
                    self.sim,
                    torch,
                    group_capacity,
                )
            )
        self.sim.prepare()
        self._id_keys = self._id_values = None
        if "seg" in modalities:
            ids = torch.cat(
                [
                    self.objects[obj["uid"]].get_user_ids().to(self.device)
                    for obj in self.scene["objects"]
                ]
            ).to(torch.int64)
            canonical = torch.tensor(
                [obj["instance_id"] for obj in self.scene["objects"]],
                device=self.device,
                dtype=torch.int32,
            ).repeat_interleave(num_envs)
            order = torch.argsort(ids)
            self._id_keys, self._id_values = ids[order], canonical[order]
        self.metadata = {
            "backend": "embodichain",
            "renderer": "DexSim " + settings["renderer"],
            "spp": settings["spp"],
            "denoising": settings["denoising"],
            "native_rgb_encoding": (
                "linear_unorm8" if self._linear_rgb_source else "display_srgb_uint8"
            ),
            "delivered_rgb_encoding": "display_srgb_uint8",
            "rgb_delivery_conversion": (
                "IEC sRGB OETF on device, included in render-to-device timing; native linear-UNORM8 precision is not recoverable"
                if self._linear_rgb_source
                else "none"
            ),
            "tone_mapping": settings["tone_mapping"],
            "postprocess_evidence": {
                "scope": "applied native WorldConfig, not renderer readback",
                "settings": postprocess_evidence,
            },
            "resolved_render_configuration": self.sim.sim_config.render_cfg.to_dict(),
            "device": self.device,
            "scene_sha256": self.source_scene_sha256,
            "num_envs": num_envs,
            "cameras_per_env": len(self.cameras),
            "camera_names": [c["name"] for c in self.scene["cameras"]],
            "batch_order": "environment_then_camera",
            "intrinsics": [
                camera.get_intrinsics()[0].tolist() for camera in self.cameras
            ],
            "camera_poses": [
                camera.get_arena_pose(to_matrix=True)[0].cpu().tolist()
                for camera in self.cameras
            ],
            "depth_semantics": "optical_axis_metres; invalid_or_beyond_far_is_infinity",
            "geometry_output_policy": (
                "native output-resolution pixel-centre AOV pass"
                if settings["denoising"] == "dlss"
                else "legacy native AOVs; NRD jitter is not removed and FastRT NRD depth is range"
            ),
            "lighting": light,
            "effective_lighting_controls": lighting_controls,
            "lighting_control_units": "native DexSim intensities; physical conversion requires measured calibration",
            "default_ground_visible": False,
            "physics_steps": 0,
            "physics_enabled": False,
            "physics_evidence": "World.enable_physics(False); transforms published without positive-time stepping",
            "publication_completion": "World.update(0) plus World._flush_render_updates after changed transforms; native render-thread/GPU transfers complete without camera exposure",
            "batch_semantics": "arena_camera_groups_chunked_by_environment",
            "native_camera_group_capacity": group_capacity,
            "native_camera_group_count": sum(
                len(camera.parts) for camera in self.cameras
            ),
            "native_camera_group_sizes": [
                stop - start
                for camera in self.cameras
                for start, stop, _ in camera.parts
            ],
            "batch_assembly": "device concatenation of completed arena chunks, then environment_then_camera ordering; included in delivery timing",
            "effective_profile": self.profile,
            "native_profiler_enabled": enable_native_profiling,
            "render_boundary": "render_camera_group followed by blocking host delivery",
            "quality_status": "not_qualified",
            "material_input_encoding": "linear reflectance encoded as display-sRGB for VisualMaterialCfg",
        }

    def _begin_frame(self) -> None:
        if self._pending_frame is None:
            self._pending_frame = self.sim.render_frame(capture_camera_images=False)
            self._pending_frame.__enter__()
            try:
                self.sim.sync_render_state()
                if getattr(self, "_pending_native_flush", False):
                    self.sim._world._flush_render_updates()
                    self._pending_native_flush = False
                    self._state_flushes += 1
                self._state_publications += 1
            except BaseException:
                self._end_frame()
                raise

    def _end_frame(self) -> None:
        if self._pending_frame is not None:
            context, self._pending_frame = self._pending_frame, None
            context.__exit__(None, None, None)

    def render_device(self) -> dict[str, Any]:
        """Submit one render and expose canonical tensors without host readback."""
        self._begin_frame()
        try:
            return self._render_published_device()
        finally:
            self._end_frame()

    def _render_published_device(self) -> dict[str, Any]:
        torch = self.torch
        self.sim.render_camera_group(
            [group for camera in self.cameras for group in camera.group_ids]
        )
        for camera in self.cameras:
            camera.update(fetch_only=True)
        arrays = {}
        for modality in self.modalities:
            key = {
                "rgb": "color",
                "depth": "depth",
                "seg": "mask",
                "normals": "normal",
            }[modality]
            value = torch.stack(
                [camera.get_data()[key] for camera in self.cameras], dim=1
            )
            if modality == "rgb":
                value = value[..., :3]
                if self._linear_rgb_source:
                    value = _linear_u8_to_srgb_u8(value, torch)
            if modality == "depth":
                far = self.scene["camera_model"]["far_m"]
                value = torch.where((value > 0) & (value < far), value, torch.inf)
            if modality == "seg":
                indices = torch.searchsorted(
                    self._id_keys, value.contiguous().to(torch.int64)
                )
                valid = indices < self._id_keys.numel()
                safe = indices.clamp(max=self._id_keys.numel() - 1)
                valid = valid & (self._id_keys[safe] == value)
                value = torch.where(valid, self._id_values[safe], 0).to(torch.int32)
            arrays[modality] = value.flatten(0, 1).contiguous()
        self._render_calls += 1
        return arrays

    def synchronize(self) -> None:
        """Wait for the selected device's native-interoperability tensor work."""
        self.torch.cuda.synchronize(self.device)

    def native_profile(self) -> dict[str, Any]:
        """Read native GPU timer evidence without treating CPU clocks as GPU time."""
        import math
        from collections.abc import Mapping

        def plain(value: Any, level: int = 0) -> Any:
            if value is None or isinstance(value, (str, bool, int)):
                return value
            if isinstance(value, float):
                return value if math.isfinite(value) else None
            if isinstance(value, Mapping):
                return {str(key): plain(item, level + 1) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [plain(item, level + 1) for item in value]
            if level >= 3:
                return str(value)
            result = {}
            for name in dir(value):
                if not name.startswith("_"):
                    item = getattr(value, name)
                    if not callable(item):
                        result[name] = plain(item, level + 1)
            return result

        return {
            "enabled": self.metadata["native_profiler_enabled"],
            "source": "dexsim.World.get_render_profile_data",
            "scope": "native renderer snapshot; may include warmup; not a measured-window aggregate",
            "data": plain(self.sim._world.get_render_profile_data()),
        }

    def to_host(self, arrays: dict[str, Any]) -> dict[str, Any]:
        """Perform one blocking D2H per modality, preserving tensor-backed arrays."""
        return {name: value.cpu().numpy() for name, value in arrays.items()}

    def capture(self) -> dict[str, Any]:
        """Return an untimed completed host capture for preview/quality evidence."""
        arrays = self.render_device()
        self.synchronize()
        result = self.to_host(arrays)
        if self._diagnostic_pose_updates:
            self.refresh_metadata()
        return result

    def refresh_metadata(self) -> dict[str, Any]:
        """Read actual poses outside timing and record explicit render submissions."""
        self.metadata["camera_poses"] = [
            camera.get_arena_pose(to_matrix=True)[0].cpu().tolist()
            for camera in self.cameras
        ]
        self.metadata["render_submissions"] = self._render_calls
        self.metadata["scene_frame_id"] = self._scene_frame_id
        self.metadata["state_publications"] = self._state_publications
        self.metadata["changed_state_flushes"] = self._state_flushes
        return self.metadata

    def set_cameras(self, specs: list[dict[str, Any]]) -> None:
        """Apply same-count local camera views to every arena without stepping.

        Args:
            specs: Look-at views in camera-group order.
        """
        if len(specs) != len(self.cameras):
            raise ValueError("Changing camera topology requires a new worker")
        self._end_frame()
        for camera, spec in zip(self.cameras, specs, strict=True):
            values = [
                self.torch.tensor(
                    [spec[key]] * self.num_envs,
                    device=self.device,
                    dtype=self.torch.float32,
                )
                for key in ("eye", "target", "up")
            ]
            camera.look_at(*values)
        self.scene["cameras"] = copy.deepcopy(specs)
        self._pending_native_flush = True

    def apply_state(self, frame: dict[str, Any]) -> None:
        """Publish frozen object and camera transforms without running physics.

        Args:
            frame: Validated replay frame with object matrices and camera views.
        """
        self._end_frame()
        for uid, pose in frame["objects"].items():
            if pose != self._object_poses[uid]:
                values = self.torch.tensor(
                    [pose] * self.num_envs, dtype=self.torch.float32
                )
                self.objects[uid].set_local_pose(values)
                self._object_poses[uid] = pose
                self._pending_native_flush = True
        views = [
            frame["cameras"].get(view["name"], view) for view in self.scene["cameras"]
        ]
        if views != self.scene["cameras"]:
            self.set_cameras(views)
        self._begin_frame()
        self._scene_frame_id = int(frame["frame_id"])

    def close(self) -> None:
        """Release native child views before their world."""
        from embodichain.lab.sim import SimulationManager

        self._end_frame()
        for camera in self.cameras:
            camera.destroy()
        self.cameras.clear()
        self.objects.clear()
        self.sim.destroy(exit_process=False)
        self.sim = None
        gc.collect()
        SimulationManager.flush_cleanup_queue()
