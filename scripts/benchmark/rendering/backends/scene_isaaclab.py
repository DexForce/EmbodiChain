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

from collections.abc import Mapping
import copy
import math
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any

from scripts.benchmark.core.artifacts import stable_hash

from ..capture import assembly_operation
from ..color import linear_u8_to_srgb_u8
from ..scenes import camera_intrinsics, camera_world_pose, load_scene, scene_fingerprint

if TYPE_CHECKING:
    import torch

__all__ = ["IsaacLabScene", "camera_prim_path"]

_RTX_OBSERVED_SETTINGS = (
    "/rtx/rendermode",
    "/rtx/post/aa/op",
    "/rtx-transient/post/aa/limitedOps",
    "/rtx/post/dlss/execMode",
    "/rtx-transient/dldenoiser/enabled",
    "/rtx-transient/post/dlss/supported",
    "/rtx-transient/dlssg/enabled",
    "/rtx/directLighting/sampledLighting/samplesPerPixel",
    "/rtx/pathtracing/spp",
    "/rtx/pathtracing/totalSpp",
    "/rtx/pathtracing/optixDenoiser/enabled",
    "/rtx/pathtracing/adaptiveSampling/enabled",
    "/rtx/ecoMode/enabled",
    "/rtx-transient/resetPtAccumOnlyWhenExternalFrameCounterChanges",
    "/rtx/externalFrameCounter",
    "/rtx/viewTile/resolution/0",
    "/rtx/viewTile/resolution/1",
)

_RTX_POSTPROCESS_SETTINGS = (
    "/rtx/post/histogram/enabled",
    "/rtx/post/histogram/filterType",
    "/rtx/post/histogram/tau",
    "/rtx/post/histogram/whiteScale",
    "/rtx/post/histogram/useExposureClamping",
    "/rtx/post/histogram/minEV",
    "/rtx/post/histogram/maxEV",
    "/rtx/post/tonemap/op",
    "/rtx/post/tonemap/filmIso",
    "/rtx/post/tonemap/exposureTime",
    "/rtx/post/tonemap/fNumber",
    "/rtx/post/tonemap/responsivity",
    "/rtx/post/tonemap/whitepoint",
    "/rtx/post/tonemap/colorMode",
    "/rtx/post/tonemap/enableSrgbToGamma",
    "/rtx/post/tonemap/wrapValue",
    "/rtx/post/tonemap/dither",
    "/rtx/post/tonemap/ocio/enabled",
    "/rtx/post/colorcorr/enabled",
    "/rtx/post/colorcorr/mode",
    "/rtx/post/colorcorr/outputMode",
    "/rtx/post/colorcorr/saturation",
    "/rtx/post/colorcorr/contrast",
    "/rtx/post/colorcorr/gamma",
    "/rtx/post/colorcorr/gain",
    "/rtx/post/colorcorr/offset",
    "/rtx/post/colorgrad/enabled",
    "/rtx/post/colorgrad/gamma",
)

_CLAMP_COLOR_SETTINGS = (
    "/rtx/post/tonemap/op",
    "/rtx/post/colorcorr/enabled",
    "/rtx/post/colorgrad/enabled",
    "/rtx/post/tonemap/ocio/enabled",
)

_RTX_LIGHT_SETTINGS = (
    "/rtx/lighting/defaultUsdLuxVersion",
    "/rtx/sceneDb/ambientLightIntensity",
    "/rtx/sceneDb/ambientLightColor",
    "/rtx/scenedb/skipMostLights",
    "/rtx/domeLight/upperLowerStrategy",
)

_USD_LIGHT_ATTRIBUTES = (
    "inputs:intensity",
    "inputs:exposure",
    "inputs:color",
    "inputs:normalize",
    "inputs:angle",
    "inputs:diffuse",
    "inputs:specular",
    "inputs:enableColorTemperature",
    "inputs:colorTemperature",
    "omni:rtx:usdluxVersion",
    "photometric:illuminance",
    "photometric:power",
)


def _light_evidence(stage: Any, sim: Any) -> dict[str, Any]:
    """Read composed USD lights and ambient settings without authoring changes."""
    from pxr import Usd, UsdGeom, UsdLux

    lights = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdLux.LightAPI):
            continue
        attributes = {}
        for name in _USD_LIGHT_ATTRIBUTES:
            attribute = prim.GetAttribute(name)
            value = attribute.Get() if attribute else None
            if value is not None and not isinstance(value, (bool, int, float, str)):
                value = list(value)
            attributes[name] = {
                "value": value,
                "authored": bool(attribute and attribute.HasAuthoredValueOpinion()),
                "type": str(attribute.GetTypeName()) if attribute else None,
            }
        lights.append(
            {
                "path": str(prim.GetPath()),
                "type": prim.GetTypeName(),
                "applied_schemas": list(prim.GetAppliedSchemas()),
                "visibility": str(UsdGeom.Imageable(prim).ComputeVisibility()),
                "attributes": attributes,
            }
        )
    settings = {name: sim.get_setting(name) for name in _RTX_LIGHT_SETTINGS}
    return {
        "scope": "composed_usd_light_attributes_and_runtime_carb_readback",
        "usd_version": list(Usd.GetVersion()),
        "lights": lights,
        "settings": settings,
        "missing_settings": [name for name, value in settings.items() if value is None],
        "interpretation": "Attribute values include USD fallbacks; authored marks explicit opinions. The defaultUsdLuxVersion command setting is not proof of an unauthored light's renderer behavior. No radiometric unit equivalence is inferred.",
    }


def _clamp_rgb_conversion(overrides: dict[str, Any], observed: dict[str, Any]) -> bool:
    """Allow conversion only for an explicit, runtime-verified bare Clamp path."""
    requested = overrides.get("/rtx/post/tonemap/op")
    if type(requested) is not int or requested != 0:
        return False
    actual = observed.get("/rtx/post/tonemap/op")
    if type(actual) is not int or actual != 0:
        raise RuntimeError(
            "Explicit Clamp RGB conversion requires observed tonemap op 0"
        )
    for name in _CLAMP_COLOR_SETTINGS[1:]:
        if observed.get(name) is not False:
            raise RuntimeError(
                f"Clamp RGB conversion requires observed disabled {name}"
            )
    return True


def _resolution_evidence(
    width: int,
    height: int,
    requested_policy: str | None,
    observed_output: list[int] | None,
    settings: dict[str, Any],
    log_text: str,
) -> dict[str, Any]:
    warnings = [
        {
            "input_size_before_increase": [int(match[0]), int(match[1])],
            "minimum_input_dimension": int(match[2]),
        }
        for match in re.findall(
            r"DLSS increasing input dimensions: Render resolution of \((\d+),\s*(\d+)\) is below minimal input resolution of (\d+)",
            log_text,
        )
    ]
    return {
        "requested_policy": requested_policy,
        "requested_output_resolution": [width, height],
        "requested_internal_resolution": (
            [width, height] if requested_policy == "native_output_resolution" else None
        ),
        "observed_output_resolution": observed_output,
        # The runtime warning supplies the old size and a minimum, not the
        # final allocated shading extent. Do not fabricate a 300x300 result.
        "resolved_internal_resolution": None,
        "internal_resolution_missing_reason": "Installed RTX Python APIs expose output/atlas extents, not resolved internal DLSS/RR shading extents",
        "native_resolution_status": "unsupported" if warnings else "not_verified",
        "runtime_resize_warnings": warnings,
        "observed_rtx_settings": settings,
    }


def camera_prim_path(env_id: int, camera_id: int) -> str:
    """Return a view with its own parent, as required by Camera's sensor count.

    Args:
        env_id: Zero-based scene replica index.
        camera_id: Zero-based camera index within the replica.

    Returns:
        USD camera path with one parent prim per logical camera exposure.
    """
    return f"/World/Envs/env_{env_id:06d}/Cameras/view_{camera_id:02d}/Camera"


def _selected_cameras(
    scene: dict[str, Any],
    names: tuple[str, ...] | None,
    specs: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    cameras = copy.deepcopy(scene["cameras"] if specs is None else specs)
    by_name = {camera["name"]: camera for camera in cameras}
    if not cameras or len(by_name) != len(cameras):
        raise ValueError("Camera selection must contain unique named views")
    if names is not None:
        if not names or len(set(names)) != len(names) or set(names) - set(by_name):
            raise ValueError("Camera selection contains missing or duplicate names")
        cameras = [by_name[name] for name in names]
    for camera in cameras:
        if camera.get("camera_model", scene["camera_model"]) != scene["camera_model"]:
            raise ValueError("Selected cameras must use the frozen scene camera model")
        for key in ("eye", "target", "up"):
            values = camera[key]
            if len(values) != 3 or not all(math.isfinite(value) for value in values):
                raise ValueError(
                    "Selected camera vectors must contain three finite values"
                )
    return cameras


def _render_settings(profile: dict[str, Any] | None) -> dict[str, Any]:
    settings = {
        "antialiasing_mode": "DLAA",
        "enable_dlssg": False,
        "enable_dl_denoiser": True,
        "samples_per_pixel": 4,
    }
    if profile is not None:
        if profile.get("frame_generation", False) or profile["isaaclab"].get(
            "enable_dlssg", False
        ):
            raise ValueError(
                "Frame generation is forbidden for measured camera exposures"
            )
        settings.update(copy.deepcopy(profile["isaaclab"]))
    settings["enable_dlssg"] = False
    for path, value in _explicit_carb_settings(settings).items():
        if path == "/rtx-transient/dlssg/enabled" and value:
            raise ValueError("Frame generation is forbidden in explicit carb settings")
    return settings


def _native_lighting(
    scene: dict[str, Any], profile: dict[str, Any] | None
) -> dict[str, Any]:
    """Resolve native USD light controls without altering the frozen manifest."""
    lighting = scene["lighting"]
    values = {
        "directional_intensity": lighting["isaac_directional_intensity"],
        "environment_intensity": lighting["isaac_dome_intensity"],
        "environment_color": [1.0, 1.0, 1.0],
    }
    overrides = (profile or {}).get("lighting_controls", {}).get("isaaclab", {})
    if not isinstance(overrides, dict) or set(overrides) - values.keys():
        raise ValueError("Unsupported Isaac lighting control")
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
            raise ValueError("Isaac light controls must be finite and nonnegative")
    values["environment_color"] = list(color)
    return values


def _isaac_light_contract(profile: dict[str, Any] | None) -> dict[str, Any]:
    """Validate only explicitly supplied USD light attributes before Kit starts."""
    contract = (profile or {}).get("isaaclab_light_contract", {})
    allowed = {
        "usdlux_version",
        "sun_normalize",
        "sun_exposure",
        "sun_angle_deg",
        "dome_exposure",
    }
    if not isinstance(contract, dict) or set(contract) - allowed:
        raise ValueError("Unsupported isaaclab_light_contract attribute")
    for name, value in contract.items():
        if name == "usdlux_version":
            if type(value) is not int or value not in {2411, 2505}:
                raise ValueError("usdlux_version must be integer 2411 or 2505")
        elif name == "sun_normalize":
            if type(value) is not bool:
                raise ValueError("sun_normalize must be a boolean")
        elif (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or (name == "sun_angle_deg" and value < 0)
        ):
            raise ValueError(
                "Light exposures must be finite numbers and sun_angle_deg nonnegative"
            )
    return copy.deepcopy(contract)


def _apply_light_contract(sun: Any, dome: Any | None, contract: dict[str, Any]) -> None:
    """Author the selected contract without applying Kit's look-preserving upgrade."""
    from pxr import Sdf

    if "usdlux_version" in contract:
        for light in (sun, dome):
            if light is not None:
                light.GetPrim().CreateAttribute(
                    "omni:rtx:usdluxVersion", Sdf.ValueTypeNames.Int, custom=True
                ).Set(contract["usdlux_version"])
    for name, create in (
        ("sun_normalize", sun.CreateNormalizeAttr),
        ("sun_exposure", sun.CreateExposureAttr),
        ("sun_angle_deg", sun.CreateAngleAttr),
    ):
        if name in contract:
            create(contract[name])
    if dome is not None and "dome_exposure" in contract:
        dome.CreateExposureAttr(contract["dome_exposure"])


def _explicit_carb_settings(settings: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in (settings.get("carb_settings") or {}).items():
        path = (
            key
            if key.startswith("/")
            else "/" + key.replace(".", "/").replace("_", "/")
        )
        result[path] = value
    return result


def _apply_carb_overrides(sim: Any, settings: dict[str, Any]) -> dict[str, Any]:
    overrides = _explicit_carb_settings(settings)
    for path, value in overrides.items():
        sim.set_setting(path, value)
    return overrides


def _configure_accumulation(sim: Any, policy: str) -> dict[str, Any]:
    if policy == "renderer_default":
        return {"policy": policy, "control": "no_override"}
    if policy != "reset_each_exposure":
        raise ValueError(f"Unsupported accumulation policy: {policy}")
    if sim.get_setting("/rtx/rendermode") != "PathTracing":
        raise RuntimeError(
            "reset_each_exposure requires explicitly selected PathTracing"
        )
    if sim.get_setting("/rtx/pathtracing/totalSpp") != 0:
        raise RuntimeError(
            "Measured PathTracing requires explicit unlimited totalSpp=0"
        )
    spp = sim.get_setting("/rtx/pathtracing/spp")
    if not isinstance(spp, int) or spp < 1:
        raise RuntimeError(
            "PathTracing samples per exposure must be explicitly positive"
        )
    before = sim.get_setting("/rtx/externalFrameCounter")
    sim.set_setting(
        "/rtx-transient/resetPtAccumOnlyWhenExternalFrameCounterChanges", True
    )
    if before is None:
        # This is an explicit renderer control input, never a measurement of
        # physical time or proof that a GPU exposure occurred.
        sim.set_setting("/rtx/externalFrameCounter", 0)
    return {
        "policy": policy,
        "pt_spp": spp,
        "pt_total_spp_cap": 0,
        "external_counter_before_setup": before,
        "external_counter_after_setup": sim.get_setting("/rtx/externalFrameCounter"),
        "reset_control_observed": sim.get_setting(
            "/rtx-transient/resetPtAccumOnlyWhenExternalFrameCounterChanges"
        ),
        "control_source": "installed Replicator orchestrator._set_motion_blur_settings; explicit external-frame-counter history control",
    }


def _advance_pt_history(sim: Any) -> dict[str, Any]:
    if (
        sim.get_setting("/rtx/rendermode") != "PathTracing"
        or sim.get_setting("/rtx/pathtracing/totalSpp") != 0
    ):
        raise RuntimeError("Frozen PathTracing mode or unlimited sample cap changed")
    if (
        sim.get_setting(
            "/rtx-transient/resetPtAccumOnlyWhenExternalFrameCounterChanges"
        )
        is not True
    ):
        raise RuntimeError(
            "Explicit PathTracing accumulation reset control was disabled"
        )
    before = sim.get_setting("/rtx/externalFrameCounter")
    if not isinstance(before, int):
        raise RuntimeError("Native PT external-frame control is not an integer")
    sim.set_setting("/rtx/externalFrameCounter", before + 1)
    after = sim.get_setting("/rtx/externalFrameCounter")
    if after != before + 1:
        raise RuntimeError("Native PT accumulation-reset control did not update")
    return {
        "before": before,
        "requested": before + 1,
        "observed_after_write": after,
        "role": "renderer history-reset input, not a measured GPU exposure counter",
    }


def _object_matrix(obj: dict[str, Any], pose: Any, offset: Any) -> Any:
    import numpy as np

    matrix = np.asarray(pose, dtype=np.float64).copy()
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], [0, 0, 0, 1])
    ):
        raise ValueError(f"Invalid object transform: {obj['uid']}")
    matrix[:3, 3] += offset
    if obj["geometry"]["kind"] == "box":
        matrix[:3, :3] = matrix[:3, :3] @ np.diag(obj["geometry"]["size"])
    return matrix


def _geometry_prototypes(
    scene: dict[str, Any], scene_path: Path
) -> tuple[Any, dict[str, str], dict[str, float]]:
    """Author immutable geometry once in a retained, non-rendered USD layer."""
    import numpy as np
    from pxr import Usd, UsdGeom, Vt

    assets = Usd.Stage.CreateInMemory()
    prototypes, by_uid, radii = {}, {}, {}
    for obj in scene["objects"]:
        geometry = obj["geometry"]
        kind = geometry["kind"]
        key = "unit_cube" if kind == "box" else geometry["sha256"]
        if key not in prototypes:
            path = f"/Assets/geometry_{len(prototypes):04d}"
            UsdGeom.Xform.Define(assets, path)
            if kind == "box":
                shape = UsdGeom.Cube.Define(assets, path + "/Geometry")
                shape.CreateSizeAttr(1.0)
                radius = math.sqrt(3) / 2
            elif kind == "mesh":
                with np.load(
                    scene_path.parent / geometry["path"], allow_pickle=False
                ) as arrays:
                    vertices = arrays["vertices"]
                    triangles = arrays["triangles"]
                    normals = arrays["normals"]
                    shape = UsdGeom.Mesh.Define(assets, path + "/Geometry")
                    shape.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices))
                    shape.CreateFaceVertexCountsAttr([3] * len(triangles))
                    shape.CreateFaceVertexIndicesAttr(triangles.reshape(-1).tolist())
                    shape.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals))
                    shape.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
                    shape.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
                    radius = float(np.linalg.norm(vertices, axis=1).max())
            else:
                raise ValueError(f"Unsupported canonical geometry: {kind}")
            prototypes[key] = (path, radius)
        by_uid[obj["uid"]] = prototypes[key][0]
        radii[obj["uid"]] = (
            float(np.linalg.norm(geometry["size"]) / 2)
            if kind == "box"
            else prototypes[key][1]
        )
    return assets, by_uid, radii


def _segmentation_map(labels: Any, object_paths: dict[str, int]) -> dict[int, int]:
    """Resolve real instance paths, including proxies, without an O(N²) scan."""
    if not isinstance(labels, dict) or not labels:
        raise RuntimeError("Native instance idToLabels mapping is missing")
    mapping = {0: 0}
    for native_id, label in labels.items():
        raw_id = int(native_id)
        if raw_id < 0:
            raise RuntimeError("Native instance IDs must be nonnegative")
        if not isinstance(label, (str, dict)):
            raise RuntimeError("Native instance label has no prim path")
        path = label if isinstance(label, str) else label.get("primPath", "")
        if not isinstance(path, str):
            raise RuntimeError("Native instance prim path must be a string")
        if path.upper() in ("BACKGROUND", "BACKGROUND_ID"):
            mapping[raw_id] = 0
            continue
        while path and path not in object_paths:
            path = path.rsplit("/", 1)[0] if "/" in path else ""
        canonical = object_paths.get(path, -1)
        if raw_id == 0 and canonical != 0:
            raise RuntimeError("Native background zero cannot identify a scene object")
        mapping[raw_id] = canonical
    if not any(value > 0 for value in mapping.values()):
        raise RuntimeError(
            "Native segmentation mapping identifies no frozen scene objects"
        )
    return mapping


def _instance_geometry(
    stage: Any,
    path: str,
    obj: dict[str, Any],
    prototype_stage: Any,
    prototype_path: str,
    offset: Any,
) -> tuple[Any, Any]:
    from pxr import Gf, UsdGeom

    shape = UsdGeom.Xform.Define(stage, path)
    shape.GetPrim().GetReferences().AddReference(
        prototype_stage.GetRootLayer().identifier, prototype_path
    )
    shape.GetPrim().SetInstanceable(True)
    operation = shape.AddTransformOp()
    matrix = _object_matrix(obj, obj["world_from_object"], offset)
    operation.Set(Gf.Matrix4d(*matrix.T.reshape(-1).tolist()))
    return shape, operation


class IsaacLabScene:
    """Author matching USD geometry and one vectorized camera render product.

    Args:
        scene_path: Frozen manifest and canonical geometry directory.
        width: Output width in pixels.
        height: Output height in pixels.
        gpu_id: Physical CUDA and graphics adapter index.
        num_envs: Number of identical scene replicas.
        modalities: Requested canonical output names.
        profile: Frozen formal profile, or ``None`` for the preview settings.
        camera_names: Optional ordered subset of named cameras.
        camera_specs: Optional explicit look-at views using the frozen intrinsics.
        diagnostic_pose_updates: Enable native per-exposure pose diagnostics.
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
    ) -> None:
        scene_path = Path(scene_path).resolve()
        self.scene = load_scene(scene_path)
        self._scene_sha256 = scene_fingerprint(self.scene)
        self.camera_specs = _selected_cameras(self.scene, camera_names, camera_specs)
        self._assembly_operations: list[dict[str, Any]] = []
        settings = _render_settings(profile)
        lighting_controls = _native_lighting(self.scene, profile)
        light_contract = _isaac_light_contract(profile)
        self._accumulation_policy = (profile or {}).get(
            "accumulation_policy", "renderer_default"
        )
        if self._accumulation_policy not in {"renderer_default", "reset_each_exposure"}:
            raise ValueError("Unknown explicitly requested accumulation policy")
        if min(width, height, num_envs) < 1 or gpu_id < 0:
            raise ValueError("Image dimensions and environment count must be positive")
        if (
            not modalities
            or len(set(modalities)) != len(modalities)
            or set(modalities) - {"rgb", "depth", "seg", "normals"}
        ):
            raise ValueError("Unsupported or duplicate rendering modalities")
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
        import warp as wp
        from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import Camera, CameraCfg
        from isaaclab_physx.renderers import IsaacRtxRendererCfg

        self.torch, self.wp, self.Gf, self.Sdf = torch, wp, Gf, Sdf
        self.modalities = modalities
        self.num_envs = num_envs
        self.width, self.height = width, height
        self.profile = copy.deepcopy(profile)
        self.diagnostic_pose_updates = diagnostic_pose_updates
        self._frame_dt = 1 / 60 if profile is None else float(profile["frame_time_s"])
        self._segmentation_labels = None
        self._segmentation_keys = self._segmentation_values = None
        self._objects = {obj["uid"]: obj for obj in self.scene["objects"]}
        self._object_ops = {uid: [] for uid in self._objects}
        self._object_poses = {
            uid: np.asarray(obj["world_from_object"]).copy()
            for uid, obj in self._objects.items()
        }
        self.sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(
                dt=1 / 240,
                device=self.device,
                render=sim_utils.RenderCfg(**settings),
                visualizer_cfgs=[],
            )
        )
        import omni.usd
        import omni.kit.app

        stage = omni.usd.get_context().get_stage()
        self.stage = stage
        self._kit_app = omni.kit.app.get_app()
        self.object_paths = {}
        self._prototype_stage, prototypes, radii = _geometry_prototypes(
            self.scene, scene_path
        )
        self._geometry_radii = radii
        self._scene_radius = max(
            float(np.linalg.norm(np.asarray(obj["world_from_object"])[:3, 3]))
            + radii[obj["uid"]]
            for obj in self.scene["objects"]
        )
        camera_radius = max(
            float(np.linalg.norm(camera["eye"])) for camera in self.camera_specs
        )
        # No RTX per-camera visibility collection exists in this frozen backend.
        # A 2D grid bounds coordinate magnitudes and keeps neighboring geometry
        # outside primary clipping range. Indirect transport remains shared.
        self._replica_spacing = 4 * (
            self.scene["camera_model"]["far_m"] + self._scene_radius + camera_radius
        )
        columns = math.ceil(math.sqrt(num_envs))
        self._env_offsets = [
            np.array([index % columns, index // columns, 0], dtype=float)
            * self._replica_spacing
            for index in range(num_envs)
        ]
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
            offset = self._env_offsets[env_id]
            prefix = f"/World/Envs/env_{env_id:06d}"
            UsdGeom.Xform.Define(stage, prefix)
            for index, obj in enumerate(self.scene["objects"]):
                path = prefix + "/Objects/" + obj["uid"]
                shape, operation = _instance_geometry(
                    stage,
                    path,
                    obj,
                    self._prototype_stage,
                    prototypes[obj["uid"]],
                    offset,
                )
                self._object_ops[obj["uid"]].append(operation)
                UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(
                    UsdShade.Material.Get(
                        stage, f"/World/Materials/material_{index:04d}"
                    )
                )
                sim_utils.add_labels(
                    shape.GetPrim(), [obj["uid"]], instance_name="class"
                )
                self.object_paths[path] = obj["instance_id"]
            for camera_index, spec in enumerate(self.camera_specs):
                camera_path = camera_prim_path(env_id, camera_index)
                view_path = camera_path.rsplit("/", 1)[0]
                UsdGeom.Xform.Define(stage, view_path)
                spawn.func(camera_path, spawn)
                eye, quaternion = camera_world_pose(spec)
                orientations.append(quaternion)
                eye_positions.append((np.asarray(eye) + offset).tolist())
        lighting = self.scene["lighting"]
        light = UsdLux.DistantLight.Define(stage, "/World/Sun")
        light.CreateIntensityAttr(lighting_controls["directional_intensity"])
        light.CreateColorAttr(Gf.Vec3f(*lighting["color"]))
        light.CreateAngleAttr(0.0)
        rotation = Gf.Rotation(
            Gf.Vec3d(0, 0, -1), Gf.Vec3d(*lighting["direction"])
        ).GetQuat()
        UsdGeom.Xformable(light).AddOrientOp().Set(
            Gf.Quatf(rotation.GetReal(), Gf.Vec3f(rotation.GetImaginary()))
        )
        dome = None
        if lighting_controls["environment_intensity"]:
            dome = UsdLux.DomeLight.Define(stage, "/World/Dome")
            dome.CreateIntensityAttr(lighting_controls["environment_intensity"])
            dome.CreateColorAttr(Gf.Vec3f(*lighting_controls["environment_color"]))
        _apply_light_contract(light, dome, light_contract)
        sim_utils.update_stage()
        self.camera = Camera(
            CameraCfg(
                prim_path="/World/Envs/env_.*/Cameras/view_.*/Camera",
                update_period=0.0,
                update_latest_camera_pose=diagnostic_pose_updates,
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
        # Camera initialization and native render-product creation may select
        # their own RTX mode. Reapply only the caller's explicitly frozen keys.
        self._post_camera_overrides = _apply_carb_overrides(self.sim, settings)
        self._clamp_rgb_requested = (
            type(self._post_camera_overrides.get("/rtx/post/tonemap/op")) is int
            and self._post_camera_overrides["/rtx/post/tonemap/op"] == 0
        )
        accumulation_setup = _configure_accumulation(
            self.sim, self._accumulation_policy
        )
        with wp.ScopedStream(
            wp.stream_from_torch(torch.cuda.current_stream(self.device))
        ):
            self.camera.set_world_poses(
                positions=torch.tensor(eye_positions, device=self.device),
                orientations=torch.tensor(orientations, device=self.device),
                convention="opengl",
            )
        self.sim.render_context.reset_transform_cadence()
        import omni.physx
        import omni.timeline

        self._physx = omni.physx.get_physx_interface()
        self._timeline = omni.timeline.get_timeline_interface()
        expected_paths = [
            camera_prim_path(env, camera)
            for env in range(num_envs)
            for camera in range(len(self.camera_specs))
        ]
        observed_paths = [str(camera.GetPath()) for camera in self.camera._sensor_prims]
        if observed_paths != expected_paths:
            raise RuntimeError(
                "Native camera view order is not environment-major camera order"
            )
        self.metadata = {
            "backend": "isaaclab",
            "renderer": "Isaac RTX tiled",
            "spp": settings["samples_per_pixel"],
            "antialiasing": settings["antialiasing_mode"],
            "dl_denoiser": settings["enable_dl_denoiser"],
            "dlss_frame_generation": False,
            "device": self.device,
            "scene_sha256": self._scene_sha256,
            "view_sha256": stable_hash(self.camera_specs),
            "profile_sha256": stable_hash(profile) if profile is not None else None,
            "render_settings": settings,
            "requested_renderer_settings": {
                "rendering_preset": settings.get("rendering_mode"),
                "mode": self._post_camera_overrides.get("/rtx/rendermode"),
                "antialiasing": settings.get("antialiasing_mode"),
                "aa_op_override": self._post_camera_overrides.get("/rtx/post/aa/op"),
                "dl_denoiser": settings.get("enable_dl_denoiser"),
                "rt_direct_light_spp": settings.get("samples_per_pixel"),
                "pt_spp": self._post_camera_overrides.get("/rtx/pathtracing/spp"),
                "pt_total_spp_cap": self._post_camera_overrides.get(
                    "/rtx/pathtracing/totalSpp"
                ),
            },
            "post_camera_carb_overrides": self._post_camera_overrides,
            "accumulation_setup": accumulation_setup,
            "spp_semantics": "requested RenderCfg direct-light setting; actual total rays per pixel are not measured",
            "diagnostic_pose_updates": diagnostic_pose_updates,
            "num_envs": num_envs,
            "cameras_per_env": len(self.camera_specs),
            "camera_names": [c["name"] for c in self.camera_specs],
            "batch_order": "environment_then_camera",
            "camera_sensor_instances": 1,
            "camera_prim_paths": observed_paths,
            "rgb_encoding": "srgb_uint8",
            "native_rgb_encoding": (
                "linear_unorm8" if self._clamp_rgb_requested else "display_srgb_uint8"
            ),
            "delivered_rgb_encoding": "display_srgb_uint8",
            "rgb_delivery_conversion": (
                "IEC sRGB OETF on device after verified Clamp, included in render-to-device timing; native linear-UNORM8 precision is not recoverable"
                if self._clamp_rgb_requested
                else "none"
            ),
            "segmentation_semantics": "canonical_instance_ids_zero_background",
            "instance_ids": [obj["instance_id"] for obj in self.scene["objects"]],
            "depth_semantics": "optical_axis_metres; invalid_or_beyond_far_is_infinity",
            "lighting": lighting,
            "effective_lighting_controls": lighting_controls,
            "requested_usd_light_contract": light_contract,
            "lighting_control_units": "native USD light intensities and linear RGB environment colour; physical conversion requires measured calibration",
            "geometry_strategy": "immutable anonymous USD layer; instanceable object references; per-instance root transforms and semantic labels",
            "geometry_prototype_count": len(set(prototypes.values())),
            "geometry_instance_count": len(self.object_paths),
            "usd_observed_prototype_count": len(stage.GetPrototypes()),
            "usd_observed_instance_count": sum(
                stage.GetPrimAtPath(path).IsInstance() for path in self.object_paths
            ),
            "replica_offsets_m": [offset.tolist() for offset in self._env_offsets],
            "replica_spacing_m": self._replica_spacing,
            "environment_isolation": {
                "primary_rays": "spatial grid, neighbor bounds outside camera far plane",
                "indirect_lighting": "not_qualified; all replicas share one RTX scene",
                "native_per_camera_isolation": "not exposed by frozen IsaacRtxRenderer",
            },
            "quality_limitations": (
                [
                    "RTX replicas share one scene; indirect-lighting environment isolation is not qualified"
                ]
                if num_envs > 1
                else []
            ),
            "render_boundary": "fresh render generation; CUDA annotators; Warp-to-Torch stream ordering; selected-device synchronization; optional explicit host delivery",
            "completion_semantics": "Replicator CUDA annotator get_data, Warp stream event, Warp device synchronize, torch.cuda.synchronize(selected device)",
            "photometric_calibration": (profile or {}).get(
                "photometric_calibration", {"status": "not_qualified"}
            ),
            "quality_status": "not_qualified",
        }
        # Initial camera data materialization belongs to setup, not a measured
        # exposure. It also lets optional diagnostics read initialized buffers.
        self._camera_data = self.camera.data
        self.refresh_metadata()
        self.sim.pause()
        self.sim.set_setting("/app/player/playSimulations", False)
        if self.sim.is_playing():
            raise RuntimeError("Pure-render setup could not pause the physics timeline")
        self.metadata["physics_enabled"] = bool(self.sim.is_playing())
        self.metadata["physics_control"] = (
            "SimulationContext.pause plus disabled Kit player simulations after setup; native counters checked during capture"
        )
        self._physics_baseline = self._physics_snapshot()
        self.metadata["no_step_evidence"] = {
            "baseline": self._physics_baseline,
            "last_observed": self._physics_baseline,
        }
        self.metadata["physics_steps"] = (
            None  # No capture interval has yet been observed.
        )

    def _physics_snapshot(self) -> dict[str, Any]:
        native_clock = getattr(self._physx, "get_simulation_time", None)
        native_time = None
        missing_reason = "Installed PhysX interface does not expose get_simulation_time"
        if callable(native_clock):
            try:
                value = float(native_clock())
                if math.isfinite(value):
                    native_time, missing_reason = value, None
                else:
                    missing_reason = "Installed PhysX clock returned a non-finite value"
            except (AttributeError, RuntimeError, TypeError) as error:
                missing_reason = str(error)
        return {
            "isaac_step_count": int(self.sim.get_physics_step_count()),
            "physx_simulation_time_s": native_time,
            "physx_time_missing_reason": missing_reason,
            "timeline_time_s": float(self._timeline.get_current_time()),
            "timeline_playing": bool(self.sim.is_playing()),
            "kit_player_simulations_enabled": self.sim.get_setting(
                "/app/player/playSimulations"
            ),
        }

    def _verify_no_step(self, before: dict[str, Any]) -> None:
        after = self._physics_snapshot()
        baseline = self._physics_baseline
        self.metadata["no_step_evidence"] = {
            "baseline": baseline,
            "before": before,
            "after": after,
        }
        self.metadata["physics_steps"] = (
            after["isaac_step_count"] - baseline["isaac_step_count"]
        )
        if before != after or after != baseline:
            raise RuntimeError(
                "Pure-render capture advanced observed Isaac or PhysX simulation state"
            )

    def _canonical_segmentation(self, raw: Any, info: dict[str, Any]) -> Any:
        segmentation_info = info.get("instance_id_segmentation_fast")
        labels = (
            segmentation_info.get("idToLabels")
            if isinstance(segmentation_info, dict)
            else None
        )
        if not isinstance(labels, dict) or not labels:
            raise RuntimeError("Native instance idToLabels mapping is missing")
        if labels != self._segmentation_labels:
            mapping = _segmentation_map(labels, self.object_paths)
            keys = sorted(mapping)
            self._segmentation_keys = self.torch.tensor(
                keys, dtype=self.torch.int64, device=self.device
            )
            self._segmentation_values = self.torch.tensor(
                [mapping[key] for key in keys],
                dtype=self.torch.int32,
                device=self.device,
            )
            self._segmentation_labels = copy.deepcopy(labels)
            self.metadata["segmentation_mapping_available"] = True
            self.metadata["segmentation_mapping_sha256"] = stable_hash(
                {str(key): value for key, value in mapping.items()}
            )
            self.metadata["segmentation_mapping_entries"] = len(mapping)
        raw = raw.to(dtype=self.torch.int64).contiguous()
        indices = self.torch.searchsorted(self._segmentation_keys, raw).clamp(
            max=len(self._segmentation_keys) - 1
        )
        valid = self._segmentation_keys[indices] == raw
        mapped = self.torch.where(valid, self._segmentation_values[indices], -1)
        # The CUDA asynchronous assertion checks one device scalar without a
        # host readback; synchronize() surfaces unknown/unlabelled foreground.
        self.torch._assert_async(
            (mapped >= 0).all(), "Unmapped native foreground instance ID"
        )
        return mapped.contiguous()

    def _canonical_rgb(self, value: torch.Tensor) -> torch.Tensor:
        if not getattr(self, "_clamp_rgb_requested", False):
            return value
        observed = {name: self.sim.get_setting(name) for name in _CLAMP_COLOR_SETTINGS}
        if not _clamp_rgb_conversion(self._post_camera_overrides, observed):
            raise RuntimeError("Clamp RGB conversion is no longer explicitly selected")
        if value.dtype != self.torch.uint8:
            raise RuntimeError("Clamp RGB normalization requires native uint8 storage")
        encoded = linear_u8_to_srgb_u8(value, self.torch)
        self.metadata["rgb_conversion_verified_settings"] = observed
        return encoded

    def render_device(self) -> dict[str, torch.Tensor]:
        """Produce a fresh canonical GPU batch without image host readback.

        Returns:
            CUDA arrays in environment-major camera order. Depth is optical-axis
            metres, segmentation is int32 with canonical IDs and zero background.
        """
        before = self._physics_snapshot()
        generation = int(self.sim.render_generation)
        kit_update_before = int(self._kit_app.get_update_number())
        accumulation_event = (
            _advance_pt_history(self.sim)
            if self._accumulation_policy == "reset_each_exposure"
            else None
        )
        self.sim.render()
        self.camera.update(dt=self._frame_dt, force_recompute=True)
        data = self.camera.data
        # Frozen RTX's app-pump helper restores this flag to True; preserve the
        # explicit no-simulation state between exposures as well as during them.
        self.sim.set_setting("/app/player/playSimulations", False)
        self._camera_data = data
        # ProxyArray.torch does not establish producer/consumer stream ordering.
        torch_stream = self.wp.stream_from_torch(
            self.torch.cuda.current_stream(self.device)
        )
        torch_stream.wait_stream(self.wp.get_stream(self.device))
        arrays = {}
        self._assembly_operations = []
        for modality in self.modalities:
            key = "instance_id_segmentation_fast" if modality == "seg" else modality
            value = data.output[key].torch
            if str(value.device) != self.device:
                raise RuntimeError(
                    f"Renderer output {modality} is not on the selected CUDA device"
                )
            if modality == "rgb":
                value = self._canonical_rgb(value[..., :3])
            elif modality in {"depth", "seg"} and value.ndim == 4:
                value = value[..., 0]
            if modality == "depth":
                far = self.scene["camera_model"]["far_m"]
                value = self.torch.where(
                    (value > 0) & (value < far), value, self.torch.inf
                )
            elif modality == "seg":
                value = self._canonical_segmentation(value, data.info)
            elif modality == "normals":
                value = value[..., :3].clone()
                self._assembly_operations.append(
                    assembly_operation("normal_channel_pack", modality, value)
                )
            expected_dtype = {
                "rgb": self.torch.uint8,
                "depth": self.torch.float32,
                "seg": self.torch.int32,
                "normals": self.torch.float32,
            }[modality]
            shape = (self.num_envs * len(self.camera_specs), self.height, self.width)
            if modality in ("rgb", "normals"):
                shape += (3,)
            if value.dtype != expected_dtype or tuple(value.shape) != shape:
                raise RuntimeError(f"Noncanonical {modality} output shape or dtype")
            arrays[modality] = value.contiguous()
            if not value.is_contiguous():
                self._assembly_operations.append(
                    assembly_operation("contiguous_pack", modality, arrays[modality])
                )
        self._verify_no_step(before)
        actual_generation = int(self.sim.render_generation)
        if actual_generation != generation + 1:
            raise RuntimeError(
                "Render generation did not advance by one fresh exposure"
            )
        self.metadata["render_generation"] = actual_generation
        if accumulation_event is not None:
            accumulation_event["observed_after_render"] = self.sim.get_setting(
                "/rtx/externalFrameCounter"
            )
            self.metadata["accumulation_event_evidence"] = accumulation_event
            if (
                accumulation_event["observed_after_render"]
                != accumulation_event["requested"]
            ):
                raise RuntimeError(
                    "Another runtime component changed the PT exposure history control"
                )
        kit_update_after = int(self._kit_app.get_update_number())
        self.metadata["exposure_event_evidence"] = {
            "isaac_render_generation_before": generation,
            "isaac_render_generation_after": actual_generation,
            "kit_update_before": kit_update_before,
            "kit_update_after": kit_update_after,
            "interpretation": "observed application render-request and Kit app-update counters, not a measured GPU ray/exposure counter",
            "native_gpu_exposure_counter": None,
            "native_gpu_exposure_counter_missing_reason": "Pinned RTX camera API does not expose a per-render-product GPU exposure counter",
        }
        if kit_update_after <= kit_update_before:
            raise RuntimeError(
                "Native camera capture did not advance an observed Kit app-update event"
            )
        self._last_capture_submission_id = actual_generation
        self._last_capture_identity = {
            "frame_id": self.metadata.get("frame_id"),
            "time_s": self.metadata.get("time_s"),
            "identity_scope": "frozen state used by the last successful render submission; not a renderer source timestamp",
            "camera_names": [view["name"] for view in self.camera_specs],
            "num_envs": self.num_envs,
            "modalities": list(self.modalities),
            "batch_order": "environment_then_camera",
        }
        self._last_assembly_operations = self._assembly_operations
        return arrays

    def capture_evidence(self) -> dict[str, Any]:
        """Return latest application identity and explicit assembly counters.

        Returns:
            Host-only evidence. Renderer GPU source timestamps and internal tile
            assembly operations remain unavailable, rather than inferred zero.
        """
        if not hasattr(self, "_last_capture_identity"):
            return {"status": "missing", "reason": "no_successful_render_submission"}
        return {
            "status": "recorded",
            "render_submission_id": self._last_capture_submission_id,
            "counter_provenance": "observed Isaac render_generation application render-request counter; not a native GPU exposure counter",
            "logical_state": copy.deepcopy(self._last_capture_identity),
            "renderer_source_frame_id": None,
            "renderer_source_timestamp_s": None,
            "renderer_source_missing_reason": "Pinned RTX camera API does not expose per-render-product GPU source frames/timestamps",
            "assembly": {
                "status": "recorded",
                "operations": copy.deepcopy(self._last_assembly_operations),
                "scope": "executed adapter buffer packing/materialization; color/geometry conversions and renderer-internal untile excluded",
                "renderer_internal_operations": None,
                "renderer_internal_missing_reason": "Isaac/Warp render-product untile operation and transfer totals are not exposed",
            },
        }

    def synchronize(self) -> None:
        """Complete renderer/Warp interop and all work on the selected CUDA device."""
        self.wp.synchronize_device(self.device)
        self.torch.cuda.synchronize(self.device)
        if hasattr(self, "_physics_baseline"):
            completed = self._physics_snapshot()
            self.metadata["no_step_evidence"]["completed"] = completed
            if completed != self._physics_baseline:
                raise RuntimeError(
                    "Physics advanced before the device-ready completion boundary"
                )

    def to_host(self, arrays: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        """Perform one real blocking D2H per modality, without another host copy.

        Args:
            arrays: Original canonical CUDA arrays from :meth:`render_device`.

        Returns:
            NumPy views of the newly allocated CPU tensors.
        """
        return {name: value.cpu().numpy() for name, value in arrays.items()}

    def refresh_metadata(self) -> dict[str, Any]:
        """Read actual camera poses outside timed capture loops.

        Returns:
            Updated renderer metadata with observed intrinsics and world poses.
        """
        # Frozen Isaac Lab Camera.set_world_poses updates USD but not CameraData.
        # Its pose-buffer helper performs no exposure; unlike data/update calls,
        # this explicitly refreshes diagnostics without rendering another frame.
        self.camera._update_poses()
        self.synchronize()
        data = self._camera_data
        self.metadata["last_capture_evidence"] = self.capture_evidence()
        self.metadata["intrinsics"] = data.intrinsic_matrices.torch.cpu().tolist()
        self.metadata["camera_positions"] = data.pos_w.torch.cpu().tolist()
        self.metadata["camera_quaternions_xyzw"] = (
            data.quat_w_opengl.torch.cpu().tolist()
        )
        settings = {name: self.sim.get_setting(name) for name in _RTX_OBSERVED_SETTINGS}
        log_path = self.sim.get_setting("/log/file")
        log_text, log_missing_reason = "", None
        if isinstance(log_path, str) and log_path:
            try:
                log_text = Path(log_path).read_text(encoding="utf-8", errors="replace")
            except OSError as error:
                log_missing_reason = str(error)
        else:
            log_missing_reason = "Current Kit log path is unavailable"
        first_output = next(iter(data.output.values()), None)
        observed_output = (
            list(reversed(first_output.torch.shape[1:3]))
            if first_output is not None
            else None
        )
        evidence = _resolution_evidence(
            self.width,
            self.height,
            (self.profile or {}).get("internal_resolution_policy"),
            observed_output,
            settings,
            log_text,
        )
        evidence["runtime_log_path"] = log_path
        evidence["runtime_log_missing_reason"] = log_missing_reason
        products = []
        for path in self.camera._render_data.render_product_paths:
            prim = self.stage.GetPrimAtPath(path)
            resolution = (
                prim.GetAttribute("resolution").Get() if prim.IsValid() else None
            )
            products.append(
                {
                    "path": str(path),
                    "output_atlas_resolution": (
                        list(resolution) if resolution is not None else None
                    ),
                }
            )
        evidence["observed_render_products"] = products
        self.metadata["resolution_evidence"] = evidence
        self.metadata["resolved_rtx_settings"] = settings
        self.metadata["observed_renderer_settings"] = {
            "mode": settings["/rtx/rendermode"],
            "aa_op": settings["/rtx/post/aa/op"],
            "dl_denoiser": settings["/rtx-transient/dldenoiser/enabled"],
            "pt_spp": settings["/rtx/pathtracing/spp"],
            "pt_total_spp_cap": settings["/rtx/pathtracing/totalSpp"],
            "rt_direct_light_spp": settings[
                "/rtx/directLighting/sampledLighting/samplesPerPixel"
            ],
        }
        postprocess = {
            name: self.sim.get_setting(name) for name in _RTX_POSTPROCESS_SETTINGS
        }
        self.metadata["postprocess_evidence"] = {
            "scope": "runtime_carb_readback",
            "settings": postprocess,
            "missing_settings": [
                name for name, value in postprocess.items() if value is None
            ],
            "requested_overrides": {
                name: self._post_camera_overrides[name]
                for name in _RTX_POSTPROCESS_SETTINGS
                if name in self._post_camera_overrides
            },
            "encoding_caution": "Installed tonemap op 0 Clamp bypasses exposure and sRGB conversion; op 1 Linear retains exposure adjustments. No operator/encoding equivalence inferred from defaults.",
        }
        self.metadata["light_evidence"] = _light_evidence(self.stage, self.sim)
        if evidence["runtime_resize_warnings"]:
            reason = "Runtime DLSS resized internal input; native_output_resolution contract is unsupported for this captured workload"
        else:
            reason = "Internal RTX shading resolution is not independently exposed; native output-size equality is not verified"
        limitations = self.metadata.setdefault("quality_limitations", [])
        if reason not in limitations:
            limitations.append(reason)
        return self.metadata

    def _publish_cameras(self, specs: list[dict[str, Any]]) -> None:
        import numpy as np

        if len(specs) != len(self.camera_specs):
            raise ValueError(
                "Camera replacement must preserve the render-product view count"
            )
        specs = _selected_cameras(self.scene, None, specs)
        camera_radius = max(float(np.linalg.norm(spec["eye"])) for spec in specs)
        if (
            self.num_envs > 1
            and camera_radius + self._scene_radius + self.scene["camera_model"]["far_m"]
            >= self._replica_spacing
        ):
            raise ValueError(
                "Camera/replay bounds violate primary-ray replica separation"
            )
        positions, orientations = [], []
        for offset in self._env_offsets:
            for spec in specs:
                eye, quaternion = camera_world_pose(spec)
                positions.append((np.asarray(eye) + offset).tolist())
                orientations.append(quaternion)
        stream = self.wp.stream_from_torch(self.torch.cuda.current_stream(self.device))
        with self.wp.ScopedStream(stream):
            self.camera.set_world_poses(
                positions=self.torch.tensor(
                    positions, dtype=self.torch.float32, device=self.device
                ),
                orientations=self.torch.tensor(
                    orientations, dtype=self.torch.float32, device=self.device
                ),
                convention="opengl",
            )
        self.camera_specs = specs
        self.metadata["camera_names"] = [spec["name"] for spec in specs]
        self.metadata["view_sha256"] = stable_hash(specs)
        self.sim.render_context.reset_transform_cadence()

    def set_cameras(self, specs: list[dict[str, Any]]) -> None:
        """Replace a fixed-count view set and refresh diagnostics outside timing.

        Args:
            specs: New look-at views using the approved scene camera model.
        """
        self._publish_cameras(specs)
        self.refresh_metadata()

    def apply_state(self, frame: dict[str, Any]) -> None:
        """Publish kinematic object/robot-link and selected camera poses to USD.

        Args:
            frame: Frozen replay frame with ``objects`` keyed by canonical UID,
                ``cameras`` keyed by selected camera name, and frame/time IDs.
                Robot-link visual transforms are already resolved by the replay.
        """
        import numpy as np

        objects = frame.get("objects", {})
        if set(objects) - set(self._objects):
            raise ValueError("Replay contains unknown scene object UIDs")
        cameras = frame.get("cameras", {})
        missing = [
            spec["name"] for spec in self.camera_specs if spec["name"] not in cameras
        ]
        if missing:
            raise ValueError(f"Replay omits selected camera views: {missing}")
        matrices = {
            uid: np.asarray(pose, dtype=np.float64) for uid, pose in objects.items()
        }
        for uid, pose in matrices.items():
            _object_matrix(self._objects[uid], pose, np.zeros(3))
        with self.Sdf.ChangeBlock():
            for uid, pose in matrices.items():
                if np.array_equal(pose, self._object_poses[uid]):
                    continue
                for operation, offset in zip(
                    self._object_ops[uid], self._env_offsets, strict=True
                ):
                    matrix = _object_matrix(self._objects[uid], pose, offset)
                    operation.Set(self.Gf.Matrix4d(*matrix.T.reshape(-1).tolist()))
                self._object_poses[uid] = pose.copy()
        self._scene_radius = max(
            float(np.linalg.norm(pose[:3, 3])) + self._geometry_radii[uid]
            for uid, pose in self._object_poses.items()
        )
        self._publish_cameras([cameras[spec["name"]] for spec in self.camera_specs])
        self.metadata["frame_id"] = frame.get("frame_id")
        self.metadata["time_s"] = frame.get("time_s")

    def capture(self) -> dict[str, Any]:
        """Return preview-compatible completed host arrays and pose diagnostics."""
        arrays = self.render_device()
        self.synchronize()
        host = self.to_host(arrays)
        self.refresh_metadata()
        return host

    def close(self) -> None:
        """Close the isolated Isaac Sim application."""
        self.camera = None
        self.app.close()
        self._prototype_stage = None
