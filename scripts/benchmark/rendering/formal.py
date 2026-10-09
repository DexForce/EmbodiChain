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

"""Version-two, document-aligned rendering cases and immutable render profiles.

These are resolved value records, not simulator configuration objects. Keeping
them standard-library-only makes matrix inspection and reporting GPU independent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import itertools
from pathlib import Path
from typing import Any

from scripts.benchmark.core.artifacts import read_json_object, stable_hash
from scripts.benchmark.core.contracts import MetricDefinition

__all__ = [
    "DEFINITION_VERSION",
    "EXPERIMENTS",
    "RenderCase",
    "experiment_cases",
    "metric_definitions",
    "render_profile",
    "validate_approval",
    "view_fingerprint",
]

DEFINITION_VERSION = "2.0"
EXPERIMENTS = ("R-01", "R-02", "R-03", "R-04", "R-05", "R-09", "modality-ablation")

# Freeze the settings actually observed in the pinned RTX runtime. This records
# its ACES response; it does not assert photometric equality with native RGB.
_ISAAC_POSTPROCESS = {
    "/rtx/post/histogram/enabled": False,
    "/rtx/post/tonemap/op": 6,
    "/rtx/post/tonemap/filmIso": 100.0,
    "/rtx/post/tonemap/exposureTime": 0.02,
    "/rtx/post/tonemap/fNumber": 5.0,
    "/rtx/post/tonemap/responsivity": 1.1026709079742432,
    "/rtx/post/tonemap/whitepoint": [1.0, 1.0, 1.0],
    "/rtx/post/tonemap/colorMode": 0,
    "/rtx/post/tonemap/enableSrgbToGamma": True,
    "/rtx/post/tonemap/wrapValue": 0.0,
    "/rtx/post/tonemap/dither": 0.0,
    "/rtx/post/tonemap/ocio/enabled": False,
    "/rtx/post/colorcorr/enabled": False,
    "/rtx/post/colorgrad/enabled": False,
}

_PROFILES = {
    "realtime": {
        "version": "1.5",
        "name": "realtime",
        "frame_time_s": 1 / 30,
        "embodichain": {
            "renderer": "hybrid",
            "spp": 1,
            "denoising": "dlss",
            "dlss_quality": 5,
            "dlss_tiled": True,
            "rgb_output_encoding": "display_srgb_uint8; normalize native Hybrid linear-UNORM8 on device",
            "tone_mapping": False,
            "tone_mapping_exposure": 1.0,
            "auto_exposure": False,
            "camera_group_capacity": 32,
        },
        "isaaclab": {
            "rendering_mode": "balanced",
            "antialiasing_mode": "DLAA",
            "samples_per_pixel": 1,
            "enable_dl_denoiser": True,
            "enable_reflections": True,
            "enable_global_illumination": True,
            "carb_settings": dict(_ISAAC_POSTPROCESS),
        },
        "frame_generation": False,
        "lighting_controls": {
            "embodichain": {
                "directional_intensity": 12.0,
                "environment_intensity": 15.0,
                "environment_color": [1.0, 1.0, 1.0],
            },
            "isaaclab": {
                "directional_intensity": 3000.0,
                "environment_intensity": 150.0,
                "environment_color": [1.0, 1.0, 1.0],
            },
        },
        "lighting_control_basis": "native environment percent maps 15 to 0.15 and directional power/4 maps 12 to irradiance 3; source-based starting units, not a calibration certificate",
        "internal_resolution_policy": "output_frozen_backend_internal_reported",
        "photometric_calibration": {"status": "not_qualified"},
    },
    "high_quality": {
        "version": "1.5",
        "name": "high_quality",
        "frame_time_s": 1 / 30,
        "embodichain": {
            "renderer": "fast-rt",
            "spp": 16,
            "denoising": "dlss",
            "dlss_quality": 5,
            "dlss_tiled": True,
            "rgb_output_encoding": "display_srgb_uint8; native FastRT output",
            "tone_mapping": False,
            "tone_mapping_exposure": 1.0,
            "auto_exposure": False,
            "camera_group_capacity": 32,
        },
        "isaaclab": {
            "rendering_mode": "quality",
            "antialiasing_mode": None,
            "samples_per_pixel": 16,
            "enable_dl_denoiser": False,
            "enable_reflections": True,
            "enable_global_illumination": True,
            "carb_settings": {
                **_ISAAC_POSTPROCESS,
                "/rtx/rendermode": "PathTracing",
                "/rtx/pathtracing/spp": 16,
                "/rtx/pathtracing/totalSpp": 0,
                "/rtx/pathtracing/adaptiveSampling/enabled": False,
                "/rtx/pathtracing/ecoMode/enabled": False,
                "/rtx/pathtracing/optixDenoiser/enabled": True,
            },
        },
        "accumulation_policy": "reset_each_exposure",
        "frame_generation": False,
        "lighting_controls": {
            "embodichain": {
                "directional_intensity": 12.0,
                "environment_intensity": 15.0,
                "environment_color": [1.0, 1.0, 1.0],
            },
            "isaaclab": {
                "directional_intensity": 3000.0,
                "environment_intensity": 150.0,
                "environment_color": [1.0, 1.0, 1.0],
            },
        },
        "lighting_control_basis": "native environment percent maps 15 to 0.15 and directional power/4 maps 12 to irradiance 3; source-based starting units, not a calibration certificate",
        "internal_resolution_policy": "output_frozen_backend_internal_reported",
        "photometric_calibration": {"status": "not_qualified"},
    },
}


def render_profile(name: str) -> dict[str, Any]:
    """Return an independent copy of a frozen cross-platform render profile.

    Args:
        name: ``realtime`` or ``high_quality``.

    Returns:
        Explicit per-renderer settings and their qualification boundary.
    """
    import copy

    if name not in _PROFILES:
        raise ValueError(f"Unknown rendering profile: {name}")
    return copy.deepcopy(_PROFILES[name])


@dataclass(frozen=True)
class RenderCase:
    """Fully resolved, portable v2 workload semantics for one matrix cell."""

    experiment_id: str
    scene_id: str
    width: int = 128
    height: int = 128
    num_envs: int = 1
    cameras_per_env: int = 1
    modalities: tuple[str, ...] = ("rgb",)
    profile: str = "realtime"
    temporal_mode: str = "static"
    delivery: str = "device_ready"
    view_count: int = 1
    sequence_frames: int = 100
    diagnostic: bool = False
    deformation: bool = False
    definition_version: str = DEFINITION_VERSION

    def __post_init__(self) -> None:
        if (
            self.experiment_id not in EXPERIMENTS
            or self.definition_version != DEFINITION_VERSION
        ):
            raise ValueError("Unsupported formal experiment or definition version")
        if self.scene_id not in {"calibration", "workshop", "franka_workstation"}:
            raise ValueError("Unknown approved scene")
        for key in (
            "width",
            "height",
            "num_envs",
            "cameras_per_env",
            "view_count",
            "sequence_frames",
        ):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.cameras_per_env not in (1, 3):
            raise ValueError("Formal layouts have one or three cameras")
        if self.cameras_per_env == 3 and self.scene_id == "calibration":
            raise ValueError("Calibration scene has no wrist camera")
        if self.profile not in _PROFILES:
            raise ValueError("Unknown profile")
        if self.temporal_mode not in {"static", "moving"}:
            raise ValueError("Unknown temporal mode")
        if self.delivery not in {
            "device_ready",
            "host_ready",
            "duplicate_d2h",
            "host_copy",
        }:
            raise ValueError("Unknown delivery boundary")
        if (
            not self.modalities
            or len(set(self.modalities)) != len(self.modalities)
            or set(self.modalities) - {"rgb", "depth", "seg", "normals"}
        ):
            raise ValueError("Unsupported or duplicate modality")

    @property
    def case_id(self) -> str:
        """Return a stable identity that cannot mix legacy R-05 meanings."""
        return f"{self.experiment_id.lower()}-{self.scene_id}-n{self.num_envs}c{self.cameras_per_env}-{stable_hash(self.to_dict())[:12]}"

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible workload values without host-dependent paths."""
        result = asdict(self)
        result["modalities"] = list(self.modalities)
        return result


def experiment_cases(experiment_id: str) -> tuple[RenderCase, ...]:
    """Expand the approved report plan's formal matrix, independently of hardware.

    Args:
        experiment_id: One of :data:`EXPERIMENTS`.

    Returns:
        All planned cases, including cases that may exceed runtime capabilities.
    """
    cases = []
    if experiment_id == "R-01":
        for scene, profile, n in itertools.product(
            ("calibration", "workshop"), _PROFILES, (1, 64)
        ):
            cases.append(
                RenderCase(
                    experiment_id,
                    scene,
                    width=512,
                    height=512,
                    num_envs=n,
                    modalities=("rgb", "depth", "seg"),
                    profile=profile,
                    delivery="host_ready",
                    view_count=20,
                )
            )
    elif experiment_id == "R-02":
        for profile in _PROFILES:
            cases.append(
                RenderCase(
                    experiment_id,
                    "franka_workstation",
                    width=512,
                    height=512,
                    cameras_per_env=3,
                    modalities=("rgb", "depth", "seg"),
                    profile=profile,
                    temporal_mode="moving",
                    delivery="host_ready",
                )
            )
            cases.append(
                RenderCase(
                    experiment_id,
                    "franka_workstation",
                    width=512,
                    height=512,
                    modalities=("rgb", "depth", "seg"),
                    profile=profile,
                    temporal_mode="moving",
                    delivery="host_ready",
                    deformation=True,
                )
            )
    elif experiment_id == "R-03":
        for scene, temporal, resolution, modalities, profile in itertools.product(
            ("calibration", "franka_workstation"),
            ("static", "moving"),
            ((256, 256), (512, 512), (1920, 1080)),
            (("rgb",), ("rgb", "depth", "seg")),
            _PROFILES,
        ):
            cases.append(
                RenderCase(
                    experiment_id,
                    scene,
                    width=resolution[0],
                    height=resolution[1],
                    modalities=modalities,
                    profile=profile,
                    temporal_mode=temporal,
                )
            )
    elif experiment_id in {"R-04", "R-05"}:
        for n in (1, 16, 64, 256, 1024):
            cases.append(
                RenderCase(
                    experiment_id,
                    "franka_workstation",
                    num_envs=n,
                    cameras_per_env=1 if experiment_id == "R-04" else 3,
                )
            )
    elif experiment_id == "R-09":
        for delivery in ("device_ready", "host_ready", "duplicate_d2h", "host_copy"):
            for diagnostic in (False, True):
                cases.append(
                    RenderCase(
                        experiment_id,
                        "franka_workstation",
                        num_envs=16,
                        cameras_per_env=3,
                        delivery=delivery,
                        diagnostic=diagnostic,
                    )
                )
    elif experiment_id == "modality-ablation":
        for modalities in (
            ("rgb",),
            ("rgb", "depth"),
            ("rgb", "normals"),
            ("rgb", "depth", "seg"),
        ):
            cases.append(
                RenderCase(
                    experiment_id,
                    "franka_workstation",
                    width=256,
                    height=256,
                    modalities=modalities,
                )
            )
    else:
        raise ValueError(f"Unknown formal experiment: {experiment_id}")
    return tuple(cases)


def view_fingerprint(
    case: RenderCase,
    scene: dict[str, Any],
    replay: dict[str, Any] | None = None,
) -> str:
    """Bind portable camera semantics and optional frozen temporal motion.

    Args:
        case: Resolved formal case, without host paths or backend offsets.
        scene: Original approved scene, before selecting its local cameras.
        replay: Complete frozen replay for a moving case.

    Returns:
        SHA-256 of the pinhole model, resolution, selected views and motion.

    Raises:
        ValueError: The declared cameras, fixed views or moving replay are absent.
    """
    from .scene_cases import fixed_views

    names = (
        ("external_front",)
        if case.cameras_per_env == 1
        else ("external_front", "external_side", "wrist")
    )
    cameras = {view["name"]: view for view in scene["cameras"]}
    if set(names) - cameras.keys():
        raise ValueError("Approved scene does not contain the declared views")
    views = (
        fixed_views(scene)
        if case.experiment_id == "R-01"
        else [cameras[name] for name in names]
    )
    expected = case.view_count if case.experiment_id == "R-01" else case.cameras_per_env
    if len(views) != expected:
        raise ValueError("Approved scene does not contain the declared views")
    if case.temporal_mode == "moving" and replay is None:
        raise ValueError("Moving view fingerprints require the frozen replay")
    return stable_hash(
        {
            "camera_model": scene["camera_model"],
            "width": case.width,
            "height": case.height,
            "cameras_per_env": case.cameras_per_env,
            "view_count": case.view_count,
            "temporal_mode": case.temporal_mode,
            "views": views,
            "replay_sha256": (
                stable_hash(replay) if case.temporal_mode == "moving" else None
            ),
        }
    )


def validate_approval(
    scene_root: Path, approval_path: Path | None = None
) -> dict[str, Any]:
    """Bind the user's scene approval to the exact frozen source manifests.

    Args:
        scene_root: Directory containing reviewed scene folders.
        approval_path: Approval receipt, defaulting to ``approval.json``.

    Returns:
        Verified scene approval; it does not certify image quality.
    """
    from .scenes import load_scene, scene_fingerprint

    approval = read_json_object(approval_path or scene_root / "approval.json")
    if (
        approval.get("status") != "approved"
        or approval.get("scope") != "scene_layout_and_camera_scheme"
    ):
        raise ValueError("Explicit scene-layout approval is required")
    for name in ("calibration", "workshop", "franka_workstation"):
        actual = scene_fingerprint(load_scene(scene_root / name / "scene.json"))
        if approval.get("scenes", {}).get(name) != actual:
            raise ValueError(f"Approved scene was changed: {name}")
    return approval


def metric_definitions() -> tuple[MetricDefinition, ...]:
    """Return definitions for completed-capture rates, latency and resource scope."""
    units = {
        "batch_steps_per_s": ("1/s", "batch"),
        "camera_frames_per_s": ("1/s", "camera_exposure"),
        "gpixel_per_s": ("Gpixel/s", "camera_exposure"),
        "latency_p50_s": ("s", "batch"),
        "latency_p95_s": ("s", "batch"),
        "state_publish_p50_s": ("s", "state_publication"),
        "device_ready_p50_s": ("s", "batch"),
        "host_ready_p50_s": ("s", "batch"),
        "native_render_s": ("s", "native_gpu_render"),
        "gpu_process_peak_bytes": ("byte", "sampled_process_gpu_memory"),
        "gpu_process_steady_bytes": ("byte", "sampled_process_gpu_memory"),
        "gpu_device_peak_bytes": ("byte", "sampled_device_memory"),
        "cpu_peak_rss_bytes": ("byte", "process_rss"),
        "torch_peak_allocated_bytes": ("byte", "torch_allocator"),
        "d2h_bytes_per_capture": ("byte", "batch"),
        "host_copy_bytes_per_capture": ("byte", "batch"),
    }
    return tuple(
        MetricDefinition(name, unit, population, DEFINITION_VERSION)
        for name, (unit, population) in units.items()
    )
