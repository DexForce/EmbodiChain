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

"""Benchmark configurable offscreen rendering and denoising throughput.

The current benchmark uses one fixed reflective scene while allowing the
camera count, output resolution, parallel environment count, renderer, sample
count, and denoising modes to be configured.

Run: python -m scripts.benchmark.rendering.offscreen_denoising
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Literal, Mapping

import cv2
import numpy as np
import psutil

RenderMode = Literal["off", "optix", "dlss", "nrd"]
BENCHMARK_MODES: tuple[RenderMode, ...] = ("optix", "dlss", "nrd")
RENDER_MODES: tuple[RenderMode, ...] = ("off", "optix", "dlss", "nrd")
VALID_RENDER_MODES = frozenset(RENDER_MODES)
FIXED_SCENE_NAME = "fixed_reflective"
_SSIM_WINDOW_SIZE = 11
_SSIM_WINDOW_SIGMA = 1.5

__all__ = [
    "BENCHMARK_MODES",
    "BenchmarkCfg",
    "build_leaderboard_rows",
    "build_parser",
    "compute_quality_metrics",
    "main",
    "run_all_benchmarks",
    "run_environment_sweep",
    "write_markdown_report",
]


@dataclass(frozen=True)
class BenchmarkCfg:
    """Settings that define one rendering benchmark run.

    The benchmark scene is intentionally fixed for now.  Keeping its name in
    the setting object gives the runner a stable extension point for adding
    scene builders without changing the report or worker protocol.
    """

    width: int = 640
    height: int = 360
    warmup_frames: int = 12
    measure_frames: int = 40
    quality_frames: int = 12
    dlss_quality: int = 3
    gpu_id: int = 0
    device: str = "cuda:0"
    camera_count: int = 1
    num_envs: int = 1
    denoising_modes: tuple[RenderMode, ...] = BENCHMARK_MODES
    reference_mode: RenderMode | None = "optix"
    renderer: Literal["auto", "hybrid", "fast-rt", "rt"] = "fast-rt"
    spp: int = 1
    scene: str = FIXED_SCENE_NAME

    def __post_init__(self) -> None:
        """Validate and normalize settings before a worker is started."""
        integer_fields = (
            "width",
            "height",
            "camera_count",
            "num_envs",
            "warmup_frames",
            "measure_frames",
            "quality_frames",
            "spp",
            "gpu_id",
            "dlss_quality",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"BenchmarkCfg.{name} must be an integer.")
        for name in ("width", "height", "camera_count", "num_envs", "spp"):
            if getattr(self, name) <= 0:
                raise ValueError(f"BenchmarkCfg.{name} must be positive.")
        if self.warmup_frames < 0:
            raise ValueError("BenchmarkCfg.warmup_frames must be non-negative.")
        if self.measure_frames <= 0:
            raise ValueError("BenchmarkCfg.measure_frames must be positive.")
        if not 1 <= self.quality_frames <= self.measure_frames:
            raise ValueError(
                "BenchmarkCfg.quality_frames must be within measure_frames."
            )
        if not 0 <= self.gpu_id:
            raise ValueError("BenchmarkCfg.gpu_id must be non-negative.")
        if not -1 <= self.dlss_quality <= 5:
            raise ValueError("BenchmarkCfg.dlss_quality must be between -1 and 5.")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("BenchmarkCfg.device must be a non-empty string.")
        if self.renderer not in {"auto", "hybrid", "fast-rt", "rt"}:
            raise ValueError(
                "BenchmarkCfg.renderer must be 'auto', 'hybrid', 'fast-rt', or 'rt'."
            )
        if self.scene != FIXED_SCENE_NAME:
            raise ValueError(
                f"Unsupported rendering benchmark scene {self.scene!r}; "
                f"only {FIXED_SCENE_NAME!r} is currently available."
            )

        modes = self.denoising_modes
        if isinstance(modes, str):
            modes = (modes,)
        else:
            modes = tuple(modes)
        if not modes:
            raise ValueError("BenchmarkCfg.denoising_modes must not be empty.")
        invalid_modes = [mode for mode in modes if mode not in VALID_RENDER_MODES]
        if invalid_modes:
            raise ValueError(
                "BenchmarkCfg.denoising_modes contains unsupported modes: "
                f"{invalid_modes!r}."
            )
        if len(set(modes)) != len(modes):
            raise ValueError(
                "BenchmarkCfg.denoising_modes must not contain duplicates."
            )
        object.__setattr__(self, "denoising_modes", modes)
        if (
            self.reference_mode is not None
            and self.reference_mode not in VALID_RENDER_MODES
        ):
            raise ValueError(
                f"BenchmarkCfg.reference_mode {self.reference_mode!r} is unsupported."
            )

    @property
    def resolution(self) -> tuple[int, int]:
        """Return the configured output resolution as ``(width, height)``."""
        return self.width, self.height

    @property
    def render_modes(self) -> tuple[RenderMode, ...]:
        """Backward-compatible alias for the configured denoising modes."""
        return self.denoising_modes

    @property
    def num_cameras(self) -> int:
        """Return the configured camera count under its descriptive alias."""
        return self.camera_count

    @property
    def resolved_reference_mode(self) -> RenderMode:
        """Return the selected reference, falling back to the first mode."""
        if self.reference_mode in self.denoising_modes:
            return self.reference_mode  # type: ignore[return-value]
        return self.denoising_modes[0]

    def to_dict(self) -> dict[str, object]:
        """Serialize settings for reports and reproducible worker runs."""
        values = asdict(self)
        values["denoising_modes"] = list(self.denoising_modes)
        values["resolution"] = f"{self.width}x{self.height}"
        values["resolved_reference_mode"] = self.resolved_reference_mode
        return values

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "BenchmarkCfg":
        """Build settings from a JSON/YAML-compatible mapping.

        ``resolution``, ``modes`` and ``denoising`` are accepted as concise
        aliases so a small external config can be used from the CLI.
        """
        data = dict(values)
        if "resolution" in data:
            resolution = data.pop("resolution")
            if isinstance(resolution, str):
                parts = resolution.lower().split("x")
                if len(parts) != 2:
                    raise ValueError("resolution must use the WIDTHxHEIGHT form.")
                resolution = (int(parts[0]), int(parts[1]))
            if not isinstance(resolution, (tuple, list)) or len(resolution) != 2:
                raise ValueError("resolution must contain width and height.")
            data.setdefault("width", int(resolution[0]))
            data.setdefault("height", int(resolution[1]))
        for alias in ("modes", "render_modes", "denoising"):
            if alias in data:
                data.setdefault("denoising_modes", data.pop(alias))
        if "num_cameras" in data:
            data.setdefault("camera_count", data.pop("num_cameras"))
        if "environments" in data:
            data.setdefault("num_envs", data.pop("environments"))
        data.pop("resolved_reference_mode", None)
        allowed = set(asdict(cls()).keys())
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(f"Unknown rendering benchmark setting(s): {unknown!r}.")
        if "denoising_modes" in data and isinstance(data["denoising_modes"], list):
            data["denoising_modes"] = tuple(data["denoising_modes"])
        return cls(**data)

    @classmethod
    def from_json(cls, path: Path) -> "BenchmarkCfg":
        """Load settings from a JSON file."""
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Benchmark config {path} must contain a JSON object.")
        return cls.from_mapping(payload)

    @classmethod
    def from_file(cls, path: Path) -> "BenchmarkCfg":
        """Load settings from JSON, YAML, or YML based on the suffix."""
        if path.suffix.lower() in {".yaml", ".yml"}:
            import yaml

            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Benchmark config {path} must contain a mapping.")
        return cls.from_mapping(payload)

    def with_overrides(self, **overrides: object) -> "BenchmarkCfg":
        """Return a copy with explicitly supplied CLI values applied."""
        values = {key: value for key, value in overrides.items() if value is not None}
        if "denoising_modes" in values and isinstance(values["denoising_modes"], list):
            values["denoising_modes"] = tuple(values["denoising_modes"])
        return replace(self, **values)


def compute_quality_metrics(
    reference: np.ndarray, candidate: np.ndarray
) -> dict[str, float]:
    """Return image similarity metrics for one RGB frame pair.

    SSIM is computed from local Gaussian windows so small edge and texture
    defects remain visible even when most of the frame is unchanged.
    """
    reference_rgb = np.asarray(reference)[..., :3].astype(np.float64) / 255.0
    candidate_rgb = np.asarray(candidate)[..., :3].astype(np.float64) / 255.0
    if reference_rgb.shape != candidate_rgb.shape:
        raise ValueError(
            "Reference and candidate images must have identical RGB shapes."
        )

    delta = reference_rgb - candidate_rgb
    mse = float(np.mean(np.square(delta)))
    mae = float(np.mean(np.abs(delta)))
    psnr_db = math.inf if mse == 0.0 else float(10.0 * math.log10(1.0 / mse))

    channel_ssim: list[float] = []
    for channel in range(3):
        ref_channel = reference_rgb[..., channel]
        candidate_channel = candidate_rgb[..., channel]
        ref_mean = cv2.GaussianBlur(
            ref_channel,
            (_SSIM_WINDOW_SIZE, _SSIM_WINDOW_SIZE),
            _SSIM_WINDOW_SIGMA,
            borderType=cv2.BORDER_REFLECT,
        )
        candidate_mean = cv2.GaussianBlur(
            candidate_channel,
            (_SSIM_WINDOW_SIZE, _SSIM_WINDOW_SIZE),
            _SSIM_WINDOW_SIGMA,
            borderType=cv2.BORDER_REFLECT,
        )
        ref_variance = cv2.GaussianBlur(
            np.square(ref_channel),
            (_SSIM_WINDOW_SIZE, _SSIM_WINDOW_SIZE),
            _SSIM_WINDOW_SIGMA,
            borderType=cv2.BORDER_REFLECT,
        ) - np.square(ref_mean)
        candidate_variance = cv2.GaussianBlur(
            np.square(candidate_channel),
            (_SSIM_WINDOW_SIZE, _SSIM_WINDOW_SIZE),
            _SSIM_WINDOW_SIGMA,
            borderType=cv2.BORDER_REFLECT,
        ) - np.square(candidate_mean)
        covariance = (
            cv2.GaussianBlur(
                ref_channel * candidate_channel,
                (_SSIM_WINDOW_SIZE, _SSIM_WINDOW_SIZE),
                _SSIM_WINDOW_SIGMA,
                borderType=cv2.BORDER_REFLECT,
            )
            - ref_mean * candidate_mean
        )
        c1 = 0.01**2
        c2 = 0.03**2
        numerator = (2.0 * ref_mean * candidate_mean + c1) * (2.0 * covariance + c2)
        denominator = (np.square(ref_mean) + np.square(candidate_mean) + c1) * (
            ref_variance + candidate_variance + c2
        )
        channel_ssim.append(float(np.mean(numerator / denominator)))

    reference_gray = np.mean(reference_rgb, axis=-1)
    candidate_gray = np.mean(candidate_rgb, axis=-1)
    reference_gradient = np.hypot(*np.gradient(reference_gray))
    candidate_gradient = np.hypot(*np.gradient(candidate_gray))
    edge_mae = float(np.mean(np.abs(reference_gradient - candidate_gradient)))
    return {
        "psnr_db": psnr_db,
        "ssim": float(np.mean(channel_ssim)),
        "mae": mae,
        "edge_mae": edge_mae,
    }


def build_leaderboard_rows(
    metric_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Rank algorithms by valid captures, similarity, and throughput."""
    ordered = sorted(
        metric_rows,
        key=lambda row: (
            float(row["success_rate"]),
            float(row["ssim"]),
            float(row.get("fps", 0.0)),
        ),
        reverse=True,
    )
    return [
        {
            "rank": rank,
            "algorithm": row["mode"],
            "overall_success_rate": row["success_rate"],
            "ssim": row["ssim"],
            "fps": row.get("fps", 0.0),
        }
        for rank, row in enumerate(ordered, start=1)
    ]


def _markdown_table(rows: list[dict[str, object]]) -> list[str]:
    """Render one Markdown table from rows sharing the same keys."""
    if not rows:
        return ["No rows were produced."]
    headers = list(rows[0])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(row[header]) for header in headers) + " |" for row in rows
    )
    return lines


def write_markdown_report(
    *,
    output_path: Path,
    perf_rows: list[dict[str, object]],
    metric_rows: list[dict[str, object]],
    leaderboard_rows: list[dict[str, object]],
    settings: Mapping[str, object] | None = None,
    notes: list[str] | None = None,
) -> Path:
    """Write one benchmark report containing exactly three Markdown tables.

    Settings are rendered as a definition list so the report remains easy to
    diff and still satisfies the benchmark report's three-table contract.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Rendering Benchmark Report",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
    ]
    if settings:
        lines.extend(["", "## Benchmark Settings", ""])
        lines.extend(
            f"- **{key}**: {json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value}"
            for key, value in settings.items()
        )
    lines.extend(
        [
            "",
            "## Time & Memory",
            "",
            *_markdown_table(perf_rows),
            "",
            "## Success & Other Metrics",
            "",
            *_markdown_table(metric_rows),
            "",
            "## Leaderboard",
            "",
            *_markdown_table(leaderboard_rows),
        ]
    )
    if notes:
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {note}" for note in notes)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _process_gpu_memory_mb() -> float:
    """Return this process's GPU allocation reported by nvidia-smi."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return 0.0
    current_pid = os.getpid()
    memory_mb = 0.0
    for line in result.stdout.splitlines():
        columns = [part.strip() for part in line.split(",")]
        if len(columns) != 2:
            continue
        try:
            if int(columns[0]) == current_pid:
                memory_mb += float(columns[1])
        except ValueError:
            continue
    return memory_mb


def _gpu_description(gpu_id: int) -> str:
    """Return the selected GPU name, driver, and total memory when available."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
                "--id",
                str(gpu_id),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return f"GPU {gpu_id} (metadata unavailable)"
    return result.stdout.strip() or f"GPU {gpu_id} (metadata unavailable)"


def _memory_snapshot() -> dict[str, float]:
    """Return CPU RSS, process GPU VRAM, and PyTorch GPU allocation in MB."""
    import torch

    mib = 1024**2
    return {
        "cpu_mb": psutil.Process(os.getpid()).memory_info().rss / mib,
        "gpu_mb": _process_gpu_memory_mb(),
        "torch_gpu_mb": (
            torch.cuda.memory_allocated() / mib if torch.cuda.is_available() else 0.0
        ),
    }


def _material(
    uid: str,
    color: tuple[float, float, float],
    *,
    roughness: float,
    metallic: float = 0.0,
    emissive: tuple[float, float, float] = (0.0, 0.0, 0.0),
):
    """Create one PBR material without importing simulation modules globally."""
    from embodichain.lab.sim.material import VisualMaterialCfg

    return VisualMaterialCfg(
        uid=uid,
        base_color=[*color, 1.0],
        roughness=roughness,
        metallic=metallic,
        emissive=list(emissive),
    )


def _add_scene(sim):
    """Build the fixed scene stressing edges, roughness, and reflections."""
    from embodichain.lab.sim.cfg import LightCfg, RigidObjectCfg
    from embodichain.lab.sim.shapes import CubeCfg, SphereCfg

    def add_cube(
        uid: str,
        size: tuple[float, float, float],
        position: tuple[float, float, float],
        color: tuple[float, float, float],
        roughness: float,
        metallic: float = 0.0,
        *,
        body_type: str = "static",
        emissive: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        return sim.add_rigid_object(
            RigidObjectCfg(
                uid=uid,
                shape=CubeCfg(
                    size=list(size),
                    visual_material=_material(
                        f"{uid}_material",
                        color,
                        roughness=roughness,
                        metallic=metallic,
                        emissive=emissive,
                    ),
                ),
                body_type=body_type,
                init_pos=position,
            )
        )

    add_cube(
        "floor",
        (5.0, 5.0, 0.08),
        (0.0, 0.0, -0.04),
        (0.32, 0.34, 0.38),
        0.72,
    )
    add_cube(
        "back_wall",
        (5.0, 0.08, 2.8),
        (0.0, 1.65, 1.4),
        (0.62, 0.65, 0.70),
        0.82,
    )
    add_cube(
        "left_wall",
        (0.08, 3.4, 2.8),
        (-2.1, 0.0, 1.4),
        (0.25, 0.30, 0.42),
        0.78,
    )

    sphere_specs = (
        ("chrome", (-1.15, 0.30, 0.42), (0.72, 0.76, 0.82), 0.05, 1.0),
        ("brushed", (-0.35, 0.45, 0.36), (0.75, 0.28, 0.08), 0.28, 1.0),
        ("ceramic", (0.45, 0.40, 0.32), (0.15, 0.55, 0.92), 0.18, 0.0),
        ("matte", (1.20, 0.28, 0.28), (0.75, 0.72, 0.16), 0.85, 0.0),
    )
    for uid, position, color, roughness, metallic in sphere_specs:
        sim.add_rigid_object(
            RigidObjectCfg(
                uid=uid,
                shape=SphereCfg(
                    radius=position[2],
                    resolution=64,
                    visual_material=_material(
                        f"{uid}_material",
                        color,
                        roughness=roughness,
                        metallic=metallic,
                    ),
                ),
                body_type="static",
                init_pos=(position[0], position[1], position[2]),
            )
        )

    for index in range(11):
        x_position = -1.35 + index * 0.27
        add_cube(
            f"slat_{index}",
            (0.035, 0.16, 0.86 - index * 0.035),
            (x_position, 1.20, 0.43 - index * 0.0175),
            (0.92 if index % 2 == 0 else 0.08,) * 3,
            0.45,
        )

    add_cube(
        "emissive_panel",
        (0.85, 0.04, 0.28),
        (1.35, 1.52, 1.30),
        (0.08, 0.30, 0.95),
        0.3,
        emissive=(0.08, 0.30, 0.95),
    )
    moving_object = add_cube(
        "moving_cube",
        (0.34, 0.34, 0.34),
        (-1.0, -0.55, 0.22),
        (0.95, 0.12, 0.10),
        0.12,
        metallic=0.35,
        body_type="static",
    )

    sim.add_light(
        LightCfg(
            uid="key_light",
            light_type="rect",
            init_pos=(0.2, -0.7, 2.6),
            direction=(0.0, 0.35, -1.0),
            color=(1.0, 0.90, 0.78),
            intensity=180.0,
            rect_width=1.4,
            rect_height=1.0,
        )
    )
    sim.add_light(
        LightCfg(
            uid="fill_light",
            light_type="point",
            init_pos=(-1.5, -0.8, 1.5),
            color=(0.35, 0.55, 1.0),
            intensity=55.0,
            radius=5.0,
        )
    )
    return moving_object


def _build_benchmark_scene(sim, scene: str):
    """Build one configured benchmark scene.

    The dispatch table is deliberately small today, but makes adding a second
    scene a local change once scene switching becomes part of the benchmark
    contract.
    """
    if scene == FIXED_SCENE_NAME:
        return _add_scene(sim)
    raise ValueError(f"Unsupported rendering benchmark scene {scene!r}.")


def _camera_pose(camera_index: int, camera_count: int) -> tuple[float, float, float]:
    """Return a deterministic local eye position for one benchmark camera."""
    if camera_count == 1:
        return 3.0, -3.1, 1.85
    base_x, base_y = 3.0, -3.1
    spread = 0.30
    angle = (camera_index - (camera_count - 1) / 2.0) * spread
    cos_angle, sin_angle = math.cos(angle), math.sin(angle)
    return (
        base_x * cos_angle - base_y * sin_angle,
        base_x * sin_angle + base_y * cos_angle,
        1.85 + 0.04 * (camera_index - (camera_count - 1) / 2.0),
    )


def _build_simulation(mode: RenderMode, cfg: BenchmarkCfg):
    """Create one isolated simulation, scene, and configured camera groups."""
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import DenoisingCfg, DLSSCfg, NRDCfg, RenderCfg
    from embodichain.lab.sim.sensors import CameraCfg

    sim_cfg = SimulationManagerCfg(
        width=cfg.width,
        height=cfg.height,
        headless=True,
        device=cfg.device,
        gpu_id=cfg.gpu_id,
        num_envs=cfg.num_envs,
        enable_entity_gizmo=False,
        robot_ik_gizmo=None,
        startup_summary="off",
        dexsim_startup_info=False,
        render_cfg=RenderCfg(
            renderer=cfg.renderer,
            spp=cfg.spp,
            denoising=DenoisingCfg(window="off", offscreen=mode),
            dlss=DLSSCfg(
                dlss_quality=cfg.dlss_quality,
                tiled_enabled=cfg.camera_count > 1,
            ),
            nrd=NRDCfg(taa_tone_mapping_enabled=False),
        ),
    )
    sim = SimulationManager(sim_cfg)
    moving_object = _build_benchmark_scene(sim, cfg.scene)
    sim.prepare()
    cameras = []
    for camera_index in range(cfg.camera_count):
        cameras.append(
            sim.add_sensor(
                CameraCfg(
                    uid=f"benchmark_camera_{camera_index}",
                    width=cfg.width,
                    height=cfg.height,
                    intrinsics=(
                        cfg.width * 0.78,
                        cfg.width * 0.78,
                        cfg.width / 2.0,
                        cfg.height / 2.0,
                    ),
                    near=0.05,
                    far=20.0,
                    enable_color=True,
                    extrinsics=CameraCfg.ExtrinsicsCfg(
                        eye=_camera_pose(camera_index, cfg.camera_count),
                        target=(0.0, 0.35, 0.55),
                        up=(0.0, 0.0, 1.0),
                    ),
                )
            )
        )
    return sim, cameras, moving_object


def _set_frame_state(sim, cameras, moving_object, frame_index: int) -> None:
    """Apply the deterministic scene, camera, and moving-object state."""
    import torch

    phase = frame_index * 0.085
    object_pose = torch.tensor(
        [
            [
                -0.95 + 1.9 * (0.5 + 0.5 * math.sin(phase)),
                -0.62 + 0.12 * math.cos(phase * 0.7),
                0.22,
                0.0,
                0.0,
                math.sin(phase * 0.5),
                math.cos(phase * 0.5),
            ]
        ]
        * sim.num_envs,
        dtype=torch.float32,
        device=sim.device,
    )
    moving_object.set_local_pose(object_pose)

    target = torch.tensor([[0.0, 0.35, 0.55]] * sim.num_envs, dtype=torch.float32)
    for camera_index, camera in enumerate(cameras):
        base_eye = _camera_pose(camera_index, len(cameras))
        eye = torch.tensor(
            [
                [
                    base_eye[0] + 0.16 * math.sin(phase * 0.5),
                    base_eye[1] + 0.12 * math.cos(phase * 0.5),
                    base_eye[2] + 0.05 * math.sin(phase * 0.35),
                ]
            ]
            * sim.num_envs,
            dtype=torch.float32,
        )
        camera.look_at(eye, target)
    sim.sync_render_state()


def _capture_frame(cameras) -> np.ndarray:
    """Render all camera groups and return ``(env, camera, H, W, RGBA)``."""
    import torch

    if not isinstance(cameras, (list, tuple)):
        cameras = [cameras]
    for camera in cameras:
        camera.update()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    camera_frames = [
        camera.get_data()["color"].detach().cpu().numpy().copy() for camera in cameras
    ]
    return np.stack(camera_frames, axis=1)


def _run_mode_worker(mode: RenderMode, cfg: BenchmarkCfg, run_dir: Path) -> None:
    """Render one mode in an isolated process and persist raw measurements."""
    import torch

    sim = None
    cameras = None
    try:
        sim, cameras, moving_object = _build_simulation(mode, cfg)
        _set_frame_state(sim, cameras, moving_object, 0)
        for _ in range(cfg.warmup_frames):
            _capture_frame(cameras)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        before = _memory_snapshot()
        peak_gpu_mb = before["gpu_mb"]
        elapsed_ms: list[float] = []
        frames: list[np.ndarray] = []
        for frame_index in range(cfg.measure_frames):
            _set_frame_state(sim, cameras, moving_object, frame_index + 1)
            start = time.perf_counter()
            frame = _capture_frame(cameras)
            elapsed_ms.append((time.perf_counter() - start) * 1000.0)
            if frame_index < cfg.quality_frames:
                frames.append(frame)
            peak_gpu_mb = max(peak_gpu_mb, _process_gpu_memory_mb())
        after = _memory_snapshot()
        torch_peak_mb = (
            torch.cuda.max_memory_allocated() / 1024**2
            if torch.cuda.is_available()
            else 0.0
        )
        np.savez_compressed(run_dir / f"{mode}_frames.npz", frames=np.stack(frames))
        metrics = {
            "mode": mode,
            "captured_frames": len(elapsed_ms),
            "expected_frames": cfg.measure_frames,
            "camera_count": cfg.camera_count,
            "num_envs": cfg.num_envs,
            "rendered_images": len(elapsed_ms) * cfg.camera_count * cfg.num_envs,
            "mean_ms": float(np.mean(elapsed_ms)),
            "median_ms": float(np.median(elapsed_ms)),
            "p95_ms": float(np.percentile(elapsed_ms, 95)),
            "fps": float(1000.0 / np.median(elapsed_ms)),
            "cpu_delta_mb": after["cpu_mb"] - before["cpu_mb"],
            "gpu_delta_mb": after["gpu_mb"] - before["gpu_mb"],
            "peak_gpu_mb": peak_gpu_mb,
            "torch_peak_gpu_mb": torch_peak_mb,
        }
        (run_dir / f"{mode}_performance.json").write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"{mode:>6}: median={metrics['median_ms']:.2f} ms  "
            f"p95={metrics['p95_ms']:.2f} ms  fps={metrics['fps']:.2f}"
        )
    finally:
        if sim is not None:
            sim.destroy(exit_process=False)
            del cameras
            del sim


def _temporal_delta_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Return consecutive-frame RGB change error with image-sized scratch space.

    Each environment/camera is processed separately. Previous-image buffers
    are reused for differences; float64 reduction preserves weighting across
    all pixels without keeping full float32 sequences or their differences.
    """
    if len(reference) < 2:
        return 0.0
    error_sum = 0.0
    element_count = 0
    for index in np.ndindex(reference.shape[1:-3]):
        previous_reference = reference[(0, *index)][..., :3].astype(np.float32)
        previous_candidate = candidate[(0, *index)][..., :3].astype(np.float32)
        previous_reference /= 255.0
        previous_candidate /= 255.0
        for frame_index in range(1, len(reference)):
            current_reference = reference[(frame_index, *index)][..., :3].astype(
                np.float32
            )
            current_candidate = candidate[(frame_index, *index)][..., :3].astype(
                np.float32
            )
            current_reference /= 255.0
            current_candidate /= 255.0
            np.subtract(current_reference, previous_reference, out=previous_reference)
            np.subtract(current_candidate, previous_candidate, out=previous_candidate)
            np.subtract(previous_reference, previous_candidate, out=previous_reference)
            np.abs(previous_reference, out=previous_reference)
            error_sum += float(np.sum(previous_reference, dtype=np.float64))
            element_count += previous_reference.size
            previous_reference = current_reference
            previous_candidate = current_candidate
        del current_reference, current_candidate
    return error_sum / element_count


def _load_frames(path: Path) -> np.ndarray:
    """Load a quality sequence and close its archive immediately."""
    with np.load(path, allow_pickle=False) as archive:
        return archive["frames"]


def _load_comparison_views(path: Path, frame_index: int) -> np.ndarray:
    """Copy selected first-environment views without retaining their sequence."""
    frame = _load_frames(path)[frame_index]
    if frame.ndim == 3:
        views = frame[None, ...]
    elif frame.ndim == 4:
        views = frame[0][None, ...]
    else:
        views = frame[0]
    return views.copy()


def _aggregate_quality(
    mode: RenderMode,
    reference: np.ndarray,
    candidate: np.ndarray,
    fps: float,
    *,
    expected_frames: int | None = None,
) -> dict[str, object]:
    """Aggregate quality metrics over frames, environments, and cameras."""
    if reference.ndim == 4:
        reference = reference[:, None, None, ...]
    elif reference.ndim == 5:
        reference = reference[:, :, None, ...]
    if candidate.ndim == 4:
        candidate = candidate[:, None, None, ...]
    elif candidate.ndim == 5:
        candidate = candidate[:, :, None, ...]
    if reference.ndim != 6 or candidate.ndim != 6:
        raise ValueError(
            "Rendering benchmark frames must have shape "
            "(frames, envs, cameras, height, width, channels)."
        )
    reference_frame_count = reference.shape[0]
    frame_count = min(reference_frame_count, candidate.shape[0])
    env_count = min(reference.shape[1], candidate.shape[1])
    camera_count = min(reference.shape[2], candidate.shape[2])
    if frame_count == 0 or env_count == 0 or camera_count == 0:
        raise ValueError("Rendering benchmark quality requires at least one frame.")
    reference = reference[:frame_count, :env_count, :camera_count]
    candidate = candidate[:frame_count, :env_count, :camera_count]
    frame_metrics = [
        compute_quality_metrics(
            reference[frame, env, camera], candidate[frame, env, camera]
        )
        for frame in range(frame_count)
        for env in range(env_count)
        for camera in range(camera_count)
    ]
    psnr_values = [float(row["psnr_db"]) for row in frame_metrics]
    finite_psnr = [value for value in psnr_values if math.isfinite(value)]
    psnr_mean = math.inf if not finite_psnr else float(np.mean(finite_psnr))
    return {
        "mode": mode,
        "success_rate": 100.0
        * (frame_count / (expected_frames or reference_frame_count)),
        "captured_frames": frame_count,
        "camera_count": camera_count,
        "num_envs": env_count,
        "rendered_images": frame_count * camera_count * env_count,
        "psnr_db": round(psnr_mean, 4) if math.isfinite(psnr_mean) else "inf",
        "ssim": round(float(np.mean([row["ssim"] for row in frame_metrics])), 6),
        "mae": round(float(np.mean([row["mae"] for row in frame_metrics])), 6),
        "edge_mae": round(
            float(np.mean([row["edge_mae"] for row in frame_metrics])), 6
        ),
        "temporal_delta_mae": round(_temporal_delta_error(reference, candidate), 6),
        "fps": round(fps, 3),
    }


def _modes_from_run(run_dir: Path) -> tuple[RenderMode, ...]:
    """Discover mode artifacts written by the parent worker coordinator."""
    discovered = {
        path.name.removesuffix("_frames.npz") for path in run_dir.glob("*_frames.npz")
    }
    modes = tuple(mode for mode in RENDER_MODES if mode in discovered)
    if not modes:
        raise FileNotFoundError(f"No rendering frame artifacts found in {run_dir}.")
    return modes  # type: ignore[return-value]


def _write_comparison_image(run_dir: Path, frame_index: int = 0) -> Path:
    """Write mode comparisons for the first environment and every camera."""
    camera_frames = {
        mode: _load_comparison_views(run_dir / f"{mode}_frames.npz", frame_index)
        for mode in _modes_from_run(run_dir)
    }
    reference_mode = "optix" if "optix" in camera_frames else next(iter(camera_frames))
    reference = camera_frames[reference_mode]
    camera_rows: list[np.ndarray] = []
    for camera_index in range(reference.shape[0]):
        top_row: list[np.ndarray] = []
        bottom_row: list[np.ndarray] = []
        for mode, mode_frames in camera_frames.items():
            rgb = mode_frames[camera_index][..., :3]
            image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(
                image,
                f"camera {camera_index} · {mode}",
                (14, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            top_row.append(image)
            difference = np.clip(
                np.abs(
                    rgb.astype(np.int16)
                    - reference[camera_index][..., :3].astype(np.int16)
                )
                * 4,
                0,
                255,
            ).astype(np.uint8)
            difference = cv2.cvtColor(difference, cv2.COLOR_RGB2BGR)
            cv2.putText(
                difference,
                "|delta| x4",
                (14, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            bottom_row.append(difference)
        camera_rows.append(
            np.concatenate(
                (np.concatenate(top_row, axis=1), np.concatenate(bottom_row, axis=1)),
                axis=0,
            )
        )
    comparison = np.concatenate(camera_rows, axis=0)
    path = run_dir / "comparison.png"
    cv2.imwrite(str(path), comparison)
    return path


def _worker_command(mode: RenderMode, cfg: BenchmarkCfg, run_dir: Path) -> list[str]:
    """Build the isolated worker command for one render mode."""
    return [
        sys.executable,
        "-m",
        "scripts.benchmark.rendering.offscreen_denoising",
        "--worker-mode",
        mode,
        "--run-dir",
        str(run_dir),
        "--width",
        str(cfg.width),
        "--height",
        str(cfg.height),
        "--camera-count",
        str(cfg.camera_count),
        "--num-envs",
        str(cfg.num_envs),
        "--renderer",
        cfg.renderer,
        "--spp",
        str(cfg.spp),
        "--scene",
        cfg.scene,
        "--warmup-frames",
        str(cfg.warmup_frames),
        "--measure-frames",
        str(cfg.measure_frames),
        "--quality-frames",
        str(cfg.quality_frames),
        "--dlss-quality",
        str(cfg.dlss_quality),
        "--gpu-id",
        str(cfg.gpu_id),
        "--device",
        cfg.device,
    ]


def run_all_benchmarks(cfg: BenchmarkCfg, output_root: Path) -> Path:
    """Run isolated mode workers and write images plus one Markdown report."""
    import dexsim

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = output_root / f"rendering_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "benchmark_settings.json").write_text(
        json.dumps(cfg.to_dict(), indent=2) + "\n", encoding="utf-8"
    )

    print("=" * 60)
    print("Rendering Quality and Efficiency Benchmark")
    print("=" * 60)
    print(
        f"Resolution={cfg.width}x{cfg.height}, cameras={cfg.camera_count}, "
        f"envs={cfg.num_envs}, modes={','.join(cfg.denoising_modes)}, "
        f"reference={cfg.resolved_reference_mode}, spp={cfg.spp}, "
        f"quality frames={cfg.quality_frames}"
    )
    for mode in cfg.denoising_modes:
        print(f"\n=== {mode.upper()} offscreen worker ===")
        subprocess.run(_worker_command(mode, cfg, run_dir), check=True)

    performance = {
        mode: json.loads(
            (run_dir / f"{mode}_performance.json").read_text(encoding="utf-8")
        )
        for mode in cfg.denoising_modes
    }
    reference_mode = cfg.resolved_reference_mode
    reference = _load_frames(run_dir / f"{reference_mode}_frames.npz")
    metric_rows: list[dict[str, object]] = []
    for mode in cfg.denoising_modes:
        candidate = (
            reference
            if mode == reference_mode
            else _load_frames(run_dir / f"{mode}_frames.npz")
        )
        metric_rows.append(
            _aggregate_quality(
                mode,
                reference,
                candidate,
                float(performance[mode]["fps"]),
                expected_frames=cfg.quality_frames,
            )
        )
        del candidate
    del reference
    perf_rows = [
        {
            "mode": mode,
            "camera_count": cfg.camera_count,
            "num_envs": cfg.num_envs,
            "rendered_images": int(performance[mode]["rendered_images"]),
            "cost_time_ms": round(float(performance[mode]["median_ms"]), 3),
            "mean_ms": round(float(performance[mode]["mean_ms"]), 3),
            "p95_ms": round(float(performance[mode]["p95_ms"]), 3),
            "fps": round(float(performance[mode]["fps"]), 3),
            "camera_env_fps": round(
                float(performance[mode]["fps"]) * cfg.camera_count * cfg.num_envs,
                3,
            ),
            "cpu_delta_mb": round(float(performance[mode]["cpu_delta_mb"]), 3),
            "gpu_delta_mb": round(float(performance[mode]["gpu_delta_mb"]), 3),
            "peak_gpu_mb": round(float(performance[mode]["peak_gpu_mb"]), 3),
            "torch_peak_gpu_mb": round(
                float(performance[mode]["torch_peak_gpu_mb"]), 3
            ),
        }
        for mode in cfg.denoising_modes
    ]
    leaderboard_rows = build_leaderboard_rows(metric_rows)
    (run_dir / "benchmark_summary.json").write_text(
        json.dumps(
            {
                "settings": cfg.to_dict(),
                "performance": perf_rows,
                "metrics": metric_rows,
                "leaderboard": leaderboard_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    comparison_path = _write_comparison_image(
        run_dir, frame_index=min(cfg.quality_frames // 2, cfg.quality_frames - 1)
    )
    report_path = write_markdown_report(
        output_path=run_dir / "report.md",
        perf_rows=perf_rows,
        metric_rows=metric_rows,
        leaderboard_rows=leaderboard_rows,
        settings=cfg.to_dict(),
        notes=[
            f"Quality values are consistency metrics relative to {cfg.resolved_reference_mode.upper()} at {cfg.spp} spp, not absolute ground-truth fidelity.",
            "Every mode uses the same fixed scene, camera trajectories, output resolution, and parallel environment count.",
            f"Run settings: {cfg.width}x{cfg.height}, cameras={cfg.camera_count}, environments={cfg.num_envs}, warmup={cfg.warmup_frames}, measured={cfg.measure_frames}, quality frames={cfg.quality_frames}, DLSS quality={cfg.dlss_quality}.",
            f"GPU {cfg.gpu_id}: {_gpu_description(cfg.gpu_id)}; DexSim {getattr(dexsim, '__version__', 'unknown')}.",
            "SSIM is computed over every captured RGB image; temporal_delta_mae compares corresponding consecutive-frame changes.",
            "Timing excludes scene/world setup and includes all configured offscreen camera groups plus GPU completion.",
            "peak_gpu_mb is process VRAM from nvidia-smi; torch_peak_gpu_mb covers PyTorch-managed allocations only.",
            f"Comparison image: {comparison_path.name}",
        ],
    )
    print("\n" + "=" * 60)
    print(f"Markdown report saved: {report_path}")
    print(f"Comparison image saved: {comparison_path}")
    print("=" * 60)
    return report_path


def _build_sweep_leaderboard_rows(
    metric_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Aggregate per-environment metrics into one row per denoising mode."""
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in metric_rows:
        grouped.setdefault(str(row["mode"]), []).append(row)
    aggregate_rows = [
        {
            "mode": mode,
            "success_rate": round(
                float(np.mean([float(row["success_rate"]) for row in rows])), 6
            ),
            "ssim": round(float(np.mean([float(row["ssim"]) for row in rows])), 6),
            "fps": round(float(np.mean([float(row["fps"]) for row in rows])), 6),
        }
        for mode, rows in grouped.items()
    ]
    return build_leaderboard_rows(aggregate_rows)


def run_environment_sweep(
    cfg: BenchmarkCfg,
    environment_counts: tuple[int, ...] | list[int],
    output_root: Path,
) -> Path:
    """Run one benchmark matrix over several parallel environment counts.

    Each environment count is isolated in its own set of mode workers. The
    returned report combines all timing/memory rows and quality rows while the
    leaderboard aggregates each denoising mode across the sweep.
    """
    counts = tuple(dict.fromkeys(int(count) for count in environment_counts))
    if not counts or any(count <= 0 for count in counts):
        raise ValueError("environment_counts must contain positive integers.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    sweep_dir = output_root / f"rendering_sweep_{timestamp}"
    runs_root = sweep_dir / "runs"
    runs_root.mkdir(parents=True, exist_ok=False)
    performance_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []

    print("=" * 60)
    print("Rendering Environment Sweep")
    print("=" * 60)
    print(
        f"Resolution={cfg.width}x{cfg.height}, cameras={cfg.camera_count}, "
        f"envs={','.join(str(count) for count in counts)}, "
        f"modes={','.join(cfg.denoising_modes)}"
    )
    for index, count in enumerate(counts, start=1):
        print(f"\n--- Sweep {index}/{len(counts)}: num_envs={count} ---")
        run_cfg = cfg.with_overrides(num_envs=count)
        report_path = run_all_benchmarks(run_cfg, runs_root)
        run_summary_path = report_path.parent / "benchmark_summary.json"
        summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
        performance_rows.extend(summary["performance"])
        metric_rows.extend(summary["metrics"])

    leaderboard_rows = _build_sweep_leaderboard_rows(metric_rows)
    settings = cfg.to_dict()
    settings["environment_counts"] = list(counts)
    settings["num_envs"] = list(counts)
    report_path = write_markdown_report(
        output_path=sweep_dir / "report.md",
        perf_rows=performance_rows,
        metric_rows=metric_rows,
        leaderboard_rows=leaderboard_rows,
        settings=settings,
        notes=[
            "Each environment count uses an isolated simulation worker for every denoising mode.",
            f"The sweep covers num_envs={list(counts)} at {cfg.width}x{cfg.height} with {cfg.camera_count} camera(s).",
            "Leaderboard values are averages across environment counts; Time & Memory and Success & Other Metrics retain one row per environment count and mode.",
            "DexSim may use the direct CUDA output path above the device's Vulkan overlay layer limit; this run emitted that warning for 64 and 128 environments.",
            "Individual run artifacts are stored under the runs/ directory.",
        ],
    )
    (sweep_dir / "sweep_summary.json").write_text(
        json.dumps(
            {
                "settings": settings,
                "performance": performance_rows,
                "metrics": metric_rows,
                "leaderboard": leaderboard_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("\n" + "=" * 60)
    print(f"Sweep report saved: {report_path}")
    print("=" * 60)
    return report_path


def build_parser() -> argparse.ArgumentParser:
    """Build command-line arguments for parent and worker execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JSON or YAML file containing BenchmarkCfg settings.",
    )
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument(
        "--resolution",
        type=_parse_resolution,
        default=None,
        metavar="WIDTHxHEIGHT",
        help="Set width and height together, for example 1280x720.",
    )
    parser.add_argument("--camera-count", "--num-cameras", type=int, default=None)
    parser.add_argument("--num-envs", "--environments", type=int, default=None)
    parser.add_argument(
        "--env-sweep",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help="Run one report over several parallel environment counts.",
    )
    parser.add_argument(
        "--denoising-modes",
        "--modes",
        nargs="+",
        choices=RENDER_MODES,
        default=None,
        help="Denoising paths to compare (off, optix, dlss, nrd).",
    )
    parser.add_argument(
        "--reference-mode",
        choices=RENDER_MODES,
        default=None,
    )
    parser.add_argument(
        "--renderer",
        choices=("auto", "hybrid", "fast-rt", "rt"),
        default=None,
    )
    parser.add_argument("--spp", type=int, default=None)
    parser.add_argument("--scene", choices=(FIXED_SCENE_NAME,), default=None)
    parser.add_argument("--warmup-frames", type=int, default=None)
    parser.add_argument("--measure-frames", type=int, default=None)
    parser.add_argument("--quality-frames", type=int, default=None)
    parser.add_argument("--dlss-quality", type=int, default=None, choices=range(-1, 6))
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/benchmarks"))
    parser.add_argument("--run-dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-mode", choices=RENDER_MODES, default=None, help=argparse.SUPPRESS
    )
    return parser


def _parse_resolution(value: str) -> tuple[int, int]:
    """Parse a command-line resolution in ``WIDTHxHEIGHT`` form."""
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("resolution must use WIDTHxHEIGHT.")
    try:
        return int(parts[0]), int(parts[1])
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "resolution dimensions must be integers."
        ) from error


def _runtime_unavailable_reason() -> str | None:
    """Return a skip reason when the rendering runtime is unavailable."""
    try:
        import dexsim  # noqa: F401
        from dexsim import engine, types  # noqa: F401
        import torch
    except ImportError as error:
        return f"DexSim/PyTorch is unavailable ({error})."
    if not torch.cuda.is_available():
        return "CUDA is unavailable; offscreen rendering requires a render GPU."
    return None


def main() -> None:
    """Run the benchmark parent or one isolated rendering worker."""
    args = build_parser().parse_args()
    cfg = (
        BenchmarkCfg.from_file(args.config)
        if args.config is not None
        else BenchmarkCfg()
    )
    resolution = args.resolution
    cfg = cfg.with_overrides(
        width=resolution[0] if resolution is not None else args.width,
        height=resolution[1] if resolution is not None else args.height,
        camera_count=args.camera_count,
        num_envs=args.num_envs,
        denoising_modes=args.denoising_modes,
        reference_mode=args.reference_mode,
        renderer=args.renderer,
        spp=args.spp,
        scene=args.scene,
        warmup_frames=args.warmup_frames,
        measure_frames=args.measure_frames,
        quality_frames=args.quality_frames,
        dlss_quality=args.dlss_quality,
        gpu_id=args.gpu_id,
        device=args.device,
    )
    unavailable_reason = _runtime_unavailable_reason()
    if unavailable_reason is not None:
        print(f"Skipped rendering benchmark: {unavailable_reason}")
        return
    if args.env_sweep is not None:
        if args.worker_mode is not None:
            raise ValueError("--env-sweep cannot be used with worker mode.")
        run_environment_sweep(cfg, args.env_sweep, args.output_root)
        return
    if args.worker_mode is not None:
        if args.run_dir is None:
            raise ValueError("--run-dir is required for worker mode.")
        _run_mode_worker(args.worker_mode, cfg, args.run_dir)
        return
    run_all_benchmarks(cfg, args.output_root)


if __name__ == "__main__":
    main()
