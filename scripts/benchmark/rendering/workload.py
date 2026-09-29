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

"""Camera-specific configuration, scene and validation over the shared core."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from embodichain.utils import configclass
from scripts.benchmark.core.artifacts import stable_hash
from scripts.benchmark.core.measurement import measure_loop

__all__ = ["PilotCfg", "scene_spec", "config_hash", "measure"]


@configclass
class PilotCfg:
    """One RGB view with explicit warm-up and measured capture counts."""

    width: int = 256
    height: int = 256
    warmup_frames: int = 30
    measured_frames: int = 300

    def __post_init__(self) -> None:
        for name in ("width", "height", "warmup_frames", "measured_frames"):
            value = getattr(self, name)
            minimum = 0 if name == "warmup_frames" else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")

    @classmethod
    def load(cls, path: Path) -> PilotCfg:
        """Read the closed pilot configuration from JSON."""
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise ValueError("Pilot configuration must be a JSON object")
        return cls(**data)


def scene_spec() -> dict:
    """Return the same procedural scene in meters for both backends."""
    return {
        "version": "three_boxes_v1",
        "eye": [1.3, 1.3, 1.0],
        "target": [0.0, 0.0, 0.12],
        "horizontal_fov_deg": 60.0,
        "near_m": 0.01,
        "far_m": 10.0,
        "boxes": [
            {
                "id": "table",
                "size": [1.4, 1.2, 0.05],
                "position": [0.0, 0.0, -0.025],
                "color": [0.5, 0.5, 0.5],
            },
            {
                "id": "red",
                "size": [0.2, 0.2, 0.2],
                "position": [-0.28, 0.0, 0.1],
                "color": [0.8, 0.05, 0.05],
            },
            {
                "id": "green",
                "size": [0.16, 0.16, 0.3],
                "position": [0.05, 0.2, 0.15],
                "color": [0.05, 0.8, 0.05],
            },
            {
                "id": "blue",
                "size": [0.24, 0.18, 0.12],
                "position": [0.25, -0.18, 0.06],
                "color": [0.05, 0.1, 0.8],
            },
        ],
    }


def config_hash(cfg: PilotCfg) -> str:
    """Hash workload semantics, independent of platform/interpreter."""
    payload = {
        "config": asdict(cfg),
        "scene": scene_spec(),
        "boundary": "render_to_cpu_rgb",
    }
    return stable_hash(payload)


def measure(
    cfg: PilotCfg,
    capture: Callable[[], np.ndarray],
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> dict:
    """Measure synchronous render-to-host RGB latency and full-loop throughput.

    The callback must return a completed, contiguous HWC uint8 RGB image.
    Warm-up is excluded. Capture latency excludes Python validation, while
    the throughput window includes that validation and loop bookkeeping.
    """

    def validate_rgb(rgb: np.ndarray) -> None:
        if (
            rgb.shape != (cfg.height, cfg.width, 3)
            or rgb.dtype != np.uint8
            or not rgb.flags.c_contiguous
        ):
            raise ValueError("capture must return contiguous HWC uint8 RGB")

    timing = measure_loop(
        capture,
        warmup=cfg.warmup_frames,
        iterations=cfg.measured_frames,
        validate=validate_rgb,
        clock=clock,
    )
    return {
        "boundary": "render_to_cpu_rgb",
        "completion": "blocking_host_readback",
        "num_envs": 1,
        "cameras_per_env": 1,
        "warmup_frames": cfg.warmup_frames,
        "measured_frames": cfg.measured_frames,
        "window_s": timing.window_s,
        "camera_frames_per_s": timing.operations_per_s,
        "gpixel_per_s": timing.operations_per_s * cfg.width * cfg.height / 1e9,
        "capture_latency_s": list(timing.latencies_s),
        "latency_p50_s": timing.p50_s,
        "latency_p95_s": timing.p95_s,
    }
