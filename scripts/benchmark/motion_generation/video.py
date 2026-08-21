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

"""Optional Atomic Task replay recording helpers.

Recording is a second, untimed physics pass. Failures never change trial
success and never add a fourth Markdown table.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from embodichain.lab.sim import SimulationManager

__all__ = [
    "DEFAULT_VIDEO_LOOK_AT",
    "VideoRecordCfg",
    "build_video_path",
    "record_with_window",
    "should_record_case",
    "summarize_video_recording",
    "video_cfg_from_args",
]

DEFAULT_VIDEO_FPS = 20
DEFAULT_VIDEO_MAX_MEMORY_MB = 2048
DEFAULT_VIDEO_WIDTH = 640
DEFAULT_VIDEO_HEIGHT = 480
DEFAULT_VIDEO_CASE_LIMIT = 0
DEFAULT_VIDEO_LOOK_AT = (
    (-1.25, -1.15, 0.95),
    (-0.25, -0.02, 0.25),
    (0.0, 0.0, 1.0),
)

LookAt = tuple[Sequence[float], Sequence[float], Sequence[float]]


@dataclass(frozen=True)
class VideoRecordCfg:
    """CLI-resolved recording policy for Atomic Task measured replays."""

    enabled: bool = False
    record_failed: bool = False
    case_limit: int = DEFAULT_VIDEO_CASE_LIMIT
    fps: int = DEFAULT_VIDEO_FPS
    width: int = DEFAULT_VIDEO_WIDTH
    height: int = DEFAULT_VIDEO_HEIGHT
    max_memory_mb: int = DEFAULT_VIDEO_MAX_MEMORY_MB
    look_at: LookAt = DEFAULT_VIDEO_LOOK_AT
    output_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.case_limit < 0:
            raise ValueError("video case_limit must be non-negative.")
        if self.fps <= 0:
            raise ValueError("video fps must be positive.")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("video width and height must be positive.")
        if self.max_memory_mb <= 0:
            raise ValueError("video max_memory_mb must be positive.")


def should_record_case(cfg: VideoRecordCfg, recorded_count: int, success: bool) -> bool:
    """Return whether one measured Atomic Task case should emit a video."""
    if not cfg.enabled:
        return False
    if not success and not cfg.record_failed:
        return False
    return cfg.case_limit == 0 or recorded_count < cfg.case_limit


def build_video_path(
    output_dir: Path,
    algorithm_id: str,
    skill_id: str,
    case_id: str,
) -> Path:
    """Build ``{algorithm}_{skill}_{case}.mp4`` under the run videos directory."""

    def _sanitize(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"{_sanitize(algorithm_id)}_{_sanitize(skill_id)}_{_sanitize(case_id)}.mp4"
    )
    return output_dir / filename


def record_with_window(
    sim: "SimulationManager",
    cfg: VideoRecordCfg,
    video_path: Path,
    replay_fn: Callable[[], None],
) -> Path | None:
    """Record one headless replay. Failures print a warning and return ``None``."""
    try:
        original_width = sim.sim_config.width
        original_height = sim.sim_config.height
        recording_started = False
        try:
            sim.sim_config.width = cfg.width
            sim.sim_config.height = cfg.height
            recording_started = sim.start_window_record(
                save_path=str(video_path),
                fps=cfg.fps,
                max_memory=cfg.max_memory_mb,
                look_at=cfg.look_at,
                use_sim_time=True,
            )
        finally:
            sim.sim_config.width = original_width
            sim.sim_config.height = original_height
        if not recording_started:
            return None

        stop_success = False
        try:
            replay_fn()
        finally:
            if sim.is_window_recording():
                stop_success = sim.stop_window_record()
            sim.wait_window_record_saves()
        return video_path if stop_success else None
    except Exception as exc:  # noqa: BLE001 - recording must not change success
        try:
            if sim.is_window_recording():
                sim.stop_window_record()
            sim.wait_window_record_saves()
        except Exception:
            pass
        print(
            "Warning: failed to record Atomic Task replay video "
            f"{video_path}: {type(exc).__name__}: {exc}"
        )
        return None


def summarize_video_recording(
    cfg: VideoRecordCfg, video_paths: Sequence[str]
) -> list[str]:
    """Return report notes describing recording coverage without extra tables."""
    if not cfg.enabled:
        return ["Video policy: disabled."]
    if cfg.record_failed:
        policy = (
            "records Atomic Task measured success replays and failed-case "
            "static scenes when capture is available."
        )
    else:
        policy = (
            "records Atomic Task measured success replays only; failed "
            "cases are reported in the tables but do not emit videos."
        )
    rendered = ", ".join(video_paths) if video_paths else "none"
    return [
        f"Video policy: {policy}",
        f"videos={len(video_paths)}",
        f"Replay videos: {rendered}",
    ]


def video_cfg_from_args(args: argparse.Namespace) -> VideoRecordCfg:
    """Build recording config from the motion-generation CLI namespace."""
    output_dir = getattr(args, "video_dir", None)
    return VideoRecordCfg(
        enabled=bool(getattr(args, "record_video", False)),
        record_failed=bool(getattr(args, "record_failed_video", False)),
        case_limit=int(getattr(args, "video_case_limit", DEFAULT_VIDEO_CASE_LIMIT)),
        fps=int(getattr(args, "video_fps", DEFAULT_VIDEO_FPS)),
        width=int(getattr(args, "video_width", DEFAULT_VIDEO_WIDTH)),
        height=int(getattr(args, "video_height", DEFAULT_VIDEO_HEIGHT)),
        max_memory_mb=int(
            getattr(args, "video_max_memory", DEFAULT_VIDEO_MAX_MEMORY_MB)
        ),
        output_dir=None if output_dir in (None, "") else Path(output_dir),
    )
