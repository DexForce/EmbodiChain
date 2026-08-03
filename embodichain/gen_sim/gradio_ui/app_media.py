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

"""Run-log archival and video/dataset preview generation."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from app_config import *  # noqa: F403 - media paths and limits are configuration.
from app_state import format_timing_lines, runtime, runtime_lock, snapshot_timing_locked


def archive_run_log(
    *,
    mode: str,
    task_description: str = "",
    scene_description: str = "",
    outcome: str,
    audience_video: Path | None = None,
) -> Path | None:
    with runtime_lock:
        run_logs = list(runtime.log_lines)
        status_text = runtime.status
        last_error = runtime.last_error
        runtime_video = runtime.video_path
        timing_durations, simulation_duration = snapshot_timing_locked()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = make_next_log_archive_dir()
    video_paths, video_errors = archive_audience_video(
        run_dir,
        audience_video or runtime_video,
    )
    log_path = run_dir / "log.md"
    content = [
        f"mode: {mode}",
        "",
        f"Timestamp: {timestamp}",
        f"Outcome: {outcome}",
        "",
        "## Task description",
        "",
        task_description or "",
        "",
        "## Scene description",
        "",
        scene_description or "",
        "",
        "## Status",
        "",
        status_text or "",
    ]
    if last_error:
        content.extend(["", "## Last error", "", last_error])
    if video_paths:
        content.extend(
            [
                "",
                "## Archived audience video",
                "",
                *[path.as_posix() for path in video_paths],
            ]
        )
    if video_errors:
        content.extend(["", "## Video archive errors", "", *video_errors])
    content.extend(
        [
            "",
            "## Logs",
            "",
            "```text",
            "\n".join(run_logs) if run_logs else "(no logs)",
            "```",
            "",
            "## Timing",
            "",
            *format_timing_lines(timing_durations, simulation_duration),
            "",
        ]
    )

    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path.write_text("\n".join(content), encoding="utf-8")
    except Exception as exc:
        with runtime_lock:
            runtime.log_lines.append(f"Failed to archive run log: {exc}")
        return None
    return log_path


def make_next_log_archive_dir() -> Path:
    AUTO_LOG_DIR.mkdir(parents=True, exist_ok=True)
    existing_indices = [
        int(path.name)
        for path in AUTO_LOG_DIR.iterdir()
        if path.is_dir() and path.name.isdigit()
    ]
    next_index = (max(existing_indices) + 1) if existing_indices else 1
    while True:
        candidate = AUTO_LOG_DIR / f"{next_index:04d}"
        if not candidate.exists():
            try:
                candidate.mkdir(parents=True, exist_ok=False)
                return candidate
            except FileExistsError:
                pass
        next_index += 1


def archive_audience_video(
    run_dir: Path,
    video_path: Path | None,
) -> tuple[list[Path], list[str]]:
    copied_paths: list[Path] = []
    errors: list[str] = []
    if video_path is None:
        return copied_paths, errors
    if not video_path.is_file():
        return copied_paths, [f"Audience video not found: {video_path}"]
    destination = run_dir / "audience_video" / video_path.name
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(video_path, destination)
    except Exception as exc:
        errors.append(f"Failed to archive audience video {video_path}: {exc}")
        return copied_paths, errors
    copied_paths.append(destination.relative_to(run_dir))
    return copied_paths, errors


def collect_output_videos() -> list[Path]:
    if not OUTPUTS_DIR.is_dir():
        return []
    return sorted(
        path
        for path in OUTPUTS_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )


def collect_audience_output_videos() -> list[Path]:
    videos = collect_output_videos()
    audience_videos = [
        path
        for path in videos
        if "audience" in path.relative_to(OUTPUTS_DIR).as_posix().lower()
    ]
    if audience_videos:
        return audience_videos
    return [
        path
        for path in videos
        if "audience" in path.relative_to(OUTPUTS_DIR).as_posix().lower()
    ]


def latest_audience_output_video(min_mtime_ns: int | None = None) -> Path | None:
    latest_path: Path | None = None
    latest_mtime = -1
    for path in collect_audience_output_videos():
        try:
            mtime = path.stat().st_mtime_ns
        except OSError:
            continue
        if min_mtime_ns is not None and mtime < min_mtime_ns:
            continue
        if mtime > latest_mtime:
            latest_path = path
            latest_mtime = mtime
    return latest_path


def configured_lerobot_roots() -> list[Path]:
    roots: list[Path] = []
    env_root = os.environ.get("EMBODICHAIN_DATASET_ROOT")
    if env_root:
        roots.append(Path(env_root).expanduser())
    roots.append(Path("~/.cache/embodichain_datasets").expanduser())

    config_roots = read_lerobot_save_paths(CURRENT_PATHS.fast_gym_config)
    roots.extend(config_roots)

    normalized: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        root = root.expanduser()
        if not root.is_absolute():
            root = EMBODICHAIN_ROOT / root
        try:
            resolved = root.resolve()
        except OSError:
            resolved = root
        if resolved in seen:
            continue
        seen.add(resolved)
        normalized.append(root)
    return normalized


def read_lerobot_save_paths(config_path: Path) -> list[Path]:
    if not config_path.is_file():
        return []
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    paths: list[Path] = []

    def visit(value: Any, key_path: tuple[str, ...] = ()) -> None:
        if isinstance(value, dict):
            if key_path[-2:] == ("lerobot", "params") and isinstance(
                value.get("save_path"), str
            ):
                paths.append(Path(value["save_path"]))
            for key, child in value.items():
                visit(child, (*key_path, str(key)))
        elif isinstance(value, list):
            for item in value:
                visit(item, key_path)

    visit(config)
    return paths


def collect_lerobot_datasets() -> list[Path]:
    datasets: list[Path] = []
    for root in configured_lerobot_roots():
        if not root.is_dir():
            continue
        try:
            candidates = list(root.iterdir())
        except OSError:
            continue
        for candidate in candidates:
            if not candidate.is_dir():
                continue
            if (candidate / "meta" / "info.json").is_file() or (
                candidate / "data"
            ).is_dir():
                datasets.append(candidate)
    return datasets


def latest_lerobot_dataset(min_mtime_ns: int | None = None) -> Path | None:
    latest_path: Path | None = None
    latest_mtime = -1
    for dataset_path in collect_lerobot_datasets():
        if not lerobot_dataset_has_frames(dataset_path):
            continue
        mtime = latest_lerobot_dataset_mtime_ns(dataset_path)
        if min_mtime_ns is not None and mtime < min_mtime_ns:
            continue
        if mtime > latest_mtime:
            latest_path = dataset_path
            latest_mtime = mtime
    return latest_path


def lerobot_dataset_has_frames(dataset_path: Path) -> bool:
    data_dir = dataset_path / "data"
    return data_dir.is_dir() and any(data_dir.rglob("*.parquet"))


def latest_lerobot_dataset_mtime_ns(dataset_path: Path) -> int:
    latest_mtime = -1
    for child in dataset_path.rglob("*"):
        if not child.is_file():
            continue
        try:
            latest_mtime = max(latest_mtime, child.stat().st_mtime_ns)
        except OSError:
            continue
    if latest_mtime >= 0:
        return latest_mtime
    try:
        return dataset_path.stat().st_mtime_ns
    except OSError:
        return -1


def build_lerobot_preview_video(dataset_path: Path) -> Path | None:
    parquet_paths = sorted((dataset_path / "data").rglob("*.parquet"))
    if not parquet_paths:
        return None

    latest_source_mtime = max(
        latest_lerobot_dataset_mtime_ns(dataset_path),
        *(path.stat().st_mtime_ns for path in parquet_paths),
    )
    output_path = LEROBOT_PREVIEW_DIR / f"{dataset_path.name}_data_preview.mp4"
    if output_path.is_file() and output_path.stat().st_mtime_ns >= latest_source_mtime:
        return output_path

    try:
        import imageio.v2 as imageio
        import pandas as pd
    except Exception as exc:
        with runtime_lock:
            runtime.log_lines.append(
                f"LeRobot preview skipped; missing dependency: {exc}"
            )
        return None

    try:
        data_frame = pd.concat(
            [pd.read_parquet(path) for path in parquet_paths],
            ignore_index=True,
        )
    except Exception as exc:
        with runtime_lock:
            runtime.log_lines.append(f"LeRobot preview skipped; read failed: {exc}")
        return None

    if data_frame.empty:
        return None

    try:
        fps = read_lerobot_fps(dataset_path) or 25
        fps = max(1, min(int(round(fps)), 30))
        frames = render_lerobot_data_frames(data_frame, dataset_path.name)
        if not frames:
            return None
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(output_path, fps=fps, codec="libx264") as writer:
            for frame in frames:
                writer.append_data(frame)
    except Exception as exc:
        with runtime_lock:
            runtime.log_lines.append(f"LeRobot preview skipped; render failed: {exc}")
        return None

    return output_path


def video_duration_seconds(video_path: Path) -> float | None:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        duration = float(result.stdout.strip())
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None
    return duration if duration > 0 else None


def build_single_env_combined_video(
    audience_video: Path | None,
    lerobot_video: Path | None,
) -> Path | None:
    """Create a synchronized side-by-side simulation and LeRobot video."""
    if (
        audience_video is None
        or lerobot_video is None
        or not audience_video.is_file()
        or not lerobot_video.is_file()
    ):
        return None

    audience_duration = video_duration_seconds(audience_video)
    lerobot_duration = video_duration_seconds(lerobot_video)
    if audience_duration is None or lerobot_duration is None:
        return None

    latest_source_mtime = max(
        audience_video.stat().st_mtime_ns,
        lerobot_video.stat().st_mtime_ns,
    )
    output_path = (
        COMBINED_PREVIEW_DIR
        / f"{safe_filename_part(audience_video.stem)}_with_lerobot.mp4"
    )
    if output_path.is_file() and output_path.stat().st_mtime_ns >= latest_source_mtime:
        return output_path

    lerobot_time_scale = audience_duration / lerobot_duration
    filter_graph = (
        f"[0:v]fps={COMBINED_VIDEO_FPS},scale=960:540:force_original_aspect_ratio=decrease,"
        "pad=960:540:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,"
        "setpts=PTS-STARTPTS[sim];"
        f"[1:v]fps={COMBINED_VIDEO_FPS},scale=960:540:force_original_aspect_ratio=decrease,"
        "pad=960:540:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,"
        f"setpts=(PTS-STARTPTS)*{lerobot_time_scale:.9f}[data];"
        "[sim][data]hstack=inputs=2:shortest=1,format=yuv420p[video]"
    )
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(audience_video),
        "-i",
        str(lerobot_video),
        "-filter_complex",
        filter_graph,
        "-map",
        "[video]",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=180,
        )
        if result.returncode == 0 and output_path.is_file():
            return output_path
        with runtime_lock:
            runtime.log_lines.append(
                "Combined video skipped: "
                + (
                    result.stderr.strip().splitlines()[-1]
                    if result.stderr
                    else "ffmpeg failed"
                )
            )
    except (OSError, subprocess.TimeoutExpired) as exc:
        with runtime_lock:
            runtime.log_lines.append(f"Combined video skipped: {exc}")
    return None


def read_lerobot_fps(dataset_path: Path) -> int | None:
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.is_file():
        return None
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    fps = info.get("fps")
    if isinstance(fps, (int, float)):
        return int(fps)
    return None


def render_lerobot_data_frames(data_frame: Any, dataset_name: str) -> list[np.ndarray]:
    total_rows = len(data_frame)
    frame_indices = np.linspace(
        0,
        total_rows - 1,
        num=min(total_rows, LEROBOT_PREVIEW_MAX_FRAMES),
        dtype=int,
    )
    state = series_to_matrix(data_frame.get("observation.state"))
    action = series_to_matrix(data_frame.get("action"))
    qvel = series_to_matrix(data_frame.get("observation.qvel"))
    timestamps = numeric_column(data_frame, "timestamp", total_rows)

    frames: list[np.ndarray] = []
    for row_index in frame_indices:
        image = Image.new("RGB", (960, 544), (247, 248, 250))
        draw = ImageDraw.Draw(image)
        draw_lerobot_header(
            draw,
            dataset_name=dataset_name,
            row_index=int(row_index),
            total_rows=total_rows,
            timestamp=float(timestamps[row_index]) if len(timestamps) else None,
        )
        draw_signal_panel(
            draw, (40, 96, 920, 220), state, row_index, "observation.state"
        )
        draw_signal_panel(draw, (40, 244, 920, 368), action, row_index, "action")
        draw_bar_panel(draw, (40, 392, 920, 506), qvel, row_index, "observation.qvel")
        frames.append(np.asarray(image))
    return frames


def series_to_matrix(series: Any, max_dims: int = 12) -> np.ndarray:
    if series is None:
        return np.empty((0, 0), dtype=float)
    rows: list[np.ndarray] = []
    for value in series:
        array = np.asarray(value, dtype=float).reshape(-1)
        if array.size:
            rows.append(array[:max_dims])
    if not rows:
        return np.empty((0, 0), dtype=float)
    width = max(row.size for row in rows)
    matrix = np.full((len(rows), width), np.nan, dtype=float)
    for index, row in enumerate(rows):
        matrix[index, : row.size] = row
    return matrix


def numeric_column(data_frame: Any, column: str, fallback_length: int) -> np.ndarray:
    if column not in data_frame:
        return np.arange(fallback_length, dtype=float)
    try:
        values = np.asarray(data_frame[column], dtype=float)
    except Exception:
        values = np.arange(fallback_length, dtype=float)
    return values


def draw_lerobot_header(
    draw: ImageDraw.ImageDraw,
    *,
    dataset_name: str,
    row_index: int,
    total_rows: int,
    timestamp: float | None,
) -> None:
    draw.text((40, 28), "LeRobot dataset preview", fill=(17, 24, 39))
    short_name = dataset_name if len(dataset_name) <= 78 else f"{dataset_name[:75]}..."
    draw.text((40, 54), short_name, fill=(75, 85, 99))
    progress = 0 if total_rows <= 1 else row_index / (total_rows - 1)
    draw.text((750, 28), f"frame {row_index + 1}/{total_rows}", fill=(17, 24, 39))
    if timestamp is not None:
        draw.text((750, 54), f"t = {timestamp:.2f}s", fill=(75, 85, 99))
    draw.rectangle((40, 78, 920, 82), fill=(224, 231, 239))
    draw.rectangle((40, 78, int(40 + 880 * progress), 82), fill=(37, 99, 235))


def draw_signal_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    matrix: np.ndarray,
    row_index: int,
    title: str,
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=8, fill=(255, 255, 255), outline=(209, 213, 219))
    draw.text((x0 + 14, y0 + 10), title, fill=(17, 24, 39))
    if matrix.size == 0:
        draw.text((x0 + 14, y0 + 48), "No numeric data", fill=(107, 114, 128))
        return
    plot_box = (x0 + 14, y0 + 36, x1 - 14, y1 - 16)
    draw_timeseries(draw, plot_box, matrix, row_index)


def draw_bar_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    matrix: np.ndarray,
    row_index: int,
    title: str,
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=8, fill=(255, 255, 255), outline=(209, 213, 219))
    draw.text((x0 + 14, y0 + 10), title, fill=(17, 24, 39))
    if matrix.size == 0 or row_index >= len(matrix):
        draw.text((x0 + 14, y0 + 48), "No numeric data", fill=(107, 114, 128))
        return
    values = matrix[row_index]
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return
    max_abs = max(float(np.nanmax(np.abs(finite))), 1e-6)
    base_y = y1 - 30
    left = x0 + 18
    available_width = x1 - x0 - 36
    bar_count = min(len(values), 12)
    bar_gap = 8
    bar_width = max(8, (available_width - bar_gap * (bar_count - 1)) // bar_count)
    for index in range(bar_count):
        value = values[index]
        if not np.isfinite(value):
            continue
        x = left + index * (bar_width + bar_gap)
        height = int((abs(float(value)) / max_abs) * 58)
        color = (22, 163, 74) if value >= 0 else (220, 38, 38)
        y_top = base_y - height
        draw.rectangle((x, y_top, x + bar_width, base_y), fill=color)
        draw.text((x, base_y + 5), str(index), fill=(107, 114, 128))


def draw_timeseries(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    matrix: np.ndarray,
    row_index: int,
) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline=(229, 231, 235))
    sample_count = min(len(matrix), LEROBOT_PREVIEW_MAX_FRAMES)
    if sample_count <= 1:
        return
    sampled = matrix[
        np.linspace(0, len(matrix) - 1, num=sample_count, dtype=int),
        : min(matrix.shape[1], 8),
    ]
    finite = sampled[np.isfinite(sampled)]
    if finite.size == 0:
        return
    minimum = float(np.nanmin(finite))
    maximum = float(np.nanmax(finite))
    if math.isclose(minimum, maximum):
        minimum -= 1.0
        maximum += 1.0
    palette = [
        (37, 99, 235),
        (5, 150, 105),
        (217, 119, 6),
        (220, 38, 38),
        (124, 58, 237),
        (8, 145, 178),
        (79, 70, 229),
        (202, 138, 4),
    ]

    def point(sample_index: int, value: float) -> tuple[int, int]:
        x = int(x0 + (x1 - x0) * sample_index / (sample_count - 1))
        y = int(y1 - (y1 - y0) * (value - minimum) / (maximum - minimum))
        return x, y

    for dim in range(sampled.shape[1]):
        points = [
            point(index, float(value))
            for index, value in enumerate(sampled[:, dim])
            if np.isfinite(value)
        ]
        if len(points) >= 2:
            draw.line(points, fill=palette[dim % len(palette)], width=2)

    cursor_x = int(x0 + (x1 - x0) * row_index / max(len(matrix) - 1, 1))
    draw.line((cursor_x, y0, cursor_x, y1), fill=(17, 24, 39), width=2)


def safe_filename_part(value: str) -> str:
    safe = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_" for char in value.strip()
    )
    return safe.strip("_")[:80]


def run_articraft_viser_preview(args: argparse.Namespace) -> None:
    """Load an Articraft URDF and publish its initial scene topology to Viser.

    The generic asset-preview command starts Viser lazily.  For an asset loaded
    before that first capture, explicitly marking the topology dirty and
    capturing once more ensures the initial scene is sent to the browser.

    Args:
        args: Parsed preview-asset command-line arguments.
    """
    from embodichain.lab.scripts import preview_asset
    from embodichain.lab.sim.sim_manager import SimulationManager
    from embodichain.utils.logger import log_info

    sim = SimulationManager(preview_asset.build_sim_cfg(args))
    try:
        if args.env_map:
            log_info(f"Setting environment map: {args.env_map} ...", color="green")
            sim.set_indirect_lighting(args.env_map)

        assets = preview_asset.load_assets(sim, args)
        log_info(f"Loaded {len(assets)} asset(s) successfully.", color="green")
        if args.viser:
            sim.start_visualization()
            sim.notify_visualization_topology_changed()
            sim.capture_visualization_safely(force=True)
        preview_asset._run_preview_mode(sim, assets, args)
    finally:
        log_info("Destroying simulation ...", color="green")
        sim.destroy()


def articraft_viser_preview_cli(argv: Sequence[str] | None = None) -> None:
    """Run the Articraft-aware variant of the generic preview-asset CLI.

    Args:
        argv: Arguments excluding the program name, or ``None`` for ``sys.argv``.
    """
    from embodichain.lab.scripts import preview_asset

    parser = preview_asset._create_parser()
    run_articraft_viser_preview(parser.parse_args(argv))


if __name__ == "__main__":
    articraft_viser_preview_cli()
