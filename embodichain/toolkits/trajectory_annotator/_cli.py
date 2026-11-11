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

"""Lightweight command entry point; optional dependencies load after parsing."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import math
import json
from typing import Any

__all__ = ["cli"]


def _add_fitting_arguments(parser: argparse.ArgumentParser) -> None:
    """Share optional web fitting defaults with the runnable annotation example."""
    group = parser.add_argument_group("轨迹拟合与连续回填（可在网页中填写或修改）")
    group.add_argument("--joints", type=Path, help="关节记录：NPZ / HDF5 / Atomic PT。")
    group.add_argument(
        "--robot-profile", type=Path, help="FK/IK、TCP、关节顺序与限制的 JSON 配置。"
    )
    group.add_argument(
        "--qpos-semantics",
        choices=("measured_state", "actual_submission", "design_reference"),
    )
    group.add_argument(
        "--time-offset", type=float, help="视频零时刻对应的关节时间，秒；必须显式指定。"
    )
    group.add_argument("--qpos-key", default="qpos")
    group.add_argument("--timestamps-key", default="timestamps")
    group.add_argument(
        "--joint-dt", type=float, help="无关节时间戳时显式指定的采样周期，秒。"
    )
    group.add_argument("--env-index", type=int, default=0)
    group.add_argument(
        "--phase-hints", type=Path, help="完整阶段分区 JSON（半开、选区相对关节索引）。"
    )
    group.add_argument("--fit-config", type=Path, help="SkillFitCfg 覆盖项 JSON。")
    group.add_argument("--device", default="cpu", help="离线 FK/IK 计算设备。")


def _fitting_options(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in (
        "joints",
        "robot_profile",
        "qpos_semantics",
        "time_offset",
        "qpos_key",
        "timestamps_key",
        "joint_dt",
        "env_index",
        "device",
    ):
        value = getattr(args, name)
        if value is not None:
            result[name] = (
                str(value.expanduser().resolve()) if isinstance(value, Path) else value
            )
    for name in ("phase_hints", "fit_config"):
        path = getattr(args, name)
        if path is not None:
            result[name] = json.loads(path.read_text())
    return result


def cli(argv: Sequence[str] | None = None) -> None:
    """Launch the local annotation website for a video or HDF5 image sequence.

    Args:
        argv: Command arguments, or process arguments when omitted.
    """
    parser = argparse.ArgumentParser(
        prog="embodichain annotate-trajectory",
        description="在浏览器逐帧标注遥操图像序列的原子技能区间。",
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--video", type=Path, help="本地视频文件（PyAV）。")
    inputs.add_argument("--hdf5", type=Path, help="本地 HDF5 图像序列。")
    parser.add_argument(
        "--image-key",
        help="HDF5 图像 dataset，默认自动定位 observations/images/cam_high（也支持 observation）。",
    )
    parser.add_argument(
        "--fps",
        type=float,
        help="HDF5 播放帧率，默认读取 fps 或 metadata_json.record_fps。",
    )
    parser.add_argument(
        "--color-order",
        choices=("rgb", "bgr"),
        default=None,
        help="HDF5 原始数组的颜色顺序，默认 rgb；JPEG/PNG 按编码解码。",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        help="标注 JSON，默认视频同名或 <hdf5>.<camera>.segments.json。",
    )
    parser.add_argument(
        "--video-stream", type=int, help="容器实际视频流索引，默认第一个视频流。"
    )
    parser.add_argument(
        "--port", type=int, default=0, help="本地端口，0 自动选择（默认）。"
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="只打印网址，不自动打开浏览器。"
    )
    _add_fitting_arguments(parser)
    args = parser.parse_args(argv)
    source_path = (args.video or args.hdf5).resolve()
    if not source_path.is_file():
        parser.error(f"输入文件不存在：{source_path}")
    if not 0 <= args.port <= 65535:
        parser.error("端口必须在 0–65535 之间。")
    if args.video_stream is not None and (args.hdf5 or args.video_stream < 0):
        parser.error("--video-stream 仅适用于视频，且不能为负数。")
    if args.video and any(
        value is not None for value in (args.image_key, args.fps, args.color_order)
    ):
        parser.error("--image-key、--fps 和 --color-order 仅适用于 --hdf5。")
    if args.image_key is not None and not args.image_key.strip("/"):
        parser.error("--image-key 不能为空。")
    if args.fps is not None and (not math.isfinite(args.fps) or args.fps <= 0):
        parser.error("--fps 必须为有限正数。")
    camera = (args.image_key or "cam_high").strip("/").rsplit("/", 1)[-1]
    default_name = (
        f"{source_path.stem}.{camera}.segments.json"
        if args.hdf5
        else f"{source_path.stem}.segments.json"
    )
    annotations = (args.annotations or source_path.with_name(default_name)).resolve()
    if annotations == source_path or (
        annotations.exists() and annotations.samefile(source_path)
    ):
        parser.error("标注文件不能与输入文件相同。")
    try:
        from .app import launch_annotation

        if args.hdf5:
            from .hdf5 import Hdf5FrameSource

            def factory() -> Hdf5FrameSource:
                return Hdf5FrameSource(
                    source_path,
                    image_key=args.image_key,
                    fps=args.fps,
                    color_order=args.color_order or "rgb",
                )

        else:
            from .video import VideoFrameSource

            def factory() -> VideoFrameSource:
                return VideoFrameSource(source_path, stream_index=args.video_stream)

        print(f"输入：{source_path}", flush=True)
        launch_annotation(
            factory,
            annotations,
            port=args.port,
            open_browser=not args.no_browser,
            fitting_options=_fitting_options(args),
        )
    except ImportError as error:
        parser.exit(
            2,
            f"缺少标注工具依赖：{error}\n请安装：python -m pip install -e '.[trajectory-annotation]'\n",
        )
    except (OSError, ValueError) as error:
        parser.exit(2, f"无法启动标注工具：{error}\n")
