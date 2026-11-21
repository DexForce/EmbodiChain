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
"""Export aligned source/simulation videos using recorded frame mappings."""

from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    import av

__all__ = ["export_comparison", "create_video_stream", "encode_video_frame"]


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def export_comparison(source: Path, output: Path) -> Path:
    """Encode every render with its explicit nearest source observation frame.

    Args:
        source: Native input package with ``trajectory_refined.npz`` and
            ``original_front.mp4``; the NPZ supplies sample times and frame IDs.
        output: Completed replay directory with ``videos/simulation.mp4`` and
            ``diagnostics/render_times.json``.

    Returns:
        ``videos/comparison_full.mp4`` in the replay directory. Its frame mapping
        is also saved to ``diagnostics/comparison_full.json``.

    Raises:
        ValueError: Source frame IDs are negative/nonmonotonic, or simulation
            frame counts do not match the supplied render times.
    """
    import av

    with np.load(source / "trajectory_refined.npz", allow_pickle=False) as data:
        times = data["timestamps"] - data["timestamps"][0]
        frame_map = (
            data["source_frame_indices"]
            if "source_frame_indices" in data
            else np.arange(len(times))
        )
    render_times = np.asarray(
        _read(output / "diagnostics/render_times.json")["timestamps"]
    )
    right = np.clip(np.searchsorted(times, render_times), 0, len(times) - 1)
    left = np.maximum(0, right - 1)
    samples = np.where(
        abs(times[left] - render_times) <= abs(times[right] - render_times), left, right
    )
    indices = np.asarray(frame_map[samples], dtype=np.int64)
    if np.any(indices < 0) or np.any(np.diff(indices) < 0):
        raise ValueError("Source frame mapping must be nonnegative and monotonic")
    target = output / "videos/comparison_full.mp4"
    with (
        av.open(str(source / "original_front.mp4")) as real,
        av.open(str(output / "videos/simulation.mp4")) as sim,
        av.open(str(target), "w") as dst,
    ):
        real.streams.video[0].codec_context.thread_count = 2
        sim.streams.video[0].codec_context.thread_count = 2
        stream = create_video_stream(dst, 1280, 480)
        iterator, current, image = iter(real.decode(video=0)), -1, None
        count = 0
        for frame, wanted in zip(sim.decode(video=0), indices, strict=True):
            while current < wanted:
                image = (
                    next(iterator)
                    .reformat(width=640, height=480)
                    .to_ndarray(format="rgb24")
                )
                current += 1
            rgb = np.concatenate([image, frame.to_ndarray(format="rgb24")], axis=1)
            encode_video_frame(dst, stream, rgb, count)
            count += 1
        for packet in stream.encode():
            dst.mux(packet)
    _write(
        output / "diagnostics/comparison_full.json",
        dict(
            source_frame_indices=indices.tolist(),
            trajectory_sample_indices=samples.tolist(),
            render_times_s=render_times.tolist(),
            frames=count,
        ),
    )
    return target


def create_video_stream(
    video: av.container.OutputContainer, width: int, height: int
) -> av.video.stream.VideoStream:
    """Create a 10 Hz H.264 stream using two encoder threads.

    Args:
        video: Open output container owned by the caller.
        width: Output frame width in pixels.
        height: Output frame height in pixels.

    Returns:
        Configured stream; the caller encodes frames, flushes and closes it.
    """
    stream = video.add_stream("libx264", rate=10)
    stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
    stream.codec_context.thread_count = 2
    stream.options = {"crf": "18", "preset": "fast"}
    return stream


def encode_video_frame(
    video: av.container.OutputContainer,
    stream: av.video.stream.VideoStream,
    rgb: np.ndarray,
    index: int,
) -> None:
    """Encode and mux one RGB frame on the 10 Hz comparison video clock.

    Args:
        video: Open destination container.
        stream: Stream returned by :func:`create_video_stream`.
        rgb: RGB uint8 array shaped ``(height, width, 3)``.
        index: Zero-based output frame index, used as presentation timestamp.
    """
    import av

    frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24")
    frame.pts, frame.time_base = index, Fraction(1, 10)
    for packet in stream.encode(frame):
        video.mux(packet)
