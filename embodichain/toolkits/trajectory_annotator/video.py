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

"""PyAV adapter preserving exact presentation-order frame indices."""

from __future__ import annotations

from array import array
from bisect import bisect_left
from collections.abc import Callable, Iterator
from fractions import Fraction
import hashlib
from pathlib import Path
import threading
from typing import Any

import av

from ._cfg import AnnotationCfg
from ._file import source_identity
from .frames import FrameSourceInfo, ImageFrame

__all__ = ["VideoFrameSource"]


class VideoFrameSource:
    """Index and decode on one thread; stat-only identity checks are thread-safe.

    Args:
        path: Local video file.
        decoder_threads: FFmpeg threads for this decoder.
        index_bytes: Maximum packed PTS and keyframe-index bytes.
        stream_index: Actual container stream index, or the first video stream.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        stream_index: int | None = None,
        decoder_threads: int = 2,
        index_bytes: int = 64 * 1024**2,
    ) -> None:
        self.path = Path(path).resolve()
        self.cfg = AnnotationCfg(
            decoder_threads=decoder_threads, index_bytes=index_bytes
        )
        self.stream_index = stream_index
        self.cancelled = threading.Event()
        self.pts = array("q")
        self.anchors = array("q")
        self.metadata: dict[str, Any] = {}
        self._container: Any = None
        self._stream: Any = None
        self._iterator: Iterator[Any] | None = None
        self._cursor = -1
        self._stat: tuple[int, int, int] | None = None
        self._time_base = Fraction(1, 1)

    def _check_cancelled(self) -> None:
        if self.cancelled.is_set():
            raise InterruptedError("视频初始化或解码已取消。")

    def _signature(self) -> tuple[int, int, int]:
        stat = self.path.stat()
        return stat.st_ino, stat.st_size, stat.st_mtime_ns

    def check_unchanged(self) -> None:
        """Reject use of an index after its source file was replaced or modified."""
        if self._stat != self._signature():
            raise ValueError("视频文件在打开后发生变化，请重新启动标注工具。")

    def _open(self) -> None:
        if self._container is not None:
            self._container.close()
        self._container = av.open(str(self.path), options={"err_detect": "explode"})
        streams = list(self._container.streams.video)
        if not streams:
            raise ValueError("输入文件中没有视频流。")
        self._stream = next(
            (stream for stream in streams if stream.index == self.stream_index), None
        )
        if self.stream_index is None:
            self._stream = streams[0]
            self.stream_index = self._stream.index
        if self._stream is None:
            raise ValueError(
                f"找不到视频流 {self.stream_index}，可选：{[stream.index for stream in streams]}。"
            )
        self._stream.codec_context.thread_count = self.cfg.decoder_threads
        self._iterator = None
        self._cursor = -1

    def prepare(
        self, cancelled: threading.Event, progress: Callable[..., None]
    ) -> FrameSourceInfo:
        """Hash and fully index PTS/anchors with bounded storage and cancellation.

        Args:
            cancelled: Cooperative cancellation event.
            progress: Callback receiving phase/completed/total keyword arguments.

        Returns:
            Identity, geometry and timing of the selected video stream.
        """
        self.cancelled = cancelled
        report = progress
        self._stat = self._signature()
        if not self.path.is_file():
            raise ValueError("视频路径必须为普通文件。")
        digest = hashlib.sha256()
        completed = 0
        report(phase="hashing", completed=0, total=self._stat[1])
        with self.path.open("rb") as handle:
            while chunk := handle.read(1024**2):
                self._check_cancelled()
                digest.update(chunk)
                completed += len(chunk)
                report(phase="hashing", completed=completed, total=self._stat[1])
        self._open()
        report(phase="indexing", completed=0, total=self._stream.frames or None)
        self.pts = array("q")
        self.anchors = array("q")
        anchor = 0
        dimensions: tuple[int, int] | None = None
        for frame in self._container.decode(self._stream):
            self._check_cancelled()
            if frame.pts is None or (self.pts and frame.pts <= self.pts[-1]):
                raise ValueError("视频帧时间戳缺失、重复或非递增；首版需要可靠的 PTS。")
            if dimensions is None:
                dimensions = (frame.width, frame.height)
            if dimensions != (frame.width, frame.height):
                raise ValueError("首版不支持中途改变分辨率的视频。")
            if (len(self.pts) + 1) * (
                self.pts.itemsize + self.anchors.itemsize
            ) > self.cfg.index_bytes:
                raise ValueError("帧索引超过配置的内存预算。")
            if frame.key_frame:
                anchor = len(self.pts)
            self.pts.append(frame.pts)
            self.anchors.append(anchor)
            if len(self.pts) % 64 == 0:
                report(
                    phase="indexing",
                    completed=len(self.pts),
                    total=self._stream.frames or None,
                )
        if not self.pts or dimensions is None:
            raise ValueError("视频没有可解码的画面。")
        self._time_base = Fraction(self._stream.time_base)
        fps = self._stream.average_rate
        if fps is not None and fps <= 0:
            fps = None
        self.metadata = dict(
            sha256=digest.hexdigest(),
            stream_index=self.stream_index,
            frame_count=len(self.pts),
            width=dimensions[0],
            height=dimensions[1],
            time_base=dict(
                numerator=self._time_base.numerator,
                denominator=self._time_base.denominator,
            ),
            average_fps=(
                dict(numerator=fps.numerator, denominator=fps.denominator)
                if fps
                else None
            ),
        )
        self.check_unchanged()
        self._open()
        return FrameSourceInfo(
            "video",
            self.path.name,
            source_identity(self.metadata),
            len(self.pts),
            dimensions[0],
            dimensions[1],
            float(fps) if fps else None,
            "pts",
            self.metadata.copy(),
            self.path,
        )

    def _walk(self, index: int, expected: int | None = None) -> Any:
        if self._iterator is None:
            self._iterator = iter(self._container.decode(self._stream))
        for frame in self._iterator:
            self._check_cancelled()
            found = (
                bisect_left(self.pts, frame.pts)
                if frame.pts is not None
                else len(self.pts)
            )
            if (
                found >= len(self.pts)
                or self.pts[found] != frame.pts
                or (expected is not None and found != expected)
                or found > index
            ):
                raise ValueError("关键帧定位后的画面无法与完整索引对齐。")
            self._cursor = found
            expected = found + 1
            if found == index:
                return frame
        raise ValueError(f"无法解码目标帧 {index}。")

    def _frame(self, index: int) -> Any:
        try:
            if self._iterator is not None and self._cursor < index <= self._cursor + 64:
                return self._walk(index, self._cursor + 1)
            self._container.seek(
                self.pts[self.anchors[index]],
                stream=self._stream,
                backward=True,
                any_frame=False,
            )
            self._iterator = None
            return self._walk(index)
        except (ValueError, av.error.FFmpegError):
            # Reopen instead of seek(0): the first source PTS need not be zero,
            # and broken container seek tables must not alter frame numbering.
            self._open()
            return self._walk(index, 0)

    def get_frame(self, index: int) -> ImageFrame:
        """Read exactly one RGB image using indexed PTS, with sequential fallback.

        Args:
            index: Zero-based presentation-order frame position.

        Returns:
            Requested RGB pixels and original PTS-based timing.
        """
        if type(index) is not int or not 0 <= index < len(self.pts):
            raise ValueError(f"帧号必须在 0–{len(self.pts) - 1} 之间。")
        self._check_cancelled()
        self.check_unchanged()
        frame = self._frame(index)
        if len(self.pts) > 1:
            offset = min(index, len(self.pts) - 2)
            delay = float((self.pts[offset + 1] - self.pts[offset]) * self._time_base)
        else:
            fps = self._stream.average_rate
            delay = float(1 / fps) if fps and fps > 0 else 1 / 30
        return ImageFrame(
            index,
            frame.to_ndarray(format="rgb24"),
            float((self.pts[index] - self.pts[0]) * self._time_base),
            delay,
            self.pts[index],
        )

    def close(self) -> None:
        """Release decoder buffers on the owning thread."""
        if self._container is not None:
            self._container.close()
            self._container = None
        self._iterator = None
