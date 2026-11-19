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

"""Source-independent preview cache and bounded, single-owner frame worker."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
import json
import math
import threading
from typing import Any

import cv2

from ._cfg import AnnotationCfg
from .frames import FrameSource, FrameSourceInfo, _rgb_image

__all__ = ["FramePreview", "FrameReader", "FrameWorker", "SupersededError"]


class SupersededError(RuntimeError):
    """A newer pending seek replaced this request before reading started."""


@dataclass(frozen=True)
class FramePreview:
    """Owned JPEG bytes and timing for exactly one displayed image."""

    index: int
    pts: int | None
    time_seconds: float
    delay_seconds: float
    jpeg: bytes


class FrameReader:
    """Consume RGB images and share resizing/JPEG/LRU behavior across adapters."""

    def __init__(
        self,
        source: FrameSource,
        cfg: AnnotationCfg,
        cancelled: threading.Event | None = None,
    ) -> None:
        self.source = source
        self.cfg = cfg
        self.cancelled = cancelled or threading.Event()
        self.info: FrameSourceInfo | None = None
        self.cache: OrderedDict[int, FramePreview] = OrderedDict()
        self.cache_bytes = 0

    def prepare(self, progress: Callable[..., None] | None = None) -> FrameSourceInfo:
        """Prepare and validate metadata before accepting requests."""
        info = self.source.prepare(self.cancelled, progress or (lambda **_: None))
        if any(
            type(value) is not int or value <= 0
            for value in (info.frame_count, info.width, info.height)
        ):
            raise ValueError("帧源需要正整数帧数和固定图像尺寸。")
        if not info.identity or not info.kind or not info.name:
            raise ValueError("帧源需要非空 identity、kind 和 name。")
        if info.fps is not None and (not math.isfinite(info.fps) or info.fps <= 0):
            raise ValueError("帧源 FPS 必须为有限正数。")
        json.dumps(info.details, allow_nan=False)
        self.info = info
        return info

    def check_unchanged(self) -> None:
        """Run the adapter's thread-safe source identity check."""
        self.source.check_unchanged()

    def preview(self, index: int) -> FramePreview:
        """Return one exact indexed frame with owned immutable JPEG bytes."""
        if self.info is None:
            raise RuntimeError("帧源尚未准备。")
        if type(index) is not int or not 0 <= index < self.info.frame_count:
            raise ValueError(f"帧号必须在 0–{self.info.frame_count - 1} 之间。")
        if self.cancelled.is_set():
            raise InterruptedError("读取已取消。")
        self.check_unchanged()
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        frame = self.source.get_frame(index)
        image = _rgb_image(frame.image)
        if frame.index != index or image.shape[:2] != (
            self.info.height,
            self.info.width,
        ):
            raise ValueError("帧源返回的帧号或图像尺寸与请求不一致。")
        if (
            not math.isfinite(frame.time_seconds)
            or frame.time_seconds < 0
            or not math.isfinite(frame.delay_seconds)
            or frame.delay_seconds <= 0
        ):
            raise ValueError("帧源返回的显示时间或帧间隔无效。")
        scale = min(1.0, self.cfg.preview_edge / max(self.info.width, self.info.height))
        if scale < 1:
            image = cv2.resize(
                image,
                (
                    max(1, round(self.info.width * scale)),
                    max(1, round(self.info.height * scale)),
                ),
                interpolation=cv2.INTER_AREA,
            )
        success, encoded = cv2.imencode(
            ".jpg",
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, self.cfg.jpeg_quality],
        )
        if not success:
            raise RuntimeError("无法编码预览图像。")
        result = FramePreview(
            index, frame.pts, frame.time_seconds, frame.delay_seconds, encoded.tobytes()
        )
        if len(result.jpeg) <= self.cfg.cache_bytes:
            while (
                self.cache
                and self.cache_bytes + len(result.jpeg) > self.cfg.cache_bytes
            ):
                self.cache_bytes -= len(self.cache.popitem(last=False)[1].jpeg)
            self.cache[index] = result
            self.cache_bytes += len(result.jpeg)
        return result

    def close(self) -> None:
        """Release the source and all cached preview bytes on the owner thread."""
        self.source.close()
        self.cache.clear()
        self.cache_bytes = 0


class FrameWorker:
    """Own a frame source on one thread with one active and one pending request."""

    def __init__(
        self,
        source_factory: Callable[[], FrameSource],
        cfg: AnnotationCfg,
        on_ready: Callable[[FrameSourceInfo], None],
    ) -> None:
        self._cancelled = threading.Event()
        self._source_factory = source_factory
        self._cfg = cfg
        self.reader: FrameReader | None = None
        self._on_ready = on_ready
        self._condition = threading.Condition()
        self._status: dict[str, Any] = dict(
            phase="starting", completed=0, total=None, error=None
        )
        self._pending: tuple[int, Future[FramePreview]] | None = None
        self._interval: (
            tuple[int, int, Future[tuple[FramePreview, FramePreview]]] | None
        ) = None
        self._thread: threading.Thread | None = None

    def status(self) -> dict[str, Any]:
        """Return an owned initialization/lifecycle status snapshot."""
        with self._condition:
            return self._status.copy()

    def _report(self, **updates: Any) -> None:
        with self._condition:
            self._status.update(updates)

    def start(self) -> None:
        """Prepare the source on its sole owning worker thread."""
        self._thread = threading.Thread(
            target=self._run, name="trajectory-frames", daemon=True
        )
        self._thread.start()

    def request(self, index: int) -> Future[FramePreview]:
        """Replace a queued seek, bounding work even after HTTP cancellation."""
        future: Future[FramePreview] = Future()
        with self._condition:
            if self._status["phase"] != "ready" or self._cancelled.is_set():
                raise RuntimeError("帧源尚未就绪或已关闭。")
            if self._pending is not None:
                previous = self._pending[1]
                if not previous.done():
                    previous.set_exception(SupersededError("已被更新的寻帧请求替代。"))
            self._pending = (index, future)
            self._condition.notify()
        return future

    def interval(
        self, first: int, last: int
    ) -> Future[tuple[FramePreview, FramePreview]]:
        """Capture exact endpoint timing on the decoder owner, without a new index.

        One pending interval is permitted independently of the latest seek. A
        video scrub cannot supersede a fitting job's clock snapshot.

        Args:
            first: Included start frame index.
            last: Included end frame index.

        Returns:
            A future containing both owned endpoint previews and their timing.
        """
        future: Future[tuple[FramePreview, FramePreview]] = Future()
        with self._condition:
            if self._status["phase"] != "ready" or self._cancelled.is_set():
                raise RuntimeError("帧源尚未就绪或已关闭。")
            if self._interval is not None:
                raise RuntimeError("已有待处理的选区时间请求。")
            self._interval = (first, last, future)
            self._condition.notify()
        return future

    def _run(self) -> None:
        try:
            self.reader = FrameReader(
                self._source_factory(), self._cfg, self._cancelled
            )
            metadata = self.reader.prepare(self._report)
            self._check_running()
            self._on_ready(metadata)
            self._report(
                phase="ready",
                completed=metadata.frame_count,
                total=metadata.frame_count,
            )
            while not self._cancelled.is_set():
                with self._condition:
                    self._condition.wait_for(
                        lambda: self._pending is not None
                        or self._interval is not None
                        or self._cancelled.is_set()
                    )
                    interval, self._interval = self._interval, None
                    pending, self._pending = self._pending, None
                if interval is not None:
                    first, last, result = interval
                    if result.set_running_or_notify_cancel():
                        try:
                            result.set_result(
                                (self.reader.preview(first), self.reader.preview(last))
                            )
                        except Exception as error:
                            result.set_exception(error)
                if pending is None:
                    continue
                index, future = pending
                if not future.set_running_or_notify_cancel():
                    continue
                try:
                    future.set_result(self.reader.preview(index))
                except Exception as error:
                    future.set_exception(error)
        except InterruptedError:
            self._report(phase="cancelled")
        except Exception as error:
            self._report(phase="error", error=str(error))
        finally:
            try:
                if self.reader is not None:
                    self.reader.close()
            except Exception as error:
                self._report(phase="error", error=f"关闭帧源失败：{error}")
            finally:
                with self._condition:
                    if self._pending is not None and not self._pending[1].done():
                        self._pending[1].set_exception(RuntimeError("帧源已关闭。"))
                    self._pending = None
                    if self._interval is not None and not self._interval[2].done():
                        self._interval[2].set_exception(RuntimeError("帧源已关闭。"))
                    self._interval = None

    def _check_running(self) -> None:
        if self._cancelled.is_set():
            raise InterruptedError("初始化已取消。")

    def close(self) -> None:
        """Cancel initialization/decoding and join its sole owning thread."""
        self._cancelled.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join()
        if self.status()["phase"] not in {"error", "cancelled"}:
            self._report(phase="closed")
