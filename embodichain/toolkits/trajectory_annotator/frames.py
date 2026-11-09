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

"""Container-independent RGB frame input for trajectory annotation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path
import threading
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

__all__ = ["ImageFrame", "FrameSourceInfo", "FrameSource", "ArrayFrameSource"]


@dataclass(frozen=True)
class ImageFrame:
    """One indexed RGB uint8 image and its display timing.

    Attributes:
        index: Zero-based position in the source sequence.
        image: H×W×3 RGB pixels, valid until the next source read or close.
        time_seconds: Nonnegative display time relative to the first image.
        delay_seconds: Positive duration before displaying the next image.
        pts: Optional original presentation timestamp; never synthesized.
    """

    index: int
    image: NDArray[np.uint8]
    time_seconds: float
    delay_seconds: float
    pts: int | None = None


@dataclass(frozen=True)
class FrameSourceInfo:
    """Prepared source identity and fixed image geometry.

    Attributes:
        kind: Adapter name, e.g. ``video``, ``hdf5`` or ``images``.
        name: Human-readable source name.
        identity: Stable identifier covering content, selection and timing.
        frame_count: Number of images, greater than zero.
        width: Original image width.
        height: Original image height.
        fps: Nominal playback rate, or None when unavailable.
        timing: ``pts`` for measured video timing or ``fixed_fps`` for derived time.
        details: JSON-serializable provenance such as image key or stream index.
        path: Optional backing file, protected against annotation overwrite.

    Names and paths are descriptive; changing image order, pixels, timing or
    camera selection must change ``identity``. Adapters must not retain all
    decoded images merely to compute this information.
    """

    kind: str
    name: str
    identity: str
    frame_count: int
    width: int
    height: int
    fps: float | None
    timing: str
    details: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None


class FrameSource(Protocol):
    """Random-access image input owned by the application's single worker.

    The app calls the source factory, prepare, get_frame and close on the same
    worker thread. A factory must return a fresh instance for each application
    lifespan. Only ``check_unchanged`` may also run on a persistence thread; it
    must use immutable metadata/stat checks, never an open decoder/HDF5 handle.
    """

    def prepare(
        self,
        cancelled: threading.Event,
        progress: Callable[..., None],
    ) -> FrameSourceInfo:
        """Prepare identity and geometry, reporting phase/completed/total keywords.

        Args:
            cancelled: Cooperative cancellation signal, checked during scans.
            progress: Callback accepting phase, completed and total keyword arguments.

        Returns:
            Complete source metadata before any frame requests are accepted.
        """
        ...

    def get_frame(self, index: int) -> ImageFrame:
        """Read exactly one indexed RGB image with its display timing.

        Args:
            index: Integer in ``[0, frame_count - 1]``.

        Returns:
            Pixels and timing belonging to the requested index.
        """
        ...

    def check_unchanged(self) -> None:
        """Raise ValueError when the prepared source has changed."""
        ...

    def close(self) -> None:
        """Release source resources, including after failed preparation."""
        ...


def _rgb_image(image: object) -> NDArray[np.uint8]:
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or min(image.shape[:2]) <= 0
    ):
        raise ValueError("图像必须为 H×W×3 的 RGB uint8 数组。")
    return image


def _positive_fps(fps: float) -> float:
    if isinstance(fps, bool) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("帧率必须为有限的正数。")
    return float(fps)


class ArrayFrameSource:
    """Adapt an existing RGB image sequence without stacking or copying the batch.

    Args:
        frames: Sequence of H×W×3 uint8 RGB arrays, or an N×H×W×3 array.
        fps: Positive rate used to derive display timing, not measured timestamps.
        name: Display name stored in annotation provenance.

    .. attention::
        The caller owns the input and must keep pixels, order and length
        unchanged until the app closes. Preparation hashes one image at a time;
        this adapter does not take a second full copy of the sequence.
    """

    def __init__(
        self,
        frames: Sequence[NDArray[np.uint8]] | NDArray[np.uint8],
        *,
        fps: float,
        name: str = "images",
    ) -> None:
        self._frames = frames
        self._fps = _positive_fps(fps)
        self._name = name
        self._count = 0
        self._cancelled = threading.Event()

    def prepare(
        self, cancelled: threading.Event, progress: Callable[..., None]
    ) -> FrameSourceInfo:
        """Validate fixed geometry and hash pixels with bounded temporary storage.

        Args:
            cancelled: Cooperative cancellation signal during hashing.
            progress: Callback accepting phase/completed/total keywords.

        Returns:
            Pixel identity, geometry and FPS-derived timing metadata.
        """
        self._cancelled = cancelled
        self._count = len(self._frames)
        if self._count == 0:
            raise ValueError("图像序列不能为空。")
        first = _rgb_image(self._frames[0])
        height, width = first.shape[:2]
        digest = hashlib.sha256(
            f"RGB:{self._count}:{width}:{height}:{self._fps}".encode()
        )
        for index in range(self._count):
            if cancelled.is_set():
                raise InterruptedError("初始化已取消。")
            image = _rgb_image(self._frames[index])
            if image.shape != first.shape:
                raise ValueError("图像序列的分辨率必须一致。")
            digest.update(memoryview(np.ascontiguousarray(image)))
            if index % 64 == 0 or index + 1 == self._count:
                progress(phase="indexing", completed=index + 1, total=self._count)
        self.check_unchanged()
        return FrameSourceInfo(
            "images",
            self._name,
            digest.hexdigest(),
            self._count,
            width,
            height,
            self._fps,
            "fixed_fps",
            {"color_order": "rgb"},
        )

    def get_frame(self, index: int) -> ImageFrame:
        """Return borrowed RGB pixels and FPS-derived time for one image.

        Args:
            index: Zero-based position in the prepared sequence.

        Returns:
            Caller-owned RGB image and timing for the requested index.
        """
        if self._cancelled.is_set():
            raise InterruptedError("读取已取消。")
        self.check_unchanged()
        if type(index) is not int or not 0 <= index < self._count:
            raise ValueError(f"帧号必须在 0–{self._count - 1} 之间。")
        return ImageFrame(
            index, _rgb_image(self._frames[index]), index / self._fps, 1 / self._fps
        )

    def check_unchanged(self) -> None:
        """Check sequence length; pixel immutability remains the caller's contract."""
        if len(self._frames) != self._count:
            raise ValueError("图像序列长度在准备后发生变化，请重新打开。")

    def close(self) -> None:
        """Leave the caller-owned image sequence intact."""
