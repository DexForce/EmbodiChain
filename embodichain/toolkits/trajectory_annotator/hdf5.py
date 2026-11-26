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

"""Read-only, on-demand HDF5 image sequence adapter."""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
import threading
from typing import Any

import cv2
import h5py
import numpy as np
from numpy.typing import NDArray

from ._file import file_signature, hash_file, source_identity
from .frames import FrameSourceInfo, ImageFrame, _positive_fps, _rgb_image

__all__ = ["Hdf5FrameSource"]


class Hdf5FrameSource:
    """Adapt compressed image bytes or an N×H×W×3 uint8 image dataset.

    Args:
        path: Local HDF5 file, opened read-only on the annotation worker.
        image_key: Dataset path. When omitted, resolve cam_high under either
            ``observations/images`` or ``observation/images``. Ambiguity is an error.
        fps: Explicit playback rate. Otherwise consult the image dataset's fps,
            root fps, then root ``metadata_json.record_fps``. No rate is guessed.
        color_order: ``rgb`` or ``bgr`` for raw arrays. Compressed JPEG/PNG images
            always use their encoded color definition.

    Time is derived as ``index / fps``; this adapter does not claim recorded
    per-image timestamps. Preparation hashes the file and decodes only its first
    selected image. Subsequent reads load a single selected-camera image.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        image_key: str | None = None,
        fps: float | None = None,
        color_order: str = "rgb",
    ) -> None:
        self.path = Path(path).resolve()
        self.image_key = image_key.strip("/") if image_key is not None else None
        self._requested_fps = _positive_fps(fps) if fps is not None else None
        if color_order not in {"rgb", "bgr"}:
            raise ValueError("color_order 必须为 rgb 或 bgr。")
        self._color_order = color_order
        self._handle: h5py.File | None = None
        self._dataset: h5py.Dataset | None = None
        self._stat: tuple[int, int, int, int] | None = None
        self._cancelled = threading.Event()
        self._info: FrameSourceInfo | None = None
        self._raw = False

    def _read_image(self, index: int) -> NDArray[np.uint8]:
        if self._cancelled.is_set():
            raise InterruptedError("读取已取消。")
        if self._dataset is None:
            raise RuntimeError("HDF5 帧源尚未准备。")
        item = self._dataset[index]
        if self._raw:
            image = _rgb_image(item)
            return image if self._color_order == "rgb" else image[:, :, ::-1].copy()
        if isinstance(item, (bytes, np.bytes_)):
            encoded = np.frombuffer(item, dtype=np.uint8)
        elif isinstance(item, np.ndarray) and item.dtype == np.uint8 and item.ndim == 1:
            encoded = item
        else:
            raise ValueError("压缩图像必须为逐帧 uint8 字节数组或 bytes。")
        if not encoded.size:
            raise ValueError(f"HDF5 第 {index} 帧为空。")
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if image is None:
            raise ValueError(f"HDF5 第 {index} 帧无法解码为 JPEG/PNG 图像。")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _read_fps(self) -> float:
        if self._requested_fps is not None:
            return self._requested_fps
        assert self._dataset is not None and self._handle is not None
        rate: Any = self._dataset.attrs.get("fps", self._handle.attrs.get("fps"))
        if rate is None and "metadata_json" in self._handle.attrs:
            try:
                metadata = json.loads(self._handle.attrs["metadata_json"])
                rate = (
                    metadata.get("record_fps") if isinstance(metadata, dict) else None
                )
            except (ValueError, TypeError, UnicodeError) as error:
                raise ValueError(
                    "无法读取 HDF5 帧率元数据，请显式指定 --fps。"
                ) from error
        try:
            return _positive_fps(float(rate))
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("缺少有效的图像帧率，请显式指定 --fps。") from error

    def prepare(
        self, cancelled: threading.Event, progress: Callable[..., None]
    ) -> FrameSourceInfo:
        """Open one image dataset and hash the file without loading the sequence.

        Args:
            cancelled: Cooperative cancellation signal for preparation and reads.
            progress: Callback accepting phase/completed/total keywords.

        Returns:
            Selected-camera identity, geometry and FPS-derived timing metadata.

        Raises:
            ValueError: Dataset selection, image format or FPS is unsupported.
        """
        self.close()
        self._cancelled = cancelled
        self._stat = file_signature(self.path)
        self._handle = h5py.File(self.path, "r")
        if self.image_key is None:
            candidates = [
                key
                for key in (
                    "observations/images/cam_high",
                    "observation/images/cam_high",
                )
                if key in self._handle
            ]
            if len(candidates) != 1:
                raise ValueError(
                    "无法唯一定位 cam_high，请用 --image-key 指定图像 dataset。"
                )
            self.image_key = candidates[0]
        if self.image_key not in self._handle:
            raise ValueError(f"HDF5 中找不到图像路径：{self.image_key}。")
        dataset = self._handle[self.image_key]
        if (
            not isinstance(dataset, h5py.Dataset)
            or dataset.ndim == 0
            or not dataset.shape[0]
        ):
            raise ValueError("图像路径必须指向非空的逐帧 dataset。")
        if (
            not Path(dataset.file.filename).samefile(self.path)
            or dataset.is_virtual
            or dataset.external
        ):
            raise ValueError(
                "图像 dataset 必须存储在输入 HDF5 内，不支持外部存储或虚拟 dataset。"
            )
        self._dataset = dataset
        self._raw = (
            dataset.ndim == 4 and dataset.shape[-1] == 3 and dataset.dtype == np.uint8
        )
        if not self._raw and dataset.ndim not in {1, 2}:
            raise ValueError("支持 N×H×W×3 uint8 图像或 N 个 JPEG/PNG 字节数组。")
        fps = self._read_fps()
        image = self._read_image(0)
        height, width = image.shape[:2]
        sha256 = hash_file(self.path, cancelled, progress)
        details = dict(
            sha256=sha256,
            image_key=self.image_key,
            encoding="raw" if self._raw else "encoded",
            color_order=self._color_order if self._raw else "encoded",
            fps=fps,
        )
        self.check_unchanged()
        self._info = FrameSourceInfo(
            "hdf5",
            f"{self.path.name} / {self.image_key}",
            source_identity(details),
            dataset.shape[0],
            width,
            height,
            fps,
            "fixed_fps",
            details,
            self.path,
        )
        return self._info

    def get_frame(self, index: int) -> ImageFrame:
        """Decode one selected-camera image with zero-based FPS-derived timing.

        Args:
            index: Zero-based row in the selected image dataset.

        Returns:
            RGB pixels and FPS-derived time, with no synthesized PTS.

        Raises:
            ValueError: Index is out of bounds or the frame cannot be decoded.
        """
        if self._info is None:
            raise RuntimeError("HDF5 帧源尚未准备。")
        if type(index) is not int or not 0 <= index < self._info.frame_count:
            raise ValueError(f"帧号必须在 0–{self._info.frame_count - 1} 之间。")
        self.check_unchanged()
        image = self._read_image(index)
        if image.shape[:2] != (self._info.height, self._info.width):
            raise ValueError(f"HDF5 第 {index} 帧的分辨率与首帧不同。")
        assert self._info.fps is not None
        return ImageFrame(index, image, index / self._info.fps, 1 / self._info.fps)

    def check_unchanged(self) -> None:
        """Check only file stat fields, safe on the persistence thread."""
        if self._stat != file_signature(self.path):
            raise ValueError("HDF5 文件在打开后发生变化，请重新启动标注工具。")

    def close(self) -> None:
        """Close the worker-owned read-only HDF5 handle."""
        if self._handle is not None:
            self._handle.close()
        self._handle = self._dataset = None
        self._info = None
