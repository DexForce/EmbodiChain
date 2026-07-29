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

"""Readers for compressed depth sidecar videos.

Decodes the ``gray12le``/HEVC MP4s written by :class:`DepthSidecarManager` and
dequantizes them back to metres or millimetres. ``load_depth_dataset`` composes
a LeRobot dataset with its depth sidecar library so callers can read RGB, state,
action, mask and depth together.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import av
import numpy as np

from embodichain.utils import logger

from .depth_utils import DEPTH_METER_UNIT, dequantize_depth

__all__ = ["DepthVideoReader", "DepthVideoLibrary", "load_depth_dataset"]


class DepthVideoReader:
    """Decode and dequantize a single depth sidecar MP4.

    Frames are decoded lazily on first access and cached, since one MP4 holds a
    single (typically short) episode.
    """

    def __init__(
        self,
        path: Path,
        depth_min: float,
        depth_max: float,
        shift: float,
        use_log: bool,
        pix_fmt: str = "gray12le",
        output_unit: str = DEPTH_METER_UNIT,
    ) -> None:
        """Initialize the reader.

        Args:
            path: Path to the sidecar MP4.
            depth_min: Quantization ``depth_min`` (metres) used at write time.
            depth_max: Quantization ``depth_max`` (metres) used at write time.
            shift: Quantization ``shift`` (metres) used at write time.
            use_log: Quantization ``use_log`` used at write time.
            pix_fmt: Pixel format of the stored video.
            output_unit: Unit to return (``"m"`` or ``"mm"``).
        """
        self._path = Path(path)
        self._depth_min = depth_min
        self._depth_max = depth_max
        self._shift = shift
        self._use_log = use_log
        self._pix_fmt = pix_fmt
        self._output_unit = output_unit
        self._codes: Optional[list[np.ndarray]] = None

    def _decode(self) -> list[np.ndarray]:
        if self._codes is not None:
            return self._codes
        if not self._path.exists():
            raise FileNotFoundError(f"Depth sidecar video not found: {self._path}")
        codes: list[np.ndarray] = []
        with av.open(str(self._path), "r") as inp:
            stream = next((s for s in inp.streams if s.type == "video"), None)
            if stream is None:
                raise ValueError(f"No video stream in {self._path}")
            for frame in inp.decode(stream):
                codes.append(frame.to_ndarray(format=self._pix_fmt))
        self._codes = codes
        return codes

    @property
    def frame_count(self) -> int:
        """Number of frames in the sidecar video."""
        return len(self._decode())

    def read(self, frame_index: int) -> np.ndarray:
        """Decode and dequantize one frame.

        Args:
            frame_index: Within-episode frame index.

        Returns:
            Depth map in the configured output unit, shape ``(1, H, W)``.
        """
        codes = self._decode()
        if frame_index < 0 or frame_index >= len(codes):
            raise IndexError(
                f"Depth frame index {frame_index} out of range [0, {len(codes)})"
            )
        return dequantize_depth(
            codes[frame_index],
            depth_min=self._depth_min,
            depth_max=self._depth_max,
            shift=self._shift,
            use_log=self._use_log,
            pix_fmt=self._pix_fmt,
            output_unit=self._output_unit,
            output_tensor=False,
        )


class DepthVideoLibrary:
    """Index of all depth sidecar videos for a dataset.

    Maps ``(episode_index, sensor_key)`` to a :class:`DepthVideoReader` and
    provides random access by within-episode frame index.
    """

    def __init__(self, dataset_root: Path, meta: Dict[str, Any]) -> None:
        """Initialize the library from parsed metadata.

        Args:
            dataset_root: Root directory of the LeRobot dataset.
            meta: Parsed ``depth_meta.json`` contents.
        """
        self._root = Path(dataset_root)
        self._meta = meta
        self._readers: Dict[tuple[int, str], DepthVideoReader] = {}

    @property
    def sensors(self) -> list[str]:
        """List of depth sensor keys recorded in the dataset."""
        return list(self._meta.get("sensors", {}).keys())

    def sensor_meta(self, sensor_key: str) -> Dict[str, Any]:
        """Return the static metadata block for a sensor."""
        return self._meta["sensors"][sensor_key]

    def _reader(self, episode_index: int, sensor_key: str) -> DepthVideoReader:
        key = (episode_index, sensor_key)
        if key in self._readers:
            return self._readers[key]
        sensors = self._meta.get("sensors", {})
        if sensor_key not in sensors:
            raise KeyError(f"Unknown depth sensor '{sensor_key}'")
        sensor_meta = sensors[sensor_key]
        episodes = sensor_meta.get("episodes", {})
        ep_meta = episodes.get(str(episode_index))
        if ep_meta is None:
            raise KeyError(
                f"No depth video for sensor '{sensor_key}' episode {episode_index}"
            )
        reader = DepthVideoReader(
            path=self._root / ep_meta["file"],
            depth_min=sensor_meta["video.depth_min"],
            depth_max=sensor_meta["video.depth_max"],
            shift=sensor_meta["video.shift"],
            use_log=sensor_meta["video.use_log"],
            pix_fmt=sensor_meta["video.pix_fmt"],
            output_unit=sensor_meta["video.output_unit"],
        )
        self._readers[key] = reader
        return reader

    def get(
        self, episode_index: int, sensor_key: str, frame_index_in_episode: int
    ) -> np.ndarray:
        """Read one depth frame.

        Args:
            episode_index: Global episode index.
            sensor_key: Sensor identifier (e.g. ``"camera"``, ``"camera_right"``).
            frame_index_in_episode: Frame index within the episode.

        Returns:
            Depth map in the unit recorded in metadata, shape ``(1, H, W)``.
        """
        return self._reader(episode_index, sensor_key).read(frame_index_in_episode)

    def frame_count(self, episode_index: int, sensor_key: str) -> int:
        """Return the number of depth frames for an episode/sensor."""
        return self._reader(episode_index, sensor_key).frame_count


def load_depth_meta(dataset_root: Path) -> Dict[str, Any]:
    """Load and return the ``depth_meta.json`` for a dataset.

    Args:
        dataset_root: Root directory of the LeRobot dataset.

    Returns:
        Parsed metadata dict.

    Raises:
        FileNotFoundError: If no depth sidecar metadata exists.
    """
    meta_path = Path(dataset_root) / "depth_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No depth_meta.json at {meta_path}; the dataset has no depth sidecar videos."
        )
    with open(meta_path) as f:
        return json.load(f)


def load_depth_dataset(
    dataset_root: Path, **lerobot_kwargs: Any
) -> tuple[Any, DepthVideoLibrary]:
    """Load a LeRobot dataset together with its depth sidecar library.

    Args:
        dataset_root: Root directory of the LeRobot dataset (the directory
            containing ``data/``, ``videos/`` and ``depth_meta.json``).
        **lerobot_kwargs: Forwarded to ``LeRobotDataset`` (e.g. ``episodes``).

    Returns:
        A ``(LeRobotDataset, DepthVideoLibrary)`` tuple.

    Raises:
        ImportError: If LeRobot is not installed.
        FileNotFoundError: If the dataset has no depth sidecar metadata.
    """
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as e:
        raise ImportError(
            "LeRobot is not installed; install it with `pip install lerobot`."
        ) from e

    root = Path(dataset_root)
    meta = load_depth_meta(root)
    dataset_name = root.name
    dataset = LeRobotDataset(repo_id=dataset_name, root=str(root), **lerobot_kwargs)
    library = DepthVideoLibrary(root, meta)
    logger.log_info(
        f"Loaded depth sidecar library: {len(library.sensors)} sensor(s) "
        f"({', '.join(library.sensors) or 'none'})"
    )
    return dataset, library
