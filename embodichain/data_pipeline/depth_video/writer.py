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

"""Streaming depth video writer and per-episode sidecar manager.

Writes camera depth maps as ``gray12le``/HEVC sidecar videos that live
alongside a LeRobot dataset (issue #424, Path A). Depth never enters LeRobot's
own image/video pipeline (which is RGB-only in 0.4.4); instead each episode is
encoded to a standalone MP4 and indexed by ``depth_meta.json``.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import av
import numpy as np
import torch

from embodichain.utils import logger

from .cfg import DepthVideoCfg, EMBODI_DEPTH_CODEBASE_VERSION
from .codec import resolve_depth_vcodec
from .depth_utils import DEPTH_QMAX, quantize_depth

__all__ = ["DepthVideoWriter", "DepthSidecarManager"]

# One episode per file, grouped under a single chunk for now. The metadata
# records ``chunk_index`` so a future migration can re-chunk if needed.
_CHUNK_INDEX = 0
_META_FILENAME = "depth_meta.json"


def _to_numpy_uint16(depth: Any) -> np.ndarray:
    """Coerce a depth frame (tensor or array) to a 2D float array for quantization.

    Args:
        depth: ``torch.Tensor`` or ``np.ndarray`` depth map.

    Returns:
        Depth as a ``np.ndarray`` (dtype preserved; ``quantize_depth`` handles
        float32/uint16 inference).
    """
    if isinstance(depth, torch.Tensor):
        depth = depth.detach().cpu().numpy()
    return np.asarray(depth)


class DepthVideoWriter:
    """Streaming writer that encodes one depth episode to a single MP4.

    Depth frames are quantized to 12-bit codes and encoded as ``gray12le``
    video. With ``lossless=True`` the 12-bit codes are bit-exact after a
    decode round-trip (verified on pyav 15.x + libx265).

    The video is written to a temporary path and atomically moved to its final
    location on :meth:`close`; on failure the partial file is removed so the
    dataset never references a corrupt sidecar.
    """

    def __init__(
        self,
        final_path: Path,
        fps: int,
        cfg: DepthVideoCfg,
        vcodec: Optional[str] = None,
    ) -> None:
        """Initialize the writer.

        Args:
            final_path: Destination MP4 path. Written via a temp file in the
                same directory and atomically renamed on success.
            fps: Episode frame rate (frames per second).
            cfg: Depth video configuration.
            vcodec: Resolved codec name. If ``None``, resolved from ``cfg``.
        """
        self._final_path = Path(final_path)
        self._fps = int(fps)
        self._cfg = cfg
        self._vcodec = vcodec or resolve_depth_vcodec(cfg.vcodec)
        self._temp_path = Path(
            tempfile.mkstemp(
                prefix=".depth_tmp_", suffix=".mp4", dir=str(self._final_path.parent)
            )[1]
        )
        self._output: Optional[av.output.OutputContainer] = None
        self._stream: Optional[av.video.stream.VideoStream] = None
        self._width: int = 0
        self._height: int = 0
        self._frame_count: int = 0
        self._closed: bool = False

    def _ensure_open(self, depth: np.ndarray) -> None:
        if self._output is not None:
            return
        # Infer spatial shape from the (squeezed) depth frame.
        h, w = int(depth.shape[-2]), int(depth.shape[-1])
        self._height, self._width = h, w
        self._final_path.parent.mkdir(parents=True, exist_ok=True)
        self._output = av.open(str(self._temp_path), "w")
        stream = self._output.add_stream(self._vcodec, self._fps)
        stream.pix_fmt = self._cfg.pix_fmt
        stream.width = w
        stream.height = h
        # libx265: ``bframes=0`` makes the encoder flush deterministic for short
        # episodes -- with the default B-frame/lookahead buffering, flushing 1-3
        # frame videos races inside libx265's thread pool and intermittently
        # raises ``PatchWelcomeError``. ``log-level=error`` silences the per-encode
        # x265 info banner. Lossless mode preserves the 12-bit codes bit-exactly.
        if self._vcodec == "libx265":
            if self._cfg.lossless:
                stream.options = {
                    "x265-params": "lossless=1:bframes=0:log-level=error",
                    "crf": "0",
                }
            else:
                stream.options = {
                    "crf": str(int(self._cfg.crf)),
                    "x265-params": "bframes=0:log-level=error",
                }
        else:
            stream.options = {"crf": str(int(self._cfg.crf))}
        self._stream = stream

    def add_frame(self, depth: Any) -> None:
        """Quantize and encode a single depth frame.

        Args:
            depth: Depth map (``torch.Tensor`` or ``np.ndarray``), shape
                ``(H, W)`` / ``(H, W, 1)`` / ``(1, H, W)``.
        """
        if self._closed:
            logger.log_error("DepthVideoWriter is closed; cannot add_frame.")
            return
        depth_np = _to_numpy_uint16(depth)
        self._ensure_open(depth_np)
        frame = quantize_depth(
            depth_np,
            depth_min=self._cfg.depth_min,
            depth_max=self._cfg.depth_max,
            shift=self._cfg.shift,
            use_log=self._cfg.use_log,
            pix_fmt=self._cfg.pix_fmt,
            input_unit=self._cfg.input_unit,
        )
        assert self._stream is not None and self._output is not None
        for packet in self._stream.encode(frame):
            self._output.mux(packet)
        self._frame_count += 1

    def close(self) -> Path:
        """Flush the encoder and finalize the MP4.

        Returns:
            The final MP4 path on success.

        Raises:
            RuntimeError: If no frames were written.
        """
        if self._closed:
            return self._final_path
        self._closed = True
        try:
            if self._output is None or self._stream is None:
                raise RuntimeError(
                    "DepthVideoWriter closed with no frames written; no MP4 produced."
                )
            for packet in self._stream.encode():
                self._output.mux(packet)
            self._output.close()
            if self._frame_count == 0:
                raise RuntimeError("DepthVideoWriter produced 0 frames.")
            # Atomic-ish move into place.
            self._temp_path.replace(self._final_path)
            return self._final_path
        except Exception:
            # Cleanup partial output so the dataset never references a corrupt file.
            self._discard_temp()
            raise

    @property
    def frame_count(self) -> int:
        """Number of frames written so far."""
        return self._frame_count

    @property
    def final_path(self) -> Path:
        """Destination MP4 path."""
        return self._final_path

    def _discard_temp(self) -> None:
        if self._temp_path.exists():
            try:
                self._temp_path.unlink()
            except OSError:
                pass

    def abort(self) -> None:
        """Abort writing and remove any partial output."""
        self._closed = True
        if self._output is not None:
            try:
                self._output.close()
            except Exception:
                pass
        self._discard_temp()


class DepthSidecarManager:
    """Owns the depth sidecar videos and metadata for one dataset recording.

    For each episode, opens one :class:`DepthVideoWriter` per depth sensor
    (e.g. ``camera`` and ``camera_right``), feeds frames, and on episode close
    moves the MP4s into the dataset's ``depth_videos/`` tree and updates
    ``depth_meta.json``.
    """

    def __init__(
        self,
        dataset_root: Path,
        fps: int,
        cfg: DepthVideoCfg,
        vcodec: Optional[str] = None,
    ) -> None:
        """Initialize the sidecar manager.

        Args:
            dataset_root: Root directory of the LeRobot dataset. Sidecar videos
                are written under ``<root>/depth_videos/`` and metadata under
                ``<root>/depth_meta.json``.
            fps: Dataset frame rate.
            cfg: Depth video configuration.
            vcodec: Resolved codec name. If ``None``, resolved from ``cfg``.
        """
        self._root = Path(dataset_root)
        self._fps = int(fps)
        self._cfg = cfg
        self._vcodec = vcodec or resolve_depth_vcodec(cfg.vcodec)
        self._meta: Dict[str, Any] = {
            "codebase_version": EMBODI_DEPTH_CODEBASE_VERSION,
            "writer": "embodichain.data_pipeline.depth_video",
            "is_depth_map": True,
            "fps": self._fps,
            "sensors": {},
        }
        self._writers: Dict[str, DepthVideoWriter] = {}
        self._episode_sensors: list[str] = []

    def has_sensor(self, sensor_key: str) -> bool:
        """Return True if *sensor_key* has been registered."""
        return sensor_key in self._meta["sensors"]

    def register_sensor(self, sensor_key: str, shape: tuple[int, ...]) -> None:
        """Register a depth sensor and its static metadata.

        Args:
            sensor_key: Sensor identifier including any side suffix, e.g.
                ``"camera"`` or ``"camera_right"``.
            shape: Spatial shape of the depth frames, e.g. ``(480, 640)``.
        """
        if sensor_key in self._meta["sensors"]:
            return
        h, w = int(shape[-2]), int(shape[-1])
        self._meta["sensors"][sensor_key] = {
            "is_depth_map": True,
            "shape": [h, w, 1],
            "video.codec": self._vcodec,
            "video.pix_fmt": self._cfg.pix_fmt,
            "video.fps": self._fps,
            "video.depth_min": self._cfg.depth_min,
            "video.depth_max": self._cfg.depth_max,
            "video.shift": self._cfg.shift,
            "video.use_log": self._cfg.use_log,
            "video.input_unit": self._cfg.input_unit,
            "video.output_unit": self._cfg.output_unit,
            "video.lossless": self._cfg.lossless,
            "video.crf": int(self._cfg.crf),
            "video.quant_bits": 12,
            "video.qmax": DEPTH_QMAX,
            "episodes": {},
        }

    def start_episode(self, episode_index: int, sensor_keys: list[str]) -> None:
        """Open a writer per sensor for a new episode.

        Args:
            episode_index: Global episode index.
            sensor_keys: Depth sensor keys present in this episode.
        """
        self._episode_sensors = list(sensor_keys)
        self._writers = {}
        for key in self._episode_sensors:
            ep_dir = self._root / "depth_videos" / key
            ep_dir.mkdir(parents=True, exist_ok=True)
            final_path = ep_dir / f"episode_{episode_index:06d}.mp4"
            self._writers[key] = DepthVideoWriter(
                final_path=final_path,
                fps=self._fps,
                cfg=self._cfg,
                vcodec=self._vcodec,
            )

    def add_frame(self, sensor_key: str, depth: Any) -> None:
        """Feed one depth frame for one sensor in the current episode.

        Args:
            sensor_key: Sensor identifier.
            depth: Depth map.
        """
        writer = self._writers.get(sensor_key)
        if writer is None:
            logger.log_warning(
                f"No depth writer for sensor '{sensor_key}'; skipping frame."
            )
            return
        writer.add_frame(depth)

    def end_episode(self, episode_index: int) -> None:
        """Close all writers for the current episode and update metadata.

        On failure of any writer the partial files are discarded and that
        sensor/episode is skipped (with a warning) rather than aborting the
        whole recording.

        Args:
            episode_index: Global episode index.
        """
        for key, writer in self._writers.items():
            try:
                writer.close()
                frame_count = writer.frame_count
            except Exception as e:
                logger.log_warning(
                    f"Failed to finalize depth video for sensor '{key}' "
                    f"episode {episode_index}: {e}"
                )
                writer.abort()
                continue
            rel_file = writer.final_path.relative_to(self._root).as_posix()
            self._meta["sensors"][key]["episodes"][str(episode_index)] = {
                "chunk_index": _CHUNK_INDEX,
                "file": rel_file,
                "frame_count": frame_count,
            }
        self._writers = {}
        self._episode_sensors = []
        self._write_meta()

    def abort_episode(self) -> None:
        """Abort the current episode: discard all partial depth videos.

        Called when the surrounding LeRobot ``save_episode`` fails, so the
        dataset never references a depth video for an episode that was not
        committed.
        """
        for writer in self._writers.values():
            writer.abort()
        self._writers = {}
        self._episode_sensors = []

    def finalize(self) -> Path:
        """Flush metadata after the last episode.

        Returns:
            Path to ``depth_meta.json``.
        """
        self._write_meta()
        return self._root / _META_FILENAME

    def _write_meta(self) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        meta_path = self._root / _META_FILENAME
        tmp_path = meta_path.with_suffix(".json.tmp")
        with open(tmp_path, "w") as f:
            json.dump(self._meta, f, indent=2, sort_keys=True)
        tmp_path.replace(meta_path)


def cleanup_partial_sidecar(dataset_root: Path) -> None:
    """Remove the depth sidecar tree and metadata for an aborted dataset.

    Args:
        dataset_root: Root directory of the LeRobot dataset.
    """
    root = Path(dataset_root)
    sidecar = root / "depth_videos"
    if sidecar.exists():
        shutil.rmtree(sidecar, ignore_errors=True)
    meta = root / _META_FILENAME
    if meta.exists():
        try:
            meta.unlink()
        except OSError:
            pass
