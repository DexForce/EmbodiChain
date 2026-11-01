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

"""Bounded native sensor frame spooling with content-verified chunk manifests."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np

__all__ = ["MediaSpool", "verify_media_manifest"]


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=".pending-")
    try:
        with os.fdopen(handle, "w") as stream:
            json.dump(value, stream, allow_nan=False, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        Path(temporary).unlink(missing_ok=True)


class MediaSpool:
    """Write aligned native sensor frames within a caller-owned byte budget.

    Args:
        directory: Fresh episode-media directory; existing contents are rejected.
        max_buffer_bytes: Maximum retained raw-frame bytes before flushing.
        parent_episode_id: Physical episode that owns the state/action sequence.
        calibration: JSON camera/sensor calibration snapshot for these frames.

    Frames include the initial and terminal observations. Chunk filenames are
    generated internally; user feature names are stored only inside manifests.
    Finishing media does not certify task success or commit an expert episode.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        max_buffer_bytes: int,
        parent_episode_id: str,
        calibration: Mapping[str, object],
    ) -> None:
        if type(max_buffer_bytes) is not int or max_buffer_bytes <= 0:
            raise ValueError("max_buffer_bytes must be positive")
        if not isinstance(parent_episode_id, str) or not parent_episode_id:
            raise ValueError("parent_episode_id must be nonempty")
        self.calibration = json.loads(json.dumps(dict(calibration), allow_nan=False))
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.max_buffer_bytes = max_buffer_bytes
        self.parent_episode_id = parent_episode_id
        self._frames: list[dict[str, np.ndarray]] = []
        self._times: list[float] = []
        self._bytes = 0
        self._last_time = None
        self._schema = None
        self._chunks = []
        self._count = 0
        self._finished = False

    @property
    def buffered_bytes(self) -> int:
        """Return retained raw-frame bytes, excluding transient flush copies."""
        return self._bytes

    def append(self, timestamp: float, frames: Mapping[str, np.ndarray]) -> None:
        """Append an owned sensor observation after validating all feature shapes.

        Args:
            timestamp: Strictly increasing control-clock observation timestamp.
            frames: Stable feature names mapped to uint8 RGB or numeric depth arrays.
        """
        if self._finished:
            raise RuntimeError("media spool is already finished")
        if (
            type(timestamp) not in (float, int)
            or not np.isfinite(timestamp)
            or timestamp < 0
        ):
            raise ValueError("timestamp must be finite and nonnegative")
        if self._last_time is not None and timestamp <= self._last_time:
            raise ValueError("media timestamps must increase")
        if not frames or any(not isinstance(key, str) or not key for key in frames):
            raise ValueError("frames require nonempty feature names")
        schema = {}
        size = 0
        for key, frame in frames.items():
            if (
                not isinstance(frame, np.ndarray)
                or frame.dtype.kind not in "uif"
                or frame.ndim not in (2, 3)
                or not frame.size
            ):
                raise ValueError("media must contain numeric image/depth arrays")
            schema[key] = {"shape": list(frame.shape), "dtype": frame.dtype.str}
            size += frame.nbytes
        if self._schema is not None and schema != self._schema:
            raise ValueError("media schema changed during an episode")
        if size > self.max_buffer_bytes:
            raise BufferError("one sensor frame exceeds the media budget")
        if self._bytes + size > self.max_buffer_bytes:
            self._flush()
        self._frames.append(
            {key: np.array(value, copy=True) for key, value in frames.items()}
        )
        self._times.append(float(timestamp))
        self._schema = schema
        self._last_time = timestamp
        self._bytes += size

    def _flush(self) -> None:
        if not self._frames:
            return
        path = self.directory / f"chunk-{len(self._chunks):06d}.npz"
        names = sorted(self._schema)
        arrays = {
            f"feature_{index}": np.stack([frame[name] for frame in self._frames])
            for index, name in enumerate(names)
        }
        arrays["timestamps"] = np.asarray(self._times, dtype=np.float64)
        with path.open("xb") as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        self._chunks.append(
            {
                "path": path.name,
                "sha256": _digest(path),
                "start": self._count,
                "count": len(self._frames),
            }
        )
        self._count += len(self._frames)
        self._frames.clear()
        self._times.clear()
        self._bytes = 0

    def finish(self) -> Path:
        """Flush frames and atomically publish the media manifest.

        Returns:
            Manifest path. Repeated calls return the same finalized manifest.
        """
        if not self._finished:
            self._flush()
            if not self._count:
                raise ValueError("empty media cannot be finalized")
            _atomic_json(
                self.directory / "manifest.json",
                {
                    "schema_version": 1,
                    "parent_episode_id": self.parent_episode_id,
                    "frame_count": self._count,
                    "features": self._schema,
                    "calibration": self.calibration,
                    "chunks": self._chunks,
                },
            )
            self._finished = True
        return self.directory / "manifest.json"


def verify_media_manifest(
    path: str | Path,
    *,
    expected_frames: int,
    parent_episode_id: str,
    expected_timestamps: np.ndarray | None = None,
) -> Mapping[str, object]:
    """Verify chunk content, shape, temporal continuity and physical parent identity.

    Args:
        path: Finalized media manifest.
        expected_frames: Required T+1 observations for the physical episode.
        parent_episode_id: Exact physical episode identity.
        expected_timestamps: Optional actual T+1 physical observation clock.
    Returns:
        Verified manifest values; no data is committed or modified.
    """
    path = Path(path)
    value = json.loads(path.read_text())
    if (
        value.get("schema_version") != 1
        or value.get("parent_episode_id") != parent_episode_id
        or value.get("frame_count") != expected_frames
    ):
        raise ValueError("media parent, version or frame count differs")
    count, last = 0, None
    names = sorted(value["features"])
    for chunk in value["chunks"]:
        file = path.parent / chunk["path"]
        if (
            file.parent.resolve() != path.parent.resolve()
            or file.is_symlink()
            or chunk["start"] != count
            or type(chunk["count"]) is not int
            or chunk["count"] <= 0
        ):
            raise ValueError("invalid media chunk address or interval")
        if _digest(file) != chunk["sha256"]:
            raise ValueError("media chunk content changed")
        with np.load(file, allow_pickle=False) as arrays:
            if set(arrays.files) != {
                "timestamps",
                *(f"feature_{i}" for i in range(len(names))),
            }:
                raise ValueError("media chunk features differ")
            times = arrays["timestamps"]
            if (
                times.shape != (chunk["count"],)
                or not np.isfinite(times).all()
                or times[0] < 0
                or np.any(np.diff(times) <= 0)
                or (last is not None and times[0] <= last)
            ):
                raise ValueError("media timestamps are not contiguous and increasing")
            if expected_timestamps is not None:
                expected = np.asarray(expected_timestamps)[
                    count : count + chunk["count"]
                ]
                if expected.shape != times.shape or not np.allclose(
                    times, expected, atol=1e-9, rtol=0
                ):
                    raise ValueError(
                        "media timestamps differ from physical observations"
                    )
            last = times[-1]
            for index, name in enumerate(names):
                frame = arrays[f"feature_{index}"]
                schema = value["features"][name]
                if (
                    list(frame.shape) != [chunk["count"], *schema["shape"]]
                    or frame.dtype.str != schema["dtype"]
                ):
                    raise ValueError("media shape or dtype differs")
        count += chunk["count"]
    if count != expected_frames or count <= 0:
        raise ValueError("media chunks do not cover the complete episode")
    return value
