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
import math
import zipfile
from bisect import bisect_right
import os
from pathlib import Path
import tempfile
import numpy as np

__all__ = ["MediaSpool", "verify_media_manifest", "MediaFrameReader"]


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mkdir(path: Path) -> None:
    """Persist newly created directory links before publishing child artifacts."""
    missing = []
    current = path
    while not current.exists():
        missing.append(current)
        current = current.parent
    for directory in reversed(missing):
        directory.mkdir(exist_ok=True)
        _sync_directory(directory.parent)


def _atomic_json(path: Path, value: object) -> None:
    _mkdir(path.parent)
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
        _mkdir(self.directory.parent)
        self.directory.mkdir()
        _sync_directory(self.directory.parent)
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
            f"feature_{index}": np.ascontiguousarray(
                np.stack([frame[name] for frame in self._frames])
            )
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


class MediaFrameReader:
    """Read owned individual frames without materializing a full NPZ chunk.

    Args:
        manifest: Verified media manifest address.
        fields: Optional feature subset; calibration matrices are ordinary features.
        max_frame_bytes: Optional bound on the requested raw frame payload.

    Streams retain only headers and ZIP handles. Returned arrays own their bytes.
    Legacy Fortran-ordered arrays use bounded block scans; newly produced chunks
    use C order. Chunk hashes and array layouts are checked before reading values.
    """

    def __init__(
        self,
        manifest: str | Path,
        *,
        fields: tuple[str, ...] | None = None,
        max_frame_bytes: int | None = None,
    ) -> None:
        self.path = Path(manifest)
        self.manifest = json.loads(self.path.read_text())
        if (
            self.manifest.get("schema_version") != 1
            or type(self.manifest.get("frame_count")) is not int
            or self.manifest["frame_count"] < 1
        ):
            raise ValueError("invalid media manifest")
        for schema in self.manifest["features"].values():
            if (
                not schema["shape"]
                or any(type(v) is not int or v < 1 for v in schema["shape"])
                or np.dtype(schema["dtype"]).kind not in "buif"
            ):
                raise ValueError("invalid media feature shape or dtype")
        self.names = tuple(sorted(self.manifest["features"]))
        self.fields = self.names if fields is None else tuple(fields)
        self._indexes = {name: index for index, name in enumerate(self.names)}
        if set(self.fields) - set(self.names):
            raise ValueError("unknown recorded media feature")
        required = sum(
            math.prod(self.manifest["features"][name]["shape"])
            * np.dtype(self.manifest["features"][name]["dtype"]).itemsize
            for name in self.fields
        )
        if max_frame_bytes is not None and required > max_frame_bytes:
            raise BufferError("recorded frame exceeds export byte budget")
        self._chunk = None
        self._archive = None
        self._streams = {}
        self._starts = [chunk["start"] for chunk in self.manifest["chunks"]]

    def close(self) -> None:
        """Release chunk handles; callers may subsequently open another frame."""
        for stream, _, _, _, _ in self._streams.values():
            stream.close()
        self._streams.clear()
        if self._archive is not None:
            self._archive.close()
        self._archive = None
        self._chunk = None

    def _open(self, index):
        if self._chunk == index:
            return
        self.close()
        chunk = self.manifest["chunks"][index]
        path = self.path.parent / chunk["path"]
        if (
            path.parent.resolve() != self.path.parent.resolve()
            or path.is_symlink()
            or _digest(path) != chunk["sha256"]
        ):
            raise ValueError("media chunk address or content changed")
        self._archive = zipfile.ZipFile(path)
        try:
            requested = {"timestamps": ((chunk["count"],), np.dtype("float64"))}
            for name in self.fields:
                schema = self.manifest["features"][name]
                requested[f"feature_{self._indexes[name]}"] = (
                    (chunk["count"], *schema["shape"]),
                    np.dtype(schema["dtype"]),
                )
            for key, (expected_shape, expected_dtype) in requested.items():
                stream = self._archive.open(key + ".npy")
                self._streams[key] = (stream, 0, (), None, False)
                version = np.lib.format.read_magic(stream)
                reader = {
                    (1, 0): np.lib.format.read_array_header_1_0,
                    (2, 0): np.lib.format.read_array_header_2_0,
                }.get(version)
                if reader is None:
                    stream.close()
                    raise ValueError("unsupported media array encoding")
                shape, fortran, dtype = reader(stream)
                offset = stream.tell()
                if (
                    tuple(shape) != tuple(expected_shape)
                    or dtype != expected_dtype
                    or dtype.kind not in "buif"
                    or self._archive.getinfo(key + ".npy").file_size
                    != offset + math.prod(shape) * dtype.itemsize
                ):
                    stream.close()
                    raise ValueError("recorded media array layout changed")
                self._streams[key] = (stream, offset, shape, dtype, fortran)
            self._chunk = index
        except BaseException:
            self.close()
            raise

    def _array(self, key, row):
        stream, offset, shape, dtype, fortran = self._streams[key]
        frame_shape = shape[1:]
        size = math.prod(frame_shape)
        if not fortran or shape[0] == 1:
            stream.seek(offset + row * size * dtype.itemsize)
            raw = stream.read(size * dtype.itemsize)
            if len(raw) != size * dtype.itemsize:
                raise ValueError("truncated media frame")
            return (
                np.frombuffer(raw, dtype=dtype)
                .reshape(frame_shape, order="F" if fortran else "C")
                .copy()
            )
        # Legacy first-axis interleaving: retain one output frame and <=1 MiB input.
        output = np.empty(size, dtype=dtype)
        stream.seek(offset)
        remaining = math.prod(shape)
        position = 0
        written = 0
        block = max(1, (1024 * 1024) // dtype.itemsize)
        while remaining:
            count = min(block, remaining)
            raw = stream.read(count * dtype.itemsize)
            if len(raw) != count * dtype.itemsize:
                raise ValueError("truncated media frame")
            values = np.frombuffer(raw, dtype=dtype)
            first = (row - position) % shape[0]
            selected = values[first :: shape[0]]
            output[written : written + len(selected)] = selected
            written += len(selected)
            position += count
            remaining -= count
        if written != size:
            raise ValueError("incomplete media frame")
        return output.reshape(frame_shape, order="F")

    def read(self, index: int) -> tuple[float, dict[str, np.ndarray]]:
        """Return one aligned timestamp and owned requested feature arrays."""
        if type(index) is not int or not 0 <= index < self.manifest["frame_count"]:
            raise IndexError(index)
        chunk_index = bisect_right(self._starts, index) - 1
        if chunk_index < 0:
            raise ValueError("media chunks have a missing initial range")
        self._open(chunk_index)
        row = index - self._starts[chunk_index]
        if row >= self.manifest["chunks"][chunk_index]["count"]:
            raise ValueError("media chunks have a gap")
        timestamp = float(self._array("timestamps", row))
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("invalid media timestamp")
        return timestamp, {
            name: self._array(f"feature_{self._indexes[name]}", row)
            for name in self.fields
        }

    def __enter__(self) -> MediaFrameReader:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
