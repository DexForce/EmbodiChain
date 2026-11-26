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

"""Joint recordings, explicit video alignment, and self-contained fit bundles."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any
import zipfile

import numpy as np

__all__ = [
    "JointRecording",
    "load_joint_recording",
    "map_video_interval",
    "save_skill_fit",
    "load_skill_fit",
    "file_digest",
]

_MAX_BYTES = 256 * 1024 * 1024


def file_digest(path: str | Path) -> str:
    """Hash a source without retaining its contents.

    Args:
        path: Existing source file.

    Returns:
        Lowercase SHA-256 digest.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class JointRecording:
    """Owned joints with explicit order, clock and observation semantics.

    Args:
        timestamps: Strictly increasing physical times in seconds.
        qpos: Numeric (T,D) samples in robot joint units.
        joint_names: Ordered, unique joint identifiers.
        semantics: measured_state, actual_submission or design_reference.
    """

    timestamps: np.ndarray
    qpos: np.ndarray
    joint_names: tuple[str, ...]
    semantics: str

    def __post_init__(self) -> None:
        q = np.asarray(self.qpos)
        t = np.asarray(self.timestamps, dtype=np.float64)
        names = tuple(self.joint_names)
        if (
            q.ndim != 2
            or q.shape[1] == 0
            or q.dtype.kind not in "fi"
            or len(q) < 12
            or q.nbytes + t.nbytes > _MAX_BYTES
            or t.shape != (len(q),)
            or not np.isfinite(q).all()
            or not np.isfinite(t).all()
            or np.any(np.diff(t) <= 0)
            or len(names) != q.shape[1]
            or len(set(names)) != len(names)
            or any(not isinstance(n, str) or not n or len(n) > 256 for n in names)
        ):
            raise ValueError("Invalid joint recording, joint order, or physical clock.")
        if self.semantics not in {
            "measured_state",
            "actual_submission",
            "design_reference",
        }:
            raise ValueError(
                "Declare measured_state, actual_submission or design_reference."
            )
        q, t = (
            np.array(
                q, dtype=q.dtype if q.dtype.kind == "f" else np.float64, copy=True
            ),
            t.copy(),
        )
        q.setflags(write=False)
        t.setflags(write=False)
        object.__setattr__(self, "qpos", q)
        object.__setattr__(self, "timestamps", t)
        object.__setattr__(self, "joint_names", names)


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    """Preflight decoded NPY allocation sizes before materializing arrays."""
    total = 0
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) > 64 or len(set(names)) != len(names):
            raise ValueError("Too many or duplicate NPZ members.")
        for member in archive.infolist():
            if not member.filename.endswith(".npy") or member.file_size > _MAX_BYTES:
                raise ValueError("Invalid or oversized NPZ member.")
            with archive.open(member) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError("Unsupported NPY header version.")
                size = math.prod(shape) * dtype.itemsize
                total += size
                if dtype.hasobject or total > _MAX_BYTES or size > member.file_size:
                    raise ValueError("NPZ decoded arrays exceed the allocation budget.")
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def load_joint_recording(
    path: str | Path,
    *,
    joint_names: tuple[str, ...],
    semantics: str,
    qpos_key: str = "qpos",
    timestamps_key: str = "timestamps",
    sample_dt: float | None = None,
    env_index: int = 0,
) -> JointRecording:
    """Load NPZ, HDF5, or a trusted local Atomic replay PT file.

    PT loading uses PyTorch's weights-only reader. NPZ is the portable input.
    Missing clocks require an explicit sample_dt; recording FPS is never guessed.

    Args:
        path: Source recording.
        joint_names: Expected robot order, also used when the file has no names.
        semantics: Explicit meaning of qpos.
        qpos_key: NPZ/HDF5 key, or PT positions key.
        timestamps_key: Physical time key.
        sample_dt: Explicit period for recordings without timestamps or dt.
        env_index: Selected row of a batched PT replay.

    Returns:
        A validated recording reordered by explicit joint names.
    """
    source = Path(path)
    if type(env_index) is not int or env_index < 0:
        raise ValueError("env_index must be a nonnegative integer.")
    if source.suffix == ".npz":
        data = _read_npz(source)
    elif source.suffix in {".h5", ".hdf5"}:
        import h5py

        with h5py.File(source, "r") as handle:
            keys = [qpos_key] + ([timestamps_key] if timestamps_key in handle else [])
            if any(handle[k].dtype.hasobject for k in keys):
                raise ValueError(
                    "Joint and time datasets must contain fixed-size numeric values."
                )
            if (
                sum(handle[k].size * handle[k].dtype.itemsize for k in keys)
                > _MAX_BYTES
            ):
                raise ValueError("HDF5 numeric input exceeds the allocation budget.")
            data = {key: handle[key][...] for key in keys}
    elif source.suffix == ".pt":
        import torch

        if source.stat().st_size > _MAX_BYTES:
            raise ValueError("PT replay exceeds the input budget.")
        raw = torch.load(source, weights_only=True, map_location="cpu")
        if qpos_key == "qpos":
            qpos_key = "positions"
        data = {}
        for name in (qpos_key, timestamps_key, "dt", "planned_dt"):
            if name in raw:
                value = raw[name]
                if not isinstance(value, torch.Tensor):
                    raise ValueError("PT numeric fields must be tensors.")
                value = value.numpy()
                if value.ndim == (3 if name == qpos_key else 2):
                    value = value[env_index]
                data[name] = value
        if timestamps_key not in data:
            interval_key = "planned_dt" if qpos_key == "planned_positions" else "dt"
            if interval_key in data:
                data[timestamps_key] = np.cumsum(data[interval_key], dtype=np.float64)
    else:
        raise ValueError("Joint recordings support .npz, .h5/.hdf5 and local .pt.")
    q = np.asarray(data[qpos_key])
    if timestamps_key in data:
        times = np.asarray(data[timestamps_key], dtype=np.float64)
    elif sample_dt is not None and math.isfinite(sample_dt) and sample_dt > 0:
        times = np.arange(len(q), dtype=np.float64) * sample_dt
    else:
        raise ValueError("Missing joint timestamps; provide an explicit sample_dt.")
    if "joint_names" in data:
        source_names = tuple(str(v) for v in data["joint_names"])
        if len(set(source_names)) != len(source_names) or set(source_names) != set(
            joint_names
        ):
            raise ValueError("Source and robot joint identifiers differ.")
        q = q[:, [source_names.index(name) for name in joint_names]]
    return JointRecording(times, q, joint_names, semantics)


def map_video_interval(
    frame_times: np.ndarray,
    frame_durations: np.ndarray,
    joint_times: np.ndarray,
    *,
    start_frame: int,
    end_frame: int,
    time_offset: float,
) -> tuple[int, int]:
    """Map an inclusive video interval to a half-open joint interval.

    Args:
        frame_times: Video presentation times relative to its first frame.
        frame_durations: Positive display duration of every indexed frame.
        joint_times: Strictly increasing physical joint times.
        start_frame: Included annotation start.
        end_frame: Included annotation end.
        time_offset: Joint time corresponding to video time zero.

    Returns:
        First included and first excluded joint indices.
    """
    ft, fd, jt = (
        np.asarray(v, dtype=np.float64)
        for v in (frame_times, frame_durations, joint_times)
    )
    if (
        ft.ndim != 1
        or fd.shape != ft.shape
        or jt.ndim != 1
        or not len(jt)
        or not np.isfinite(ft).all()
        or not np.isfinite(fd).all()
        or not np.isfinite(jt).all()
        or not math.isfinite(time_offset)
        or np.any(np.diff(ft) <= 0)
        or np.any(fd <= 0)
        or np.any(np.diff(jt) <= 0)
        or type(start_frame) is not int
        or type(end_frame) is not int
        or not 0 <= start_frame <= end_frame < len(ft)
    ):
        raise ValueError("Invalid indexed video clock or inclusive frame interval.")
    left = ft[start_frame] + time_offset
    right = ft[end_frame] + fd[end_frame] + time_offset
    tolerance = 1e-7
    if left < jt[0] - tolerance or left > jt[-1] + tolerance or right <= jt[0]:
        raise ValueError("Video interval is outside the joint recording clock.")
    if (
        right
        > jt[-1] + max(float(fd[end_frame]), float(np.median(np.diff(jt)))) + tolerance
    ):
        raise ValueError("Video interval exceeds the joint recording.")
    return (
        int(np.searchsorted(jt, left - tolerance, side="left")),
        int(np.searchsorted(jt, right - tolerance, side="left")),
    )


def save_skill_fit(
    path: str | Path, arrays: dict[str, np.ndarray], metadata: dict[str, Any]
) -> None:
    """Atomically write a standalone numeric result, without pickle.

    Args:
        path: Destination NPZ.
        arrays: Original, native and patched samples plus visualization geometry.
        metadata: Versioned fit parameters, profile, clocks and diagnostics.
    """
    destination = Path(path)
    if metadata.get("schema_version") != 1:
        raise ValueError("Fit metadata must declare schema_version=1.")
    values = {k: np.asarray(v) for k, v in arrays.items()}
    if "metadata_json" in values or any(v.dtype.hasobject for v in values.values()):
        raise ValueError("Result arrays cannot use objects or reserved names.")
    if sum(v.nbytes for v in values.values()) > _MAX_BYTES:
        raise ValueError("Result exceeds the numeric budget.")
    values["metadata_json"] = np.array(
        json.dumps(metadata, allow_nan=False, sort_keys=True)
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=destination.parent, suffix=".npz")
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **values)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_skill_fit(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read a self-contained result without the source recording or simulator.

    Args:
        path: NPZ written by save_skill_fit.

    Returns:
        Owned numeric arrays and metadata.
    """
    data = _read_npz(Path(path))
    metadata = json.loads(str(data.pop("metadata_json")))
    if metadata.get("schema_version") != 1:
        raise ValueError("Unsupported skill-fit schema.")
    required = {"timestamps", "original_qpos", "native_qpos", "patched_qpos"}
    if not required <= data.keys():
        raise ValueError("Fit bundle is missing trajectory arrays.")
    return data, metadata
