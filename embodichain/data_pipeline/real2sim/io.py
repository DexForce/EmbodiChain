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

"""Bounded JSON and non-pickle NPZ persistence for source trajectories."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np

from .contracts import JointSpec, PhaseAnnotation, Real2SimTrajectory
from .normalize import (
    _DEFAULT_MAX_OUTPUT_BYTES,
    _DEFAULT_MAX_SAMPLES,
    normalize_trajectory,
)
from .prototype import SkillMotionPrototype, build_motion_prototype

__all__ = ["load_trajectory", "save_trajectory", "prepare_prototype", "load_prototype"]

_ARRAY_FIELDS = frozenset(
    {
        "qpos",
        "timestamps",
        "qvel",
        "initial_observed_qpos",
        "commands",
        "eef_poses",
        "world_from_pose_frame",
        "observed_link_to_tcp",
        "object_pose",
        "target_pose",
    }
)
_MAX_MANIFEST_BYTES = 1024 * 1024
_MAX_ARRAY_BYTES = 512 * 1024 * 1024


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def save_trajectory(source: Real2SimTrajectory, directory: str | Path) -> Path:
    """Write one immutable trajectory bundle and return its manifest path.

    The manifest is published last and contains a digest of the NPZ bytes. An
    existing destination is rejected, including a partial prior bundle.

    Args:
        source: Typed source trajectory.
        directory: New or empty destination directory.

    Returns:
        Path to the saved manifest.
    """
    if not isinstance(source, Real2SimTrajectory):
        raise TypeError("source must be a Real2SimTrajectory")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "trajectory.json"
    data_path = root / "trajectory.npz"
    if manifest_path.exists() or data_path.exists():
        raise FileExistsError("trajectory bundle already exists or is incomplete")
    arrays = {
        name: getattr(source, name)
        for name in sorted(_ARRAY_FIELDS)
        if getattr(source, name) is not None
    }
    metadata = {
        field.name: getattr(source, field.name)
        for field in fields(source)
        if field.name not in _ARRAY_FIELDS and field.name != "phases"
    }
    metadata["phases"] = [asdict(phase) for phase in source.phases]
    data_tmp = None
    manifest_tmp = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=root, suffix=".npz", delete=False
        ) as stream:
            data_tmp = Path(stream.name)
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        payload = {
            "schema_version": 1,
            "array_file": data_path.name,
            "sha256": _digest(data_tmp),
            "metadata": metadata,
        }
        encoded = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False).encode(
            "utf-8"
        )
        if len(encoded) > _MAX_MANIFEST_BYTES:
            raise ValueError("trajectory manifest exceeds the size limit")
        with tempfile.NamedTemporaryFile(
            dir=root, suffix=".json", delete=False
        ) as stream:
            manifest_tmp = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        # Exclusive hard links avoid overwriting another writer's bundle.
        os.link(data_tmp, data_path)
        os.link(manifest_tmp, manifest_path)
        return manifest_path
    finally:
        if data_tmp is not None:
            data_tmp.unlink(missing_ok=True)
        if manifest_tmp is not None:
            manifest_tmp.unlink(missing_ok=True)


def load_trajectory(manifest: str | Path) -> Real2SimTrajectory:
    """Load and validate one bounded, hash-checked trajectory bundle.

    Args:
        manifest: Path to a trajectory JSON manifest.

    Returns:
        Typed source trajectory with independently owned arrays.

    Raises:
        ValueError: Manifest, NPZ, digest or source contract is invalid.
    """
    path = Path(manifest)
    if path.stat().st_size > _MAX_MANIFEST_BYTES:
        raise ValueError("trajectory manifest exceeds the size limit")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("invalid trajectory manifest") from exc
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema_version", "array_file", "sha256", "metadata"}
        or payload["schema_version"] != 1
    ):
        raise ValueError("invalid trajectory manifest schema")
    name = payload["array_file"]
    if not isinstance(name, str) or Path(name).name != name or name in {".", ".."}:
        raise ValueError("array_file must be a local file name")
    root = path.parent.resolve()
    array_path = (root / name).resolve()
    if array_path.parent != root or not array_path.is_file():
        raise ValueError("trajectory NPZ is outside the bundle or missing")
    if _digest(array_path) != payload["sha256"]:
        raise ValueError("trajectory NPZ digest mismatch")
    try:
        with zipfile.ZipFile(array_path) as archive:
            if sum(item.file_size for item in archive.infolist()) > _MAX_ARRAY_BYTES:
                raise ValueError("trajectory arrays exceed the size limit")
        with np.load(array_path, allow_pickle=False) as archive:
            if (
                len(archive.files) != len(set(archive.files))
                or not {"qpos", "timestamps"} <= set(archive.files)
                or set(archive.files) - _ARRAY_FIELDS
            ):
                raise ValueError("unknown or missing trajectory arrays")
            arrays = {
                name: np.array(archive[name], copy=True) for name in archive.files
            }
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError("invalid trajectory NPZ") from exc
    metadata = payload["metadata"]
    allowed = {field.name for field in fields(Real2SimTrajectory)} - _ARRAY_FIELDS
    if not isinstance(metadata, dict) or set(metadata) - allowed:
        raise ValueError("unknown trajectory metadata")
    try:
        metadata = dict(metadata)
        metadata["phases"] = tuple(
            PhaseAnnotation(**phase) for phase in metadata["phases"]
        )
        return Real2SimTrajectory(**metadata, **arrays)
    except (TypeError, KeyError) as exc:
        raise ValueError("invalid trajectory metadata") from exc


def prepare_prototype(
    source: Real2SimTrajectory,
    *,
    joints: tuple[JointSpec, ...],
    control_dt: float,
    directory: str | Path,
    event_tolerance_s: float = 1e-6,
    mimic_tolerance: float = 1e-6,
    max_samples: int = _DEFAULT_MAX_SAMPLES,
    max_output_bytes: int = _DEFAULT_MAX_OUTPUT_BYTES,
) -> Path:
    """Normalize, freeze and publish a portable prototype bundle.

    The complete source is copied into the bundle, so loading never depends on
    the original source path. The final manifest is written only after its NPZ
    and source manifest exist. Existing output is not replaced.

    Args:
        source: Annotated source trajectory.
        joints: Complete ordered target-joint layout.
        control_dt: Destination command period in seconds.
        directory: New or empty output directory.
        event_tolerance_s: Maximum hard-event clock error.
        mimic_tolerance: Tolerance for redundant mimic values.
        max_samples: Maximum normalized sample count before publishing files.
        max_output_bytes: Estimated normalization array/work-buffer budget.

    Returns:
        The prepared prototype manifest path.
    """
    unit = normalize_trajectory(
        source,
        joints=joints,
        control_dt=control_dt,
        event_tolerance_s=event_tolerance_s,
        mimic_tolerance=mimic_tolerance,
        max_samples=max_samples,
        max_output_bytes=max_output_bytes,
    )
    prototype = build_motion_prototype(unit)
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "prototype.json"
    data_path = root / "normalized.npz"
    if manifest_path.exists() or data_path.exists() or (root / "source").exists():
        raise FileExistsError("prototype bundle already exists or is incomplete")
    source_manifest = save_trajectory(source, root / "source")
    time_map = unit.report.time_map
    values = {
        "qpos": unit.qpos,
        "initial_qpos": unit.initial_qpos,
        "timestamps": unit.timestamps,
        "source_left": time_map.left,
        "source_right": time_map.right,
        "source_weight": time_map.weight,
        "source_event_indices": time_map.source_event_indices,
        "execution_event_indices": time_map.execution_event_indices,
    }
    if unit.world_tcp is not None:
        values["world_tcp"] = unit.world_tcp
        values["object_to_tcp"] = prototype.object_to_tcp
    data_tmp = None
    manifest_tmp = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=root, suffix=".npz", delete=False
        ) as stream:
            data_tmp = Path(stream.name)
            np.savez_compressed(stream, **values)
            stream.flush()
            os.fsync(stream.fileno())
        payload = {
            "schema_version": 1,
            "source_manifest": "source/trajectory.json",
            "source_manifest_sha256": _digest(source_manifest),
            "normalized_file": data_path.name,
            "normalized_sha256": _digest(data_tmp),
            "joints": [asdict(joint) for joint in joints],
            "control_dt": control_dt,
            "event_tolerance_s": event_tolerance_s,
            "mimic_tolerance": mimic_tolerance,
            "corrections": list(unit.report.corrections),
            "max_event_error_s": time_map.max_event_error_s,
        }
        encoded = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False).encode(
            "utf-8"
        )
        if len(encoded) > _MAX_MANIFEST_BYTES:
            raise ValueError("prototype manifest exceeds the size limit")
        with tempfile.NamedTemporaryFile(
            dir=root, suffix=".json", delete=False
        ) as stream:
            manifest_tmp = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(data_tmp, data_path)
        os.link(manifest_tmp, manifest_path)
        return manifest_path
    finally:
        if data_tmp is not None:
            data_tmp.unlink(missing_ok=True)
        if manifest_tmp is not None:
            manifest_tmp.unlink(missing_ok=True)


def load_prototype(
    manifest: str | Path,
    *,
    max_samples: int = _DEFAULT_MAX_SAMPLES,
    max_output_bytes: int = _DEFAULT_MAX_OUTPUT_BYTES,
) -> SkillMotionPrototype:
    """Verify a portable prototype and reconstruct its canonical contract.

    Args:
        manifest: Path to the prepared prototype manifest.
        max_samples: Reader-owned normalized sample budget; the manifest cannot
            enlarge it.
        max_output_bytes: Reader-owned estimated normalization memory budget.

    Returns:
        An independently owned skill motion prototype.

    Raises:
        ValueError: A source, manifest or prepared array does not verify.
    """
    path = Path(manifest)
    if path.stat().st_size > _MAX_MANIFEST_BYTES:
        raise ValueError("prototype manifest exceeds the size limit")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("invalid prototype manifest") from exc
    required = {
        "schema_version",
        "source_manifest",
        "source_manifest_sha256",
        "normalized_file",
        "normalized_sha256",
        "joints",
        "control_dt",
        "event_tolerance_s",
        "mimic_tolerance",
        "corrections",
        "max_event_error_s",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != required
        or payload["schema_version"] != 1
    ):
        raise ValueError("invalid prototype manifest schema")
    if (
        payload["source_manifest"] != "source/trajectory.json"
        or payload["normalized_file"] != "normalized.npz"
    ):
        raise ValueError("invalid prototype bundle paths")
    root = path.parent.resolve()
    source_manifest = (root / payload["source_manifest"]).resolve()
    normalized_path = (root / payload["normalized_file"]).resolve()
    if source_manifest.parent != root / "source" or normalized_path.parent != root:
        raise ValueError("prototype paths must remain inside the bundle")
    if (
        _digest(source_manifest) != payload["source_manifest_sha256"]
        or _digest(normalized_path) != payload["normalized_sha256"]
    ):
        raise ValueError("prototype bundle digest mismatch")
    try:
        joints = tuple(JointSpec(**item) for item in payload["joints"])
        source = load_trajectory(source_manifest)
        unit = normalize_trajectory(
            source,
            joints=joints,
            control_dt=payload["control_dt"],
            event_tolerance_s=payload["event_tolerance_s"],
            mimic_tolerance=payload["mimic_tolerance"],
            max_samples=max_samples,
            max_output_bytes=max_output_bytes,
        )
        prototype = build_motion_prototype(unit)
        with zipfile.ZipFile(normalized_path) as archive:
            if sum(item.file_size for item in archive.infolist()) > _MAX_ARRAY_BYTES:
                raise ValueError("prototype arrays exceed the size limit")
        expected = {
            "qpos": unit.qpos,
            "initial_qpos": unit.initial_qpos,
            "timestamps": unit.timestamps,
            "source_left": unit.report.time_map.left,
            "source_right": unit.report.time_map.right,
            "source_weight": unit.report.time_map.weight,
            "source_event_indices": unit.report.time_map.source_event_indices,
            "execution_event_indices": unit.report.time_map.execution_event_indices,
        }
        if unit.world_tcp is not None:
            expected["world_tcp"] = unit.world_tcp
            expected["object_to_tcp"] = prototype.object_to_tcp
        with np.load(normalized_path, allow_pickle=False) as archive:
            if (
                len(archive.files) != len(set(archive.files))
                or set(archive.files) != set(expected)
                or any(
                    archive[name].shape != value.shape
                    or (
                        not np.array_equal(archive[name], value)
                        if np.issubdtype(value.dtype, np.integer)
                        else not np.allclose(archive[name], value, atol=1e-12, rtol=0)
                    )
                    for name, value in expected.items()
                )
            ):
                raise ValueError("prepared arrays differ from normalized source")
        if (
            payload["corrections"] != list(unit.report.corrections)
            or payload["max_event_error_s"] != unit.report.time_map.max_event_error_s
        ):
            raise ValueError("normalization report differs from normalized source")
        return prototype
    except (TypeError, KeyError, OSError, zipfile.BadZipFile) as exc:
        raise ValueError("invalid prepared prototype bundle") from exc
