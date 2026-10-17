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

"""Content-checked, non-executable real2sim source bundles."""

from __future__ import annotations

from dataclasses import asdict, fields
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np

from .contracts import (
    EefPoseSamples,
    MimicJoint,
    Real2SimTrajectory,
    RobotJointLayout,
    SourcePhase,
    _plain,
)
from .normalization import (
    Real2SimNormalizationCfg,
    NormalizedReal2SimUnit,
    normalize_real2sim,
)

__all__ = ["write_real2sim_bundle", "load_real2sim_bundle"]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verified_bytes(path: Path, expected: str, limit: int, name: str) -> bytes:
    # Decode the same immutable bytes that were hashed, even if a caller replaces
    # an external file concurrently with admission.
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit or hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"Source {name} digest mismatch or oversized content")
    return data


def _closed(value: object, keys: set[str], name: str) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{name} must contain exactly {sorted(keys)}")
    return value


def write_real2sim_bundle(
    source: Real2SimTrajectory,
    layout: RobotJointLayout,
    cfg: Real2SimNormalizationCfg,
    directory: str | Path,
) -> Path:
    """Write raw evidence and explicit normalization settings into a new directory.

    Args:
        source: Owned raw source values; review states are preserved verbatim.
        layout: Complete target joint model and restored initial state.
        cfg: Explicit normalization settings.
        directory: New output directory, separate from the original recording.

    Returns:
        Path to the bundle JSON; its SHA-256 pins the entire declaration.
    """
    normalize_real2sim(source, layout, cfg)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    arrays = {
        "positions": source.positions,
        "timestamps": source.timestamps,
        "initial_positions": layout.initial_positions,
    }
    if source.velocities is not None:
        arrays["velocities"] = source.velocities
    data = {
        f.name: _plain(getattr(source, f.name))
        for f in fields(source)
        if f.name not in {"positions", "timestamps", "velocities", "eef", "phases"}
    }
    data["phases"] = [asdict(p) for p in source.phases]
    data["has_velocities"] = source.velocities is not None
    data["eef"] = None
    if source.eef is not None:
        arrays["eef_poses"] = source.eef.poses
        data["eef"] = {
            f.name: getattr(source.eef, f.name)
            for f in fields(source.eef)
            if f.name != "poses"
        }
        for name in ("reference_to_base", "link_to_tcp"):
            data["eef"][name] = data["eef"][name].tolist()
    target = {
        f.name: getattr(layout, f.name)
        for f in fields(layout)
        if f.name not in {"initial_positions", "mimics"}
    }
    target["mimics"] = [asdict(m) for m in layout.mimics]
    options = {f.name: getattr(cfg, f.name) for f in fields(cfg)}
    np.savez_compressed(directory / "source.npz", **arrays)
    payload = {
        "schema_version": 1,
        "arrays": "source.npz",
        "arrays_sha256": _digest(directory / "source.npz"),
        "source": data,
        "layout": target,
        "normalization": options,
    }
    path = directory / "bundle.json"
    path.write_text(
        json.dumps(payload, sort_keys=True, allow_nan=False, indent=2) + "\n"
    )
    return path


def load_real2sim_bundle(
    path: str | Path, *, expected_sha256: str, require_verified: bool = True
) -> NormalizedReal2SimUnit:
    """Validate bundle content and rerun normalization before any simulation.

    Args:
        path: Bundle JSON file with a local NPZ sidecar.
        expected_sha256: Required content digest from trusted integration config.
        require_verified: Require reviewed prerequisites for formal execution.
            False allows offline inspection only.

    Returns:
        Fresh owned normalized values; no simulator or executable imports are loaded.
    """
    path = Path(path)
    declaration = _verified_bytes(path, expected_sha256, 1_000_000, "bundle")
    payload = _closed(
        json.loads(declaration),
        {
            "schema_version",
            "arrays",
            "arrays_sha256",
            "source",
            "layout",
            "normalization",
        },
        "bundle",
    )
    if payload["schema_version"] != 1 or payload["arrays"] != "source.npz":
        raise ValueError("Unsupported bundle version or sidecar path")
    sidecar = path.parent / payload["arrays"]
    array_bytes = _verified_bytes(
        sidecar, payload["arrays_sha256"], 256_000_000, "array"
    )
    raw = _closed(
        payload["source"],
        {f.name for f in fields(Real2SimTrajectory)}
        - {"positions", "timestamps", "velocities"}
        | {"has_velocities"},
        "source",
    )
    layout = _closed(
        payload["layout"],
        {f.name for f in fields(RobotJointLayout)} - {"initial_positions"},
        "layout",
    )
    options = _closed(
        payload["normalization"],
        {f.name for f in fields(Real2SimNormalizationCfg)},
        "normalization",
    )
    with zipfile.ZipFile(io.BytesIO(array_bytes)) as archive:
        if sum(info.file_size for info in archive.infolist()) > 256_000_000:
            raise ValueError("Source arrays exceed the uncompressed size limit")
    if type(raw["has_velocities"]) is not bool:
        raise ValueError("has_velocities must be boolean")
    with np.load(io.BytesIO(array_bytes), allow_pickle=False) as arrays:
        expected = {"positions", "timestamps", "initial_positions"}
        if raw["has_velocities"]:
            expected.add("velocities")
        if raw["eef"] is not None:
            expected.add("eef_poses")
        if set(arrays.files) != expected:
            raise ValueError("Source bundle array keys do not match declarations")
        eef = raw["eef"]
        if eef is not None:
            _closed(eef, {f.name for f in fields(EefPoseSamples)} - {"poses"}, "EEF")
            eef = EefPoseSamples(poses=arrays["eef_poses"], **eef)
        source = Real2SimTrajectory(
            **{
                k: v
                for k, v in raw.items()
                if k not in {"phases", "has_velocities", "eef"}
            },
            positions=arrays["positions"],
            timestamps=arrays["timestamps"],
            velocities=arrays["velocities"] if raw["has_velocities"] else None,
            phases=tuple(SourcePhase(**p) for p in raw["phases"]),
            eef=eef,
        )
        target = RobotJointLayout(
            **{k: v for k, v in layout.items() if k != "mimics"},
            initial_positions=arrays["initial_positions"],
            mimics=tuple(MimicJoint(**m) for m in layout["mimics"]),
        )
    unit = normalize_real2sim(source, target, Real2SimNormalizationCfg(**options))
    if require_verified and not unit.eligible:
        raise ValueError("Source bundle prerequisites remain unverified")
    return unit
