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

"""Bounded portable bundles for raw robot demonstrations and DemoSkills."""

from __future__ import annotations

from dataclasses import asdict, fields, replace
import json
import os
import shutil
from pathlib import Path
import tempfile
from collections.abc import Mapping

import numpy as np

from .contracts import JointSpec, TrajectoryTimeMap
from .demo_skill import (
    RobotDemonstration,
    DemoSkillReference,
    DemoSkillArtifact,
    build_demo_skill,
    decode_demo_skill_spec,
)
from .io import (
    _ARRAY_FIELDS,
    _MAX_MANIFEST_BYTES,
    _MAX_ARRAY_BYTES,
    _digest,
    _validate_npz_arrays,
)
from .normalize import _DEFAULT_MAX_SAMPLES, _DEFAULT_MAX_OUTPUT_BYTES

__all__ = [
    "save_robot_demonstration",
    "load_robot_demonstration",
    "save_demo_skill",
    "load_demo_skill",
]


def _read_json(path: Path) -> dict:
    if path.stat().st_size > _MAX_MANIFEST_BYTES:
        raise ValueError("DemoSkill manifest exceeds size limit")
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError("manifest must be an object")
    return value


def _publish_json(path: Path, value: Mapping) -> None:
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode()
    if len(payload) > _MAX_MANIFEST_BYTES:
        raise ValueError("DemoSkill manifest exceeds size limit")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _contained(root: Path, name: str) -> Path:
    if not isinstance(name, str) or Path(name).name != name or name in {".", ".."}:
        raise ValueError("bundle references must be local filenames")
    path = (root / name).resolve()
    if path.parent != root.resolve() or not path.is_file():
        raise ValueError("missing or escaping bundle reference")
    return path


def save_robot_demonstration(source: RobotDemonstration, directory: str | Path) -> Path:
    """Publish an immutable raw-source JSON/NPZ bundle.

    Args:
        source: Typed recording; no standard skill identity is required.
        directory: New or empty destination.

    Returns:
        Manifest path, published after numeric arrays.
    """
    if type(source) is not RobotDemonstration:
        raise TypeError("expected RobotDemonstration")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    manifest, data = root / "demonstration.json", root / "demonstration.npz"
    if manifest.exists() or data.exists():
        raise FileExistsError("demonstration bundle already exists or is incomplete")
    arrays, metadata = {}, {}
    for field in fields(source):
        value = getattr(source, field.name)
        if field.name in _ARRAY_FIELDS:
            if value is not None:
                arrays[field.name] = value
        else:
            metadata[field.name] = value
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=root, delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, data)
        _publish_json(
            manifest,
            {
                "schema": "robot-demonstration/v1",
                "array_file": data.name,
                "sha256": _digest(data),
                "metadata": metadata,
            },
        )
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return manifest


def load_robot_demonstration(
    manifest: str | Path, *, max_bytes: int = _MAX_ARRAY_BYTES
) -> RobotDemonstration:
    """Read a hash-checked raw source without pickle or simulation imports.

    Args:
        manifest: Raw source manifest path.
        max_bytes: Caller-owned decoded numeric byte budget.

    Returns:
        Independently owned, validated source recording.
    """
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    path = Path(manifest)
    value = _read_json(path)
    if (
        set(value) != {"schema", "array_file", "sha256", "metadata"}
        or value["schema"] != "robot-demonstration/v1"
    ):
        raise ValueError("invalid robot demonstration schema")
    data = _contained(path.parent, value["array_file"])
    if _digest(data) != value["sha256"]:
        raise ValueError("demonstration array digest mismatch")
    _validate_npz_arrays(data, max_bytes=max_bytes)
    allowed = {field.name for field in fields(RobotDemonstration)}
    with np.load(data, allow_pickle=False) as archive:
        if not {"qpos", "timestamps"} <= set(archive.files) or set(archive.files) - (
            allowed & _ARRAY_FIELDS
        ):
            raise ValueError("unknown or missing demonstration arrays")
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    metadata = value["metadata"]
    if type(metadata) is not dict or set(metadata) - (allowed - _ARRAY_FIELDS):
        raise ValueError("unknown demonstration metadata")
    return RobotDemonstration(**metadata, **arrays)


def save_demo_skill(
    artifact: DemoSkillArtifact,
    directory: str | Path,
    *,
    original_recording: str | Path | None = None,
) -> Path:
    """Save a self-contained asset with source and reproducible preparation inputs.

    Args:
        artifact: Prepared exemplar with no implied physical qualification.
        directory: New or empty destination.
        original_recording: Optional original raw file, verified against the
            recording digest and copied without interpreting its format.

    Returns:
        Immutable DemoSkill manifest path.
    """
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "demo_skill.json"
    if manifest.exists():
        raise FileExistsError("DemoSkill bundle already exists")
    source = save_robot_demonstration(artifact.demonstration, root)
    spec = artifact.spec
    mapping = dict(
        zip(
            artifact.report.time_map.execution_event_indices,
            artifact.report.time_map.source_event_indices,
        )
    )
    spec.phases = tuple(
        replace(
            phase,
            start=int(mapping[phase.start]),
            stop=(
                len(artifact.demonstration.timestamps)
                if index == len(spec.phases) - 1
                else int(mapping[phase.stop])
            ),
        )
        for index, phase in enumerate(spec.phases)
    )
    spec.events = tuple(
        replace(event, sample_index=int(mapping[event.sample_index]))
        for event in spec.events
    )
    reference_payload = {}
    if original_recording is not None:
        original = Path(original_recording)
        expected = artifact.demonstration.recording_sha256
        if _digest(original) != expected:
            raise ValueError("original recording digest mismatch")
        copied = root / "original_recording.bin"
        with original.open("rb") as reader, copied.open("xb") as writer:
            shutil.copyfileobj(reader, writer, 1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
        if _digest(copied) != expected:
            raise ValueError("original recording changed during copy")
        reference_payload["original_recording"] = {
            "file": copied.name,
            "sha256": expected,
        }
    if artifact.reference is not None:
        ref = artifact.reference
        refpath = root / "reference.npz"
        with refpath.open("xb") as stream:
            np.savez_compressed(
                stream,
                qpos=ref.qpos,
                timestamps=ref.timestamps,
                source_mask=ref.source_mask,
                **({"world_tcp": ref.world_tcp} if ref.world_tcp is not None else {}),
                **(
                    {"initial_qpos": ref.initial_qpos}
                    if ref.initial_qpos is not None
                    else {}
                ),
                **(
                    {"input_qpos": ref.input_qpos} if ref.input_qpos is not None else {}
                ),
                **{
                    "map_" + name: getattr(ref.time_map, name)
                    for name in (
                        "left",
                        "right",
                        "weight",
                        "source_event_indices",
                        "execution_event_indices",
                    )
                },
            )
            stream.flush()
            os.fsync(stream.fileno())
        reference_payload.update(
            {
                "reference_file": refpath.name,
                "reference_sha256": _digest(refpath),
                "reference_metadata": {
                    "phases": [asdict(phase) for phase in ref.phases],
                    "events": [asdict(event) for event in ref.events],
                    "preparation_json": ref.preparation_json,
                    "role_poses": ref.role_poses,
                    "max_event_error_s": ref.time_map.max_event_error_s,
                },
            }
        )
    _publish_json(
        manifest,
        {
            **reference_payload,
            "schema": "demo-skill/v1",
            "source_manifest": source.name,
            "source_sha256": _digest(source),
            "content_sha256": artifact.content_sha256,
            "spec": asdict(spec),
            "joints": [asdict(joint) for joint in artifact.joint_specs],
            "control_dt": float(artifact.timestamps[1] - artifact.timestamps[0]),
            "event_tolerance_s": artifact.event_tolerance_s,
        },
    )
    return manifest


def load_demo_skill(
    manifest: str | Path,
    *,
    max_samples: int = _DEFAULT_MAX_SAMPLES,
    max_output_bytes: int = _DEFAULT_MAX_OUTPUT_BYTES,
) -> DemoSkillArtifact:
    """Verify and reconstruct a portable asset under caller-owned limits.

    Args:
        manifest: DemoSkill manifest path.
        max_samples: Maximum normalized sample count.
        max_output_bytes: Maximum decoded source and normalization budget.

    Returns:
        Reconstructed asset with exact content identity.
    """
    path = Path(manifest)
    value = _read_json(path)
    if (
        set(value)
        - {
            "reference_file",
            "reference_sha256",
            "reference_metadata",
            "original_recording",
        }
    ) != {
        "schema",
        "source_manifest",
        "source_sha256",
        "content_sha256",
        "spec",
        "joints",
        "control_dt",
        "event_tolerance_s",
    } or value[
        "schema"
    ] != "demo-skill/v1":
        raise ValueError("invalid DemoSkill manifest schema")
    source = _contained(path.parent, value["source_manifest"])
    if _digest(source) != value["source_sha256"]:
        raise ValueError("DemoSkill source manifest digest mismatch")
    result = build_demo_skill(
        load_robot_demonstration(source, max_bytes=max_output_bytes),
        decode_demo_skill_spec(value["spec"]),
        joints=tuple(JointSpec(**joint) for joint in value["joints"]),
        control_dt=value["control_dt"],
        event_tolerance_s=value["event_tolerance_s"],
        max_samples=max_samples,
        max_output_bytes=max_output_bytes,
    )
    refkeys = {"reference_file", "reference_sha256", "reference_metadata"}
    if set(value) & refkeys:
        if not refkeys <= set(value):
            raise ValueError("incomplete nominal reference declaration")
        refpath = _contained(path.parent, value["reference_file"])
        if _digest(refpath) != value["reference_sha256"]:
            raise ValueError("nominal reference digest mismatch")
        _validate_npz_arrays(refpath, max_bytes=max_output_bytes)
        from .demo_skill import DemoSkillPhase, DemoSkillEvent

        with np.load(refpath, allow_pickle=False) as archive:
            required = {"qpos", "timestamps", "source_mask"} | {
                "map_" + name
                for name in (
                    "left",
                    "right",
                    "weight",
                    "source_event_indices",
                    "execution_event_indices",
                )
            }
            if not required <= set(archive.files) or set(archive.files) - required - {
                "world_tcp",
                "initial_qpos",
                "input_qpos",
            }:
                raise ValueError("unknown or missing nominal reference arrays")
            values = {
                name: np.array(archive[name], copy=True) for name in archive.files
            }
        if len(values["timestamps"]) > max_samples:
            raise ValueError("nominal reference exceeds max_samples")
        meta = value["reference_metadata"]
        if type(meta) is not dict or (set(meta) - {"role_poses"}) != {
            "phases",
            "events",
            "preparation_json",
            "max_event_error_s",
        }:
            raise ValueError("invalid nominal reference metadata")
        mapping = TrajectoryTimeMap(
            *(
                values.pop("map_" + name)
                for name in (
                    "left",
                    "right",
                    "weight",
                    "source_event_indices",
                    "execution_event_indices",
                )
            ),
            meta["max_event_error_s"],
        )
        ref = DemoSkillReference(
            **values,
            phases=tuple(DemoSkillPhase(**p) for p in meta["phases"]),
            events=tuple(DemoSkillEvent(**e) for e in meta["events"]),
            time_map=mapping,
            preparation_json=meta["preparation_json"],
            role_poses=tuple(
                (name, tuple(pose)) for name, pose in meta.get("role_poses", ())
            ),
        )
        result = replace(result, reference=ref)
    if "original_recording" in value:
        original = value["original_recording"]
        if (
            type(original) is not dict
            or set(original) != {"file", "sha256"}
            or original["sha256"] != result.demonstration.recording_sha256
        ):
            raise ValueError("invalid original recording reference")
        if _digest(_contained(path.parent, original["file"])) != original["sha256"]:
            raise ValueError("original recording digest mismatch")
    if result.content_sha256 != value["content_sha256"]:
        raise ValueError("DemoSkill content digest mismatch")
    return result
