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

"""Read-only references and integrity checks for existing Gym projects."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from embodichain.data import get_data_path
from embodichain.data.asset_dependencies import asset_dependency_files
from .source_scene import resolve_source_scene

__all__ = [
    "SceneSourceFingerprint",
    "SceneSourceRef",
    "fingerprint_scene_source",
    "scene_revision_id",
    "verify_scene_source_fingerprint",
]

_SCENE_SECTIONS = ("background", "rigid_object", "articulation")


@dataclass(frozen=True)
class SceneSourceRef:
    """Reference an existing scene without copying or owning its files."""

    path: Path | str
    robot_profile: str = "dual_franka"
    z_rotation_degrees: float | None = None
    body_scale_policy: str = "preserve"
    body_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path).expanduser())


@dataclass(frozen=True)
class SceneSourceFingerprint:
    """Content evidence for one externally owned scene source."""

    source_format: str
    config_path: Path
    config_sha256: str
    asset_sha256: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe audit view."""
        return {
            "source_format": self.source_format,
            "config_path": self.config_path.as_posix(),
            "config_sha256": self.config_sha256,
            "asset_sha256": dict(sorted(self.asset_sha256.items())),
        }


def fingerprint_scene_source(
    source: SceneSourceRef | str | Path,
) -> SceneSourceFingerprint:
    """Hash a source config and referenced assets without copying either."""
    source_path = source.path if isinstance(source, SceneSourceRef) else source
    resolved = resolve_source_scene(source_path)
    config_bytes = resolved.path.read_bytes()
    try:
        config = json.loads(config_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Scene config is not valid JSON: {resolved.path}") from exc
    if not isinstance(config, Mapping):
        raise ValueError(f"Scene config must contain an object: {resolved.path}")

    asset_hashes: dict[str, str] = {}
    for section in _SCENE_SECTIONS:
        entries = config.get(section, ())
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            continue
        for index, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                continue
            references: list[tuple[str, Any]] = []
            shape = entry.get("shape")
            if isinstance(shape, Mapping) and shape.get("fpath"):
                references.append(("shape.fpath", shape["fpath"]))
            if section == "articulation" and entry.get("fpath"):
                references.append(("fpath", entry["fpath"]))
            for field_name, reference in references:
                asset_path = Path(str(reference)).expanduser()
                if not asset_path.is_absolute():
                    asset_path = resolved.path.parent / asset_path
                asset_path = asset_path.resolve()
                if not asset_path.is_file() and not Path(str(reference)).is_absolute():
                    asset_path = (
                        Path(get_data_path(str(reference))).expanduser().resolve()
                    )
                if not asset_path.is_file():
                    raise FileNotFoundError(
                        f"Scene asset does not exist: {asset_path} "
                        f"({section}[{index}].{field_name})."
                    )
                for dependency in asset_dependency_files(asset_path):
                    asset_hashes[dependency.as_posix()] = _sha256(
                        dependency.read_bytes()
                    )
    return SceneSourceFingerprint(
        source_format=resolved.source_format,
        config_path=resolved.path,
        config_sha256=_sha256(config_bytes),
        asset_sha256=asset_hashes,
    )


def verify_scene_source_fingerprint(expected: Mapping[str, Any]) -> None:
    """Raise when an externally owned source changed after preparation."""
    required = {"source_format", "config_path", "config_sha256", "asset_sha256"}
    if set(expected) != required:
        raise ValueError("Scene source fingerprint fields are invalid.")
    actual = fingerprint_scene_source(str(expected["config_path"])).to_dict()
    normalized = {
        "source_format": str(expected["source_format"]),
        "config_path": Path(str(expected["config_path"])).resolve().as_posix(),
        "config_sha256": str(expected["config_sha256"]),
        "asset_sha256": dict(expected["asset_sha256"]),
    }
    if actual != normalized:
        raise RuntimeError(
            "Source Gym project changed after Task Engine preparation; "
            "prepare a new bundle before running it."
        )


def scene_revision_id(source: SceneSourceRef | str | Path) -> str:
    """Return a location-independent content identity for one scene revision.

    Volatile exporter IDs and absolute asset paths are excluded. Referenced
    asset content remains part of the identity through SHA-256 placeholders.

    Args:
        source: Scene project, configuration path, or Task Engine source reference.

    Returns:
        Stable SHA-256 identity of scene semantics and referenced asset content.
    """
    source_path = source.path if isinstance(source, SceneSourceRef) else source
    resolved = resolve_source_scene(source_path)
    try:
        config = json.loads(resolved.path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Scene config is not valid JSON: {resolved.path}") from exc
    if not isinstance(config, Mapping):
        raise ValueError(f"Scene config must contain an object: {resolved.path}")
    normalized = _normalize_revision_value(
        dict(config),
        config_root=resolved.path.parent,
    )
    normalized.pop("scene_id", None)
    payload = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256(payload)


def _normalize_revision_value(value: Any, *, config_root: Path) -> Any:
    if isinstance(value, Mapping):
        result = {
            str(key): _normalize_revision_value(item, config_root=config_root)
            for key, item in value.items()
        }
        for key in ("fpath",):
            raw = result.get(key)
            if not isinstance(raw, str) or not raw:
                continue
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = config_root / path
            path = path.resolve()
            if path.is_file():
                files = asset_dependency_files(path)
                result[key] = {
                    "sha256": _sha256(path.read_bytes()),
                    "dependency_sha256": {
                        Path(
                            os.path.relpath(dependency, start=path.parent)
                        ).as_posix(): (_sha256(dependency.read_bytes()))
                        for dependency in files
                        if dependency != path
                    },
                }
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [
            _normalize_revision_value(item, config_root=config_root) for item in value
        ]
    return value


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
