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
from pathlib import Path
from typing import Any

from embodichain.gen_sim.action_engine.generation.source_scene import (
    resolve_source_scene,
)

__all__ = [
    "SceneSourceFingerprint",
    "SceneSourceRef",
    "fingerprint_scene_source",
    "verify_scene_source_fingerprint",
]

_SCENE_SECTIONS = ("background", "rigid_object", "articulation")


@dataclass(frozen=True)
class SceneSourceRef:
    """Reference an existing scene without copying or owning its files."""

    path: Path | str
    robot_profile: str = "franka"
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
            shape = entry.get("shape")
            if not isinstance(shape, Mapping) or not shape.get("fpath"):
                continue
            asset_path = Path(str(shape["fpath"])).expanduser()
            if not asset_path.is_absolute():
                asset_path = resolved.path.parent / asset_path
            asset_path = asset_path.resolve()
            if not asset_path.is_file():
                raise FileNotFoundError(
                    f"Scene asset does not exist: {asset_path} "
                    f"({section}[{index}])."
                )
            asset_hashes[asset_path.as_posix()] = _sha256(asset_path.read_bytes())
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


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
