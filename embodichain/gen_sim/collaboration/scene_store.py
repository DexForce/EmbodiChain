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

"""Content-addressed storage for immutable collaboration scene packages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from embodichain.gen_sim.action_engine.generation.source_scene import (
    resolve_source_scene,
)

__all__ = [
    "ScenePackageCorruptError",
    "ScenePackageNotFoundError",
    "ScenePackageRef",
    "ScenePackageStore",
    "SceneSourceRef",
]


_PACKAGE_SCHEMA = "action_engine_scene_package_v1"
_ADAPTER_POLICY_VERSION = "action_engine_scene_adapter_v1"
_MANIFEST_FILENAME = "scene_package.json"
_PACKAGE_KEYS = frozenset(
    {
        "schema_version",
        "package_id",
        "adapter_policy_version",
        "source_format",
        "adaptation",
        "config_path",
        "config_sha256",
        "assets",
    }
)
_ASSET_KEYS = frozenset({"path", "sha256", "size"})
_ADAPTATION_KEYS = frozenset({"z_rotation_degrees", "body_scale_policy", "body_scale"})


class ScenePackageCorruptError(ValueError):
    """A scene package failed its path or content-integrity contract."""


class ScenePackageNotFoundError(FileNotFoundError):
    """A requested content-addressed scene package does not exist."""


@dataclass(frozen=True)
class SceneSourceRef:
    """Reference to an existing exported scene and its adaptation policy."""

    path: Path | str
    robot_profile: str = "franka"
    z_rotation_degrees: float | None = None
    body_scale_policy: str = "preserve"
    body_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path).expanduser())


@dataclass(frozen=True)
class ScenePackageRef:
    """A verified package reference returned by :class:`ScenePackageStore`."""

    package_id: str
    package_path: Path | None = None
    config_path: Path | None = None
    manifest: Mapping[str, Any] = field(default_factory=dict)
    robot_profile: str = "franka"
    z_rotation_degrees: float | None = None
    body_scale_policy: str = "preserve"
    body_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)


class ScenePackageStore:
    """Import and verify immutable scene packages in a local CAS."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = _data_bank_root(root)
        self.packages_root = self.root / "scene_packages" / "sha256"

    def import_scene(
        self,
        source: SceneSourceRef | str | Path,
    ) -> ScenePackageRef:
        """Copy a source scene and its assets into the content-addressed bank."""
        source_ref = _coerce_source_ref(source)
        adaptation = _adaptation_policy(source_ref)
        resolved = resolve_source_scene(source_ref.path)
        source_config = _read_json(resolved.path, context="source scene config")
        packaged_config, assets = _package_assets(
            source_config,
            source_dir=resolved.path.parent,
        )
        package_id = _package_digest(
            packaged_config,
            source_format=resolved.source_format,
            assets=assets,
            adaptation=adaptation,
        )
        package_dir = self._package_dir(package_id)
        if package_dir.exists():
            loaded = self._verify_package(package_dir, expected_id=package_id)
            return _with_source_ref(loaded, source_ref)

        package_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{package_id}.staging-",
                dir=package_dir.parent,
            )
        )
        try:
            config_name = resolved.path.name
            config_path = staging / config_name
            config_bytes = _canonical_json_bytes(packaged_config) + b"\n"
            config_path.write_bytes(config_bytes)
            for asset in assets:
                target = _safe_package_path(staging, str(asset["path"]))
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(Path(str(asset["source_path"])), target)
            manifest = {
                "schema_version": _PACKAGE_SCHEMA,
                "package_id": package_id,
                "adapter_policy_version": _ADAPTER_POLICY_VERSION,
                "source_format": resolved.source_format,
                "adaptation": adaptation,
                "config_path": config_name,
                "config_sha256": _sha256_bytes(config_bytes),
                "assets": [
                    {
                        "path": str(asset["path"]),
                        "sha256": str(asset["sha256"]),
                        "size": int(asset["size"]),
                    }
                    for asset in assets
                ],
            }
            (staging / _MANIFEST_FILENAME).write_bytes(
                _canonical_json_bytes(manifest) + b"\n"
            )
            # Verify staged bytes before publication. A same-digest concurrent
            # importer may win the rename; in that case its package is verified.
            self._verify_package(staging, expected_id=package_id)
            try:
                os.rename(staging, package_dir)
            except OSError:
                if not package_dir.is_dir():
                    raise
                self._verify_package(package_dir, expected_id=package_id)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        loaded = self._verify_package(package_dir, expected_id=package_id)
        return _with_source_ref(loaded, source_ref)

    def load(self, package: ScenePackageRef | str) -> ScenePackageRef:
        """Resolve an exact package ID and verify every referenced byte."""
        requested = (
            package.package_id if isinstance(package, ScenePackageRef) else package
        )
        package_id = _validate_package_id(requested)
        package_dir = self._package_dir(package_id)
        if not package_dir.is_dir():
            raise ScenePackageNotFoundError(
                f"Scene package {package_id!r} does not exist in {self.root}."
            )
        loaded = self._verify_package(package_dir, expected_id=package_id)
        profile = (
            package.robot_profile if isinstance(package, ScenePackageRef) else "franka"
        )
        return _with_robot_profile(loaded, profile)

    def _package_dir(self, package_id: str) -> Path:
        package_id = _validate_package_id(package_id)
        return self.packages_root / package_id[:2] / package_id

    def _verify_package(
        self,
        package_dir: Path,
        *,
        expected_id: str,
    ) -> ScenePackageRef:
        try:
            if package_dir.is_symlink() or not package_dir.is_dir():
                raise ScenePackageCorruptError("Package root must be a real directory.")
            manifest_path = package_dir / _MANIFEST_FILENAME
            if manifest_path.is_symlink():
                raise ScenePackageCorruptError(
                    "Package manifest must not be a symlink."
                )
            manifest = _read_json(manifest_path, context="scene package manifest")
            _validate_manifest(manifest, expected_id=expected_id)
            config_path = _safe_package_path(package_dir, str(manifest["config_path"]))
            _verify_file(
                config_path,
                expected_hash=str(manifest["config_sha256"]),
                label="scene config",
            )
            config = _read_json(config_path, context="packaged scene config")
            assets: list[dict[str, Any]] = []
            for raw in manifest["assets"]:
                asset_path = _safe_package_path(package_dir, str(raw["path"]))
                _verify_file(
                    asset_path,
                    expected_hash=str(raw["sha256"]),
                    expected_size=int(raw["size"]),
                    label="scene asset",
                )
                assets.append(
                    {
                        "path": str(raw["path"]),
                        "sha256": str(raw["sha256"]),
                        "size": int(raw["size"]),
                    }
                )
            actual_id = _package_digest(
                config,
                source_format=str(manifest["source_format"]),
                assets=assets,
                adaptation=manifest["adaptation"],
            )
            if actual_id != expected_id:
                raise ScenePackageCorruptError(
                    "Scene package canonical digest does not match its package ID."
                )
            return ScenePackageRef(
                package_id=expected_id,
                package_path=package_dir.resolve(),
                config_path=config_path.resolve(),
                manifest=deepcopy(manifest),
                z_rotation_degrees=manifest["adaptation"]["z_rotation_degrees"],
                body_scale_policy=manifest["adaptation"]["body_scale_policy"],
                body_scale=tuple(manifest["adaptation"]["body_scale"]),
            )
        except ScenePackageCorruptError:
            raise
        except (OSError, TypeError, ValueError) as exc:
            raise ScenePackageCorruptError(
                f"Scene package {expected_id!r} is corrupt: {exc}"
            ) from exc


def _data_bank_root(value: str | Path | None) -> Path:
    if value is not None:
        return Path(value).expanduser().resolve()
    configured = os.environ.get("EMBODICHAIN_DATA_BANK")
    if configured:
        return Path(configured).expanduser().resolve()
    xdg_home = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg_home).expanduser() if xdg_home else Path.home() / ".local" / "share"
    return (base / "embodichain" / "data_bank").resolve()


def _coerce_source_ref(value: SceneSourceRef | str | Path) -> SceneSourceRef:
    return value if isinstance(value, SceneSourceRef) else SceneSourceRef(value)


def _with_robot_profile(value: ScenePackageRef, profile: str) -> ScenePackageRef:
    return ScenePackageRef(
        package_id=value.package_id,
        package_path=value.package_path,
        config_path=value.config_path,
        manifest=value.manifest,
        robot_profile=str(profile),
        z_rotation_degrees=value.z_rotation_degrees,
        body_scale_policy=value.body_scale_policy,
        body_scale=value.body_scale,
    )


def _with_source_ref(
    value: ScenePackageRef,
    source: SceneSourceRef,
) -> ScenePackageRef:
    return ScenePackageRef(
        package_id=value.package_id,
        package_path=value.package_path,
        config_path=value.config_path,
        manifest=value.manifest,
        robot_profile=source.robot_profile,
        z_rotation_degrees=source.z_rotation_degrees,
        body_scale_policy=source.body_scale_policy,
        body_scale=source.body_scale,
    )


def _validate_package_id(value: Any) -> str:
    package_id = str(value).strip().lower()
    if len(package_id) != 64 or any(
        char not in "0123456789abcdef" for char in package_id
    ):
        raise ValueError("Scene package ID must be a 64-character SHA-256 digest.")
    return package_id


def _read_json(path: Path, *, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid {context} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{context.capitalize()} must contain a JSON object: {path}")
    return value


def _package_assets(
    config: Mapping[str, Any],
    *,
    source_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result = deepcopy(dict(config))
    by_target: dict[str, dict[str, Any]] = {}

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in list(value.items()):
                if (
                    str(key) == "fpath"
                    and isinstance(child, (str, os.PathLike))
                    and str(child)
                ):
                    raw_path = Path(child).expanduser()
                    if raw_path.is_absolute():
                        source_path = raw_path.resolve(strict=True)
                    else:
                        if ".." in raw_path.parts:
                            raise ValueError(
                                "Relative scene asset paths may not traverse outside "
                                f"the scene export: {raw_path}"
                            )
                        source_root = source_dir.resolve(strict=True)
                        source_path = (source_root / raw_path).resolve(strict=True)
                        if (
                            source_path != source_root
                            and source_root not in source_path.parents
                        ):
                            raise ValueError(
                                "Relative scene asset path escapes the scene export: "
                                f"{raw_path}"
                            )
                    if not source_path.is_file():
                        raise FileNotFoundError(
                            f"Scene asset is not a file: {source_path}"
                        )
                    digest = _sha256_file(source_path)
                    suffix = source_path.suffix.lower()
                    relative = Path("assets") / f"{digest}{suffix}"
                    value[key] = relative.as_posix()
                    by_target.setdefault(
                        relative.as_posix(),
                        {
                            "path": relative.as_posix(),
                            "source_path": source_path,
                            "sha256": digest,
                            "size": source_path.stat().st_size,
                        },
                    )
                else:
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(result)
    return result, [by_target[key] for key in sorted(by_target)]


def _package_digest(
    config: Mapping[str, Any],
    *,
    source_format: str,
    assets: Sequence[Mapping[str, Any]],
    adaptation: Mapping[str, Any],
) -> str:
    canonical_config = _without_ephemeral_scene_identity(config)
    payload = {
        "adapter_policy_version": _ADAPTER_POLICY_VERSION,
        "source_format": source_format,
        "adaptation": deepcopy(dict(adaptation)),
        "config": canonical_config,
        "assets": [
            {
                "path": str(asset["path"]),
                "sha256": str(asset["sha256"]),
                "size": int(asset["size"]),
            }
            for asset in sorted(assets, key=lambda item: str(item["path"]))
        ],
    }
    return _sha256_bytes(_canonical_json_bytes(payload))


def _without_ephemeral_scene_identity(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _without_ephemeral_scene_identity(child)
            for key, child in value.items()
            if str(key).strip().lower()
            not in {"scene_id", "created_at", "updated_at", "timestamp"}
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_without_ephemeral_scene_identity(child) for child in value]
    return value


def _validate_manifest(value: Mapping[str, Any], *, expected_id: str) -> None:
    if set(value) != _PACKAGE_KEYS:
        raise ScenePackageCorruptError(
            f"Scene package manifest fields must be exactly {sorted(_PACKAGE_KEYS)}."
        )
    if value["schema_version"] != _PACKAGE_SCHEMA:
        raise ScenePackageCorruptError("Unsupported scene package schema version.")
    if value["adapter_policy_version"] != _ADAPTER_POLICY_VERSION:
        raise ScenePackageCorruptError("Unsupported scene adapter policy version.")
    if value["package_id"] != expected_id:
        raise ScenePackageCorruptError(
            "Manifest package ID does not match its CAS path."
        )
    if not isinstance(value["source_format"], str) or not value["source_format"]:
        raise ScenePackageCorruptError("Manifest source_format must be non-empty.")
    _validate_adaptation(value["adaptation"])
    _validate_relative_path(value["config_path"], label="config_path")
    _validate_hex_digest(value["config_sha256"], label="config_sha256")
    raw_assets = value["assets"]
    if not isinstance(raw_assets, list):
        raise ScenePackageCorruptError("Manifest assets must be a list.")
    seen: set[str] = set()
    for index, raw in enumerate(raw_assets):
        if not isinstance(raw, Mapping) or set(raw) != _ASSET_KEYS:
            raise ScenePackageCorruptError(
                f"Manifest asset {index} fields must be exactly {sorted(_ASSET_KEYS)}."
            )
        path = _validate_relative_path(raw["path"], label=f"assets[{index}].path")
        if path in seen:
            raise ScenePackageCorruptError(f"Duplicate packaged asset path {path!r}.")
        seen.add(path)
        _validate_hex_digest(raw["sha256"], label=f"assets[{index}].sha256")
        if (
            not isinstance(raw["size"], int)
            or isinstance(raw["size"], bool)
            or raw["size"] < 0
        ):
            raise ScenePackageCorruptError(f"Manifest assets[{index}].size is invalid.")


def _validate_relative_path(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ScenePackageCorruptError(f"Manifest {label} must be a non-empty path.")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ScenePackageCorruptError(
            f"Manifest {label} must be a normalized relative path."
        )
    return value


def _safe_package_path(root: Path, relative: str) -> Path:
    _validate_relative_path(relative, label="referenced path")
    root_resolved = root.resolve()
    candidate = (root / relative).resolve(strict=False)
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise ScenePackageCorruptError("Scene package path escapes the package root.")
    return candidate


def _verify_file(
    path: Path,
    *,
    expected_hash: str,
    label: str,
    expected_size: int | None = None,
) -> None:
    if path.is_symlink() or not path.is_file():
        raise ScenePackageCorruptError(
            f"Referenced {label} is missing or is a symlink: {path}"
        )
    if expected_size is not None and path.stat().st_size != expected_size:
        raise ScenePackageCorruptError(
            f"Referenced {label} has an unexpected size: {path}"
        )
    if _sha256_file(path) != expected_hash:
        raise ScenePackageCorruptError(
            f"Referenced {label} failed SHA-256 verification: {path}"
        )


def _validate_hex_digest(value: Any, *, label: str) -> None:
    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ScenePackageCorruptError(f"Manifest {label} must be a SHA-256 digest.")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _adaptation_policy(source: SceneSourceRef) -> dict[str, Any]:
    value = {
        "z_rotation_degrees": source.z_rotation_degrees,
        "body_scale_policy": source.body_scale_policy,
        "body_scale": list(source.body_scale),
    }
    return _validate_adaptation(value)


def _validate_adaptation(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _ADAPTATION_KEYS:
        raise ScenePackageCorruptError("Scene package adaptation fields are invalid.")
    rotation = value["z_rotation_degrees"]
    if rotation is not None and (
        isinstance(rotation, bool)
        or not isinstance(rotation, (int, float))
        or not math.isfinite(float(rotation))
    ):
        raise ScenePackageCorruptError(
            "Scene package z_rotation_degrees must be finite or null."
        )
    policy = value["body_scale_policy"]
    if policy not in {"preserve", "multiply", "absolute"}:
        raise ScenePackageCorruptError("Scene package body_scale_policy is invalid.")
    scale = value["body_scale"]
    if (
        not isinstance(scale, Sequence)
        or isinstance(scale, (str, bytes))
        or len(scale) != 3
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or float(item) <= 0.0
            for item in scale
        )
    ):
        raise ScenePackageCorruptError(
            "Scene package body_scale must contain three positive finite values."
        )
    return {
        "z_rotation_degrees": None if rotation is None else float(rotation),
        "body_scale_policy": str(policy),
        "body_scale": [float(item) for item in scale],
    }


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
