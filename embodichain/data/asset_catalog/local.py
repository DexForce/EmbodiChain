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

"""Local catalog providers with verified, content-addressed asset packages.

Each package contains ``asset.catalog.json`` with a descriptor, a relative
``visual_path``, and optional ``collision_path`` (defaults to the visual model).
An optional ``simready_metadata`` path imports existing SimReady category and
provenance; normalization and usable regions always require explicit annotation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import struct
import tempfile
from typing import Any
from urllib.parse import unquote, urlparse

from .contracts import (
    AssetCandidate,
    AssetDescriptor,
    AssetQuery,
    PreparedAsset,
    PreparationContext,
)

__all__ = [
    "DirectoryAssetProvider",
    "ExplicitAssetProvider",
    "verify_prepared_asset",
]

_MANIFEST = "asset.catalog.json"
_FORMATS = {".obj", ".glb", ".gltf", ".stl", ".ply"}


def _relative_file(root: Path, value: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Package paths must be non-empty relative strings.")
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"Package path must be relative: {value}")
    target = root / relative
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Package dependency escapes its root: {value}")
    # Disallow even in-package links: each byte must be owned by the snapshot.
    current = target
    while current != root:
        if current.is_symlink():
            raise ValueError(f"Package symlinks are unsupported: {value}")
        if current == current.parent:
            break
        current = current.parent
    if not target.is_file():
        raise ValueError(f"Missing package dependency: {value}")
    return target.resolve()


def _read_manifest(path: Path) -> tuple[AssetDescriptor, str, str]:
    if path.is_symlink():
        raise ValueError("Catalog manifests cannot be symlinks.")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Asset catalog manifest must be an object.")
    unknown = set(raw) - {
        "descriptor",
        "visual_path",
        "collision_path",
        "simready_metadata",
    }
    if unknown:
        raise ValueError(f"Unknown asset catalog fields: {sorted(unknown)}")
    if "descriptor" not in raw or "visual_path" not in raw:
        raise ValueError("Catalog requires descriptor and visual_path.")
    descriptor_data = dict(raw["descriptor"])
    if "simready_metadata" in raw:
        metadata_file = _relative_file(path.parent, raw["simready_metadata"])
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("SimReady metadata must be an object.")
        identity = metadata.get("identity") or {}
        if not isinstance(identity, dict):
            raise ValueError("SimReady identity must be an object.")
        if descriptor_data.get("category") is None:
            descriptor_data["category"] = identity.get("category") or metadata.get(
                "category"
            )
        provenance = dict(descriptor_data.get("provenance", {}))
        provenance["simready"] = {
            "source": raw["simready_metadata"],
            "asset_id": metadata.get("asset_id"),
            "category_evidence": "imported",
            "provenance": metadata.get("provenance", {}),
        }
        descriptor_data["provenance"] = provenance
    descriptor = AssetDescriptor.from_dict(descriptor_data)
    visual = raw["visual_path"]
    collision = raw.get("collision_path", visual)
    for filename in (visual, collision):
        model = _relative_file(path.parent, filename)
        if model.suffix.lower() not in _FORMATS:
            raise ValueError(f"Unsupported rigid model format: {model.suffix}")
    return descriptor, visual, collision


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory(root: Path) -> dict[str, str]:
    inventory: dict[str, str] = {}
    for directory, folders, files in os.walk(root, followlinks=False):
        for name in folders + files:
            path = Path(directory) / name
            if path.is_symlink():
                raise ValueError(f"Package symlinks are unsupported: {path}")
        for name in files:
            path = Path(directory) / name
            if not path.is_file():
                raise ValueError(f"Package contains a nonregular file: {path}")
            inventory[path.relative_to(root).as_posix()] = _file_hash(path)
    return dict(sorted(inventory.items()))


def _json_chunk(path: Path) -> Mapping[str, Any]:
    if path.suffix.lower() == ".gltf":
        result = json.loads(path.read_text(encoding="utf-8"))
    else:
        with path.open("rb") as stream:
            header = stream.read(12)
            if len(header) != 12:
                raise ValueError(f"Invalid GLB header: {path}")
            magic, version, length = struct.unpack("<4sII", header)
            if magic != b"glTF" or version != 2 or length != path.stat().st_size:
                raise ValueError(f"Invalid GLB header: {path}")
            chunk_header = stream.read(8)
            if len(chunk_header) != 8:
                raise ValueError(f"Missing GLB JSON chunk: {path}")
            chunk_length, kind = struct.unpack("<I4s", chunk_header)
            if kind != b"JSON" or chunk_length > length - 20:
                raise ValueError(f"Invalid GLB JSON chunk: {path}")
            result = json.loads(stream.read(chunk_length))
    if not isinstance(result, dict):
        raise ValueError(f"Invalid glTF document: {path}")
    return result


def _dependency(root: Path, parent: Path, locator: str) -> Path:
    parsed = urlparse(locator)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError(f"External asset references are unsupported: {locator}")
    target = parent / unquote(locator)
    return _relative_file(root, os.path.relpath(target, root))


def _check_dependencies(root: Path, model: Path) -> None:
    suffix = model.suffix.lower()
    if suffix in (".glb", ".gltf"):
        document = _json_chunk(model)
        for field in ("buffers", "images"):
            for item in document.get(field, []):
                uri = item.get("uri")
                if uri is not None and not uri.startswith("data:"):
                    _dependency(root, model.parent, uri)
    elif suffix == ".obj":
        for line in model.read_text(encoding="utf-8").splitlines():
            tokens = shlex.split(line, comments=True)
            if tokens and tokens[0].lower() == "mtllib":
                for locator in tokens[1:]:
                    material = _dependency(root, model.parent, locator)
                    for material_line in material.read_text(
                        encoding="utf-8"
                    ).splitlines():
                        values = shlex.split(material_line, comments=True)
                        if values and (
                            values[0].lower().startswith("map_")
                            or values[0].lower()
                            in {"bump", "disp", "decal", "refl", "norm"}
                        ):
                            if len(values) < 2:
                                raise ValueError(
                                    f"Missing texture reference in {material}"
                                )
                            _dependency(root, material.parent, values[-1])


def _check_geometry(descriptor: AssetDescriptor, visual: Path, collision: Path) -> None:
    # Lazy import keeps source listing and catalog contracts independent of mesh tooling.
    import trimesh

    if not descriptor.normalized or descriptor.bounds_m is None:
        raise ValueError(
            "Asset requires explicit normalized meter/Z-up geometry and bounds_m annotations."
        )
    for model in dict.fromkeys((visual, collision)):
        mesh = trimesh.load(str(model), force="mesh", process=False)
        bounds = mesh.bounds
        if bounds is None or not len(mesh.vertices) or not len(mesh.faces):
            raise ValueError(f"Model has no usable rigid mesh: {model}")
        if any(not math.isfinite(float(v)) for row in bounds for v in row):
            raise ValueError(f"Model geometry is not finite: {model}")
        if model == visual:
            expected = (descriptor.bounds_m.lower, descriptor.bounds_m.upper)
            if any(
                not math.isclose(float(actual), wanted, rel_tol=1e-5, abs_tol=1e-6)
                for row, target in zip(bounds, expected)
                for actual, wanted in zip(row, target)
            ):
                raise ValueError(
                    "Declared bounds_m does not match the visual mesh in meters."
                )


def _content_digest(descriptor: AssetDescriptor, inventory: Mapping[str, str]) -> str:
    payload = {"descriptor": descriptor.to_dict(), "files": dict(inventory)}
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def verify_prepared_asset(asset: PreparedAsset) -> None:
    """Verify the common prepared package contract before copy or execution.

    Any provider can produce this format: ``provenance.package_root`` names an
    absolute package directory and ``provenance.files`` maps every relative file
    to its SHA-256. The package contains ``asset.catalog.json``. Content identity
    is SHA-256 of UTF-8 JSON ``{"descriptor": descriptor.to_dict(), "files": files}``,
    with sorted keys, compact separators and no nonfinite values. Provider
    provenance outside the descriptor is deliberately excluded from identity.

    Args:
        asset: Prepared snapshot from any provider using the common package format.

    Raises:
        ValueError: Provenance is absent or package metadata/files have changed.
    """
    package_root = asset.provenance.get("package_root")
    recorded_files = asset.provenance.get("files")
    if not isinstance(package_root, str) or not isinstance(recorded_files, Mapping):
        raise ValueError("Prepared assets require package_root and files provenance.")
    root = Path(package_root)
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise ValueError("Prepared package root is missing or invalid.")
    inventory = _inventory(root)
    if inventory != dict(recorded_files):
        raise ValueError("Prepared package files changed after materialization.")
    if _content_digest(asset.descriptor, inventory) != asset.content_sha256:
        raise ValueError("Prepared content identity does not match package bytes.")
    descriptor, visual, collision = _read_manifest(root / _MANIFEST)
    if descriptor != asset.descriptor:
        raise ValueError("Prepared descriptor does not match package metadata.")
    if (root / visual).resolve() != Path(asset.visual_path).resolve() or (
        root / collision
    ).resolve() != Path(asset.collision_path).resolve():
        raise ValueError("Prepared model paths do not match package metadata.")


class ExplicitAssetProvider:
    """Read a fixed collection of local asset packages.

    Args:
        paths: Catalog manifests, package directories, or model files with an
            adjacent ``asset.catalog.json`` naming that model.

    Raw models without annotated catalogs are rejected with a preparation
    diagnostic. This provider does not invoke online generation or SimReady.
    """

    def __init__(self, paths: Sequence[str | Path]) -> None:
        if isinstance(paths, (str, Path)):
            raise TypeError("paths must be a sequence of asset locations.")
        manifests: set[Path] = set()
        for value in paths:
            source = Path(value).absolute()
            if source.is_symlink():
                raise ValueError(f"Asset locations cannot be symlinks: {source}")
            manifest = (
                source / _MANIFEST
                if source.is_dir()
                else (source if source.name == _MANIFEST else source.parent / _MANIFEST)
            )
            if not manifest.is_file():
                raise ValueError(
                    f"Missing {_MANIFEST}; annotate normalized geometry and semantic metadata for {source}."
                )
            descriptor, visual, collision = _read_manifest(manifest)
            if (
                source.is_file()
                and source != manifest
                and source.resolve()
                not in {
                    (manifest.parent / visual).resolve(),
                    (manifest.parent / collision).resolve(),
                }
            ):
                raise ValueError(f"Catalog does not describe selected model: {source}")
            manifests.add(manifest.resolve())
        if not manifests:
            raise ValueError("At least one asset package is required.")
        self._manifests = tuple(sorted(manifests))

    def candidates(
        self, request: AssetQuery, *, limit: int
    ) -> tuple[AssetCandidate, ...]:
        """Rank known semantic matches without excluding task-level aliases.

        Args:
            request: Source-neutral category, property and capability constraints.
            limit: Positive maximum number of candidates.

        Returns:
            Stable candidate snapshots ordered by known matches and locator.
        """
        if not isinstance(request, AssetQuery):
            raise TypeError("request must be AssetQuery.")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer.")
        ranked = []
        for manifest in self._manifests:
            descriptor, _, _ = _read_manifest(manifest)
            score = int(
                request.category is not None and request.category == descriptor.category
            )
            score += sum(
                k in descriptor.attributes and descriptor.attributes[k] == v
                for k, v in request.attributes.items()
            )
            score += len(set(request.capabilities) & set(descriptor.capabilities))
            candidate = AssetCandidate(
                descriptor.asset_id, manifest.as_uri(), descriptor
            )
            ranked.append((-score, manifest.as_posix(), candidate))
        return tuple(row[2] for row in sorted(ranked)[:limit])

    def materialize(
        self, candidate: AssetCandidate, *, context: PreparationContext
    ) -> PreparedAsset:
        """Verify and cache a complete package, detecting missing or changed bytes.

        Args:
            candidate: Candidate returned by this provider.
            context: Cache destination outside the source package.

        Returns:
            Validated local geometry with immutable descriptor and file provenance.
        """
        matches = [
            path for path in self._manifests if path.as_uri() == candidate.source_uri
        ]
        if not matches:
            raise ValueError("Candidate does not belong to this provider.")
        manifest = matches[0]
        root = manifest.parent
        descriptor, visual, collision = _read_manifest(manifest)
        cache = Path(context.cache_dir).resolve()
        if cache.is_relative_to(root):
            raise ValueError("Preparation cache cannot be inside an asset package.")
        inventory = _inventory(root)
        for filename in dict.fromkeys((visual, collision)):
            _check_dependencies(root, root / filename)
        _check_geometry(descriptor, root / visual, root / collision)
        digest = _content_digest(descriptor, inventory)
        destination = cache / digest
        cache.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.is_symlink() or _inventory(destination) != inventory:
                raise ValueError(f"Prepared cache is corrupt: {destination}")
        else:
            temporary = Path(tempfile.mkdtemp(prefix=".prepare-", dir=cache))
            try:
                shutil.copytree(root, temporary, dirs_exist_ok=True, symlinks=True)
                if _inventory(temporary) != inventory:
                    raise ValueError(
                        "Source asset changed during preparation; retry from a stable snapshot."
                    )
                try:
                    temporary.rename(destination)
                except OSError:
                    # POSIX may report ENOTEMPTY rather than EEXIST when another
                    # preparer atomically publishes this same content package.
                    if not destination.exists():
                        raise
                    if destination.is_symlink() or _inventory(destination) != inventory:
                        raise ValueError(f"Prepared cache is corrupt: {destination}")
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
        if _read_manifest(destination / manifest.name) != (
            descriptor,
            visual,
            collision,
        ):
            raise ValueError(
                "Source metadata changed during preparation; retry from a stable snapshot."
            )
        return PreparedAsset(
            descriptor=descriptor,
            visual_path=str(destination / visual),
            collision_path=str(destination / collision),
            content_sha256=digest,
            provenance={
                "source_uri": candidate.source_uri,
                "package_root": str(destination),
                "files": inventory,
            },
        )


class DirectoryAssetProvider(ExplicitAssetProvider):
    """Discover annotated packages recursively under one local directory.

    Args:
        directory: Root containing one or more ``asset.catalog.json`` packages.
    """

    def __init__(self, directory: str | Path) -> None:
        root = Path(directory).absolute()
        if not root.is_dir() or root.is_symlink():
            raise ValueError("Asset directory must be an existing ordinary directory.")
        manifests = sorted(root.rglob(_MANIFEST))
        if not manifests:
            raise ValueError(
                f"No {_MANIFEST} packages found in {root}; raw files require explicit asset annotations."
            )
        super().__init__(manifests)
