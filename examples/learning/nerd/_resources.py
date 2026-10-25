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

"""Explicit, pinned resources for the NeRD showcase, without SDK registration."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
from typing import Any
import zipfile

__all__ = ["prepare_reference_assets", "prepare_resources"]

_ASSET_REPO = "DexForceAI/embodichain_data"
_ASSET_REVISION = "97e202e41fe7fb504656a6b163291305e400780d"
_ASSET_PREFIX = "demo/nerd/reference_assets/v1"
_ASSET_ROOT = "nerd_reference_assets"
_UPSTREAM_REVISION = "476dc67fccc6a99be293bf611f897b1c06f13e30"
_MANIFEST_SHA256 = "10dd356b1c818111c31d4db8b5011fa9c0df1bfc55b01632ed992ecd9806711e"
_ARCHIVE_SHA256 = "008d90d395422be81f8bbd9707fc05fafd358b44794b29ff0f6158a75cd1da22"
_ASSET_FILES = {
    "ATTRIBUTION.md": "4d86dc2651a4e24903162e8bac2a96122952d49d91c7ffb52a7fa28119438ca5",
    "LICENSE.md": "59899c6091b540582ed617e8eeaac4919dc985ccfc35459ee9752b699be5205b",
    "ant.xml": "23e9a720465a71fde3fd9fd2d5a18f8c11ec642f3dca8921c66cb4e169421fa7",
    "cartpole_single.urdf": "f31276531c03050a3ac3a3e24443899f38a4424586db17140978383433dec237",
}
_DATASET_REPO = "DexForceAI/embodichain_datasets"
_DATASET_REVISION = "b9ec58f36da99cc99a93f97adfe74ebc9fd66586"
_MODEL_REPO = "DexForceAI/embodichain_model"
_MODEL_REVISION = "78e581526570069410bf269b38c0c52c800b24a7"
_PAYLOAD_MANIFEST_SHA256 = {
    "dataset": {
        "cartpole": "1cba6f65535a4e84c23cab4e633623ed0d5ef4f8b85e67781c5a4316c4f14775",
        "ant": "5fe070411f3b61c87915ec61923d976b399059f0d93e071f119e58ed0e68127a",
    },
    "model": {
        "cartpole": "6e507149c9aacdc3f47665f91c3eb141cbcc0bd3bdcd697700a081d08077cf6a",
        "ant": "52f0975aed5cd5946647f8ae60c52be307d51956dda4d7e9b9b3e2414cd23e9f",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_hash(path: Path, expected: str) -> None:
    if not path.is_file() or _sha256(path) != expected:
        raise ValueError(f"Resource SHA256 mismatch: {path}")


def _valid_assets(path: Path) -> bool:
    if not path.is_dir() or path.is_symlink():
        return False
    try:
        if {entry.name for entry in path.iterdir()} != set(_ASSET_FILES):
            return False
        return all(
            not (path / name).is_symlink()
            and (path / name).is_file()
            and _sha256(path / name) == digest
            for name, digest in _ASSET_FILES.items()
        )
    except OSError:
        return False


def _asset_payloads(archive: Path, manifest_path: Path) -> dict[str, bytes]:
    _require_hash(manifest_path, _MANIFEST_SHA256)
    _require_hash(archive, _ARCHIVE_SHA256)
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema_version") != 1
        or manifest.get("license") != "Apache-2.0"
        or manifest["upstream"]["revision"] != _UPSTREAM_REVISION
        or manifest["archive"]["root"] != _ASSET_ROOT
        or manifest["archive"]["sha256"] != _ARCHIVE_SHA256
        or manifest["archive"]["size"] != archive.stat().st_size
        or set(manifest["files"]) != set(_ASSET_FILES)
    ):
        raise ValueError("Unexpected NeRD reference asset manifest")
    directory = f"{_ASSET_ROOT}/"
    expected = {directory, *(f"{directory}{name}" for name in _ASSET_FILES)}
    payloads = {}
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        if (
            not names
            or names[0] != directory
            or len(names) != len(set(names))
            or set(names) != expected
        ):
            raise ValueError("Unexpected NeRD archive member layout")
        for entry in bundle.infolist():
            name = entry.filename
            parts = name.rstrip("/").split("/")
            mode = entry.external_attr >> 16
            if (
                PurePosixPath(name).is_absolute()
                or "\\" in name
                or ":" in name
                or any(part in ("", ".", "..") for part in parts)
                or stat.S_ISLNK(mode)
            ):
                raise ValueError("Unsafe NeRD archive member")
            if entry.is_dir():
                if name != directory or entry.file_size:
                    raise ValueError("Unexpected NeRD archive directory")
                continue
            if stat.S_IFMT(mode) not in (0, stat.S_IFREG):
                raise ValueError("Unsupported NeRD archive member type")
            filename = PurePosixPath(name).name
            record = manifest["files"][filename]
            if (
                record["path"] != name
                or record["sha256"] != _ASSET_FILES[filename]
                or entry.file_size != record["size"]
                or entry.file_size > 1024 * 1024
            ):
                raise ValueError("Unexpected NeRD asset member identity")
            value = bundle.read(entry)
            if hashlib.sha256(value).hexdigest() != _ASSET_FILES[filename]:
                raise ValueError(f"NeRD member SHA256 mismatch: {filename}")
            payloads[filename] = value
    return payloads


def _resource_root() -> Path:
    """Resolve the showcase resource root without preparing any resources."""
    default = (
        Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        / "embodichain"
        / "examples"
        / "nerd"
    )
    return (
        Path(os.environ.get("EMBODICHAIN_NERD_RESOURCE_ROOT", default))
        .expanduser()
        .resolve()
    )


def prepare_reference_assets(
    cache_dir: str | Path | None = None, *, local_files_only: bool = False
) -> Path:
    """Download, verify and repair the four pinned reference asset files.

    Args:
        cache_dir: Explicit asset cache root, taking precedence over the example's
            ``EMBODICHAIN_NERD_RESOURCE_ROOT/assets``. Without either setting, use
            the showcase cache under ``XDG_CACHE_HOME`` or ``~/.cache``.
        local_files_only: Use verified extracted files or the existing Hub cache;
            missing downloads fail without network access.

    Returns:
        Directory containing the two robot files, license and attribution.
    """
    from filelock import FileLock

    if cache_dir is not None:
        base = Path(cache_dir).expanduser()
    else:
        base = _resource_root() / "assets"
    base = base.resolve()
    locks = base / ".locks"
    locks.mkdir(parents=True, exist_ok=True)
    with FileLock(str(locks / f"reference-assets-{_ASSET_REVISION}.lock")):
        revision = base / "reference_assets" / _ASSET_REVISION
        if revision.is_symlink() or not revision.resolve().is_relative_to(base):
            raise ValueError("Reference asset revision must remain inside its cache")
        target = revision / _ASSET_ROOT
        if _valid_assets(target):
            return target
        from huggingface_hub import hf_hub_download

        options = {
            "repo_id": _ASSET_REPO,
            "repo_type": "dataset",
            "revision": _ASSET_REVISION,
            "cache_dir": base / "hub",
            "local_files_only": local_files_only,
        }
        manifest = Path(
            hf_hub_download(filename=f"{_ASSET_PREFIX}/manifest.json", **options)
        )
        archive = Path(
            hf_hub_download(filename=f"{_ASSET_PREFIX}/reference_assets.zip", **options)
        )
        payloads = _asset_payloads(archive, manifest)
        revision.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".staging-", dir=revision) as temporary:
            staged = Path(temporary) / _ASSET_ROOT
            staged.mkdir()
            for name, value in payloads.items():
                (staged / name).write_bytes(value)
            if not _valid_assets(staged):
                raise ValueError("Extracted NeRD reference assets failed verification")
            if target.is_symlink() or (target.exists() and not target.is_dir()):
                target.unlink()
            elif target.exists():
                shutil.rmtree(target)
            staged.replace(target)
        return target


def _resource_file(root: Path, relative: str) -> Path:
    if (
        not isinstance(relative, str)
        or "\\" in relative
        or ":" in relative
        or PurePosixPath(relative).is_absolute()
        or any(part in ("", ".", "..") for part in relative.split("/"))
    ):
        raise ValueError("Unsafe published resource filename")
    target = root.joinpath(*PurePosixPath(relative).parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("Published resource resolves outside its directory")
    return target


def _verify_payload(path: Path, robot: str, kind: str) -> dict[str, Any]:
    _require_hash(path / "manifest.json", _PAYLOAD_MANIFEST_SHA256[kind][robot])
    manifest = json.loads((path / "manifest.json").read_text())
    if kind == "dataset":
        if (
            manifest.get("schema_version") != 1
            or manifest["metadata"]["codec"]["robot"] != robot
        ):
            raise ValueError("Unexpected NeRD dataset manifest")
        shards = manifest["shards"]
        if not shards or len({row["file"] for row in shards}) != len(shards):
            raise ValueError("NeRD dataset shards must be nonempty and unique")
        for row in shards:
            _require_hash(_resource_file(path, row["file"]), row["sha256"])
    else:
        if manifest.get("format_version") != 1 or manifest["codec"]["robot"] != robot:
            raise ValueError("Unexpected NeRD model manifest")
        _require_hash(path / "weights.pt", manifest["weights_sha256"])
    return manifest


def _prepare_payload(root: Path, robot: str, kind: str) -> tuple[Path, dict[str, Any]]:
    prefix = f"neural_dynamics/nerd/{robot}/v1"
    path = root / prefix
    return path, _verify_payload(path, robot, kind)


def prepare_resources(
    root: Path,
    robot: str,
    *,
    assets_only: bool = False,
    local_files_only: bool = False,
) -> dict[str, Path]:
    """Prepare pinned showcase assets and optional published data/model bundles.

    Args:
        root: Caller-owned resource root for assets, datasets and models.
        robot: ``cartpole`` or ``ant``.
        assets_only: Prepare assets without starting model or dataset downloads.
        local_files_only: Forbid remote access when a required cache is missing.

    Returns:
        Verified ``assets`` directory; also ``dataset`` and ``model`` directories
        when ``assets_only`` is false.
    """
    robot = robot.lower()
    if robot not in ("cartpole", "ant"):
        raise ValueError("NeRD resources support cartpole or ant")
    root = Path(root).expanduser().resolve()
    resources = {
        "assets": prepare_reference_assets(
            root / "assets", local_files_only=local_files_only
        )
    }
    if assets_only:
        return resources
    manifests = {}
    for kind, directory, repo, revision in (
        ("dataset", "datasets", _DATASET_REPO, _DATASET_REVISION),
        ("model", "models", _MODEL_REPO, _MODEL_REVISION),
    ):
        base = root / directory
        try:
            path, manifest = _prepare_payload(base, robot, kind)
        except (OSError, ValueError, KeyError, TypeError):
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=repo,
                repo_type="dataset" if kind == "dataset" else "model",
                revision=revision,
                allow_patterns=[f"neural_dynamics/nerd/{robot}/v1/*"],
                local_dir=base,
                local_files_only=local_files_only,
                max_workers=4,
            )
            path, manifest = _prepare_payload(base, robot, kind)
        resources[kind], manifests[kind] = path, manifest
    metadata = manifests["dataset"]["metadata"]
    for field in (
        "codec",
        "dt",
        "asset_sha256",
        "physics_signature",
        "gravity",
        "joint_names",
    ):
        if manifests["model"].get(field) != metadata.get(field):
            raise ValueError(f"Published NeRD model/dataset {field} mismatch")
    filename = "cartpole_single.urdf" if robot == "cartpole" else "ant.xml"
    if metadata.get("asset_sha256") != _ASSET_FILES[filename]:
        raise ValueError("Published NeRD resource/reference asset mismatch")
    return resources
