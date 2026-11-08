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

"""Atomic publication and static verification of sealed declaration bundles."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
from types import MappingProxyType
from typing import Any

from .bundle_identity import BundleIdentity, verify_bundle_files
from .bundle_inputs import SOURCE_INPUT_PROVENANCE_FILE, _validated_provenance

__all__ = [
    "BundleSnapshot",
    "read_bundle_identity",
    "read_bundle_snapshot",
    "write_bundle_atomic",
]

_IDENTITY_FILE = "integration_fingerprint.json"


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate bundle identity field: {key}.")
        result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class BundleSnapshot:
    """Owned declaration bytes and their verified immutable identity.

    No simulator, configured adapter or historical execution evidence is stored.
    Recomposition must additionally check the effective runtime declarations.
    """

    identity: BundleIdentity
    files: Mapping[str, bytes]

    def __post_init__(self) -> None:
        identity = BundleIdentity.from_dict(self.identity)
        files = dict(self.files)
        verify_bundle_files(identity, files)
        _validated_provenance(
            files[SOURCE_INPUT_PROVENANCE_FILE], identity.to_dict()["external_inputs"]
        )
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "files", MappingProxyType(files))


def _regular_file(path: Path, root: Path) -> bytes:
    for selected in (path, *path.parents):
        if selected.is_symlink():
            raise ValueError(f"Bundle declarations cannot follow links: {selected}.")
        if selected == root:
            break
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(f"Bundle declaration is not a regular file: {path}.")
    return path.read_bytes()


def read_bundle_identity(root: str | Path) -> BundleIdentity:
    """Read and verify an envelope before declaration traversal or runtime imports.

    Args:
        root: Existing published declaration directory.

    Returns:
        Independently owned runtime identity under the existing regular-file,
        no-links and duplicate-field checks. Unsupported old contracts reject
        without inspecting their components or constructing runtime resources.
    """
    directory = Path(root).expanduser().absolute()
    envelope = json.loads(
        _regular_file(directory / _IDENTITY_FILE, directory),
        object_pairs_hook=_unique_object,
    )
    return BundleIdentity.from_dict(envelope)


def read_bundle_snapshot(
    root: str | Path,
    *,
    external_input_digests: Mapping[str, str],
    implementation_revisions: Mapping[str, str],
) -> BundleSnapshot:
    """Reject unsupported or changed bundles before decoding runtime components.

    Args:
        root: Published declaration directory, excluding run outputs.
        external_input_digests: Current owner-resolved input hashes, not values
            copied from the envelope being checked.
        implementation_revisions: Current selected implementation revisions.

    Returns:
        Owned verified bytes suitable for static configured-deployment preflight.

    Raises:
        ValueError: Contract, file inventory, bytes or current inputs disagree.
        OSError: Required declaration files cannot be read.
    """
    directory = Path(root).expanduser().resolve()
    # Version rejection deliberately precedes file traversal and runtime imports.
    identity = read_bundle_identity(directory)
    payload = identity.to_dict()
    if dict(external_input_digests) != payload["external_inputs"]:
        raise ValueError("Bundle external inputs differ from the current owner inputs.")
    if dict(implementation_revisions) != payload["implementation_revisions"]:
        raise ValueError("Bundle implementation revisions differ; rebuild the bundle.")
    expected = set(payload["component_files"]) | {_IDENTITY_FILE}
    actual = set()
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Bundle declarations cannot follow links: {path}.")
        if not path.is_dir():
            actual.add(path.relative_to(directory).as_posix())
    if actual != expected:
        raise ValueError("Published bundle file inventory differs from its identity.")
    return BundleSnapshot(
        identity,
        {
            name: _regular_file(directory / name, directory)
            for name in payload["component_files"]
        },
    )


def _publish_directory(staging: Path, destination: Path) -> None:
    """Atomically publish without replacing even an existing empty directory."""
    if sys.platform == "win32":
        os.rename(staging, destination)
        return
    if not sys.platform.startswith("linux"):
        raise NotImplementedError(
            "Atomic no-replace bundle publication requires Linux or Windows."
        )
    import ctypes

    library = ctypes.CDLL(None, use_errno=True)
    rename = getattr(library, "renameat2", None)
    if rename is None:
        raise NotImplementedError("Atomic bundle publication requires renameat2.")
    rename.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(staging), -100, os.fsencode(destination), 1) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def write_bundle_atomic(
    root: str | Path,
    snapshot: BundleSnapshot,
    *,
    validate_inputs: Callable[[], None] | None = None,
) -> Path:
    """Publish a preflighted, sealed declaration set as one fresh directory.

    All hashes are rechecked before any directory is created. A private sibling
    staging directory is removed on failure; an existing destination is never
    replaced. A sealed ancestor rejects before staging, including symbolic-link
    aliases. This function does not preflight capabilities itself: callers must
    use the public configured deployment loader before sealing the snapshot.
    Runtime outputs and physical certificates belong to separate run storage.
    An explicit input validator runs before any writes and again immediately
    before publication; caller-owned qualification inputs need no filesystem
    interpretation by this byte-oriented writer.

    Args:
        root: Fresh destination outside existing sealed declarations.
        snapshot: Owned preflighted declaration bytes and their identity.
        validate_inputs: Explicit source-owner validation callback, when supplied.

    Returns:
        Published destination after the atomic no-replace rename.
    """
    owned = BundleSnapshot(snapshot.identity, snapshot.files)
    if validate_inputs is not None:
        if not callable(validate_inputs):
            raise TypeError("validate_inputs must be callable or None.")
        validate_inputs()
    destination = Path(root).expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Bundle destination already exists: {destination}.")
    from .orchestration.workflow_artifacts import validate_run_output_directory

    destination = validate_run_output_directory(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.partial-", dir=destination.parent)
    )
    try:
        files = dict(owned.files)
        files[_IDENTITY_FILE] = (
            json.dumps(owned.identity.to_dict(), sort_keys=True, indent=2) + "\n"
        ).encode()
        for name, content in files.items():
            path = staging / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        if validate_inputs is not None:
            validate_inputs()
        validate_run_output_directory(destination)
        _publish_directory(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return destination
