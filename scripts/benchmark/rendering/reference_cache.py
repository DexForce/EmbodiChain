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

"""Checksum-verified, exact-identity reference reuse across campaign experiments."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from scripts.benchmark.core.artifacts import read_json_object, stable_hash

__all__ = ["transfer_reference"]


def transfer_reference(
    source_root: Path, target_root: Path, key: str, identity: dict[str, Any]
) -> bool:
    """Copy one complete immutable reference closure after checking all sources.

    Args:
        source_root: Experiment or shared-cache root owning ``references/``.
        target_root: Independent destination trust boundary.
        key: SHA-256 of the exact camera/geometry/resolution/generator settings.
        identity: Expected complete reference identity, including seeds and spp.

    Returns:
        Whether a reference was available. Corrupt, incomplete or conflicting
        sources raise before publication. Hardlinks save space on one filesystem;
        consumers always reverify bytes and preserve their own proof bindings.
    """
    if key != stable_hash(identity):
        raise ValueError("Reference key does not match requested identity")
    source_root = Path(source_root).resolve()
    target_root = Path(target_root).resolve()
    source = source_root / "references" / key
    descriptor = source / "cache_binding.json"
    if not source.exists():
        return False
    binding = read_json_object(descriptor)
    if binding.get("identity") != identity or len(binding.get("artifacts", [])) != 3:
        raise ValueError("Shared reference identity or source closure mismatch")
    files = [(descriptor, "cache_binding.json")]
    for artifact in binding["artifacts"]:
        path = (source_root / artifact["path"]).resolve()
        if not path.is_relative_to(source) or not path.is_file():
            raise ValueError("Shared reference source escapes its cache entry")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != artifact["sha256"] or len(
            data
        ) != artifact.get("bytes", len(data)):
            raise ValueError("Shared reference checksum mismatch")
        files.append((path, path.relative_to(source).as_posix()))
    target = target_root / "references" / key
    if target.exists():
        for path, relative in files:
            other = target / relative
            if (
                not other.is_file()
                or hashlib.sha256(path.read_bytes()).digest()
                != hashlib.sha256(other.read_bytes()).digest()
            ):
                raise ValueError("Existing reference cache entry conflicts")
        return True
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".reference-", dir=target.parent))
    try:
        for path, relative in files:
            destination = temporary / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(path, destination)
            except OSError:
                shutil.copyfile(path, destination)
            if (
                hashlib.sha256(path.read_bytes()).digest()
                != hashlib.sha256(destination.read_bytes()).digest()
            ):
                raise ValueError("Reference changed during transfer")
        temporary.rename(target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return True
