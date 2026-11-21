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

"""Filesystem snapshots for relocatable robot resource bundles."""

from __future__ import annotations

from pathlib import Path
import shutil

__all__ = ["snapshot_urdf_resources"]


def snapshot_urdf_resources(
    urdf_path: str | Path,
    directory: str | Path,
    *,
    provenance_root: str | Path | None = None,
) -> Path:
    """Copy a prepared URDF resource tree without importing simulation code.

    The caller supplies a prepared URDF whose relative meshes/materials live
    below its parent directory. Copying that complete tree preserves transitive
    OBJ/MTL/texture references without changing geometry or rewriting the URDF.
    An optional source-package root contributes its manifest and license/notice
    files from that root and its immediate model subdirectories. Model names
    and simulator configuration do not enter this filesystem operation.

    Args:
        urdf_path: Existing prepared URDF entry point.
        directory: Caller-owned destination for the standalone robot tree.
        provenance_root: Optional package root holding ``source_manifest.json``
            and original model license/notice files.

    Returns:
        Resolved entry-point path inside the destination.

    Raises:
        FileNotFoundError: The entry-point file does not exist.
        ValueError: A new snapshot destination is nested inside its source tree.
    """
    source = Path(urdf_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    target = Path(directory).resolve()
    if target != source.parent:
        if target.is_relative_to(source.parent):
            raise ValueError(
                "Robot snapshot destination must be outside its source tree."
            )
        shutil.copytree(source.parent, target, dirs_exist_ok=True)
    if provenance_root is not None:
        provenance = Path(provenance_root).resolve()
        manifest = provenance / "source_manifest.json"
        if manifest.is_file() and manifest != target / manifest.name:
            shutil.copy2(manifest, target / manifest.name)
        licenses = target / "licenses"
        for pattern in ("LICENSE*", "NOTICE*", "*/LICENSE*", "*/NOTICE*"):
            for path in sorted(provenance.glob(pattern)):
                if not path.is_file():
                    continue
                licenses.mkdir(parents=True, exist_ok=True)
                name = (
                    path.name
                    if path.parent == provenance
                    else f"{path.parent.name}.{path.name}"
                )
                destination = licenses / name
                if path.resolve() != destination.resolve():
                    shutil.copy2(path, destination)
    return target / source.name
