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

"""Install the pinned robot and mesh inputs used to qualify the cloth-fold task.

The wheel owns the adjusted command stream; this installs its matching URDF,
collision/visual meshes and cloth mesh into the ordinary EmbodiChain data root.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import urllib.request
import zipfile

from embodichain.data.constants import EMBODICHAIN_DEFAULT_DATA_ROOT
from embodichain.data.dataset import _dataset_download_lock

__all__: list[str] = []

_URL = (
    "https://huggingface.co/datasets/DexForceAI/embodichain_reports/resolve/"
    "cafe3e14f650f93123d47fe338e49c003443f3a4/reports/"
    "dexuni-physics-2026-10-27/inputs/m02_cloth_single_fold/"
    "m02_cloth_single_fold.zip"
)
_ARCHIVE_SHA256 = "d6059987b9d5bc6bc820e884655b5cb6657477699f8580f249168f64795ba157"
_FILES = {
    "w1_v030/robot.urdf": "5a800a7811c2d6d8012e7f0a09c55509e84b7b48bf3658bc7b73ac2d66c1e558",
    "cloth_mesh.npz": "a99816d80199900c2180b97be83835acf7f7c148d89007469d37e731d92f7b79",
}


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest()


def _prepare(data_root: Path) -> None:
    """Install the exact qualified cloth inputs without replacing local variants."""
    data_root = data_root.expanduser().resolve()
    prefix = "ClothFoldQualifiedInputs"
    with _dataset_download_lock(str(data_root), prefix):
        for name, digest in _FILES.items():
            path = data_root / name
            if path.exists() and _sha256(path) != digest:
                raise ValueError(
                    f"Existing qualified input has a different digest: {path}"
                )
        archive = data_root / "download" / prefix / "m02_cloth_single_fold.zip"
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.exists() or _sha256(archive) != _ARCHIVE_SHA256:
            temporary = archive.with_suffix(".partial")
            urllib.request.urlretrieve(_URL, temporary)
            if _sha256(temporary) != _ARCHIVE_SHA256:
                temporary.unlink()
                raise ValueError("Cloth input archive SHA-256 mismatch.")
            temporary.replace(archive)
        extracted = data_root / "extract" / prefix
        with zipfile.ZipFile(archive) as bundle:
            for entry in bundle.infolist():
                if (
                    not (extracted / entry.filename)
                    .resolve()
                    .is_relative_to(extracted.resolve())
                ):
                    raise ValueError("Cloth input archive contains an escaping path.")
            bundle.extractall(extracted)
        inputs = extracted / "m02_cloth_single_fold" / "inputs"
        for name, digest in _FILES.items():
            if _sha256(inputs / name) != digest:
                raise ValueError(f"Qualified input SHA-256 mismatch: {name}")
        # Copy the complete URDF-relative mesh tree, including collision meshes.
        sources = [*(inputs / "w1_v030").rglob("*"), inputs / "cloth_mesh.npz"]
        for source in sources:
            if not source.is_file():
                continue
            destination = data_root / source.relative_to(inputs)
            if destination.exists() and _sha256(source) != _sha256(destination):
                raise ValueError(
                    f"Existing input differs from the qualified bundle: {destination}"
                )
        for source in sources:
            if source.is_file():
                destination = data_root / source.relative_to(inputs)
                if not destination.exists():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
        for name in _FILES:
            print(data_root / name)


def main() -> None:
    """Install pinned, hash-verified inputs for the packaged cloth task."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=Path(EMBODICHAIN_DEFAULT_DATA_ROOT)
    )
    args = parser.parse_args()
    _prepare(args.data_root)


if __name__ == "__main__":
    main()
