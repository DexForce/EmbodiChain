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
"""Explicit resource preparation with pinned archive and per-file identities."""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from urllib.request import urlopen
import zipfile

_ARCHIVE = "worldparticle-demo-assets-2eab8d200762.zip"
_SHA256 = "2eab8d200762df1dcab257be6ca5499dd28223d5a91f8e4ce8b3bec3fcf67a7b"
_URL = "http://192.168.3.120/CoreEngine/DexSim/resources/" + _ARCHIVE


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(root: Path, *, local_files_only: bool = False) -> None:
    manifest = json.loads(Path(__file__).with_name("assets-manifest.json").read_text())

    def verify() -> bool:
        return all(
            (root / item["path"]).is_file()
            and (root / item["path"]).stat().st_size == item["size"]
            and _sha256(root / item["path"]) == item["sha256"]
            for item in manifest["files"]
        )

    if verify():
        print(f"Verified {len(manifest['files'])} WorldParticle assets: {root}")
        return
    if local_files_only:
        raise ValueError(f"Missing or corrupt WorldParticle assets: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root.parent) as directory:
        archive = Path(directory) / _ARCHIVE
        with urlopen(_URL, timeout=60) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output)
        if _sha256(archive) != _SHA256:
            raise ValueError("WorldParticle archive SHA256 mismatch")
        with zipfile.ZipFile(archive) as bundle:
            # Extract only committed, verified file names, never archive paths.
            for item in manifest["files"]:
                relative = Path(item["path"])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("Invalid resource manifest path")
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with (
                    bundle.open("worldparticle/" + relative.as_posix()) as source,
                    target.open("wb") as output,
                ):
                    shutil.copyfileobj(source, output)
        if not verify():
            raise ValueError("Extracted WorldParticle file SHA256 mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assets", type=Path, default=Path("outputs/worldparticle/assets")
    )
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    prepare(args.assets, local_files_only=args.local_files_only)


if __name__ == "__main__":
    main()
