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

"""Pin and prepare the public inputs to the Cube showcase."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
from tempfile import TemporaryDirectory
from typing import Any

from .._official_cube import CUBE_SWM_REV, CUBE_WEIGHTS_REV, OGBENCH_REV, _revision
from .._reference import _sha

_ARCHIVE_REV = "02a19a67a0dc8c9d6215f89c19e0a597691e152a"
_ARCHIVE_SHA = "3725d6a01abd492164441ef0a27e588f52b94a118fab56b96987b1a34a6c2600"
_DATA_BYTES = 101942558720
_SOURCE_REPOS = {
    "stable-worldmodel": (
        "https://github.com/galilai-group/stable-worldmodel.git",
        CUBE_SWM_REV,
    ),
    "ogbench": ("https://github.com/seohongpark/ogbench.git", OGBENCH_REV),
}


@dataclass(frozen=True)
class _CubeResources:
    root: Path

    @property
    def data(self) -> Path:
        return self.root / "data/cube_single_expert.h5"

    @property
    def weights(self) -> Path:
        return self.root / "weights/weights.pt"

    @property
    def queries(self) -> Path:
        return self.root / "queries/official-queries.h5"

    @property
    def reference(self) -> Path:
        return self.root / "native-reference"

    @property
    def swm(self) -> Path:
        return self.root / "upstream/stable-worldmodel"

    @property
    def ogbench(self) -> Path:
        return self.root / "upstream/ogbench"

    def require_prepared(self) -> None:
        for path in (
            self.root / "resources.json",
            self.data,
            self.weights,
            self.queries,
            self.queries.with_suffix(".json"),
            self.reference / "dexuni-render-materials.json",
            self.swm,
            self.ogbench,
        ):
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing Cube resource {path}. Run prepare first."
                )


def _checkout(destination: Path, url: str, revision: str, local_only: bool) -> None:
    if not destination.exists():
        if local_only:
            raise FileNotFoundError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=destination.parent) as scratch:
            staging = Path(scratch) / destination.name
            staging.mkdir()
            for args in (
                ["init", "--quiet"],
                ["fetch", "--quiet", "--depth", "1", url, revision],
                ["checkout", "--quiet", "--detach", revision],
            ):
                subprocess.run(["git", "-C", str(staging), *args], check=True)
            _revision(staging, revision)
            staging.rename(destination)
    _revision(destination, revision)


def _extract_dataset(archive: Path, destination: Path) -> None:
    import zstandard

    if _sha(archive) != _ARCHIVE_SHA:
        raise ValueError("Official Cube archive checksum mismatch")
    with TemporaryDirectory(dir=destination.parent) as scratch:
        target = Path(scratch) / destination.name
        with archive.open("rb") as compressed:
            with zstandard.ZstdDecompressor().stream_reader(compressed) as stream:
                with tarfile.open(fileobj=stream, mode="r|") as bundle:
                    for member in bundle:
                        if Path(member.name).name != destination.name:
                            continue
                        if not member.isfile() or member.size != _DATA_BYTES:
                            raise ValueError("Unexpected Cube HDF5 archive member")
                        with (
                            bundle.extractfile(member) as source,
                            target.open("xb") as out,
                        ):
                            shutil.copyfileobj(source, out, 1024 * 1024)
                        break
        if not target.is_file() or target.stat().st_size != _DATA_BYTES:
            raise ValueError("Official Cube HDF5 file is absent or incomplete")
        target.rename(destination)


def prepare_cube(
    root: Path,
    *,
    local_files_only: bool,
    reference_python: Path | None = None,
) -> dict[str, Any]:
    """Fetch checked inputs and compile the reference and native camera assets."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    from huggingface_hub import hf_hub_download

    from .._official_cube import audit_reference
    from .._official_data import prepare_queries
    from .._official_native import _OFFICIAL_WEIGHTS_SHA256, prepare_dexuni_reference

    resources = _CubeResources(root.resolve())
    resources.root.mkdir(parents=True, exist_ok=True)
    for name, (url, rev) in _SOURCE_REPOS.items():
        _checkout(resources.root / "upstream" / name, url, rev, local_files_only)
    if not resources.weights.is_file():
        hf_hub_download(
            "quentinll/lewm-cube",
            "weights.pt",
            revision=CUBE_WEIGHTS_REV,
            local_dir=resources.weights.parent,
            local_files_only=local_files_only,
        )
    if _sha(resources.weights) != _OFFICIAL_WEIGHTS_SHA256:
        raise ValueError("Official Cube checkpoint checksum mismatch")
    if not resources.data.is_file():
        resources.data.parent.mkdir(parents=True, exist_ok=True)
        archive = hf_hub_download(
            "quentinll/lewm-cube",
            "cube_single_expert.tar.zst",
            revision=_ARCHIVE_REV,
            repo_type="dataset",
            local_dir=resources.data.parent,
            local_files_only=local_files_only,
        )
        _extract_dataset(Path(archive), resources.data)
    queries = prepare_queries(resources.data, resources.queries)
    if not resources.reference.is_dir():
        with TemporaryDirectory(dir=resources.root) as scratch:
            compiled = Path(scratch) / "reference"
            prepared = Path(scratch) / "native-reference"
            if reference_python is None:
                audit_reference(compiled, resources.swm, resources.ogbench, steps=1)
            else:
                subprocess.run(
                    [
                        str(reference_python.absolute()),
                        "-m",
                        "embodichain",
                        "run-example",
                        "learning/lewm",
                        "official-reference",
                        "--steps",
                        "1",
                        "--swm-root",
                        str(resources.swm),
                        "--ogbench-root",
                        str(resources.ogbench),
                        "--output",
                        str(compiled),
                    ],
                    check=True,
                )
            prepare_dexuni_reference(compiled, prepared)
            prepared.rename(resources.reference)
    renderer = json.loads(
        (resources.reference / "dexuni-render-materials.json").read_text()
    )
    for name, digest in renderer["sha256"].items():
        if _sha(resources.reference / name) != digest:
            raise ValueError(
                f"Prepared Cube mesh or material checksum mismatch: {name}"
            )
    report = {
        "reference_mujoco": json.loads(
            (resources.reference / "reference-report.json").read_text()
        )["runtime"]["mujoco"],
        "stable_worldmodel_revision": CUBE_SWM_REV,
        "ogbench_revision": OGBENCH_REV,
        "dataset_revision": _ARCHIVE_REV,
        "weights_revision": CUBE_WEIGHTS_REV,
        "dataset_sha256": queries["source_sha256"],
        "queries_sha256": queries["queries_sha256"],
        "weights_sha256": _OFFICIAL_WEIGHTS_SHA256,
        "native_assets_sha256": _sha(
            resources.reference / "dexuni-render-materials.json"
        ),
    }
    (resources.root / "resources.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
