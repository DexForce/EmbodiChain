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

"""Prepare an experiment's inputs before launching or timing its simulator."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Any

from filelock import FileLock

from .download import (
    DATASET_REVISION,
    REPORT_ID,
    _DEFAULT_CACHE,
    _file_matches,
    download_report_assets,
)
from .upstream import download_nvidia_assets

__all__ = [
    "ReportInputs",
    "add_asset_arguments",
    "add_runtime_arguments",
    "prepare_inputs",
    "runtime_python",
]

_BENCHMARKS = ("l-02", "l-03", "l-04", "p-01", "p-02", "p-03")
_GRASP_FILES = ("commands.npz", "fingers.npz", "kinematics.json", "asset-manifest.json")


@dataclass(frozen=True)
class ReportInputs:
    """Verified report bundle and paths for any external or derived inputs."""

    bundle: Path
    provenance: dict[str, Any]
    robots: dict[str, Path] = field(default_factory=dict)
    grasp_inputs: Path | None = None


def add_asset_arguments(parser: argparse.ArgumentParser) -> None:
    """Expose the same asset controls on all report launchers."""
    parser.add_argument(
        "--revision", default=DATASET_REVISION, help="Report dataset commit or tag"
    )
    parser.add_argument(
        "--cache-dir", type=Path, help="Cache for report and upstream assets"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use cached inputs without network requests",
    )


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    """Configure runtime executables separately from the asset resolver."""
    add_asset_arguments(parser)
    parser.add_argument("--embodichain-python", type=Path)
    parser.add_argument("--isaaclab-python", type=Path)
    parser.add_argument(
        "--isaaclab-repo",
        type=Path,
        help="Optional source checkout for version provenance",
    )


def runtime_python(args: argparse.Namespace, platform: str, legacy: bool) -> str:
    """Resolve an interpreter without changing the historical launcher defaults."""
    configured = getattr(args, f"{platform}_python")
    if configured is not None:
        return str(configured.absolute())
    if legacy:
        return f"/home/oem/miniconda3/envs/{platform}/bin/python"
    if platform == "embodichain":
        return sys.executable
    raise ValueError("Provide --isaaclab-python for the Isaac Lab runtime")


def _grasp_inputs(asset: Path, cache: Path, commit: str) -> Path:
    source = Path(__file__).resolve().parents[1] / "physics/experiments/p03_prepare.py"
    materializer = hashlib.sha256(source.read_bytes()).hexdigest()
    parent = cache / "derived" / commit / materializer
    parent.mkdir(parents=True, exist_ok=True)
    prepared = parent / "p-03"
    with FileLock(str(prepared) + ".lock"):
        inventory = prepared / "generated.json"
        if inventory.is_file():
            records = json.loads(inventory.read_text())
            if all(
                _file_matches(prepared / name, *records[name]) for name in _GRASP_FILES
            ):
                return prepared
        with TemporaryDirectory(dir=parent) as temp:
            output = Path(temp) / "p-03"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.benchmark.tech_reports.physics.experiments.p03_prepare",
                    "--asset",
                    str(asset),
                    "--output",
                    str(output),
                ],
                cwd=source.parents[5],
                check=True,
            )
            records = {}
            for name in _GRASP_FILES:
                path = output / name
                with path.open("rb") as stream:
                    records[name] = [
                        path.stat().st_size,
                        hashlib.file_digest(stream, "sha256").hexdigest(),
                    ]
            (output / "generated.json").write_text(json.dumps(records, indent=2) + "\n")
            if prepared.exists():
                shutil.rmtree(prepared)
            output.rename(prepared)
    return prepared


def prepare_inputs(
    benchmark: str,
    *,
    revision: str = DATASET_REVISION,
    cache_dir: Path | None = None,
    local_files_only: bool = False,
    robots: tuple[str, ...] = ("cartpole", "franka", "go2"),
) -> ReportInputs:
    """Resolve report inputs, including upstream models and derived grasp targets.

    Inputs are fetched and checked once before physics startup. Offline mode
    verifies cached bytes and can generate data from cached source files.
    """
    cache = (cache_dir or _DEFAULT_CACHE).resolve()
    bundle, provenance = download_report_assets(
        f"{REPORT_ID}-{benchmark}",
        revision=revision,
        cache_dir=cache,
        local_files_only=local_files_only,
    )
    if benchmark not in {"l-03", "p-03"}:
        return ReportInputs(bundle, provenance)
    sources = bundle / "sources.json"
    if benchmark == "p-03":
        loading, _ = download_report_assets(
            f"{REPORT_ID}-l-03",
            revision=provenance["revision"],
            cache_dir=cache,
            local_files_only=local_files_only,
        )
        sources = loading / "sources.json"
        robots = ("franka",)
    assets = download_nvidia_assets(
        sources, cache, robots=robots, local_files_only=local_files_only
    )
    derived = (
        _grasp_inputs(assets["franka"], cache, provenance["revision"])
        if benchmark == "p-03"
        else None
    )
    return ReportInputs(bundle, provenance, assets, derived)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=_BENCHMARKS)
    add_asset_arguments(parser)
    args = parser.parse_args()
    result = prepare_inputs(
        args.benchmark,
        revision=args.revision,
        cache_dir=args.cache_dir,
        local_files_only=args.offline,
    )
    print(
        json.dumps(
            {
                "input_directory": result.bundle,
                "robots": result.robots,
                "grasp_inputs": result.grasp_inputs,
                **result.provenance,
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
