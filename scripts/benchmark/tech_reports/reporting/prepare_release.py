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

"""Inventory an existing report workspace for a deferred Hugging Face upload.

This offline command copies only authored input files and records output hashes.
It does not upload, modify measurements, or implement the planned asset CLI.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

REPORT_ID = "simulation-report-2026-10-28"
SOURCE = Path(__file__).resolve().parents[1]
RELEASE = Path(__file__).parent / "releases" / REPORT_ID


def sha256(path: Path) -> str:
    """Hash a file without loading large raw arrays or videos into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(workspace: Path, output: Path) -> dict:
    """Freeze an inventory and copy input candidates into a new staging folder.

    Args:
        workspace: Existing simulation-report workspace containing raw artifacts.
        output: New staging directory outside the source workspace.

    Returns:
        Counts and bytes inventoried, with publication explicitly pending.
    """
    workspace, output = workspace.resolve(), output.resolve()
    if output.is_relative_to(workspace):
        raise ValueError("Use a staging directory outside the evidence workspace")
    output.mkdir(parents=True, exist_ok=False)
    report = output / "reports" / REPORT_ID
    report.mkdir(parents=True)
    bundles, inputs = {}, []
    for family in ("physics", "loading"):
        for experiment in sorted((SOURCE / "assets" / family).iterdir()):
            if not experiment.is_dir():
                continue
            benchmark = experiment.name.lower()
            files = []
            for source in sorted(experiment.rglob("*")):
                if not source.is_file() or source.name == "README.md":
                    continue
                relative = source.relative_to(experiment)
                destination = report / "inputs" / benchmark / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                record = {
                    "path": destination.relative_to(output).as_posix(),
                    "size_bytes": destination.stat().st_size,
                    "sha256": sha256(destination),
                }
                files.append(record)
                inputs.append(
                    {
                        **record,
                        "source": source.relative_to(SOURCE).as_posix(),
                        "provenance": "project-authored report input",
                        "license": "NOASSERTION",
                        "license_note": "Data license must be recorded before upload; code license does not automatically apply.",
                    }
                )
            if files:
                bundles[f"{REPORT_ID}-{benchmark}"] = {
                    "report_id": REPORT_ID,
                    "benchmark_id": benchmark,
                    "files": files,
                }

    for relative, name, provenance in [
        (
            "outputs/benchmarks/P-03/prepared/grasp-v1/commands.npz",
            "commands.npz",
            "frozen project-generated shared grasp commands",
        ),
        (
            "outputs/benchmarks/P-03/prepared/collision-v1/fingers.npz",
            "fingers.npz",
            "evaluation geometry derived from the NVIDIA Franka source asset; upstream license review required",
        ),
    ]:
        source = workspace / relative
        if not source.is_file():
            continue
        destination = report / "inputs/p-03/frozen" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        record = {
            "path": destination.relative_to(output).as_posix(),
            "size_bytes": source.stat().st_size,
            "sha256": sha256(source),
        }
        bundles[f"{REPORT_ID}-p-03"]["files"].append(record)
        inputs.append(
            {
                **record,
                "source": relative,
                "provenance": provenance,
                "license": "NOASSERTION",
                "license_note": "Review before upload; no license is granted by this inventory.",
            }
        )

    roots = [workspace / "results/P-01", workspace / "reports"]
    roots.extend(sorted((workspace / "outputs/benchmarks").iterdir()))
    roots.extend([workspace / "configs", workspace / "code", workspace / "specs"])
    artifacts, excluded = [], []
    for root in roots:
        for source in sorted(root.rglob("*")):
            if not source.is_file():
                continue
            relative = source.relative_to(workspace)
            parts = relative.parts
            reason = None
            if source.is_symlink():
                reason = (
                    "symlink; resolve and review original asset provenance separately"
                )
            elif any(p in parts for p in ("runtime", "__pycache__", ".cache")):
                reason = "installed runtime/cache; reproduce from separately licensed dependencies"
            elif (
                source.suffixes[-2].lower()
                if source.suffix == ".gz" and len(source.suffixes) > 1
                else source.suffix.lower()
            ) in {
                ".usd",
                ".usda",
                ".usdc",
                ".urdf",
                ".obj",
                ".dae",
                ".stl",
                ".mtl",
                ".so",
                ".pyc",
                ".patch",
                ".diff",
            } or source.name == "parser.py":
                reason = "generic asset or engine-source evidence; retain source reference and review redistribution separately"
            elif "prepared" in parts and source.suffix.lower() not in {
                ".json",
                ".md",
                ".log",
                ".npz",
                ".csv",
                ".txt",
            }:
                reason = "prepared generic asset/media; use original repository"
            if reason:
                excluded.append(
                    {
                        "source": relative.as_posix(),
                        "size_bytes": source.stat().st_size,
                        "reason": reason,
                    }
                )
                continue
            benchmark = next(
                (
                    p.lower()
                    for p in parts
                    if p in {"P-01", "P-02", "P-03", "L-02", "L-03", "L-04"}
                ),
                "shared",
            )
            if relative.as_posix().startswith("reports/P-01"):
                benchmark = "p-01"
            if "usd-import-fix" in parts:
                benchmark = "usd-import-fix"
            category = (
                "videos"
                if source.suffix.lower() in {".mp4", ".webm", ".gif"}
                else (
                    "figures"
                    if source.suffix.lower()
                    in {".png", ".jpg", ".jpeg", ".svg", ".pdf"}
                    else "results"
                )
            )
            artifacts.append(
                {
                    "source": relative.as_posix(),
                    "path": f"reports/{REPORT_ID}/{category}/{benchmark}/{relative.as_posix()}",
                    "size_bytes": source.stat().st_size,
                    "sha256": sha256(source),
                    "provenance": "existing workspace artifact; exact commands and upstream inputs remain in run manifests",
                    "license": "NOASSERTION",
                    "license_note": "Pending per-file rights review, including source-asset notices for derived geometry/media.",
                }
            )
    summary = {
        "report_id": REPORT_ID,
        "dataset": "DexForceAI/embodichain_reports",
        "publication_status": "not_uploaded",
        "dataset_revision": None,
        "input_files": len(inputs),
        "artifact_files": len(artifacts),
        "artifact_bytes": sum(x["size_bytes"] for x in artifacts),
        "excluded_files": len(excluded),
        "artifact_groups": dict(Counter(x["path"].split("/")[2] for x in artifacts)),
        "limitations": [
            "Only input candidates are copied. Artifacts remain at their original paths; verify hashes again before uploading.",
            "NOASSERTION is a review marker, not a redistribution license; resolve per-file rights before upload.",
            "This snapshot does not include installed runtimes or generic robot/scene assets.",
            "Merge index.fragment.json into the current dataset index; never replace it with this fragment or the empty bootstrap index.",
            "Publish reviewed inputs, report metadata and merged index in one dataset commit, then verify and pin its full SHA.",
        ],
    }
    for name, value in {
        "inventory.json": {
            "inputs": inputs,
            "artifacts": artifacts,
            "excluded": excluded,
        },
        "release.json": summary,
    }.items():
        (report / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n"
        )
    (output / "index.fragment.json").write_text(
        json.dumps({"schema_version": 1, "bundles": bundles}, indent=2) + "\n"
    )
    shutil.copy2(RELEASE / "README.md", report / "README.md")
    shutil.copy2(RELEASE / "source-map.json", report / "source-map.json")
    return summary


def main() -> None:
    """Prepare a local inventory without contacting Hugging Face."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(
        json.dumps(prepare(args.workspace, args.output), ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    main()
