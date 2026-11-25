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

"""Stage report-owned benchmark inputs and the attributed ForgeCAD cabinet."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

from huggingface_hub import snapshot_download

from ..assets.download import REPORT_ID

__all__ = ["prepare"]


def prepare(output: Path, cache_dir: Path | None = None) -> Path:
    """Build a ready-to-review release tree and a mergeable input index."""
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parents[1]
    repo = source.parents[2]
    with TemporaryDirectory(dir=output.parent) as temp:
        staged = Path(temp) / "stage"
        report = staged / "reports" / REPORT_ID
        inputs = report / "inputs"
        for family in ("loading", "physics"):
            for asset in sorted((source / "assets" / family).iterdir()):
                if asset.is_dir():
                    shutil.copytree(asset, inputs / asset.name.lower())
        subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.benchmark.tech_reports.loading.experiments.l02_prepare",
                "--shared-only",
                "--output",
                str(inputs / "l-02/shared"),
            ],
            cwd=repo,
            check=True,
        )
        p01 = inputs / "p-01"
        cfg_dir = source / "physics/experiments/p01_configs"
        for name in ("embodichain-dexuni.json", "isaaclab.json"):
            shutil.copy2(cfg_dir / name, p01 / name)
        shutil.copy2(
            source / "physics/experiments/p-01-dynamics-spec.md",
            p01 / "p-01-dynamics-spec.md",
        )
        l04 = inputs / "l-04"
        cabinet = json.loads((l04 / "cabinet-candidate.json").read_text())
        upstream = Path(
            snapshot_download(
                repo_id=cabinet["repo_id"],
                repo_type=cabinet["repo_type"],
                revision=cabinet["revision"],
                allow_patterns=cabinet["allow_patterns"],
                cache_dir=cache_dir,
                token=False,
            )
        )
        name = "kitchen_base_cabinet"
        upstream_package = upstream / name
        metadata = json.loads(
            (upstream_package / "com.nvidia.simready.packaging.json").read_text()
        )
        if metadata["license"].lower() != "cc-by-4.0":
            raise ValueError("ForgeCAD cabinet has an unexpected license declaration")
        if not (upstream / cabinet["entrypoint"]).is_file():
            raise FileNotFoundError(cabinet["entrypoint"])
        shutil.copytree(upstream_package, l04 / name)
        (l04 / "FORGECAD-NOTICE.md").write_text(
            "# Kitchen base cabinet\n\n"
            "The kitchen base cabinet is published by ForgeCAD and copied unchanged "
            "from its SimReady Assets dataset. It is licensed under "
            "[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).\n\n"
            f"[Source and attribution](https://huggingface.co/datasets/{cabinet['repo_id']}"
            f"/tree/{cabinet['revision']}/{name})\n"
        )
        bundles = {}
        for case in sorted(inputs.iterdir()):
            files = []
            for path in sorted(case.rglob("*")):
                if path.is_file():
                    with path.open("rb") as stream:
                        digest = hashlib.file_digest(stream, "sha256").hexdigest()
                    files.append(
                        {
                            "path": path.relative_to(staged).as_posix(),
                            "size_bytes": path.stat().st_size,
                            "sha256": digest,
                        }
                    )
            bundles[f"{REPORT_ID}-{case.name}"] = {
                "report_id": REPORT_ID,
                "benchmark_id": case.name,
                "files": files,
            }
        (staged / "index.fragment.json").write_text(
            json.dumps({"schema_version": 1, "bundles": bundles}, indent=2) + "\n"
        )
        shutil.copy2(repo / "LICENSE", report / "LICENSE-EmbodiChain")
        (report / "README.md").write_text(
            "# EmbodiChain simulation report inputs\n\n"
            "Run inputs for L-02, L-03, the L-04 workstation candidate, "
            "and P-01 through P-03. The input bundle index records each file's "
            "size and SHA-256. Use a fixed dataset commit when running the benchmarks.\n\n"
            "L-03 and P-03 resolve the exact NVIDIA robot files directly from NVIDIA. "
            "The file paths and hashes are in the L-03 sources.json manifest. "
            "P-03's command arrays and finger geometry can be regenerated locally "
            "from the pinned Franka asset by the project preparation script.\n\n"
            "The L-04 model is the ForgeCAD candidate. See "
            "[the attribution notice](inputs/l-04/FORGECAD-NOTICE.md).\n"
        )
        staged.rename(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    print(prepare(args.output, args.cache_dir))


if __name__ == "__main__":
    main()
