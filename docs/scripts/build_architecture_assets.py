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

"""Build and publish a complete documentation-only architecture bundle."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence

__all__ = ["build_assets"]
ROOT = Path(__file__).resolve().parents[2]


def build_assets(repo_root: Path, output: Path, *, html: bool = True) -> None:
    """Generate source data and replace assets only after a successful build.

    Args:
        repo_root: Checkout whose HEAD is being documented.
        output: Destination for one complete, disposable generated bundle.
        html: Build the frontend; false generates a text-only documentation bundle.
    """
    repo_root, output = repo_root.resolve(), output.resolve()
    web = repo_root / "docs/architecture/web"
    if html and not (web / "node_modules").is_dir():
        raise RuntimeError(
            f"Install frontend dependencies first: npm --prefix {web} ci (Node.js 22.12+ required)"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".architecture-build-", dir=output.parent
    ) as folder:
        stage = Path(folder)
        data = stage / "data"
        data.mkdir()
        subprocess.run(
            [
                sys.executable,
                str(repo_root / "docs/scripts/build_architecture.py"),
                "--repo-root",
                str(repo_root),
                "--curated",
                str(repo_root / "docs/architecture/curated.json"),
                "--output-dir",
                str(data),
            ],
            cwd=repo_root,
            check=True,
        )
        bundle = stage / "bundle"
        if html:
            subprocess.run(
                ["npm", "run", "build", "--", "--outDir", str(bundle)],
                cwd=web,
                check=True,
                env={
                    **os.environ,
                    "ARCHITECTURE_DATA_PATH": str(data / "architecture.json"),
                },
            )
            if not (bundle / "index.html").is_file():
                raise RuntimeError("Frontend build did not produce index.html")
            if (bundle / "architecture.json").read_bytes() != (
                data / "architecture.json"
            ).read_bytes():
                raise RuntimeError(
                    "Frontend snapshot does not match the generated source data"
                )
        else:
            bundle.mkdir()
            shutil.copy2(data / "architecture.json", bundle / "architecture.json")
        shutil.copy2(data / "summary.md", bundle / "summary.md")
        previous = stage / "previous"
        if output.exists():
            output.rename(previous)
        try:
            bundle.rename(output)
        except OSError:
            if previous.exists():
                previous.rename(output)
            raise


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "docs/source/_static/architecture"
    )
    parser.add_argument("--text-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        build_assets(ROOT, args.output_dir, html=not args.text_only)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Architecture asset build failed: {error}\n")
    print(f"Architecture assets ready: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
