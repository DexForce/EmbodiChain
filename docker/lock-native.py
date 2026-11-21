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

"""Resolve native conda packages into explicit, checksum-qualified URL locks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

__all__ = []

_SPECS = {
    "py311": ["python=3.11.16"],
    "py312": ["python=3.12.14"],
    "media": ["ffmpeg=7.1.*=gpl_*", "hdf5"],
}


def _write_lock(name: str, plan: dict, cache_dirs: list[str]) -> None:
    if not plan.get("success"):
        raise RuntimeError(plan.get("message", str(plan)))
    records = {
        f"{record['name']}-{record['version']}-{record['build']}": record
        for record in plan["actions"].get("FETCH", [])
    }
    urls = []
    for record in plan["actions"]["LINK"]:
        key = record["dist_name"]
        full = records.get(key)
        if full is None:
            for directory in cache_dirs:
                metadata = Path(directory) / key / "info/repodata_record.json"
                if metadata.is_file():
                    full = json.loads(metadata.read_text(encoding="utf-8"))
                    break
        if full is None:
            raise RuntimeError(f"Missing package checksum: {key}")
        urls.append(full["url"] + "#" + full["md5"])
    target = Path(__file__).parent / "locks" / f"conda-{name}.txt"
    target.write_text(
        "# platform: linux-64\n# Regenerate with docker/lock-native.py\n@EXPLICIT\n"
        + "\n".join(urls)
        + "\n",
        encoding="utf-8",
    )
    print(f"{target}: {len(urls)} packages")


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=tuple(_SPECS))
    args = parser.parse_args()
    info = json.loads(subprocess.check_output(["conda", "info", "--json"], text=True))
    with tempfile.TemporaryDirectory(prefix="embodichain-conda-lock-") as directory:
        result = subprocess.run(
            [
                "conda",
                "create",
                "--dry-run",
                "--json",
                "--override-channels",
                "-c",
                "conda-forge",
                "-p",
                str(Path(directory) / "prefix"),
                *_SPECS[args.profile],
            ],
            env={**os.environ, "CONDA_OVERRIDE_CUDA": "12.8"},
            capture_output=True,
            text=True,
        )
        plan = json.loads(result.stdout)
        _write_lock(args.profile, plan, info["pkgs_dirs"])


if __name__ == "__main__":
    _main()
