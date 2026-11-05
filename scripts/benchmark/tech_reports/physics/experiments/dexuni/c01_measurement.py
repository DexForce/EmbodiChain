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
"""Guard and run the authorized EC-only five-process C01 performance batch.

Run: python -m scripts.benchmark.tech_reports.physics.experiments.dexuni.c01_measurement
The protocol is frozen before execution; all prior previews and pilots stay intact.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .c01_common import OUTPUT
from .c01_pilot import _gpu_processes
from .c01_replay import sha256, write_json

__all__ = ["measurement_directory", "main"]


def measurement_directory(repeat: int) -> Path:
    """Verify the frozen EC-only scope, immutable inputs and idle compute GPU."""
    root = OUTPUT / "measurement"
    protocol = json.loads((root / "protocol.json").read_text())
    if not (
        protocol["formal_measurement"]
        and protocol["platforms"] == ["embodichain"]
        and protocol["episodes_per_platform"] == 5
        and repeat in range(1, 6)
    ):
        raise ValueError("Expected the authorized five-process EC-only protocol")
    for name, expected in protocol["protected_sha256"].items():
        if sha256(OUTPUT / name) != expected:
            raise ValueError(f"Protected C01 file changed: {name}")
    for name, expected in protocol["source_sha256"].items():
        if sha256(OUTPUT.parents[1] / name) != expected:
            raise ValueError(f"Frozen execution source changed: {name}")
    foreign = [p for p in _gpu_processes() if p["pid"] != os.getpid()]
    if foreign:
        raise RuntimeError(f"Wait for foreign GPU compute: {foreign}")
    output = root / "embodichain" / f"r{repeat:02d}"
    if output.exists():
        raise FileExistsError(f"Retain attempts; do not overwrite {output}")
    return output


def main() -> None:
    """Launch exactly five sequential fresh processes and retain all attempt logs."""
    root = OUTPUT / "measurement"
    ledger = root / "attempts.json"
    if ledger.exists():
        raise FileExistsError("Batch already started; inspect its retained attempts")
    attempts = []
    tech = OUTPUT.parents[1]
    repo = tech.parents[2]
    env = os.environ.copy()
    env.update(
        PYTHONUNBUFFERED="1",
        PYTHONPATH="/home/oem/code/repos/dexsimforembodychain/dexsim/modules/dexsim/build_Release/lib/python_package:/home/oem/miniconda3/envs/dexsim1/lib/python3.11/site-packages",
    )
    for repeat in range(1, 6):
        measurement_directory(repeat)
        log = root / "embodichain" / f"r{repeat:02d}.log"
        log.parent.mkdir(exist_ok=True)
        row = dict(
            platform="embodichain",
            repeat=repeat,
            status="running",
            log=str(log.relative_to(root)),
        )
        attempts.append(row)
        write_json(ledger, attempts)
        command = [
            str(repo / ".venv/bin/python"),
            str(tech / "physics/platforms/dexuni/embodichain/c01.py"),
            "--measure",
            str(repeat),
        ]
        row["command"] = command
        write_json(ledger, attempts)
        print(f"C01 formal EC r{repeat:02d} starting", flush=True)
        with log.open("w") as stream:
            result = subprocess.run(
                command,
                env=env,
                cwd=repo,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        row.update(
            exit_code=result.returncode,
            status="failed" if result.returncode else "completed",
        )
        write_json(ledger, attempts)
        if result.returncode:
            sys.exit(result.returncode)
        summary = json.loads(
            (root / "embodichain" / f"r{repeat:02d}/summary.json").read_text()
        )
        if not summary["performance_eligible"]:
            raise RuntimeError(
                "Foreign GPU compute observed; retain this sample and inspect batch"
            )
        print(
            f"C01 formal EC r{repeat:02d} completed mean={summary['cost_time_ms']:.6f}ms",
            flush=True,
        )
    shutil.copy2(Path(__file__), root / "batch_execution.py")


if __name__ == "__main__":
    main()
