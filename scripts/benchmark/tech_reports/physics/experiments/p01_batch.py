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

"""Run a specified P-01 matrix serially and retain failures and console logs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(
    os.environ.get("SIM_REPORT_ROOT", Path(__file__).resolve().parents[5])
).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    env = os.environ.copy()
    env.update(
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
        PYTHONPATH="/home/oem/code/EmbodiChain:/home/oem/code/EmbodiChain/extensions:/home/oem/code/EmbodiChain/extensions/embodichain_tasks",
        EMBODICHAIN_SIM_EXIT_PROCESS="0",
    )
    ledger = args.plan.with_suffix(".execution.json")
    assert not ledger.exists(), ledger
    records = []
    for entry in plan["runs"]:
        stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
        out = ROOT / "results/P-01" / f"{stamp}_embodichain_dexuni_{entry['id']}_N1"
        out.mkdir()
        cmd = [
            sys.executable,
            str(
                Path(__file__).resolve().parents[1]
                / "platforms/embodichain/p01_dynamics.py"
            ),
            "--case",
            entry["case"],
            "--dt",
            str(entry["dt"]),
            "--phase",
            entry["phase"],
            "--count",
            str(entry.get("count", 10)),
            "--output",
            str(out),
        ]
        if entry.get("record"):
            cmd.append("--record")
        record = dict(run_id=out.name, entry=entry, command=cmd, status="running")
        records.append(record)
        ledger.write_text(json.dumps(records, indent=2) + "\n")
        with (
            (out / "stdout.log").open("w") as stdout,
            (out / "stderr.log").open("w") as stderr,
        ):
            try:
                result = subprocess.run(
                    cmd, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, timeout=1200
                )
                record.update(
                    returncode=result.returncode,
                    status="completed" if result.returncode == 0 else "failed",
                )
            except subprocess.TimeoutExpired:
                record.update(returncode=124, status="timeout")
        ledger.write_text(json.dumps(records, indent=2) + "\n")
        print(json.dumps(record), flush=True)
        if record["returncode"]:
            print((out / "stderr.log").read_text()[-4000:], flush=True)
            raise SystemExit(record["returncode"])


if __name__ == "__main__":
    main()
