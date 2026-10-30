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

"""Serial capture of the six P-01 common-view demonstrations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo

PROJECT = Path(
    os.environ.get("SIM_REPORT_ROOT", Path(__file__).resolve().parents[5])
).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    ledger = args.plan.with_suffix(".execution.json")
    assert not ledger.exists(), ledger
    results = []
    for entry in plan:
        platform, case = entry["platform"], entry["case"]
        stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
        out = PROJECT / "results/P-01" / f'{stamp}_{entry["id"]}'
        out.mkdir()
        cmd = [
            f"/home/oem/miniconda3/envs/{platform}/bin/python",
            str(
                Path(__file__).resolve().parents[1]
                / f"platforms/{platform}/p01_dynamics.py"
            ),
            "--case",
            case,
            "--dt",
            "0.005",
            "--count",
            "1",
            "--duration",
            "10",
            "--phase",
            "capture",
            "--record",
            "--common-view",
            str(PROJECT / "assets/P-01/common-view.json"),
            "--output",
            str(out),
        ]
        env = dict(os.environ)
        if platform == "embodichain":
            env["PYTHONPATH"] = (
                "/home/oem/code/EmbodiChain:/home/oem/code/EmbodiChain/extensions:/home/oem/code/EmbodiChain/extensions/embodichain_tasks"
            )
        else:
            env.pop("PYTHONPATH", None)
            env["OMNI_KIT_ACCEPT_EULA"] = "YES"
            cmd += [
                "--solver",
                entry["solver"],
                "--position-iterations",
                "4",
                "--velocity-iterations",
                "1",
                "--visualizer",
                "none",
                "--device",
                "cuda:0",
            ]
        result = dict(**entry, run_id=out.name, command=cmd, status="running")
        results.append(result)
        ledger.write_text(json.dumps(results, indent=2) + "\n")
        print("Starting", out.name, flush=True)
        with (
            (out / "stdout.log").open("w") as stdout,
            (out / "stderr.log").open("w") as stderr,
        ):
            try:
                process = subprocess.run(
                    cmd,
                    env=env,
                    cwd=PROJECT,
                    stdout=stdout,
                    stderr=stderr,
                    timeout=1200,
                )
                result["returncode"] = process.returncode
                result["status"] = "completed" if process.returncode == 0 else "failed"
            except subprocess.TimeoutExpired:
                result["status"] = "timeout"
        ledger.write_text(json.dumps(results, indent=2) + "\n")
        print(result["status"], out.name, flush=True)
        if result["status"] != "completed":
            raise RuntimeError(f"Capture failed; inspect {out}")


if __name__ == "__main__":
    main()
