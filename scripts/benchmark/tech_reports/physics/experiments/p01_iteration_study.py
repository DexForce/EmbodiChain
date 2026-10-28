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

"""Execute the recorded P-01 investigation plan in separate, sequential processes."""

from __future__ import annotations

import os

import json
import subprocess
from pathlib import Path

ROOT = Path(
    os.environ.get("SIM_REPORT_ROOT", Path(__file__).resolve().parents[5])
).resolve()


def main() -> None:
    plan = json.loads((ROOT / "configs/P-01/iteration-study.json").read_text())
    for item in plan["runs"]:
        output = ROOT / "results/P-01" / f"20261014_study_{item['id']}"
        if output.exists():
            meta = json.loads((output / "run.json").read_text())
            assert (
                meta["status"] == "completed"
            ), f"Inspect failed run before resuming: {output}"
            print(f"Already complete: {output.name}", flush=True)
            continue
        output.mkdir()
        command = [
            "/home/oem/miniconda3/bin/conda",
            "run",
            "--no-capture-output",
            "-n",
            "isaaclab",
            "python",
            str(
                Path(__file__).resolve().parents[1]
                / "platforms/isaaclab/p01_dynamics.py"
            ),
            "--case",
            "pendulum",
            "--phase",
            "investigation",
            "--probe-body-velocity",
            "--visualizer",
            "none",
            "--output",
            str(output),
        ]
        for key, default in {
            "dt": 0.005,
            "position_iterations": 4,
            "velocity_iterations": 1,
            "count": 1,
            "duration": 10,
            "solver": "TGS",
            "device": "cuda:0",
            "body_frame": "pivot",
        }.items():
            command += ["--" + key.replace("_", "-"), str(item.get(key, default))]
        if "theta_deg" in item:
            command += ["--theta-deg", str(item["theta_deg"])]
        if not item.get("external_forces_every_iteration", True):
            command.append("--no-external-forces-every-iteration")
        (output / "launch-command.json").write_text(
            json.dumps(command, indent=2) + "\n"
        )
        print(f"Starting {item['group']}: {output.name}", flush=True)
        with (
            (output / "stdout.log").open("w") as stdout,
            (output / "stderr.log").open("w") as stderr,
        ):
            result = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr)
        meta = json.loads((output / "run.json").read_text())
        print(f"{meta['status']}: {output.name}", flush=True)
        if result.returncode or meta["status"] != "completed":
            raise RuntimeError(f"Run failed; inspect preserved logs: {output}")


if __name__ == "__main__":
    main()
