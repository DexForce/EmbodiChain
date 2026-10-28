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

"""Repeat fixed stack inputs without recording to check process reproducibility.

Run: python -m scripts.benchmark.tech_reports.physics.experiments.p02_stack_repeat
"""

from __future__ import annotations

import os

import json
from pathlib import Path
import subprocess
import sys
import numpy as np

from scripts.benchmark.tech_reports.reporting.p02_stack_metrics import measure_stack

__all__ = ["main"]


def main() -> None:
    """Run two fresh processes for each preselected boundary initial."""
    project = Path(
        os.environ.get("SIM_REPORT_ROOT", Path(__file__).resolve().parents[5])
    ).resolve()
    parent = project / "outputs/benchmarks/P-02/stack-limit-10s"
    root = parent / "repeatability"
    root.mkdir(exist_ok=True)
    plan_path = root / "plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
    else:
        plan = [
            dict(layers=n, initial=i, repeat=r)
            for n, i in [(51, 0), (52, 4)]
            for r in [1, 2]
        ]
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    for job in plan:
        if "run_id" in job:
            continue
        tag = f"_repeat{job['repeat']}_stack{job['layers']}"
        command = [
            sys.executable,
            "-u",
            "-m",
            "scripts.benchmark.tech_reports.physics.experiments.p02_batch",
            "--phase",
            "diagnostic",
            "--config",
            str(
                project
                / "scripts/benchmark/tech_reports/assets/physics/P-02/stack-limit-10s-sleep.json"
            ),
            "--output-root",
            str(root),
            "--platforms",
            "embodichain",
            "--scenes",
            f"stack_{job['layers']:03d}",
            "--count",
            "1",
            "--episode-start",
            str(job["initial"]),
            "--duration",
            "10",
            "--iterations",
            "100",
            "--contact-grouping",
            "--tag",
            tag,
        ]
        job["command"] = command
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")
        subprocess.run(command, cwd=project, check=True)
        path = sorted((root / "runs").glob(f"*{tag}_*/run.json"))[-1]
        meta = json.loads(path.read_text())
        assert meta["status"] == "completed"
        metric, curves = measure_stack(
            path.parent / f"episode_{job['initial']:02d}.npz", meta
        )
        dest = root / "analysis"
        dest.mkdir(exist_ok=True)
        (dest / f"{path.parent.name}.json").write_text(
            json.dumps(metric, indent=2) + "\n"
        )
        np.savez_compressed(dest / f"{path.parent.name}.npz", **curves)
        job["run_id"] = path.parent.name
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")
        print(
            "REPEATED",
            job["layers"],
            job["initial"],
            job["repeat"],
            metric["first_geometry_event_s"],
            flush=True,
        )


if __name__ == "__main__":
    main()
