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

"""Capture observed stack boundaries through each platform's native camera.

Run after the sweep: python -m scripts.benchmark.tech_reports.physics.experiments.p02_stack_capture
"""

from __future__ import annotations

import os

import json
from pathlib import Path
import subprocess
import sys

from scripts.benchmark.tech_reports.physics.experiments.p02_stack_sweep import PROFILES

__all__ = ["main"]


def main() -> None:
    """Keep 10 s original frames and shared cameras at each platform's boundary."""
    project = Path(
        os.environ.get("SIM_REPORT_ROOT", Path(__file__).resolve().parents[5])
    ).resolve()
    root = project / "outputs/benchmarks/P-02/stack-limit-10s"
    assets = project / "scripts/benchmark/tech_reports/assets/physics/P-02"
    boundaries = json.loads((root / "boundary-results.json").read_text())
    metrics = [
        json.loads(p.read_text()) for p in (root / "analysis/by-run").glob("*.json")
    ]
    selected = ["dexuni100group", "physx100"]
    config = assets / "stack-limit-native.json"
    cfg = json.loads((assets / "stack-limit-10s-sleep.json").read_text())
    cfg["capture"]["stack_view_layers"] = max(
        boundaries[k]["highest_verified_layers"] + 1 for k in selected
    )
    cfg["revision"] = "stack-limit-10s-sleep-native-v1"
    config.write_text(json.dumps(cfg, indent=2) + "\n")
    plan_path = root / "capture-plan.json"
    if plan_path.exists():
        jobs = json.loads(plan_path.read_text())
    else:
        jobs = []
        for key in selected:
            n = boundaries[key]["highest_verified_layers"]
            jobs.append(
                dict(
                    profile=key,
                    layers=n,
                    initial=0,
                    role="stable_boundary",
                    reason="Initial 0 selected before recording; highest layer count observed intact for ten initials.",
                )
            )
            if n < boundaries[key]["search_ceiling"]:
                failed = sorted(
                    (
                        r
                        for r in metrics
                        if r["profile"] == key
                        and r["layers"] == n + 1
                        and not r["intact_full_window"]
                    ),
                    key=lambda r: r["initial"],
                )
                jobs.append(
                    dict(
                        profile=key,
                        layers=n + 1,
                        initial=failed[0]["initial"],
                        role="next_layer",
                        reason="Lowest-index initial with a measured structural event at the next layer. Deliberate failure example, not an average trial.",
                    )
                )
        plan_path.write_text(json.dumps(jobs, indent=2) + "\n")
    for job in jobs:
        if "run_id" in job:
            continue
        platform, _, options = PROFILES[job["profile"]]
        tag = f"_{job['profile']}_{job['role']}_native"
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "scripts.benchmark.tech_reports.physics.experiments.p02_batch",
            "--phase",
            "capture",
            "--config",
            str(config),
            "--output-root",
            str(root),
            "--scenes",
            f"stack_{job['layers']:03d}",
            "--platforms",
            platform,
            "--count",
            "1",
            "--episode-start",
            str(job["initial"]),
            "--duration",
            "10",
            "--tag",
            tag,
            *options,
        ]
        job["command"] = cmd
        plan_path.write_text(json.dumps(jobs, indent=2) + "\n")
        subprocess.run(cmd, cwd=project, check=True)
        p = sorted((root / "runs").glob(f"*{tag}_capture_{platform}_*/run.json"))[-1]
        meta = json.loads(p.read_text())
        assert meta["status"] == "completed"
        job["run_id"] = meta["run_id"]
        plan_path.write_text(json.dumps(jobs, indent=2) + "\n")
        print("CAPTURED", job["role"], job["profile"], job["layers"], flush=True)


if __name__ == "__main__":
    main()
