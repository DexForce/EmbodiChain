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
"""Run a fixed three-repeat coupling validation matrix, serializing GPU processes.

Run: .venv/bin/python scripts/benchmark/tech_reports/physics/experiments/dexuni/verify_coupling.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[6]
TECH = REPO / "scripts/benchmark/tech_reports"
ROOTS = {
    "s03": TECH.parents[2]
    / "outputs/benchmarks/tech_reports/dexuni/s03_cloth_corner_lift",
    "s04": TECH.parents[2]
    / "outputs/benchmarks/tech_reports/dexuni/s04_rigid_soft_squeeze_grasp",
}
__all__ = ["main"]


def main() -> None:
    """Verify preselected solutions against fresh native baselines without profiler instrumentation."""
    protocol = ROOTS["s04"] / "analysis/validation_protocol.json"
    attempts_path = ROOTS["s04"] / "analysis/validation_attempts.json"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if protocol.exists() and not args.resume:
        raise ValueError("Do not overwrite an existing validation matrix")
    script = TECH / "physics/experiments/dexuni/diagnose_coupling.py"
    py = {
        "embodichain": REPO / ".venv/bin/python",
        "isaaclab": Path("/home/oem/app/IsaacLab/.venv/bin/python"),
    }
    candidates = {"s03": "rne_off_matched_surface", "s04": "rne_off_uniform"}
    identity = {
        str(p.relative_to(TECH)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (script, TECH / "physics/experiments/dexuni/coupling_tuning.py")
    }
    plan = []
    for repeat in range(11, 14):
        for task in ROOTS:
            order = [
                ("embodichain", "baseline"),
                ("isaaclab", "baseline"),
                ("embodichain", candidates[task]),
            ]
            if repeat % 2 == 0:
                order.reverse()
            plan.extend((task, p, c, repeat) for p, c in order)
    if args.resume:
        prior = json.loads(protocol.read_text())
        attempts = json.loads(attempts_path.read_text())
        completed_prefix = len(attempts)
        prior.setdefault("source_sha256_revisions", []).append(identity)
        prior["resume_reason"] = (
            "generated-kernel license/header syntax fix; first five complete samples unchanged and failed initialization retained"
        )
        failures = sum(bool(a["returncode"]) for a in attempts)
        for extra in range(failures):
            plan.append(("s04", "embodichain", candidates["s04"], 14 + extra))
        prior["plan"] = plan
        protocol.write_text(json.dumps(prior, indent=2) + "\n")
        plan = plan[completed_prefix:]
    else:
        protocol.write_text(
            json.dumps(
                {
                    "scope": "diagnostic validation; existing five-repeat official records untouched",
                    "repeats": 3,
                    "first_repeat": 11,
                    "candidates": candidates,
                    "source_sha256": identity,
                    "events": False,
                    "profiler": False,
                    "warmup": 30,
                    "plan": plan,
                },
                indent=2,
            )
            + "\n"
        )
        attempts = []
    for task, p, c, repeat in plan:
        out = ROOTS[task] / "analysis/coupling" / f"{p}_{c}_r{repeat:02d}"
        log = ROOTS[task] / "analysis" / f"{p}_{c}_r{repeat:02d}.log"
        if out.exists() or log.exists():
            raise ValueError(f"Existing attempt: {out}")
        command = [
            str(py[p]),
            str(script),
            "--task",
            task,
            "--platform",
            p,
            "--case",
            c,
            "--repeat",
            str(repeat),
        ]
        print(f"[{len(attempts)+1}/{len(plan)}] {task} {p} {c} r{repeat}", flush=True)
        started = time.perf_counter()
        with log.open("w") as stream:
            result = subprocess.run(
                command,
                cwd=REPO,
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=dict(os.environ, PYTHONUNBUFFERED="1"),
                check=False,
            )
        attempts.append(
            {
                "task": task,
                "platform": p,
                "case": c,
                "repeat": repeat,
                "returncode": result.returncode,
                "wall_s": time.perf_counter() - started,
                "run": str(out.relative_to(TECH)),
                "log": str(log.relative_to(TECH)),
            }
        )
        attempts_path.write_text(json.dumps(attempts, indent=2) + "\n")
        print(
            f"Completed rc={result.returncode}, wall={attempts[-1]['wall_s']:.2f}s",
            flush=True,
        )
        if result.returncode:
            raise RuntimeError(f"Retained failed attempt: {log}")
    print(f"Validation attempts: {attempts_path}", flush=True)


if __name__ == "__main__":
    main()
