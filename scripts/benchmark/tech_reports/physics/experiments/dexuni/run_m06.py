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
"""Run five isolated paired M06 processes under the selected solver policies.

Run: .venv/bin/python scripts/benchmark/tech_reports/physics/experiments/dexuni/run_m06.py
Resume with --resume; offline report rebuilding uses --rebuild-report.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO))

from scripts.benchmark.tech_reports.physics.experiments.dexuni.s04_measurement import (
    file_hashes,
    write_json,
)
from scripts.benchmark.tech_reports.physics.platforms.dexuni.dexsim.m06 import (
    DEFAULT_OUTPUT as TASK,
    digest,
)

TECH = REPO / "scripts/benchmark/tech_reports"
MEASUREMENT = TASK / "measurement"
PYTHONS = {
    "embodichain": REPO / ".venv/bin/python",
    "isaaclab": Path("/home/oem/app/IsaacLab/.venv/bin/python"),
}
__all__ = ["main"]


def prepare() -> None:
    """Freeze input, preview and execution-source identities before the first attempt."""
    if MEASUREMENT.exists():
        raise ValueError("Existing formal M06 batch requires --resume")
    ec = json.loads((TASK / "preview/embodichain/manifest.json").read_text())
    il = json.loads((TASK / "preview/isaaclab/manifest.json").read_text())
    if ec["status"] != "completed" or il["status"] != "completed":
        raise ValueError("Both current scene previews must be complete")
    if (
        ec["effective"]["vbd_preset"] != "surface-fast"
        or ec["effective"]["vbd"]["iterations"] != 8
    ):
        raise ValueError("EC preview does not use the selected eight-sweep preset")
    if il["vbd_iterations"] != 15:
        raise ValueError("Isaac preview does not use its selected fifteen-sweep policy")
    source_names = [
        "physics/experiments/dexuni/run_m06.py",
        "physics/experiments/dexuni/m06_measurement.py",
        "physics/experiments/dexuni/s04_measurement.py",
        "physics/experiments/dexuni/sdk_identity.py",
        "physics/platforms/dexuni/_lib/benchlib/gpu_memory.py",
        "reporting/dexuni/build_m06_report.py",
        *[f"physics/platforms/dexuni/{p}/m06.py" for p in (*PYTHONS, "dexsim")],
    ]
    sources = [TECH / name for name in source_names]
    sources += [
        REPO / name
        for name in (
            "embodichain/lab/sim/sim_manager.py",
            "embodichain/lab/sim/cfg/deformable.py",
            "embodichain/lab/sim/spawn/descriptors.py",
        )
    ]
    sdk = REPO / ".venv/lib/python3.11/site-packages/dexsim/engine/newton_physics"
    sources += [
        sdk / name
        for name in ("solvers_cfg.py", "newton_backend.py", "_dexuni_options.py")
    ]
    sources += sorted((sdk / "_jit/modules/mjvbd_v2").rglob("*.py"))
    (MEASUREMENT / "logs").mkdir(parents=True)
    for path in sources:
        target = MEASUREMENT / "source" / path.relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    preview_names = [
        "preview/comparison.mp4",
        "preview/embodichain/embodichain.mp4",
        "preview/isaaclab/isaaclab.mp4",
    ]
    plan = [
        (platform, repeat)
        for repeat in range(1, 6)
        for platform in (list(PYTHONS) if repeat % 2 else list(PYTHONS)[::-1])
    ]
    protocol = {
        "task": "M06",
        "authorized": True,
        "authorization": "User: 好，按这个测试一版正式数据",
        "authorization_scope": "selected DexUni surface-fast policy, rather than equal iteration budgets",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repeats_per_platform": 5,
        "outer_steps": 906,
        "warmup_steps": 30,
        "measured_steps_per_episode": 876,
        "dt": 1 / 60,
        "num_substeps": 10,
        "policies": {
            "embodichain": {
                "vbd_preset": "surface-fast",
                "base_iterations": 8,
                "adaptive_fallback_iterations": 20,
                "displacement_deadband_m_per_substep": 5e-6,
            },
            "isaaclab": {"vbd_preset": None, "base_iterations": 15},
        },
        "comparison": "same frozen garment/materials/kinematic joint replay; different solver algorithms and iteration budgets; task success reported separately from performance eligibility",
        "sampling": "five fresh no-camera processes per platform, serialized GPU, alternating order; all attempts retained, no geometry-based exclusion or replacement",
        "preview_note": "EC video regenerated using selected fast policy; Isaac video retains unchanged physical policy; user authorized this selected configuration, videos are not separately marked human-approved",
        "input_sha256": file_hashes(TASK / "inputs"),
        "source_sha256": {str(path): digest(path) for path in sources},
        "preview_sha256": {name: digest(TASK / name) for name in preview_names},
        "plan": [{"platform": platform, "repeat": repeat} for platform, repeat in plan],
        "task_checks": {
            "max_tcp_error_m": 0.0005,
            "minimum_sleeve_lift_m": 0.05,
            "minimum_rightward_motion_m": 0.15,
        },
        "memory": "per-process NVML snapshots, sampled maximum rather than transient peak; one separate untimed warmed memory call after each episode; Torch allocator peak has limited coverage",
        "repo_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
    }
    write_json(MEASUREMENT / "protocol.json", protocol)
    write_json(MEASUREMENT / "attempts.json", [])


def gpu_status() -> str:
    """Record unlocked GPU clocks, temperature and utilization outside timing."""
    return subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,pstate,temperature.gpu,clocks.sm,utilization.gpu,power.draw",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


def main() -> None:
    """Execute the immutable attempt plan and rebuild the report after completion."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rebuild-report", action="store_true")
    args = parser.parse_args()
    if not args.resume and not args.rebuild_report:
        prepare()
    if not args.rebuild_report:
        protocol = json.loads((MEASUREMENT / "protocol.json").read_text())
        attempts = json.loads((MEASUREMENT / "attempts.json").read_text())
        prefix = [{"platform": a["platform"], "repeat": a["repeat"]} for a in attempts]
        if prefix != protocol["plan"][: len(attempts)] or any(
            not a.get("performance_valid") for a in attempts
        ):
            raise ValueError(
                "Inspect incomplete/failed attempts; no automatic replacement is allowed"
            )
        for item in protocol["plan"][len(attempts) :]:
            platform, repeat = item["platform"], item["repeat"]
            command = [
                str(PYTHONS[platform]),
                str(TECH / f"physics/platforms/dexuni/{platform}/m06.py"),
                "--measure",
                "--repeat",
                str(repeat),
            ]
            log = MEASUREMENT / f"logs/{platform}_r{repeat:02d}.log"
            if log.exists():
                raise ValueError("Unrecorded attempt log already exists")
            attempt = dict(
                item,
                command=command,
                log=str(log.relative_to(TASK)),
                started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                gpu_before=gpu_status(),
            )
            attempts.append(attempt)
            write_json(MEASUREMENT / "attempts.json", attempts)
            print(f"[{len(attempts)}/10] {platform} r{repeat:02d}", flush=True)
            start = time.perf_counter()
            try:
                with log.open("w") as stream:
                    result = subprocess.run(
                        command,
                        cwd=REPO,
                        env=dict(os.environ, PYTHONUNBUFFERED="1"),
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        timeout=1200,
                    )
                attempt["returncode"] = result.returncode
                run = MEASUREMENT / platform / f"r{repeat:02d}"
                manifest = json.loads((run / "manifest.json").read_text())
                attempt.update(
                    run=str(run.relative_to(TASK)),
                    performance_valid=result.returncode == 0
                    and manifest["status"] == "completed"
                    and all(manifest["performance_checks"].values()),
                    task_success=manifest.get("task_success", False),
                    mean_ms=manifest.get("summary", {}).get("cost_time_ms"),
                )
            except BaseException as error:
                attempt.update(performance_valid=False, error=repr(error))
            attempt.update(
                process_wall_s=time.perf_counter() - start,
                gpu_after=gpu_status(),
                ended_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            )
            write_json(MEASUREMENT / "attempts.json", attempts)
            if not attempt["performance_valid"]:
                raise RuntimeError(f"Failed attempt retained: {log}")
            print(
                f"  {attempt['mean_ms']:.3f} ms; task_success={attempt['task_success']}",
                flush=True,
            )
    from scripts.benchmark.tech_reports.reporting.dexuni.build_m06_report import (
        build_report,
    )

    build_report()


if __name__ == "__main__":
    main()
