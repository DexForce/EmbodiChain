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
"""Remeasure permanent SDK integration with five isolated paired runs per task.

Run: .venv/bin/python scripts/benchmark/tech_reports/physics/experiments/dexuni/remeasure_sdk.py
Resume: same command with --resume. No runtime monkeypatches are installed.
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

import numpy as np

REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO))
from scripts.benchmark.tech_reports.physics.experiments.dexuni.run_s03 import _metrics
from scripts.benchmark.tech_reports.physics.experiments.dexuni.run_s04 import _analyze
from scripts.benchmark.tech_reports.physics.experiments.dexuni.s04_measurement import (
    file_hashes,
    write_json,
)

TECH = REPO / "scripts/benchmark/tech_reports"
ROOTS = {
    "s03": TECH.parents[2]
    / "outputs/benchmarks/tech_reports/dexuni/s03_cloth_corner_lift",
    "s04": TECH.parents[2]
    / "outputs/benchmarks/tech_reports/dexuni/s04_rigid_soft_squeeze_grasp",
}
AUDIT = ROOTS["s04"] / "analysis/sdk_integration"
SDK = Path("/home/oem/code/repos/dexsimforembodychain/dexsim/modules/dexsim")
INSTALLED = REPO / ".venv/lib/python3.11/site-packages/dexsim/engine/newton_physics"
PYTHONS = {
    "embodichain": REPO / ".venv/bin/python",
    "isaaclab": Path("/home/oem/app/IsaacLab/.venv/bin/python"),
}
S03_OVERRIDES = {
    "solver.dexsim_solver_extra": {
        "request_body_acceleration": False,
        "particle_coloring_include_bending": False,
        "particle_coloring_balance_colors": False,
    },
    "solver.dexsim_vbd_extra": {
        "particle_collision_detection_interval": -1,
        "particle_enable_surface_cache": True,
        "particle_enable_truncation_cache": True,
    },
}
__all__ = ["main"]


def _hash(path: Path) -> str:
    """Hash immutable artifact bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prepare() -> None:
    """Archive previous formal records and freeze the newly authorized protocol."""
    if (AUDIT / "remeasurement_protocol.json").exists():
        raise ValueError("Existing SDK measurements require --resume")
    installed = {}
    for name in ("newton_backend.py", "solvers_cfg.py", "_dexuni_options.py"):
        assert (INSTALLED / name).read_bytes() == (
            SDK / "python/dexsim/engine/newton_physics" / name
        ).read_bytes()
        installed[str(INSTALLED / name)] = _hash(INSTALLED / name)
    for name in (
        "modules/mjvbd_v2/vbd/solver_vbd.py",
        "modules/mjvbd_v2/vbd/particle_vbd_kernels.py",
    ):
        assert (INSTALLED / "_jit" / name).read_bytes() == (
            AUDIT / "staged_jit" / name
        ).read_bytes()
        installed[str(INSTALLED / "_jit" / name)] = _hash(INSTALLED / "_jit" / name)
    assert "265 passed" in (AUDIT / "regression_tests.log").read_text()
    assert "102 passed" in (AUDIT / "final_policy_tests.log").read_text()
    identity = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "authorization": "User explicitly authorized permanent changes in the specified SDK checkout and remeasurement of tasks 3 and 4.",
        "sdk_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=SDK, text=True
        ).strip(),
        "sdk_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "diff", "HEAD"], cwd=SDK)
        ).hexdigest(),
        "installed_sdk_sha256": installed,
        "scope": "performance_only",
        "repeats": 5,
        "warmup_steps": 30,
        "s03_overrides": S03_OVERRIDES,
        "s04_optimization": {
            "request_body_acceleration": False,
            "particle_enable_uniform_tet_specialization": True,
        },
        "numerical_validation": "all colors / rest and 0.1mm, 1mm perturbations / displacements and optional Hessians bitwise equal; uniform material rebuild and generic fallback tested",
        "configuration_note": "S03 aligns self-contact/coloring schedule to Isaac; S04 preserves original schedule. Both retain graph, dt, substeps, iterations, geometry/materials and named commands.",
        "sampling": "five isolated no-camera processes per platform/task; alternating platform/task order; serialized GPU; every attempt retained, no geometry filtering or replacement",
        "sources": {},
    }
    source_names = [
        "physics/experiments/dexuni/remeasure_sdk.py",
        "physics/experiments/dexuni/sdk_identity.py",
        "physics/experiments/dexuni/s03_common.py",
        "physics/experiments/dexuni/s04_common.py",
        "physics/experiments/dexuni/s04_measurement.py",
        *[f"physics/platforms/dexuni/{p}/{t}.py" for p in PYTHONS for t in ROOTS],
    ]
    identity["sources"] = {name: _hash(TECH / name) for name in source_names}
    for task, root in ROOTS.items():
        history = root / "history/formal_before_sdk"
        if not history.exists():
            history.mkdir(parents=True, exist_ok=False)
            names = (
                [
                    "REPORT.md",
                    "measurement",
                    "attempts.json",
                    "protocol.json",
                    "results",
                    "logs",
                    "completion.json",
                    "raw_hashes.json",
                    "report_builder.py",
                    "source",
                    "VALIDATION.md",
                    "parameter_audit.json",
                    "PARAMETER_CHECK.md",
                ]
                if task == "s03"
                else ["REPORT.md", "measurement"]
            )
            for name in names:
                if (root / name).exists():
                    shutil.move(str(root / name), str(history / name))
            write_json(
                history / "ARCHIVE.json",
                {
                    "reason": "immutable formal measurements before SDK integration",
                    "sha256": file_hashes(history),
                },
            )
        (root / "measurement/logs").mkdir(parents=True)
        for name in source_names:
            target = root / "measurement/source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(TECH / name, target)
        preview = root / (
            "videos/s03_comparison.mp4" if task == "s03" else "preview/comparison.mp4"
        )
        protocol = dict(
            identity,
            task=task,
            authorized=True,
            input_sha256=file_hashes(root / "inputs"),
            approved_preview_sha256=_hash(preview),
            approved_preview=str(preview),
        )
        protocol["preview_note"] = (
            "Previously accepted scene preview; latest explicit SDK optimization/remeasurement instruction authorizes this batch. S03 scheduling changes are disclosed separately."
        )
        if task == "s03":
            protocol["inputs_sha256"] = protocol["input_sha256"]
            write_json(root / "protocol.json", protocol)
        write_json(root / "measurement/protocol.json", protocol)
    (AUDIT / "sdk.patch").write_bytes(
        subprocess.check_output(["git", "diff", "HEAD"], cwd=SDK)
    )
    write_json(AUDIT / "remeasurement_protocol.json", identity)
    write_json(AUDIT / "remeasurement_attempts.json", [])


def _validate(attempt: dict) -> dict:
    """Audit full episode, source identity and effective SDK settings without shape filtering."""
    root = ROOTS[attempt["task"]]
    path = root / attempt["run"]
    manifest = json.loads((path / "manifest.json").read_text())
    metrics = (
        _metrics("s03", root, path) if attempt["task"] == "s03" else _analyze(attempt)
    )
    effective = manifest["effective"]
    protocol = json.loads((root / "measurement/protocol.json").read_text())
    checks = dict(metrics.get("performance_checks", metrics.get("checks", {})))
    checks["immutable_inputs"] = (
        file_hashes(root / "inputs") == protocol["input_sha256"]
    )
    checks["graph_captured"] = effective["cuda_graph_status"] == "captured"
    checks["budget"] = effective["num_substeps"] == 10 and effective.get("vbd", {}).get(
        "iterations", effective.get("vbd_iterations")
    ) == (10 if attempt["task"] == "s03" else 12)
    checks["outer_step_cadence"] = effective["outer_steps_per_command"] == 1
    count = 793 if attempt["task"] == "s03" else 720
    checks["elapsed_time"] = bool(
        np.isclose(effective["simulated_elapsed_s"], count / 60, rtol=0, atol=1e-8)
    )
    if attempt["task"] == "s03":
        checks["no_cameras"] = manifest["images"] == "none"
        if attempt["platform"] == "embodichain":
            checks["no_renderer"] = manifest["renderer"] == "no-render"
    if attempt["platform"] == "embodichain":
        loaded = {
            value["path"]: value["sha256"] for value in manifest["sdk_runtime"].values()
        }
        checks["installed_sdk_identity"] = all(
            loaded.get(path) == sha
            for path, sha in protocol["installed_sdk_sha256"].items()
        )
        checks["unused_outputs_omitted"] = not any(
            effective["state_attributes"].values()
        )
        if attempt["task"] == "s03":
            checks["reference_schedule_and_cache"] = (
                effective["particle_color_groups"] == 3
                and effective["vbd"]["particle_collision_detection_interval"] == -1
                and effective["vbd"]["surface_cache"]
                and effective["vbd"]["truncation_cache"]
            )
        else:
            checks["native_uniform_kernel_active"] = (
                effective["vbd"]["particle_uniform_tet_status"] == "specialized"
            )
    metrics.update(
        success=attempt["returncode"] == 0 and all(checks.values()),
        sdk_checks=checks,
        process_wall_s=attempt["process_wall_s"],
        effective=effective,
    )
    write_json(path / "validation.json", metrics)
    return metrics


def _save(attempts: list[dict]) -> None:
    """Keep the shared and per-task attempt ledgers in sync."""
    write_json(AUDIT / "remeasurement_attempts.json", attempts)
    for task, root in ROOTS.items():
        selected = [a for a in attempts if a["task"] == task]
        write_json(root / "measurement/attempts.json", selected)
        if task == "s03":
            write_json(root / "attempts.json", selected)


def main() -> None:
    """Run the fixed matrix, stopping on any unclassified failed attempt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.resume:
        _prepare()
    protocol = json.loads((AUDIT / "remeasurement_protocol.json").read_text())
    for path, sha in protocol["installed_sdk_sha256"].items():
        if _hash(Path(path)) != sha:
            raise ValueError("Installed SDK changed during measurement")
    for name, sha in protocol["sources"].items():
        if _hash(TECH / name) != sha:
            raise ValueError("Worker source changed during measurement")
    attempts = json.loads((AUDIT / "remeasurement_attempts.json").read_text())
    plan = [
        (t, p, r)
        for r in range(1, 6)
        for t in (list(ROOTS) if r % 2 else list(ROOTS)[::-1])
        for p in (list(PYTHONS) if r % 2 else list(PYTHONS)[::-1])
    ]
    if [(a["task"], a["platform"], a["repeat"]) for a in attempts] != plan[
        : len(attempts)
    ]:
        raise ValueError("Attempt prefix changed")
    if any(not a.get("success", False) for a in attempts):
        raise ValueError("Inspect failed attempts before continuing")
    for task, platform, repeat in plan[len(attempts) :]:
        root = ROOTS[task]
        command = [
            str(PYTHONS[platform]),
            str(TECH / f"physics/platforms/dexuni/{platform}/{task}.py"),
        ]
        if task == "s04":
            command += ["--measure", "--repeat", str(repeat)]
        else:
            command += [
                "--state",
                "s00",
                "--variant",
                "nominal",
                "--repeat",
                str(repeat),
                "--kind",
                "full",
                "--images",
                "none",
                "--out-root",
                str(root / "measurement/s03"),
                "--inputs",
                str(root / "inputs/s03"),
            ]
            for name, value in S03_OVERRIDES.items():
                command += [
                    "--set",
                    name + "=" + json.dumps(value, separators=(",", ":")),
                ]
        log = root / f"measurement/logs/{platform}_r{repeat:02d}.log"
        if log.exists():
            raise ValueError("An unrecorded attempt already exists")
        attempt = dict(
            task=task,
            platform=platform,
            repeat=repeat,
            mode="measurement",
            command=command,
            log=str(log.relative_to(root)),
            started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        )
        attempts.append(attempt)
        _save(attempts)
        print(f"[{len(attempts)}/20] {task} {platform} r{repeat:02d}", flush=True)
        start = time.perf_counter()
        env = dict(
            os.environ,
            DEXUNI_ALLOW_PILOT="1",
            DEXUNI_INPUT_ROOT=str(root / "inputs"),
            PYTHONUNBUFFERED="1",
        )
        with log.open("w") as stream:
            result = subprocess.run(
                command,
                cwd=REPO,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=1200,
            )
        attempt.update(
            returncode=result.returncode, process_wall_s=time.perf_counter() - start
        )
        run = (
            root / f"measurement/{platform}/r{repeat:02d}"
            if task == "s04"
            else next(
                (root / f"measurement/s03/raw_data/{platform}").glob(
                    f"s03_*r{repeat:02d}_*"
                ),
                None,
            )
        )
        try:
            if run is None:
                raise ValueError("No run directory")
            attempt["run"] = str(run.relative_to(root))
            attempt["metrics"] = _validate(attempt)
            attempt["success"] = attempt["metrics"]["success"]
        except Exception as error:
            attempt.update(success=False, error=repr(error))
        _save(attempts)
        if not attempt["success"]:
            raise RuntimeError(f"Failed attempt retained: {attempt}")
        metric = attempt["metrics"]
        print(
            f"  {metric.get('cost_time_ms', metric.get('step_ms')):.3f} ms; all checks passed",
            flush=True,
        )
    print("All 20 SDK measurements complete", flush=True)


if __name__ == "__main__":
    main()
