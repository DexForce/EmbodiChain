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

"""Compare bounded/workspace sampling and FIFO/coverage scheduling.

Run: python -m scripts.benchmark.expansion.task_pose_coverage --output DIR
Use --suite formal only after the smoke scene qualification succeeds.
The worker uses real physics, measured acceptance and native persistence.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import statistics
import yaml

ROOT = Path(__file__).resolve().parents[3]
TASK = (
    ROOT
    / "extensions/embodichain_tasks/configs/tasks/manipulation/workspace_pose_pick_place"
)
GROUPS = {
    "A": ("bounded_pose", "fifo"),
    "B": ("bounded_pose", "task_pose_coverage"),
    "C": ("workspace_joint", "fifo"),
    "D": ("workspace_joint", "task_pose_coverage"),
}


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def _worker(args) -> None:
    sys.path[:0] = [
        str(ROOT),
        str(ROOT / "extensions/embodichain_tasks"),
        str(ROOT / "extensions/embodichain_robots"),
    ]
    from embodichain.lab.expansion.task_program_jobs import run_task_program_expansion

    args.output.mkdir(parents=True, exist_ok=False)
    cfg = yaml.safe_load((TASK / "expansion.yaml").read_text())
    cfg["scene_sampling"]["strategy"], cfg["scheduling"]["policy"] = GROUPS[args.group]
    cfg["engine"].update(
        episodes_per_operator=args.rollouts,
        max_attempts=args.candidates,
        batch_size=args.rows,
        max_wall_time_s=args.seconds,
    )
    cfg["scene_sampling"].update(
        max_candidates=args.candidates,
        max_rollouts=args.rollouts,
        max_ik_calls=args.ik_calls,
        max_planning_calls=args.planning_calls,
    )
    if args.nominal:
        for v in cfg["scene"].values():
            v.update(translation_m=[[0, 0], [0, 0], [0, 0]], yaw_rad=[0, 0])
    cfg["trajectory"]["operators"] = [args.operator]
    config = args.output / "configuration.yaml"
    config.write_text(yaml.safe_dump(cfg, sort_keys=False))
    deployment = TASK / (
        "task.obstacle.ur5.yaml" if args.scene == "obstacle" else "task.ur5.yaml"
    )
    raw = yaml.safe_load(deployment.read_text())
    # Resolve every component before writing an installation-specific deployment.
    for key in ("environment", "embodiment"):
        raw[key]["component"] = str((TASK / raw[key]["component"]).resolve())
    for key in ("program", "integration", "execution_policy"):
        raw["task_program"][key] = str((TASK / raw["task_program"][key]).resolve())
    if args.installation != "nominal":
        sign = 1 if args.installation == "positive" else -1
        raw["embodiment"]["overrides"].update(
            init_pos=[sign * 0.05, 0, 0], init_rot=[0, 0, sign * 10]
        )
    if args.media:
        cfg["engine"]["capture_media"] = True
        config.write_text(yaml.safe_dump(cfg, sort_keys=False))
    dep = args.output / "deployment.yaml"
    dep.write_text(yaml.safe_dump(raw, sort_keys=False))
    import hashlib

    source_paths = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        text=True,
    ).splitlines()
    selected = [
        ROOT / path
        for path in source_paths
        if path.startswith(
            (
                "embodichain/lab/expansion/",
                "embodichain/lab/sim/atomic_skills/",
                "embodichain/lab/gym/envs/task_program/",
                "embodichain/lab/task_program/",
                "extensions/embodichain_tasks/configs/tasks/manipulation/workspace_pose_pick_place/",
                "scripts/benchmark/expansion/task_pose_coverage.py",
            )
        )
        and (ROOT / path).is_file()
    ]
    _write(
        args.output / "source-manifest.json",
        {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in selected
        },
    )
    start = time.perf_counter()
    result = None
    error = None
    try:
        result = run_task_program_expansion(
            dep, config, args.output / "collection", seed=args.seed
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        import traceback

        traceback.print_exc()
    samples = list((args.output / "collection").rglob("pose-sampling.json"))
    data = json.loads(samples[0].read_text()) if samples else {}
    events = data.get("events", [])
    outcomes = [e for e in events if e["event"] == "outcome"]
    confirmed = data.get("coverage", {}).get("confirmed", {})
    cells = {tuple(cell) for _, values in confirmed.values() for cell in values}
    counts = {p: sum(c[0] == p for c in cells) for p in ("pick", "place")}
    failure = {}
    for record in (args.output / "collection").rglob("attempt.json"):
        d = json.loads(record.read_text())
        if d.get("status") != "committed":
            reason = d.get("reason", d.get("status", "unknown"))
            failure[reason] = failure.get(reason, 0) + 1
    versions = {
        name: importlib.metadata.version(name) for name in ("torch", "dexsim-engine")
    }
    summary = {
        "source_digest": hashlib.sha256(
            (args.output / "source-manifest.json").read_bytes()
        ).hexdigest(),
        "suite": args.suite,
        "declared_budget": {
            "rollouts": args.rollouts,
            "candidates": args.candidates,
            "ik_calls": args.ik_calls,
            "planning_calls": args.planning_calls,
            "seconds": args.seconds,
        },
        "group": args.group,
        "scene": args.scene,
        "installation": args.installation,
        "seed": args.seed,
        "rows": args.rows,
        "operator": args.operator,
        "nominal": args.nominal,
        "media": args.media,
        "execution_status": "error" if error else "completed",
        "error": error,
        "wall_s": time.perf_counter() - start,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "committed": len(confirmed),
        "physical_accepted": sum(e["physical_accepted"] for e in outcomes),
        "executed": sum(e["executed"] for e in outcomes),
        "coverage": counts,
        "unique_cells": len(cells),
        "budget": data.get("budget", {}),
        "failures": failure,
        "versions": versions,
        "result": result,
        "coverage_domain": data.get("domain_id"),
        "path": str(args.output),
    }
    _write(args.output / "metrics.json", summary)
    print(json.dumps(summary), flush=True)
    if error:
        raise SystemExit(1)


def _report(output: Path, rows: list[dict]) -> None:
    _write(output / "results.json", rows)
    lines = [
        "# Task pose coverage acceptance",
        "",
        "Default CPU physics; cold end-to-end preparation, planning, rollout, validation and export. "
        "Grasp classification may use CUDA. RSS is a process high-water mark; GPU memory unmeasured. "
        "Only receipt-confirmed, physically validated episodes increase coverage.",
        "",
        "## Time & Memory",
        "",
        "|Case|Group|Seed|cost_time_ms|peak RSS MiB|cpu_delta_mb|gpu_delta_mb|peak_gpu_mb|",
        "|---|---|---:|---:|---:|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"|{r['scene']}/{r['installation']}/B={r['rows']}|{r['group']}|{r['seed']}|{r['wall_s']*1000:.1f}|{r['peak_rss_mib']:.1f}|N/A|N/A|N/A|"
        )
    lines += [
        "",
        "## Success & Other Metrics",
        "",
        "|Case|Group|Executed|Physically accepted|Committed|Pick cells|Place cells|Outcome|",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"|{r['scene']}/{r['installation']}|{r['group']}|{r['executed']}|{r['physical_accepted']}|{r['committed']}|{r['coverage']['pick']}|{r['coverage']['place']}|{r['error'] or 'completed'}|"
        )
    lines += [
        "",
        "## Leaderboard",
        "",
        "Ranked by measured acceptance rate; coverage gates are evaluated separately.",
        "",
        "|Group|Accepted / executed|Acceptance rate|Confirmed unique cells (sum across separate runs)|",
        "|---|---|---:|---:|",
    ]
    aggregates = []
    for group in GROUPS:
        selected = [r for r in rows if r["group"] == group]
        if selected:
            n = sum(r["executed"] for r in selected)
            k = sum(r["physical_accepted"] for r in selected)
            aggregates.append((group, k, n, sum(r["unique_cells"] for r in selected)))
    for g, k, n, u in sorted(
        aggregates, key=lambda v: v[1] / max(v[2], 1), reverse=True
    ):
        lines.append(f"|{g}|{k}/{n}|{k/max(n,1):.3f}|{u}|")
    gates = {}
    for scene in ("open", "obstacle"):
        pairs = []
        for seed in (7, 17, 27, 37, 47):
            matched = {
                r["group"]: r
                for r in rows
                if r["scene"] == scene
                and r["seed"] == seed
                and r["installation"] == "nominal"
                and not r["nominal"]
                and not r["media"]
                and r["rows"] == 1
                and r["operator"] == "object_pose"
                and r.get("suite") == "formal"
                and r.get("execution_status") == "completed"
                and r.get("error") is None
                and r.get("declared_budget")
                == {
                    "rollouts": 128,
                    "candidates": 4096,
                    "ik_calls": 32768,
                    "planning_calls": 4096,
                    "seconds": 1800,
                }
            }
            if (
                set(matched) >= {"C", "D"}
                and matched["C"].get("coverage_domain")
                == matched["D"].get("coverage_domain")
                and matched["C"].get("versions") == matched["D"].get("versions")
                and matched["C"].get("source_digest")
                == matched["D"].get("source_digest")
                and bool(matched["C"].get("source_digest"))
            ):
                pairs.append((matched["C"], matched["D"]))
        if len(pairs) != 5 or any(c["unique_cells"] == 0 for c, d in pairs):
            gates[scene] = {
                "status": "unqualified",
                "reason": "Need five complete paired seeds with nonzero baseline coverage",
            }
            continue
        gains = [d["unique_cells"] / c["unique_cells"] - 1 for c, d in pairs]
        phase_ok = all(
            statistics.median(d["coverage"][p] for c, d in pairs)
            >= statistics.median(c["coverage"][p] for c, d in pairs)
            for p in ("pick", "place")
        )
        rate = lambda r: r["physical_accepted"] / max(1, r["executed"])
        passed = (
            statistics.median(gains) >= 0.15
            and sum(g > 0 for g in gains) >= 4
            and phase_ok
            and statistics.median(rate(d) - rate(c) for c, d in pairs) >= -0.05
        )
        gates[scene] = {
            "status": "passed" if passed else "failed",
            "median_gain": statistics.median(gains),
            "positive_seeds": sum(g > 0 for g in gains),
        }
    _write(output / "acceptance.json", gates)
    lines += [
        "",
        "## Acceptance gates",
        "",
        json.dumps(gates, indent=2),
        "",
        "Raw proposal/outcome logs and receipt checkpoints remain in each run directory. Single-point nominal pilots do not qualify broad-domain improvement.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    # Offline plot, without importing simulation during report rebuild.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for r in rows:
            logs = list(Path(r["path"]).rglob("pose-sampling.json"))
            if not logs:
                continue
            events = [
                e
                for e in json.loads(logs[0].read_text())["events"]
                if e["event"] == "outcome"
            ]
            for ax, key in zip(axes, ("proposals", "rollouts", "elapsed_s")):
                ax.step(
                    [e.get(key, 0) for e in events],
                    [sum(e["coverage"].values()) for e in events],
                    where="post",
                    label=f"{r['group']}/{r['scene']}/{r['seed']}",
                )
                ax.set_xlabel(key)
                ax.set_ylabel("Confirmed task pose cells")
        if axes[-1].lines:
            axes[-1].legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(output / "coverage.svg")
        plt.close(fig)
    except ImportError:
        pass


def main() -> None:
    """Run isolated comparable trials or rebuild reports from their artifacts."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--worker", action="store_true")
    p.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seed subset; omitted uses suite defaults",
    )
    p.add_argument(
        "--scenes", default=None, help="Comma-separated open/obstacle subset"
    )
    p.add_argument(
        "--suite",
        choices=("pilot", "smoke", "formal", "installation", "parallel", "trajectory"),
        default="smoke",
    )
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--group", choices=tuple(GROUPS), default="A")
    p.add_argument("--scene", choices=("open", "obstacle"), default="open")
    p.add_argument(
        "--installation", choices=("nominal", "positive", "negative"), default="nominal"
    )
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--rows", type=int, default=1)
    p.add_argument("--rollouts", type=int, default=12)
    p.add_argument("--candidates", type=int, default=384)
    p.add_argument("--ik-calls", type=int, default=32768)
    p.add_argument("--planning-calls", type=int, default=4096)
    p.add_argument("--seconds", type=float, default=300)
    p.add_argument("--nominal", action="store_true")
    p.add_argument("--media", action="store_true")
    p.add_argument(
        "--operator", choices=("object_pose", "joint_residual"), default="object_pose"
    )
    args = p.parse_args()
    args.output = args.output.resolve()
    if args.worker:
        _worker(args)
        return
    if args.rebuild:
        _report(
            args.output,
            [
                json.loads(f.read_text())
                for f in sorted(args.output.glob("*/metrics.json"))
            ],
        )
        return
    args.output.mkdir(parents=True, exist_ok=False)
    seeds = (
        [7, 17, 27, 37, 47]
        if args.suite == "formal"
        else [7, 17, 27] if args.suite == "installation" else [7]
    )
    scenes = ["open", "obstacle"] if args.suite in ("formal", "pilot") else [args.scene]
    if args.seeds is not None:
        seeds = [int(value) for value in args.seeds.split(",")]
    if args.scenes is not None:
        scenes = args.scenes.split(",")
        if not set(scenes) <= {"open", "obstacle"}:
            p.error("Unknown scene")
    installations = (
        ["positive", "negative"] if args.suite == "installation" else ["nominal"]
    )
    groups = ["C", "D"] if args.suite == "installation" else list(GROUPS)
    rows = []
    _write(
        args.output / "provenance.json",
        {
            "head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "diff": subprocess.check_output(["git", "diff"], cwd=ROOT, text=True),
            "scope": vars(args),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
        },
    )
    for scene in scenes:
        for installation in installations:
            for i, seed in enumerate(seeds):
                for group in (
                    groups
                    if (
                        (7, 17, 27, 37, 47).index(seed)
                        if seed in (7, 17, 27, 37, 47)
                        else i
                    )
                    % 2
                    == 0
                    else groups[::-1]
                ):
                    destination = args.output / f"{scene}-{installation}-{group}-{seed}"
                    n = (
                        128
                        if args.suite == "formal"
                        else (
                            64
                            if args.suite == "installation"
                            else 1 if args.suite == "pilot" else args.rollouts
                        )
                    )
                    cap = (
                        4096
                        if args.suite == "formal"
                        else 2048 if args.suite == "installation" else args.candidates
                    )
                    seconds = (
                        1800
                        if args.suite == "formal"
                        else 900 if args.suite == "installation" else args.seconds
                    )
                    command = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker",
                        "--suite",
                        args.suite,
                        "--output",
                        str(destination),
                        "--group",
                        group,
                        "--scene",
                        scene,
                        "--seed",
                        str(seed),
                        "--installation",
                        installation,
                        "--rows",
                        str(4 if args.suite == "parallel" else args.rows),
                        "--rollouts",
                        str(n),
                        "--candidates",
                        str(cap),
                        "--seconds",
                        str(seconds),
                        "--operator",
                        (
                            "joint_residual"
                            if args.suite == "trajectory"
                            else args.operator
                        ),
                    ]
                    if args.suite == "installation":
                        command += ["--ik-calls", "16384", "--planning-calls", "2048"]
                    if args.suite == "pilot" or args.nominal:
                        command += ["--nominal"]
                    if args.media:
                        command += ["--media"]
                    with (args.output / (destination.name + ".log")).open("w") as log:
                        try:
                            result = subprocess.run(
                                command,
                                cwd=ROOT,
                                env={
                                    **os.environ,
                                    "OMP_NUM_THREADS": "1",
                                    "MKL_NUM_THREADS": "1",
                                    "SDL_VIDEODRIVER": "offscreen",
                                },
                                stdout=log,
                                stderr=subprocess.STDOUT,
                                timeout=seconds + 180,
                            )
                        except subprocess.TimeoutExpired:
                            raise RuntimeError(
                                f"Worker timeout; preserve log {log.name}"
                            )
                    if (destination / "metrics.json").exists():
                        rows.append(
                            json.loads((destination / "metrics.json").read_text())
                        )
                        _report(args.output, rows)
                    if result.returncode:
                        raise RuntimeError(f"Worker failed; inspect {destination}")
                    if args.suite == "pilot" and not rows[-1]["committed"]:
                        raise RuntimeError(
                            "Nominal scene qualification failed; do not relax acceptance or narrow the declared domain"
                        )
    print(args.output / "report.md")


if __name__ == "__main__":
    main()
