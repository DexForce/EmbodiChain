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

"""Compare four layout proposal strategies on three native Task Programs.

Run: python -m scripts.benchmark.expansion.scene_first --output DIR
Use --pilot first. Each case uses a fresh process, Default CPU physics, the
same full-state geometric/task-path screens and measured acceptance/receipts.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
from statistics import median
import subprocess
import sys
import time
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
TASKS = ROOT / "extensions/embodichain_tasks/configs/tasks/manipulation"
CASES = {
    "pick_place": TASKS / "workspace_pose_pick_place/acceptance/scene_first",
    "stack": TASKS / "tableware/stack_blocks_two/acceptance/scene_first",
    "dual_arm_sort": TASKS / "tableware/blocks_ranking_rgb/acceptance/scene_first",
}
STRATEGIES = ("bounded_pose", "scene_first", "robot_first", "bidirectional")
PLACEMENTS = {
    "pick_place": {"cube": ("goal", (0.0, 0.0))},
    "stack": {"block_2": ("block_1", (0.0, 0.0))},
    "dual_arm_sort": {
        "block_1": ("block_2", (-0.08, 0.0)),
        "block_3": ("block_2", (0.08, 0.0)),
    },
}


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def _xy_cell(pose, origin, bounds):
    """Four fixed XY bins per authored nonzero axis; never fit bins to results."""
    delta = np.asarray(pose)[:2] - np.asarray(origin)[:2]
    lo, hi = np.array(bounds)[:2, 0], np.array(bounds)[:2, 1]
    if np.any(delta < lo - 1e-5) or np.any(delta > hi + 1e-5):
        return None
    return tuple(
        min(3, max(0, int(np.floor(4 * (v - a) / (b - a))))) if b > a else 0
        for v, a, b in zip(delta, lo, hi)
    )


def _confirmed_episodes(root):
    """Read only receipt-confirmed evidence; a staged NPZ is not a commit."""
    from embodichain.lab.expansion.persistence import (
        ArtifactEpisodeSink,
        load_expansion_episode,
    )

    if not root.exists():
        return ()
    return tuple(
        load_expansion_episode(receipt.storage_id)
        for receipt in ArtifactEpisodeSink(root).inspect()
        if receipt.confirmed
    )


def _task_pose_cells(episode, case, cfg, nominal):
    """Count each manipulated object on frozen initial and placement domains."""
    cells = set()
    for uid, (goal, offset) in PLACEMENTS[case].items():
        values = episode.observations[f"rigid/{uid}/state"].cpu().numpy()
        initial_uid = cfg["scene"][uid].get("follow", uid)
        initial = _xy_cell(
            values[0], nominal[uid], cfg["scene"][initial_uid]["translation_m"]
        )
        if initial is not None:
            cells.add((uid, "initial", *initial))
        bounds = cfg["scene"].get(goal, {}).get("translation_m", [[0, 0]] * 3)
        if all(pair[0] == pair[1] for pair in bounds[:2]):
            # A fixed target contributes one category after measured acceptance;
            # tracking error inside its tolerance cannot create pose diversity.
            cells.add((uid, "placed-fixed"))
        else:
            origin = np.asarray(nominal[goal], dtype=float).copy()
            origin[:2] += offset
            placed = _xy_cell(values[-1], origin, bounds)
            if placed is not None:
                cells.add((uid, "placed", *placed))
    return cells


def _strategy_order(strategies: list[str], seed_index: int) -> list[str]:
    """Balance position and preceding strategy over four seeds in the full comparison."""
    width = len(strategies)
    order = [0, *(i // 2 + 1 if i % 2 else width - i // 2 for i in range(1, width))]
    return [strategies[(index + seed_index) % width] for index in order]


def _source_breakdown(events, checks):
    """Attribute screened and executed outcomes to the actual raw source."""
    origins = defaultdict(Counter)
    selections = {}
    rejected = Counter()
    for e in events:
        if e["event"] == "proposal":
            source = origins[e["origin"]]
            source["proposed"] += 1
            source["screened"] += int(e["screened"])
            if not e["screened"]:
                rejected[e["reason"]] += 1
            elif e["origin"] == "robot_first":
                anchor = checks["workspace_anchor_checks"][str(e["raw_ordinal"])]
                actual = e["pose_changes"][anchor["root_uid"]]
                if not np.allclose(actual, anchor["pose"], atol=1e-9, rtol=0):
                    raise ValueError(
                        "Conditioned layout did not preserve the workspace anchor"
                    )
        elif e["event"] == "selection":
            selections[e["ordinal"]] = e["origin"]
        elif e["event"] == "outcome":
            source = origins[selections[e["ordinal"]]]
            for key in ("executed", "physical_accepted", "committed"):
                source[key] += int(e[key])
    return {k: dict(v) for k, v in origins.items()}, dict(rejected)


def worker(args):
    """Execute one isolated source/case/seed with complete failure denominators."""
    sys.path[:0] = [
        str(ROOT),
        str(ROOT / "extensions/embodichain_tasks"),
        str(ROOT / "extensions/embodichain_robots"),
    ]
    from embodichain.lab.expansion.task_program_jobs import run_task_program_expansion

    directory = args.output
    directory.mkdir(parents=True, exist_ok=False)
    cfg = yaml.safe_load((CASES[args.case] / "expansion.yaml").read_text())
    cfg["scene_sampling"]["strategy"] = args.strategy
    cfg["engine"].update(
        episodes_per_operator=args.rollouts,
        max_attempts=args.candidates,
        max_wall_time_s=args.seconds,
    )
    cfg["scene_sampling"].update(
        max_candidates=args.candidates, max_rollouts=args.rollouts
    )
    # One raw draw per attempt across all four variants. The screening budgets
    # and measured acceptance remain those declared by the fixture.
    cfg["scheduling"]["pool_size"] = 1
    if args.pilot:
        for value in cfg["scene"].values():
            if "follow" not in value:
                value["translation_m"] = [[0, 0]] * 3
        for key in cfg["scene_sampling"]["layout"]["policy"][
            "group_xy_offset_bounds_m"
        ]:
            cfg["scene_sampling"]["layout"]["policy"]["group_xy_offset_bounds_m"][
                key
            ] = [[0, 0], [0, 0]]
    config = directory / "expansion.yaml"
    config.write_text(yaml.safe_dump(cfg, sort_keys=False))
    files = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        text=True,
    ).splitlines()
    paths = [
        p
        for p in files
        if p.startswith(
            (
                "embodichain/lab/expansion/",
                "embodichain/lab/sim/scene/",
                "embodichain/compute/geometry/",
                "extensions/embodichain_tasks/configs/tasks/manipulation/",
            )
        )
        and (ROOT / p).is_file()
    ]
    paths.append(str(Path(__file__).resolve().relative_to(ROOT)))
    _write(
        directory / "source-manifest.json",
        {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths},
    )
    started = time.perf_counter()
    result = None
    error = None
    try:
        result = run_task_program_expansion(
            CASES[args.case] / "task.yaml",
            config,
            directory / "collection",
            seed=args.seed,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        import traceback

        traceback.print_exc()
    run_wall_s = time.perf_counter() - started
    root = directory / "collection/dataset/object_pose"
    source = root / "pose-sampling.json"
    data = json.loads(source.read_text()) if source.exists() else {}
    events = data.get("events", [])
    outcomes = [e for e in events if e["event"] == "outcome"]
    screen_path = root / "scene-first-checks.json"
    screens = json.loads(screen_path.read_text()) if screen_path.exists() else {}
    origins, proposal_rejections = _source_breakdown(events, screens)
    failures = {}
    for path in root.glob("attempts/*/attempt.json"):
        d = json.loads(path.read_text())
        if d.get("status") != "committed":
            key = d.get("reason", d.get("status", "unavailable"))
            failures[key] = failures.get(key, 0) + 1
    nominal_path = directory / "collection/reference-scene.json"
    nominal = (
        json.loads(nominal_path.read_text())["poses_xyzw"]
        if nominal_path.exists()
        else {}
    )
    independent = [uid for uid, opts in cfg["scene"].items() if "follow" not in opts]
    joint = set()
    marginal = set()
    task = set()
    audits = []
    committed = 0
    for episode in _confirmed_episodes(root):
        committed += 1
        row = []
        for uid in independent:
            states = episode.observations[f"rigid/{uid}/state"]
            cell = _xy_cell(
                states[0].cpu().numpy(),
                nominal[uid],
                cfg["scene"][uid]["translation_m"],
            )
            if cell is not None:
                marginal.add((uid, *cell))
            row.append((uid, cell))
        if all(cell is not None for _, cell in row):
            joint.add(tuple((uid, *cell) for uid, cell in row))
        task.update(_task_pose_cells(episode, args.case, cfg, nominal))
        steps = len(episode.actions)
        if (
            not episode.validation.accepted
            or len(episode.timestamps) != steps + 1
            or any(len(v) != steps + 1 for v in episode.observations.values())
            or episode.metadata["source_env_id"] != 0
        ):
            raise ValueError("Invalid measured episode evidence at confirmed readback")
        arm_motion = {}
        if args.case == "dual_arm_sort":
            q = episode.observations["full_joint_positions"].cpu().numpy()
            names = episode.metadata["joint_names"]
            for arm in ("left", "right"):
                columns = [
                    i for i, name in enumerate(names) if name.startswith(f"{arm}_joint")
                ]
                arm_motion[arm] = float(np.ptp(q[:, columns], axis=0).max())
            if min(arm_motion.values()) <= 0.1:
                raise ValueError(
                    "Dual-arm sorting requires measured motion in both arms"
                )
        audits.append(
            {
                "episode_id": episode.episode_id,
                "T": len(episode.actions),
                "observations": len(episode.timestamps),
                "source_row": episode.metadata["source_env_id"],
                "validation": episode.validation.accepted,
                "arm_joint_range_rad": arm_motion,
            }
        )
    slots_per_uid = {
        uid: int(
            np.prod(
                [
                    4 if pair[1] > pair[0] else 1
                    for pair in cfg["scene"][uid]["translation_m"][:2]
                ]
            )
        )
        for uid in independent
    }
    metric = {
        "case": args.case,
        "strategy": args.strategy,
        "seed": args.seed,
        "pilot": args.pilot,
        "budget": data.get("budget", {}),
        "declared_budget": {
            "candidates": args.candidates,
            "rollouts": args.rollouts,
            "seconds": args.seconds,
        },
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "execution": (
            json.loads((directory / "collection/execution.json").read_text())
            if (directory / "collection/execution.json").exists()
            else None
        ),
        "error": error,
        "wall_s": run_wall_s,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "executed": sum(bool(e["executed"]) for e in outcomes),
        "physically_accepted": sum(bool(e["physical_accepted"]) for e in outcomes),
        "committed": committed,
        "scene_joint_cells": len(joint),
        "scene_joint_declared_cells": int(np.prod(list(slots_per_uid.values()))),
        "scene_marginal_cells": len(marginal),
        "scene_marginal_declared_cells": sum(slots_per_uid.values()),
        "task_pose_cells": len(task),
        "coverage_cells": {
            "scene_joint": sorted(joint),
            "scene_marginal": sorted(marginal),
            "task": sorted(task),
        },
        "preparation_rejected": sum(not e["executed"] for e in outcomes),
        "proposal_rejections": proposal_rejections,
        "origins": origins,
        "workspace_prior": screens.get("workspace_prior", []),
        "workspace_prepare_wall_s": screens.get("workspace_prepare_wall_s"),
        "proposal_screen_wall_s": screens.get("proposal_screen_wall_s"),
        "failures": failures,
        "episode_audit": audits,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["torch", "dexsim-engine"]
        },
        "result": result,
    }
    _write(directory / "metrics.json", metric)
    print(json.dumps(metric), flush=True)
    if error:
        raise SystemExit(1)


def _aggregate(rows):
    """Combine unique receipt-confirmed coverage across seeds in a fixed domain."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["case"], row["strategy"]].append(row)
    result = []
    for (case, strategy), runs in sorted(groups.items()):
        origins = defaultdict(Counter)
        for run in runs:
            for origin, values in run.get("origins", {}).items():
                origins[origin].update(values)
        coverage = {}
        for kind in ("scene_joint", "scene_marginal", "task"):
            if all("coverage_cells" in run for run in runs):
                coverage[kind] = len(
                    {
                        json.dumps(cell)
                        for run in runs
                        for cell in run["coverage_cells"][kind]
                    }
                )
            else:
                coverage[kind] = None
        totals = {
            k: sum(run[k] for run in runs)
            for k in (
                "executed",
                "physically_accepted",
                "committed",
                "preparation_rejected",
            )
        }
        proposed = [run["budget"].get("candidates") for run in runs]
        result.append(
            dict(
                case=case,
                strategy=strategy,
                seeds=[r["seed"] for r in runs],
                raw_proposals=(
                    None if any(v is None for v in proposed) else sum(proposed)
                ),
                median_wall_s=median(run["wall_s"] for run in runs),
                total_wall_s=sum(run["wall_s"] for run in runs),
                median_screen_s=median(
                    [
                        run["proposal_screen_wall_s"]
                        for run in runs
                        if run.get("proposal_screen_wall_s") is not None
                    ]
                    or [0.0]
                ),
                peak_rss_mib=max(run["peak_rss_mib"] for run in runs),
                scene_joint_declared=max(
                    run["scene_joint_declared_cells"] for run in runs
                ),
                scene_marginal_declared=max(
                    run["scene_marginal_declared_cells"] for run in runs
                ),
                coverage=coverage,
                origins={key: dict(val) for key, val in origins.items()},
                failed_runs=[
                    f'{run["seed"]}: {run["error"]}' for run in runs if run["error"]
                ],
                **totals,
            )
        )
    return result


def report(output, rows):
    """Write aggregated comparisons and preserve seed-level results separately."""
    aggregate = _aggregate(rows)
    _write(output / "results.json", rows)
    _write(output / "aggregate.json", aggregate)
    lines = [
        "# Conditional layout sampling across three tasks",
        "",
        "Default CPU physics, B=1, sensors disabled. Each strategy uses the same",
        "task domains, layout and path checks, runtime replanning, physical",
        "acceptance and artifact/LeRobot export. CUDA grasp classification may be",
        "used. Cold wall time covers the expansion call, including initialization",
        "and export, excluding process startup and benchmark readback. RSS is the",
        "process high-water mark; incremental CPU/GPU allocations are unmeasured.",
        "",
        "## Time & Memory",
        "",
        "|Task|Strategy|Seeds|Median cost_time_ms|Median screen s|Wall s / receipt|Peak RSS MiB|cpu_delta_mb|gpu_delta_mb|peak_gpu_mb|",
        "|---|---|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for r in aggregate:
        per_episode = (
            f'{r["total_wall_s"]/r["committed"]:.2f}' if r["committed"] else "N/A"
        )
        lines.append(
            f"|{r['case']}|{r['strategy']}|{len(r['seeds'])}|{r['median_wall_s']*1000:.1f}|"
            f"{r['median_screen_s']:.2f}|{per_episode}|{r['peak_rss_mib']:.1f}|N/A|N/A|N/A|"
        )
    lines += [
        "",
        "## Success & Other Metrics",
        "",
        "|Task|Strategy|Raw proposals|Prep rejected|Accepted / executed|Committed|Marginal XY cells|Joint XY cells|Task pose cells|Failed runs|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in aggregate:
        coverage = r["coverage"]
        number = lambda x: "N/A" if x is None else str(x)
        lines.append(
            f"|{r['case']}|{r['strategy']}|{number(r['raw_proposals'])}|{r['preparation_rejected']}|"
            f"{r['physically_accepted']}/{r['executed']}|{r['committed']}|"
            f"{number(coverage['scene_marginal'])}/{r['scene_marginal_declared']}|"
            f"{number(coverage['scene_joint'])}/{r['scene_joint_declared']}|"
            f"{number(coverage['task'])}|{len(r['failed_runs'])}|"
        )
    lines += [
        "",
        "## Leaderboard",
        "",
        "Order within each task by physical acceptance per executed rollout.",
        "",
        "|Task|Strategy|Physical acceptance|Confirmed episodes|",
        "|---|---|---:|---:|",
    ]
    for r in sorted(
        aggregate,
        key=lambda r: (
            r["case"],
            -r["physically_accepted"] / r["executed"] if r["executed"] else 1.0,
        ),
    ):
        rate = (
            f"{r['physically_accepted']/r['executed']:.3f}"
            if r["executed"]
            else "N/A (no rollout)"
        )
        lines.append(f"|{r['case']}|{r['strategy']}|{rate}|{r['committed']}|")
    lines += [
        "",
        "Scene coverage is the union of fixed XY grid cells from receipt-confirmed",
        "measured initial layouts. Task cells use measured initial/terminal object",
        "positions; each fixed placement is a single category. The counted cells",
        "describe observed sample diversity in the fixture domains.",
        "See results.json for each seed, proposal failures, source usage, recorded",
        "budgets, hardware affinity, and readback audit. Aggregated wall times",
        "include all recorded runs, including errors.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def main():
    """Run fresh worker processes sequentially, alternating source order by seed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--cases", default=",".join(CASES))
    parser.add_argument("--strategies", default=",".join(STRATEGIES))
    parser.add_argument("--case", choices=tuple(CASES), default="pick_place")
    parser.add_argument("--strategy", choices=STRATEGIES, default="scene_first")
    parser.add_argument("--seeds", default="7,17")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--seconds", type=float, default=600)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.worker:
        worker(args)
        return
    if args.rebuild:
        report(
            args.output,
            [
                json.loads(p.read_text())
                for p in sorted(args.output.rglob("metrics.json"))
            ],
        )
        return
    cases = args.cases.split(",")
    strategies = args.strategies.split(",")
    if any(case not in CASES for case in cases) or len(set(cases)) != len(cases):
        parser.error("Select distinct task cases from " + ",".join(CASES))
    if any(s not in STRATEGIES for s in strategies) or len(set(strategies)) != len(
        strategies
    ):
        parser.error("Select distinct strategies from " + ",".join(STRATEGIES))
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for case in cases:
        for index, seed in enumerate(map(int, args.seeds.split(","))):
            for strategy in _strategy_order(strategies, index):
                directory = args.output / f"{case}-{strategy}-{seed}"
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--output",
                    str(directory),
                    "--case",
                    case,
                    "--strategy",
                    strategy,
                    "--seed",
                    str(seed),
                    "--rollouts",
                    str(args.rollouts),
                    "--candidates",
                    str(args.candidates),
                    "--seconds",
                    str(args.seconds),
                ]
                if args.pilot:
                    command += ["--pilot"]
                attempt_started = time.perf_counter()
                env = {
                    **os.environ,
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "SDL_VIDEODRIVER": "offscreen",
                }
                with (args.output / (directory.name + ".log")).open("w") as log:
                    try:
                        proc = subprocess.run(
                            command,
                            cwd=ROOT,
                            env=env,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            timeout=args.seconds + 180,
                        )
                        worker_error = f"Worker exited with status {proc.returncode}"
                    except subprocess.TimeoutExpired:
                        worker_error = "Worker exceeded the hard process timeout"
                metrics = directory / "metrics.json"
                if not metrics.exists():
                    directory.mkdir(exist_ok=True)
                    _write(
                        metrics,
                        {
                            "case": case,
                            "strategy": strategy,
                            "seed": seed,
                            "budget": {},
                            "declared_budget": {
                                "candidates": args.candidates,
                                "rollouts": args.rollouts,
                                "seconds": args.seconds,
                            },
                            "error": worker_error,
                            "wall_s": time.perf_counter() - attempt_started,
                            "peak_rss_mib": 0,
                            "executed": 0,
                            "physically_accepted": 0,
                            "committed": 0,
                            "scene_joint_declared_cells": 0,
                            "scene_marginal_declared_cells": 0,
                            "preparation_rejected": 0,
                        },
                    )
                row = json.loads(metrics.read_text())
                rows.append(row)
                report(args.output, rows)
                if row["error"]:
                    print(
                        f"Worker failed: {directory}; diagnostics retained", flush=True
                    )
    print(args.output / "report.md", flush=True)


if __name__ == "__main__":
    main()
