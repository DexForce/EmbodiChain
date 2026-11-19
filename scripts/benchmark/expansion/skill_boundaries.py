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

"""Analyze real pick/place handoffs and plot paired measured/command signals.

Run: python -m scripts.benchmark.expansion.skill_boundaries --input COMPARISON_DIR
Input contains runs.json and A/D-repeat-before/after native benchmark outputs.
The collection command and fixed CPU/GPU settings are retained in each run log.
"""

from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
import hashlib
from pathlib import Path
import numpy as np

__all__: list[str] = []


def _episodes(directory: Path) -> list[dict]:
    result = []
    for path in sorted(
        (directory / "collection/dataset/object_pose/episodes").glob("*/episode.npz")
    ):
        with np.load(path, allow_pickle=False) as arrays:
            m = json.loads(arrays["manifest"].tobytes())
            meta = m["metadata"]
            names = meta["expert_action_spec"]["joint_names"]
            # This acceptance task explicitly controls the first six UR5 arm joints.
            q = arrays[
                "observation_" + str(m["observations"].index("full_joint_positions"))
            ][:, :6]
            a = arrays["actions"][:, :6]
            t = arrays["timestamps"]
            for segment in meta["program_segments"][1:]:
                k = segment["start_step"]
                if not 2 <= k < len(a) - 1:
                    continue
                dt = np.diff(t)
                command_speed = np.diff(a, axis=0) / dt[:-1, None]
                measured_speed = np.diff(q, axis=0) / dt[:, None]
                row = {
                    "episode": m["episode_id"],
                    "ordinal": meta["expansion_recipe"]["ordinal"],
                    "boundary": segment["name"],
                    "step": k,
                    "duration_s": float(t[-1] - t[0]),
                    "steps": len(a),
                    "joint_names": names[:6],
                    "command_jump_rad": float(np.abs(a[k] - a[k - 1]).max()),
                    "command_difference_speed_rad_s": float(
                        np.abs(command_speed[k - 1]).max()
                    ),
                    "command_difference_acceleration_rad_s2": float(
                        np.abs(
                            (command_speed[k - 1] - command_speed[k - 2]) / dt[k]
                        ).max()
                    ),
                    "measured_control_speed_rad_s": float(
                        np.abs(measured_speed[k - 1]).max()
                    ),
                    "initial_objects": {
                        v["uid"]: v["pose"]
                        for v in meta["initial_state"]
                        if v["kind"] == "rigid_object"
                    },
                    "artifact": str(path),
                }
                ix = np.arange(max(1, k - 15), min(len(a) - 1, k + 16))
                row["curve"] = {
                    "time_s": (t[ix] - t[k]).tolist(),
                    "command_q": a[ix].tolist(),
                    "measured_q": q[ix].tolist(),
                    "command_difference_velocity": command_speed[ix - 1].tolist(),
                    "measured_difference_velocity": measured_speed[ix - 1].tolist(),
                }
                result.append(row)
    return result


def _plot(before: dict, after: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(6, 2, figsize=(12, 15), sharex=True)
    colors = {"before": "#3267a8", "after": "#df7c28"}
    for label, row in (("before", before), ("after", after)):
        c = row["curve"]
        t = c["time_s"]
        for j in range(6):
            for col, keys in enumerate(
                (
                    ("command_q", "measured_q"),
                    ("command_difference_velocity", "measured_difference_velocity"),
                )
            ):
                for key, style in zip(keys, ("-", "--")):
                    axes[j, col].plot(
                        t,
                        np.asarray(c[key])[:, j],
                        style,
                        color=colors[label],
                        lw=1.5,
                        label=label + (" command" if style == "-" else " measured"),
                    )
    for j in range(6):
        for col in range(2):
            axes[j, col].axvline(0, color="#555", lw=0.8)
            axes[j, col].grid(alpha=0.2)
            axes[j, col].set_ylabel(
                before["joint_names"][j] + (" [rad]" if col == 0 else " [rad/s]")
            )
    axes[0, 0].set_title("Joint position: command and measured")
    axes[0, 1].set_title("Velocity: control-grid finite differences")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Seconds relative to first Place command")
    fig.suptitle(
        "Same proposed layout, ordinal "
        + str(before["ordinal"])
        + "; no smoothing of recorded signals",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output / "joint-boundaries.png", dpi=160)
    fig.savefig(output / "joint-boundaries.pdf")
    plt.close(fig)


def _collect(root: Path, baseline: Path, revision: Path) -> None:
    if root.exists():
        raise FileExistsError("Collection requires a fresh output directory")
    root.mkdir(parents=True)
    cpus = sorted(os.sched_getaffinity(0))[:8]
    environment = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        SDL_VIDEODRIVER="offscreen",
    )
    runs = []
    for group in ("A", "D"):
        for repeat, order in enumerate((("before", "after"), ("after", "before"))):
            for label in order:
                workspace = baseline if label == "before" else revision
                case = f"{group}-{repeat}-{label}"
                command = [
                    "taskset",
                    "-c",
                    ",".join(map(str, cpus)),
                    sys.executable,
                    "-m",
                    "scripts.benchmark.expansion.task_pose_coverage",
                    "--worker",
                    "--group",
                    group,
                    "--scene",
                    "open",
                    "--rows",
                    "4",
                    "--rollouts",
                    "16",
                    "--candidates",
                    "512",
                    "--seconds",
                    "600",
                    "--output",
                    str(root / case),
                ]
                start = time.perf_counter()
                with (root / f"{case}.log").open("w") as log:
                    process = subprocess.run(
                        command,
                        cwd=workspace,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                metrics = json.loads((root / case / "metrics.json").read_text())
                files = (
                    "embodichain/lab/sim/atomic_skills/execution.py",
                    "embodichain/lab/sim/atomic_skills/tracking.py",
                    "embodichain/lab/expansion/task_program_jobs.py",
                    "embodichain/lab/task_program/integrations/configured.py",
                )
                runs.append(
                    {
                        "case": case,
                        "group": group,
                        "repeat": repeat,
                        "revision": label,
                        "returncode": process.returncode,
                        "command": command,
                        "affinity": cpus,
                        "cold_process_s": time.perf_counter() - start,
                        "source": {
                            f: hashlib.sha256((workspace / f).read_bytes()).hexdigest()
                            for f in files
                        },
                        "metrics": {
                            k: metrics[k]
                            for k in (
                                "wall_s",
                                "committed",
                                "executed",
                                "physical_accepted",
                                "error",
                                "peak_rss_mib",
                            )
                        },
                    }
                )
                (root / "runs.json").write_text(json.dumps(runs, indent=2))
                print(case, runs[-1]["metrics"], flush=True)
                if process.returncode or metrics["error"]:
                    raise RuntimeError(f"Benchmark worker failed: {case}")


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument(
        "--revision", type=Path, default=Path(__file__).resolve().parents[3]
    )
    args = parser.parse_args()
    root = args.input.resolve()
    if args.collect:
        if args.baseline is None:
            parser.error("--collect requires --baseline")
        _collect(root, args.baseline.resolve(), args.revision.resolve())
    runs = json.loads((root / "runs.json").read_text())
    rows = []
    by_case = {}
    for run in runs:
        data = _episodes(root / run["case"])
        if len({row["episode"] for row in data}) != run["metrics"]["committed"]:
            raise ValueError(
                f"Boundary records do not match confirmed episode count: {run['case']}"
            )
        by_case[run["case"]] = data
        for row in data:
            rows.append(
                {
                    **row,
                    "case": run["case"],
                    "group": run["group"],
                    "revision": run["revision"],
                }
            )
    summaries = []
    for group in ("A", "D"):
        for revision in ("before", "after"):
            selected = [
                r for r in rows if r["group"] == group and r["revision"] == revision
            ]
            matching = [
                r for r in runs if r["group"] == group and r["revision"] == revision
            ]
            if not matching:
                continue
            summary = {
                "group": group,
                "revision": revision,
                "confirmed": sum(r["metrics"]["committed"] for r in matching),
                "executed": sum(r["metrics"]["executed"] for r in matching),
                "wall_s_runs": [r["metrics"]["wall_s"] for r in matching],
                "boundary_count": len(selected),
            }
            for metric in (
                "command_jump_rad",
                "command_difference_speed_rad_s",
                "command_difference_acceleration_rad_s2",
                "measured_control_speed_rad_s",
                "duration_s",
                "steps",
            ):
                summary[metric + "_p50_p95_max"] = (
                    np.quantile([r[metric] for r in selected], [0.5, 0.95, 1]).tolist()
                    if selected
                    else None
                )
            summaries.append(summary)
    pairs = []
    b = by_case.get("A-0-before", [])
    a = {r["ordinal"]: r for r in by_case.get("A-0-after", [])}
    for row in b:
        other = a.get(row["ordinal"])
        if other is None:
            continue
        error = max(
            np.max(
                np.abs(np.asarray(value) - np.asarray(other["initial_objects"][uid]))
            )
            for uid, value in row["initial_objects"].items()
        )
        if error <= 1e-4:
            pairs.append((row, other, float(error)))
    plot = None
    if pairs:
        before, after, error = max(pairs, key=lambda pair: pair[0]["command_jump_rad"])
        _plot(before, after, root)
        plot = {
            "selection": "largest before command jump among successful same-ordinal pairs with initial object poses within 1e-4",
            "ordinal": before["ordinal"],
            "max_initial_pose_component_error": error,
            "before": before["artifact"],
            "after": after["artifact"],
        }
    report = {
        "schema_version": 1,
        "summary": summaries,
        "plot_pair": plot,
        "rows": rows,
        "limitations": [
            "Accepted-episode boundary metrics exclude failed attempts; success denominator includes all executed attempts.",
            "Velocity and acceleration here are control-grid finite differences, not native physics readouts.",
            "Two orders, one seed, same host: latency is descriptive, not a general speedup claim.",
        ],
    }
    (root / "boundary-report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False)
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    _main()
