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

"""Recompute the combined P-01 comparison from the 240 preserved trajectories."""

from __future__ import annotations

import os

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.benchmark.tech_reports.reporting.p01_analysis import measure

ROOT = Path(
    os.environ.get("SIM_REPORT_ROOT", Path(__file__).resolve().parents[4])
).resolve()
OUT = ROOT / "reports/P-01/comparison"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    cfg_path = ROOT / "configs/P-01/embodichain-dexuni.json"
    cfg = json.loads(cfg_path.read_text())
    inputs_path = ROOT / cfg["initial_states"]
    source = ROOT / "reports/P-01/dexuni/metrics.csv"
    ref_path = ROOT / "reports/P-01/dexuni/reference.npz"
    ref = np.load(ref_path)
    assert np.max(np.abs(ref["state"] - ref["refined_state"])) < 1e-8
    original = list(csv.DictReader(source.open()))
    rows, hashes = [], {}
    for old in original:
        path = ROOT / old["source"]
        data = np.load(path)
        t, state = data["t"], data["state"]
        dt, case, initial = float(old["dt_s"]), old["case"], int(old["initial_id"])
        reference = (
            np.column_stack(
                (
                    state[0, 0] - 0.5 * cfg["gravity_m_s2"] * t**2,
                    -cfg["gravity_m_s2"] * t,
                )
            )
            if case == "freefall"
            else ref["state"][:: round(dt / 0.0005), initial]
        )
        metrics = measure(case, t, state, reference, cfg)
        for key, value in metrics.items():
            if isinstance(value, bool):
                assert old[key] == str(value)
            elif value is not None:
                np.testing.assert_allclose(
                    value, float(old[key]), rtol=1e-11, atol=1e-13
                )
        rows.append(
            dict(
                run_id=old["run_id"],
                source=old["source"],
                platform=old["platform"],
                solver=old["solver"],
                case=case,
                phase=old["phase"],
                dt_s=dt,
                initial_id=initial,
                **metrics,
            )
        )
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        run_meta = path.with_name("run.json")
        meta = json.loads(run_meta.read_text())
        assert (
            meta["hashes"][cfg["initial_states"]]
            == hashlib.sha256(inputs_path.read_bytes()).hexdigest()
        )
        assert meta["status"] == "completed"
        hashes[str(run_meta.relative_to(ROOT))] = hashlib.sha256(
            run_meta.read_bytes()
        ).hexdigest()
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / "metrics.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for run_id in dict.fromkeys(r["run_id"] for r in rows):
        subset = [r for r in rows if r["run_id"] == run_id]
        assert len(subset) == 10
        s = {
            k: subset[0][k]
            for k in ["run_id", "platform", "solver", "case", "phase", "dt_s"]
        }
        s.update(trials=10, pass_count=sum(r["candidate_pass"] for r in subset))
        for key in subset[0]:
            if key in s or key in {
                "initial_id",
                "candidate_pass",
                "first_pointwise_threshold_failure_s",
            }:
                continue
            if isinstance(subset[0][key], (float, int)):
                values = [r[key] for r in subset]
                s[key] = dict(
                    mean=float(np.mean(values)), min=min(values), max=max(values)
                )
        summary.append(s)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    labels = {
        "DexUni": ("EmbodiChain DexUni", "#19779f", "o"),
        "PGS 4/1": ("Isaac Lab PGS 4/1", "#d17a22", "s"),
        "TGS 4/1": ("Isaac Lab TGS 4/1", "#97508a", "D"),
        "TGS 1/1": ("Isaac Lab TGS 1/1", "#333333", "^"),
    }
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.9))
    for ax, case, key, scale, title, unit, threshold in zip(
        axes,
        ["freefall", "pendulum", "freefall"],
        ["position_max_0_1s_m", "energy_max_relative_drift", "position_max_error"],
        [1000, 100, 1],
        [
            "A  Freefall, 0–1 s",
            "B  Pendulum, 0–10 s",
            "C  Freefall, 0–10 s stress test",
        ],
        [
            "Max. position error (mm)",
            "Max. energy change (%)",
            "Max. position error (m)",
        ],
        [5, 2, None],
    ):
        for label, (display, color, marker) in labels.items():
            entries = sorted(
                [s for s in summary if s["case"] == case and s["solver"] == label],
                key=lambda s: s["dt_s"],
            )
            x = np.array([s["dt_s"] * 1000 for s in entries])
            y = np.array([s[key]["mean"] * scale for s in entries])
            low = np.array([s[key]["min"] * scale for s in entries])
            high = np.array([s[key]["max"] * scale for s in entries])
            ax.errorbar(
                x,
                y,
                yerr=[y - low, high - y],
                fmt=marker + ("-" if len(entries) > 1 else ""),
                color=color,
                label=display,
                capsize=3,
                linewidth=1.6,
            )
        if threshold is not None:
            ax.axhline(threshold, color="#777777", ls="--", lw=1)
        ax.set(
            xscale="log",
            yscale="log",
            xlabel="Physics step (ms)",
            ylabel=unit,
            title=title,
        )
        ax.set_xticks([1, 2.5, 5, 10], labels=["1", "2.5", "5", "10"])
        ax.grid(alpha=0.18, which="both")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "accuracy-comparison.png", dpi=180)
    plt.close(fig)
    table = [
        "| 步长 | 配置 | 落体位置误差 / mm | 单摆能量漂移 / % | 落体通过 | 单摆通过 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for dt, solvers in [
        (0.005, ["DexUni", "PGS 4/1", "TGS 4/1"]),
        (0.001, ["DexUni", "PGS 4/1", "TGS 1/1"]),
    ]:
        for solver in solvers:
            f = next(
                s
                for s in summary
                if (s["solver"], s["dt_s"], s["case"]) == (solver, dt, "freefall")
            )
            p = next(
                s
                for s in summary
                if (s["solver"], s["dt_s"], s["case"]) == (solver, dt, "pendulum")
            )
            table.append(
                f'| {dt*1000:g} ms | {labels[solver][0]} | {f["position_max_0_1s_m"]["mean"]*1000:.3f} | {p["energy_max_relative_drift"]["mean"]*100:.3f} | {f["pass_count"]}/10 | {p["pass_count"]}/10 |'
            )
    (OUT / "key-table.md").write_text("\n".join(table) + "\n")
    for path in [
        Path(__file__),
        ROOT / "code/analysis/p01_analysis.py",
        cfg_path,
        inputs_path,
        source,
        ref_path,
    ]:
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    (OUT / "analysis.json").write_text(
        json.dumps(
            dict(
                trajectory_count=len(rows),
                configuration_count=len(summary),
                metric_definition="Per-trajectory maximum within the stated physical window, then mean and min/max across 10 distinct initial states; not repeatability confidence intervals.",
                reference="Analytic fall and independently validated RK4, never the other simulator.",
                all_metrics_match_previous_analysis=True,
                sources_sha256=hashes,
            ),
            indent=2,
        )
        + "\n"
    )
    print(
        "Recomputed",
        len(rows),
        "trajectories;",
        len(summary),
        "configurations; all metrics match preserved analysis.",
    )
    print("\n".join(table))


if __name__ == "__main__":
    main()
