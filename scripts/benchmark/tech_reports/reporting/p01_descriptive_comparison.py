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

"""Rebuild P-01 continuous comparisons without changing historical analyses."""

from __future__ import annotations

import os

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from scripts.benchmark.tech_reports.reporting.p01_analysis import measure

ROOT = Path(
    os.environ.get("SIM_REPORT_ROOT", Path(__file__).resolve().parents[4])
).resolve()
OUT = ROOT / "reports/P-01/comparison-descriptive"
LABELS = {
    "DexUni": ("EmbodiChain DexUni", "#19779f", "o"),
    "PGS 4/1": ("Isaac Lab PGS 4/1", "#d17a22", "s"),
    "TGS 4/1": ("Isaac Lab TGS 4/1", "#97508a", "D"),
    "TGS 1/1": ("Isaac Lab TGS 1/1", "#333333", "^"),
}


def save_figure(fig: plt.Figure, name: str) -> None:
    """Keep raster previews and editable vector exports together."""
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)


def plot_ranges(
    ax: plt.Axes, groups: list[dict], key: str, scale: float, solver: str
) -> None:
    """Plot means and initial-state ranges, with no acceptance line."""
    entries = sorted(groups, key=lambda s: s["dt_s"])
    x = np.array([s["dt_s"] * 1000 for s in entries])
    y = np.array([s[key]["mean"] * scale for s in entries])
    low = np.array([s[key]["min"] * scale for s in entries])
    high = np.array([s[key]["max"] * scale for s in entries])
    label, color, marker = LABELS[solver]
    ax.errorbar(
        x,
        y,
        yerr=np.maximum([y - low, high - y], 0),
        fmt=marker + ("-" if len(entries) > 1 else ""),
        color=color,
        label=label,
        capsize=3,
        linewidth=1.6,
        markerfacecolor="none" if solver == "TGS 1/1" else color,
        markersize=8 if solver == "TGS 1/1" else 5,
    )


def main() -> None:
    OUT.mkdir(exist_ok=True)
    cfg_path = ROOT / "configs/P-01/embodichain-dexuni.json"
    cfg = json.loads(cfg_path.read_text())
    inputs_path = ROOT / cfg["initial_states"]
    source = ROOT / "reports/P-01/dexuni/metrics.csv"
    ref_path = ROOT / "reports/P-01/dexuni/reference.npz"
    ref = np.load(ref_path)
    original = list(csv.DictReader(source.open()))
    rows, hashes, traces, plot_arrays, plot_sources = [], {}, {}, {}, {}
    legacy_fields = {"candidate_pass", "first_pointwise_threshold_failure_s"}
    for old in original:
        path = ROOT / old["source"]
        data = np.load(path)
        t, state = data["t"], data["state"]
        dt, case, initial = float(old["dt_s"]), old["case"], int(old["initial_id"])
        assert state.shape == (round(10 / dt) + 1, 2)
        assert t[0] == 0 and t[-1] == 10 and np.isfinite(state).all()
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
        # Reuse frozen metric definitions; historical gates are not new metrics.
        metrics = {
            k: v
            for k, v in measure(case, t, state, reference, cfg).items()
            if k not in legacy_fields
        }
        for key, value in metrics.items():
            if value is None:
                assert old[key] == ""
            else:
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
        run_meta = path.with_name("run.json")
        meta = json.loads(run_meta.read_text())
        assert meta["status"] == "completed"
        assert (
            meta["hashes"][cfg["initial_states"]]
            == hashlib.sha256(inputs_path.read_bytes()).hexdigest()
        )
        for p in (path, run_meta):
            hashes[str(p.relative_to(ROOT))] = hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
        if initial == 0:
            traces[old["solver"], case, dt] = (t, state, reference)
            plot_sources[old["run_id"]] = old["source"]
            for key, values in (("t", t), ("state", state), ("reference", reference)):
                plot_arrays[f"{old['run_id']}__{key}"] = values

    assert len(rows) == 240
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / "metrics.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for run_id in dict.fromkeys(r["run_id"] for r in rows):
        subset = [r for r in rows if r["run_id"] == run_id]
        assert {r["initial_id"] for r in subset} == set(range(10))
        s = {
            k: subset[0][k]
            for k in ("run_id", "platform", "solver", "case", "phase", "dt_s")
        }
        s["trials"] = len(subset)
        for key in subset[0]:
            if key in s or key == "initial_id":
                continue
            if isinstance(subset[0][key], (float, int)):
                values = [r[key] for r in subset]
                s[key] = dict(
                    mean=float(np.mean(values)),
                    median=float(np.median(values)),
                    min=min(values),
                    max=max(values),
                )
        summary.append(s)
    assert len(summary) == 24
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(OUT / "plot-traces.npz", **plot_arrays)

    table = [
        "| 步长 | 配置 | 落体位置误差均值 [范围] / mm | 单摆能量漂移均值 / % |",
        "| --- | --- | ---: | ---: |",
    ]
    for dt, solvers in (
        (0.005, ("DexUni", "PGS 4/1", "TGS 4/1")),
        (0.001, ("DexUni", "PGS 4/1", "TGS 1/1")),
    ):
        for solver in solvers:
            f = next(
                s
                for s in summary
                if (s["solver"], s["dt_s"], s["case"]) == (solver, dt, "freefall")
            )["position_max_0_1s_m"]
            p = next(
                s
                for s in summary
                if (s["solver"], s["dt_s"], s["case"]) == (solver, dt, "pendulum")
            )["energy_max_relative_drift"]
            table.append(
                f"| {dt*1000:g} ms | {LABELS[solver][0]} | "
                f"{f['mean']*1000:.3f} [{f['min']*1000:.3f}, {f['max']*1000:.3f}] | "
                f"{p['mean']*100:.3f} |"
            )
    (OUT / "key-table.md").write_text("\n".join(table) + "\n")

    font_path = ROOT / "assets/fonts/NotoSansSC-Regular.ttf"
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update(
        {
            "font.family": "Noto Sans SC",
            "font.size": 10,
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 180,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
    for ax, case, key, scale, title, unit in zip(
        axes,
        ("freefall", "pendulum", "freefall"),
        ("position_max_0_1s_m", "energy_max_relative_drift", "position_max_error"),
        (1000, 100, 1),
        ("A  落体：0–1 s", "B  单摆：0–10 s", "C  落体：0–10 s 长程测试"),
        ("最大位置误差（mm）", "最大相对能量漂移（%）", "最大位置误差（m）"),
    ):
        for solver in LABELS:
            plot_ranges(
                ax,
                [s for s in summary if s["case"] == case and s["solver"] == solver],
                key,
                scale,
                solver,
            )
        ax.set(
            xscale="log",
            yscale="log",
            xlabel="物理步长（ms）",
            ylabel=unit,
            title=title,
        )
        ax.set_xticks([1, 2.5, 5, 10], labels=["1", "2.5", "5", "10"])
        ax.grid(alpha=0.18, which="both")
    axes[0].legend(fontsize=8, loc="upper left")
    save_figure(fig, "accuracy-comparison")

    dt_colors = {0.01: "#97508a", 0.005: "#d17a22", 0.0025: "#19779f"}
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    for dt, color in dt_colors.items():
        t, state, reference = traces["TGS 4/1", "freefall", dt]
        mask = t <= 1 + 1e-12
        axes[0].plot(
            t[mask],
            np.abs(state[mask, 0] - reference[mask, 0]) * 1000,
            color=color,
            label=f"{dt*1000:g} ms",
        )
    plot_ranges(
        axes[1],
        [s for s in summary if s["case"] == "freefall" and s["solver"] == "TGS 4/1"],
        "position_max_0_1s_m",
        1000,
        "TGS 4/1",
    )
    axes[0].set(
        xlabel="物理时间（s）", ylabel="位置绝对误差（mm）", title="A  TGS 4/1：初态 0"
    )
    axes[1].set(
        xlabel="物理步长（ms）",
        ylabel="前 1 s 最大位置误差（mm）",
        title="B  10 初态均值及范围",
        xticks=[2.5, 5, 10],
    )
    for ax in axes:
        ax.grid(alpha=0.18)
        ax.legend(fontsize=9)
    save_figure(fig, "freefall_accuracy")

    m, g, length, radius = (
        cfg[k] for k in ("mass_kg", "gravity_m_s2", "pendulum_length_m", "bob_radius_m")
    )
    inertia = m * (length * length + 0.4 * radius * radius)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    t, state, reference = traces["TGS 4/1", "pendulum", 0.005]
    axes[0].plot(
        t, np.rad2deg(state[:, 0]), label="TGS 4/1 · 5 ms", color=LABELS["TGS 4/1"][1]
    )
    axes[0].plot(
        t, np.rad2deg(reference[:, 0]), "--", label="独立参考", color="#333333"
    )
    for dt, color in dt_colors.items():
        t, state, _ = traces["TGS 4/1", "pendulum", dt]
        energy = inertia * state[:, 1] ** 2 / 2 + m * g * length * (
            1 - np.cos(state[:, 0])
        )
        axes[1].plot(
            t, 100 * (energy / energy[0] - 1), label=f"{dt*1000:g} ms", color=color
        )
    axes[0].set(xlabel="物理时间（s）", ylabel="摆角（°）", title="A  初态 0：摆角")
    axes[1].set(
        xlabel="物理时间（s）", ylabel="相对初始能量变化（%）", title="B  初态 0：能量"
    )
    for ax in axes:
        ax.grid(alpha=0.18)
        ax.legend(fontsize=9)
    save_figure(fig, "pendulum_accuracy")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
    for ax, case, key, scale, title, unit in (
        (
            axes[0],
            "freefall",
            "position_max_0_1s_m",
            1000,
            "A  落体：0–1 s",
            "最大位置误差（mm）",
        ),
        (
            axes[1],
            "pendulum",
            "energy_max_relative_drift",
            100,
            "B  单摆：0–10 s",
            "最大相对能量漂移（%）",
        ),
    ):
        for solver in ("PGS 4/1", "TGS 4/1"):
            plot_ranges(
                ax,
                [s for s in summary if s["case"] == case and s["solver"] == solver],
                key,
                scale,
                solver,
            )
        ax.set(
            yscale="log",
            xlabel="物理步长（ms）",
            ylabel=unit,
            title=title,
            xticks=[1, 2.5, 5, 10],
        )
    for solver in ("PGS 4/1", "TGS 4/1"):
        t, state, _ = traces[solver, "pendulum", 0.005]
        energy = inertia * state[:, 1] ** 2 / 2 + m * g * length * (
            1 - np.cos(state[:, 0])
        )
        axes[2].plot(
            t,
            (energy / energy[0] - 1) * 100,
            color=LABELS[solver][1],
            label=LABELS[solver][0],
        )
    axes[2].set(
        xlabel="物理时间（s）",
        ylabel="相对初始能量变化（%）",
        title="C  单摆：5 ms，初态 0",
    )
    for ax in axes:
        ax.grid(alpha=0.18)
        ax.legend(fontsize=8)
    save_figure(fig, "solver_comparison")

    # Mechanism studies are preserved analyses; replot their continuous results.
    study_path = ROOT / "reports/P-01/investigation/summary.json"
    study = json.loads(study_path.read_text())
    study = [{k: v for k, v in s.items() if k != "pass_count"} for s in study]
    (OUT / "iteration-plot-data.json").write_text(json.dumps(study, indent=2) + "\n")
    matrix = [s for s in study if s["group"] == "matrix"]
    colors = ["#168575", "#3478A4", "#BA543B", "#8B64A4", "#B48734"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    for n, color in zip((1, 2, 4, 8, 16), colors):
        selected = sorted(
            [s for s in matrix if s["position_iterations"] == n],
            key=lambda s: s["dt_s"],
        )
        x = np.array([s["dt_s"] * 1000 for s in selected])
        y, low, high = (
            np.array([s["energy_max_relative_drift"][k] * 100 for s in selected])
            for k in ("mean", "min", "max")
        )
        axes[0].errorbar(
            x,
            y,
            yerr=np.maximum([y - low, high - y], 0),
            fmt="o-",
            capsize=3,
            color=color,
            label=f"位置迭代 {n} 次",
        )
        axes[1].plot(
            x,
            [s["log_energy_growth_s_inv"]["mean"] for s in selected],
            "o",
            color=color,
        )
        axes[1].plot(
            x,
            [s["linearized_growth_prediction_s_inv"]["mean"] for s in selected],
            "--",
            color=color,
        )
    axes[0].set(
        xlabel="物理步长（ms）",
        ylabel="最大相对能量漂移（%）",
        yscale="log",
        title="A  10 初态均值及最小—最大范围",
    )
    axes[1].set(
        xlabel="物理步长（ms）",
        ylabel="能量增长率（1/s）",
        title="B  圆点：实测；虚线：小角度预测",
    )
    for ax in axes:
        ax.set_xticks([2.5, 5, 10])
        ax.grid(alpha=0.18)
    axes[0].legend(fontsize=9, ncol=2)
    save_figure(fig, "iteration_sweep")

    picked = [
        ("20261014_main_pendulum_dt005", "TGS · 5 ms · 4 次"),
        ("20261014_study_tgs_dt005_p1_v1", "TGS · 5 ms · 1 次"),
        ("20261014_study_tgs_dt00125_p1", "TGS · 1.25 ms · 1 次"),
        ("20261014_study_tgs_dt005_p4_force_once", "TGS · 5 ms · 4 次\n关闭逐迭代施力"),
        ("20261014_study_pgs_dt005_p4", "PGS · 5 ms · 4 次"),
        ("20261014_study_cpu_tgs_dt005_p4", "CPU TGS · 5 ms · 4 次"),
    ]
    groups = [next(s for s in study if s["run_id"] == rid) for rid, _ in picked]
    vals = [s["energy_max_relative_drift"]["mean"] * 100 for s in groups]
    fig, ax = plt.subplots(figsize=(10.8, 4.5), constrained_layout=True)
    ax.barh(np.arange(len(vals)), vals, color="#3478A4", height=0.58)
    for i, (v, s) in enumerate(zip(vals, groups)):
        ax.text(v + 0.8, i, f"{v:.3f}%  (n={s['trials']})", va="center", fontsize=10)
    ax.set(
        yticks=np.arange(len(vals)),
        yticklabels=[label for _, label in picked],
        xlabel="10 s 最大相对能量漂移（%）",
        xlim=(0, max(vals) * 1.33),
    )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.18)
    save_figure(fig, "configuration_controls")

    for path in (
        Path(__file__),
        ROOT / "code/analysis/p01_analysis.py",
        cfg_path,
        inputs_path,
        source,
        ref_path,
        font_path,
        study_path,
        ROOT / "reports/P-01/investigation/analysis.json",
        ROOT / "code/analysis/p01_iteration_analysis.py",
    ):
        key = (
            str(path.relative_to(ROOT))
            if path.is_relative_to(ROOT)
            else "source:" + str(path.relative_to(Path(__file__).resolve().parents[4]))
        )
        hashes[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT / "analysis.json").write_text(
        json.dumps(
            dict(
                evaluation_revision="2026-10-22-descriptive",
                trajectory_count=len(rows),
                configuration_count=len(summary),
                metric_definition="Per-trajectory metrics in unchanged physical windows, then mean/median/min/max across 10 distinct initial states; not repeatability confidence intervals.",
                change="Remove categorical gates from current presentation; preserve raw states, windows, reference and metric definitions. No new simulations or solver changes.",
                historical_labels="reports/P-01/comparison/ and reports/P-01/dexuni/ retained unchanged.",
                all_continuous_metrics_match_previous_analysis=True,
                iteration_figures="Replotted preserved investigation/summary.json; original analysis.json identifies every raw input and hash.",
                sources_sha256=hashes,
            ),
            indent=2,
        )
        + "\n"
    )
    (OUT / "plotting.json").write_text(
        json.dumps(
            dict(
                statistic="Mean with min/max across distinct initial states; no acceptance lines or filtering.",
                exports=["png", "svg", "pdf"],
                labels=LABELS,
                font_source=str(font_path.relative_to(ROOT)),
                trajectory_array_sources=plot_sources,
                trajectory_array_keys="<run_id>__t, __state, __reference; initial state 0, no smoothing or resampling",
                figures={
                    "accuracy-comparison": "summary.json; all 24 groups; fall 0–1 s position in mm, pendulum 0–10 s energy in %, fall 0–10 s position in m; log axes",
                    "freefall_accuracy": "plot-traces.npz and summary.json; TGS 4/1, 10/5/2.5 ms; absolute position error in mm, 0–1 s",
                    "pendulum_accuracy": "plot-traces.npz; TGS 4/1, initial 0; 5 ms angle vs reference in degrees; 10/5/2.5 ms signed energy change in %, 0–10 s",
                    "solver_comparison": "summary.json and plot-traces.npz; PGS/TGS 4/1; 1 ms PGS only; 5 ms initial 0 energy curves",
                    "iteration_sweep": "iteration-plot-data.json; all group=matrix; energy % and growth 1/s",
                    "configuration_controls": [s["run_id"] for s in groups],
                },
            ),
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    print(
        f"Recomputed {len(rows)} trajectories / {len(summary)} configurations; continuous metrics unchanged."
    )
    print("\n".join(table))


if __name__ == "__main__":
    main()
