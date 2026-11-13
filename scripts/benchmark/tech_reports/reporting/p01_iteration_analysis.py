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

"""Analyze the P-01 iteration study and test an unfitted one-step model."""

from __future__ import annotations

import os

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from scripts.benchmark.tech_reports.reporting.p01_analysis import measure

ROOT = Path(
    os.environ.get("SIM_REPORT_ROOT", Path(__file__).resolve().parents[4])
).resolve()
OUT = ROOT / "reports/P-01/investigation"


def recurrence(
    initial: np.ndarray, steps: int, h: float, k: float, c: float
) -> np.ndarray:
    """Predict a trajectory from its initial state, without using later observations."""
    y = np.empty((steps + 1, 2))
    y[0] = initial
    for i in range(steps):
        q, v = y[i]
        a = -k * np.sin(q)
        y[i + 1] = q + h * v + c * h * h * a, v + h * a
    return y


def main() -> None:
    OUT.mkdir(exist_ok=True)
    cfg = json.loads((ROOT / "configs/P-01/isaaclab.json").read_text())
    plan = json.loads((ROOT / "configs/P-01/iteration-study.json").read_text())
    runs = [
        (ROOT / "results/P-01" / f"20261014_study_{r['id']}", r["group"])
        for r in plan["runs"]
    ]
    runs += [
        (ROOT / f"results/P-01/20261014_main_pendulum_dt{tag}", "matrix")
        for tag in ["010", "005", "0025"]
    ]
    runs += [(ROOT / "results/P-01/20261014_study_control_tgs_dt005_p4", "control")]
    m, g, l, r = (
        cfg[key]
        for key in ["mass_kg", "gravity_m_s2", "pendulum_length_m", "bob_radius_m"]
    )
    inertia = m * (l * l + 0.4 * r * r)
    kappa = m * g * l / inertia
    rows, manifests, summaries, traces, cache = [], {}, [], {}, {}
    for run, group in runs:
        meta = json.loads((run / "run.json").read_text())
        assert meta["status"] == "completed", run.name
        n, h = meta["position_iterations"], meta["dt_s"]
        solver = meta["solver"].split()[-1]
        every = meta["effective_config"]["physics"][
            "enable_external_forces_every_iteration"
        ]
        c = (n + 1) / (2 * n) if solver == "TGS" and every else 1.0
        manifest = {
            "run_json_sha256": hashlib.sha256(
                (run / "run.json").read_bytes()
            ).hexdigest(),
            "episodes": {},
        }
        group_rows = []
        for i in range(meta["completed_episodes"]):
            file = run / f"episode_{i:02d}.npz"
            a = np.load(file)
            t, state = a["t"], a["state"]
            key = (float(state[0, 0]), float(t[-1]), h)
            if key not in cache:
                ref = solve_ivp(
                    lambda _, y: [y[1], -kappa * np.sin(y[0])],
                    [0, t[-1]],
                    state[0],
                    t_eval=t,
                    method="DOP853",
                    rtol=1e-12,
                    atol=1e-14,
                )
                assert ref.success
                cache[key] = ref.y.T
            expected = cache[key]
            metrics = measure("pendulum", t, state, expected, cfg)
            q, v = state.T
            acc = -kappa * np.sin(q[:-1])
            dq, dv = np.diff(q), np.diff(v)
            e = inertia * v * v / 2 + m * g * l * (1 - np.cos(q))
            prediction = recurrence(state[0], len(t) - 1, h, kappa, c)
            model_energy = inertia * prediction[:, 1] ** 2 / 2 + m * g * l * (
                1 - np.cos(prediction[:, 0])
            )
            c_fit = np.dot(dq - h * v[:-1], h * h * acc) / np.dot(
                h * h * acc, h * h * acc
            )
            geometric = np.column_stack(
                (-l * np.sin(q), np.zeros(len(t)), 1 - l * np.cos(q))
            )
            row = dict(
                run_id=run.name,
                group=group,
                initial_id=i,
                initial_deg=float(np.rad2deg(q[0])),
                duration_s=float(t[-1]),
                dt_s=h,
                position_iterations=n,
                velocity_iterations=meta["velocity_iterations"],
                solver=solver,
                device=meta["device"],
                external_each_iteration=every,
                c_theory=c,
                c_observed=float(c_fit),
                **metrics,
                step_angle_residual_rad=float(
                    np.max(np.abs(dq - h * v[:-1] - c * h * h * acc))
                ),
                step_velocity_residual_rad_s=float(np.max(np.abs(dv - h * acc))),
                model_angle_max_difference_deg=float(
                    np.rad2deg(np.max(np.abs(prediction[:, 0] - q)))
                ),
                model_energy_drift_error_pp=float(
                    100
                    * abs(
                        np.max(np.abs(model_energy / model_energy[0] - 1))
                        - metrics["energy_max_relative_drift"]
                    )
                ),
                log_energy_growth_s_inv=float(np.polyfit(t, np.log(e / e[0]), 1)[0]),
                linearized_growth_prediction_s_inv=float(
                    np.log1p((1 - c) * kappa * h * h) / h
                ),
                com_kinematic_max_error_m=float(
                    np.max(np.linalg.norm(a["com_position"] - geometric, axis=1))
                ),
            )
            assert row["com_kinematic_max_error_m"] < 2e-5
            if "body_com_velocity" in a:
                bv = a["body_com_velocity"]
                body_e = (
                    0.5 * m * np.sum(bv[:, :3] ** 2, axis=1)
                    + 0.5 * (0.4 * m * r * r) * np.sum(bv[:, 3:] ** 2, axis=1)
                    + m * g * (a["com_position"][:, 2] - (1 - l))
                )
                row["body_energy_max_relative_drift"] = float(
                    np.max(np.abs(body_e / body_e[0] - 1))
                )
                row["body_vs_joint_energy_max_difference_j"] = float(
                    np.max(np.abs(body_e - e))
                )
            rows.append(row)
            group_rows.append(row)
            manifest["episodes"][file.name] = hashlib.sha256(
                file.read_bytes()
            ).hexdigest()
            if i == 0:
                traces[run.name] = (t, state, expected, prediction, e)
                np.savez_compressed(
                    OUT / f"{run.name}_prediction.npz",
                    t=t,
                    predicted_state=prediction,
                    reference=expected,
                )
        numeric = [
            "energy_max_relative_drift",
            "period_relative_error",
            "position_rmse",
            "c_observed",
            "step_angle_residual_rad",
            "step_velocity_residual_rad_s",
            "model_angle_max_difference_deg",
            "model_energy_drift_error_pp",
            "log_energy_growth_s_inv",
            "linearized_growth_prediction_s_inv",
            "com_kinematic_max_error_m",
            "body_energy_max_relative_drift",
            "body_vs_joint_energy_max_difference_j",
        ]
        summary = {
            key: group_rows[0][key]
            for key in [
                "run_id",
                "group",
                "dt_s",
                "position_iterations",
                "velocity_iterations",
                "solver",
                "device",
                "external_each_iteration",
                "duration_s",
                "initial_deg",
            ]
        }
        summary.update(
            trials=len(group_rows),
            pass_count=sum(x["candidate_pass"] for x in group_rows),
        )
        for key in numeric:
            if key in group_rows[0]:
                vals = [r[key] for r in group_rows]
                summary[key] = {
                    "mean": float(np.mean(vals)),
                    "min": float(min(vals)),
                    "max": float(max(vals)),
                }
        summaries.append(summary)
        manifests[run.name] = manifest
    with (OUT / "metrics.csv").open("w") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(dict.fromkeys(k for row in rows for k in row))
        )
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    (OUT / "analysis.json").write_text(
        json.dumps(
            dict(
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                plan_sha256=hashlib.sha256(
                    (ROOT / "configs/P-01/iteration-study.json").read_bytes()
                ).hexdigest(),
                input_runs=manifests,
                trajectories=len(rows),
                reference="DOP853 rtol=1e-12 atol=1e-14; actual readback initial state",
                model="No fitted coefficients: c=(N+1)/(2N) for TGS with per-iteration external forces, else c=1",
                small_angle_k_s2=kappa,
            ),
            indent=2,
        )
        + "\n"
    )

    # The adaptive check evaluates one shared setting against both original P-01 gates.
    followup = []
    followup_manifest = {}
    followup_plan = ROOT / "configs/P-01/iteration-followup.json"
    for item in json.loads(followup_plan.read_text())["runs"]:
        run = ROOT / "results/P-01" / f"20261014_study_{item['id']}"
        meta = json.loads((run / "run.json").read_text())
        assert meta["status"] == "completed", run.name
        measured = []
        hashes = {
            "run.json": hashlib.sha256((run / "run.json").read_bytes()).hexdigest()
        }
        for i in range(meta["completed_episodes"]):
            path = run / f"episode_{i:02d}.npz"
            data = np.load(path)
            t, state = data["t"], data["state"]
            if item["case"] == "freefall":
                ref = np.column_stack((state[0, 0] - g * t * t / 2, -g * t))
            else:
                solve = solve_ivp(
                    lambda _, y: [y[1], -kappa * np.sin(y[0])],
                    [0, t[-1]],
                    state[0],
                    t_eval=t,
                    method="DOP853",
                    rtol=1e-12,
                    atol=1e-14,
                )
                assert solve.success
                ref = solve.y.T
            result = measure(item["case"], t, state, ref, cfg)
            result.update(initial_id=i, run_id=run.name)
            measured.append(result)
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        keys = (
            ["position_max_0_1s_m", "velocity_max_0_1s_m_s"]
            if item["case"] == "freefall"
            else ["energy_max_relative_drift", "period_relative_error", "position_rmse"]
        )
        group = dict(
            item,
            run_id=run.name,
            trials=len(measured),
            pass_count=sum(x["candidate_pass"] for x in measured),
            episodes=measured,
        )
        for key in keys:
            values = [x[key] for x in measured]
            group[key] = {
                "mean": float(np.mean(values)),
                "min": float(min(values)),
                "max": float(max(values)),
            }
        followup.append(group)
        followup_manifest[run.name] = hashes
    (OUT / "followup.json").write_text(json.dumps(followup, indent=2) + "\n")
    (OUT / "followup-analysis.json").write_text(
        json.dumps(
            dict(
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                plan_sha256=hashlib.sha256(followup_plan.read_bytes()).hexdigest(),
                input_runs=followup_manifest,
            ),
            indent=2,
        )
        + "\n"
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font_manager.fontManager.addfont(str(ROOT / "assets/fonts/NotoSansSC-Regular.ttf"))
    plt.rcParams.update(
        {
            "font.family": "Noto Sans SC",
            "font.size": 11,
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 180,
        }
    )
    colors = ["#168575", "#3478A4", "#BA543B", "#8B64A4", "#B48734"]
    matrix = [s for s in summaries if s["group"] == "matrix"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    for n, color in zip([1, 2, 4, 8, 16], colors):
        selected = sorted(
            [s for s in matrix if s["position_iterations"] == n],
            key=lambda s: s["dt_s"],
        )
        x = np.array([s["dt_s"] * 1000 for s in selected])
        y = np.array([s["energy_max_relative_drift"]["mean"] * 100 for s in selected])
        low = np.array([s["energy_max_relative_drift"]["min"] * 100 for s in selected])
        high = np.array([s["energy_max_relative_drift"]["max"] * 100 for s in selected])
        axes[0].errorbar(
            x,
            y,
            yerr=[y - low, high - y],
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
    axes[0].axhline(2, color="#525A65", ls=":", label="2% 判据")
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
    for ext in ["png", "svg"]:
        fig.savefig(OUT / f"iteration_sweep.{ext}")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    for n, color in zip([1, 4, 16], [colors[0], colors[2], colors[4]]):
        s = next(
            s for s in matrix if s["position_iterations"] == n and s["dt_s"] == 0.005
        )
        t, y, ref, pred, e = traces[s["run_id"]]
        pe = inertia * pred[:, 1] ** 2 / 2 + m * g * l * (1 - np.cos(pred[:, 0]))
        axes[0].plot(t, 100 * (e / e[0] - 1), color=color, lw=2, label=f"{n} 次：实测")
        axes[0].plot(
            t,
            100 * (pe / pe[0] - 1),
            color="#243548",
            ls="--",
            lw=0.85,
            label="离散模型" if n == 16 else None,
        )
    selected = sorted(
        [s for s in matrix if s["dt_s"] == 0.005],
        key=lambda s: s["position_iterations"],
    )
    x = np.array([s["position_iterations"] for s in selected])
    axes[1].plot(x, (x + 1) / (2 * x), "--", color="#243548", label="理论：(N+1)/(2N)")
    axes[1].scatter(
        x,
        [s["c_observed"]["mean"] for s in selected],
        s=40,
        color=colors[1],
        label="实测一步更新系数",
    )
    axes[0].set(
        xlabel="物理时间（s）",
        ylabel="能量变化 / 初始能量（%）",
        title="A  从初态独立预测，未拟合轨迹",
    )
    axes[1].set(
        xlabel="位置迭代次数 N",
        ylabel="摆角更新系数 c",
        title="B  一步更新规律",
        xticks=x,
    )
    for ax in axes:
        ax.grid(alpha=0.18)
        ax.legend(fontsize=9)
    for ext in ["png", "svg"]:
        fig.savefig(OUT / f"mechanism_validation.{ext}")
    plt.close(fig)

    picked = [
        "20261014_main_pendulum_dt005",
        "20261014_study_tgs_dt005_p1_v1",
        "20261014_study_tgs_dt00125_p1",
        "20261014_study_tgs_dt005_p4_force_once",
        "20261014_study_pgs_dt005_p4",
        "20261014_study_cpu_tgs_dt005_p4",
    ]
    labels = [
        "TGS · 5 ms · 4 次",
        "TGS · 5 ms · 1 次",
        "TGS · 1.25 ms · 1 次",
        "TGS · 5 ms · 4 次\n关闭逐迭代施力",
        "PGS · 5 ms · 4 次",
        "CPU TGS · 5 ms · 4 次",
    ]
    groups = [next(s for s in summaries if s["run_id"] == rid) for rid in picked]
    fig, ax = plt.subplots(figsize=(10.8, 4.5), constrained_layout=True)
    vals = [s["energy_max_relative_drift"]["mean"] * 100 for s in groups]
    bars = ax.barh(
        np.arange(len(vals)),
        vals,
        color=[colors[2], colors[0], colors[0], colors[1], colors[1], colors[2]],
        height=0.58,
    )
    for i, (v, s) in enumerate(zip(vals, groups)):
        ax.text(v + 0.8, i, f"{v:.3f}%  (n={s['trials']})", va="center", fontsize=10)
    ax.axvline(2, color="#525A65", ls=":", label="2% 判据")
    ax.set(
        yticks=np.arange(len(vals)),
        yticklabels=labels,
        xlabel="10 s 最大相对能量漂移（%）",
        xlim=(0, max(vals) * 1.33),
    )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.18)
    ax.legend(loc="lower right", fontsize=9)
    for ext in ["png", "svg"]:
        fig.savefig(OUT / f"configuration_controls.{ext}")
    plt.close(fig)
    for s in summaries:
        print(
            s["run_id"],
            f'n={s["trials"]}, E={100*s["energy_max_relative_drift"]["mean"]:.5f}%, pass={s["pass_count"]}, c={s["c_observed"]["mean"]:.6f}, model_dq_deg={s["model_angle_max_difference_deg"]["max"]:.6g}',
        )


if __name__ == "__main__":
    main()
