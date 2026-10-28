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

"""Independent P-01 references, metrics and reproducible figures (no simulator)."""

from __future__ import annotations

import os

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import ellipk

PROJECT = Path(
    os.environ.get("SIM_REPORT_ROOT", Path(__file__).resolve().parents[4])
).resolve()


def reference(
    theta0: np.ndarray, duration: float, h: float, output_dt: float, coefficient: float
) -> np.ndarray:
    """Integrate theta''=-coefficient*sin(theta) with float64 classical RK4."""
    stride = round(output_dt / h)
    steps = round(duration / h)
    assert abs(stride * h - output_dt) < 1e-12 and steps % stride == 0
    theta = np.asarray(theta0, dtype=np.float64).copy()
    omega = np.zeros_like(theta)
    result = np.empty((steps // stride + 1, len(theta), 2))
    result[0] = np.stack((theta, omega), axis=-1)
    for k in range(1, steps + 1):
        a = -coefficient * np.sin(theta)
        b = -coefficient * np.sin(theta + h * omega / 2)
        c = -coefficient * np.sin(theta + h * omega / 2 + h * h * a / 4)
        d = -coefficient * np.sin(theta + h * omega + h * h * b / 2)
        theta += h * omega + h * h * (a + b + c) / 6
        omega += h * (a + 2 * b + 2 * c + d) / 6
        if k % stride == 0:
            result[k // stride] = np.stack((theta, omega), axis=-1)
    return result


def periods(t: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Periods between descending zero crossings, with linear interpolation."""
    ix = np.flatnonzero((theta[:-1] > 0) & (theta[1:] <= 0))
    crossings = t[ix] - theta[ix] * (t[ix + 1] - t[ix]) / (theta[ix + 1] - theta[ix])
    return np.diff(crossings)


def measure(
    case: str, t: np.ndarray, state: np.ndarray, ref: np.ndarray, cfg: dict
) -> dict:
    """Evaluate raw, unsmoothed samples using frozen candidate thresholds."""
    error = state - ref
    out = {
        "position_rmse": float(np.sqrt(np.mean(error[:, 0] ** 2))),
        "position_max_error": float(np.max(np.abs(error[:, 0]))),
        "velocity_rmse": float(np.sqrt(np.mean(error[:, 1] ** 2))),
        "velocity_max_error": float(np.max(np.abs(error[:, 1]))),
        "position_final_error": float(error[-1, 0]),
        "velocity_final_error": float(error[-1, 1]),
    }
    limits = cfg["candidate_thresholds"]
    if case == "freefall":
        mask = t <= 1.0 + 1e-12
        out["position_max_0_1s_m"] = float(np.max(np.abs(error[mask, 0])))
        out["velocity_max_0_1s_m_s"] = float(np.max(np.abs(error[mask, 1])))
        out["position_normalized_max"] = out["position_max_error"] / (
            cfg["gravity_m_s2"] * t[-1] ** 2 / 2
        )
        out["velocity_normalized_max"] = out["velocity_max_error"] / (
            cfg["gravity_m_s2"] * t[-1]
        )
        passed = (
            out["position_max_0_1s_m"] <= limits["fall_position_0_1s_m"]
            and out["velocity_max_0_1s_m_s"] <= limits["fall_velocity_0_1s_m_s"]
        )
        bad = mask & (
            (np.abs(error[:, 0]) > limits["fall_position_0_1s_m"])
            | (np.abs(error[:, 1]) > limits["fall_velocity_0_1s_m_s"])
        )
    else:
        m, g, l, r = (
            cfg["mass_kg"],
            cfg["gravity_m_s2"],
            cfg["pendulum_length_m"],
            cfg["bob_radius_m"],
        )
        inertia = m * (l * l + 0.4 * r * r)
        e = inertia * state[:, 1] ** 2 / 2 + m * g * l * (1 - np.cos(state[:, 0]))
        relative = (e - e[0]) / e[0]
        measured_periods, reference_periods = periods(t, state[:, 0]), periods(
            t, ref[:, 0]
        )
        out["period_s"] = float(np.mean(measured_periods))
        out["reference_period_s"] = float(np.mean(reference_periods))
        out["period_count"] = len(measured_periods)
        out["period_relative_error"] = abs(
            out["period_s"] / out["reference_period_s"] - 1
        )
        out["energy_initial_j"] = float(e[0])
        out["energy_max_relative_drift"] = float(np.max(np.abs(relative)))
        out["energy_final_relative_drift"] = float(relative[-1])
        out["energy_max_absolute_drift_j"] = float(np.max(np.abs(e - e[0])))
        out["energy_max_drift_over_mgl"] = out["energy_max_absolute_drift_j"] / (
            m * g * l
        )
        passed = (
            out["period_relative_error"] <= limits["pendulum_period_relative"]
            and out["energy_max_relative_drift"] <= limits["pendulum_energy_relative"]
        )
        bad = np.abs(relative) > limits["pendulum_energy_relative"]
    out["candidate_pass"] = bool(passed)
    out["first_pointwise_threshold_failure_s"] = (
        float(t[np.flatnonzero(bad)[0]]) if np.any(bad) else None
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((PROJECT / "configs/P-01/isaaclab.json").read_text())
    metadata = [json.loads((p / "run.json").read_text()) for p in args.runs]
    assert all(m["status"] == "completed" and m["phase"] == "main" for m in metadata)
    assert {(m["case"], m["dt_s"]) for m in metadata} == {
        (c, d) for c in ["freefall", "pendulum"] for d in cfg["dt_s"]
    }
    assert len(metadata) == 6 and all(m["completed_episodes"] == 10 for m in metadata)
    assert len({m["hashes"]["code/isaaclab/p01_dynamics.py"] for m in metadata}) == 1
    assert all(
        m["hashes"]["configs/P-01/isaaclab.json"]
        == hashlib.sha256(
            (PROJECT / "configs/P-01/isaaclab.json").read_bytes()
        ).hexdigest()
        for m in metadata
    )
    pendulum_run = next(
        p for p, m in zip(args.runs, metadata) if m["case"] == "pendulum"
    )
    theta0 = np.array(
        [
            np.load(pendulum_run / f"episode_{i:02d}.npz")["state"][0, 0]
            for i in range(10)
        ]
    )
    m, g, l, r = (
        cfg["mass_kg"],
        cfg["gravity_m_s2"],
        cfg["pendulum_length_m"],
        cfg["bob_radius_m"],
    )
    inertia = m * (l * l + 0.4 * r * r)
    coefficient = m * g * l / inertia
    base_dt = min(cfg["dt_s"])
    ref = reference(theta0, 10.0, cfg["reference_dt_s"], base_dt, coefficient)
    refined = reference(
        theta0, 10.0, cfg["reference_refined_dt_s"], base_dt, coefficient
    )
    check = {
        "max_theta_difference_rad": float(
            np.max(np.abs(ref[:, :, 0] - refined[:, :, 0]))
        ),
        "max_omega_difference_rad_s": float(
            np.max(np.abs(ref[:, :, 1] - refined[:, :, 1]))
        ),
    }
    # A different integrator and the exact elliptic-integral period check the RK4 implementation.
    times = np.arange(ref.shape[0]) * base_dt
    independent = solve_ivp(
        lambda t, y: [y[1], -coefficient * np.sin(y[0])],
        [0.0, 10.0],
        [theta0[0], 0.0],
        method="DOP853",
        rtol=1e-12,
        atol=1e-14,
        t_eval=times,
    )
    check["dop853_max_theta_difference_rad"] = float(
        np.max(np.abs(ref[:, 0, 0] - independent.y[0]))
    )
    check["elliptic_period_s"] = float(
        4 * np.sqrt(inertia / (m * g * l)) * ellipk(np.sin(theta0[0] / 2) ** 2)
    )
    check["rk4_sampled_period_s"] = float(periods(times, ref[:, 0, 0]).mean())
    energy = inertia * ref[:, :, 1] ** 2 / 2 + m * g * l * (1 - np.cos(ref[:, :, 0]))
    check["reference_energy_max_relative_drift"] = float(
        np.max(np.abs(energy / energy[0] - 1))
    )
    assert check["max_theta_difference_rad"] < 1e-8
    assert check["max_omega_difference_rad_s"] < 1e-7
    assert check["dop853_max_theta_difference_rad"] < 1e-8
    assert abs(check["rk4_sampled_period_s"] / check["elliptic_period_s"] - 1) < 1e-6
    assert check["reference_energy_max_relative_drift"] < 1e-8
    np.savez_compressed(
        out / "pendulum_reference.npz",
        t=times,
        state=ref,
        refined_state=refined,
        theta0=theta0,
    )
    (out / "reference_validation.json").write_text(json.dumps(check, indent=2) + "\n")
    rows = []
    traces = {}
    inputs = {}
    for path, meta in zip(args.runs, metadata):
        case, dt = meta["case"], meta["dt_s"]
        inputs[meta["run_id"]] = {
            "run_json_sha256": hashlib.sha256(
                (path / "run.json").read_bytes()
            ).hexdigest(),
            "episodes": {},
        }
        for i in range(10):
            file = path / f"episode_{i:02d}.npz"
            data = np.load(file)
            t, state = data["t"], data["state"]
            assert (
                state.shape == (round(10 / dt) + 1, 2)
                and t[-1] == 10
                and np.isfinite(state).all()
            )
            assert abs(state[0, 1]) < 1e-8
            if case == "freefall":
                expected = np.stack((state[0, 0] - g * t * t / 2, -g * t), axis=-1)
            else:
                assert abs(state[0, 0] - theta0[i]) < 1e-8
                expected = ref[:: round(dt / base_dt), i]
            row = dict(
                run_id=meta["run_id"],
                case=case,
                dt_s=dt,
                initial_id=i,
                initial=float(state[0, 0]),
                **measure(case, t, state, expected, cfg),
            )
            if case == "pendulum":
                geometric = np.stack(
                    (
                        -l * np.sin(state[:, 0]),
                        np.zeros(len(t)),
                        1 - l * np.cos(state[:, 0]),
                    ),
                    axis=-1,
                )
                row["com_kinematic_max_error_m"] = float(
                    np.max(np.linalg.norm(data["com_position"] - geometric, axis=1))
                )
                assert row["com_kinematic_max_error_m"] < 1e-5
            rows.append(row)
            if i == 0:
                traces[case, dt] = (t, state, expected)
            inputs[meta["run_id"]]["episodes"][file.name] = hashlib.sha256(
                file.read_bytes()
            ).hexdigest()
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with (out / "metrics.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for case in ["freefall", "pendulum"]:
        for dt in cfg["dt_s"]:
            subset = [x for x in rows if x["case"] == case and x["dt_s"] == dt]
            group = {
                "case": case,
                "dt_s": dt,
                "trials": len(subset),
                "candidate_pass_count": sum(x["candidate_pass"] for x in subset),
            }
            for key in subset[0]:
                if key in [
                    "initial",
                    "initial_id",
                    "dt_s",
                    "candidate_pass",
                    "first_pointwise_threshold_failure_s",
                ] or not isinstance(subset[0][key], (float, int)):
                    continue
                values = [row[key] for row in subset]
                group[key] = {
                    "mean": float(np.mean(values)),
                    "min": float(min(values)),
                    "max": float(max(values)),
                }
            summary.append(group)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "analysis.json").write_text(
        json.dumps(
            {
                "script_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "inputs": inputs,
                "reference_check": check,
                "packages": {
                    n: importlib.metadata.version(n)
                    for n in ["numpy", "scipy", "matplotlib"]
                },
            },
            indent=2,
        )
        + "\n"
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 180,
        }
    )
    colors = ["#C85139", "#3478A4", "#29916C"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    t, s, r0 = traces["freefall", cfg["dt_s"][0]]
    axes[0, 0].plot(t, r0[:, 0], color="black", ls="--", label="Analytical")
    for dt, color in zip(cfg["dt_s"], colors):
        t, s, r0 = traces["freefall", dt]
        label = f"{dt*1000:g} ms"
        axes[0, 0].plot(t, s[:, 0], color=color, label=label, alpha=0.8)
        mask = t <= 1
        axes[0, 1].plot(t[mask], (s - r0)[mask, 0] * 1000, color=color, label=label)
        axes[1, 0].plot(t, (s - r0)[:, 0] * 1000, color=color, label=label)
        axes[1, 1].plot(t, (s - r0)[:, 1] * 1000, color=color, label=label)
    axes[0, 1].axhline(-5, color="gray", ls=":", label="Candidate limit (-5 mm)")
    labels = [
        ("Height (m)", "Free fall: trajectory"),
        ("Position error (mm)", "First second: signed error"),
        ("Position error (mm)", "Full 10 s: signed error"),
        ("Velocity error (mm/s)", "Velocity error"),
    ]
    for ax, (ylabel, title) in zip(axes.flat, labels):
        ax.set(xlabel="Simulation time (s)", ylabel=ylabel, title=title)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("P-01 | Isaac Lab / GPU PhysX TGS | Initial state 0", fontsize=13)
    fig.savefig(out / "freefall.png")
    fig.savefig(out / "freefall.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    t, s, r0 = traces["pendulum", cfg["dt_s"][0]]
    axes[0, 0].plot(
        t, np.rad2deg(r0[:, 0]), color="black", ls="--", label="RK4 reference"
    )
    for dt, color in zip(cfg["dt_s"], colors):
        t, s, r0 = traces["pendulum", dt]
        label = f"{dt*1000:g} ms"
        axes[0, 0].plot(t, np.rad2deg(s[:, 0]), color=color, label=label, alpha=0.8)
        axes[0, 1].plot(t, np.rad2deg((s - r0)[:, 0]), color=color, label=label)
        e = inertia * s[:, 1] ** 2 / 2 + m * g * l * (1 - np.cos(s[:, 0]))
        axes[1, 0].plot(t, 100 * (e / e[0] - 1), color=color, label=label)
        axes[1, 1].plot(t, (s - r0)[:, 1], color=color, label=label)
    for ax, (ylabel, title) in zip(
        axes.flat,
        [
            ("Angle (deg)", "Pendulum: trajectory"),
            ("Angle error (deg)", "Signed angle error"),
            ("Energy change / E0 (%)", "Energy drift"),
            ("Angular velocity error (rad/s)", "Angular velocity error"),
        ],
    ):
        ax.set(xlabel="Simulation time (s)", ylabel=ylabel, title=title)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("P-01 | Passive pendulum | Initial state 0", fontsize=13)
    fig.savefig(out / "pendulum.png")
    fig.savefig(out / "pendulum.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    panels = [
        ("freefall", "position_max_0_1s_m", 1000, "Max position error, 0-1 s (mm)", 5),
        (
            "freefall",
            "position_max_error",
            1000,
            "Max position error, 0-10 s (mm)",
            None,
        ),
        ("pendulum", "period_relative_error", 100, "Period error (%)", 1),
        ("pendulum", "energy_max_relative_drift", 100, "Max energy drift / E0 (%)", 2),
    ]
    for ax, (case, key, scale, label, limit) in zip(axes.flat, panels):
        groups = sorted(
            [x for x in summary if x["case"] == case], key=lambda x: x["dt_s"]
        )
        x = np.array([v["dt_s"] * 1000 for v in groups])
        y = np.array([v[key]["mean"] * scale for v in groups])
        low = np.array([v[key]["min"] * scale for v in groups])
        high = np.array([v[key]["max"] * scale for v in groups])
        ax.errorbar(
            x,
            y,
            yerr=[y - low, high - y],
            fmt="o-",
            capsize=4,
            color="#3478A4",
            label="Mean / min-max (10 initial states)",
        )
        if limit is not None:
            ax.axhline(limit, color="#C85139", ls="--", label="Candidate threshold")
        ax.set(xlabel="Physics time step (ms)", ylabel=label, xticks=x)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle(
        "P-01 | Time-step sweep | Fixed TGS iterations: 4 position / 1 velocity",
        fontsize=12,
    )
    fig.savefig(out / "convergence.png")
    fig.savefig(out / "convergence.pdf")
    plt.close(fig)
    if args.diagnostic:
        diagnostic_meta = json.loads((args.diagnostic / "run.json").read_text())
        assert (
            diagnostic_meta["phase"] == "diagnostic"
            and diagnostic_meta["position_iterations"] == 1
        )
        data = np.load(args.diagnostic / "episode_00.npz")
        t, s = data["t"], data["state"]
        assert s[0, 0] == theta0[0]
        expected = ref[:: round(diagnostic_meta["dt_s"] / base_dt), 0]
        diagnostic_metrics = measure("pendulum", t, s, expected, cfg)
        diagnostic_metrics["run_id"] = diagnostic_meta["run_id"]
        diagnostic_metrics["position_iterations"] = 1
        diagnostic_metrics["source_npz_sha256"] = hashlib.sha256(
            (args.diagnostic / "episode_00.npz").read_bytes()
        ).hexdigest()
        (out / "diagnostic.json").write_text(
            json.dumps(diagnostic_metrics, indent=2) + "\n"
        )
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for count, arrays, color in [
            (4, traces["pendulum", 0.005], colors[0]),
            (1, (t, s, expected), colors[2]),
        ]:
            tt, ss, rr = arrays
            energy = inertia * ss[:, 1] ** 2 / 2 + m * g * l * (1 - np.cos(ss[:, 0]))
            axes[0].plot(
                tt,
                100 * (energy / energy[0] - 1),
                color=color,
                label=f"{count} position iterations",
            )
            axes[1].plot(
                tt,
                np.rad2deg((ss - rr)[:, 0]),
                color=color,
                label=f"{count} position iterations",
            )
        axes[0].axhline(2, color="gray", ls=":", label="Candidate +2% threshold")
        axes[0].set(ylabel="Energy change / E0 (%)", xlabel="Simulation time (s)")
        axes[1].set(ylabel="Angle error (deg)", xlabel="Simulation time (s)")
        for ax in axes:
            ax.grid(alpha=0.2)
            ax.legend(fontsize=9)
        fig.suptitle(
            "Diagnostic only | 5 ms, initial state 0 | 4 vs 1 TGS position iterations"
        )
        fig.savefig(out / "iteration_diagnostic.png")
        fig.savefig(out / "iteration_diagnostic.pdf")
        plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
