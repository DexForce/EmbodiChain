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

"""Recompute P-01 DexUni and preserved Isaac Lab metrics from raw trajectories."""

from __future__ import annotations

import os

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from scripts.benchmark.tech_reports.reporting.p01_analysis import measure, reference

ROOT = Path(
    os.environ.get("SIM_REPORT_ROOT", Path(__file__).resolve().parents[4])
).resolve()
OUT = ROOT / "reports/P-01/dexuni"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT / "configs/P-01/embodichain-dexuni.json").read_text())
    ledger_path = ROOT / "configs/P-01/embodichain-dexuni-runs.execution.json"
    ledger = json.loads(ledger_path.read_text())
    assert len(ledger) == 8 and all(e["status"] == "completed" for e in ledger)
    inputs_hash = hashlib.sha256(
        (ROOT / cfg["initial_states"]).read_bytes()
    ).hexdigest()
    entries = [(e["run_id"], "DexUni", e["entry"]["phase"]) for e in ledger]
    for case in ["freefall", "pendulum"]:
        for tag in ["010", "005", "0025"]:
            entries.append((f"20261014_main_{case}_dt{tag}", "TGS 4/1", "main"))
            entries.append((f"20261014_pgs_baseline_{case}_dt{tag}", "PGS 4/1", "main"))
        entries.append((f"20261014_study_{case}_dt001_p1", "TGS 1/1", "supplement"))
        entries.append((f"20261014_pgs_baseline_{case}_dt001", "PGS 4/1", "supplement"))
    first_pendulum = next(r for r, s, _ in entries if s == "DexUni" and "pendulum" in r)
    theta0 = np.array(
        [
            np.load(ROOT / "results/P-01" / first_pendulum / f"episode_{i:02d}.npz")[
                "state"
            ][0, 0]
            for i in range(10)
        ]
    )
    m, g, length, radius = (
        cfg[k] for k in ["mass_kg", "gravity_m_s2", "pendulum_length_m", "bob_radius_m"]
    )
    inertia = m * (length**2 + 0.4 * radius**2)
    coefficient = m * g * length / inertia
    sample_dt = 0.0005
    base = reference(theta0, 10.0, cfg["reference_dt_s"], sample_dt, coefficient)
    refined = reference(
        theta0, 10.0, cfg["reference_refined_dt_s"], sample_dt, coefficient
    )
    times = np.arange(len(base)) * sample_dt
    independent = solve_ivp(
        lambda _, y: [y[1], -coefficient * np.sin(y[0])],
        [0, 10],
        [theta0[0], 0],
        t_eval=times,
        method="DOP853",
        rtol=1e-12,
        atol=1e-14,
    )
    assert independent.success
    reference_check = dict(
        refinement_angle_rad=float(np.max(np.abs(base[:, :, 0] - refined[:, :, 0]))),
        refinement_velocity_rad_s=float(
            np.max(np.abs(base[:, :, 1] - refined[:, :, 1]))
        ),
        dop853_angle_rad=float(np.max(np.abs(base[:, 0, 0] - independent.y[0]))),
        dop853_velocity_rad_s=float(np.max(np.abs(base[:, 0, 1] - independent.y[1]))),
    )
    assert max(reference_check.values()) < 1e-8
    np.savez_compressed(
        OUT / "reference.npz", t=times, state=base, refined_state=refined
    )
    rows, summaries, provenance, traces = [], [], {}, {}
    recurrence_checks = []
    for run_id, label, phase in entries:
        run = ROOT / "results/P-01" / run_id
        meta = json.loads((run / "run.json").read_text())
        assert meta["status"] == "completed" and meta["completed_episodes"] == 10
        assert meta["duration_s"] == 10 and meta["num_envs"] == 1
        assert meta["hashes"][cfg["initial_states"]] == inputs_hash
        if label == "DexUni":
            assert meta["resolved_solver"] == "dexuni" and meta["num_substeps"] == 1
            assert meta["repository_commit"] == cfg["source_commit"]
            assert meta["repository_status"] == ""
            assert all(
                str(Path(cfg["repository"])) in p
                for p in meta["module_origins"].values()
            )
        case, dt = meta["case"], meta["dt_s"]
        subset = []
        for episode in range(10):
            file = run / f"episode_{episode:02d}.npz"
            a = np.load(file)
            t, state = a["t"], a["state"]
            assert state.shape == (round(10 / dt) + 1, 2) and np.isfinite(state).all()
            np.testing.assert_allclose(t, np.arange(len(t)) * dt, rtol=0, atol=1e-12)
            assert state[0, 1] == 0
            if label == "DexUni":
                np.testing.assert_array_equal(
                    state, np.load(run / f"episode_{episode:02d}_state.npy")
                )
            if case == "pendulum":
                assert state[0, 0] == theta0[episode], (
                    run_id,
                    episode,
                    state[0, 0],
                    theta0[episode],
                )
                ref = base[:: round(dt / sample_dt), episode]
                geometric = np.column_stack(
                    (
                        -length * np.sin(state[:, 0]),
                        np.zeros(len(t)),
                        1 - length * np.cos(state[:, 0]),
                    )
                )
                geometry_error = float(
                    np.max(np.linalg.norm(a["com_position"] - geometric, axis=1))
                )
                assert geometry_error < 1e-5, (run_id, geometry_error)
            else:
                ref = np.column_stack((state[0, 0] - g * t**2 / 2, -g * t))
                geometry_error = None
                if label == "DexUni":
                    # Local VBD source: velocity kick, position drift, then
                    # overwrite velocity using the rounded position difference.
                    # float64 products/sums here emulate one rounding of these
                    # float32 FMA operands; this is an offline numerical model.
                    predicted = np.empty_like(state)
                    z, velocity = map(np.float32, state[0])
                    h32, gravity32 = np.float32(dt), np.float32(-g)
                    predicted[0] = (z, velocity)
                    for k in range(1, len(t)):
                        kicked = np.float32(
                            float(velocity) + float(gravity32) * float(h32)
                        )
                        next_z = np.float32(float(z) + float(kicked) * float(h32))
                        velocity = np.float32(np.float32(next_z - z) / h32)
                        z = next_z
                        predicted[k] = (z, velocity)
                    difference = np.abs(predicted - state)
                    recurrence_checks.append(
                        dict(
                            run_id=run_id,
                            initial_id=episode,
                            exactly_equal=bool(np.array_equal(predicted, state)),
                            max_position_difference_m=float(difference[:, 0].max()),
                            max_velocity_difference_m_s=float(difference[:, 1].max()),
                        )
                    )
                    model_dir = OUT / "roundoff_replay" / run_id
                    model_dir.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        model_dir / f"episode_{episode:02d}.npz",
                        t=t,
                        predicted_state=predicted,
                        float64_semi_implicit_state=np.column_stack(
                            (ref[:, 0] - g * dt * t / 2, ref[:, 1])
                        ),
                    )
            row = dict(
                run_id=run_id,
                source=str(file.relative_to(ROOT)),
                platform="EmbodiChain" if label == "DexUni" else "Isaac Lab",
                solver=label,
                phase=phase,
                case=case,
                dt_s=dt,
                initial_id=episode,
                initial=float(state[0, 0]),
                com_geometry_max_error_m=geometry_error,
                **measure(case, t, state, ref, cfg),
            )
            if case == "pendulum":
                assert row["period_count"] >= 2
            rows.append(row)
            subset.append(row)
            if label == "DexUni":
                derived = OUT / "derived" / run_id
                derived.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    derived / f"episode_{episode:02d}.npz",
                    t=t,
                    reference=ref,
                    error=state - ref,
                )
            if episode == 0:
                traces[label, case, dt] = (t, state, ref)
        summary = dict(
            run_id=run_id,
            solver=label,
            phase=phase,
            case=case,
            dt_s=dt,
            trials=10,
            pass_count=sum(r["candidate_pass"] for r in subset),
        )
        for key in subset[0]:
            if key in summary or key in [
                "initial_id",
                "candidate_pass",
                "first_pointwise_threshold_failure_s",
            ]:
                continue
            if isinstance(subset[0][key], (float, int)):
                values = [r[key] for r in subset]
                summary[key] = dict(
                    mean=float(np.mean(values)),
                    min=float(min(values)),
                    max=float(max(values)),
                )
        if label == "DexUni":
            summary["diagnostic_wall_s_mean"] = float(
                np.mean(meta["diagnostic_wall_s_per_episode"])
            )
            summary["process_peak_rss_mib"] = meta["process_peak_rss_mib"]
        summaries.append(summary)
        files = (
            sorted(run.rglob("*"))
            if label == "DexUni"
            else [run / "run.json", *sorted(run.glob("episode_*.npz"))]
        )
        provenance[run_id] = {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files
            if p.is_file()
        }
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    manifest = dict(
        experiment="P-01",
        reference_check=reference_check,
        inputs_sha256=inputs_hash,
        analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        measurement_script_sha256=hashlib.sha256(
            Path(__file__).with_name("p01_analysis.py").read_bytes()
        ).hexdigest(),
        runs=provenance,
    )
    (OUT / "analysis.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUT / "roundoff-replay.json").write_text(
        json.dumps(recurrence_checks, indent=2) + "\n"
    )
    # Simple diagnostic plots; no publication layout or image synthesis.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    for dt in [0.01, 0.005, 0.0025, 0.001]:
        t, state, ref = traces["DexUni", "freefall", dt]
        axes[0].plot(t, (state[:, 0] - ref[:, 0]) * 1000, label=f"{dt*1000:g} ms")
        axes[1].plot(t, state[:, 1] - ref[:, 1], label=f"{dt*1000:g} ms")
    for ax in axes:
        ax.set_xlabel("Time (s)")
        ax.grid(alpha=0.25)
        ax.legend()
    axes[0].set_ylabel("Height error (mm)")
    axes[1].set_ylabel("Velocity error (m/s)")
    fig.savefig(OUT / "freefall-errors.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    for dt in [0.01, 0.005, 0.0025, 0.001]:
        t, state, ref = traces["DexUni", "pendulum", dt]
        energy = inertia * state[:, 1] ** 2 / 2 + m * g * length * (
            1 - np.cos(state[:, 0])
        )
        axes[0].plot(
            t, (state[:, 0] - ref[:, 0]) * 180 / np.pi, label=f"{dt*1000:g} ms"
        )
        axes[1].plot(t, (energy / energy[0] - 1) * 100, label=f"{dt*1000:g} ms")
    for ax in axes:
        ax.set_xlabel("Time (s)")
        ax.grid(alpha=0.25)
        ax.legend()
    axes[0].set_ylabel("Angle error (deg)")
    axes[1].set_ylabel("Energy drift (%)")
    fig.savefig(OUT / "pendulum-errors.png", dpi=150)
    plt.close(fig)
    print(
        json.dumps(
            {
                "trajectories": len(rows),
                "reference_check": reference_check,
                "dexuni": [s for s in summaries if s["solver"] == "DexUni"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
