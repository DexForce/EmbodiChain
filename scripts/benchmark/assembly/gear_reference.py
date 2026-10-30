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
"""Measure true hinged-gear source dynamics, solver choice and step convergence.

Run after preparing assembly manifests with ``asset_gallery``::

    python scripts/benchmark/assembly/gear_reference.py --manifests assets/*.json

This reference runner imports only NumPy and MuJoCo, so the same immutable
manifests run in both the historical and backend-matched reference environments.
It never prescribes follower motion or adds a gear constraint. Reversal phase
span is a dynamic diagnostic, not an independent static backlash measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import time

import mujoco
import numpy as np

__all__: list[str] = []


def _load_manifest(path: Path) -> tuple[dict, str]:
    document = json.loads(path.read_text())
    asset = document["asset"]
    encoded = json.dumps(asset, sort_keys=True, separators=(",", ":"), allow_nan=False)
    actual = hashlib.sha256(encoded.encode()).hexdigest()
    if actual != document["sha256"]:
        raise ValueError(f"Manifest checksum mismatch: {path}")
    if asset["config"].get("kind") != "gear":
        raise ValueError(f"Not a gear manifest: {path}")
    if len(asset["config"]["teeth"]) != 2:
        raise ValueError("Ratio measurement requires a gear pair.")
    return asset, actual


def _backlash(asset: dict) -> dict:
    """Bracket first tooth contact in both directions with the follower fixed.

    This is a static geometry probe. It changes no mass, contact parameter or
    dynamics trajectory; nominal pitch-circle play is reported separately.
    """
    model = mujoco.MjModel.from_xml_string(asset["source_mjcf"])
    return _backlash_model(model, asset["config"])


def _backlash_model(model: mujoco.MjModel, config: dict) -> dict:
    """Scan a source or compiled pair using its actual public MuJoCo model."""
    data = mujoco.MjData(model)
    ids = {
        next(i for i in range(model.nbody) if model.body(i).name.endswith(name))
        for name in ("gear_a", "gear_b")
    }
    if model.nq != 2 or len(ids) != 2:
        raise ValueError("Backlash scan requires precisely two hinged gears.")
    teeth = config["teeth"]
    width = float(config.get("tooth_width", 0.48))

    def touches(angle: float) -> bool:
        data.qpos[:] = [angle, 0.0]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_collision(model, data)
        return any(
            {
                int(model.geom_bodyid[int(data.contact[i].geom[0])]),
                int(model.geom_bodyid[int(data.contact[i].geom[1])]),
            }
            == ids
            and data.contact[i].dist <= 0
            for i in range(data.ncon)
        )

    if touches(0.0):
        return {"status": "initial_contact", "driver_free_play_rad": 0.0}
    bounds = []
    for sign in [-1, 1]:
        low = 0.0
        high = None
        # Find the FIRST collision interval, rather than assuming tooth-contact
        # occupancy is monotone over an entire revolution.
        for angle in np.linspace(0.0, np.pi / teeth[0], 65)[1:]:
            if touches(sign * angle):
                high = float(angle)
                break
            low = float(angle)
        if high is None:
            return {"status": "no_contact_in_half_tooth", "direction": sign}
        for _ in range(32):
            middle = (low + high) / 2
            if touches(sign * middle):
                high = middle
            else:
                low = middle
        bounds.append([sign * low, sign * high])
    total = sum(abs(interval[1]) for interval in bounds)
    module = float(config["module"]) * float(config.get("scale", 1.0))
    return {
        "status": "bracketed",
        "driver_contact_angle_brackets_rad": bounds,
        "driver_free_play_rad": total,
        "follower_equivalent_play_rad": total * teeth[0] / teeth[1],
        "tangential_play_m": total * teeth[0] * module / 2,
        "nominal_pitch_circle_play_m": 2 * (0.5 - width) * np.pi * module,
        "angular_bracket_tolerance_rad": float(max(abs(b - a) for a, b in bounds)),
        "note": "Source mesh/contact geometry probe; nominal involute pitch-circle play is an approximate analytic comparison, not a pass criterion for the polygonal outline.",
    }


def _trajectory(model: mujoco.MjModel, duration: float, scenario: str) -> tuple:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    steps = round(duration / model.opt.timestep)
    q = np.empty((steps + 1, 2), dtype=np.float64)
    v = np.empty_like(q)
    contacts = np.empty(steps + 1, dtype=np.int32)
    q[0], v[0], contacts[0] = data.qpos, data.qvel, data.ncon
    start = time.perf_counter()
    for step in range(steps):
        reverse = scenario == "reversal" and step >= steps // 2
        data.ctrl[0] = -1.0 if reverse else 1.0
        # A positive follower torque opposes its expected negative rotation.
        data.qfrc_applied[1] = 0.002 if scenario == "loaded" else 0.0
        mujoco.mj_step(model, data)
        q[step + 1], v[step + 1], contacts[step + 1] = data.qpos, data.qvel, data.ncon
    elapsed = time.perf_counter() - start
    return q, v, contacts, elapsed, sum(int(w.number) for w in data.warning)


def _metrics(q: np.ndarray, v: np.ndarray, teeth: list[int], scenario: str) -> dict:
    expected = -teeth[0] / teeth[1]
    # Evaluate the final quarter after startup or a mid-run torque reversal.
    first = len(q) * 3 // 4
    delta = q[-1] - q[first]
    eligible = bool(abs(delta[0]) >= np.pi / 2)
    ratio = float(delta[1] / delta[0]) if abs(delta[0]) > 1e-12 else None
    error = abs(ratio / expected - 1) if ratio is not None else None
    phase = q[:, 1] - expected * q[:, 0]
    finite = bool(np.isfinite(q).all() and np.isfinite(v).all())
    return {
        "finite": finite,
        "driver_rotation_rad": float(q[-1, 0]),
        "follower_rotation_rad": float(q[-1, 1]),
        "steady_driver_travel_rad": float(delta[0]),
        "sufficient_steady_travel": eligible,
        "expected_ratio": expected,
        "steady_ratio": ratio,
        "steady_ratio_relative_error": error,
        "cumulative_ratio_relative_error": (
            float(abs((q[-1, 1] / q[-1, 0]) / expected - 1))
            if abs(q[-1, 0]) > 1e-12
            else None
        ),
        "steady_phase_span_rad": float(np.ptp(phase[first:])),
        "phase_span_rad": float(np.ptp(phase)),
        "final_phase_rad": float(phase[-1]),
        "maximum_speed_rad_s": float(np.abs(v).max()),
        "scenario": scenario,
    }


def _report(output: Path, rows: list[dict], metadata: dict) -> Path:
    lines = [
        "# Hinged gear source reference",
        "",
        json.dumps(metadata, ensure_ascii=False),
        "",
        "## Time & Memory",
        "",
        "| config | solver | dt_ms | scenario | cost_time_ms | cpu_delta_mb | gpu_delta_mb | peak_gpu_mb |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['config']} | {r['solver']} | {r['timestep']*1000:g} | {r['scenario']} | {r['cost_time_ms']:.3f} | unmeasured | n/a | n/a |"
        )
    lines += [
        "",
        "## Success & Other Metrics",
        "",
        "| config | solver | dt_ms | scenario | success_rate | steady ratio error % | phase span rad | warnings |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        error = r["metrics"]["steady_ratio_relative_error"]
        lines.append(
            f"| {r['config']} | {r['solver']} | {r['timestep']*1000:g} | {r['scenario']} | {int(r['passed'])} | {error*100 if error is not None else 'n/a'} | {r['metrics']['steady_phase_span_rad']:.6g} | {r['warning_count']} |"
        )
    lines += [
        "",
        "## Leaderboard",
        "",
        "| rank | algorithm | overall_success_rate | cells |",
        "| --- | --- | --- | --- |",
    ]
    solvers = {r["solver"] for r in rows}
    totals = [
        (
            solver,
            sum(r["passed"] for r in rows if r["solver"] == solver),
            sum(r["solver"] == solver for r in rows),
        )
        for solver in solvers
    ]
    for rank, (solver, passed, count) in enumerate(
        sorted(totals, key=lambda t: -t[1] / t[2]), 1
    ):
        lines.append(f"| {rank} | {solver} | {passed/count:.6f} | {count} |")
    lines += [
        "",
        "Ratio acceptance is 0.5% relative error, finite state, no warnings, and at least pi/2 driver travel over the final quarter. Phase span is a dynamic diagnostic, not a static backlash oracle. Reversal cases may fail the travel gate if inertia prevents sustained reverse rotation within the run. Source geometry, contacts and mass remain unchanged. Timing is CPU wall time for complete trajectories with host trace recording, median over repeats after one untimed warmup; generation/compile is separate. CPU delta and GPU metrics are unmeasured. Process maximum RSS is separately recorded in raw metadata; it is not incremental peak memory.",
        "",
    ]
    path = output / "report.md"
    path.write_text("\n".join(lines))
    return path


def run_all_benchmarks(args: argparse.Namespace) -> Path:
    """Run source-only reference cells and persist raw measurements and traces."""
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    static_geometry = []
    metadata = {
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "threads": {
            k: os.getenv(k)
            for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "repeats": args.repeats,
        "duration": args.duration,
        "seed": 0,
        "precision": "MuJoCo CPU FP64",
    }
    for path in args.manifests:
        asset, digest = _load_manifest(Path(path))
        static_geometry.append(
            {"config": Path(path).stem, "asset_sha256": digest, **_backlash(asset)}
        )
        for solver in args.solvers:
            for dt in args.timesteps:
                start = time.perf_counter()
                model = mujoco.MjModel.from_xml_string(asset["source_mjcf"])
                model.opt.timestep = dt
                model.opt.solver = getattr(mujoco.mjtSolver, f"mjSOL_{solver.upper()}")
                model.opt.iterations = args.iterations
                compile_ms = (time.perf_counter() - start) * 1000
                for scenario in args.scenarios:
                    _trajectory(model, args.duration, scenario)
                    runs = [
                        _trajectory(model, args.duration, scenario)
                        for _ in range(args.repeats)
                    ]
                    q, v, contact, _, warnings = runs[-1]
                    metrics = _metrics(q, v, asset["config"]["teeth"], scenario)
                    passed = bool(
                        metrics["finite"]
                        and metrics["sufficient_steady_travel"]
                        and metrics["steady_ratio_relative_error"] is not None
                        and metrics["steady_ratio_relative_error"] <= 0.005
                        and not warnings
                    )
                    row = {
                        "config": Path(path).stem,
                        "asset_sha256": digest,
                        "solver": solver,
                        "timestep": dt,
                        "scenario": scenario,
                        "compile_ms": compile_ms,
                        "cost_time_ms": statistics.median(r[3] for r in runs) * 1000,
                        "trajectory_time_samples_ms": [r[3] * 1000 for r in runs],
                        "warning_count": warnings,
                        "max_contacts": int(contact.max()),
                        "metrics": metrics,
                        "passed": passed,
                    }
                    trace = f"trace_{len(rows):03d}.npz"
                    np.savez_compressed(
                        output / trace, qpos=q, qvel=v, contacts=contact, timestep=dt
                    )
                    row["trace"] = trace
                    rows.append(row)
                    print(
                        f"{row['config']} {solver} {dt:g} {scenario}: {'passed' if passed else 'failed'}, error={metrics['steady_ratio_relative_error']}",
                        flush=True,
                    )
    metadata["process_max_rss_mb"] = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    )
    (output / "raw.json").write_text(
        json.dumps(
            {"metadata": metadata, "static_geometry": static_geometry, "rows": rows},
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    report = _report(output, rows, metadata)
    print(f"Markdown report saved: {report}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument(
        "--solvers",
        nargs="+",
        choices=["pgs", "newton", "cg"],
        default=["pgs", "newton", "cg"],
    )
    parser.add_argument(
        "--timesteps", nargs="+", type=float, default=[0.001, 0.0005, 0.00025]
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=["constant", "loaded", "reversal"],
        default=["constant", "loaded", "reversal"],
    )
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output", default="outputs/assembly_many_gears/gear_reference"
    )
    arguments = parser.parse_args()
    if (
        min(arguments.timesteps) <= 0
        or arguments.duration <= 0
        or arguments.repeats < 1
        or arguments.iterations < 1
    ):
        parser.error("timesteps, duration, repeats and iterations must be positive")
    run_all_benchmarks(arguments)
