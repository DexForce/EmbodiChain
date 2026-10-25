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
"""Diagnose native hinged-gear accuracy against immutable source references.

Run with the documented assembly dependency overlays::

    python -m scripts.benchmark.assembly.gear_native --manifests assets/*.json

Each native cell runs in an isolated subprocess. By default constant-drive
cells cover all timesteps and load/reversal cells use --transient-timestep.
Private SDK runtime inspection is confined to this diagnostic benchmark.
Its health evidence does not change the Scene facade's unverified status.
Timing includes per-step state/health collection and is not a performance claim.
"""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import hashlib
import inspect
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from typing import Any

import mujoco
import numpy as np

from scripts.benchmark.assembly.gear_reference import (
    _backlash_model,
    _load_manifest,
    _metrics,
    _trajectory,
)

__all__: list[str] = []


def _array(value: Any) -> np.ndarray:
    return np.asarray(value.numpy() if hasattr(value, "numpy") else value)


def _model_properties(model: mujoco.MjModel) -> dict[str, Any]:
    """Snapshot the actual compiled model, independent of XML authoring state."""
    return {
        "solver": int(model.opt.solver),
        "disableflags": int(model.opt.disableflags),
        "integrator": int(model.opt.integrator),
        "timestep": float(model.opt.timestep),
        "ngeom": int(model.ngeom),
        "nbody": int(model.nbody),
        "njnt": int(model.njnt),
        "body_names": [model.body(i).name for i in range(model.nbody)],
        "geom_names": [model.geom(i).name for i in range(model.ngeom)],
        **{
            name: getattr(model, name).tolist()
            for name in (
                "body_mass",
                "body_inertia",
                "body_ipos",
                "body_iquat",
                "geom_bodyid",
                "geom_contype",
                "geom_conaffinity",
                "geom_condim",
                "geom_priority",
                "geom_friction",
                "geom_solref",
                "geom_solimp",
                "geom_solmix",
                "geom_margin",
                "geom_gap",
                "dof_damping",
                "dof_armature",
                "dof_frictionloss",
                "jnt_bodyid",
                "jnt_qposadr",
                "jnt_dofadr",
            )
        },
    }


def _active_properties(solver: Any) -> dict[str, Any]:
    """Record live solver buffers, since host MJB options can lag Warp state."""
    model = solver.mj_model if solver.use_mujoco_cpu else solver.mjw_model
    names = (
        "body_mass",
        "body_inertia",
        "geom_friction",
        "geom_solref",
        "geom_solimp",
        "geom_solmix",
        "geom_gap",
        "geom_margin",
        "dof_damping",
        "dof_armature",
        "dof_frictionloss",
    )
    options = (
        "solver",
        "integrator",
        "timestep",
        "iterations",
        "tolerance",
        "disableflags",
        "enableflags",
    )
    return {
        "model_buffer_source": "mj_model" if solver.use_mujoco_cpu else "mjw_model",
        "properties": {
            name: _array(getattr(model, name)).tolist()
            for name in names
            if hasattr(model, name)
        },
        "options": {
            name: _array(getattr(model.opt, name)).tolist()
            for name in options
            if hasattr(model.opt, name)
        },
    }


def _physics_errors(source: mujoco.MjModel, actual: mujoco.MjModel) -> dict[str, float]:
    errors = {
        "mass_relative_error": 0.0,
        "principal_inertia_relative_error": 0.0,
        "damping_absolute_error": 0.0,
        "armature_absolute_error": 0.0,
    }
    for name in ("gear_a", "gear_b"):
        source_id = source.body(name).id
        actual_id = next(
            i for i in range(actual.nbody) if actual.body(i).name.endswith(name)
        )
        errors["mass_relative_error"] = max(
            errors["mass_relative_error"],
            float(abs(actual.body_mass[actual_id] / source.body_mass[source_id] - 1)),
        )
        expected = np.sort(source.body_inertia[source_id])
        got = np.sort(actual.body_inertia[actual_id])
        errors["principal_inertia_relative_error"] = max(
            errors["principal_inertia_relative_error"],
            float(np.max(np.abs(got / expected - 1))),
        )
        sj = int(np.flatnonzero(source.jnt_bodyid == source_id)[0])
        aj = int(np.flatnonzero(actual.jnt_bodyid == actual_id)[0])
        sd, ad = int(source.jnt_dofadr[sj]), int(actual.jnt_dofadr[aj])
        errors["damping_absolute_error"] = max(
            errors["damping_absolute_error"],
            float(abs(actual.dof_damping[ad] - source.dof_damping[sd])),
        )
        errors["armature_absolute_error"] = max(
            errors["armature_absolute_error"],
            float(abs(actual.dof_armature[ad] - source.dof_armature[sd])),
        )
    return errors


def _run_cell(spec: dict[str, Any], directory: Path) -> dict[str, Any]:
    """Run one actual native scene with read-only per-step diagnostics."""
    import torch
    from embodichain.gen_sim.assembly import AssemblyAsset
    from embodichain.gen_sim.assembly.precision import (
        PrecisionRunCfg,
        _run_embodichain,
        run_passive_probe,
    )

    asset = AssemblyAsset.load(spec["manifest"])
    _, digest = _load_manifest(Path(spec["manifest"]))
    cfg = PrecisionRunCfg(
        backend="newton",
        route="direct",
        duration=spec["duration"],
        timestep=spec["timestep"],
        solver=spec["solver"],
        use_mujoco_cpu=spec["use_mujoco_cpu"],
        solver_tolerance=spec.get("solver_tolerance"),
        enable_multiccd=spec.get("enable_multiccd"),
        solver_iterations=spec["iterations"],
        contact_capacity=spec["contact_capacity"],
        work_dir=str(directory / "runtime"),
    )
    steps = round(cfg.duration / cfg.timestep)
    q = np.full((steps + 1, 2), np.nan, dtype=np.float64)
    v = np.full_like(q, np.nan)
    native_times = np.full(steps + 1, np.nan)
    contacts = np.full(steps + 1, -1, dtype=np.int32)
    warning_trace = np.zeros(steps + 1, dtype=np.int32)
    overflow_trace = np.zeros(steps + 1, dtype=np.int32)
    health: dict[str, Any] = {
        "private_path": "objects['gear_a']._entities[0]._physics_binding._runtime.solver",
        "warning_count": 0,
        "overflow_seen": False,
        "overflow_observation_available": False,
        "runtime_time_available": False,
        "time_reset_count": 0,
        "observed_steps": 0,
        "finite": True,
    }
    previous_warnings = None
    last_time = None
    solver = None
    initial_props = None
    final_props = None
    active_initial = None
    active_final = None
    dependency = None
    snapshot_steps = []
    driver_torque = next(b.joint.motor_gear for b in asset.bodies if b.name == "gear_a")

    def before_step(sim: Any, objects: dict[str, Any], step: int) -> None:
        reverse = spec["scenario"] == "reversal" and step >= steps // 2
        objects["gear_a"].set_qf(
            torch.full(
                (1, 1), -driver_torque if reverse else driver_torque, device=sim.device
            )
        )
        # Only external torques are prescribed. No follower pose/velocity or
        # ideal gear relation is ever written.
        objects["gear_b"].set_qf(
            torch.full(
                (1, 1),
                0.002 if spec["scenario"] == "loaded" else 0.0,
                device=sim.device,
            )
        )

    def observe(sim: Any, objects: dict[str, Any], step: int) -> None:
        nonlocal solver, previous_warnings, last_time, initial_props, final_props
        nonlocal active_initial, active_final, dependency
        if solver is None:
            solver = objects["gear_a"]._entities[0]._physics_binding._runtime.solver
            path = Path(inspect.getfile(type(solver)))
            dependency = {
                "solver_source": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "model_device": str(solver.model.device),
                "gpu": torch.cuda.get_device_name(0),
            }
        q[step] = [
            objects[name].get_qpos().detach().cpu().numpy().reshape(-1)[0]
            for name in ("gear_a", "gear_b")
        ]
        v[step] = [
            objects[name].get_qvel().detach().cpu().numpy().reshape(-1)[0]
            for name in ("gear_a", "gear_b")
        ]
        health["finite"] = bool(
            health["finite"]
            and np.isfinite(q[step]).all()
            and np.isfinite(v[step]).all()
        )
        data = solver.mj_data if cfg.use_mujoco_cpu else solver.mjw_data
        if cfg.use_mujoco_cpu:
            values = np.asarray([int(w.number) for w in data.warning], dtype=np.int64)
            increments = (
                values
                if previous_warnings is None
                else np.where(
                    values >= previous_warnings, values - previous_warnings, values
                )
            )
            health["warning_count"] += int(increments.sum())
            previous_warnings = values.copy()
            warning_trace[step] = int(values.sum())
            health["overflow_observation_available"] = True
            health["overflow_kind"] = (
                "all native MuJoCo warning counters, sampled every physical step"
            )
        elif hasattr(data, "overflow"):
            values = _array(data.overflow)
            overflow_trace[step] = int(np.count_nonzero(values))
            health["overflow_seen"] = bool(
                health["overflow_seen"] or overflow_trace[step]
            )
            health["overflow_observation_available"] = True
            health["overflow_kind"] = (
                "MuJoCo Warp overflow flags, sampled every physical step"
            )
        for field in ("ncon", "nacon"):
            if hasattr(data, field):
                contacts[step] = int(_array(getattr(data, field)).sum())
                break
        if hasattr(data, "time"):
            current = float(_array(data.time).reshape(-1)[0])
            native_times[step] = current
            health["runtime_time_available"] = True
            if last_time is not None and current < last_time - 1e-8:
                health["time_reset_count"] += 1
            last_time = current
        health["observed_steps"] += 1
        if step in (0, steps):
            props = _model_properties(solver.mj_model)
            path = directory / f"native_{step}.mjb"
            mujoco.mj_saveModel(solver.mj_model, str(path), None)
            snapshot_steps.append({"step": step, "file": path.name})
            if step == 0:
                initial_props = props
                active_initial = _active_properties(solver)
            else:
                final_props = props
                active_final = _active_properties(solver)

    started = time.perf_counter()
    result = _run_embodichain(asset, cfg, before_step=before_step, on_step=observe)
    wall_ms = (time.perf_counter() - started) * 1000
    np.savez_compressed(
        directory / "trace.npz",
        qpos=q,
        qvel=v,
        contacts=contacts,
        warning_count=warning_trace,
        overflow_flags_count=overflow_trace,
        actual_runtime_time=native_times,
        timestep=cfg.timestep,
    )
    metrics = _metrics(q, v, asset.config["teeth"], spec["scenario"])
    health["maximum_speed_rad_s"] = float(np.max(np.abs(v)))
    health["speed_limit_rad_s"] = spec["speed_limit"]
    health["all_steps_observed"] = health["observed_steps"] == steps + 1
    health["peak_contacts"] = int(contacts.max())
    health["contact_count_observation_available"] = bool(np.any(contacts >= 0))
    health["overflow_steps"] = np.flatnonzero(overflow_trace).tolist()
    health["time_initial_final"] = [float(native_times[0]), float(native_times[-1])]
    actual = mujoco.MjModel.from_binary_path(str(directory / "native_0.mjb"))
    source = mujoco.MjModel.from_xml_string(asset.source_mjcf)
    physics_errors = _physics_errors(source, actual)
    # Static inspection uses public CPU collision queries on an independent
    # compiled-model copy; it does not alter the live native scene.
    scan_model = copy.copy(actual)
    scan_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_MIDPHASE
    static = {
        "compiled": _backlash_model(scan_model, asset.config),
        "source": _backlash_model(source, asset.config),
        "scope": "public MuJoCo CPU geometry queries on actual compiled model; not a Warp contact-runtime oracle",
    }
    static["tangential_play_error_m"] = (
        abs(
            static["compiled"]["tangential_play_m"]
            - static["source"]["tangential_play_m"]
        )
        if static["compiled"]["status"] == static["source"]["status"] == "bracketed"
        else None
    )
    references = []
    for name in dict.fromkeys(["pgs", spec["solver"]]):
        model = mujoco.MjModel.from_xml_string(asset.source_mjcf)
        model.opt.timestep = cfg.timestep
        model.opt.iterations = cfg.solver_iterations
        if cfg.solver_tolerance is not None:
            model.opt.tolerance = cfg.solver_tolerance
        if cfg.enable_multiccd is not None:
            if cfg.enable_multiccd:
                model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
            else:
                model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_MULTICCD
        model.opt.solver = getattr(mujoco.mjtSolver, f"mjSOL_{name.upper()}")
        rq, rv, rc, elapsed, warnings = _trajectory(
            model, cfg.duration, spec["scenario"]
        )
        rm = _metrics(rq, rv, asset.config["teeth"], spec["scenario"])
        trace = f"reference_{name}.npz"
        np.savez_compressed(
            directory / trace, qpos=rq, qvel=rv, contacts=rc, timestep=cfg.timestep
        )
        references.append(
            {
                "solver": name,
                "mujoco": mujoco.__version__,
                "metrics": rm,
                "warning_count": warnings,
                "actual_tolerance": float(model.opt.tolerance),
                "actual_multiccd": not bool(
                    model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_MULTICCD
                ),
                "trace": trace,
                "diagnostic_wall_ms": elapsed * 1000,
            }
        )
    independent_warp: dict[str, Any] = {"status": "not_requested"}
    if spec.get("warp_reference", False):
        if spec["profile"] != "warp_cg":
            independent_warp = {
                "status": "not_run",
                "reason": "Independent Warp endpoint reference applies to the warp_cg profile only.",
            }
        elif spec["scenario"] != "constant":
            independent_warp = {
                "status": "not_run",
                "reason": "Passive Warp reference does not prescribe the loaded or reversal scenarios.",
            }
        else:
            # The native world has already been destroyed by _run_embodichain.
            # This is the public source-MJCF Warp path, with no native scene.
            reference_cfg = PrecisionRunCfg(
                backend="mujoco_warp",
                route="direct",
                duration=cfg.duration,
                timestep=cfg.timestep,
                solver="cg",
                solver_tolerance=cfg.solver_tolerance,
                enable_multiccd=cfg.enable_multiccd,
                warp_reference_initialization=spec.get(
                    "warp_reference_initialization", "cpu_compiled"
                ),
                solver_iterations=cfg.solver_iterations,
                contact_capacity=cfg.contact_capacity,
            )
            started_reference = time.perf_counter()
            independent_warp = run_passive_probe(asset, reference_cfg)
            independent_warp["end_to_end_wall_ms"] = (
                time.perf_counter() - started_reference
            ) * 1000
            independent_warp["scope"] = (
                "independent source-MJCF MuJoCo Warp endpoint after native world cleanup"
            )
    passed = bool(
        metrics["finite"]
        and metrics["sufficient_steady_travel"]
        and metrics["steady_ratio_relative_error"] is not None
        and metrics["steady_ratio_relative_error"] <= 0.005
        and health["finite"]
        and not health["warning_count"]
        and not health["overflow_seen"]
        and health["overflow_observation_available"]
        and not health["time_reset_count"]
        and health["all_steps_observed"]
        and health["maximum_speed_rad_s"] <= spec["speed_limit"]
        and physics_errors["mass_relative_error"] <= 1e-5
        and physics_errors["principal_inertia_relative_error"] <= 1e-5
        and physics_errors["damping_absolute_error"] <= 1e-8
        and physics_errors["armature_absolute_error"] <= 1e-10
        and static["tangential_play_error_m"] is not None
        and static["tangential_play_error_m"] <= 1e-6
    )
    return {
        **spec,
        "asset_sha256": digest,
        "diagnostic_status": "passed" if passed else "failed",
        "passed": passed,
        "formal_certification": False,
        "native_runner_result": result,
        "metrics": metrics,
        "health": health,
        "physics_errors": physics_errors,
        "static_geometry": static,
        "initial_model_properties": initial_props,
        "final_model_properties": final_props,
        "initial_active_properties": active_initial,
        "final_active_properties": active_final,
        "dependency": dependency,
        "model_snapshot_scope": "host MuJoCo model geometry/properties; live Warp options/buffers recorded separately",
        "compiled_models": snapshot_steps,
        "references": references,
        "independent_warp_reference": independent_warp,
        "trace": "trace.npz",
        "cost_time_ms": wall_ms,
        "timing_scope": "one end-to-end cold native scene with per-step state/health diagnostics, model snapshots, and cleanup; not a performance benchmark",
        "mujoco": mujoco.__version__,
    }


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _safe(v) for key, v in value.items()}
    if isinstance(value, list):
        return [_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_report(
    directory: Path, rows: list[dict[str, Any]], metadata: dict[str, Any]
) -> Path:
    lines = [
        "# Native gear diagnostic alignment",
        "",
        json.dumps(metadata),
        "",
        "## Time & Memory",
        "",
        "| config | profile | dt_ms | scenario | cost_time_ms | cpu_delta_mb | gpu_delta_mb | peak_gpu_mb |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['config']} | {r['profile']} | {r['timestep']*1000:g} | {r['scenario']} | {r.get('cost_time_ms','unavailable')} | unmeasured | unmeasured | unmeasured |"
        )
    lines += [
        "",
        "## Success & Other Metrics",
        "",
        "| config | profile | dt_ms | scenario | success_rate | ratio error % | health | backlash error um | status |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        error = r.get("metrics", {}).get("steady_ratio_relative_error")
        backlash = r.get("static_geometry", {}).get("tangential_play_error_m")
        h = r.get("health", {})
        status = f"warnings={h.get('warning_count','?')}, overflow={h.get('overflow_seen','?')}, resets={h.get('time_reset_count','?')}"
        lines.append(
            f"| {r['config']} | {r['profile']} | {r['timestep']*1000:g} | {r['scenario']} | {int(r.get('passed',False))} | {error*100 if error is not None else 'n/a'} | {status} | {backlash*1e6 if backlash is not None else 'n/a'} | {r.get('diagnostic_status','failed')} |"
        )
    reference_rows = {}
    for row in rows:
        for reference in row.get("references", []):
            key = (
                row["asset_sha256"],
                row["timestep"],
                row["duration"],
                row["scenario"],
                reference["solver"],
            )
            value = reference["metrics"]
            ok = bool(
                value["finite"]
                and value["sufficient_steady_travel"]
                and value["steady_ratio_relative_error"] is not None
                and value["steady_ratio_relative_error"] <= 0.005
                and not reference["warning_count"]
            )
            reference_rows[key] = (f"source_cpu_{reference['solver']}", ok)
        warp_reference = row.get("independent_warp_reference", {})
        if warp_reference.get("status") not in {None, "not_requested", "not_run"}:
            key = (
                row.get("asset_sha256"),
                row["timestep"],
                row["duration"],
                row["scenario"],
                "source_warp_cg",
            )
            reference_rows[key] = (
                "source_warp_cg",
                warp_reference.get("status") == "passed",
            )
    lines += [
        "",
        "## Leaderboard",
        "",
        "| rank | algorithm | overall_success_rate | cells |",
        "| --- | --- | --- | --- |",
    ]
    totals = []
    for profile in sorted({r["profile"] for r in rows}):
        matches = [r for r in rows if r["profile"] == profile]
        totals.append(
            (
                profile,
                sum(r.get("passed", False) for r in matches) / len(matches),
                len(matches),
            )
        )
    for profile in sorted({name for name, _ in reference_rows.values()}):
        outcomes = [
            passed for name, passed in reference_rows.values() if name == profile
        ]
        totals.append((profile, sum(outcomes) / len(outcomes), len(outcomes)))
    for rank, (profile, success, count) in enumerate(
        sorted(totals, key=lambda v: -v[1]), 1
    ):
        lines.append(f"| {rank} | {profile} | {success:.6f} | {count} |")
    lines += [
        "",
        "Diagnostic acceptance requires finite full traces, no observed warnings/overflows/resets, <=0.5% steady ratio error with sufficient travel, matching authored mass/inertia/damping/armature and <=1um source-to-compiled static backlash difference. These are diagnostic passes through private SDK inspection; the public Scene facade remains unverified. Every physical step is observed, including transient failures. Timing includes inspection and setup and must not be used as throughput. Memory is unmeasured. Initial/final MJB snapshots come from the actual solver's mj_model using mj_saveModel, not process-global mj_saveLastXML. Static geometry probes use independent CPU model copies. No follower motion or ideal gear constraint is prescribed. Leaderboard source CPU references are deduplicated by asset/dt/duration/scenario/solver and evaluated by ratio/travel/finite/warning criteria; native profiles additionally require the integration checks above. Optional source Warp references use the public endpoint/overflow acceptance and are separately ranked; their actual effective options are recorded, including backend tolerance clamps.",
        "",
    ]
    path = directory / "report.md"
    path.write_text("\n".join(lines))
    return path


def run_all_benchmarks(args: argparse.Namespace) -> Path:
    """Run isolated native cells, retaining failures and subprocess diagnostics."""
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for manifest in args.manifests:
        _load_manifest(Path(manifest))
        for profile in args.profiles:
            for scenario in args.scenarios:
                timesteps = (
                    args.timesteps
                    if scenario == "constant" or args.full_scenario_sweep
                    else [args.transient_timestep]
                )
                for dt in timesteps:
                    directory = output / f"cell_{len(rows):03d}"
                    directory.mkdir(exist_ok=True)
                    spec = {
                        "config": Path(manifest).stem,
                        "manifest": str(Path(manifest).resolve()),
                        "profile": profile,
                        "solver": "pgs" if profile == "cpu_pgs" else "cg",
                        "use_mujoco_cpu": profile == "cpu_pgs",
                        "solver_tolerance": args.solver_tolerance,
                        "enable_multiccd": args.multiccd,
                        "warp_reference": args.warp_reference,
                        "warp_reference_initialization": args.warp_reference_initialization,
                        "timestep": dt,
                        "duration": args.duration,
                        "scenario": scenario,
                        "iterations": args.iterations,
                        "contact_capacity": args.contact_capacity,
                        "speed_limit": args.speed_limit,
                    }
                    spec_path = directory / "spec.json"
                    spec_path.write_text(json.dumps(spec, indent=2) + "\n")
                    (directory / "result.json").unlink(missing_ok=True)
                    started = time.perf_counter()
                    command = [
                        sys.executable,
                        "-m",
                        "scripts.benchmark.assembly.gear_native",
                        "--cell",
                        str(spec_path),
                    ]
                    try:
                        completed = subprocess.run(
                            command,
                            capture_output=True,
                            text=True,
                            timeout=args.cell_timeout,
                        )
                        (directory / "stdout.log").write_text(completed.stdout)
                        (directory / "stderr.log").write_text(completed.stderr)
                        result_path = directory / "result.json"
                        row = (
                            json.loads(result_path.read_text())
                            if result_path.exists()
                            else {
                                **spec,
                                "passed": False,
                                "diagnostic_status": "failed",
                                "reason": f"Process exit {completed.returncode}: {completed.stderr[-1500:]}",
                            }
                        )
                        if completed.returncode:
                            row.update(
                                passed=False,
                                diagnostic_status="failed",
                                process_exit_code=completed.returncode,
                            )
                    except subprocess.TimeoutExpired as exc:
                        row = {
                            **spec,
                            "passed": False,
                            "diagnostic_status": "timeout",
                            "reason": str(exc),
                        }
                    row["subprocess_wall_ms"] = (time.perf_counter() - started) * 1000
                    row["cell_directory"] = directory.name
                    rows.append(row)
                    (output / "raw.json").write_text(
                        json.dumps(_safe({"rows": rows}), indent=2, allow_nan=False)
                        + "\n"
                    )
                    print(
                        f"{spec['config']} {profile} {dt:g} {scenario}: {row['diagnostic_status']}",
                        flush=True,
                    )
    versions = {}
    for name in (
        "dexsim-engine",
        "newton",
        "mujoco",
        "mujoco-warp",
        "warp-lang",
        "torch",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    metadata = {
        "versions": versions,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dtype": "Newton state FP32; native CPU solve FP64 or Warp FP32",
        "num_envs": 1,
        "duration": args.duration,
        "requested_solver_tolerance": args.solver_tolerance,
        "requested_enable_multiccd": args.multiccd,
        "requested_warp_reference": args.warp_reference,
        "warp_reference_initialization": args.warp_reference_initialization,
        "repeats": 1,
        "scope": "accuracy diagnostics, not timed throughput",
        "formal_certification": False,
    }
    (output / "raw.json").write_text(
        json.dumps(
            _safe({"metadata": metadata, "rows": rows}), indent=2, allow_nan=False
        )
        + "\n"
    )
    report = _write_report(output, rows, metadata)
    print(f"Markdown report saved: {report}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", nargs="+")
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=["cpu_pgs", "warp_cg"],
        default=["cpu_pgs", "warp_cg"],
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
    parser.add_argument("--transient-timestep", type=float, default=0.001)
    parser.add_argument("--full-scenario-sweep", action="store_true")
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--solver-tolerance", type=float)
    parser.add_argument(
        "--warp-reference-initialization",
        choices=["cpu_compiled", "warp_recomputed"],
        default="cpu_compiled",
        help="Constants protocol for the optional independent Warp reference only; does not alter native initialization.",
    )
    parser.add_argument(
        "--warp-reference",
        action="store_true",
        help="After native cleanup, run an independent source Warp CG endpoint reference for constant-drive warp_cg cells.",
    )
    parser.add_argument(
        "--multiccd", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--contact-capacity", type=int, default=4096)
    parser.add_argument("--speed-limit", type=float, default=1000.0)
    parser.add_argument("--cell-timeout", type=float, default=300.0)
    parser.add_argument("--output", default="outputs/assembly_many_gears/gear_native")
    parser.add_argument("--cell", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.cell:
        directory = args.cell.resolve().parent
        spec = json.loads(args.cell.read_text())
        try:
            result = _run_cell(spec, directory)
        except Exception as exc:
            result = {
                **spec,
                "passed": False,
                "diagnostic_status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        (directory / "result.json").write_text(
            json.dumps(_safe(result), indent=2, allow_nan=False) + "\n"
        )
    else:
        if not args.manifests:
            parser.error("--manifests is required")
        if (
            args.warp_reference_initialization != "cpu_compiled"
            and not args.warp_reference
        ):
            parser.error("--warp-reference-initialization requires --warp-reference.")
        if args.solver_tolerance is not None and (
            not math.isfinite(args.solver_tolerance) or args.solver_tolerance < 0
        ):
            parser.error("Solver tolerance must be finite and non-negative.")
        if (
            min(
                [
                    *args.timesteps,
                    args.transient_timestep,
                    args.duration,
                    args.cell_timeout,
                    args.speed_limit,
                ]
            )
            <= 0
            or min(args.iterations, args.contact_capacity) < 1
        ):
            parser.error("numeric settings must be positive")
        run_all_benchmarks(args)
