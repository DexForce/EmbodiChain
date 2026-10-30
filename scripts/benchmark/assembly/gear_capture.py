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
"""Capture complete live gear properties before and after one physical step.

Run only after other native jobs finish, with the same dependency overlays::

    python -m scripts.benchmark.assembly.gear_capture NATIVE_ALIGNMENT_DIRECTORY

One isolated scene is prepared for each distinct asset/profile in completed
results. This checks static live-model representation, not ratio accuracy or
long-run health. Original benchmark cells remain immutable.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import mujoco
import numpy as np

from scripts.benchmark.assembly.gear_postvalidate import (
    _compare_models,
    _global_options,
)
from scripts.benchmark.assembly.gear_reference import _load_manifest

__all__: list[str] = []

_FIELDS = (
    "body_pos",
    "body_quat",
    "body_ipos",
    "body_iquat",
    "body_mass",
    "body_inertia",
    "geom_bodyid",
    "geom_type",
    "geom_dataid",
    "geom_size",
    "geom_pos",
    "geom_quat",
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
    "mesh_vertadr",
    "mesh_vertnum",
    "mesh_vert",
    "jnt_bodyid",
    "jnt_type",
    "jnt_limited",
    "jnt_axis",
    "jnt_pos",
    "jnt_dofadr",
    "dof_damping",
    "dof_armature",
    "dof_frictionloss",
)


def _capture_cell(spec: dict[str, Any], directory: Path) -> dict[str, Any]:
    """Capture actual live arrays without prescribing any follower motion."""
    from embodichain.toolkits.assembly import AssemblyAsset
    from embodichain_benchmarks.assembly.precision import (
        PrecisionRunCfg,
        _run_embodichain,
    )

    asset_dict, digest = _load_manifest(Path(spec["manifest"]))
    asset = AssemblyAsset.load(spec["manifest"])
    source = mujoco.MjModel.from_xml_string(asset_dict["source_mjcf"])
    cfg = PrecisionRunCfg(
        backend="newton",
        route="direct",
        duration=0.001,
        timestep=0.001,
        solver="pgs" if spec["profile"] == "cpu_pgs" else "cg",
        use_mujoco_cpu=spec["profile"] == "cpu_pgs",
        work_dir=str(directory / "runtime"),
    )
    stages = []
    dependency = None

    def observe(_sim: Any, objects: dict[str, Any], step: int) -> None:
        nonlocal dependency
        solver = objects["gear_a"]._entities[0]._physics_binding._runtime.solver
        solver_path = Path(inspect.getfile(type(solver)))
        dependency = {
            "solver_source": str(solver_path),
            "sha256": hashlib.sha256(solver_path.read_bytes()).hexdigest(),
        }
        live = solver.mj_model if cfg.use_mujoco_cpu else solver.mjw_model
        host = solver.mj_model
        mirror = copy.copy(host)
        arrays = {}
        missing, invalid = [], []
        for name in _FIELDS:
            if not hasattr(live, name):
                missing.append(name)
                continue
            value = getattr(live, name)
            array = np.asarray(value.numpy() if hasattr(value, "numpy") else value)
            target = getattr(mirror, name)
            if array.shape == (1, *target.shape):
                array = array[0]
            arrays[name] = array.copy()
            if array.shape != target.shape:
                invalid.append(
                    {
                        "field": name,
                        "live_shape": list(array.shape),
                        "host_shape": list(target.shape),
                    }
                )
                continue
            target[:] = array
        arrays_path = directory / f"live_{step}.npz"
        np.savez_compressed(arrays_path, **arrays)
        model_path = directory / f"host_{step}.mjb"
        mujoco.mj_saveModel(host, str(model_path), None)
        options = {}
        for name in (
            "solver",
            "integrator",
            "timestep",
            "iterations",
            "tolerance",
            "disableflags",
            "enableflags",
            "cone",
            "jacobian",
            "ls_iterations",
            "ls_tolerance",
            "impratio",
            "impratio_invsqrt",
            "ccd_tolerance",
            "ccd_iterations",
        ):
            if not hasattr(live.opt, name):
                continue
            value = getattr(live.opt, name)
            options[name] = np.asarray(
                value.numpy() if hasattr(value, "numpy") else value
            ).tolist()
        verification = _compare_models(source, mirror)
        option_checks = {
            "solver": int(np.asarray(options["solver"]).reshape(-1)[0])
            == (0 if cfg.use_mujoco_cpu else 1),
            "integrator": int(np.asarray(options["integrator"]).reshape(-1)[0])
            == int(source.opt.integrator),
            "iterations": int(np.asarray(options["iterations"]).reshape(-1)[0])
            == cfg.solver_iterations,
            "timestep": step == 0
            or abs(float(np.asarray(options["timestep"]).reshape(-1)[0]) - cfg.timestep)
            <= 1e-8,
        }
        stages.append(
            {
                "step": step,
                "arrays": arrays_path.name,
                "arrays_sha256": hashlib.sha256(arrays_path.read_bytes()).hexdigest(),
                "host_model": model_path.name,
                "captured_fields": sorted(arrays),
                "missing_fields": missing,
                "invalid_shapes": invalid,
                "active_options": options,
                "option_checks": option_checks,
                "global_options": _global_options(source, host, options),
                "verification": verification,
                "passed": bool(
                    not missing
                    and not invalid
                    and verification["passed"]
                    and all(option_checks.values())
                ),
                "quaternion_order": "wxyz, including MuJoCo Warp's raw wp.quat storage",
            }
        )

    started = time.perf_counter()
    native = _run_embodichain(asset, cfg, on_step=observe)
    return {
        **spec,
        "asset_sha256": digest,
        "scope": "Complete live static-model properties, initial and after one 1ms physical step; no ratio or long-run acceptance.",
        "passed": bool(
            len(stages) == 2
            and all(s["passed"] for s in stages)
            and native["metrics"]["finite"]
        ),
        "formal_certification": False,
        "stages": stages,
        "dependency": dependency,
        "native_runner_result": native,
        "capture_wall_ms": (time.perf_counter() - started) * 1000,
        "mujoco": mujoco.__version__,
    }


def _report(output: Path, rows: list[dict[str, Any]]) -> None:
    """Write raw evidence plus three tables without a performance claim."""
    (output / "capture.json").write_text(
        json.dumps(
            {"cells": rows, "formal_certification": False}, indent=2, allow_nan=False
        )
        + "\n"
    )
    lines = [
        "# Complete live gear property capture",
        "",
        "## Time & Memory",
        "",
        "| asset | profile | setup/capture ms | CPU peak MB | GPU peak MB |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {Path(r['manifest']).stem} | {r['profile']} | {r.get('capture_wall_ms', 'n/a')} | unmeasured | unmeasured |"
        for r in rows
    ]
    lines += [
        "",
        "## Success & Other Metrics",
        "",
        "| asset | profile | complete live property capture passed |",
        "| --- | --- | --- |",
    ]
    lines += [
        f"| {Path(r['manifest']).stem} | {r['profile']} | {r.get('passed', False)} |"
        for r in rows
    ]
    lines += [
        "",
        "## Leaderboard",
        "",
        "| rank | profile | property capture success rate | cells |",
        "| --- | --- | --- | --- |",
    ]
    totals = []
    for profile in sorted({r["profile"] for r in rows}):
        subset = [r for r in rows if r["profile"] == profile]
        totals.append(
            (
                profile,
                sum(r.get("passed", False) for r in subset) / len(subset),
                len(subset),
            )
        )
    lines += [
        f"| {i} | {p} | {rate:.6f} | {n} |"
        for i, (p, rate, n) in enumerate(sorted(totals, key=lambda t: -t[1]), 1)
    ]
    lines += [
        "",
        "This separate six-scene check captures actual CPU/Warp body principal orientation, COM, full inertia inputs, collision geometry/transforms, masks, condim, priority and raw contacts before and after one physical step. CPU mirrors are constructed from every captured field for independent MuJoCo comparison; missing/invalid fields fail the capture. It does not retroactively observe missing fields in the original 30 trajectories or certify their lifetime. No follower pose or velocity is prescribed. Timing includes cold setup, transfers and inspection and is not throughput. Memory is unmeasured. Original benchmark data are unchanged. No formal certification.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))


def main() -> None:
    """Run explicitly requested isolated capture cells sequentially."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="?")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cell", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.cell:
        spec = json.loads(args.cell.read_text())
        result = _capture_cell(spec, args.cell.parent)
        (args.cell.parent / "result.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
        return
    if args.input is None:
        parser.error("A completed native alignment directory is required.")
    source = args.input.resolve()
    output = (
        args.output or source.with_name(source.name + "_live_properties")
    ).resolve()
    if output == source:
        parser.error("Capture output must be separate from the original benchmark.")
    output.mkdir(parents=True, exist_ok=True)
    specs = {}
    for path in sorted(source.glob("cell_*/result.json")):
        row = json.loads(path.read_text())
        if row.get("profile") in {"cpu_pgs", "warp_cg"} and "manifest" in row:
            specs[(row["manifest"], row["profile"])] = {
                "manifest": row["manifest"],
                "profile": row["profile"],
            }
    rows = []
    for index, spec in enumerate(specs.values()):
        directory = output / f"cell_{index:03d}"
        directory.mkdir(exist_ok=True)
        path = directory / "spec.json"
        path.write_text(json.dumps(spec, indent=2) + "\n")
        (directory / "result.json").unlink(missing_ok=True)
        print(
            f"Capturing {Path(spec['manifest']).stem} / {spec['profile']}", flush=True
        )
        try:
            with (directory / "process.log").open("w") as log:
                process = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "scripts.benchmark.assembly.gear_capture",
                        "--cell",
                        str(path),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=180,
                )
        except subprocess.TimeoutExpired:
            rows.append(
                {
                    **spec,
                    "passed": False,
                    "reason": "Capture subprocess exceeded 180 seconds.",
                }
            )
            _report(output, rows)
            continue
        result_path = directory / "result.json"
        result = (
            json.loads(result_path.read_text())
            if process.returncode == 0 and result_path.exists()
            else {
                **spec,
                "passed": False,
                "reason": f"Capture process exited {process.returncode}; inspect {directory / 'process.log'}.",
            }
        )
        rows.append(result)
        _report(output, rows)
        print(f"Capture passed: {result['passed']}", flush=True)
    print(f"Capture report: {output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
