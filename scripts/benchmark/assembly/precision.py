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
"""Measure geometry roundtrips and four passive precision-assembly backends.

Run: python -m scripts.benchmark.assembly.precision --configs circle usb_c m10 spur_pair
Each native cell runs in its own process. Reports retain failed and unavailable
cells; successful passive probes do not certify the unmeasured S2-S4 probes.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from embodichain.gen_sim.assembly import AssemblyAsset, build_assembly
from embodichain.gen_sim.assembly.precision import (
    PrecisionRunCfg,
    compare_usd_geometry,
    run_passive_probe,
)

__all__: list[str] = []


def run_all_benchmarks(args: argparse.Namespace) -> Path:
    """Run isolated cells and write exactly three comparison tables plus data."""
    if args.probe != "passive" and args.num_envs != [1]:
        raise ValueError(
            "Constrained probes batch one world per direction; use --directions. Use --probe passive for an environment-capacity sweep."
        )
    if args.probe != "passive" and (
        args.solver_tolerance is not None
        or args.multiccd is not None
        or args.warp_reference_initialization != "cpu_compiled"
    ):
        raise ValueError(
            "Solver tolerance and MultiCCD overrides currently apply to passive probes only."
        )
    directory = Path(args.output).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    geometry = []
    selections = []
    for name in args.configs:
        if args.clearances:
            if name.endswith(".json"):
                raise ValueError(
                    "Clearance sweeps require generator configs; manifests are immutable."
                )
            for clearance in args.clearances:
                if clearance <= 0:
                    raise ValueError(
                        "Generated clearance sweeps require positive clearances."
                    )
                selections.append(
                    (
                        f"{name}@{clearance*1e6:g}um",
                        build_assembly(name, clearance=clearance),
                    )
                )
        else:
            selections.append(
                (
                    name,
                    (
                        AssemblyAsset.load(name)
                        if name.endswith(".json")
                        else build_assembly(name)
                    ),
                )
            )
    for name, asset in selections:
        manifest = asset.save(directory / f"{asset.content_hash}.json")
        usd = asset.write_usd(directory / f"{asset.content_hash}.usda")
        geometry.append({"config": name, **compare_usd_geometry(asset, usd)})
        for backend in args.backends:
            for route in (
                args.routes if backend in {"newton", "default"} else ["direct"]
            ):
                for timestep in args.timesteps:
                    for worlds in args.num_envs if backend != "mujoco_cpu" else [1]:
                        cfg = PrecisionRunCfg(
                            backend=backend,
                            route=route,
                            num_envs=worlds,
                            duration=args.duration,
                            timestep=timestep,
                            contact_offset=args.contact_offset,
                            rest_offset=args.rest_offset,
                            solver_iterations=args.solver_iterations,
                            contact_capacity=args.contact_capacity,
                            source_contact=not args.backend_contacts,
                            solver=args.solver,
                            use_mujoco_cpu=args.use_mujoco_cpu,
                            solver_tolerance=args.solver_tolerance,
                            enable_multiccd=args.multiccd,
                            warp_reference_initialization=args.warp_reference_initialization,
                            work_dir=str(directory / "runtime"),
                        )
                        cell = directory / f"cell_{len(rows)}.json"
                        spec = directory / f"spec_{len(rows)}.json"
                        spec.write_text(
                            json.dumps(
                                {
                                    **asdict(cfg),
                                    "constrained_probe": args.probe,
                                    "num_directions": args.directions,
                                }
                            ),
                            encoding="utf-8",
                        )
                        command = [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "--cell",
                            str(manifest),
                            "--cell-config",
                            str(spec),
                            "--cell-output",
                            str(cell),
                        ]
                        try:
                            completed = subprocess.run(
                                command,
                                capture_output=True,
                                text=True,
                                timeout=args.cell_timeout,
                            )
                            (directory / f"cell_{len(rows)}.log").write_text(
                                completed.stdout + "\n" + completed.stderr,
                                encoding="utf-8",
                            )
                            if cell.exists():
                                result = json.loads(cell.read_text())
                            else:
                                result = {
                                    "backend": backend,
                                    "route": route,
                                    "status": "failed",
                                    "reason": f"Process exit {completed.returncode}: {completed.stderr[-1800:]}",
                                }
                        except subprocess.TimeoutExpired:
                            result = {
                                "backend": backend,
                                "route": route,
                                "status": "failed",
                                "reason": "Probe process timed out.",
                            }
                        result.update(
                            config_name=name,
                            requested_num_envs=worlds,
                            timestep=timestep,
                        )
                        result.setdefault("num_envs", worlds)
                        cell.write_text(
                            json.dumps(result, indent=2, allow_nan=False) + "\n"
                        )
                        result["probe"] = args.probe
                        rows.append(result)
                        print(
                            f"{name} {backend}/{route} B={worlds} dt={timestep}: {result['status']}",
                            flush=True,
                        )
    versions = {}
    for package in [
        "mujoco",
        "mujoco-warp",
        "warp-lang",
        "newton",
        "dexsim-engine",
        "torch",
        "mjfit",
    ]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "versions": versions,
        "geometry": geometry,
        "cells": rows,
        "unqualified_probes": [
            "S0 contact onset sweep",
            "S2 lateral play",
            "S3 tilt play",
            "S4 yaw play",
            "robot control",
            "heterogeneous capacity",
        ],
    }
    stem = "assembly_precision_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    raw = directory / f"{stem}.json"
    raw.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    report = directory / f"{stem}.md"
    lines = [
        "# Assembly precision measurements",
        "",
        f"Raw results: `{raw.name}`.",
        "",
        "Timing scopes and the selected probe are recorded per cell. Missing native health measurements and unselected probes remain unqualified.",
        "",
        "## Time & Memory",
        "",
        "| asset | backend/route | worlds | dt | cost_time_ms/world-step | cpu_delta_mb | gpu_delta_mb | peak_gpu_mb |",
        "|---|---|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['config_name']} | {row['backend']}/{row['route']} | {row['num_envs']} | {row['timestep']} | {row.get('step_time_ms', 'unmeasured')} | unmeasured | unmeasured | unmeasured |"
        )
    lines += [
        "",
        "## Success & Other Metrics",
        "",
        "| asset | backend/route | status | measured success | geometric error m | mass relative error | inertia relative error |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        measured = next(g for g in geometry if g["config"] == row["config_name"])
        lines.append(
            f"| {row['config_name']} | {row['backend']}/{row['route']} | {row['status']} | {row.get('metrics', {}).get('success_rate', 'unmeasured')} | {measured['geometry_error_m']:.3g} | {measured['mass_relative_error']:.3g} | {measured['inertia_relative_error']:.3g} |"
        )
    lines += [
        "",
        "## Leaderboard",
        "",
        "| rank | algorithm | qualified passive cells | attempted cells | overall_success_rate |",
        "|---:|---|---:|---:|---:|",
    ]
    algorithms = sorted({f"{r['backend']}/{r['route']}" for r in rows})
    ranked = []
    for algorithm in algorithms:
        attempts = [r for r in rows if f"{r['backend']}/{r['route']}" == algorithm]
        passed = sum(r["status"] == "passed" for r in attempts)
        ranked.append((passed / len(attempts), algorithm, passed, len(attempts)))
    for rank, (rate, algorithm, passed, attempted) in enumerate(
        sorted(ranked, reverse=True), 1
    ):
        lines.append(f"| {rank} | {algorithm} | {passed} | {attempted} | {rate:.4f} |")
    for row in rows:
        if row.get("reason"):
            lines += [
                "",
                f"- {row['config_name']} {row['backend']}/{row['route']}: {row['reason'].replace(chr(10), ' ')}",
            ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Markdown report saved: {report}")
    return report


def main() -> None:
    """Parse matrix selections or execute one isolated benchmark cell."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs", nargs="+", default=["circle", "usb_c", "m10", "spur_pair"]
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["mujoco_cpu", "mujoco_warp", "newton", "default"],
        default=["mujoco_cpu", "mujoco_warp", "newton", "default"],
    )
    parser.add_argument(
        "--routes", nargs="+", choices=["direct", "usd"], default=["direct", "usd"]
    )
    parser.add_argument("--num-envs", nargs="+", type=int, default=[1])
    parser.add_argument("--timesteps", nargs="+", type=float, default=[0.001])
    parser.add_argument("--duration", type=float, default=1.5)
    parser.add_argument(
        "--clearances",
        nargs="+",
        type=float,
        help="Generate peg/thread clearance sweep values in meters.",
    )
    parser.add_argument("--contact-offset", type=float, default=0.0)
    parser.add_argument("--rest-offset", type=float, default=0.0)
    parser.add_argument("--solver-iterations", type=int, default=100)
    parser.add_argument("--solver", choices=["pgs", "cg", "newton"])
    parser.add_argument("--use-mujoco-cpu", action="store_true")
    parser.add_argument("--solver-tolerance", type=float)
    parser.add_argument(
        "--warp-reference-initialization",
        choices=["cpu_compiled", "warp_recomputed"],
        default="cpu_compiled",
        help="Declare how the mujoco_warp reference initializes derived constants; non-default is accepted only by that backend.",
    )
    parser.add_argument(
        "--multiccd", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--contact-capacity", type=int, default=4096)
    parser.add_argument(
        "--probe", choices=["passive", "S0", "S2", "S3", "S4"], default="passive"
    )
    parser.add_argument("--directions", type=int, default=24)
    parser.add_argument("--cell-timeout", type=int, default=300)
    parser.add_argument(
        "--backend-contacts",
        action="store_true",
        help="Explicitly measure backend defaults instead of exact raw source contacts.",
    )
    parser.add_argument("--output", default="outputs/benchmarks/assembly")
    parser.add_argument("--cell", help=argparse.SUPPRESS)
    parser.add_argument("--cell-config", help=argparse.SUPPRESS)
    parser.add_argument("--cell-output", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.cell:
        parameters = json.loads(Path(args.cell_config).read_text())
        probe = parameters.pop("constrained_probe", "passive")
        directions = parameters.pop("num_directions", 24)
        asset = AssemblyAsset.load(args.cell)
        cfg = PrecisionRunCfg(**parameters)
        if probe == "passive":
            result = run_passive_probe(asset, cfg)
        else:
            from embodichain.gen_sim.assembly.probes import (
                ConstrainedProbeCfg,
                run_native_probe,
                run_reference_probe,
                run_warp_probe,
            )

            selected = ConstrainedProbeCfg(
                probe=probe,
                num_directions=directions,
                duration=cfg.duration,
                timestep=cfg.timestep,
                source_contact=cfg.source_contact,
                solver_iterations=cfg.solver_iterations,
                contact_offset=cfg.contact_offset,
                rest_offset=cfg.rest_offset,
                contact_capacity=cfg.contact_capacity,
            )
            try:
                if cfg.backend == "mujoco_cpu":
                    result = run_reference_probe(asset, selected)
                elif cfg.backend == "mujoco_warp":
                    result = run_warp_probe(asset, selected)
                elif cfg.route != "direct":
                    result = {
                        "status": "skipped",
                        "reason": "Constrained USD articulation route is not qualified.",
                    }
                else:
                    result = run_native_probe(
                        asset, selected, backend=cfg.backend, work_dir=cfg.work_dir
                    )
            except (
                ValueError,
                RuntimeError,
                ImportError,
                NotImplementedError,
            ) as error:
                result = {"status": "failed", "reason": str(error)}
            rows = result.get("rows", [])
            result.update(
                backend=cfg.backend,
                route=cfg.route,
                asset_sha256=asset.content_hash,
                metrics={
                    "success_rate": (
                        sum(bool(row.get("passed")) for row in rows) / len(rows)
                        if rows
                        else None
                    )
                },
                config=parameters,
            )
        Path(args.cell_output).write_text(
            json.dumps(result, allow_nan=False), encoding="utf-8"
        )
    else:
        run_all_benchmarks(args)


if __name__ == "__main__":
    main()
