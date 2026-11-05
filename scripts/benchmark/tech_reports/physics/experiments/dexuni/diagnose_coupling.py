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
"""Profile unchanged S03/S04 workers and reversible instance-local optimizations.

Run: .venv/bin/python scripts/benchmark/tech_reports/physics/experiments/dexuni/diagnose_coupling.py --task s04 --platform embodichain --case baseline
The installed DexSim/Newton sources and accepted measurements are never edited.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import replace
import importlib
import inspect
import types
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO))
TECH = REPO / "scripts/benchmark/tech_reports"
ROOTS = {
    "s03": TECH.parents[2]
    / "outputs/benchmarks/tech_reports/dexuni/s03_cloth_corner_lift",
    "s04": TECH.parents[2]
    / "outputs/benchmarks/tech_reports/dexuni/s04_rigid_soft_squeeze_grasp",
}
__all__ = ["main"]


def _write(path: Path, value: object) -> None:
    """Persist diagnostic metadata with native scalar support."""
    path.write_text(
        json.dumps(
            value,
            indent=2,
            default=lambda x: x.item() if isinstance(x, np.generic) else str(x),
        )
        + "\n"
    )


def main() -> None:
    """Run the original task with scoped instrumentation and unchanged quality settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=ROOTS, required=True)
    parser.add_argument(
        "--platform", choices=("embodichain", "isaaclab"), required=True
    )
    parser.add_argument(
        "--case",
        choices=(
            "baseline",
            "rne_last",
            "fast",
            "rne_last_fast",
            "rne_last_cache",
            "rne_last_fast_cache",
            "rne_last_pruned",
            "pruned",
            "rne_last_compact",
            "rne_last_surface",
            "rne_last_fused",
            "rne_off",
            "rne_off_surface",
            "rne_off_fused",
            "rne_off_no_support",
            "rne_off_matched_surface",
            "rne_off_specialized",
            "rne_off_uniform",
        ),
        default="baseline",
    )
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--events", action="store_true")
    parser.add_argument("--profile-start", type=int, default=0)
    parser.add_argument("--profile-steps", type=int, default=5)
    args = parser.parse_args()
    out = (
        ROOTS[args.task]
        / "analysis/coupling"
        / f"{args.platform}_{args.case}_r{args.repeat:02d}"
    )
    out.mkdir(parents=True, exist_ok=False)
    os.environ["DEXUNI_ALLOW_PILOT"] = "1"
    os.environ["DEXUNI_INPUT_ROOT"] = str(ROOTS[args.task] / "inputs")
    import torch
    import warp as wp
    import hashlib
    from scripts.benchmark.tech_reports.physics.platforms.dexuni._lib.benchlib.gpu_memory import (
        GpuProcessMemoryTracker,
    )

    memory = GpuProcessMemoryTracker()
    memory.start()

    data = {
        "task": args.task,
        "platform": args.platform,
        "case": args.case,
        "repeat": args.repeat,
        "scope": "diagnostic native platform run; original measurements unchanged",
        "profile_window": [args.profile_start, args.profile_start + args.profile_steps],
        "torch": torch.__version__,
        "warp": wp.__version__,
        "python": sys.version,
        "graph_launches": 0,
    }
    data["driver_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    data["input_config_sha256"] = hashlib.sha256(
        (
            ROOTS[args.task]
            / (
                "inputs/s03/s03_config.json"
                if args.task == "s03"
                else "inputs/s04_config.json"
            )
        ).read_bytes()
    ).hexdigest()
    frame = 0
    gpu_rows = []
    captured = Counter()
    original_launch = wp.launch
    original_capture_launch = wp.capture_launch
    specialized = None
    generic_elasticity = None
    kernel_inputs = None
    kernel_launch_parameters = None

    def launch(kernel, *pos, **kw):
        nonlocal kernel_inputs, kernel_launch_parameters
        if specialized is not None and kernel is generic_elasticity:
            kernel_inputs = kw["inputs"]
            kernel_launch_parameters = dict(kw)
            kernel = specialized
        device = wp.get_device(kw.get("device"))
        if device.is_cuda and device.is_capturing:
            captured[
                (
                    kernel.module.name,
                    kernel.key,
                    str(kw.get("dim", pos[0] if pos else None)),
                )
            ] += 1
        return original_launch(kernel, *pos, **kw)

    def capture_launch(graph, *pos, **kw):
        data["graph_launches"] += 1
        return original_capture_launch(graph, *pos, **kw)

    wp.launch = launch
    wp.capture_launch = capture_launch

    def install_solver(solver) -> None:
        """Audit live model/solver sizes before graph capture; schedule optional RNE outputs."""
        nonlocal specialized, generic_elasticity
        if data.get("solver"):
            return
        mj = getattr(solver, "mujoco_solver", None)
        vbd = getattr(solver, "vbd_solver", None)
        impl = getattr(solver, "backend", solver)
        entries = getattr(impl, "_entries", {})
        if mj is None:
            mj = next(
                (
                    e.solver
                    for e in entries.values()
                    if "MuJoCo" in type(e.solver).__name__
                ),
                None,
            )
        if vbd is None:
            vbd = next(
                (
                    e.solver
                    for e in entries.values()
                    if "VBD" in type(e.solver).__name__
                ),
                None,
            )
        info = {
            "class": type(solver).__name__,
            "file": sys.modules[type(solver).__module__].__file__,
            "entries": {},
        }
        for name, e in entries.items():
            m = e.view
            info["entries"][name] = {
                "solver": type(e.solver).__name__,
                "file": sys.modules[type(e.solver).__module__].__file__,
                "bodies": m.body_count,
                "joints": m.joint_count,
                "particles": m.particle_count,
                "shapes": m.shape_count,
                "body_qdd_allocated": e.state_0.body_qdd is not None,
                "body_parent_f_allocated": e.state_0.body_parent_f is not None,
            }
        if mj is not None:
            info["mujoco"] = {
                "nbody": mj.mj_model.nbody,
                "nv": mj.mj_model.nv,
                "nq": mj.mj_model.nq,
                "ngeom": mj.mj_model.ngeom,
                "nsensor": mj.mj_model.nsensor,
                "iterations": mj.mj_model.opt.iterations,
                "ls_iterations": mj.mj_model.opt.ls_iterations,
                "integrator": int(mj.mj_model.opt.integrator),
                "cone": int(mj.mj_model.opt.cone),
                "sensor_rne_postconstraint": mj.mjw_model.sensor_rne_postconstraint,
                "run_rne_postconstraint": getattr(
                    mj.mjw_model.opt, "run_rne_postconstraint", None
                ),
                "disableflags": int(mj.mj_model.opt.disableflags),
            }
        if vbd is not None:
            info["vbd"] = {
                "iterations": vbd.iterations,
                "enable_cuda_fast_path": getattr(vbd, "enable_cuda_fast_path", None),
                "surface_fast_eligible": getattr(
                    getattr(vbd, "_cuda_surface", None), "eligible", None
                ),
                "particle_self_contact": vbd.particle_enable_self_contact,
                "particle_collision_detection_interval": vbd.particle_collision_detection_interval,
                "surface_cache": getattr(vbd, "_surface_cached_kernel", None)
                is not None,
                "truncation_cache": getattr(vbd, "_particle_truncation_cache", None)
                is not None,
                "particle_color_groups": len(vbd.model.particle_color_groups),
                "body_color_groups": len(vbd.model.body_color_groups),
            }
        if "specialized" in args.case or "uniform" in args.case:
            if (
                not vbd.model.tet_count
                or np.any(vbd.model.tri_materials.numpy())
                or np.any(vbd.model.edge_bending_properties.numpy())
            ):
                raise ValueError(
                    "Specialization requires tetrahedra and identically zero triangle/bending materials"
                )
            from newton._src.solvers.mjvbd_v2.vbd import particle_vbd_kernels as kernels

            generic_elasticity = kernels.solve_elasticity_tile
            code = inspect.getsource(generic_elasticity.func)
            code = code.replace("@wp.kernel", '@wp.kernel(module="unique")', 1).replace(
                "def solve_elasticity_tile(", "def solve_tet_only_tile(", 1
            )
            code = (
                code.replace("if tri_indices.shape[0] > 0:", "if wp.static(False):")
                .replace("if edge_indices.shape[0] > 0:", "if wp.static(False):")
                .replace("if tet_indices.shape[0] > 0:", "if wp.static(True):")
            )
            if "uniform" in args.case:
                from scripts.benchmark.tech_reports.physics.experiments.dexuni.coupling_tuning import (
                    specialize_uniform_materials,
                )

                code = specialize_uniform_materials(
                    code, vbd.model.tet_materials.numpy()
                )
            generated = out / "generated_tet_kernel.py"
            import dexsim

            vendor = (
                Path(dexsim.__file__).parent
                / "engine/newton_physics/_vendor/newton_mjvbd_v2/vbd/particle_vbd_kernels.py"
            )
            if not vendor.is_file():
                vendor = Path(
                    "/home/oem/code/repos/dexsimforembodychain/dexsim/modules/dexsim/python/dexsim/engine/newton_physics/_vendor/newton_mjvbd_v2/vbd/particle_vbd_kernels.py"
                )
            header_lines = []
            for line in vendor.read_text().splitlines():
                if line.startswith("#") or not line.strip():
                    header_lines.append(line)
                else:
                    break
            license_header = "\n".join(header_lines) + "\n"
            code = license_header + "from __future__ import annotations\n\n" + code
            generated.write_text(code)
            module = types.ModuleType("dexuni_diagnostic_tet_kernel")
            module.__dict__.update(kernels.__dict__)
            module.__dict__["__name__"] = "dexuni_diagnostic_tet_kernel"
            module.__dict__["__file__"] = str(generated)
            sys.modules[module.__name__] = module
            exec(compile(code, str(generated), "exec"), module.__dict__)
            specialized = module.solve_tet_only_tile
            info["tet_specialization"] = (
                "same kernel/ABI and tetrahedron math; statically omit identically zero membrane/bending loops"
            )
        data["solver"] = info
        info["collision_pipelines"] = []
        for cfg in getattr(impl, "_proxy_collision_configs", {}).values():
            pp = cfg.pipeline
            info["collision_pipelines"].append(
                {
                    "class": type(pp).__name__,
                    "shape_pair_capacity": pp.shape_pairs_max,
                    "rigid_contact_max": pp.rigid_contact_max,
                    "soft_contact_max": pp.soft_contact_max,
                    "convex_support_acceleration": pp.narrow_phase.convex_support_acceleration,
                    "sparse_gjk_pairs": pp.narrow_phase.sparse_gjk_pairs,
                    "split_gjk_mpr": pp.narrow_phase.split_gjk_mpr,
                }
            )
        info["convex_mesh_vertices"] = [
            len(src.vertices)
            for src in solver.model.shape_source
            if hasattr(src, "vertices")
        ]
        model = solver.model
        _write(
            out / "model_labels.json",
            {
                key: list(getattr(model, key, []))
                for key in ("body_label", "joint_label", "shape_label")
            },
        )
        np.savez_compressed(
            out / "geometry.npz",
            **{
                key: getattr(model, key).numpy()
                for key in (
                    "body_mass",
                    "body_q",
                    "body_flags",
                    "shape_body",
                    "shape_transform",
                    "shape_scale",
                    "shape_type",
                    "shape_flags",
                    "shape_collision_group",
                    "tri_materials",
                    "tet_materials",
                    "edge_bending_properties",
                )
                if hasattr(getattr(model, key, None), "numpy")
            },
        )
        if "pruned" in args.case:
            if vbd.proxy_body_feedback != "momentum" or vbd.one_way_proxy_bodies:
                raise ValueError(
                    "Early pruning requires the existing momentum-feedback contact predicate"
                )
            from newton import BodyFlags

            for config in impl._proxy_collision_configs.values():
                pipeline = config.pipeline
                pm = pipeline.model
                bodies, flags = pm.shape_body.numpy(), pm.body_flags.numpy()
                proxy = [
                    b >= 0 and bool(int(flags[b]) & int(BodyFlags.PROXY))
                    for b in bodies
                ]
                rejected = [
                    (i, j)
                    for i in range(pm.shape_count)
                    for j in range(i + 1, pm.shape_count)
                    if (proxy[i] and (proxy[j] or bodies[j] < 0))
                    or (proxy[j] and bodies[i] < 0)
                ]
                previous = (
                    pipeline.shape_pairs_excluded.numpy().tolist()
                    if pipeline.shape_pairs_excluded is not None
                    else []
                )
                pairs = sorted(set(map(tuple, previous)) | set(rejected))
                pipeline.shape_pairs_excluded = wp.array(
                    np.asarray(pairs, dtype=np.int32).reshape(-1, 2),
                    dtype=wp.vec2i,
                    device=pm.device,
                )
                pipeline.shape_pairs_excluded_count = len(pairs)
                info["early_proxy_pruning"] = {
                    "old_exclusions": len(previous),
                    "discarded_proxy_pairs": len(rejected),
                    "new_exclusions": len(pairs),
                    "guard": "same predicate as existing _filter_vbd_proxy_contacts_kernel; no soft-contact change",
                }
        if "rne_last" in args.case and mj is not None:
            if mj.mj_model.nsensor:
                raise ValueError(
                    "This diagnostic only scopes optional RNE output for a sensor-free model"
                )
            original_step = mj.step
            calls = 0

            def rne_step(state_in, state_out, control, contacts, dt):
                nonlocal calls
                final = calls % 10 == 9
                calls += 1
                qdd, parent = state_out.body_qdd, state_out.body_parent_f
                mj.mjw_model.sensor_rne_postconstraint = final
                if hasattr(mj.mjw_model.opt, "run_rne_postconstraint"):
                    mj.mjw_model.opt.run_rne_postconstraint = final
                if not final:
                    state_out.body_qdd = state_out.body_parent_f = None
                try:
                    return original_step(state_in, state_out, control, contacts, dt)
                finally:
                    state_out.body_qdd, state_out.body_parent_f = qdd, parent

            mj.step = rne_step
        memory.sample("after_solver_initialization")
        _write(out / "diagnostics.json", data)

    if args.platform == "embodichain":
        import dexsim
        from dexsim.engine.newton_physics.solver_runtime import ensure_solver_available
        from dexsim.engine.newton_physics.backend_registry import get_newton_backend
        from embodichain.lab.sim import SimulationManager

        data["dexsim"] = dexsim.__version__
        if "no_support" in args.case:
            from newton._src.geometry.narrow_phase import NarrowPhase

            original_narrow_init = NarrowPhase.__init__

            def narrow_init(self, *pos, **kw):
                kw["convex_support_acceleration"] = False
                original_narrow_init(self, *pos, **kw)

            NarrowPhase.__init__ = narrow_init
        if "matched" in args.case:
            from newton import ModelBuilder

            original_color = ModelBuilder.color

            def color(self, *pos, **kw):
                kw["include_bending"] = False
                kw["balance_colors"] = False
                return original_color(self, *pos, **kw)

            ModelBuilder.color = color
        if "rne_off" in args.case:
            from newton import ModelBuilder

            original_request = ModelBuilder.request_state_attributes

            def request(self, *names):
                retained = tuple(
                    n for n in names if n not in ("body_qdd", "body_parent_f")
                )
                if retained:
                    return original_request(self, *retained)
                return None

            ModelBuilder.request_state_attributes = request
        if "compact" in args.case:
            from newton import ShapeFlags
            from newton._src.solvers.coupled.solver_coupled_proxy import (
                SolverCoupledProxy,
            )

            original_proxy_init = SolverCoupledProxy.__init__

            def proxy_init(self, *pos, **kw):
                model, config = kw["model"], kw["coupling"]
                shape_body, shape_flags = (
                    model.shape_body.numpy(),
                    model.shape_flags.numpy(),
                )
                active_mask = int(ShapeFlags.COLLIDE_SHAPES) | int(
                    ShapeFlags.COLLIDE_PARTICLES
                )
                active_bodies = {
                    int(b)
                    for b, f in zip(shape_body, shape_flags)
                    if b >= 0 and int(f) & active_mask
                }
                proxies = []
                for proxy in config.proxies:
                    if proxy.source == "mujoco" and proxy.destination == "vbd":
                        original_bodies = list(proxy.bodies)
                        proxy = replace(
                            proxy,
                            bodies=[b for b in original_bodies if b in active_bodies],
                        )
                        data["compact_proxy_bodies"] = {
                            "before": len(original_bodies),
                            "after": len(proxy.bodies),
                            "excluded": sorted(
                                set(original_bodies) - set(proxy.bodies)
                            ),
                            "source_ownership": "unchanged; only collisionless destination proxies removed",
                        }
                    proxies.append(proxy)
                kw["coupling"] = replace(config, proxies=proxies)
                original_proxy_init(self, *pos, **kw)

            SolverCoupledProxy.__init__ = proxy_init
        cls = ensure_solver_available("dexuni")
        original_init = cls.__init__

        def solver_init(self, *pos, **kw):
            if "fast" in args.case:
                kw["vbd_options"] = dict(
                    kw.get("vbd_options") or {}, enable_cuda_fast_path=True
                )
                kw["collision_options"] = dict(
                    kw.get("collision_options") or {}, enable_cuda_fast_path=True
                )
            if "matched" in args.case:
                kw["vbd_options"] = dict(
                    kw.get("vbd_options") or {},
                    particle_collision_detection_interval=-1,
                )
            if "surface" in args.case:
                kw["vbd_options"] = dict(
                    kw.get("vbd_options") or {},
                    particle_enable_surface_cache=True,
                    particle_enable_truncation_cache=True,
                )
            if "fused" in args.case:
                kw["vbd_options"] = dict(
                    kw.get("vbd_options") or {},
                    particle_enable_truncation_cache=True,
                    enable_cuda_fast_path=True,
                )
            if "cache" in args.case:
                kw["collision_options"] = dict(
                    kw.get("collision_options") or {},
                    stationary_rigid_contact_cache=True,
                )
            original_init(self, *pos, **kw)
            install_solver(self)

        cls.__init__ = solver_init
        original_update = SimulationManager.update

        def update(self, *pos, **kw):
            return tracked_step(lambda: original_update(self, *pos, **kw))

        SimulationManager.update = update
    else:
        if args.case != "baseline":
            raise ValueError(
                "Candidate optimizations only apply to the EC installed DexUni implementation"
            )
        from isaaclab_newton.physics import NewtonManager

        original_step = NewtonManager.step

        def manager_step(cls):
            if not data.get("solver"):
                install_solver(cls._solver)
            return tracked_step(original_step)

        NewtonManager.step = classmethod(manager_step)

    def tracked_step(operation) -> None:
        """Measure device stream duration without inserting synchronization inside the platform step."""
        nonlocal frame
        frame += 1
        if not args.events and not args.profile_start:
            return operation()
        if args.profile_start and frame == args.profile_start:
            wp.synchronize_device("cuda:0")
            torch.cuda.cudart().cudaProfilerStart()
        start = wp.Event("cuda:0", enable_timing=True)
        end = wp.Event("cuda:0", enable_timing=True)
        wp.record_event(start)
        began = time.perf_counter()
        torch.cuda.nvtx.range_push(
            f"{args.task}:{args.platform}:{args.case}:frame{frame}"
        )
        try:
            operation()
        finally:
            torch.cuda.nvtx.range_pop()
        host = time.perf_counter() - began
        wp.record_event(end)
        wp.synchronize_device("cuda:0")
        gpu = wp.get_event_elapsed_time(start, end)
        gpu_rows.append((frame, host * 1000, gpu))
        if args.profile_start and frame == args.profile_start + args.profile_steps - 1:
            torch.cuda.cudart().cudaProfilerStop()
        if frame % 120 == 0:
            print(f"diagnostic {args.platform} {args.case}: {frame} steps", flush=True)

    sys.argv = (
        ["worker", "--measure", "--repeat", "1"]
        if args.task == "s04"
        else ["worker", "--kind", "full", "--images", "none", "--out-root", str(out)]
    )
    module = importlib.import_module(
        f"scripts.benchmark.tech_reports.physics.platforms.dexuni.{args.platform}.{args.task}"
    )
    if args.task == "s04":
        module.measurement_directory = lambda *_: out

        def record(
            platform,
            output,
            inputs,
            set_target,
            advance,
            readings,
            synchronize,
            diagnostics,
            init_s,
        ):
            states = {}
            timings = []
            for i in range(inputs["config"]["steps"] + 1):
                if i:
                    set_target(i)
                    synchronize()
                    before = time.perf_counter()
                    advance()
                    synchronize()
                    timings.append(time.perf_counter() - before)
                row = readings()
                row.update(
                    time=np.asarray(inputs["time"][i]), target_q=inputs["targets"][i]
                )
                for key, value in row.items():
                    if not np.isfinite(value).all():
                        raise ValueError(f"Nonfinite {key} frame {i}")
                    states.setdefault(key, []).append(np.asarray(value).copy())
            np.savez_compressed(
                out / "states.npz", **{k: np.stack(v) for k, v in states.items()}
            )
            np.save(out / "step_seconds.npy", np.asarray(timings))
            data["effective"] = diagnostics()
            data["init_s"] = init_s
            data["step_mean_ms"] = float(np.mean(timings[30:]) * 1000)
            data["step_median_ms"] = float(np.median(timings[30:]) * 1000)

        module.record_measurement = record
        sys.argv = [module.__file__, "--measure", "--repeat", "1"]
    else:
        sys.argv = [
            module.__file__,
            "--kind",
            "full",
            "--images",
            "none",
            "--out-root",
            str(out),
        ]
    try:
        if args.task == "s03" and args.platform == "isaaclab":
            module.run(module.args_cli)
        else:
            module.main()
    finally:
        if specialized is not None and kernel_inputs is not None:
            errors = []
            for amplitude in (0.0, 0.0001, 0.001):
                inputs = list(kernel_inputs)
                if amplitude:
                    rng = np.random.default_rng(42)
                    positions = inputs[3].numpy()
                    positions += rng.normal(0, amplitude, positions.shape).astype(
                        np.float32
                    )
                    inputs[3] = wp.array(positions, dtype=wp.vec3, device="cuda:0")
                source = kernel_launch_parameters["outputs"][0]
                first, second = wp.clone(source), wp.clone(source)
                base = dict(kernel_launch_parameters, inputs=inputs, outputs=[first])
                original_launch(generic_elasticity, **base)
                base["outputs"] = [second]
                original_launch(specialized, **base)
                wp.synchronize_device("cuda:0")
                error = float(np.max(np.abs(first.numpy() - second.numpy())))
                errors.append(error)
                if error > 1e-8:
                    raise ValueError(f"Tet specialization differs: {error}")
            data["kernel_equivalence_max_errors"] = errors
        memory.sample("after_episode")
        data["gpu_memory"] = memory.metrics()
        memory.close()
        data["steps"] = frame
        _write(out / "diagnostics.json", data)
        _write(
            out / "capture_nodes.json",
            [
                {"module": m, "kernel": k, "dim": d, "count": n}
                for (m, k, d), n in captured.items()
            ],
        )
        with (out / "stream_timing.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["frame", "host_dispatch_ms", "stream_elapsed_ms"])
            writer.writerows(gpu_rows)
        print(f"Diagnostic output: {out}", flush=True)


if __name__ == "__main__":
    main()
