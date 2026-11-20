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

"""Compare sensor-free Task Program execution on pinned source trees.

Run with --help. Results measure execution, not qualified dataset production.
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
import resource
import subprocess
import cProfile
import pstats


def write_report(path: Path, rows: list[dict]) -> None:
    """Write three tables, explicitly separating program acceptance from quality."""
    text = [
        "# Sensor-free execution diagnostic",
        "",
        (
            "Profiler enabled: these times diagnose hotspots and must not be compared as benchmark latency."
            if any(row.get("profiled") for row in rows)
            else "Profiler disabled."
        ),
        "",
        "Cold execution; CPU Default physics and solvers, with legacy CUDA grasp classification included. Program acceptance is not physical qualification. No sensors, recording, external collision checks or persistence. Process RSS peaks include startup; their sum is an upper bound, not a simultaneous sampled peak.",
        "",
        "## Time & Memory",
        "",
        "| Scale | Mode | cost_time_ms | Action rows/s | Sum env.step s | Sum sim.update s | Sum peak RSS MiB | cpu_delta_mb | gpu_delta_mb | peak_gpu_mb |",
        "|---|---|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        text.append(
            f"| {row['scale']} | {row['mode']} | {row['execution_s']*1000:.3f} | {row['action_rows']/row['execution_s']:.1f} | {row['env_step_s']:.3f} | {row['sim_update_s']:.3f} | {row['peak_rss_mib']:.1f} | N/A | N/A | N/A |"
        )
    text += [
        "",
        "## Success & Other Metrics",
        "",
        "| Scale | Mode | Attempts | Program accepted | Action rows |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        text.append(
            f"| {row['scale']} | {row['mode']} | {row['attempts']} | {row['accepted']} | {row['action_rows']} |"
        )
    text += [
        "",
        "## Leaderboard",
        "",
        "Ranked by program acceptance fraction only; modes with different action counts are not equal-work latency comparisons.",
        "",
        "| Rank | Scale | Mode | Program acceptance |",
        "|---:|---|---|---:|",
    ]
    for i, row in enumerate(
        sorted(rows, key=lambda r: r["accepted"] / max(r["attempts"], 1), reverse=True),
        1,
    ):
        text.append(
            f"| {i} | {row['scale']} | {row['mode']} | {row['accepted']/max(row['attempts'],1):.3f} |"
        )
    path.write_text("\n".join(text) + "\n")


def matrix_main() -> None:
    """Run variants sequentially, with concurrent main workers per scale."""
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", action="store_true")
    p.add_argument("--baseline-root", type=Path, required=True)
    p.add_argument("--pr-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--waves", type=int, default=2)
    p.add_argument("--seeds", default="7")
    p.add_argument("--scales", default="small,medium,large")
    p.add_argument("--order", choices=("forward", "reverse"), default="forward")
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=False)
    cpus = sorted(os.sched_getaffinity(0))[:8]
    os.sched_setaffinity(0, cpus)
    settings = {"small": (1, 1), "medium": (2, 4), "large": (4, 8)}
    rows = []
    for seed_index, seed in enumerate(map(int, a.seeds.split(","))):
        for scale in a.scales.split(","):
            w, b = settings[scale]
            modes = ["cohort", "rolling-off", "rolling-on"]
            if (seed_index % 2) ^ (a.order == "reverse"):
                modes.reverse()
            for mode in modes:
                processes = []
                for worker in range(w if mode == "cohort" else 1):
                    name = f"{scale}-{mode}-seed{seed}-worker{worker}"
                    output = a.output_dir / (name + ".json")
                    log = (a.output_dir / (name + ".log")).open("w")
                    command = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--root",
                        str(a.baseline_root if mode == "cohort" else a.pr_root),
                        "--mode",
                        mode,
                        "--rows",
                        str(b if mode == "cohort" else w * b),
                        "--groups",
                        str(1 if mode == "cohort" else w),
                        "--episodes",
                        str(a.waves * b if mode == "cohort" else a.waves * w * b),
                        "--seed",
                        str(seed),
                        "--output",
                        str(output),
                    ]
                    process = subprocess.Popen(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env={
                            **os.environ,
                            "OMP_NUM_THREADS": "1",
                            "MKL_NUM_THREADS": "1",
                            "PYTHONHASHSEED": "0",
                            "SDL_VIDEODRIVER": "offscreen",
                        },
                    )
                    processes.append((process, log, output))
                failed = []
                for process, log, output in processes:
                    code = process.wait()
                    log.close()
                    if code:
                        failed.append(str(output.with_suffix(".log")))
                if failed:
                    raise RuntimeError(f"Workers failed: {failed}")
                reports = [json.loads(output.read_text()) for _, _, output in processes]
                episodes = [e for report in reports for e in report["episodes"]]
                row = {
                    "scale": scale,
                    "mode": mode,
                    "seed": seed,
                    "execution_s": max(r["execution_end"] for r in reports)
                    - min(r["execution_start"] for r in reports),
                    "env_step_s": sum(r["env_step_s"] for r in reports),
                    "sim_update_s": sum(r["sim_update_s"] for r in reports),
                    "peak_rss_mib": sum(r["peak_rss_mib"] for r in reports),
                    "attempts": len(episodes),
                    "accepted": sum(e["program_accepted"] for e in episodes),
                    "action_rows": sum(e["steps"] for e in episodes),
                }
                rows.append(row)
                (a.output_dir / "results.json").write_text(
                    json.dumps(
                        {
                            "cpu_affinity": cpus,
                            "initial_order": a.order,
                            "waves": a.waves,
                            "baseline_root": str(a.baseline_root),
                            "pr_root": str(a.pr_root),
                            "rows": rows,
                        },
                        indent=2,
                    )
                )
                write_report(a.output_dir / "report.md", rows)
                print(json.dumps(row), flush=True)


def main() -> None:
    """Execute a bounded real-physics diagnostic in one isolated process."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument(
        "--mode", choices=["cohort", "rolling-off", "rolling-on"], required=True
    )
    p.add_argument("--rows", type=int, default=1)
    p.add_argument("--groups", type=int, default=1)
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--profile",
        action="store_true",
        help="Profile execution only; these times are not benchmark results.",
    )
    a = p.parse_args()
    root = a.root.resolve()
    sys.path[:0] = [
        str(root),
        str(root / "embodichain/extensions/tasks"),
        str(root / "embodichain/extensions/robots"),
    ]
    import torch
    from embodichain.lab.gym.utils.gym_utils import config_to_cfg
    from embodichain.lab.gym.utils.registration import (
        REGISTERED_ENVS,
        discover_task_packages,
        execute_init_hooks,
    )
    from embodichain.lab.sim import SimulationManager
    from embodichain.utils.utility import load_config
    from embodichain.lab.expansion.program import collect_program_batch

    torch.set_num_threads(1)
    path = (
        root
        / "embodichain/extensions/tasks/configs/tasks/manipulation/repeated_pick_place/task.ur5.yaml"
    )
    discover_task_packages()
    execute_init_hooks()
    cfg = config_to_cfg(load_config(path), source_path=path)
    cfg.num_envs = cfg.sim_cfg.num_envs = a.rows
    cfg.sim_cfg.device = "cpu"
    cfg.sim_cfg.headless = True
    cfg.sim_cfg.render_cfg.renderer = "no-render"
    cfg.sensor = []
    cfg.enable_sensor = False
    cfg.dataset = None
    cfg.init_rollout_buffer = cfg.record_trajectory = False
    timings = {"env_step_s": 0.0, "reset_s": 0.0}
    sim_time = [0.0]
    episodes = []
    adapters = {}
    env = None
    start = time.perf_counter()
    try:
        env = REGISTERED_ENVS[load_config(path)["id"]].make(cfg=cfg)
        assert env.sim.physics_backend == "default"
        assert not env.sim.get_sensor_uid_list(), env.sim.get_sensor_uid_list()
        env.reset(seed=a.seed)
        program = env.compile_task_program(cfg.task_program)
        step = env.step
        reset = env.reset

        def timed_step(*args, **kwargs):
            tick = time.perf_counter()
            try:
                return step(*args, **kwargs)
            finally:
                timings["env_step_s"] += time.perf_counter() - tick

        def timed_reset(*args, **kwargs):
            tick = time.perf_counter()
            try:
                return reset(*args, **kwargs)
            finally:
                timings["reset_s"] += time.perf_counter() - tick

        env.step = timed_step
        env.reset = timed_reset
        update = env.sim.update

        def timed_update(*args, **kwargs):
            tick = time.perf_counter()
            try:
                return update(*args, **kwargs)
            finally:
                sim_time[0] += time.perf_counter() - tick

        env.sim.update = timed_update
        setup = time.perf_counter() - start

        def bridge(row):
            adapter = env.task_program_adapter.for_env_ids((row,))
            adapters[row] = adapter
            return adapter.create_bridge(program, isolate_policy_sessions=True)

        def completed(result):
            for row, length, accepted, reason in zip(
                result.env_ids, result.lengths, result.accepted, result.reasons
            ):
                episodes.append(
                    {
                        "row": row,
                        "steps": length,
                        "program_accepted": accepted,
                        "reason": reason,
                    }
                )
                adapters.pop(row).close()

        profiler = cProfile.Profile() if a.profile else None
        cpu_before = resource.getrusage(resource.RUSAGE_SELF)
        if profiler is not None:
            profiler.enable()
        begin = time.perf_counter()
        before = env.sim.physics_step_index
        planning = {}
        if a.mode == "cohort":
            remaining = a.episodes
            while remaining:
                n = min(a.rows, remaining)
                env.reset(seed=a.seed, options={"save_data": False})
                # Same minimum settling, at ordinary environment cadence.
                from embodichain.lab.gym.envs.types import ControllerAction

                physical = env.sim.physics_step_index
                while env.sim.physics_step_index - physical < 5:
                    env.step(ControllerAction(env.robot.get_qpos(target=True).clone()))
                env._elapsed_steps[:] = 0
                env._reset_physical_objective(tuple(range(n)))
                result = collect_program_batch(
                    env,
                    {r: bridge(r) for r in range(n)},
                    max_steps=1200,
                    observe_step=lambda *args: None,
                )
                completed(result)
                remaining -= n
        else:
            from embodichain.lab.expansion.workloads import (
                WorkloadProgramExecutor,
                WorkloadExecutionCfg,
                ExpansionWorkloadCfg,
            )

            class Source:
                def prepare(self, assignments):
                    pass

                def activate(self, assignment, observation):
                    return bridge(assignment.env_id)

                def started(self, assignment, boundary):
                    pass

                def observe(self, assignments, action, observation):
                    return ()

                def complete(self, result):
                    completed(result.program)
                    # No persistence in this diagnostic: never fabricate receipts.
                    return None

            assert a.rows % a.groups == 0 and a.episodes % a.groups == 0
            count = a.episodes // a.groups
            executor = WorkloadProgramExecutor(
                env,
                [
                    ExpansionWorkloadCfg(f"g{i}", count, count, 3600)
                    for i in range(a.groups)
                ],
                Source(),
                WorkloadExecutionCfg(
                    workload_count=a.groups,
                    batch_size=a.rows // a.groups,
                    max_steps=1200,
                    settling_steps=5,
                    seed=a.seed,
                    batch_planning=a.mode == "rolling-on",
                ),
            )
            outcome = executor.run()
            planning = outcome["planning"]
        elapsed = time.perf_counter() - begin
        if profiler is not None:
            profiler.disable()
        cpu_after = resource.getrusage(resource.RUSAGE_SELF)
        report = {
            "mode": a.mode,
            "rows": a.rows,
            "seed": a.seed,
            "setup_s": setup,
            "execution_s": elapsed,
            "profiled": a.profile,
            "physics_backend": env.sim.physics_backend,
            "physics_device": str(env.sim.device),
            "solver_devices": {
                name: str(solver.device) for name, solver in env.robot._solvers.items()
            },
            "grasp_collision_device_policy": "legacy current CUDA device",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_device_after_execution": (
                torch.cuda.current_device() if torch.cuda.is_initialized() else None
            ),
            "process_cpu_s": cpu_after.ru_utime
            + cpu_after.ru_stime
            - cpu_before.ru_utime
            - cpu_before.ru_stime,
            "execution_start": begin,
            "execution_end": begin + elapsed,
            "sim_update_s": sim_time[0],
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            **timings,
            "remaining_execution_s": elapsed - sum(timings.values()),
            "physics_substeps": env.sim.physics_step_index - before,
            "episodes": episodes,
            "planning": planning,
            "sensors": env.sim.get_sensor_uid_list(),
            "scope": "cold execution including planning, preparation and env.step; no sensors, external collision qualification, media, artifact persistence or export; program acceptance is not physical qualification",
        }
        a.output.parent.mkdir(parents=True, exist_ok=True)
        if profiler is not None:
            profiler.dump_stats(str(a.output.with_suffix(".pstats")))
            with a.output.with_suffix(".profile.txt").open("w") as stream:
                stream.write(
                    "Profiled execution only. Cumulative times overlap; greenlet switching can distort caller attribution. Use leaf self-times and call counts, not sums of cumulative times.\n"
                )
                stats = pstats.Stats(profiler, stream=stream)
                stats.sort_stats("tottime").print_stats(70)
                stats.sort_stats("cumulative").print_stats(70)
        a.output.write_text(json.dumps(report, indent=2))
        write_report(
            a.output.with_suffix(".md"),
            [
                {
                    **report,
                    "scale": str(a.rows),
                    "attempts": len(episodes),
                    "accepted": sum(e["program_accepted"] for e in episodes),
                    "action_rows": sum(e["steps"] for e in episodes),
                }
            ],
        )
        print(
            json.dumps(
                {k: v for k, v in report.items() if k not in ("episodes", "planning")}
            ),
            flush=True,
        )
    finally:
        for adapter in adapters.values():
            adapter.close()
        if env is not None:
            env.close(exit_process=False)
        SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    matrix_main() if "--matrix" in sys.argv else main()
