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

"""Compare G2's original numerical IK with strict, seed-local Warp SSIK.

Run: python -m scripts.benchmark.robotics.kinematic_solver.agibot_g2 --output report.md
Targets and acceptance use the original offset URDF; both solvers use 1e-5
position/rotation tolerances. The legacy SRS solver is structurally inapplicable:
its shoulder reconstruction assumes a zero common-normal upper-arm offset.
No fabricated SRS calibration or zero-offset surrogate is benchmarked.
"""

from __future__ import annotations

import argparse
import platform
from pathlib import Path
import statistics
import time

import psutil
import torch
import warp as wp

from embodichain.lab.sim.motion.solvers import PytorchSolverCfg, SSIKSolverCfg
from embodichain_robots import AgibotG2Cfg

__all__ = []


def _sync(device: str) -> None:
    """Complete Torch and Warp work before sampling wall time or memory."""
    if device == "cuda":
        wp.synchronize_device(device)
        torch.cuda.synchronize()


def _table(rows: list[dict[str, object]]) -> list[str]:
    """Render rows with one stable Markdown header."""
    keys = list(rows[0])
    return [
        "| " + " | ".join(keys) + " |",
        "| " + " | ".join("---" for _ in keys) + " |",
    ] + ["| " + " | ".join(str(row[key]) for key in keys) + " |" for row in rows]


def _cpu_name() -> str:
    """Report the Linux CPU model, falling back to platform information."""
    info = Path("/proc/cpuinfo")
    if info.is_file():
        for line in info.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def _main() -> None:
    """Measure matched inputs and write exactly three result tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--devices", nargs="+", choices=("cpu", "cuda"), default=["cpu", "cuda"]
    )
    parser.add_argument("--sizes", nargs="+", type=int, default=[1, 32, 128])
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1 or any(n < 1 for n in args.sizes):
        parser.error("repeats and sizes must be positive")
    if "cpu" not in args.devices and not torch.cuda.is_available():
        parser.error("no usable device selected: CUDA is unavailable; select cpu")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    wp.init()
    cfg = AgibotG2Cfg.from_dict({})
    perf, metrics = [], []
    notes = [
        f"Python {platform.python_version()}, Torch {torch.__version__}, Warp {wp.__version__}; CPU {_cpu_name()}, Torch threads=1.",
        f"CUDA: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'unavailable'}.",
        f"Target and numerical sampler RNG seed=42; float32 device-resident targets/seeds; two warmups per shape; {args.repeats} completed-call repeats, median and min/max wall latency; excludes setup and transfers.",
        "Success requires solver validity, finite in-limit joints, translation error <=1e-5 m, and rotation Frobenius error <=sqrt(2)*1e-5. Failed targets remain in the denominator.",
        "Memory uses a separate warmed call, retains one output, and reports RSS retained delta, not CPU transient peak. GPU statistics cover Torch allocations only, including total peak, and exclude Warp/native persistent scratch. These are not total device-memory measurements.",
        "Legacy SRSSolver: not applicable without changing its shoulder/elbow reconstruction; no latency or success percentage is assigned to an unsupported configuration.",
    ]
    for device in args.devices:
        if device == "cuda" and not torch.cuda.is_available():
            notes.append("CUDA skipped: hardware unavailable.")
            continue
        for part, chain in cfg.build_pk_serial_chain(torch.device(device)).items():
            low, high = (
                torch.tensor(v, device=device) for v in chain.get_joint_limits()
            )
            for label, options in (
                ("Pytorch", PytorchSolverCfg(pos_eps=1e-5, rot_eps=1e-5)),
                ("SSIK", SSIKSolverCfg(srs_search="seed_local")),
            ):
                options.joint_names = cfg.control_parts[part]
                options.root_link_name = cfg.solver_cfg[part].root_link_name
                options.end_link_name = cfg.solver_cfg[part].end_link_name
                solver = options.init_solver(device=device, pk_serial_chain=chain)
                for size in args.sizes:
                    fractions = 0.05 + 0.9 * torch.rand(
                        size, 7, generator=torch.Generator().manual_seed(42)
                    )
                    q = low + fractions.to(device) * (high - low)
                    target = chain.forward_kinematics(q).get_matrix()
                    for seed_name in ("midpoint", "nearby"):
                        seed = (
                            ((low + high) / 2).expand_as(q)
                            if seed_name == "midpoint"
                            else (q + 0.02).clamp(low, high)
                        )

                        def operation():
                            return solver.get_ik_batch(target, qpos_seed=seed)

                        for _ in range(2):
                            operation()
                        _sync(device)
                        timings = []
                        for _ in range(args.repeats):
                            _sync(device)
                            start = time.perf_counter()
                            result = operation()
                            _sync(device)
                            timings.append((time.perf_counter() - start) * 1000)
                            del result
                        rss = psutil.Process().memory_info().rss
                        if device == "cuda":
                            torch.cuda.reset_peak_memory_stats()
                            allocated = torch.cuda.memory_allocated()
                        mask, solution = operation()
                        _sync(device)
                        cpu_delta = (psutil.Process().memory_info().rss - rss) / 1024**2
                        gpu_delta = (
                            (torch.cuda.memory_allocated() - allocated) / 1024**2
                            if device == "cuda"
                            else "n/a"
                        )
                        gpu_peak = (
                            torch.cuda.max_memory_allocated() / 1024**2
                            if device == "cuda"
                            else "n/a"
                        )
                        actual = chain.forward_kinematics(solution).get_matrix()
                        pos = (actual[:, :3, 3] - target[:, :3, 3]).norm(dim=-1)
                        rot = (
                            (actual[:, :3, :3] - target[:, :3, :3])
                            .flatten(1)
                            .norm(dim=-1)
                        )
                        accepted = (
                            mask
                            & torch.isfinite(solution).all(-1)
                            & (solution >= low).all(-1)
                            & (solution <= high).all(-1)
                            & (pos <= 1e-5)
                            & (rot <= 2**0.5 * 1e-5)
                        )
                        key = dict(
                            algorithm=label,
                            device=device,
                            arm=part,
                            seed=seed_name,
                            batch=size,
                        )
                        perf.append(
                            dict(
                                **key,
                                cost_time_ms=round(statistics.median(timings), 3),
                                min_ms=round(min(timings), 3),
                                max_ms=round(max(timings), 3),
                                cpu_delta_mb=round(cpu_delta, 3),
                                gpu_delta_mb=gpu_delta,
                                peak_gpu_mb=gpu_peak,
                            )
                        )
                        metrics.append(
                            dict(
                                **key,
                                successes=int(accepted.sum()),
                                success_rate=float(accepted.float().mean()),
                                max_position_m=(
                                    float(pos[accepted].max())
                                    if accepted.any()
                                    else "n/a"
                                ),
                                max_rotation_frobenius=(
                                    float(rot[accepted].max())
                                    if accepted.any()
                                    else "n/a"
                                ),
                            )
                        )
                        print(perf[-1], metrics[-1], flush=True)
                        del mask, solution, actual
    leaderboard = []
    for label in ("Pytorch", "SSIK"):
        rows = [r for r in metrics if r["algorithm"] == label]
        leaderboard.append(
            dict(
                algorithm=label,
                success_rate=sum(r["successes"] for r in rows)
                / sum(r["batch"] for r in rows),
            )
        )
    leaderboard.sort(key=lambda row: row["success_rate"], reverse=True)
    lines = [
        "# AgiBot G2 IK qualification",
        "",
        *notes,
        "",
        "## Time & Memory",
        "",
        *_table(perf),
        "",
        "## Success & Other Metrics",
        "",
        *_table(metrics),
        "",
        "## Leaderboard",
        "",
        *_table(leaderboard),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(args.output)


if __name__ == "__main__":
    _main()
