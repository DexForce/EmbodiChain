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
"""GPU VRAM leak tracker for run_env episodes.

Runs the same environment as `python -m embodichain run-env` but records GPU
memory after every episode reset.  At the end it prints a summary table,
performs a simple linear-regression slope test, and writes a CSV report.

Usage
-----
::

    python scripts/vram_tracker.py \\
        --gym_config  configs/gym/pour_water/gym_config.json \\
        --action_config configs/gym/pour_water/action_config.json \\
        --headless \\
        --filter_visual_rand \\
        --renderer fast-rt \\
        --filter_dataset_saving \\
        --episodes 30 \\
        --output vram_report.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

# Ensure repository root is on sys.path so `embodichain` can be imported
# when running this script directly (without installing the package).
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


# ---------------------------------------------------------------------------
# VRAM sampling helpers
# ---------------------------------------------------------------------------

def _torch_vram_mb() -> tuple[float, float]:
    """Return (allocated_MB, reserved_MB) from PyTorch CUDA allocator."""
    if not torch.cuda.is_available():
        return 0.0, 0.0
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    return allocated, reserved


def _nvml_process_vram_mb(pid: int, gpu_index: int = 0) -> Optional[float]:
    """Return process-level VRAM usage (MiB) via nvidia-smi, or None on failure."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
                f"--id={gpu_index}",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2 and int(parts[0]) == pid:
                return float(parts[1])
        return None
    except Exception:
        return None


def _nvml_total_used_vram_mb(gpu_index: int = 0) -> Optional[float]:
    """Return total used VRAM on the GPU (MiB) via nvidia-smi."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return float(out.strip().splitlines()[0].strip())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EpisodeSample:
    episode: int
    wall_time_s: float          # seconds since start
    torch_alloc_mb: float
    torch_reserved_mb: float
    proc_vram_mb: Optional[float]   # nvidia-smi process-level
    gpu_total_used_mb: Optional[float]  # nvidia-smi device total used


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _linear_slope(xs: List[float], ys: List[float]) -> float:
    """Least-squares slope of y = a*x + b."""
    if len(xs) < 2:
        return 0.0
    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    x -= x.mean()
    denom = float(np.dot(x, x))
    if denom == 0:
        return 0.0
    return float(np.dot(x, y - y.mean()) / denom)


def _print_table(samples: List[EpisodeSample]) -> None:
    header = (
        f"{'Ep':>4}  {'Time(s)':>8}  "
        f"{'Torch Alloc(MB)':>16}  {'Torch Resv(MB)':>14}  "
        f"{'Proc VRAM(MB)':>13}  {'GPU Used(MB)':>12}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for s in samples:
        proc = f"{s.proc_vram_mb:>13.1f}" if s.proc_vram_mb is not None else f"{'N/A':>13}"
        gpu = f"{s.gpu_total_used_mb:>12.1f}" if s.gpu_total_used_mb is not None else f"{'N/A':>12}"
        print(
            f"{s.episode:>4}  {s.wall_time_s:>8.1f}  "
            f"{s.torch_alloc_mb:>16.1f}  {s.torch_reserved_mb:>14.1f}  "
            f"{proc}  {gpu}"
        )
    print("=" * len(header))


def _print_slope_analysis(samples: List[EpisodeSample]) -> None:
    if len(samples) < 3:
        print("\n[WARN] Too few samples for slope analysis.")
        return

    xs = [float(s.episode) for s in samples]
    fields_map = {
        "Torch Alloc (MB/ep)":   [s.torch_alloc_mb for s in samples],
        "Torch Reserved (MB/ep)":[s.torch_reserved_mb for s in samples],
    }
    if any(s.proc_vram_mb is not None for s in samples):
        fields_map["Proc VRAM (MB/ep)"] = [
            s.proc_vram_mb if s.proc_vram_mb is not None else float("nan")
            for s in samples
        ]
    if any(s.gpu_total_used_mb is not None for s in samples):
        fields_map["GPU Total Used (MB/ep)"] = [
            s.gpu_total_used_mb if s.gpu_total_used_mb is not None else float("nan")
            for s in samples
        ]

    print("\n[Slope Analysis]  (slope > ~1 MB/ep suggests a leak)")
    print(f"  {'Metric':<28}  {'Slope':>10}  {'Verdict'}")
    print("  " + "-" * 55)
    for label, ys in fields_map.items():
        ys_clean = [y for y in ys if not (isinstance(y, float) and y != y)]
        xs_clean = [xs[i] for i, y in enumerate(ys) if not (isinstance(y, float) and y != y)]
        slope = _linear_slope(xs_clean, ys_clean)
        verdict = "LEAK?" if abs(slope) > 1.0 else "OK"
        print(f"  {label:<28}  {slope:>10.3f}  {verdict}")
    print()


def _save_csv(samples: List[EpisodeSample], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "episode",
                "wall_time_s",
                "torch_alloc_mb",
                "torch_reserved_mb",
                "proc_vram_mb",
                "gpu_total_used_mb",
            ]
        )
        for s in samples:
            writer.writerow(
                [
                    s.episode,
                    f"{s.wall_time_s:.2f}",
                    f"{s.torch_alloc_mb:.2f}",
                    f"{s.torch_reserved_mb:.2f}",
                    f"{s.proc_vram_mb:.2f}" if s.proc_vram_mb is not None else "",
                    f"{s.gpu_total_used_mb:.2f}" if s.gpu_total_used_mb is not None else "",
                ]
            )
    print(f"[vram_tracker] Report saved to: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _sample(episode: int, t0: float, pid: int, gpu_index: int) -> EpisodeSample:
    torch_alloc, torch_resv = _torch_vram_mb()
    proc_vram = _nvml_process_vram_mb(pid, gpu_index)
    gpu_used = _nvml_total_used_vram_mb(gpu_index)
    return EpisodeSample(
        episode=episode,
        wall_time_s=time.perf_counter() - t0,
        torch_alloc_mb=torch_alloc,
        torch_reserved_mb=torch_resv,
        proc_vram_mb=proc_vram,
        gpu_total_used_mb=gpu_used,
    )


def main() -> None:
    # ---- Parse args (superset of run_env args) ----
    parser = argparse.ArgumentParser(
        description="Track GPU VRAM across run_env episodes to detect memory leaks."
    )
    # run_env pass-through args
    from embodichain.lab.gym.utils.gym_utils import add_env_launcher_args_to_parser
    add_env_launcher_args_to_parser(parser)
    # tracker-specific args
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of episodes to run (overrides gym_config max_episodes). Default: 20.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="vram_report.csv",
        help="Path for the CSV output report. Default: vram_report.csv.",
    )
    parser.add_argument(
        "--gpu_index",
        type=int,
        default=0,
        help="GPU device index to query via nvidia-smi. Default: 0.",
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="Skip matplotlib plot even if it is available.",
    )

    args = parser.parse_args()

    # ---- Build env ----
    import gymnasium
    from embodichain.lab.gym.utils.gym_utils import build_env_cfg_from_args

    np.set_printoptions(5, suppress=True)
    torch.set_printoptions(precision=5, sci_mode=False)

    env_cfg, gym_config, action_config = build_env_cfg_from_args(args)

    # Override max_episodes with --episodes
    num_episodes = args.episodes

    print(f"[vram_tracker] Building environment '{gym_config['id']}' ...")
    env = gymnasium.make(id=gym_config["id"], cfg=env_cfg, **action_config)

    pid = os.getpid()
    gpu_index = args.gpu_index
    t0 = time.perf_counter()
    samples: List[EpisodeSample] = []

    # ---- Episode loop ----
    from embodichain.lab.scripts.run_env import generate_function

    print(f"[vram_tracker] Running {num_episodes} episodes — tracking VRAM ...\n")

    for ep in range(num_episodes):
        # Sample BEFORE reset (captures any residual from previous episode)
        s_before = _sample(ep, t0, pid, gpu_index)
        samples.append(s_before)
        print(
            f"[ep {ep:03d}]  torch_alloc={s_before.torch_alloc_mb:.1f} MB  "
            f"torch_resv={s_before.torch_reserved_mb:.1f} MB  "
            f"proc_vram={s_before.proc_vram_mb} MB  "
            f"gpu_used={s_before.gpu_total_used_mb} MB"
        )

        generate_function(
            env,
            num_traj=1,
            time_id=ep,
            save_path=getattr(args, "save_path", ""),
            save_video=getattr(args, "save_video", False),
            debug_mode=getattr(args, "debug_mode", False),
            regenerate=getattr(args, "regenerate", False),
        )

    # Final sample after all episodes
    s_final = _sample(num_episodes, t0, pid, gpu_index)
    samples.append(s_final)
    print(
        f"[ep {num_episodes:03d} FINAL]  "
        f"torch_alloc={s_final.torch_alloc_mb:.1f} MB  "
        f"torch_resv={s_final.torch_reserved_mb:.1f} MB  "
        f"proc_vram={s_final.proc_vram_mb} MB  "
        f"gpu_used={s_final.gpu_total_used_mb} MB"
    )

    # ---- Report ----
    _print_table(samples)
    _print_slope_analysis(samples)
    _save_csv(samples, args.output)

    # ---- Optional plot ----
    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle("GPU VRAM per Episode", fontsize=14)
            xs = [s.episode for s in samples]
            plot_data = [
                (axes[0, 0], [s.torch_alloc_mb for s in samples],    "Torch Alloc (MB)",    "blue"),
                (axes[0, 1], [s.torch_reserved_mb for s in samples], "Torch Reserved (MB)", "orange"),
                (axes[1, 0], [s.proc_vram_mb or 0 for s in samples], "Proc VRAM / nvidia-smi (MB)", "green"),
                (axes[1, 1], [s.gpu_total_used_mb or 0 for s in samples], "GPU Total Used (MB)", "red"),
            ]
            for ax, ys, title, color in plot_data:
                ax.plot(xs, ys, marker="o", color=color, linewidth=1.5, markersize=4)
                # Linear trend line
                if len(xs) >= 2:
                    slope = _linear_slope([float(x) for x in xs], [float(y) for y in ys])
                    trend = [slope * (x - xs[0]) + ys[0] for x in xs]
                    ax.plot(xs, trend, "--", color="grey", linewidth=1, label=f"slope={slope:.2f} MB/ep")
                    ax.legend(fontsize=8)
                ax.set_title(title, fontsize=10)
                ax.set_xlabel("Episode")
                ax.set_ylabel("MB")
                ax.grid(True, linestyle="--", alpha=0.5)

            plt.tight_layout()
            plot_path = args.output.replace(".csv", ".png")
            plt.savefig(plot_path, dpi=120)
            print(f"[vram_tracker] Plot saved to: {plot_path}")
            plt.close()
        except ImportError:
            print("[vram_tracker] matplotlib not available, skipping plot.")

    # ---- Env cleanup ----
    env.close()


if __name__ == "__main__":
    main()
