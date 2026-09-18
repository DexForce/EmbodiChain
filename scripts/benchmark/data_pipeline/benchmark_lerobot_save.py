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

"""Benchmark LeRobot data saving under parallel environments.

Compares four saving strategies on the same real ``StayStillSave-v1`` rollout
(N parallel envs x K episodes x 100 steps, each frame carrying an RGB image
from ``cam_high``):

- ``baseline_sync``      : ``LeRobotRecorder`` (current default; synchronous).
- ``opt_a_async_image``  : ``LeRobotRecorder`` + lerobot ``AsyncImageWriter``
                           (per-frame PNG writes offloaded to a thread pool).
- ``opt_b_async_episode``: ``AsyncLeRobotRecorder`` (whole-episode convert+save
                           offloaded to a background worker; sim never blocks).
- ``opt_ab_async_both``  : ``AsyncLeRobotRecorder`` + ``AsyncImageWriter``
                           (two levels of async).

Metrics (per variant):
- ``t_run``      : step loop. Sync baseline includes the save stall on every
                   reset; async variants do not.
- ``t_finalize`` : ``dataset_manager.finalize()`` time - drains the async worker
                   and flushes the dataset. This is the saving close cost.
- ``t_total``    : ``t_run + t_finalize`` (total time to generate AND persist).

.. note::
   ``env.close()`` calls ``sim.destroy()`` (a dexsim C++ teardown that exits the
   process without returning to Python), so each variant runs in its own
   subprocess. Results are written to a JSON file before the subprocess exits.

Run: python -m scripts.benchmark.data_pipeline.benchmark_lerobot_save
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil

# Variant -> (recorder func, image_writer_threads).
VARIANTS: dict[str, dict[str, Any]] = {
    "baseline_sync": {"func": "LeRobotRecorder", "image_writer_threads": 0},
    "opt_a_async_image": {"func": "LeRobotRecorder", "image_writer_threads": 4},
    "opt_b_async_episode": {"func": "AsyncLeRobotRecorder", "image_writer_threads": 0},
    "opt_ab_async_both": {"func": "AsyncLeRobotRecorder", "image_writer_threads": 4},
}

DEFAULT_GYM_CONFIG = (
    "embodichain_tasks/configs/tasks/special/stay_still_save/env_ur10.json"
)


# --------------------------------------------------------------------------- #
# Child process: run one variant (or sim baseline) and write a JSON result.    #
# --------------------------------------------------------------------------- #
def _make_modifier(
    variant: str,
    num_envs: int,
    save_root: Path,
    camera_hw: tuple[int, int],
    skip_save: bool = False,
):
    """Build a gym_config_modifier that selects a variant and output path."""

    def modifier(cfg_dict: dict) -> None:
        cfg_dict["num_envs"] = num_envs
        for sensor in cfg_dict.get("sensor", []):
            sensor["width"], sensor["height"] = camera_hw
        ds = cfg_dict["env"]["dataset"]["lerobot"]
        if skip_save:
            # Keep the dataset block but mark saving filtered -> no DatasetManager,
            # no rollout buffer, pure sim.
            return
        v = VARIANTS[variant]
        ds["func"] = v["func"]
        # Save every completed episode on reset (not only successes) so that
        # saves happen *during* the step loop. This is what makes the sync
        # baseline block the sim on every reset and lets the async recorder
        # pipeline saves into sim time.
        ds["save_failed_episodes"] = True
        ds["params"]["image_writer_threads"] = v["image_writer_threads"]
        ds["params"]["save_path"] = str(save_root / variant)

    return modifier


def _read_dataset_meta(save_root: Path, variant: str) -> tuple[int, int]:
    """Return (total_episodes, total_frames) saved for a variant."""
    ds_dirs = sorted(glob.glob(str(save_root / variant / "*")))
    if not ds_dirs:
        return 0, 0
    info_path = Path(ds_dirs[0]) / "meta" / "info.json"
    if not info_path.exists():
        return 0, 0
    info = json.loads(info_path.read_text())
    return int(info.get("total_episodes", 0)), int(info.get("total_frames", 0))


def _run_child(args: argparse.Namespace) -> int:
    """Run a single variant or the sim baseline; write JSON; exit."""
    import gymnasium  # imported lazily so the parent doesn't need dexsim
    import torch

    from embodichain.lab.gym.utils.gym_utils import (
        add_env_launcher_args_to_parser,
        build_env_cfg_from_args,
    )
    from embodichain.lab.gym.utils.registration import (
        discover_task_packages,
        execute_init_hooks,
    )

    discover_task_packages()
    execute_init_hooks()

    save_root = Path(args.save_root)
    camera_hw = (args.width, args.height)
    skip_save = args.variant == "sim_only"

    if not skip_save:
        shutil.rmtree(save_root / args.variant, ignore_errors=True)

    parser = argparse.ArgumentParser()
    # This child consumes a Gym config, so let the file own the physics
    # backend and its device default.  The shared standalone-parser defaults
    # (``physics=default``/``renderer=auto``) would otherwise be interpreted
    # as explicit overrides before a Newton config is decoded.
    add_env_launcher_args_to_parser(parser, require_gym_config=True)
    launcher_args = parser.parse_args(["--gym_config", args.gym_config, "--headless"])
    env_cfg, gcfg, action_config = build_env_cfg_from_args(
        launcher_args,
        gym_config_modifier=_make_modifier(
            args.variant, args.num_envs, save_root, camera_hw, skip_save
        ),
    )
    if skip_save:
        env_cfg.filter_dataset_saving = True

    proc = psutil.Process()
    mem_before = proc.memory_info().rss / 1024**2
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    env = gymnasium.make(id=gcfg["id"], cfg=env_cfg, **action_config)
    # Initial reset: nothing to save yet, so skip the save path.
    env.reset(options={"save_data": False})
    qpos = env.get_wrapper_attr("robot").get_qpos().clone()

    # --- Timed: K episodes x `steps` steps, with an explicit save-reset ---
    # --- at each episode boundary.                                  ---
    # EmbodiChain's BaseEnv does not auto-reset on truncation, so we drive
    # episode boundaries ourselves. The reset(save_data=True) call is where
    # the sync baseline blocks on saving and where the async recorder only
    # enqueues.
    t0 = time.perf_counter()
    for _ in range(args.episodes):
        for _ in range(args.steps):
            env.step(qpos)
        env.reset(options={"save_data": True})
    t_run = time.perf_counter() - t0

    # --- Timed: finalize = drain async worker + flush dataset. ---
    # NOTE: we call dataset_manager.finalize() directly instead of env.close(),
    # because env.close() -> sim.destroy() exits the process without returning.
    dataset_manager = getattr(env.unwrapped, "dataset_manager", None)
    t_f0 = time.perf_counter()
    if dataset_manager is not None:
        dataset_manager.finalize()
    t_finalize = time.perf_counter() - t_f0

    mem_after = proc.memory_info().rss / 1024**2
    peak_gpu = (
        torch.cuda.max_memory_allocated() / 1024**2
        if torch.cuda.is_available()
        else 0.0
    )
    saved_episodes, saved_frames = (
        _read_dataset_meta(save_root, args.variant) if not skip_save else (0, 0)
    )

    result = {
        "variant": args.variant,
        "func": VARIANTS[args.variant]["func"] if args.variant in VARIANTS else "-",
        "image_writer_threads": (
            VARIANTS[args.variant]["image_writer_threads"]
            if args.variant in VARIANTS
            else 0
        ),
        "num_envs": args.num_envs,
        "episodes": args.episodes,
        "steps": args.steps,
        "camera": list(camera_hw),
        "t_run_s": round(t_run, 4),
        "t_finalize_s": round(t_finalize, 4),
        "t_total_s": round(t_run + t_finalize, 4),
        "cpu_delta_mb": round(mem_after - mem_before, 1),
        "peak_gpu_mb": round(peak_gpu, 1),
        "saved_episodes": saved_episodes,
        "saved_frames": saved_frames,
    }

    Path(args.result_file).write_text(json.dumps(result))
    # Results are persisted. Force-exit to skip lerobot/dexsim interpreter
    # teardown, which can hang (image-writer threads / dataset stats) at higher
    # resolutions and would otherwise block the parent on subprocess.run.
    os._exit(0)


# --------------------------------------------------------------------------- #
# Parent process: spawn children, aggregate into a markdown report.            #
# --------------------------------------------------------------------------- #
def _spawn_child(
    args: argparse.Namespace, variant: str, result_file: Path
) -> dict | None:
    """Run one variant in a subprocess and return its result dict."""
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "scripts.benchmark.data_pipeline.benchmark_lerobot_save",
        "--child",
        "--variant",
        variant,
        "--result_file",
        str(result_file),
        "--gym_config",
        args.gym_config,
        "--num_envs",
        str(args.num_envs),
        "--episodes",
        str(args.episodes),
        "--steps",
        str(args.steps),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--save_root",
        args.save_root,
    ]
    print(f"  spawning subprocess for '{variant}'...", flush=True)
    try:
        completed = subprocess.run(
            cmd, capture_output=True, text=True, timeout=args.child_timeout
        )
    except subprocess.TimeoutExpired:
        print(
            f"  [WARN] '{variant}' subprocess timed out after "
            f"{args.child_timeout}s (result JSON may still be present)",
            flush=True,
        )
        completed = None
    if completed is not None and completed.returncode != 0:
        print(
            f"  [WARN] '{variant}' subprocess exited with code "
            f"{completed.returncode}",
            flush=True,
        )
        tail = "\n".join(completed.stderr.strip().splitlines()[-15:])
        if tail:
            print(f"  stderr tail:\n{tail}", flush=True)
    if not result_file.exists():
        print(f"  [ERROR] no result file for '{variant}'", flush=True)
        return None
    return json.loads(result_file.read_text())


def write_markdown_report(
    num_envs: int,
    episodes: int,
    steps: int,
    camera_hw: tuple[int, int],
    t_sim: float,
    results: list[dict[str, Any]],
    report_path: Path,
) -> Path:
    """Write the 3-table benchmark report."""
    baseline_total = next(
        (r["t_total_s"] for r in results if r["variant"] == "baseline_sync"),
        None,
    )
    expected_episodes = num_envs * episodes
    expected_frames = expected_episodes * steps

    perf_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    leaderboard: list[dict[str, object]] = []

    for r in results:
        save_overhead = max(0.0, r["t_run_s"] - t_sim)
        speedup = (
            round(baseline_total / r["t_total_s"], 2)
            if baseline_total and r["t_total_s"] > 0
            else "-"
        )
        frames_per_s = (
            round(r["saved_frames"] / r["t_total_s"], 1) if r["t_total_s"] > 0 else 0.0
        )
        integrity = (
            "ok"
            if (
                r["saved_episodes"] == expected_episodes
                and r["saved_frames"] == expected_frames
            )
            else "MISMATCH"
        )
        perf_rows.append(
            {
                "variant": r["variant"],
                "func": r["func"],
                "img_threads": r["image_writer_threads"],
                "t_run_s": r["t_run_s"],
                "t_finalize_s": r["t_finalize_s"],
                "t_total_s": r["t_total_s"],
                "save_overhead_s": round(save_overhead, 4),
                "cpu_delta_mb": r["cpu_delta_mb"],
                "peak_gpu_mb": r["peak_gpu_mb"],
            }
        )
        metric_rows.append(
            {
                "variant": r["variant"],
                "saved_episodes": r["saved_episodes"],
                "expected_episodes": expected_episodes,
                "saved_frames": r["saved_frames"],
                "expected_frames": expected_frames,
                "integrity": integrity,
                "frames_per_s": frames_per_s,
                "speedup_vs_baseline": speedup,
            }
        )

    ranked = sorted(results, key=lambda r: r["t_total_s"])
    for rank, r in enumerate(ranked, start=1):
        speedup = (
            round(baseline_total / r["t_total_s"], 2)
            if baseline_total and r["t_total_s"] > 0
            else "-"
        )
        leaderboard.append(
            {
                "rank": rank,
                "variant": r["variant"],
                "t_total_s": r["t_total_s"],
                "speedup_vs_baseline": speedup,
                "frames_per_s": (
                    round(r["saved_frames"] / r["t_total_s"], 1)
                    if r["t_total_s"] > 0
                    else 0.0
                ),
            }
        )

    def _table(rows: list[dict[str, object]]) -> list[str]:
        if not rows:
            return ["No rows."]
        headers = list(rows[0].keys())
        out = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for row in rows:
            out.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
        return out

    lines: list[str] = [
        "# LeRobot Save Benchmark Report",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "- Env: StayStillSave-v1 (robot holds still; `cam_high` camera).",
        f"- Parallel envs (num_envs): {num_envs}",
        f"- Episodes per env: {episodes}  x  {steps} steps/episode",
        f"- Camera: {camera_hw[0]}x{camera_hw[1]} RGB (PNG, use_videos=False)",
        f"- Expected saved per variant: {expected_episodes} episodes / {expected_frames} frames",
        f"- Pure-sim baseline (no save), {episodes}x{steps} steps: {t_sim:.4f} s",
        "",
        "`t_run` = step loop (sync baseline includes the save stall on every reset).",
        "`t_finalize` = dataset_manager.finalize() (async recorder drains worker here).",
        "`save_overhead` = t_run - t_sim (time the sim was blocked by saving).",
        "`t_total` = t_run + t_finalize (total time to generate AND persist).",
        "",
        "## Time & Memory",
        "",
    ]
    lines += _table(perf_rows)
    lines += ["", "## Success & Other Metrics", ""]
    lines += _table(metric_rows)
    lines += ["", "## Leaderboard (ranked by t_total, fastest first)", ""]
    lines += _table(leaderboard)
    lines += [
        "",
        "## Notes",
        "- `baseline_sync`: current EmbodiChain default; saving blocks env.reset().",
        "- `opt_a_async_image`: lerobot official `AsyncImageWriter` (per-frame PNG",
        "  writes offloaded to 4 threads). EmbodiChain now wires this through.",
        "- `opt_b_async_episode`: `AsyncLeRobotRecorder` clones each completed",
        "  episode and persists it on a background worker; env.reset() returns at once.",
        "- `opt_ab_async_both`: both optimizations stacked.",
        "- All variants write the same on-disk LeRobot format; integrity is verified",
        "  by checking episode/frame counts against the expected totals.",
        "- Each variant runs in its own subprocess because env.close()",
        "  (sim.destroy()) exits the process without returning to Python.",
    ]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def run_all_benchmarks(args: argparse.Namespace) -> None:
    """Spawn a subprocess per variant (+ sim baseline) and aggregate."""
    save_root = Path(args.save_root)
    save_root.mkdir(parents=True, exist_ok=True)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    camera_hw = (args.width, args.height)

    print("=" * 70, flush=True)
    print("LeRobot Save Benchmark (parallel environments)", flush=True)
    print("=" * 70, flush=True)
    print(
        f"num_envs={args.num_envs} episodes={args.episodes} steps={args.steps} "
        f"camera={camera_hw[0]}x{camera_hw[1]}",
        flush=True,
    )

    print("\n--- Pure-sim baseline (no saving) ---", flush=True)
    sim_result_file = work_dir / "sim_only.json"
    sim = _spawn_child(args, "sim_only", sim_result_file)
    t_sim = sim["t_run_s"] if sim else float("nan")
    print(f"  t_sim = {t_sim:.4f} s", flush=True)

    results: list[dict[str, Any]] = []
    for variant in VARIANTS:
        print(f"\n--- Variant: {variant} ---", flush=True)
        r = _spawn_child(args, variant, work_dir / f"{variant}.json")
        if r is not None:
            results.append(r)
            print(
                f"  t_run={r['t_run_s']:.4f}s  t_finalize={r['t_finalize_s']:.4f}s  "
                f"t_total={r['t_total_s']:.4f}s  cpu_delta={r['cpu_delta_mb']:+.1f}MB",
                flush=True,
            )
            print(
                f"  saved: {r['saved_episodes']} episodes / {r['saved_frames']} frames",
                flush=True,
            )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path(args.report_dir) / f"lerobot_save_{ts}.md"
    write_markdown_report(
        args.num_envs,
        args.episodes,
        args.steps,
        camera_hw,
        t_sim,
        results,
        report_path,
    )
    print("\n" + "=" * 70, flush=True)
    print(f"Markdown report saved: {report_path}", flush=True)
    print("=" * 70, flush=True)


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gym_config",
        default=DEFAULT_GYM_CONFIG,
        help="Path to the stay-still gym config JSON.",
    )
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument(
        "--save_root", default="/tmp/lerobot_save_bench", help="Dataset output root."
    )
    parser.add_argument(
        "--report_dir", default="outputs/benchmarks", help="Markdown report dir."
    )
    parser.add_argument(
        "--work_dir", default="/tmp/lerobot_save_bench_work", help="JSON result dir."
    )
    parser.add_argument(
        "--child_timeout",
        type=int,
        default=900,
        help="Per-variant subprocess timeout in seconds (safety net).",
    )
    # Child-mode flags (used when the parent re-invokes this script):
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--variant", default="", help=argparse.SUPPRESS)
    parser.add_argument("--result_file", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.child:
        sys.exit(_run_child(args))
    run_all_benchmarks(args)


if __name__ == "__main__":
    cli()
