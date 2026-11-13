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

"""Shared inputs and immutable run records for P-02.

Run: imported by the two platform runners; no simulator is imported here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
from scipy.spatial.transform import Rotation


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add only the options used by both P-02 platform runners."""
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--duration", type=float)
    parser.add_argument(
        "--phase",
        choices=[
            "pilot",
            "main",
            "supplement",
            "capture",
            "diagnostic",
            "independent",
            "independent_supplement",
        ],
        default="main",
    )
    parser.add_argument("--record", action="store_true")


def scene_geometry(cfg: dict, scene: dict, inputs: dict, initial: int) -> dict:
    """Build the same SI box geometry and xyzw poses for both simulators."""
    if scene["kind"] == "slope":
        rotation = Rotation.from_euler("y", scene["angle_deg"], degrees=True)
        origin = np.array(cfg["slope_top_origin_m"])
        normal = rotation.apply([0, 0, 1])
        downhill = rotation.apply([1, 0, 0])
        size = np.array(cfg["slope_body_size_m"])
        pos = (
            origin
            + downhill * inputs["slope_start_s_m"][initial]
            + np.array([0, inputs["slope_start_y_m"][initial], 0])
            + normal * (size[2] / 2 + cfg["initial_gap_m"])
        )
        poses = np.array([[*pos, *rotation.as_quat()]])
        support_size = np.array(cfg["slope_size_m"])
        support_pose = np.r_[origin - normal * support_size[2] / 2, rotation.as_quat()]
    else:
        count = scene["layers"]
        size = np.array(cfg["stack_cube_size_m"])
        poses = np.zeros((count, 7))
        poses[:, :2] = np.array(inputs["stack_xy_m"][initial])[:count]
        poses[:, 2] = (
            cfg["stack_support_top_z_m"]
            + size[2] / 2
            + np.arange(count) * size[2]
            + (np.arange(count) + 1) * cfg["initial_gap_m"]
        )
        poses[:, 3:] = Rotation.from_euler(
            "z",
            np.asarray(inputs["stack_yaw_deg"][initial][:count])[:, None],
            degrees=True,
        ).as_quat()
        support_size = np.array(cfg["stack_support_size_m"])
        support_pose = np.array(
            [0, 0, cfg["stack_support_top_z_m"] - support_size[2] / 2, 0, 0, 0, 1]
        )
        origin = np.array([0, 0, cfg["stack_support_top_z_m"]])
        normal = np.array([0, 0, 1])
        downhill = np.array([1, 0, 0])
    inertia = (
        cfg["mass_kg"]
        / 12
        * np.array(
            [
                size[1] ** 2 + size[2] ** 2,
                size[0] ** 2 + size[2] ** 2,
                size[0] ** 2 + size[1] ** 2,
            ]
        )
    )
    return dict(
        body_size=size,
        body_poses=poses,
        body_inertia=inertia,
        support_size=support_size,
        support_pose=support_pose,
        surface_origin=origin,
        surface_normal=normal,
        downhill=downhill,
    )


def begin_run(
    args: argparse.Namespace, platform: str, source: Path
) -> tuple[dict, dict, dict, dict]:
    """Freeze inputs, source identity and launch information before simulation."""
    cfg = json.loads(args.config.read_text())
    input_path = args.config.parent / cfg["initial_states"]
    inputs = json.loads(input_path.read_text())
    scene = next(s for s in cfg["scenarios"] if s["id"] == args.scene)
    assert 0 <= args.episode_start < 10 and 0 < args.count <= 10 - args.episode_start
    assert args.dt > 0 and (not args.record or args.count == 1)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "run.json").exists():
        raise FileExistsError(args.output / "run.json")
    snap = args.output / "config"
    snap.mkdir()
    sources = [source, Path(__file__), args.config, input_path]
    if args.spec is not None:
        sources.append(args.spec)
    hashes = {}
    for path in sources:
        shutil.copy2(path, snap / path.name)
        hashes[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    duration = args.duration or (
        cfg["slope_duration_s"]
        if scene["kind"] == "slope"
        else cfg["stack_settle_s"] + cfg["stack_observation_s"]
    )
    args.duration = float(duration)
    assert abs(round(duration / args.dt) * args.dt - duration) < 1e-10
    repo = args.source_repo.resolve()
    packages = (
        ["embodichain", "dexsim-engine", "newton", "warp-lang", "torch", "numpy"]
        if platform == "embodichain"
        else ["isaaclab", "isaaclab-physx", "isaacsim", "warp-lang", "torch", "numpy"]
    )
    meta = dict(
        experiment="P-02",
        run_id=args.output.name,
        platform=platform,
        phase=args.phase,
        scene=scene,
        dt_s=args.dt,
        duration_s=duration,
        num_envs=1,
        initial_count=args.count,
        episode_start=args.episode_start,
        completed_episodes=0,
        status="running",
        started_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        command=[sys.executable, *sys.argv],
        python=sys.version,
        pythonpath=os.environ.get("PYTHONPATH"),
        source_repository=str(repo),
        source_commit=subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        source_status=subprocess.check_output(
            ["git", "-C", str(repo), "status", "--porcelain"], text=True
        ),
        packages={n: importlib.metadata.version(n) for n in packages},
        source_sha256=hashes,
        protocol=cfg,
        performance_benchmark=False,
        diagnostic_timing_scope="Full rollout with per-step state readback and optional recording; excludes preparation/reset; not physics throughput.",
        gpu=subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip(),
        cpu_threads={
            key: os.environ.get(key)
            for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]
        },
        state_columns=[
            "x",
            "y",
            "z",
            "qx",
            "qy",
            "qz",
            "qw",
            "vx",
            "vy",
            "vz",
            "wx",
            "wy",
            "wz",
        ],
        physics_dtype="float32",
        storage_dtype="float64",
    )
    write_run(args.output, meta)
    return cfg, inputs, scene, meta


def write_run(output: Path, meta: dict) -> None:
    """Persist completion state, including any partial-run location."""
    (output / "run.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")


def finish_run(output: Path, meta: dict) -> None:
    """Finish provenance and a three-table diagnostic run summary."""
    meta["ended_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    meta["process_peak_rss_mib"] = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    )
    write_run(output, meta)
    times = meta.get("diagnostic_wall_s_per_episode", [])
    time_ms = f"{np.median(times)*1000:.3f}" if times else "未测得"
    text = f"""# P-02 单次运行摘要\n\n{meta['run_id']}；诊断采样耗时，不作为物理吞吐。完整指标由离线分析生成。\n\n## Time & Memory\n\n| platform | cost_time_ms | cpu_delta_mb | gpu_delta_mb | peak_gpu_mb | process_peak_rss_mib |\n| --- | ---: | --- | --- | --- | ---: |\n| {meta['platform']} | {time_ms} | 未测 | 未测 | 未测 | {meta['process_peak_rss_mib']:.1f} |\n\n## Success & Other Metrics\n\n| scene | success_rate | completed_episodes | run_status |\n| --- | --- | ---: | --- |\n| {meta['scene']['id']} | 不适用：连续误差实验 | {meta['completed_episodes']} | {meta['status']} |\n\n## Leaderboard\n\n| configuration | rank | reason |\n| --- | --- | --- |\n| {meta.get('solver','unresolved')} / {meta['dt_s']} s | 不设综合排名 | 连续指标与几何事件分别比较 |\n"""
    (output / "run-summary.md").write_text(text)
    print(f'Markdown report saved: {output / "run-summary.md"}', flush=True)
