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

"""Run verified Pick/Place in the Library scan and record every RGB-D boundary."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import traceback
from uuid import uuid4

import yaml

from _recording import (
    CAMERA,
    STEP_DT,
    EpisodeCapture,
    atomic_json,
    digest,
    json_value,
    preview,
)

EXAMPLE = Path(__file__).resolve().parent
REPOSITORY = EXAMPLE.parents[2]

__all__ = ["main"]


def _write_yaml(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False))


def _prepare(args: argparse.Namespace, directory: Path) -> tuple[object, dict]:
    import dexsim
    from embodichain.lab.gym.utils.gym_utils import config_to_cfg
    from embodichain.utils.config_paths import resolve_config_path
    from embodichain.utils.utility import load_config

    preset = json.loads(
        (REPOSITORY / "examples/sim/assets/gaussian_scenes.json").read_text()
    )["library-usd"]
    subprocess.run(
        [
            sys.executable,
            str(REPOSITORY / "examples/sim/gaussian_scenes.py"),
            "--scene",
            "library-usd",
            "--download-only",
            "--data-root",
            str(args.data_root),
        ],
        check=True,
    )
    asset_root = args.data_root.resolve() / preset["cache_dir"]
    if digest(asset_root / preset["download"]) != preset["sha256"]:
        raise ValueError("Library archive hash mismatch")
    for name, expected in preset["members"].items():
        if digest(asset_root / name) != expected:
            raise ValueError(f"Library dependency hash mismatch: {name}")

    resolved = directory / "resolved"
    resolved.mkdir()
    for source in (EXAMPLE / "configs").rglob("*.yaml"):
        target = resolved / source.relative_to(EXAMPLE / "configs")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    environment = load_config(resolved / "env.yaml")
    environment["simulation"]["usd_scene"]["source"]["file_path"] = str(
        asset_root / preset["entry"]
    )
    _write_yaml(resolved / "env.yaml", environment)

    sdk = "extensions/embodichain_tasks/configs/components/"
    embodiment_path = resolve_config_path(
        sdk + "embodiments/dual_ur5_dh_pgi_140_80.yaml"
    )
    embodiment = load_config(embodiment_path)
    embodiment["sensor"] = [
        {
            "sensor_type": "Camera",
            "uid": CAMERA,
            "width": 960,
            "height": 540,
            "intrinsics": [700, 700, 480, 270],
            "extrinsics": {
                "eye": [0.8, -1.6, 1.5],
                "target": [0.8, 0, 0.8],
                "up": [0, 0, 1],
            },
            "enable_color": True,
            "enable_depth": True,
            "enable_mask": False,
            "enable_normal": False,
            "enable_position": False,
            "ground_plane_visible": False,
        }
    ]
    _write_yaml(resolved / "embodiment.rgbd.yaml", embodiment)
    policy_path = resolve_config_path(
        sdk + "execution_policies/motion_gen_verified.yaml"
    )
    shutil.copyfile(policy_path, resolved / "policy.yaml")
    task_path = resolved / "task.dual_ur5.yaml"
    cfg = config_to_cfg(load_config(task_path), source_path=task_path)
    inventory = cfg._usd_scene_inventory
    if set(inventory["resource_uids"].values()) != {"library_background"} or any(
        "Collision" in path for path in inventory["resource_source_paths"].values()
    ):
        raise ValueError(f"Library subtree imported unexpected resources: {inventory}")
    cfg.record_trajectory = True
    cfg.trajectory_auto_save = False
    cfg.init_rollout_buffer = False
    cfg.sim_cfg.headless = not args.viewer
    cfg.sim_cfg.startup_summary = "compact"
    cfg.seed = args.seed
    if cfg.num_envs != 1 or str(cfg.sim_cfg.device) != "cpu":
        raise ValueError("Library example requires one CPU physics environment")
    # Save the effective typed settings as well as runnable YAML snapshots.
    atomic_json(
        resolved / "recording.json",
        {
            "record_trajectory": True,
            "trajectory_auto_save": False,
            "init_rollout_buffer": False,
            "step_dt": STEP_DT,
            "headless": not args.viewer,
            "seed": args.seed,
        },
    )
    # config_to_cfg has already performed physical binding, catalog and lowering
    # preflight. Keep this evidence usable in installed examples without .agents.
    atomic_json(
        resolved / "preflight.json",
        {
            "valid": True,
            "gym_id": "LibraryPickPlace-DualUR5-v1",
            "usd_inventory": inventory,
        },
    )
    report = {
        "schema_version": 1,
        "status": "prepared",
        "run_uuid": directory.name,
        "requested_episodes": args.episodes,
        "seed": args.seed,
        "library": preset,
        "library_pose": environment["simulation"]["usd_scene"]["pose"],
        "usd_inventory": json_value(inventory),
        "sdk_inputs": {
            str(path): digest(path) for path in (embodiment_path, policy_path)
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("dexsim_engine", "torch", "numpy")
        },
        "python": sys.version,
        "dexsim_commit": getattr(dexsim, "__commit_id__", None),
        "example_files": {
            name: digest(EXAMPLE / name)
            for name in ("run.py", "_recording.py", "verify.py")
        },
        "source_commit": (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
            ).strip()
            if (REPOSITORY / ".git").exists()
            else None
        ),
        "config_files": {
            path.relative_to(directory).as_posix(): digest(path)
            for path in sorted(resolved.rglob("*"))
            if path.is_file()
        },
        "episodes": [],
    }
    atomic_json(directory / "run.json", report)
    return cfg, report


def _collect(
    cfg: object, args: argparse.Namespace, directory: Path, report: dict
) -> str | None:
    import gymnasium as gym
    from embodichain.lab.gym.envs.demo import execute_demo_episode
    from verify import verify_episode

    env = capture = None
    failure = None
    report["status"] = "running"
    atomic_json(directory / "run.json", report)
    try:
        env = gym.make("LibraryPickPlace-DualUR5-v1", cfg=cfg)
        target = env.unwrapped
        if (
            target.sim.get_visual_object_uid_list() != ["library_background"]
            or set(target.sim.get_rigid_object_uid_list())
            != {"cube", "support_surface"}
            or set(target.sensors) != {CAMERA}
        ):
            raise ValueError(
                "Runtime resources differ from the qualified Library scene"
            )
        if abs(target.step_dt - STEP_DT) > 1e-12:
            raise ValueError("Library control cadence must be 25 Hz")
        for index in range(args.episodes):
            capture = None
            started = time.perf_counter()
            observation, _ = env.reset(
                seed=args.seed + index, options={"save_data": False}
            )
            capture = EpisodeCapture(directory / "episodes", target, observation)
            result = execute_demo_episode(
                env,
                episode_index=index,
                step_observer=capture.observe,
            )
            if capture.directory is None:
                raise RuntimeError(
                    f"Task emitted no control steps: {result.terminal_reason}"
                )
            episode = capture.directory
            result_metadata = result.to_metadata()
            atomic_json(episode / "execution_result.json", result_metadata)
            if not result.completed or not result.all_success or capture.error:
                raise RuntimeError(
                    f"Pick/Place or capture failed: {result.terminal_reason}; {capture.error}"
                )
            steps = result.lengths[0]
            capture.finish(steps)
            rgbd = {
                "schema_version": 1,
                "media": "media/manifest.json",
                "captures": "captures.jsonl",
                "media_sha256": digest(episode / "media/manifest.json"),
                "captures_sha256": digest(episode / "captures.jsonl"),
            }
            target.save_trajectory(
                str(episode / "trajectory.pt"), extra_meta={"rgbd_recording": rgbd}
            )
            preview(episode, result_metadata)
            elapsed = time.perf_counter() - started
            completion = {
                "schema_version": 1,
                "status": "complete",
                "episode_uuid": capture.episode_uuid,
                "steps": steps,
                "frames": capture.count,
                "step_dt": STEP_DT,
                "seed": args.seed + index,
                "metrics": {
                    "wall_seconds": elapsed,
                    "control_steps_per_second": steps / elapsed,
                    "capture_seconds": capture.capture_seconds,
                    "peak_buffered_bytes": capture.peak_buffer_bytes,
                    "max_buffer_bytes": 64 * 1024 * 1024,
                    "process_peak_rss_bytes": resource.getrusage(
                        resource.RUSAGE_SELF
                    ).ru_maxrss
                    * 1024,
                },
                "files": {
                    name: digest(episode / name)
                    for name in (
                        "trajectory.pt",
                        "execution_result.json",
                        "captures.jsonl",
                        "media/manifest.json",
                        "preview.png",
                    )
                },
            }
            checked = verify_episode(episode, completion)
            atomic_json(episode / "episode.json", completion)
            report["episodes"].append(
                {
                    "path": episode.relative_to(directory).as_posix(),
                    "sha256": digest(episode / "episode.json"),
                    "validation": checked,
                }
            )
            atomic_json(directory / "run.json", report)
            print(
                f"Episode {index + 1}/{args.episodes}: {steps} steps, {capture.count} RGB-D frames, {elapsed:.2f}s",
                flush=True,
            )
        report["status"] = "complete"
    except BaseException:
        failure = traceback.format_exc()
        report["status"] = "failed"
        report["failed_attempt"] = {
            "attempt": len(report["episodes"]),
            "error": failure,
            "path": (
                str(capture.directory.relative_to(directory))
                if capture is not None and capture.directory is not None
                else None
            ),
        }
    finally:
        try:
            if capture is not None:
                capture.close()
            if env is not None:
                env.unwrapped.close(exit_process=False)
        except BaseException:
            cleanup_error = traceback.format_exc()
            failure = (failure or "") + cleanup_error
            report["status"] = "failed"
            report["cleanup_error"] = cleanup_error
        atomic_json(directory / "run.json", report)
    return failure


def main() -> None:
    """Prepare or collect a fresh Library recording run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            os.environ.get(
                "EMBODICHAIN_DATA_ROOT", Path.home() / ".cache/embodichain_data"
            )
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/library_pick_place")
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    os.environ.setdefault("EMBODICHAIN_SIM_EXIT_PROCESS", "0")
    if not args.viewer:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    directory = args.output.expanduser().resolve() / uuid4().hex
    directory.mkdir(parents=True)
    print(f"Run directory: {directory}", flush=True)
    try:
        cfg, report = _prepare(args, directory)
    except Exception:
        atomic_json(
            directory / "run.json",
            {
                "schema_version": 1,
                "status": "failed",
                "stage": "prepare",
                "requested_episodes": args.episodes,
                "episodes": [],
                "error": traceback.format_exc(),
            },
        )
        raise
    if args.prepare_only:
        print(f"Prepared: {directory / 'resolved/task.dual_ur5.yaml'}", flush=True)
        return
    failure = _collect(cfg, args, directory, report)
    # All native scene locals and caught exception tracebacks have unwound.
    gc.collect()
    from embodichain.lab.sim import SimulationManager

    SimulationManager.flush_cleanup_queue()
    if failure is not None:
        raise RuntimeError(failure)
    from verify import verify_run

    print(json.dumps(verify_run(directory), indent=2), flush=True)


if __name__ == "__main__":
    main()
