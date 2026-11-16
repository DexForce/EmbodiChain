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

"""Run measured physical Pick/Place and capture every RGB state boundary."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo


from _prepare import SCENES, digest, prepare, write_json
from _recording import CAMERAS, STEP_DT, Capture

__all__ = ["main"]


def export_video(episode: Path, metrics: dict[str, Any]) -> None:
    from _export import export_video as export_pair

    export_pair(episode, metrics)


def collect(cfg: Any, report: dict[str, Any], root: Path, seed: int) -> str | None:
    import gymnasium as gym
    from embodichain.lab.gym.envs.demo import execute_demo_episode
    from _verify import verify_episode

    env = capture = None
    started = time.perf_counter()
    failure = None
    report["status"] = "running"
    write_json(root / "run.json", report)
    try:
        env = gym.make(report["task_id"], cfg=cfg)
        target = env.unwrapped
        if abs(target.step_dt - STEP_DT) > 1e-12:
            raise ValueError("Expected 25 Hz physical control")
        cube = target.sim.get_rigid_object("cube")
        if cube.is_non_dynamic or abs(float(cube.get_mass()[0]) - 0.05) > 1e-6:
            raise ValueError(
                "Cube is not the expected unconstrained 50 g dynamic rigid body"
            )
        report["runtime_physics"] = {
            "cube_body_type": cube.body_type,
            "cube_mass_kg": float(cube.get_mass()[0]),
            "support_static": target.sim.get_rigid_object("support_surface").is_static,
        }
        observation, _ = env.reset(seed=seed, options={"save_data": False})
        capture = Capture(root, target, observation)
        print(
            "RESET completed; starting verified FR3 Task Program with two cameras",
            flush=True,
        )
        result = execute_demo_episode(
            env,
            step_observer=capture.observe,
            on_segment_completed=capture.segment_completed,
        )
        if capture.directory is None:
            raise RuntimeError(f"Task produced no steps: {result.terminal_reason}")
        episode = capture.directory
        write_json(episode / "execution_result.json", result.to_metadata())
        capture.finish(result.lengths[0])
        report["episode"] = episode.relative_to(root).as_posix()
        if not result.completed or not result.all_success:
            target.save_trajectory(
                str(episode / "trajectory.failed.pt"), purpose="diagnostic"
            )
            raise RuntimeError(f"Physical Pick/Place failed: {result.terminal_reason}")
        reference = {
            "media": "media/manifest.json",
            "captures": "captures.jsonl",
            "media_sha256": digest(episode / "media/manifest.json"),
            "captures_sha256": digest(episode / "captures.jsonl"),
        }
        target.save_trajectory(
            str(episode / "trajectory.pt"), extra_meta={"rgb_recording": reference}
        )
        metrics = verify_episode(episode, result.lengths[0])
        write_json(episode / "physical_metrics.json", metrics)
        report["metrics"] = metrics
        report["capture"] = {
            "frames": capture.count,
            "peak_buffered_bytes": capture.peak_buffer_bytes,
            "max_buffer_bytes": 64 * 1024 * 1024,
            "capture_seconds": capture.capture_seconds,
        }
        report["wall_seconds_before_video"] = time.perf_counter() - started
        report["status"] = "captured"
        print("PHYSICAL " + json.dumps(metrics), flush=True)
    except BaseException:
        failure = traceback.format_exc()
        report["status"] = "failed"
        report["error"] = failure
        if env is not None and capture is not None and capture.directory is not None:
            report["episode"] = capture.directory.relative_to(root).as_posix()
            try:
                write_json(
                    capture.directory / "failure_metadata.json",
                    target.get_demo_episode_metadata(0),
                )
                target.save_trajectory(
                    str(capture.directory / "trajectory.failed.pt"),
                    purpose="diagnostic",
                )
                if capture.error is None:
                    capture.finish()
            except Exception:
                report["diagnostic_error"] = traceback.format_exc()
    finally:
        if capture is not None:
            capture.close()
        if env is not None:
            env.unwrapped.close(exit_process=False)
        write_json(root / "run.json", report)
    return failure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=SCENES, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/gaussian_pick_place")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--spp", type=int, default=8)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("EMBODICHAIN_SIM_EXIT_PROCESS", "0")
    if not args.viewer:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S")
    root = args.output.resolve() / f"{args.scene}-{stamp}-{uuid4().hex[:8]}"
    root.mkdir(parents=True)
    print(f"RUN {root}", flush=True)
    try:
        cfg, report = prepare(args.scene, root, args.seed, args.spp, args.viewer)
    except BaseException:
        write_json(
            root / "run.json",
            {
                "status": "failed",
                "stage": "prepare",
                "scene": args.scene,
                "error": traceback.format_exc(),
            },
        )
        raise
    if args.prepare_only:
        print("PREPARED " + str(root / "resolved" / report["task_config"]), flush=True)
        return
    failure = collect(cfg, report, root, args.seed)
    gc.collect()
    from embodichain.lab.sim import SimulationManager

    SimulationManager.flush_cleanup_queue()
    if failure:
        raise RuntimeError(failure)
    episode = root / report["episode"]
    try:
        export_video(episode, report["metrics"])
        names = [
            "trajectory.pt",
            "execution_result.json",
            "physical_metrics.json",
            "captures.jsonl",
            "media/manifest.json",
            "sensor_full.mp4",
            "cam_third.mp4",
            "cam_wrist.mp4",
            "preview.jpg",
        ]
        marker = {
            "status": "complete",
            "episode_uuid": episode.name,
            "steps": report["metrics"]["steps"],
            "frames": report["metrics"]["frames"],
            "step_dt": STEP_DT,
            "files": {name: digest(episode / name) for name in names},
        }
        write_json(episode / "episode.json", marker)
        report["status"] = "complete"
        report["video"] = (episode / "sensor_full.mp4").relative_to(root).as_posix()
        write_json(root / "run.json", report)
        print("COMPLETE " + str(episode / "sensor_full.mp4"), flush=True)
    except BaseException:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        write_json(root / "run.json", report)
        raise


if __name__ == "__main__":
    main()
