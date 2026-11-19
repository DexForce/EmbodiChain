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

"""Qualify live batched Pick/Place, camera isolation and every RGB boundary."""

from __future__ import annotations

import argparse
import gc
import json
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np

from _prepare import SCENES, digest, prepare, write_json
from _recording import CAMERAS, json_value

__all__ = ["main"]


def _isolation(env: Any) -> dict[str, Any]:
    """Hide one arena's background and measure changes in both camera batches."""
    visual = env.sim.get_visual_object("scan_background")
    resources = visual.get_visual_elements()[0]["resource_ids"]
    if len(resources) != env.num_envs or len(set(resources)) != 1:
        raise ValueError("Gaussian arenas must share one immutable resource")

    def capture() -> dict[str, np.ndarray]:
        env.sim.render_camera_group(
            [group for name in CAMERAS for group in env.sensors[name].group_ids]
        )
        result = {}
        for name in CAMERAS:
            camera = env.sensors[name]
            camera.update(fetch_only=True)
            result[name] = camera.get_data()["color"][..., :3].cpu().numpy().copy()
        return result

    before = capture()
    visual.set_visible(False, [1])
    try:
        hidden = capture()
    finally:
        visual.reset([1])
    differences = {}
    for name in CAMERAS:
        error = np.abs(before[name].astype(np.int16) - hidden[name].astype(np.int16))
        means = error.mean(axis=(1, 2, 3))
        if means[1] < 5 or np.max(np.delete(means, 1)) > 2:
            raise ValueError(f"Cross-arena RGB isolation failed: {name}: {means}")
        differences[name] = means.tolist()
    return {"hidden_env_id": 1, "mean_absolute_rgb_changes": differences}


def _collect(cfg: Any, report: dict[str, Any], root: Path) -> dict[str, Any]:
    import gymnasium as gym
    from PIL import Image
    from embodichain.data_pipeline.recording.media import MediaSpool
    from embodichain.lab.gym.envs.demo import execute_demo_episode

    env = gym.make(report["task_id"], cfg=cfg)
    host = env.unwrapped
    spools = []
    states = []
    previous = {name: -1 for name in CAMERAS}
    fixed_pose = None
    frame_count = 0
    try:
        env.reset(seed=cfg.seed, options={"save_data": False})
        report["isolation"] = _isolation(host)
        observation, _ = env.reset(seed=cfg.seed, options={"save_data": False})
        start = host.sim.simulation_time
        mount = host.sensors["cam_wrist"].cfg.extrinsics.transformation.cpu().numpy()
        for row in range(host.num_envs):
            spools.append(
                MediaSpool(
                    root / f"env_{row}" / "media",
                    max_buffer_bytes=32 * 1024 * 1024,
                    parent_episode_id=f"{root.name}-env-{row}",
                    calibration={"env_id": row, "wrist_mount_pose": mount.tolist()},
                )
            )
        with (root / "captures.jsonl").open("x") as stream:

            def capture(active: Any, action: Any, obs: Any) -> None:
                nonlocal frame_count, fixed_pose
                captures, frames = {}, {}
                physical = (
                    host.sim.get_rigid_object("cube").body_state.cpu().numpy().copy()
                )
                hand = (
                    host.robot.get_link_pose("fr3_hand", to_matrix=True).cpu().numpy()
                )
                for name in CAMERAS:
                    camera = host.sensors[name]
                    meta = json_value(camera.get_capture_metadata())
                    status, contract = meta["status"], meta["output_contract"]
                    if not (
                        meta["valid"]
                        and status["valid"]
                        and status["completed"]
                        and status["outcome"] == "SUCCESS"
                        and contract["completed"]
                        and contract["same_frame_identity"]
                        and meta["frame_id"]
                        == status["frame_index"]
                        == contract["frame_index"]
                        and meta["frame_id"] > previous[name]
                        and meta["camera_id"] == name
                        and meta["env_ids"] == list(range(host.num_envs))
                        and status["camera_count"] == host.num_envs
                        and all(c["selected_nodes"] > 0 for c in status["cameras"])
                    ):
                        raise ValueError(
                            f"Invalid completed Gaussian camera batch: {name}"
                        )
                    if (
                        abs(
                            meta["simulation_time"] - start - frame_count * host.step_dt
                        )
                        > 1e-7
                    ):
                        raise ValueError("Camera and physical control clocks differ")
                    rgb = obs["sensor"][name]["color"][..., :3].cpu().numpy()
                    if (
                        rgb.shape != (host.num_envs, 540, 960, 3)
                        or rgb.dtype != np.uint8
                    ):
                        raise ValueError(
                            "Expected native uint8 960x540 RGB for every arena"
                        )
                    if np.any(rgb.std(axis=(1, 2, 3)) < 5):
                        raise ValueError("Blank RGB frame")
                    poses = np.asarray(meta["arena_poses"])
                    if name == "cam_third":
                        if fixed_pose is None:
                            fixed_pose = poses.copy()
                        np.testing.assert_allclose(poses, fixed_pose, atol=1e-6, rtol=0)
                    else:
                        np.testing.assert_allclose(
                            poses, hand @ mount, atol=2e-6, rtol=0
                        )
                    if frame_count == 0:
                        for row in range(host.num_envs):
                            point = np.linalg.solve(
                                poses[row], np.r_[physical[row, :3], 1]
                            )
                            k = np.asarray(meta["intrinsics"][row])
                            u = int(round(k[0, 0] * point[0] / -point[2] + k[0, 2]))
                            v = int(round(k[1, 2] - k[1, 1] * point[1] / -point[2]))
                            pixel = np.median(
                                rgb[row, v - 2 : v + 3, u - 2 : u + 3], axis=(0, 1)
                            )
                            if not (pixel[0] > 64 and pixel[0] > 2 * max(pixel[1:])):
                                raise ValueError(
                                    f"Projected cube is not red RGB: {name}/{row}"
                                )
                            Image.fromarray(rgb[row]).save(
                                root / f"env_{row}" / f"initial_{name}.png"
                            )
                    previous[name] = meta["frame_id"]
                    captures[name], frames[name] = meta, rgb
                for row, spool in enumerate(spools):
                    spool.append(
                        frame_count * host.step_dt,
                        {f"{name}/rgb": frames[name][row] for name in CAMERAS},
                    )
                stream.write(
                    json.dumps(
                        {
                            "frame": frame_count,
                            "active": active,
                            "captures": captures,
                            "cube_state": physical.tolist(),
                            "hand_pose": hand.tolist(),
                        }
                    )
                    + "\n"
                )
                states.append(physical)
                frame_count += 1
                if frame_count % 100 == 0:
                    print(
                        f"CAPTURE {frame_count}: {host.num_envs} arenas x 2 cameras",
                        flush=True,
                    )

            capture(None, None, observation)
            result = execute_demo_episode(env, step_observer=capture)
        write_json(root / "execution_result.json", result.to_metadata())
        if not result.completed or not result.all_success:
            raise ValueError(f"Batched Pick/Place failed: {result.terminal_reason}")
        if frame_count != max(result.lengths) + 1:
            raise ValueError("Expected T+1 RGB boundaries")
        host.save_trajectory(str(root / "trajectory.pt"), purpose="diagnostic")
        for spool in spools:
            spool.finish()
        return {
            "states": np.asarray(states),
            "frames": frame_count,
            "lengths": list(result.lengths),
        }
    finally:
        host.close(exit_process=False)


def main() -> None:
    """Validate one official deployment and export evidence for every arena."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=SCENES, required=True)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/gaussian_parallel")
    )
    args = parser.parse_args()
    if args.num_envs < 2:
        parser.error("--num-envs must be at least two for the isolation check")
    os.environ.setdefault("EMBODICHAIN_SIM_EXIT_PROCESS", "0")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S")
    root = args.output.resolve() / f"{args.scene}-{stamp}-{uuid4().hex[:8]}"
    root.mkdir(parents=True)
    print(f"RUN {root}", flush=True)
    cfg, report = prepare(args.scene, root, args.seed, 8, False)
    cfg.num_envs = args.num_envs
    overrides = root / "resolved/runtime_overrides.json"
    values = json.loads(overrides.read_text()) | {"num_envs": args.num_envs}
    write_json(overrides, values)
    report["configuration_hashes"][overrides.name] = digest(overrides)
    report["num_envs"] = args.num_envs
    report["status"] = "running"
    write_json(root / "run.json", report)
    try:
        captured = _collect(cfg, report, root)
        gc.collect()
        from embodichain.lab.sim import SimulationManager

        SimulationManager.flush_cleanup_queue()
        from _export import export_video
        from _bundle import _video_integrity

        program = json.loads((root / "execution_result.json").read_text())
        report["execution_success"] = program["success"]
        report["frames_per_camera"] = captured["frames"]
        report["rgb_images"] = captured["frames"] * args.num_envs * 2
        report["rows"] = []
        target = np.asarray(report["physics"]["target_center"])
        for row in range(args.num_envs):
            state = captured["states"][:, row]
            lift = float(state[:, 2].max() - state[0, 2])
            error = float(np.linalg.norm(state[-1, :3] - target))
            if lift < 0.04 or error > 0.02 or np.linalg.norm(state[-1, 7:10]) > 0.01:
                raise ValueError(
                    f"Independent physical acceptance failed for env {row}"
                )
            metrics = {
                "frames": captured["frames"],
                "max_lift_frame": int(state[:, 2].argmax()),
            }
            directory = root / f"env_{row}"
            export_video(directory, metrics)
            report["rows"].append(
                {
                    "env_id": row,
                    "actions": captured["lengths"][row],
                    "lift_m": lift,
                    "position_error_m": error,
                    "video_integrity": _video_integrity(directory, captured["frames"]),
                }
            )
            print(f"AUDITED env_{row}", flush=True)
        report["status"] = "complete"
    except BaseException:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        raise
    finally:
        write_json(root / "run.json", report)
    print(f"COMPLETE {root}", flush=True)


if __name__ == "__main__":
    main()
