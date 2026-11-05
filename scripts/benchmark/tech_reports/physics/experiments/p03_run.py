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

"""Execute a frozen P-03 input through either platform in its own process.

Run: python -m scripts.benchmark.tech_reports.physics.experiments.p03_run --help
"""

from __future__ import annotations

import os

import argparse
from datetime import datetime
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import time
import traceback


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--platform", choices=["embodichain", "isaaclab"], required=True
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shape", choices=["cube", "cylinder"], default="cube")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--latency", action="store_true")
    parser.add_argument("--mode", choices=["physics", "env"], default="physics")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output / "run.json").exists()
    cfg = json.loads(args.config.read_text())
    assert abs(cfg["control_dt"] - 4 * cfg["dt"]) < 1e-12
    cfg["execution_mode"] = args.mode
    assert not (args.record and args.mode == "env")
    cfg["shape_dir"] = str(
        (args.config.parent / cfg.get("shapes_subdir", "shapes-v1")).resolve()
    )
    meta = dict(
        experiment="P-03",
        status="running",
        platform=args.platform,
        started_at=datetime.now().astimezone().isoformat(),
        arguments=vars(args),
        config=cfg,
        completed_episodes=0,
        state_layout={
            "joint_position": [0, 9],
            "joint_velocity": [9, 18],
            "tcp_pose_xyzw": [18, 25],
            "leftfinger_pose_xyzw": [25, 32],
            "rightfinger_pose_xyzw": [32, 39],
            "object_pose_xyzw": [39, 46],
            "object_linear_velocity": [46, 49],
            "object_angular_velocity": [49, 52],
        },
        timing_scope="Public target/force writes, physics step, public state reads and GPU trajectory sampling. Setup, warmup, reset, CPU copies and file output excluded. Rendering and synchronized per-step latency are separate runs.",
    )
    snapshot = args.output / "config"
    snapshot.mkdir()
    source_root = Path(
        os.environ.get("SIM_REPORT_ROOT", Path(__file__).resolve().parents[1])
    ).resolve()
    files = [
        Path(__file__),
        source_root / f"platforms/{args.platform}/p03_grasp.py",
        args.config,
        args.inputs / "commands.npz",
        args.inputs / "kinematics.json",
        Path(cfg["shape_dir"]) / f"{args.shape}.usda",
    ]
    meta["source_files"] = {}
    for path in files:
        shutil.copy2(path, snapshot / path.name)
        meta["source_files"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    meta["source_commits"] = {
        name: subprocess.check_output(
            ["git", "-C", f"/home/oem/code/{name}", "rev-parse", "HEAD"], text=True
        ).strip()
        for name in ["EmbodiChain", "IsaacLab", "dexsim"]
    }
    meta["gpu_inventory"] = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv",
        ],
        text=True,
    )

    def save() -> None:
        (args.output / "run.json").write_text(
            json.dumps(meta, indent=2, default=str) + "\n"
        )

    save()
    app = None
    simulation = None
    try:
        if args.platform == "isaaclab":
            from isaaclab.app import AppLauncher

            app = AppLauncher(
                headless=True, visualizer="none", enable_cameras=args.record
            ).app
        import numpy as np
        import torch

        inputs = dict(np.load(args.inputs / "commands.npz"))
        for key in (
            ["embodichain", "embodichain_robots", "embodichain_tasks", "dexsim"]
            if args.platform == "embodichain"
            else ["isaaclab", "isaacsim"]
        ):
            meta.setdefault("module_origins", {})[key] = importlib.util.find_spec(
                key
            ).origin
        meta["packages"] = {
            n: importlib.metadata.version(n)
            for n in (
                ["dexsim-engine", "newton", "warp-lang", "torch"]
                if args.platform == "embodichain"
                else ["isaaclab", "isaacsim", "warp-lang", "torch"]
            )
        }
        if args.platform == "embodichain":
            import dexsim.kit.usd.parser as usd_parser

            assert hasattr(
                usd_parser, "_apply_physics_mimic_joints"
            ), "P-03 requires MR1501 parser"
            meta["parser"] = {
                "path": usd_parser.__file__,
                "sha256": hashlib.sha256(
                    Path(usd_parser.__file__).read_bytes()
                ).hexdigest(),
            }
        module = importlib.import_module(
            f"scripts.benchmark.tech_reports.physics.platforms.{args.platform}.p03_grasp"
        )
        simulation = module.Simulation(
            cfg, inputs, args.shape, args.num_envs, args.record, args.output
        )
        meta["model_readback"] = simulation.meta
        save()
        device = simulation.device
        n = args.num_envs
        dt = cfg["dt"]
        q_all = torch.tensor(inputs["q_target"], dtype=torch.float32, device=device)
        forces = torch.tensor(inputs["force_world"], dtype=torch.float32, device=device)
        zero = torch.zeros((n, 3), device=device)
        if args.calibrate:
            simulation.reset(0)
            pose = torch.tensor([[1.5, 0.5, 3.0, 0, 0, 0, 1]], device=device).repeat(
                n, 1
            )
            if args.platform == "embodichain":
                simulation.target.set_local_pose(pose)
            else:
                pose[:, :3] += simulation.origins
                simulation.target.write_root_pose_to_sim_index(root_pose=pose)
                simulation.scene.update(0)
            states = [simulation.state().cpu().numpy()]
            applied = []
            f = zero.clone()
            f[:, 0] = torch.arange(1, n + 1, device=device) * 0.1
            for k in range(40):
                value = f if k < 20 else zero
                simulation.step(value)
                states.append(simulation.state().cpu().numpy())
                applied.append(value.cpu().numpy())
            np.savez_compressed(
                args.output / "force_calibration.npz",
                state=states,
                force=applied,
                t=np.arange(41) * dt,
            )
            meta["completed_episodes"] = 1
        else:
            simulation.reset(args.episode_start)
            if args.mode == "env":
                for _ in range(50):
                    simulation.control_step(
                        q_all[args.episode_start, 0].expand(n, -1),
                        zero[None].expand(4, -1, -1),
                    )
            else:
                for _ in range(200):
                    simulation.step(zero)
            steps = round(cfg["duration_s"] / dt)
            times = np.arange(steps + 1) * dt
            for episode in range(args.episode_start, args.episode_start + args.count):
                simulation.reset(episode)
                trajectory = torch.empty((steps + 1, n, 52), device=device)
                frames = []
                latencies = []
                if args.record:
                    assert args.count == 1 and n == 1
                    camera_path = args.output / "native_camera"
                    camera_path.mkdir()
                    from PIL import Image
                torch.cuda.synchronize()
                epoch_start = time.time()
                tic = time.perf_counter()
                trajectory[0] = simulation.state()
                for k in range(0, steps, 4 if args.mode == "env" else 1):
                    if args.latency:
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                    if args.mode == "env":
                        simulation.control_step(
                            q_all[episode, k // 4].expand(n, -1),
                            forces[k : k + 4, None, :].expand(-1, n, -1),
                        )
                        trajectory[k + 1 : k + 5] = simulation.last_states
                    else:
                        if k % 4 == 0:
                            simulation.command(q_all[episode, k // 4].expand(n, -1))
                        simulation.step(forces[k].expand(n, -1))
                        trajectory[k + 1] = simulation.state()
                    if args.latency:
                        torch.cuda.synchronize()
                        latencies.append(time.perf_counter() - start)
                    if args.record and (k + 1) % 10 == 0:
                        rgb, cam = simulation.render()
                        filename = f"frame_{len(frames):04d}.png"
                        Image.fromarray(rgb).save(camera_path / filename)
                        frames.append(
                            dict(file=filename, step=k + 1, time_s=(k + 1) * dt, **cam)
                        )
                torch.cuda.synchronize()
                wall = time.perf_counter() - tic
                epoch_end = time.time()
                finite = torch.isfinite(trajectory).all().item()
                # Retain all channels for every replica, including anomalous states.
                raw = trajectory.cpu().numpy()
                np.save(args.output / f"episode_{episode:02d}_state.npy", raw)
                np.save(args.output / f"episode_{episode:02d}_time.npy", times)
                assert finite, "Nonfinite physical state; raw trajectory retained"
                np.save(
                    args.output / f"episode_{episode:02d}_latency_s.npy",
                    np.asarray(latencies),
                )
                meta.setdefault("episodes", []).append(
                    dict(
                        initial=episode,
                        wall_s=wall,
                        epoch_start=epoch_start,
                        epoch_end=epoch_end,
                        physics_steps=steps,
                        control_steps=steps // 4,
                        env_physics_steps_s=n * steps / wall,
                        recorded=args.record,
                        mode=args.mode,
                        env_control_steps_s=(
                            n * (steps // 4) / wall if args.mode == "env" else None
                        ),
                        per_step_synchronized=args.latency,
                    )
                )
                if args.record:
                    from scripts.benchmark.tech_reports.physics.experiments.p02_media_inputs import (
                        encode,
                    )

                    meta["capture"] = encode(
                        camera_path,
                        frames,
                        cfg["capture"],
                        f"{args.platform} live native camera",
                    )
                meta["completed_episodes"] += 1
                save()
                print(f"Completed initial {episode}; {wall:.3f} s", flush=True)
        meta["status"] = "completed"
    except BaseException as exc:
        meta["status"] = "failed"
        meta["failure"] = repr(exc)
        traceback.print_exc()
        raise
    finally:
        meta["ended_at"] = datetime.now().astimezone().isoformat()
        save()
        if simulation is not None:
            simulation.close()
        if app is not None:
            app.close()


if __name__ == "__main__":
    main()
