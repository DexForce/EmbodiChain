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

"""Generate a qualified Task Program trajectory, then evaluate fluid Gym replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from embodichain.lab.gym.utils.gym_utils import config_to_cfg, load_trajectory
from embodichain.lab.gym.utils.registration import make, discover_task_packages
from embodichain.lab.gym.envs.demo import execute_demo_episode
from embodichain.lab.sim import SimulationManager
from embodichain.lab.sim.cfg import RenderCfg
from embodichain.utils.config_paths import resolve_config_path
from embodichain.utils.utility import load_config

__all__ = ["main"]

_TASK = "embodichain_tasks/configs/tasks/manipulation/tableware/pour_cup/"


def _generate(source_path: Path, output: Path) -> None:
    declaration = load_config(str(source_path))
    cfg = config_to_cfg(declaration, source_path=source_path)
    if cfg.task_program is None:
        raise ValueError("Source deployment must select a Task Program.")
    cfg.num_envs = 1
    cfg.sim_cfg.headless = True
    cfg.sim_cfg.render_cfg = RenderCfg(renderer="hybrid")
    cfg.enable_sensor = False
    if hasattr(cfg.events, "record_demo"):
        delattr(cfg.events, "record_demo")
    cfg.record_trajectory = True
    env = make(declaration["id"], cfg=cfg)
    try:
        env.reset(seed=0)
        result = execute_demo_episode(env)
        report = result.to_metadata()
        (output / "task_program_result.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        if not result.completed or not result.all_success:
            raise RuntimeError("Task Program failed; fluid replay will not start.")
        path = output / "task_program_trajectory.pt"
        env.save_trajectory(str(path))
        data = load_trajectory(path)
        poses = data["states"]["rigid_objects", "cup", "pose"][0].cpu().numpy()
        lift = float(np.max(poses[:, 2]) - poses[0, 2])
        z_axis = Rotation.from_quat(poses[:, 3:]).as_matrix()[:, :, 2]
        tilt = float(np.rad2deg(np.arccos(np.clip(z_axis[:, 2], -1, 1))).max())
        return_error = float(np.linalg.norm(poses[-1, :3] - poses[0, :3]))
        accepted = lift >= 0.05 and tilt >= 45.0 and return_error <= 0.03
        qualification = {
            "kind": "task_program",
            "accepted": accepted,
            "source_config": str(source_path),
            "program_hash": data["meta"].get("program_hash"),
            "lift_m": lift,
            "max_tilt_deg": tilt,
            "return_error_m": return_error,
            "task_program_completed": result.completed,
        }
        data["meta"]["fluid_source"] = qualification
        torch.save(data, path)
        (output / "source_qualification.json").write_text(
            json.dumps(qualification, indent=2) + "\n"
        )
        if not accepted:
            raise RuntimeError(
                "Measured cup lift/tilt/return failed source qualification."
            )
        print(json.dumps(qualification), flush=True)
    finally:
        env.close(exit_process=False)
    del env
    SimulationManager.flush_cleanup_queue()


def _replay(args, trajectory: Path, output: Path) -> None:
    from embodichain.lab.gym.envs.fluid_replay import FluidReplayEnv

    path = resolve_config_path(args.replay_task).resolve()
    cfg = config_to_cfg(load_config(str(path)), source_path=path)
    cfg.num_envs = args.num_envs
    cfg.sim_cfg.headless = True
    cfg.sim_cfg.render_cfg = RenderCfg(renderer=args.renderer)
    if args.renderer == "no-render":
        cfg.enable_sensor = False
        cfg.events.record_camera = None
    else:
        cfg.events.record_camera.params["save_path"] = str(output / "videos")
        cfg.events.record_camera.params["max_env_num"] = cfg.num_envs
    if args.particle_spacing is not None:
        # Resample the configured fill volume at the smoke-test resolution.
        fluid = cfg.fluid_object[0]
        if fluid.spawn.kind == "points":
            raise ValueError(
                "particle-spacing override requires a volume source; imported points have fixed sampling."
            )
        fluid.particle_spacing = args.particle_spacing
        fluid.particle_radius = 0.3 * args.particle_spacing
        cfg.sim_cfg.physics_cfg.solver_cfg["particle_support_radius"] = (
            2.25 * args.particle_spacing
        )
    env = FluidReplayEnv(cfg, trajectory=str(trajectory), speed=args.speed)
    try:
        obs, info = env.reset(seed=0)
        duration = (
            env.replay_duration
            if args.seconds is None
            else min(args.seconds, env.replay_duration)
        )
        steps = int(np.ceil(duration / env.step_dt))
        action = torch.zeros(
            (env.num_envs, len(env.active_joint_ids)), device=env.device
        )
        samples = []
        started = time.perf_counter()
        for index in range(steps):
            obs, reward, terminated, truncated, info = env.step(action)
            if args.partial_reset_step is not None and index == args.partial_reset_step:
                before = (
                    env.sim.get_fluid_object("water").data.particle_state_w[1].clone()
                )
                env.reset(
                    options={
                        "reset_ids": torch.tensor([0], device=env.device),
                        "save_data": False,
                    }
                )
                torch.testing.assert_close(
                    env.sim.get_fluid_object("water").data.particle_state_w[1],
                    before,
                    rtol=0,
                    atol=0,
                )
            if index % max(1, round(1 / env.step_dt)) == 0 or index == steps - 1:
                from embodichain.lab.gym.envs.managers.observations import (
                    get_fluid_summary,
                    get_fluid_volume_in_region,
                )
                from embodichain.lab.gym.envs.managers.cfg import SceneEntityCfg

                summary = get_fluid_summary(env, obs, SceneEntityCfg(uid="water"))
                received = get_fluid_volume_in_region(
                    env,
                    obs,
                    [SceneEntityCfg(uid="water"), SceneEntityCfg(uid="receiver")],
                    lower=[-0.1, -0.1, 0.0045],
                    upper=[0.1, 0.1, 0.0675],
                    sphere_center=[0, 0, 0.088],
                    sphere_radius=0.093,
                )
                if not torch.isfinite(summary).all() or not torch.all(
                    summary[:, -1] == 1
                ):
                    raise RuntimeError("Fluid state became non-finite.")
                sample = {
                    "time_s": info["fluid_replay_time_s"].cpu().tolist(),
                    "received_ml": (received[:, 0] * 1e6).cpu().tolist(),
                    "summary": summary.cpu().tolist(),
                }
                samples.append(sample)
                print(json.dumps(sample), flush=True)
        videos = []
        if args.renderer != "no-render":
            # Closing discards uncommitted recordings. Commit the last episode
            # explicitly, including short qualification runs without a reset.
            env.event_manager.get_functor("record_camera").save_and_clear()
            name = cfg.events.record_camera.params["name"]
            videos = [
                str(Path("videos") / f"episode_0_{name}_env{env_id}.mp4")
                for env_id in range(env.num_envs)
            ]
        report = {
            "source": env.source_qualification,
            "coupling": "one-way measured Task Program state replay",
            "num_envs": env.num_envs,
            "renderer": args.renderer,
            "videos": videos,
            "video_fps": env.control_frequency if videos else None,
            "particle_count_per_env": env.sim.get_fluid_object("water").num_particles,
            "fluid_source": cfg.fluid_object[0].spawn.to_dict(),
            "particle_spacing_m": cfg.fluid_object[0].particle_spacing,
            "density_kg_m3": cfg.fluid_object[0].density,
            "partial_reset_step": args.partial_reset_step,
            "replay_complete": info["fluid_replay_finished"].cpu().tolist(),
            "wall_seconds": time.perf_counter() - started,
            "samples": samples,
        }
        (output / "fluid_gym_result.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
    finally:
        env.close(exit_process=False)
    del env
    SimulationManager.flush_cleanup_queue()


def main() -> None:
    """Run Task Program source generation and independent fluid evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-task", default=_TASK + "task.franka.newton.yaml")
    parser.add_argument("--replay-task", default=_TASK + "task.franka.fluid.yaml")
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/fluid-task-program")
    )
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--speed", type=float, default=2.0)
    parser.add_argument(
        "--renderer", choices=["hybrid", "fast-rt", "no-render"], default="hybrid"
    )
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--particle-spacing", type=float)
    parser.add_argument("--partial-reset-step", type=int)
    parser.add_argument(
        "--phase",
        choices=["all", "generate", "replay"],
        default="all",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    import math

    if args.num_envs < 1 or not math.isfinite(args.speed) or args.speed <= 0:
        parser.error("num-envs and speed must be positive")
    if args.seconds is not None and (
        not math.isfinite(args.seconds) or args.seconds <= 0
    ):
        parser.error("seconds must be finite and positive")
    if args.particle_spacing is not None and (
        not math.isfinite(args.particle_spacing) or args.particle_spacing <= 0
    ):
        parser.error("particle-spacing must be finite and positive")
    if args.partial_reset_step is not None and (
        args.num_envs < 2 or args.partial_reset_step < 0
    ):
        parser.error(
            "partial-reset-step requires at least two environments and a nonnegative step"
        )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    discover_task_packages()
    source = resolve_config_path(args.source_task).resolve()
    if args.phase == "generate":
        _generate(source, output)
        return
    trajectory = (
        args.trajectory.resolve()
        if args.trajectory is not None
        else output / "task_program_trajectory.pt"
    )
    if args.trajectory is None and args.phase == "all":
        bootstrap = "import json,runpy,sys; sys.path[:]=json.loads(sys.argv.pop(1)); runpy.run_module('embodichain.lab.scripts.run_fluid_task_program',run_name='__main__')"
        subprocess.run(
            [
                sys.executable,
                "-c",
                bootstrap,
                json.dumps(sys.path),
                "--phase",
                "generate",
                "--source-task",
                str(source),
                "--output",
                str(output),
            ],
            check=True,
        )
    _replay(args, trajectory, output)


if __name__ == "__main__":
    main()
