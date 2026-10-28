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

"""Run a Task Program with live robot/cup contact and one-way SPH fluids."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Iterator
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from embodichain.lab.gym.envs.demo import execute_demo_episode
from embodichain.lab.gym.envs.embodied_env import EmbodiedEnvCfg
from embodichain.lab.gym.envs.managers.cfg import SceneEntityCfg
from embodichain.lab.gym.envs.managers.observations import (
    get_fluid_summary,
    get_fluid_volume_in_region,
)
from embodichain.lab.gym.utils.gym_utils import config_to_cfg
from embodichain.lab.gym.utils.registration import discover_task_packages, make
from embodichain.lab.sim import SimulationManager
from embodichain.utils.config_paths import resolve_config_path
from embodichain.utils.utility import load_config

__all__ = ["main"]

_TASK = (
    "extensions/embodichain_tasks/configs/tasks/manipulation/tableware/pour_cup/"
    "task.franka.fluid.yaml"
)


def _configure(
    task: Path,
    *,
    num_envs: int,
    renderer: str,
    output: Path,
    particle_spacing: float | None = None,
    record_trajectory: bool = False,
) -> tuple[str, EmbodiedEnvCfg]:
    declaration = load_config(str(task))
    cfg = config_to_cfg(declaration, source_path=task)
    if cfg.task_program is None:
        raise ValueError("The online fluid deployment must select a Task Program.")
    solver = cfg.sim_cfg.physics_cfg.solver_cfg
    if solver.get("solver_type") != "dexuni" or not isinstance(
        solver.get("sph_cfg"), dict
    ):
        raise ValueError("Online fluid execution requires DexUni with sph_cfg.")
    if num_envs < 1:
        raise ValueError("num_envs must be positive.")
    cfg.num_envs = num_envs
    cfg.sim_cfg.headless = True
    cfg.sim_cfg.render_cfg.renderer = renderer
    cfg.enable_sensor = False
    cfg.record_trajectory = record_trajectory
    if renderer == "no-render":
        cfg.events.record_camera = None
        for fluid in cfg.fluid_object:
            fluid.render.mode = "none"
    else:
        cfg.events.record_camera.params["save_path"] = str(output / "videos")
        cfg.events.record_camera.params["max_env_num"] = num_envs
    if particle_spacing is not None:
        if not np.isfinite(particle_spacing) or particle_spacing <= 0:
            raise ValueError("particle_spacing must be finite and positive.")
        fluid = cfg.fluid_object[0]
        if fluid.spawn.kind == "points":
            raise ValueError("particle-spacing requires a procedural volume source.")
        scale = particle_spacing / fluid.particle_spacing
        if fluid.render.render_radius is not None:
            fluid.render.render_radius *= scale
        for key in ("cell_size", "kernel_radius"):
            if key in fluid.render.mesh_surface:
                fluid.render.mesh_surface[key] *= scale
        fluid.particle_spacing = particle_spacing
        fluid.particle_radius = 0.3 * particle_spacing
        solver["sph_cfg"]["particle_support_radius"] = 2.25 * particle_spacing
    return declaration["id"], cfg


def _sample(env: Any) -> dict[str, Any]:
    summary = get_fluid_summary(env, {}, SceneEntityCfg(uid="water"))
    if not torch.isfinite(summary).all() or not torch.all(summary[:, -1] == 1):
        raise RuntimeError("Fluid state became non-finite.")
    received = get_fluid_volume_in_region(
        env,
        {},
        [SceneEntityCfg(uid="water"), SceneEntityCfg(uid="receiver")],
        lower=[-0.1, -0.1, 0.0045],
        upper=[0.1, 0.1, 0.0675],
        sphere_center=[0, 0, 0.088],
        sphere_radius=0.093,
    )
    return {
        "time_s": (env.elapsed_steps * env.step_dt).cpu().tolist(),
        "received_ml": (received[:, 0] * 1e6).cpu().tolist(),
        "summary": summary.cpu().tolist(),
        "cup_pose": env.sim.get_rigid_object("cup").get_local_pose().cpu().tolist(),
    }


def _cup_qualification(poses: list[np.ndarray]) -> dict[str, Any]:
    measured = np.stack(poses, axis=1)
    lift = measured[..., 2].max(axis=1) - measured[:, 0, 2]
    vertical = 1 - 2 * (measured[..., 3] ** 2 + measured[..., 4] ** 2)
    tilt = np.rad2deg(np.arccos(np.clip(vertical, -1, 1))).max(axis=1)
    return_error = np.linalg.norm(measured[:, -1, :3] - measured[:, 0, :3], axis=1)
    return {
        "lift_m": lift.tolist(),
        "max_tilt_deg": tilt.tolist(),
        "return_error_m": return_error.tolist(),
        "accepted": ((lift >= 0.05) & (tilt >= 45) & (return_error <= 0.03)).tolist(),
    }


def _run(args: argparse.Namespace, output: Path) -> dict[str, Any]:
    path = resolve_config_path(args.task).resolve()
    task_id, cfg = _configure(
        path,
        num_envs=args.num_envs,
        renderer=args.renderer,
        output=output,
        particle_spacing=args.particle_spacing,
        record_trajectory=args.record_trajectory,
    )
    discover_task_packages()
    env = make(task_id, cfg=cfg)
    try:
        env.reset(seed=0)
        cup = env.sim.get_rigid_object("cup")
        poses = [cup.get_local_pose().cpu().numpy().copy()]
        samples = [_sample(env)]
        started = time.perf_counter()
        sample_stride = max(1, round(1 / env.step_dt))

        def observe(actions: Iterable[Any], description: str) -> Iterator[Any]:
            print(description, flush=True)
            for action in actions:
                yield action
                poses.append(cup.get_local_pose().cpu().numpy().copy())
                if int(env.elapsed_steps.max()) % sample_stride == 0:
                    sample = _sample(env)
                    samples.append(sample)
                    print(json.dumps(sample), flush=True)

        def stop() -> bool:
            return (
                args.seconds is not None
                and float(env.elapsed_steps.max()) * env.step_dt >= args.seconds
            )

        result = execute_demo_episode(env, progress=observe, should_stop=stop)
        if result.completed and args.settle_seconds:
            target = env.robot.get_qpos()[:, env.active_joint_ids].clone()
            for _ in range(round(args.settle_seconds / env.step_dt)):
                env.step(target)
                poses.append(cup.get_local_pose().cpu().numpy().copy())
        samples.append(_sample(env))
        qualification = _cup_qualification(poses)
        accepted = bool(
            result.completed
            and result.all_success
            and all(qualification["accepted"])
            and all(v >= 50 for v in samples[-1]["received_ml"])
        )
        videos = []
        if args.renderer != "no-render":
            recorder = env.event_manager.get_functor("record_camera")
            purpose = "recording" if accepted else "diagnostic"
            name = cfg.events.record_camera.params["name"]
            video_root = output / "videos"
            if purpose == "diagnostic":
                video_root /= "diagnostics"
            existing = {
                path: (path.stat().st_ino, path.stat().st_mtime_ns)
                for path in video_root.glob(f"env_*/episode_*_{name}.mp4")
            }
            recorder.save_and_clear(purpose=purpose)
            for i in range(env.num_envs):
                matches = []
                for path in (video_root / f"env_{i:03d}").glob(f"episode_*_{name}.mp4"):
                    stat = path.stat()
                    if existing.get(path) != (stat.st_ino, stat.st_mtime_ns):
                        matches.append(path)
                if matches:
                    videos.append(str(sorted(matches)[-1].relative_to(output)))
        if args.record_trajectory:
            env.save_trajectory(str(output / "online_trajectory.pt"))
        report = {
            "task": str(path),
            "coupling": "online one-way rigid-to-fluid",
            "num_envs": env.num_envs,
            "renderer": args.renderer,
            "videos": videos,
            "video_fps": (
                cfg.events.record_camera.params.get("fps") or env.control_frequency
                if videos
                else None
            ),
            "particle_count_per_env": env.sim.get_fluid_object("water").num_particles,
            "fluid_source": cfg.fluid_object[0].spawn.to_dict(),
            "particle_spacing_m": cfg.fluid_object[0].particle_spacing,
            "physics_dt": cfg.sim_cfg.physics_dt,
            "control_dt": env.step_dt,
            "task_program": result.to_metadata(),
            "cup": qualification,
            "wall_seconds": time.perf_counter() - started,
            "samples": samples,
        }
        report["accepted"] = accepted
        (output / "fluid_gym_result.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(
            json.dumps(
                {
                    key: report[key]
                    for key in ("accepted", "coupling", "cup", "wall_seconds")
                }
            ),
            flush=True,
        )
        if args.seconds is None and not report["accepted"]:
            raise RuntimeError(
                "Online pouring failed physical qualification; inspect fluid_gym_result.json."
            )
        return report
    finally:
        env.close(exit_process=False)
        del env
        SimulationManager.flush_cleanup_queue()


def main() -> None:
    """Execute the configured Task Program and fluid simulation in one World."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=_TASK)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/fluid-task-program-online")
    )
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument(
        "--renderer", choices=["fast-rt", "no-render"], default="fast-rt"
    )
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--particle-spacing", type=float)
    parser.add_argument("--settle-seconds", type=float, default=2.0)
    parser.add_argument(
        "--record-trajectory",
        action="store_true",
        help="Optionally save measured online states for inspection.",
    )
    args = parser.parse_args()
    if args.seconds is not None and (
        not np.isfinite(args.seconds) or args.seconds <= 0
    ):
        parser.error("seconds must be finite and positive")
    if not np.isfinite(args.settle_seconds) or args.settle_seconds < 0:
        parser.error("settle-seconds must be finite and non-negative")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _run(args, output)


if __name__ == "__main__":
    main()
