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

"""Run a repository Task Program on matched physics profiles, without sensors.

Run: python -m scripts.benchmark.robotics.dexuni.tasks pick auto 0 1 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from tensordict import TensorDict

import gymnasium
import warp as wp

from embodichain.lab.gym.envs.demo import execute_demo_episode
from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv
from embodichain.lab.sim.cfg import NewtonPhysicsCfg
from embodichain.lab.gym.utils.gym_utils import (
    add_env_launcher_args_to_parser,
    build_env_cfg_from_args,
)
from embodichain.lab.gym.utils.registration import (
    discover_task_packages,
    execute_init_hooks,
)
from embodichain.lab.sim.sim_manager import SimulationManager

ROOT = Path("embodichain/extensions/tasks/configs/tasks/manipulation")
OUT = Path("outputs/dexuni")
TASKS = {
    "pick": ("workspace_pose_pick_place", ["cube", "goal"]),
    "stack": ("tableware/stack_blocks_two", ["block_1", "block_2"]),
    "drawer": ("drawer_storage", ["cube"]),
}


def make_env(
    task: str, mode: str, seed: int, batch: int = 1
) -> tuple[EmbodiedEnv, NewtonPhysicsCfg, dict[str, Any]]:
    """Construct the authored deployment with paired randomization and solver ownership."""
    path, _ = TASKS[task]
    parser = argparse.ArgumentParser()
    add_env_launcher_args_to_parser(parser, require_gym_config=True)
    args = parser.parse_args(
        [
            "--gym_config",
            str(ROOT / path / "task.ur5.dexuni.yaml"),
            "--seed",
            str(seed),
            "--num_envs",
            str(batch),
            "--headless",
            "--disable_sensor",
            "--filter_dataset_saving",
            "--renderer",
            "no-render",
        ]
    )
    discover_task_packages()
    execute_init_hooks()

    # Resolve the deployed robot and task using the authored profile first.
    def variation(config):
        if task != "stack":
            config["env"].setdefault("events", {})["paired_cube_pose"] = {
                "func": "randomize_rigid_object_pose",
                "mode": "reset",
                "params": {
                    "entity_cfg": {"uid": "cube"},
                    "position_range": [[-0.006, -0.006, 0], [0.006, 0.006, 0]],
                    "rotation_range": [[0, 0, -3], [0, 0, 3]],
                },
            }

    cfg, merged, _ = build_env_cfg_from_args(args, gym_config_modifier=variation)
    physics = cfg.sim_cfg.physics_cfg
    if mode != "all_mj":
        physics.dexuni_mujoco_joints_per_arena = None
    if mode in ("one_way", "two_way"):
        physics.solver_cfg["coupling"] = mode
    if mode == "mj":
        physics.solver_cfg = {
            "solver_type": "mujoco_warp",
            **physics.solver_cfg["mujoco_options"],
        }
    return gymnasium.make(id=merged["id"], cfg=cfg).unwrapped, physics, merged


def execute(task: str, mode: str, seeds: list[int]) -> None:
    """Run the configured expert episode and retain its measured segment outcomes."""
    _, objects = TASKS[task]
    env, physics, merged = make_env(task, mode, seeds[0])
    result = {
        "task": merged["id"],
        "mode": mode,
        "solver": physics.solver_cfg,
        "dt": physics.physics_dt,
        "substeps": physics.num_substeps,
        "episodes": [],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    diagnostics = env.sim._spawn_scene.builder.result.get_solver_diagnostics()
    backend = diagnostics.get("features", {}).get("backend", diagnostics["solver_type"])
    expected = {
        "auto": "pure_mujoco",
        "all_mj": "pure_mujoco",
        "mj": "mujoco_warp",
        "one_way": "coupled",
        "two_way": "coupled",
    }[mode]
    if backend != expected:
        raise RuntimeError(
            f"{task}/{mode} reached {backend}, expected {expected}: {diagnostics}"
        )
    payload = env.sim.get_rigid_object("block_2" if task == "stack" else "cube")
    nominal = payload.get_mass().clone()
    result["initial_diagnostics"] = diagnostics
    try:
        for seed in seeds:
            env.reset(seed=seed, options={"save_data": False})

            payload.set_mass(nominal * (0.75, 1.0, 1.25)[seed % 3])

            def poses():
                return {
                    uid: env.sim.get_rigid_object(uid)
                    .get_local_pose()[0]
                    .cpu()
                    .tolist()
                    for uid in objects
                }

            initial = poses()
            step0 = env.sim.physics_step_index
            wp.synchronize_device("cuda:0")
            start = time.perf_counter()
            boundaries = []

            def record(segment):
                boundaries.append({"name": segment.name, "poses": poses()})

            tape = []
            save_tape = mode == "all_mj" and not (OUT / f"control_{task}.npz").exists()

            def capture(active, action, observation):
                value = getattr(action, "value", action)
                fields = (
                    value.items()
                    if isinstance(value, TensorDict)
                    else [("qpos", value)]
                )
                tape.append({k: v[0].detach().cpu().numpy().copy() for k, v in fields})

            episode = execute_demo_episode(
                env,
                on_segment_completed=record,
                step_observer=capture if save_tape else None,
            )
            if save_tape and episode.completed and all(episode.success):
                if any(frame.keys() != tape[0].keys() for frame in tape):
                    raise ValueError("Control schema changed within a trajectory")
                np.savez(
                    OUT / f"control_{task}.npz",
                    **{key: np.stack([v[key] for v in tape]) for key in tape[0]},
                    seed=seed,
                    initial=json.dumps(initial),
                    stages=json.dumps([(s.name, s.end_step) for s in episode.segments]),
                )
            wp.synchronize_device("cuda:0")
            elapsed = time.perf_counter() - start
            summary = episode.to_metadata()
            result["diagnostics"] = (
                env.sim._spawn_scene.builder.result.get_solver_diagnostics()
            )
            result["episodes"].append(
                {
                    "seed": seed,
                    "mass": payload.get_mass().item(),
                    "initial": initial,
                    "final": poses(),
                    "seconds": elapsed,
                    "world_steps": env.sim.physics_step_index - step0,
                    "boundaries": boundaries,
                    "episode": summary,
                }
            )
            print(
                seed,
                summary["completed"],
                summary["success"],
                summary["terminal_reason"],
                elapsed,
                flush=True,
            )
            (OUT / f"deploy_{task}_{mode}_{seeds[0]}.json").write_text(
                json.dumps(result, default=str)
            )
    finally:
        env.close(exit_process=False)
        SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=TASKS)
    parser.add_argument("mode", choices=("auto", "one_way", "two_way", "all_mj", "mj"))
    parser.add_argument("seeds", nargs="+", type=int)
    options = parser.parse_args()
    execute(options.task, options.mode, options.seeds)
