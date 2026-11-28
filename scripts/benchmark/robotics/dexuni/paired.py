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
"""Replay an accepted task trajectory on 30 paired physical initial states.

Run: python -m scripts.benchmark.robotics.dexuni.paired pick auto
Generate the reference with the tasks benchmark in all_mj mode first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time

import numpy as np
import torch
import warp as wp
from tensordict import TensorDict

from embodichain.lab.gym.envs.types import ControllerAction
from embodichain.lab.sim.sim_manager import SimulationManager
from embodichain.utils.math import quat_mul
from .tasks import OUT, TASKS, make_env


def run(task: str, mode: str, size: int, repeats: int) -> None:
    """Record stage geometry and time equal-length commands, including failures."""
    path = OUT / f"control_{task}.npz"
    with np.load(path, allow_pickle=False) as tape:
        initial = json.loads(str(tape["initial"]))
        stages = {n: int(i) for n, i in json.loads(str(tape["stages"]))}
        seed = int(tape["seed"])
        cpu = {key: tape[key] for key in ("qpos", "qvel", "qf") if key in tape}
    env, _, _ = make_env(task, mode, seed, size)
    try:
        device = env.device
        assert env.num_envs == size
        diag = env.sim._spawn_scene.builder.result.get_solver_diagnostics()
        branch = diag.get("features", {}).get("backend", diag["solver_type"])
        expected = (
            "coupled"
            if mode in ("one_way", "two_way")
            else "mujoco_warp" if mode == "mj" else "pure_mujoco"
        )
        if branch != expected:
            raise RuntimeError(diag)
        # A controller-ready tensor means qpos. Normalize recorded controls once.
        data = {k: torch.from_numpy(v).to(device) for k, v in cpu.items()}
        actions = [
            ControllerAction(
                TensorDict(
                    {k: v[i].expand(size, -1) for k, v in data.items()},
                    batch_size=[size],
                )
            )
            for i in range(len(next(iter(data.values()))))
        ]
        objects = {uid: env.sim.get_rigid_object(uid) for uid in TASKS[task][1]}
        payload = "block_2" if task == "stack" else "cube"
        nominal = objects[payload].get_mass().clone()
        output = dict(
            task=task,
            mode=mode,
            batch=size,
            steps=len(actions),
            stages=stages,
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            diagnostics=diag,
            trials=[],
        )
        for repeat in range(repeats):
            env.reset(seed=seed, options={"save_data": False})
            offsets = {}
            for i, (uid, obj) in enumerate(objects.items()):
                if uid == "goal":
                    continue
                # Center every mode on the actual reference poses. The first row
                # reproduces the reference, other rows perturb XY/yaw and load.
                rng = np.random.default_rng(20261128 + i)
                shift = rng.uniform(-1, 1, (size, 3))
                shift[0] = 0.0
                offsets[uid] = shift.tolist()
                pose = torch.tensor(initial[uid], device=device).repeat(size, 1)
                shift = torch.tensor(shift, device=device, dtype=pose.dtype)
                pose[:, :2] += 0.006 * shift[:, :2]
                angle = shift[:, 2] * np.deg2rad(3) / 2
                rot = torch.zeros_like(pose[:, 3:])
                rot[:, 2] = angle.sin()
                rot[:, 3] = angle.cos()
                pose[:, 3:] = quat_mul(rot, pose[:, 3:])
                obj.set_local_pose(pose)
                zeros = torch.zeros((size, 3), device=device)
                obj.set_velocity(zeros, zeros)
            factors = torch.tensor(
                [(0.75, 1.0, 1.25)[(i + seed) % 3] for i in range(size)], device=device
            )
            objects[payload].set_mass(nominal * factors)
            env._demo_no_auto_reset = True
            env._demo_active_mask.fill_(True)
            start_height = objects[payload].get_local_pose()[:, 2].clone()
            # Keep graph recapture and warmup outside the matched trajectory.
            env.sim.update(step=4)
            wp.synchronize_device("cuda:0")
            stages_met = {}
            observations = {}

            def measure(name):
                obj = objects[payload]
                pos = obj.get_local_pose()
                if name.startswith("pick"):
                    delta = pos[:, 2] - start_height
                    limit = {"pick": 0.04, "stack": 0.05, "drawer": 0.08}[task]
                    return delta > limit, delta
                if task == "drawer":
                    rail = env.sim.get_articulation("drawer").get_qpos()[:, 0]
                    if name == "open_drawer":
                        return (rail >= 0.25) & (rail <= 0.28), rail
                    if name == "store_cube":
                        return env._cube_stored(), rail
                    return env.is_task_success(), rail
                if task == "pick":
                    target = objects["goal"].get_local_pose()[:, :3].clone()
                    target[:, 2] += 0.12
                    err = torch.linalg.vector_norm(pos[:, :3] - target, dim=1)
                    up = obj.get_local_pose(to_matrix=True)[:, 2, 2]
                    return (err <= 0.02) & (up >= np.cos(np.deg2rad(15))), err
                lower = objects["block_1"]
                delta = pos[:, :3] - lower.get_local_pose()[:, :3]
                delta[:, 2] -= 0.05
                up = obj.get_local_pose(to_matrix=True)[:, 2, 2]
                up0 = lower.get_local_pose(to_matrix=True)[:, 2, 2]
                return (
                    (delta.abs() <= delta.new_tensor([0.025, 0.025, 0.012])).all(1)
                    & (up > 0.7071)
                    & (up0 > 0.7071)
                ), torch.linalg.vector_norm(delta, dim=1)

            elapsed = []
            end_to_name = {v: k for k, v in stages.items()}
            expected_intervals = len(actions) * env.cfg.sim_steps_per_control
            initial_step = env.sim.physics_step_index
            assert env.sim.physics.cuda_graph_status == "captured"
            started = time.perf_counter()
            first = 0
            for i, action in enumerate(actions, 1):
                env.step(action)
                if i in end_to_name or i % 32 == 0 or i == len(actions):
                    wp.synchronize_device("cuda:0")
                    elapsed.append([i - first, time.perf_counter() - started])
                    if i in end_to_name:
                        label = end_to_name[i]
                        passed, metric = measure(label)
                        stages_met[label] = passed.cpu().tolist()
                        observations[label] = metric.cpu().tolist()
                        print(
                            f"{task}/{mode} {label}: {sum(stages_met[label])}/{size}, "
                            f"frame {i}/{len(actions)}",
                            flush=True,
                        )
                    first = i
                    wp.synchronize_device("cuda:0")
                    started = time.perf_counter()
            elapsed_intervals = env.sim.physics_step_index - initial_step
            if (
                elapsed_intervals != expected_intervals
                or stages_met.keys() != stages.keys()
            ):
                raise RuntimeError(
                    f"Incomplete replay at {elapsed_intervals} World steps: {stages_met.keys()}"
                )
            output["trials"].append(
                dict(
                    world_steps=elapsed_intervals,
                    graph=env.sim.physics.cuda_graph_status,
                    offsets=offsets,
                    mass=objects[payload].get_mass().cpu().tolist(),
                    stages=stages_met,
                    observations=observations,
                    elapsed=elapsed,
                )
            )
            (OUT / f"paired_{task}_{mode}_{size}.json").write_text(
                json.dumps(output, default=str)
            )
    finally:
        env.close(exit_process=False)
        SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=TASKS)
    parser.add_argument("mode", choices=("auto", "all_mj", "mj", "one_way", "two_way"))
    parser.add_argument("--size", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    run(args.task, args.mode, args.size, args.repeats)
