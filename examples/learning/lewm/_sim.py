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

"""Single-environment push-cube collection and visual goal preparation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

CONFIG = Path(__file__).parent / "configs/push_cube_visual.json"
SPLIT_SEEDS = {"train": 100_000, "validation": 200_000, "test": 300_000}


def make_env(device: str, render_mode: str = "default") -> tuple[Any, dict[str, Any]]:
    import gymnasium as gym
    from embodichain.lab.gym.utils.gym_utils import config_to_cfg
    from embodichain.lab.gym.utils.registration import discover_task_packages

    config = json.loads(CONFIG.read_text())
    if render_mode not in {"default", "legacy", "stable"}:
        raise ValueError("render_mode must be default, legacy, or stable")
    if render_mode == "stable":
        config["render_cfg"] = {"denoising": {"window": "off", "offscreen": "off"}}
    config.update(
        device=device, gpu_id=int(device.split(":")[-1]) if ":" in device else 0
    )
    discover_task_packages()
    cfg = config_to_cfg(config)
    cfg.init_rollout_buffer = False
    cfg.filter_dataset_saving = True
    env = gym.make(config["id"], cfg=cfg, max_episode_steps=config["max_episode_steps"])
    env.unwrapped._replay_no_auto_reset = True
    return env, config


def render_contract(env: Any) -> dict[str, str]:
    """Record the simulator's resolved offscreen rendering settings."""
    render = env.unwrapped.sim.sim_config.render_cfg
    return {
        "renderer": render.renderer,
        "offscreen_denoising": render.denoising.offscreen,
    }


def environment_contract(env: Any, config: dict[str, Any]) -> dict[str, Any]:
    """Describe raw observations and controls independently of model blocking."""
    raw = env.unwrapped
    return {
        "action_dim": raw.single_action_space.shape[0],
        "action_semantics": "joint_position.relative@1",
        "step_dt": raw.step_dt,
        "action_scale": 0.1,
        "action_clip": 1.0,
        "camera": config["sensor"][0],
        "environment": "PushCubeRL",
        "physics": config["physics"],
        "render": render_contract(env),
    }


def close_env(env: Any) -> None:
    from embodichain.lab.sim import SimulationManager

    env.unwrapped.sim.destroy(exit_process=False)
    SimulationManager.flush_cleanup_queue()


def frame(obs: Any) -> np.ndarray:
    return obs["sensor"]["main_cam"]["color"][0, ..., :3].detach().cpu().numpy().copy()


def reset(env: Any, seed: int) -> tuple[Any, np.random.Generator]:
    obs, _ = env.reset(seed=seed, options={"save_data": False})
    raw = env.unwrapped
    rng = np.random.default_rng(seed)
    cube = raw.sim.get_rigid_object("cube").get_local_pose()
    angle, distance = rng.uniform(-np.pi, np.pi), rng.uniform(0.12, 0.18)
    offset = cube.new_tensor([np.cos(angle) * distance, np.sin(angle) * distance])
    raw.goal_pose[:, :2, 3] = cube[:, :2] + offset
    raw.goal_pose[:, 2, 3] = cube[:, 2]
    return obs, rng


def goal_image(env: Any, render_frames: int = 32) -> np.ndarray:
    """Render a settled visual goal and wash its history out after restoration.

    Rendering advances the denoiser history without stepping physics. The caller
    renders its next observation after this function, completing the same number
    of restored-scene frames. ``render_frames=1`` retains the original protocol.
    """
    from embodichain.lab.sim.scene import SceneView, capture_state, restore_state

    if type(render_frames) is not int or render_frames < 1:
        raise ValueError("render_frames must be a positive integer")
    raw = env.unwrapped
    view = SceneView(raw.sim)
    before = capture_state(view)
    cube = raw.sim.get_rigid_object("cube")
    pose = cube.get_local_pose().clone()
    pose[:, :2] = raw.goal_pose[:, :2, 3]
    try:
        cube.set_local_pose(pose)
        view.sync_render_state()
        for _ in range(render_frames):
            with raw.sim.render_frame():
                target = frame(raw.get_obs())
    finally:
        restore_state(view, before, qpos_tolerance=1e-4)
    for _ in range(render_frames - 1):
        with raw.sim.render_frame():
            raw.get_obs()
    after = capture_state(view)
    for initial, restored in zip(before.batches[0], after.batches[0]):
        for name in (
            "pose",
            "velocity",
            "qpos",
            "qvel",
            "target_qpos",
            "target_qvel",
            "qf",
        ):
            if not np.allclose(
                getattr(initial, name), getattr(restored, name), atol=1e-4, rtol=0
            ):
                raise RuntimeError(f"goal rendering changed {initial.uid}.{name}")
    return target


def guided_action(
    env: Any, phase: int, rng: np.random.Generator
) -> tuple[torch.Tensor, int, bool]:
    """Use privileged geometry only for collection of raw policy actions."""
    raw = env.unwrapped
    robot = raw.robot
    joints = robot.get_joint_ids("arm")
    qpos = robot.get_qpos()[:, joints]
    tcp = robot.compute_fk(qpos, name="arm", to_matrix=True)
    cube = raw.sim.get_rigid_object("cube").get_local_pose()
    direction = raw.goal_pose[:, :2, 3] - cube[:, :2]
    direction = direction / torch.linalg.vector_norm(
        direction, dim=-1, keepdim=True
    ).clamp_min(1e-6)
    target = tcp.clone()
    target[:, :2, 3] = cube[:, :2] - direction * 0.14
    target[:, 2, 3] = 0.25 if phase == 0 else 0.10
    if phase == 2:
        target[:, :2, 3] = cube[:, :2] + direction * 0.10
    error = torch.linalg.vector_norm(target[:, :3, 3] - tcp[:, :3, 3], dim=-1)
    if phase < 2 and bool(error[0] < 0.025):
        phase += 1
    solution = robot.compute_ik(target, joint_seed=qpos, name="arm")
    if solution is None or not bool(torch.as_tensor(solution[0]).all()):
        action = torch.zeros_like(qpos)
        ok = False
    else:
        action = ((solution[1].reshape_as(qpos) - qpos) / 0.1).clamp(-0.25, 0.25)
        ok = True
    noise = torch.as_tensor(
        rng.normal(0, 0.02, size=action.shape), device=action.device, dtype=action.dtype
    )
    return (action + noise).clamp(-1, 1), phase, ok


def collect(
    path: str | Path,
    split: str,
    episodes: int,
    device: str,
    render_mode: str = "default",
    start_index: int = 0,
) -> dict[str, Any]:
    """Collect complete episodes to a split-owned, chunked HDF5 file."""
    if split not in SPLIT_SEEDS or episodes < 1 or start_index < 0:
        raise ValueError("invalid split or episode count")
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    env, config = make_env(device, render_mode)
    raw = env.unwrapped
    metadata = {
        "action_dim": 6,
        "action_semantics": "joint_position.relative@1",
        "step_dt": raw.step_dt,
        "action_scale": 0.1,
        "action_clip": 1.0,
        "camera": config["sensor"][0],
        "environment": "PushCubeRL",
        "physics": "default",
        "render": render_contract(env),
    }
    successes = 0
    ik_failures = 0
    try:
        with h5py.File(temporary, "w") as output:
            output.attrs["format"] = "embodichain.lewm.trajectory.v1"
            output.attrs["split"] = split
            output.attrs["metadata"] = json.dumps(metadata, sort_keys=True)
            group = output.create_group("episodes")
            for index in range(episodes):
                seed = SPLIT_SEEDS[split] + start_index + index
                obs, rng = reset(env, seed)
                frames, actions = [frame(obs)], []
                phase, success = 0, False
                goal = raw.goal_pose[0, :3, 3].detach().cpu().numpy().copy()
                for _ in range(raw.max_episode_steps):
                    if (start_index + index) % 5 == 4:
                        action = torch.as_tensor(
                            rng.uniform(-1, 1, size=(1, 6)),
                            dtype=torch.float32,
                            device=raw.device,
                        )
                    else:
                        action, phase, valid = guided_action(env, phase, rng)
                        ik_failures += int(not valid)
                    obs, _, terminated, truncated, info = env.step(action)
                    actions.append(action[0].detach().cpu().numpy().copy())
                    frames.append(frame(obs))
                    if bool(
                        torch.as_tensor(terminated).any()
                        or torch.as_tensor(truncated).any()
                    ):
                        success = bool(info["success"][0])
                        break
                success = bool(info["success"][0])
                successes += int(success)
                episode = group.create_group(f"{index:06d}")
                episode.attrs.update(
                    seed=seed,
                    success=success,
                    guided=(start_index + index) % 5 != 4,
                    goal=goal,
                    final_distance=float(info["metrics"]["distance_to_goal"][0]),
                )
                episode.create_dataset(
                    "pixels",
                    data=np.stack(frames),
                    chunks=(1, *frames[0].shape),
                    compression="lzf",
                )
                episode.create_dataset(
                    "actions", data=np.stack(actions).astype(np.float32)
                )
                episode.create_dataset(
                    "timestamp",
                    data=np.arange(len(frames), dtype=np.float64) * raw.step_dt,
                )
                output.flush()
                print(
                    json.dumps(
                        {
                            "episode": index + 1,
                            "total": episodes,
                            "success": success,
                            "steps": len(actions),
                        }
                    ),
                    flush=True,
                )
        temporary.replace(path)
        return {
            "path": str(path.resolve()),
            "split": split,
            "episodes": episodes,
            "successes": successes,
            "ik_failures": ik_failures,
            "metadata": metadata,
        }
    finally:
        close_env(env)
