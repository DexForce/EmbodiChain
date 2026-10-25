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

"""Evaluate a frozen SL visual BC checkpoint in independent PickPlace physics."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import torch

from embodichain_models.policy.bc.model import (
    ChunkBCModel,
    ChunkBCModelCfg,
    ChunkBCExecutor,
)
from ._pick_place import CAMERAS, TARGET_SEMANTICS, decode_request


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def evaluate_checkpoint(
    checkpoint: Path,
    *,
    scenes: int,
    seed: int,
    output: Path,
    sim_device: str = "cuda:0",
    policy_device: str = "cuda:0",
    action_horizon: int | None = None,
    diagnostic: bool = False,
    trace: bool = False,
) -> dict[str, Any]:
    """Measure frozen student behavior without constructing an expert engine."""
    if action_horizon is not None and not diagnostic:
        raise ValueError("Action-horizon overrides require an explicit diagnostic run")
    if trace and not diagnostic:
        raise ValueError("Step traces require an explicit diagnostic run")
    invocation_started = time.monotonic()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    signature = state["signature"]
    weights = state["learner"]["model"]
    model = ChunkBCModel(ChunkBCModelCfg(**weights["_extra_state"]["config"]))
    if (
        model.cfg.cameras != CAMERAS
        or model.cfg.target_semantics != TARGET_SEMANTICS
        or signature.get("camera_profile_version") != "pick-place-v2"
    ):
        raise ValueError(
            "Checkpoint input or target contract differs from the qualified task"
        )
    model.load_state_dict(weights)
    model = model.to(policy_device).eval()
    if torch.device(policy_device).type == "cuda":
        torch.cuda.synchronize(policy_device)
    policy_load_seconds = time.monotonic() - invocation_started
    executor = ChunkBCExecutor(
        model,
        action_horizon=(
            signature["action_horizon"] if action_horizon is None else action_horizon
        ),
    )
    policy_calls = 0

    def count_forward(_module: Any, _inputs: Any, _output: Any) -> None:
        nonlocal policy_calls
        policy_calls += 1

    forward_hook = model.register_forward_hook(count_forward)
    config = deepcopy(signature["environment"])
    config.update(
        device=sim_device,
        gpu_id=int(sim_device.split(":")[-1]),
        num_envs=1,
        seed=seed,
        headless=True,
    )
    config["max_episode_steps"] = 600
    os.environ["EMBODICHAIN_SIM_EXIT_PROCESS"] = "0"
    from embodichain.lab.sim import SimulationManager
    from embodichain.lab.gym.utils.gym_utils import config_to_cfg, get_manager_modules
    from embodichain.lab.gym.utils.registration import discover_task_packages
    from examples.ods.simple_bc_demo.observations import policy_observations
    import gymnasium as gym

    discover_task_packages()
    cfg = config_to_cfg(config, manager_modules=get_manager_modules())
    cfg.init_rollout_buffer = False
    cfg.filter_dataset_saving = True
    cfg.max_episode_steps = 600
    generator = torch.Generator(device="cpu").manual_seed(seed)
    positions = (
        torch.tensor([-0.42, -0.08])
        + (torch.rand(scenes, 2, generator=generator) - 0.5) * 0.08
    )
    panel_hash = hashlib.sha256(positions.numpy().tobytes()).hexdigest()
    records: list[dict[str, Any]] = []
    env = None
    started = time.monotonic()
    reset_seconds, step_seconds, cleanup_seconds = 0.0, 0.0, 0.0
    simulation_setup_seconds = 0.0
    try:
        env = gym.make(config["id"], cfg=cfg)
        simulation_setup_seconds = time.monotonic() - started
        raw = env.unwrapped
        raw._replay_no_auto_reset = True
        for ordinal, xy in enumerate(positions):
            reset_started = time.monotonic()
            observation, _ = env.reset(
                seed=seed + ordinal,
                options={"save_data": False, "cube_xy": [xy.tolist()]},
            )
            reset_seconds += time.monotonic() - reset_started
            executor.reset()
            scene_calls_started = policy_calls
            trace_rows = []
            terminal, truncated = False, False
            steps_started = time.monotonic()
            with torch.no_grad():
                for length in range(1, 601):
                    geometry = policy_observations(observation).to(policy_device)
                    images = {}
                    for camera in CAMERAS:
                        color = observation["sensor"][camera]["color"]
                        if color.dtype != torch.uint8 or color.shape not in (
                            (1, 256, 256, 3),
                            (1, 256, 256, 4),
                        ):
                            raise ValueError(
                                "Student evaluation requires measured uint8 RGB(A) camera frames"
                            )
                        # Match offline CPU uint8->FP32 scaling exactly before
                        # transferring a measured history to the policy device.
                        images[camera] = (
                            color[..., :3]
                            .movedim(-1, -3)
                            .to(device="cpu", dtype=torch.float32)
                            .div_(255)
                            .to(policy_device)
                        )
                    calls_before = policy_calls
                    target = executor.action(geometry[:, :27], images)
                    if not torch.isfinite(target).all():
                        raise FloatingPointError("Student produced non-finite targets")
                    action = decode_request(target, geometry)
                    observation, _, terminated, timed_out, _ = env.step(
                        action.to(raw.device)
                    )
                    terminal, truncated = bool(torch.as_tensor(terminated).any()), bool(
                        torch.as_tensor(timed_out).any()
                    )
                    if trace:
                        success, failed, metrics = raw.compute_task_state()
                        cube, tcp = geometry[0, 27:30], geometry[0, 18:21]
                        distance = float(torch.linalg.vector_norm(cube - tcp))
                        goal_error = float(metrics["goal_error"][0])
                        lifted = bool(metrics["lifted"][0])
                        # Diagnostic labels use measured geometry, not expert state.
                        phase = (
                            "near_goal"
                            if goal_error <= 0.03
                            else (
                                "transport"
                                if lifted
                                else "approach" if distance > 0.08 else "near_cube"
                            )
                        )
                        trace_rows.append(
                            {
                                "step": length,
                                "geometric_phase": phase,
                                "cube_xyz_before": cube.tolist(),
                                "tcp_xyz_before": tcp.tolist(),
                                "gripper_qpos_before": geometry[0, 7:9].tolist(),
                                "tcp_cube_distance_before": distance,
                                "target_tcp_error_before": float(
                                    torch.linalg.vector_norm(target[0, :3] - tcp)
                                ),
                                "goal_error_after": goal_error,
                                "lifted_after": lifted,
                                "success_after": bool(success[0]),
                                "target": target[0].tolist(),
                                "request": action[0].tolist(),
                                "policy_forward_calls": policy_calls - calls_before,
                                "scene_policy_forward_calls": policy_calls
                                - scene_calls_started,
                            }
                        )
                    if terminal or truncated:
                        break
            step_seconds += time.monotonic() - steps_started
            success, failed, metrics = raw.compute_task_state()
            record = {
                "scene": ordinal,
                "seed": seed + ordinal,
                "cube_xy": xy.tolist(),
                "success": bool(success[0]),
                "failure": bool(failed[0]),
                "lifted": bool(metrics["lifted"][0]),
                "goal_error": float(metrics["goal_error"][0]),
                "length": length,
                "terminated": terminal,
                "truncated": truncated,
                "policy_forward_calls": policy_calls - scene_calls_started,
            }
            if trace:
                record["trace"] = trace_rows
            records.append(record)
            _write(
                output,
                {
                    "status": "in_progress",
                    "policy_step": state["learner"]["num_updates"],
                    "scene_digest": panel_hash,
                    "scenes": scenes,
                    "records": records,
                },
            )
            print(json.dumps(record, allow_nan=False), flush=True)
        if getattr(raw, "_expert_engine", None) is not None:
            raise RuntimeError(
                "Student evaluation unexpectedly initialized an expert engine"
            )
    finally:
        cleanup_started = time.monotonic()
        forward_hook.remove()
        try:
            if env is not None:
                env.unwrapped.close(exit_process=False)
        finally:
            SimulationManager.flush_cleanup_queue()
            cleanup_seconds = time.monotonic() - cleanup_started
    report = {
        "status": "complete",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "policy_step": state["learner"]["num_updates"],
        "scene_digest": panel_hash,
        "scenes": scenes,
        "seed": seed,
        "records": records,
        "success_rate": sum(record["success"] for record in records) / scenes,
        "goal_error": sum(record["goal_error"] for record in records) / scenes,
        "lift_rate": sum(record["lifted"] for record in records) / scenes,
        "wall_seconds": time.monotonic() - invocation_started,
        "timing": {
            "policy_load_seconds": policy_load_seconds,
            "simulation_setup_seconds": simulation_setup_seconds,
            "reset_seconds": reset_seconds,
            "step_seconds": step_seconds,
            "cleanup_seconds": cleanup_seconds,
        },
        "expert_initialized": False,
        "closed": True,
        "diagnostic": diagnostic,
        "action_horizon": executor.action_horizon,
        "policy_forward_calls": policy_calls,
        "trace_enabled": trace,
    }
    _write(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--scenes", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--policy-device", default="cuda:0")
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--action-horizon", type=int)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    if args.scenes < 1 or args.seed < 0:
        parser.error("scenes must be positive and seed nonnegative")
    torch.set_num_threads(1)
    evaluate_checkpoint(
        args.checkpoint,
        scenes=args.scenes,
        seed=args.seed,
        output=args.output,
        sim_device=args.sim_device,
        policy_device=args.policy_device,
        action_horizon=args.action_horizon,
        diagnostic=args.diagnostic,
        trace=args.trace,
    )


if __name__ == "__main__":
    main()
