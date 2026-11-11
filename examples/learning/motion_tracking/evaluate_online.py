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

"""Evaluate an online-training checkpoint from every clip's first frame."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
import numpy as np
import torch
import warp as wp
from datetime import datetime, timezone
from embodichain.lab.gym.utils.gym_utils import config_to_cfg
from embodichain_models.policies.motion_tracking import (
    MotionTrackingPolicy,
    MotionTrackingPolicyCfg,
)
from embodichain_models.dynamics.rwm.robot_state import pack_robot_state
from embodichain.lab.gym.utils.registration import build_env
from embodichain_tasks.locomotion.tracking import g1_motion  # Register physical task.
from run_metadata import write_run_metadata
from _configuration import normalize_recipe


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path)
    parser.add_argument("--motion-path", type=Path)
    parser.add_argument(
        "--clips",
        help="Comma-separated evaluation clips; defaults to the checkpoint selection.",
    )
    parser.add_argument("--quaternion-order", choices=("xyzw", "wxyz"))
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument(
        "--mode", choices=("quality", "physics-regression"), default="quality"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Bound a functional evaluation; results record the shortened horizon.",
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    wp.config.deterministic = (
        wp.DeterministicMode.RUN_TO_RUN
        if args.mode == "physics-regression"
        else wp.DeterministicMode.NOT_GUARANTEED
    )
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    training_config = normalize_recipe(state["config"])["workflow"]
    gym = deepcopy(state.get("evaluation_gym_config", state["gym_config"]))
    gym["num_envs"] = 1
    options = gym["env"]["extensions"]["motion_tracking"]
    if args.motion_path is not None:
        options["motion_path"] = str(args.motion_path)
    if args.clips is not None:
        options["clips"] = args.clips.split(",")
    if args.quaternion_order is not None:
        options["quaternion_order"] = args.quaternion_order
    options["observation_noise"] = False
    options["random_start"] = False
    options["adaptive_sampling"] = False
    options["fixed_clip_assignment"] = True
    options["reset_noise"] = False
    options["domain_randomization"] = False
    # Read clip count before constructing the single physical scene.
    from embodichain.data import get_data_path
    from embodichain.data_pipeline.datasets.motion import MotionLibrary

    joints = state["identity"]["wm"]["joint_names"]
    motion = MotionLibrary.from_npz(
        get_data_path(options["motion_path"]),
        joints,
        quaternion_order=options["quaternion_order"],
        clips=options.get("clips"),
    )
    gym["num_envs"] = motion.num_clips
    gym["max_episode_steps"] = 2**31 - 1
    cfg = config_to_cfg(gym)
    cfg.init_rollout_buffer = False
    env = build_env(gym["id"], base_env_cfg=cfg)
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        metadata = write_run_metadata(
            args.output.parent,
            invocation_started_at=datetime.now(timezone.utc).isoformat(),
            config={
                "kind": "evaluation",
                "mode": args.mode,
                "seed": args.seed,
                "amp": False,
            },
            gym_config=gym,
            motion={
                "names": env.motion.names,
                "lengths": env.motion.lengths.cpu().tolist(),
                "fps": env.motion.fps,
            },
            motion_path=Path(get_data_path(options["motion_path"])),
            device=env.device,
            resume_checkpoint=args.checkpoint.resolve(),
        )
        pcfg = MotionTrackingPolicyCfg(**state["identity"]["policy"])
        pcfg.amp_bfloat16 = False
        policy = MotionTrackingPolicy(pcfg, env.device)
        policy.load_state_dict(state["policy"])
        policy.eval()
        tracking_errors = env.task.tracking_errors
        if env.motion_cfg.compile_mode is not None:
            # Fuse metric FK without adding a CUDA graph output lifetime to evaluation.
            tracking_errors = torch.compile(
                tracking_errors,
                fullgraph=True,
                options={"triton.cudagraphs": False},
            )
        obs, _ = env.reset(seed=args.seed, options={"motion_frames": env.motion.starts})
        active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        total = torch.zeros(env.num_envs, device=env.device)
        steps = torch.zeros_like(total, dtype=torch.int64)
        failed = torch.zeros_like(active)
        measured_steps = torch.zeros_like(steps)
        error_sums = {
            name: torch.zeros_like(total) for name in env.task.reward_names[:6]
        }
        failure_reasons = {
            name: torch.zeros_like(active) for name in ("height", "tilt", "non_finite")
        }
        trajectories = []
        references = []
        alive = []
        start = time.perf_counter()
        full_steps = int(env.motion.lengths.max()) - 1
        horizon = min(full_steps, args.max_steps) if args.max_steps else full_steps
        for _ in range(horizon):
            if not bool(active.any()):
                break
            action = policy.actor(obs["policy"])
            obs, reward, terminated, truncated, info = env.step(action)
            total += torch.where(active, reward, 0)
            steps += active
            transition = info["transition"]
            reference = env.motion.frame(transition["frame"])
            measured = active & transition["usable"]
            measured_steps += measured
            for name, error in tracking_errors(transition["state"], reference).items():
                error_sums[name] += torch.where(measured, error, 0)
            for name, mask in env.task.failure_reasons(
                transition["state"], reference
            ).items():
                failure_reasons[name] |= (
                    active & transition["usable"] & terminated & mask
                )
            failure_reasons["non_finite"] |= active & ~transition["usable"]
            if args.trajectories:
                trajectories.append(
                    pack_robot_state(info["transition"]["state"]).cpu().numpy()
                )
                references.append(pack_robot_state(reference).cpu().numpy())
                alive.append(active.cpu().numpy())
            failed |= terminated & active
            active &= ~(terminated | truncated)
        result = {
            "checkpoint": str(args.checkpoint),
            "iteration": state["schedule"]["iterations"],
            "kind": "evaluation",
            "mode": args.mode,
            "environment_metadata": metadata.name,
            "full_horizon": horizon == full_steps,
            "max_steps": args.max_steps,
            "training_kind": training_config["kind"],
            "evaluation": "full-clip physical rollout; fixed reference starts; no observation/action noise",
            "seed": args.seed,
            "evaluation_dataset": {
                "path": str(get_data_path(options["motion_path"])),
                "quaternion_order": options["quaternion_order"],
                "clips": env.motion.names,
                "frames": env.motion.lengths.cpu().tolist(),
                "fps": env.motion.fps,
            },
            "reset": {
                "fixed_clip_assignment": True,
                "assigned_start_frames": env.motion.starts.cpu().tolist(),
                "final_episode_start_frames": env._episode_start.cpu().tolist(),
            },
            "precision": {
                "policy": "float32",
                "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            },
            "tracking_error_definition": "RMSE over usable physical successor frames, including the terminal frame; body pose errors average tracked links after heading/horizontal-root alignment; root pose and body velocities use world coordinates; position in m, angle in rad, linear velocity in m/s, angular velocity in rad/s",
            "seconds": time.perf_counter() - start,
            "clips": [
                {
                    "clip": name,
                    "steps": int(steps[i]),
                    "reference_steps": int(env.motion.lengths[i]) - 1,
                    "failed": bool(failed[i]),
                    "end_reasons": (
                        [
                            name
                            for name, mask in failure_reasons.items()
                            if bool(mask[i])
                        ]
                        if bool(failed[i])
                        else [
                            "evaluation_limit" if bool(active[i]) else "reference_end"
                        ]
                    ),
                    "measured_steps": int(measured_steps[i]),
                    "tracking_rmse": {
                        name: (
                            float((value[i] / measured_steps[i]).sqrt())
                            if int(measured_steps[i])
                            else None
                        )
                        for name, value in error_sums.items()
                    },
                    "return": float(total[i]),
                    "normalized_return": float(
                        total[i]
                        / ((env.motion.lengths[i] - 1) * env.task.max_step_reward)
                    ),
                }
                for i, name in enumerate(env.motion.names)
            ],
        }
        result["failures"] = sum(x["failed"] for x in result["clips"])
        result["macro_normalized_return"] = sum(
            x["normalized_return"] for x in result["clips"]
        ) / len(result["clips"])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        if args.trajectories:
            args.trajectories.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.trajectories,
                format_version=1,
                state=np.asarray(trajectories),
                reference=np.asarray(references),
                active=np.asarray(alive),
                names=env.motion.names,
                joint_names=env.motion.joint_names,
                tracked_link_names=[
                    env.task.kinematics.link_names[i]
                    for i in env.task.tracked.cpu().tolist()
                ],
                state_layout="root_position[3],root_rotation_row_major[9],root_linear_velocity[3],root_angular_velocity[3],joint_position[dof],joint_velocity[dof]",
                dt=env.step_dt,
            )
        print(json.dumps(result, indent=2), flush=True)
    finally:
        env.close(exit_process=False)


if __name__ == "__main__":
    main()
