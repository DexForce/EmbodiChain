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

"""Measure frozen WM prediction on newly collected, episode-local physical windows."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import torch

from embodichain.data import get_data_path
from embodichain_models.dynamics.rwm.robot_state import unpack_robot_state
from run_metadata import write_run_metadata
from train import MotionTraining
from _configuration import normalize_recipe


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control", choices=("policy", "reference"), default="policy")
    parser.add_argument("--num-envs", type=int, default=80)
    parser.add_argument("--collect-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 8, 16, 32])
    parser.add_argument("--seed", type=int, default=10042)
    parser.add_argument("--motion-path", type=Path)
    parser.add_argument("--quaternion-order", choices=("xyzw", "wxyz"))
    args = parser.parse_args()
    horizons = sorted(set(args.horizons))
    if min(horizons) < 1 or args.collect_steps < max(64, horizons[-1]):
        parser.error(
            "Collect at least 64 steps and cover the longest prediction horizon."
        )
    if min(args.num_envs, args.batch_size) < 1:
        parser.error("Environment and batch sizes must be positive.")
    args.output.mkdir(parents=True, exist_ok=False)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    recipe = normalize_recipe(saved["config"])
    config = recipe["workflow"]
    gym = deepcopy(saved["gym_config"])
    options = gym["env"]["extensions"]["motion_tracking"]
    if args.motion_path:
        options["motion_path"] = str(args.motion_path.resolve())
    if args.quaternion_order:
        options["quaternion_order"] = args.quaternion_order
    options["compile_mode"] = None
    (args.output / "gym.json").write_text(json.dumps(gym, indent=2) + "\n")
    config.update(
        kind="test",
        seed=args.seed,
        num_envs=args.num_envs,
        buffer_steps=args.collect_steps,
        compile_mode=None,
        collection_env=str((args.output / "gym.json").resolve()),
        evaluation_env=str((args.output / "gym.json").resolve()),
    )
    recipe["trainer"]["num_envs"] = args.num_envs
    recipe["trainer"]["learning_env"].get("cfg", {}).pop("compile_mode", None)
    config.setdefault("task_options", {}).update(options)
    workflow = MotionTraining(recipe, args.output)
    try:
        model, policy, env = workflow.model, workflow.policy, workflow.env
        model.load_state_dict(saved["model"]["model"])
        policy.load_state_dict(saved["policy"])
        model.eval()
        policy.eval()
        metadata = write_run_metadata(
            args.output,
            invocation_started_at=datetime.now(timezone.utc).isoformat(),
            config=config,
            gym_config=gym,
            motion=workflow.identity["motion"],
            motion_path=Path(get_data_path(env.motion_cfg.motion_path)),
            device=env.device,
            resume_checkpoint=args.checkpoint.resolve(),
        )
        start = time.perf_counter()
        for _ in range(args.collect_steps):
            workflow.collect(
                update_normalization=False,
                reference_control=args.control == "reference",
            )
        collection_seconds = time.perf_counter() - start
        for _ in range(8):
            windows = workflow.replay.sample_windows(args.batch_size * 4, horizons[-1])
            valid = windows["valid"].all(1) & windows["usable"].all(1)
            if int(valid.sum()) >= args.batch_size:
                windows = {k: v[valid][: args.batch_size] for k, v in windows.items()}
                break
        else:
            raise RuntimeError(
                "Insufficient complete physical windows; increase collection steps."
            )
        # Preserve the actual measured controls and contexts for every horizon.
        predicted = windows["state"][:, 0]
        identity = predicted.clone()
        metrics = {}
        prediction_started = time.perf_counter()
        for step in range(horizons[-1]):
            predicted = model(
                predicted, windows["action"][:, step], windows["context"][:, step]
            )
            if not bool(torch.isfinite(predicted).all()):
                raise RuntimeError(f"Non-finite WM prediction at horizon {step + 1}")
            if step + 1 not in horizons:
                continue
            target = windows["next_state"][:, step]
            reference = env.motion.frame(windows["frame"][:, step] + 1)
            actual_state = unpack_robot_state(target)
            actual_reward, _ = env.task.reward(
                actual_state,
                reference,
                windows["policy_action"][:, step],
                windows["previous_action"][:, step],
            )
            actual_done, actual_continuation = env.task.termination(
                actual_state, reference
            )
            row = {}
            for label, state in (("model", predicted), ("state_hold", identity)):
                error = model.state_error(state, target)
                joints = env.task.dof
                measured = unpack_robot_state(state)
                reward, _ = env.task.reward(
                    measured,
                    reference,
                    windows["policy_action"][:, step],
                    windows["previous_action"][:, step],
                )
                done, continuation = env.task.termination(measured, reference)
                row[label] = {
                    "normalized_mse": float(
                        (error / model.error_scale()).square().mean()
                    ),
                    "root_position_m": float(error[:, :3].norm(dim=-1).mean()),
                    "root_rotation_rad": float(error[:, 3:6].norm(dim=-1).mean()),
                    "joint_position_rmse_rad": float(
                        error[:, 12 : 12 + joints].square().mean().sqrt()
                    ),
                    "joint_velocity_rmse_rad_s": float(
                        error[:, 12 + joints :].square().mean().sqrt()
                    ),
                    "normalized_reward_mae": float(
                        (reward - actual_reward).abs().mean() / env.task.max_step_reward
                    ),
                    "hard_termination_disagreement": float(
                        (done != actual_done).float().mean()
                    ),
                    "continuation_mae": float(
                        (continuation - actual_continuation).abs().mean()
                    ),
                }
            row["physical_terminal_count"] = int(actual_done.sum())
            metrics[str(step + 1)] = row
        unchanged = all(
            torch.equal(value.cpu(), saved["model"]["model"][key])
            for key, value in model.state_dict().items()
            if isinstance(value, torch.Tensor)
        )
        policy_unchanged = all(
            torch.equal(value.cpu(), saved["policy"][key])
            for key, value in policy.state_dict().items()
            if isinstance(value, torch.Tensor)
        )
        assert (
            unchanged and policy_unchanged
        ), "Validation changed weights or normalization statistics."
        report = {
            "kind": "validation",
            "checkpoint": str(args.checkpoint.resolve()),
            "environment_metadata": metadata.name,
            "control": args.control,
            "seed": args.seed,
            "collection_steps": args.collect_steps,
            "num_envs": args.num_envs,
            "windows": args.batch_size,
            "sampled_clip_counts": {
                env.motion.names[i]: int(n)
                for i, n in enumerate(
                    torch.bincount(
                        env.motion.clip_ids(windows["frame"][:, 0]),
                        minlength=len(env.motion.names),
                    )
                    .cpu()
                    .tolist()
                )
            },
            "horizons": horizons,
            "prediction_precision": "Checkpoint network precision; FP32 state integration and errors",
            "normalization_floor": None,
            "split": "Fresh physical episodes from the frozen checkpoint; no fitting or normalization updates. Complete windows respect episode boundaries.",
            "model_and_statistics_unchanged": unchanged,
            "policy_and_statistics_unchanged": policy_unchanged,
            "invalid_transitions": workflow.invalid_transitions,
            "collection_seconds": collection_seconds,
            "prediction_seconds": time.perf_counter() - prediction_started,
            "metrics": metrics,
        }
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        workflow.close()


if __name__ == "__main__":
    main()
