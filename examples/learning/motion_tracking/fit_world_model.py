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

"""Collect G1 physical transitions and fit an eight-step world model through shared SL."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml

from embodichain.lab.gym.utils.gym_utils import config_to_cfg
from embodichain.learning.sl import SupervisedBatch, SupervisedLearner
from embodichain_models.dynamics.rwm import (
    KinematicWorldModel,
    KinematicWorldModelCfg,
    TransitionReplay,
    RobotWorldModelObjective,
    replay_supervised_batch,
)
from embodichain_models.dynamics.rwm.robot_state import pack_robot_state
from embodichain_tasks.locomotion.tracking.g1_motion import UnitreeG1MotionEnv


def sample_batch(
    replay: TransitionReplay, batch_size: int, horizon: int
) -> SupervisedBatch:
    """Draw complete usable windows with bounded retries and a fixed training batch."""
    for _ in range(4):
        windows = replay.sample_windows(batch_size * 4, horizon)
        selected = windows["valid"].all(1) & windows["usable"].all(1)
        if int(selected.sum()) >= batch_size:
            return replay_supervised_batch(
                {k: v[selected][:batch_size] for k, v in windows.items()},
                complete_windows=True,
            )
    raise RuntimeError(
        "Insufficient complete windows; collect longer episodes or reduce the horizon"
    )


@torch.no_grad()
def prediction_metrics(
    model: KinematicWorldModel, batch: SupervisedBatch, *, identity: bool = False
) -> dict[str, float]:
    """Evaluate fixed held-out episodes without updating normalizer statistics."""
    state = batch.inputs["initial_state"]
    values = {}
    for step in range(batch.inputs["actions"].shape[1]):
        if not identity:
            state = model(
                state,
                batch.inputs["actions"][:, step],
                batch.inputs["contexts"][:, step],
            )
        error = model.state_error(state, batch.targets[:, step])
        if step in (0, batch.inputs["actions"].shape[1] - 1):
            prefix = f"step_{step+1}"
            joints = (state.shape[-1] - 18) // 2
            values.update(
                {
                    f"{prefix}/normalized_mse": float(
                        (error / model.error_scale()).square().mean()
                    ),
                    f"{prefix}/root_position_m": float(
                        error[:, :3].norm(dim=-1).mean()
                    ),
                    f"{prefix}/root_rotation_rad": float(
                        error[:, 3:6].norm(dim=-1).mean()
                    ),
                    f"{prefix}/joint_position_rmse_rad": float(
                        error[:, 12 : 12 + joints].square().mean().sqrt()
                    ),
                    f"{prefix}/joint_velocity_rmse_rad_s": float(
                        error[:, 12 + joints :].square().mean().sqrt()
                    ),
                }
            )
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, required=True, help="G1 Gym deployment YAML"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=160)
    parser.add_argument("--eval-envs", type=int, default=32)
    parser.add_argument("--collect-steps", type=int, default=512)
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument(
        "--hidden-dims", type=int, nargs="+", default=[2048, 2048, 2048]
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path(__file__).with_name("world_model_weights.yaml"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--fp32", action="store_true", help="Disable BF16 network arithmetic"
    )
    args = parser.parse_args()
    if (
        not 0 < args.eval_envs < args.num_envs
        or args.collect_steps < args.horizon
        or args.updates < 1
    ):
        parser.error(
            "Need nonempty train/evaluation partitions, enough collection steps and positive updates"
        )
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    config = yaml.safe_load(args.config.read_text())
    config["num_envs"] = args.num_envs
    cfg = config_to_cfg(config)
    cfg.init_rollout_buffer = False
    env = UnitreeG1MotionEnv(cfg)
    try:
        env.reset(seed=args.seed)
        task = env.task
        wm_cfg = KinematicWorldModelCfg(
            joint_names=env.motion_joint_names,
            resting_position=tuple(task.resting_position.cpu().tolist()),
            action_dim=task.dof,
            context_dim=task.dof + 4,
            hidden_dims=tuple(args.hidden_dims),
            step_dt=env.step_dt,
            amp_bfloat16=not args.fp32,
        )
        model = KinematicWorldModel(Path(cfg.robot.fpath).read_text(), wm_cfg).to(
            env.device
        )
        table = yaml.safe_load(args.weights.read_text())
        if set(table) != set(env.motion_joint_names):
            raise ValueError(
                "Joint error weights must cover the configured joint names"
            )
        joint_weights = torch.tensor(
            [table[name] for name in env.motion_joint_names], device=env.device
        ).clamp(0.25, 4)
        joint_weights = joint_weights / joint_weights.mean()
        weights = torch.cat((joint_weights.new_ones(12), joint_weights, joint_weights))
        objective = RobotWorldModelObjective(
            None, temporal_reduction="sum", error_weights=tuple(weights.cpu().tolist())
        )
        learner = SupervisedLearner(
            model,
            objective,
            torch.optim.Adam(model.parameters(), lr=2e-4),
            max_grad_norm=1.0,
        )
        extras = {
            "frame": (None, torch.int64),
            "policy_action": (task.dof, torch.float32),
            "previous_action": (task.dof, torch.float32),
            "usable": (None, torch.bool),
        }
        train_envs = args.num_envs - args.eval_envs
        train = TransitionReplay(
            args.collect_steps,
            train_envs,
            model.state_dim,
            task.dof,
            context_dim=wm_cfg.context_dim,
            device=env.device,
            seed=args.seed + 1,
            extra_fields=extras,
        )
        holdout = TransitionReplay(
            args.collect_steps,
            args.eval_envs,
            model.state_dim,
            task.dof,
            context_dim=wm_cfg.context_dim,
            device=env.device,
            seed=args.seed + 2,
            extra_fields=extras,
        )
        manifest = {
            "args": {
                k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
            },
            "model": wm_cfg.to_dict(),
            "error_weights": weights.cpu().tolist(),
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "collection": "reference joint-position commands with 0.1 Gaussian action noise in 75% of environments",
            "holdout": "last eval_envs worlds; their episodes never enter training or normalization",
            "replay_bytes": train.allocated_bytes + holdout.allocated_bytes,
        }
        (args.output / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")
        begin = time.perf_counter()
        invalid = 0
        with torch.no_grad():
            for step in range(args.collect_steps):
                before = env.capture_transition_start()
                reference = env.motion.future(
                    env.motion_frame, task.future_offsets.new_ones(1)
                )["q"][:, 0]
                action = (
                    (reference - task.resting_position)
                    / task.action_scale
                    * (1 - task.reference_weight)
                )
                noise = torch.randn_like(action) * 0.1
                noise[: round(args.num_envs * 0.25)] = 0
                _, _, terminated, truncated, info = env.step(action + noise)
                after = info["transition"]
                state, successor = pack_robot_state(before["state"]), pack_robot_state(
                    after["state"]
                )
                usable = after["usable"] & torch.isfinite(state).all(-1)
                invalid += int((~usable).sum())
                payload = {
                    "frame": before["frame"],
                    "previous_action": before["previous_action"],
                    "policy_action": after["action"],
                    "usable": usable,
                }
                for replay, rows in (
                    (train, slice(None, train_envs)),
                    (holdout, slice(train_envs, None)),
                ):
                    replay.append(
                        state[rows],
                        after["setpoint_action"][rows],
                        successor[rows],
                        context=before["context"][rows],
                        terminated=terminated[rows],
                        truncated=truncated[rows],
                        extras={k: v[rows] for k, v in payload.items()},
                    )
                valid_train = usable[:train_envs]
                model.update_normalization(
                    state[:train_envs][valid_train],
                    after["setpoint_action"][:train_envs][valid_train],
                    successor[:train_envs][valid_train],
                    before["context"][:train_envs][valid_train],
                )
                if (step + 1) % 64 == 0:
                    print(
                        json.dumps(
                            {
                                "collected_vector_steps": step + 1,
                                "invalid_transitions": invalid,
                            }
                        ),
                        flush=True,
                    )
        torch.cuda.synchronize()
        collection_seconds = time.perf_counter() - begin
        validation = sample_batch(holdout, args.eval_batch_size, args.horizon)
        identity = prediction_metrics(model, validation, identity=True)
        initial = prediction_metrics(model, validation)
        print(json.dumps({"identity": identity, "initial": initial}), flush=True)
        torch.cuda.reset_peak_memory_stats()
        begin = time.perf_counter()
        with (args.output / "metrics.jsonl").open("w") as stream:
            for update in range(args.updates):
                metrics = learner.update(
                    sample_batch(train, args.batch_size, args.horizon)
                )
                if (update + 1) % 50 == 0 or update == 0:
                    metrics["evaluation"] = prediction_metrics(model, validation)
                    print(json.dumps(metrics), flush=True)
                stream.write(json.dumps(metrics) + "\n")
                stream.flush()
        torch.cuda.synchronize()
        training_seconds = time.perf_counter() - begin
        final = prediction_metrics(model, validation)
        report = {
            "identity": identity,
            "initial": initial,
            "final": final,
            "updates": learner.num_updates,
            "train_transitions": args.collect_steps * train_envs,
            "holdout_transitions": args.collect_steps * args.eval_envs,
            "invalid_transitions": invalid,
            "collection_seconds": collection_seconds,
            "training_seconds": training_seconds,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "replay_bytes": manifest["replay_bytes"],
            "normalization_count": int(model.input_normalizer.count),
            "improved_over_identity": final[f"step_{args.horizon}/normalized_mse"]
            < identity[f"step_{args.horizon}/normalized_mse"],
            "improved_over_initial": final[f"step_{args.horizon}/normalized_mse"]
            < initial[f"step_{args.horizon}/normalized_mse"],
        }
        torch.save(
            {"learner": learner.state_dict(), "recipe": manifest},
            args.output / "world_model.pt",
        )
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        if not report["improved_over_identity"] or not report["improved_over_initial"]:
            raise RuntimeError("Held-out eight-step prediction did not improve")
    finally:
        env.close(exit_process=False)


if __name__ == "__main__":
    main()
