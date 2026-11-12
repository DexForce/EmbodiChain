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

"""Collect real Sharpa transitions, fit RWM, optimize APG, and evaluate physics."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import torch
from _simulation import SharpaSimulation
from embodichain_models.dynamics.rwm import TransitionReplay
from embodichain_tasks.manipulation.dexterous.sharpa_inhand import (
    SharpaInHandModelEnv,
    SharpaInHandWorldModel,
    _cylinder_axis_observation,
    _joint_limit_aware_action,
    _stable_rotation_mask,
    _transition_rates,
    _WorldObjective,
)
from tensordict import TensorDict

from embodichain.lab.sim import SimulationManager
from embodichain.learning.models import MLP
from embodichain.learning.rl.algo.apg import APG, APGCfg
from embodichain.learning.rl.differentiable_trainer import (
    DifferentiableTrainer,
    DifferentiableTrainerCfg,
)
from embodichain.learning.rl.policies import ActorOnly
from embodichain.learning.rl.utils import OptimizerCfg
from embodichain.learning.sl import SupervisedBatch, SupervisedLearner

__all__ = ["main"]

_GRASP_URL = (
    "https://raw.githubusercontent.com/Mingrui-Yu/sharpa_rl_unilab/"
    "9816885a3b1fbdb947db71e72c27ed75effcbf9f/"
    "src/sharpa_rl_unilab/assets/caches/sharpa_grasp_linspace_1.npy"
)
_GRASP_SHA256 = "22a2dac7f6ecdf73764d06c74f3810b21ec9af43c609d283d61b788b8383c842"


def _grasp_cache(path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(_GRASP_URL, timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != _GRASP_SHA256:
            raise ValueError("Upstream grasp cache checksum differs")
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(data)
        temporary.replace(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != _GRASP_SHA256:
        raise ValueError("Grasp cache is not the pinned scale-1 source")
    return path


class _JointLimitActor(torch.nn.Module):
    def __init__(self, limits: torch.Tensor, *, axis_policy: bool = False) -> None:
        super().__init__()
        self.axis_policy = axis_policy
        self.network = MLP(
            109, 22, (128, 128), activation="elu", last_activation="tanh"
        )
        self.register_buffer("lower", limits[:, 0].detach().clone())
        self.register_buffer("upper", limits[:, 1].detach().clone())

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        features = (
            _cylinder_axis_observation(observation) if self.axis_policy else observation
        )
        return _joint_limit_aware_action(
            self.network(features), observation, self.lower, self.upper
        )


def _policy(
    device: torch.device,
    limits: torch.Tensor,
    *,
    joint_limit_aware: bool = False,
    axis_policy: bool = False,
) -> ActorOnly:
    actor = (
        _JointLimitActor(limits, axis_policy=axis_policy)
        if joint_limit_aware or axis_policy
        else MLP(109, 22, (128, 128), activation="elu", last_activation="tanh")
    )
    policy = ActorOnly(109, 22, device, actor)
    with torch.no_grad():
        policy.log_std.fill_(-2)
        final = [
            module for module in actor.modules() if isinstance(module, torch.nn.Linear)
        ][-1]
        final.weight.zero_()
        final.bias.zero_()
    return policy


def _same_physics(
    saved: dict[str, Any], current: dict[str, Any], *, model_only: bool = False
) -> bool:
    saved, current = dict(saved), dict(current)
    # Reset distributions may change for domain-randomized collection or fixed
    # evaluation. Geometry, control and state/action contracts must still match.
    saved.pop("randomization", None)
    current.pop("randomization", None)
    if model_only:
        saved.pop("observation", None)
        current.pop("observation", None)
    return saved == current


def _action(policy: ActorOnly, observation: torch.Tensor) -> torch.Tensor:
    batch = TensorDict(
        {"obs": observation}, batch_size=[len(observation)], device=observation.device
    )
    return policy.get_action(batch, deterministic=True)["action"].clamp(-1, 1)


def _collect(
    source: SharpaSimulation,
    replay: TransitionReplay,
    model: SharpaInHandWorldModel,
    policy: ActorOnly | None,
    steps: int,
    noise: float,
    *,
    update_statistics: bool,
    grasp_exploration: float = 0.0,
) -> dict[str, float]:
    source.reset()
    rewards, drops = [], []
    perturbation = torch.zeros(source.num_envs, 22, device=source.device)
    closing = torch.zeros_like(perturbation)
    closing[::2, [0, 2, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 18, 20, 21]] = (
        grasp_exploration
    )
    for index in range(steps):
        with torch.no_grad():
            action = (
                torch.zeros(source.num_envs, 22, device=source.device)
                if policy is None
                else _action(policy, source.observation())
            )
            perturbation = 0.8 * perturbation + 0.6 * noise * torch.randn_like(
                perturbation
            )
            action = action + perturbation + (source.steps < 10)[:, None] * closing
            transition = source.step(action.clamp(-1, 1))
            # Each collection block ends a replay episode before the next full
            # reset; windows must never cross evaluation or collection resets.
            truncated = transition["truncated"] | (index == steps - 1)
            replay.append(
                transition["state"],
                transition["action"],
                transition["next_state"],
                context=transition["context"],
                terminated=transition["terminated"],
                truncated=truncated,
            )
            if update_statistics:
                model.update_normalization(
                    transition["state"],
                    transition["action"],
                    transition["next_state"],
                )
            rewards.append(transition["reward"])
            drops.append(transition["terminated"])
            ids = (transition["terminated"] | truncated).nonzero().flatten()
            if ids.numel():
                source.reset(ids)
        if (index + 1) % 128 == 0:
            print(
                json.dumps(
                    {
                        "collection_steps": index + 1,
                        "collection_total": steps,
                        "training_statistics": update_statistics,
                    }
                ),
                flush=True,
            )
    return {
        "real_reward": float(torch.stack(rewards).mean()),
        "real_drops": float(torch.stack(drops).sum()),
        "real_transitions": float(steps * source.num_envs),
    }


def _batch(replay: TransitionReplay, batch_size: int, horizon: int) -> SupervisedBatch:
    windows = replay.sample_windows(batch_size, horizon)
    # RWM derives fixed-palm context from its own current predicted state.
    # The separate replay context is task reset/target state, never model input.
    return SupervisedBatch(
        {"initial_state": windows["state"][:, 0], "actions": windows["action"]},
        windows["next_state"],
        {"valid": windows["valid"]},
    )


@torch.no_grad()
def _evaluate(
    source: SharpaSimulation,
    policy: ActorOnly,
    steps: int,
    seed: int,
    trajectory_path: Path | None = None,
) -> dict[str, Any]:
    initial_state = source.reset(seed=seed)
    initial_context = source.context.clone()
    alive = torch.ones(source.num_envs, device=source.device, dtype=torch.bool)
    angle = torch.zeros(source.num_envs, device=source.device)
    survival = torch.zeros_like(angle)
    returns = torch.zeros_like(angle)
    positive_steps = torch.zeros_like(angle, dtype=torch.int64)
    window_angles = torch.zeros(
        source.num_envs, (steps + 99) // 100, device=source.device
    )
    positive_windows = torch.zeros_like(window_angles, dtype=torch.int64)
    states, actions, masks = [], [], []
    for index in range(steps):
        action = _action(policy, source.observation())
        transition = source.step(action)
        _, angular, _ = _transition_rates(
            transition["state"], transition["next_state"], source.cfg.step_dt
        )
        if trajectory_path is not None:
            states.append(transition["next_state"])
            actions.append(action)
            masks.append(alive.clone())
        angle += alive * angular[:, 2] * source.cfg.step_dt
        window_angles[:, index // 100] += alive * angular[:, 2] * source.cfg.step_dt
        positive_steps += alive * (angular[:, 2] > 0.1)
        positive_windows[:, index // 100] += alive * (angular[:, 2] > 0.1)
        returns += alive * transition["reward"]
        survival += alive * source.cfg.step_dt
        alive &= ~transition["terminated"]
    if trajectory_path is not None:
        torch.save(
            {
                "format": "sharpa-inhand-evaluation-v1",
                "initial_state": initial_state.cpu(),
                "initial_context": initial_context.cpu(),
                "state": torch.stack(states).cpu(),
                "action": torch.stack(actions).cpu(),
                "alive_before_step": torch.stack(masks).cpu(),
                "step_dt": source.cfg.step_dt,
                "seed": seed,
                "physics_identity": source.identity,
                "reward_identity": source.reward_identity,
                "rotation_measurement": "world SO(3) control-period increments",
            },
            trajectory_path,
        )
    window_lengths = torch.tensor(
        [min(100, steps - i * 100) for i in range(window_angles.shape[1])],
        device=source.device,
    )
    stable = _stable_rotation_mask(
        alive,
        angle,
        positive_steps,
        positive_windows[:, -1],
        steps=steps,
        final_window_steps=min(100, steps - 100 * (window_angles.shape[1] - 1)),
    )
    positive_windows = positive_windows / window_lengths
    return {
        "return": float(returns.mean()),
        "rotation_rad": float(angle.mean()),
        "speed_fixed_window_rad_s": float(angle.mean() / (steps * source.cfg.step_dt)),
        "survival_seconds": float(survival.mean()),
        "drop_rate": float((~alive).float().mean()),
        "window_seconds": steps * source.cfg.step_dt,
        "scenes": source.num_envs,
        "seed": seed,
        "reset_domain": source.identity["randomization"],
        "reward_identity": source.reward_identity,
        "rotation_measurement": "world SO(3) increments over each control period",
        "per_scene_rotation_rad": angle.cpu().tolist(),
        "per_scene_survival_seconds": survival.cpu().tolist(),
        "per_scene_positive_fraction": (positive_steps / steps).cpu().tolist(),
        "per_scene_window_rotation_rad": window_angles.cpu().tolist(),
        "per_scene_window_positive_fraction": positive_windows.cpu().tolist(),
        "stable_scene_fraction": float(stable.float().mean()),
    }


def _evaluation_rank(metrics: dict[str, Any]) -> tuple[float, ...]:
    windows = torch.tensor(metrics["per_scene_window_rotation_rad"]).mean(0)
    return (
        metrics["stable_scene_fraction"],
        1 - metrics["drop_rate"],
        metrics["survival_seconds"],
        float(windows.min()),
        metrics["rotation_rad"],
    )


def _snapshot(
    output: Path,
    source: SharpaSimulation,
    model: SharpaInHandWorldModel,
    policy: ActorOnly,
    replay: TransitionReplay,
    trainer: DifferentiableTrainer,
    learner: SupervisedLearner,
    args: argparse.Namespace,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(output / "apg.pt")
    starts = replay.sample_starts(args.start_bank_size, 1)["start"]
    payloads = {
        "world_model.pt": {
            "format": "sharpa-inhand-world-model-v1",
            "model": model.state_dict(),
            "hidden_dims": list(args.world_hidden),
            "servo_context": model.servo_context,
            "world_frame": model.world_frame,
            "axial_symmetry": model.axial_symmetry,
            "initial_states": starts["state"].cpu(),
            "initial_contexts": starts["context"].cpu(),
            "joint_limits": source.joint_limits.cpu(),
            "physics_identity": source.identity,
            "reward_identity": source.reward_identity,
        },
        "policy.pt": {
            "format": "sharpa-inhand-policy-v1",
            "policy": policy.state_dict(),
            "joint_limit_aware": isinstance(policy.actor, _JointLimitActor),
            "axis_policy": getattr(policy.actor, "axis_policy", False),
            "physics_identity": source.identity,
            "reward_identity": source.reward_identity,
        },
        "world_learner.pt": learner.state_dict(),
    }
    for name, payload in payloads.items():
        temporary = output / (name + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(output / name)
    (output / "training.json").write_text(
        json.dumps(
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            indent=2,
        )
        + "\n"
    )


def _run(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    grasp = _grasp_cache(args.grasp_cache.expanduser().resolve())
    source = evaluation = imagined = None
    try:
        source = SharpaSimulation(
            args.hand_urdf,
            grasp,
            output / "physics",
            num_envs=args.num_envs,
            device=args.device,
            seed=args.seed,
            renderer=args.renderer,
            randomize_yaw=args.randomize_training_yaw and args.evaluate_policy is None,
        )
        if args.evaluate_policy is not None:
            saved = torch.load(
                args.evaluate_policy, map_location=device, weights_only=True
            )
            if saved.get("format") != "sharpa-inhand-policy-v1":
                raise ValueError("Expected the online workflow's policy.pt export")
            if not _same_physics(saved["physics_identity"], source.identity):
                raise ValueError(
                    "Saved policy physical identity differs from evaluation"
                )
            policy = _policy(
                device,
                source.joint_limits,
                joint_limit_aware=saved.get("joint_limit_aware", False),
                axis_policy=saved.get("axis_policy", False),
            )
            policy.load_state_dict(saved["policy"])
            metrics = _evaluate(
                source,
                policy,
                args.eval_steps,
                args.seed + 10000,
                output / "evaluation.pt",
            )
            (output / "report.json").write_text(json.dumps(metrics, indent=2) + "\n")
            print(json.dumps(metrics), flush=True)
            return
        evaluation = SharpaSimulation(
            args.hand_urdf,
            grasp,
            output / "evaluation",
            num_envs=args.num_envs,
            device=args.device,
            seed=args.seed + 10000,
            renderer=args.renderer,
        )
        model = SharpaInHandWorldModel(
            tuple(args.world_hidden),
            servo_context=args.servo_context,
            world_frame=args.world_frame,
            axial_symmetry=args.axial_symmetry,
        ).to(device)
        policy = _policy(
            device,
            source.joint_limits,
            joint_limit_aware=args.joint_limit_aware_policy,
            axis_policy=args.axis_policy,
        )
        if args.initialize_from is not None:
            previous = torch.load(
                args.initialize_from / "world_model.pt",
                map_location=device,
                weights_only=True,
            )
            if not _same_physics(
                previous["physics_identity"],
                source.identity,
                model_only=args.reset_policy,
            ):
                raise ValueError("Warm-start physics identity differs")
            model.load_state_dict(previous["model"])
            if not args.reset_policy:
                previous = torch.load(
                    args.initialize_from / "policy.pt",
                    map_location=device,
                    weights_only=True,
                )
                old_aware = previous.get("joint_limit_aware", False)
                args.axis_policy = args.axis_policy or previous.get(
                    "axis_policy", False
                )
                if old_aware:
                    args.joint_limit_aware_policy = True
                    policy = _policy(
                        device,
                        source.joint_limits,
                        joint_limit_aware=True,
                        axis_policy=args.axis_policy,
                    )
                    policy.load_state_dict(previous["policy"])
                elif args.joint_limit_aware_policy:
                    original = _policy(device, source.joint_limits)
                    original.load_state_dict(previous["policy"])
                    policy.actor.network.load_state_dict(original.actor.state_dict())
                    with torch.no_grad():
                        policy.log_std.copy_(original.log_std)
                else:
                    policy.load_state_dict(previous["policy"])
        replay = TransitionReplay(
            args.replay_capacity,
            args.num_envs,
            62,
            22,
            context_dim=47,
            device=device,
            seed=args.seed,
            max_bytes=args.replay_max_mb * 1024**2,
        )
        holdout = TransitionReplay(
            args.warmup_steps,
            args.num_envs,
            62,
            22,
            context_dim=47,
            device=device,
            seed=args.seed + 20000,
            max_bytes=args.replay_max_mb * 1024**2,
        )
        _collect(
            evaluation,
            holdout,
            model,
            None,
            args.warmup_steps,
            0.3,
            update_statistics=False,
            grasp_exploration=args.grasp_exploration,
        )
        _collect(
            source,
            replay,
            model,
            None,
            args.warmup_steps,
            0.3,
            update_statistics=args.initialize_from is None,
            grasp_exploration=args.grasp_exploration,
        )
        initial_eval = _evaluate(
            evaluation,
            policy,
            args.eval_steps,
            args.seed + 10000,
            output / "initial_evaluation.pt",
        )
        starts = replay.sample_starts(args.start_bank_size, 1)["start"]
        imagined = SharpaInHandModelEnv(
            args.imagination_envs,
            device,
            max_episode_steps=args.horizon,
            model=SharpaInHandWorldModel(
                tuple(args.world_hidden),
                servo_context=args.servo_context,
                world_frame=args.world_frame,
                axial_symmetry=args.axial_symmetry,
            ),
            initial_states=starts["state"],
            initial_contexts=starts["context"],
            joint_limits=source.joint_limits,
            height_penalty=args.height_penalty,
        )
        learner = SupervisedLearner(
            model,
            _WorldObjective(),
            torch.optim.Adam(model.parameters(), lr=args.world_lr),
            max_grad_norm=1,
        )
        algorithm = APG(
            APGCfg(
                device=str(device),
                optimizer=OptimizerCfg(learning_rate=args.policy_lr),
                gamma=0.99,
                max_grad_norm=0.5,
                max_grad_norm_before_clip=1e6,
            ),
            policy,
        )
        trainer = DifferentiableTrainer(
            DifferentiableTrainerCfg(
                segment_length=args.horizon,
                update_horizon=args.horizon,
                rollout_mode="complete",
                deterministic_actions=True,
                clip_actions_to_space=True,
                action_adjoint_max_norm=1.0,
                checkpoint_dir=str(output / "checkpoints"),
                experiment_name="sharpa_inhand",
            ),
            imagined,
            policy,
            algorithm,
        )
        objective = _WorldObjective()
        for index in range(args.world_pretrain_updates):
            learner.update(_batch(replay, args.batch_size, args.world_horizon))
            if (index + 1) % 100 == 0:
                print(json.dumps({"world_pretrain_update": index + 1}), flush=True)
        history = []
        best_rank = (-1.0,)
        started = time.perf_counter()
        for iteration in range(args.iterations):
            collection = _collect(
                source,
                replay,
                model,
                policy,
                args.collection_steps,
                args.exploration,
                update_statistics=False,
                grasp_exploration=args.grasp_exploration,
            )
            for _ in range(args.world_updates):
                learner.update(_batch(replay, args.batch_size, args.world_horizon))
            with torch.no_grad():
                validation = objective.compute_loss(
                    model, _batch(holdout, args.batch_size, args.world_horizon)
                )
            # All previous policy backward work is complete before replacing
            # the inference weights or starts. Training model remains unfrozen.
            imagined.model.load_state_dict(model.state_dict())
            starts = replay.sample_starts(args.start_bank_size, 1)["start"]
            imagined.replace_start_bank(starts["state"], starts["context"])
            before = [parameter.detach().clone() for parameter in policy.parameters()]
            imagined.cfg.rotation_weight = 2.5 * min(
                1.0, (iteration + 1) / max(1, args.rotation_warmup_iterations)
            )
            outcome = trainer.train(
                total_updates=trainer.num_updates + args.policy_updates
            )
            metrics = {
                "iteration": iteration,
                **collection,
                "world_holdout_loss": float(validation.loss),
                "world_position_error_m": float(validation.metrics["position_error_m"]),
                "policy_updates": trainer.num_updates,
                "imagined_rotation_weight": imagined.cfg.rotation_weight,
                "imagined_height_penalty": imagined.cfg.height_penalty,
                "seconds": time.perf_counter() - started,
                "policy_parameter_delta": float(
                    torch.stack(
                        [
                            (parameter.detach() - old).square().sum()
                            for parameter, old in zip(policy.parameters(), before)
                        ]
                    )
                    .sum()
                    .sqrt()
                ),
                "policy_metrics": outcome["last_train_metrics"],
            }
            if (
                iteration == args.iterations - 1
                or (iteration + 1) % args.eval_every == 0
            ):
                metrics["evaluation"] = _evaluate(
                    evaluation,
                    policy,
                    args.eval_steps,
                    args.seed + 10000,
                    output / "final_evaluation.pt",
                )
                rank = _evaluation_rank(metrics["evaluation"])
                snapshot = output / "snapshots" / f"iteration_{iteration:05d}"
                _snapshot(
                    snapshot, source, model, policy, replay, trainer, learner, args
                )
                trace = (output / "final_evaluation.pt").read_bytes()
                (snapshot / "evaluation.pt").write_bytes(trace)
                (snapshot / "evaluation.json").write_text(
                    json.dumps(metrics["evaluation"], indent=2) + "\n"
                )
                if rank > best_rank:
                    best_rank = rank
                    _snapshot(
                        output / "best",
                        source,
                        model,
                        policy,
                        replay,
                        trainer,
                        learner,
                        args,
                    )
                    (output / "best" / "evaluation.pt").write_bytes(trace)
                    (output / "best" / "evaluation.json").write_text(
                        json.dumps(metrics["evaluation"], indent=2) + "\n"
                    )
            history.append(metrics)
            with (output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics) + "\n")
            print(json.dumps(metrics), flush=True)
        _snapshot(output, source, model, policy, replay, trainer, learner, args)
        report = {
            "physics_identity": source.identity,
            "reward_identity": source.reward_identity,
            "evaluation_identity": evaluation.identity,
            "initial_evaluation": initial_eval,
            "final_evaluation": history[-1]["evaluation"],
            "history": history,
            "replay_bytes": replay.allocated_bytes,
            "heldout_replay_bytes": holdout.allocated_bytes,
            "physical_transitions": (
                args.warmup_steps + args.iterations * args.collection_steps
            )
            * args.num_envs,
            "heldout_physical_transitions": args.warmup_steps * args.num_envs,
            "policy_checkpoint": trainer.latest_checkpoint_path,
            "arguments": {
                k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
            },
        }
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        if imagined is not None:
            imagined.close()
        if evaluation is not None:
            evaluation.close()
        if source is not None:
            source.close()


def main() -> None:
    """Run the bounded online real-data/world-model/APG workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hand-urdf", default="SharpaWave/right_sharpa_wave/right_sharpa_wave.urdf"
    )
    parser.add_argument(
        "--grasp-cache",
        type=Path,
        default=Path("~/.cache/embodichain/sharpa/grasp_1.npy"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--evaluate-policy",
        type=Path,
        help="Evaluate a saved online policy.pt without training",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--initialize-from",
        type=Path,
        help="Warm-start exported weights into a new measured replay, not exact recovery",
    )
    parser.add_argument("--reset-policy", action="store_true")
    parser.add_argument("--world-pretrain-updates", type=int, default=0)
    parser.add_argument("--grasp-exploration", type=float, default=0.3)
    parser.add_argument("--height-penalty", type=float, default=0.0)
    parser.add_argument("--rotation-warmup-iterations", type=int, default=0)
    parser.add_argument(
        "--renderer", choices=["no-render", "hybrid"], default="no-render"
    )
    for name, default in {
        "num-envs": 8,
        "imagination-envs": 64,
        "iterations": 50,
        "warmup-steps": 128,
        "collection-steps": 64,
        "world-updates": 30,
        "policy-updates": 4,
        "horizon": 16,
        "world-horizon": 4,
        "batch-size": 128,
        "start-bank-size": 1024,
        "replay-capacity": 1024,
        "replay-max-mb": 256,
        "eval-steps": 400,
        "eval-every": 10,
        "torch-threads": 4,
    }.items():
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--world-hidden", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--servo-context", action="store_true")
    parser.add_argument("--world-frame", action="store_true")
    parser.add_argument("--axial-symmetry", action="store_true")
    parser.add_argument("--joint-limit-aware-policy", action="store_true")
    parser.add_argument("--axis-policy", action="store_true")
    parser.add_argument("--randomize-training-yaw", action="store_true")
    parser.add_argument("--world-lr", type=float, default=1e-3)
    parser.add_argument("--policy-lr", type=float, default=2.5e-4)
    parser.add_argument("--exploration", type=float, default=0.15)
    args = parser.parse_args()
    if args.axis_policy:
        args.joint_limit_aware_policy = True
    if args.world_pretrain_updates < 0:
        parser.error("world-pretrain-updates must be nonnegative")
    if args.rotation_warmup_iterations < 0:
        parser.error("rotation-warmup-iterations must be nonnegative")
    if not 0 <= args.height_penalty < float("inf"):
        parser.error("height-penalty must be finite and nonnegative")
    for name in (
        "num_envs",
        "imagination_envs",
        "iterations",
        "warmup_steps",
        "collection_steps",
        "world_updates",
        "policy_updates",
        "horizon",
        "world_horizon",
        "batch_size",
        "start_bank_size",
        "replay_capacity",
        "replay_max_mb",
        "eval_steps",
        "eval_every",
        "torch_threads",
    ):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    if args.world_horizon > min(args.warmup_steps, args.replay_capacity):
        parser.error("world-horizon exceeds available replay steps")
    _run(args)
    # Successful return has released acquisition frames and native children.
    SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    main()
