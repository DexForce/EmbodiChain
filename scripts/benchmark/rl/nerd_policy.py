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

"""Matched, multi-seed APG/PPO training and analytical transfer benchmark.

Run from the repository root with ``python -m scripts.benchmark.rl.nerd_policy``.
Models/results stay outside Git. See nerd_policy.md for the frozen protocol.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
import warp as wp
import yaml

from embodichain.utils import configclass

__all__: list[str] = []


@configclass
class BenchmarkCfg:
    """Shared transition budget and fixed algorithm hyperparameters."""

    seeds: tuple[int, ...] = (0, 1, 2)
    robots: tuple[str, ...] = ("cartpole", "ant")
    algorithms: tuple[str, ...] = ("apg", "ppo")
    num_envs: int = 64
    horizon: int = 16
    updates: int = 512
    checkpoints: int = 8
    eval_episodes: int = 32
    eval_seed: int = 10000
    cpu_threads: int = 2
    warmup_updates: int = 2
    learning_rate: float = 0.00025
    gamma: float = 0.99
    max_grad_norm: float = 0.5
    initial_log_std: float = -2.0
    ppo_epochs: int = 4
    ppo_minibatches: int = 4

    def check(self) -> None:
        """Reject incomplete or ambiguous comparison recipes."""
        for name in (
            "num_envs",
            "horizon",
            "updates",
            "checkpoints",
            "eval_episodes",
            "cpu_threads",
            "warmup_updates",
            "ppo_epochs",
            "ppo_minibatches",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.updates % self.checkpoints:
            raise ValueError("updates must be divisible by checkpoints")
        if self.num_envs * self.horizon % self.ppo_minibatches:
            raise ValueError("PPO minibatches must divide the rollout")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Training seeds must be nonempty and unique")
        if set(self.robots) - {"cartpole", "ant"} or set(self.algorithms) - {
            "apg",
            "ppo",
        }:
            raise ValueError("Unsupported task or algorithm")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_digest(values: list[torch.Tensor]) -> str:
    result = hashlib.sha256()
    for value in values:
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def _json(path: Path, value: Any) -> None:
    """Commit complete JSON atomically; null represents unavailable measurements."""

    def clean(item: Any) -> Any:
        if isinstance(item, dict):
            return {key: clean(val) for key, val in item.items()}
        if isinstance(item, (tuple, list)):
            return [clean(val) for val in item]
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wp.synchronize_device(str(device))


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class InitialStates:
    """Pair each world's episode-indexed starts across algorithms, regardless of done order.

    The complete bank is bounded by one reset per collected vector step. Sampling
    on CPU makes its content independent of GPU identity and scheduling.
    """

    def __init__(
        self, robot: str, count: int, episodes: int, seed: int, device: torch.device
    ) -> None:
        from examples.learning.nerd._support._task_reset import sample_task_reset

        generator = torch.Generator(device="cpu").manual_seed(seed)
        q, qd = sample_task_reset(robot, count * episodes, "cpu", generator)
        self.sha256 = _tensor_digest([q, qd])
        self.q = q.reshape(episodes, count, -1).to(device)
        self.qd = qd.reshape(episodes, count, -1).to(device)
        self.cursor = torch.zeros(count, dtype=torch.long, device=device)

    def __call__(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        episode = self.cursor[ids]
        if bool((episode >= self.q.shape[0]).any()):
            raise RuntimeError(
                "Initial-state bank exhausted; recipe budget was exceeded"
            )
        q, qd = self.q[episode, ids], self.qd[episode, ids]
        self.cursor[ids] += 1
        return q, qd


def _validate_physics(model: dict[str, Any], actual: dict[str, Any]) -> None:
    for key in (
        "physics_signature",
        "physics_signature_schema",
        "dt",
        "gravity",
        "joint_names",
        "contact_schema",
        "codec",
    ):
        if key not in model or key not in actual or model[key] != actual[key]:
            raise ValueError(f"Model/analytical physical identity mismatch: {key}")


def _policy_config(cfg: BenchmarkCfg) -> dict[str, Any]:
    actor = {
        "type": "mlp",
        "network_cfg": {
            "hidden_sizes": [64, 64],
            "activation": "tanh",
            "last_activation": "tanh",
            "orthogonal_init": [1.414, 1.414, 0.01],
        },
    }
    critic = {
        "type": "mlp",
        "network_cfg": {
            "hidden_sizes": [64, 64],
            "activation": "tanh",
            "orthogonal_init": [1.414, 1.414, 1.0],
        },
    }
    # Both algorithms have the exact same initialized actor and action distribution.
    return {
        "name": "actor_critic",
        "actor": actor,
        "critic": critic,
        "initial_log_std": cfg.initial_log_std,
        "action_std_range": [float(np.exp(-5.0)), float(np.exp(2.0))],
    }


def _build_training(
    cfg: BenchmarkCfg,
    robot: str,
    algorithm: str,
    seed: int,
    model: Path,
    device: torch.device,
    directory: Path,
) -> tuple[Any, Any, InitialStates]:
    from examples.learning.nerd._support._cartpole_env import make_cartpole_dynamics_env
    from examples.learning.nerd._support._ant_env import make_ant_dynamics_env
    from embodichain.learning.rl.runtime import _build_learning_policy
    from embodichain.learning.rl.algo import build_algo
    from embodichain.learning.rl.differentiable_trainer import (
        DifferentiableTrainer,
        DifferentiableTrainerCfg,
    )
    from embodichain.learning.rl.utils.trainer import Trainer

    manifest = json.loads((model / "manifest.json").read_text())
    bank = InitialStates(
        robot, cfg.num_envs, cfg.updates * cfg.horizon + 4, seed + 1000, device
    )
    factory = (
        make_cartpole_dynamics_env if robot == "cartpole" else make_ant_dynamics_env
    )
    env = factory(
        cfg.num_envs,
        device,
        checkpoint=str(model),
        seed=seed,
        physics_profile=manifest["teacher"]["physics_profile"],
        reset_source=bank,
    )
    try:
        _validate_physics(manifest, env._binding.metadata(env.physics_dt))
        # Scene construction can consume random numbers; policy initialization owns its seed.
        _seed(seed)
        policy = _build_learning_policy(_policy_config(cfg), env, device)
        settings = {
            "optimizer": {"name": "adam", "learning_rate": cfg.learning_rate},
            "gamma": cfg.gamma,
            "max_grad_norm": cfg.max_grad_norm,
            "ent_coef": 0.0,
        }
        if algorithm == "ppo":
            settings.update(
                n_epochs=cfg.ppo_epochs,
                num_mini_batches=cfg.ppo_minibatches,
                gae_lambda=0.95,
                clip_coef=0.2,
                vf_coef=0.5,
            )
        algo = build_algo(algorithm, settings, policy, device)
        if algorithm == "apg":
            trainer = DifferentiableTrainer(
                DifferentiableTrainerCfg(
                    segment_length=cfg.horizon,
                    update_horizon=cfg.horizon,
                    rollout_seed=seed,
                ),
                env,
                policy,
                algo,
            )
        else:
            trainer = Trainer(
                policy,
                env,
                algo,
                cfg.horizon,
                cfg.num_envs * cfg.horizon // cfg.ppo_minibatches,
                None,
                0,
                0,
                str(directory),
                f"{robot}_{algorithm}",
                use_wandb=False,
            )
        return env, trainer, bank
    except BaseException:
        env.close()
        raise


@torch.no_grad()
def _evaluate(
    teacher: Any,
    policy: torch.nn.Module,
    initial_q: torch.Tensor,
    initial_qd: torch.Tensor,
    max_steps: int,
) -> dict[str, Any]:
    """Evaluate every fixed start once, without selecting fast-finishing episodes."""
    from examples.learning.nerd._support._task_values import reference_task_values

    teacher.reset(initial_q, initial_qd)
    count = teacher.num_envs
    active = torch.ones(count, dtype=torch.bool, device=teacher.device)
    returns = torch.zeros(count, dtype=torch.float64, device=teacher.device)
    lengths = torch.zeros(count, dtype=torch.long, device=teacher.device)
    failed = torch.zeros_like(active)
    nonfinite = torch.zeros_like(active)
    quality = torch.zeros_like(returns)
    policy.eval()
    for _ in range(max_steps):
        state, _ = teacher.snapshot()
        obs, _, _ = reference_task_values(teacher.robot, state["q"], state["qd"])
        action = policy.actor(torch.nan_to_num(obs)).clamp(-1, 1)
        action = torch.where(active[:, None], action, torch.zeros_like(action))
        effort = torch.zeros(count, teacher.profile["qd_dim"], device=teacher.device)
        if teacher.robot == "cartpole":
            effort[:, :1] = action * 1500.0
        else:
            effort[:, 6:] = action * 400.0
        teacher.step(effort)
        state, _ = teacher.snapshot()
        obs, reward, done = reference_task_values(
            teacher.robot, state["q"], state["qd"]
        )
        bad = ~torch.isfinite(obs).all(-1) | ~torch.isfinite(reward)
        nonfinite |= active & bad
        returns += torch.where(active & ~bad, reward.double(), 0)
        lengths += active.long()
        failed |= active & done
        metric = obs[:, 1].abs() if teacher.robot == "cartpole" else state["qd"][:, 1]
        quality += torch.where(active & ~bad, metric.double(), 0)
        active &= ~(done | bad)
        if not bool(active.any()):
            break
    episodes = []
    for index in range(count):
        episodes.append(
            {
                "episode": index,
                "return": None if bool(nonfinite[index]) else float(returns[index]),
                "length": int(lengths[index]),
                "failed": bool(failed[index]),
                "nonfinite": bool(nonfinite[index]),
                "quality": float(quality[index] / lengths[index].clamp_min(1)),
            }
        )
    invalid = int(nonfinite.sum())
    return {
        "return_mean": None if invalid else float(returns.mean()),
        "return_std_episodes": (
            None if invalid else float(returns.std(unbiased=count > 1))
        ),
        "length_mean": float(lengths.double().mean()),
        "failure_rate": float(failed.double().mean()),
        "nonfinite_episodes": invalid,
        "quality_mean": float((quality / lengths.clamp_min(1)).mean()),
        "episodes": episodes,
    }


def _checkpoint(
    trainer: Any, path: Path, config: dict[str, Any], identity: dict[str, Any]
) -> None:
    # Policy-only exports are explicit, tensor-safe, and never labeled resumable runs.
    torch.save(
        {
            "schema": "embodichain.nerd.policy-benchmark.v1",
            "policy": trainer.policy.state_dict(),
            "policy_config": config,
            "global_step": trainer.global_step,
            "identity": identity,
        },
        path,
    )


def run_one(
    cfg: BenchmarkCfg,
    robot: str,
    algorithm: str,
    seed: int,
    models: Path,
    output: Path,
    device: torch.device,
) -> Path:
    """Train one fixed-budget seed, then evaluate its saved checkpoints independently."""
    from examples.learning.nerd._support._scene import (
        DynamicsSimulation,
        _runtime_identity,
    )
    from examples.learning.nerd._support._task_reset import sample_task_reset

    directory = output / robot / algorithm / f"seed-{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "result.json"
    if result_path.exists():
        raise FileExistsError(f"Completed run exists: {result_path}")
    manifest = json.loads((models / robot / "manifest.json").read_text())
    root = Path(__file__).resolve().parents[3]
    source_paths = [
        Path(__file__),
        *(root / "examples/learning/nerd/_support").glob("_*.py"),
    ]
    identity = {
        "source_sha256": {
            str(path.relative_to(root)): _digest(path) for path in source_paths
        },
        "robot": robot,
        "algorithm": algorithm,
        "seed": seed,
        "model_weights_sha256": _digest(models / robot / "weights.pt"),
        "model_manifest_sha256": _digest(models / robot / "manifest.json"),
        "physical_signature": manifest["physics_signature"],
        "recipe": cfg.to_dict(),
        "runtime": _runtime_identity(device),
    }
    teacher_cfg = manifest["teacher"]
    # Fail physical-domain drift before spending the training budget.
    setup_start = time.perf_counter()
    teacher = DynamicsSimulation(
        robot,
        device=str(device),
        num_envs=cfg.eval_episodes,
        solver=teacher_cfg["solver"],
        num_substeps=teacher_cfg["num_substeps"],
        physics_profile=teacher_cfg["physics_profile"],
        use_cuda_graph=False,
    )
    try:
        _validate_physics(manifest, teacher.manifest)
        evaluation_identity = teacher.manifest
    finally:
        teacher.close()
    # Warm compilation/optimizers using throwaway policy/environment/RNG, then reconstruct.
    env, trainer, bank = _build_training(
        cfg, robot, algorithm, seed, models / robot, device, directory
    )
    try:
        with redirect_stdout(io.StringIO()):
            trainer.train(cfg.warmup_updates * cfg.num_envs * cfg.horizon)
        _sync(device)
    finally:
        del trainer, bank
        env.close()
        del env
    warmup_seconds = time.perf_counter() - setup_start
    _seed(seed)
    actual_setup_start = time.perf_counter()
    env, trainer, bank = _build_training(
        cfg, robot, algorithm, seed, models / robot, device, directory
    )
    identity["initial_actor_sha256"] = _tensor_digest(
        [*trainer.policy.actor.parameters(), trainer.policy.log_std]
    )
    identity["initial_state_bank_sha256"] = bank.sha256
    identity["evaluation_teacher"] = evaluation_identity
    _json(directory / "identity.json", identity)
    _sync(device)
    actual_setup_seconds = time.perf_counter() - actual_setup_start
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    training_seconds = 0.0
    milestones = [{"global_step": 0, "training_seconds": 0.0}]
    _checkpoint(trainer, directory / "policy-0.pt", _policy_config(cfg), identity)
    try:
        for chunk in range(1, cfg.checkpoints + 1):
            target = (
                chunk * (cfg.updates // cfg.checkpoints) * cfg.num_envs * cfg.horizon
            )
            _sync(device)
            start = time.perf_counter()
            with redirect_stdout(io.StringIO()):
                trainer.train(target)
            _sync(device)
            training_seconds += time.perf_counter() - start
            if not all(
                bool(torch.isfinite(p).all()) for p in trainer.policy.parameters()
            ):
                raise FloatingPointError(
                    "Nonfinite policy parameters; incomplete run retained"
                )
            _checkpoint(
                trainer,
                directory / f"policy-{target}.pt",
                _policy_config(cfg),
                identity,
            )
            milestones.append(
                {"global_step": target, "training_seconds": training_seconds}
            )
            _json(
                directory / "training.json",
                {"milestones": milestones, "history": trainer.train_history},
            )
            print(
                json.dumps(
                    {
                        "robot": robot,
                        "algorithm": algorithm,
                        "seed": seed,
                        "step": target,
                        "training_seconds": round(training_seconds, 2),
                    }
                ),
                flush=True,
            )
        final_policy = trainer.policy
        history = list(trainer.train_history)
        peak_torch_mb = (
            torch.cuda.max_memory_allocated(device) / 2**20
            if device.type == "cuda"
            else None
        )
    finally:
        del trainer, bank
        env.close()
        del env
    teacher = DynamicsSimulation(
        robot,
        device=str(device),
        num_envs=cfg.eval_episodes,
        solver=teacher_cfg["solver"],
        num_substeps=teacher_cfg["num_substeps"],
        physics_profile=teacher_cfg["physics_profile"],
        use_cuda_graph=False,
    )
    try:
        _validate_physics(manifest, teacher.manifest)
        generator = torch.Generator(device="cpu").manual_seed(cfg.eval_seed)
        q, qd = sample_task_reset(robot, cfg.eval_episodes, "cpu", generator)
        identity["evaluation_initial_states_sha256"] = _tensor_digest([q, qd])
        q, qd = q.to(device), qd.to(device)
        evaluations = []
        # Warm analytical kernels outside the measured evaluation windows.
        teacher.reset(q, qd)
        teacher.step(
            torch.zeros(cfg.eval_episodes, teacher.profile["qd_dim"], device=device)
        )
        _sync(device)
        for point in milestones:
            artifact = torch.load(
                directory / f"policy-{point['global_step']}.pt",
                map_location=device,
                weights_only=True,
            )
            final_policy.load_state_dict(artifact["policy"])
            _sync(device)
            start = time.perf_counter()
            metrics = _evaluate(
                teacher, final_policy, q, qd, 500 if robot == "cartpole" else 1000
            )
            _sync(device)
            evaluations.append(
                {**point, **metrics, "evaluation_seconds": time.perf_counter() - start}
            )
            _json(directory / "evaluation.json", evaluations)
    finally:
        teacher.close()
    _json(directory / "identity.json", identity)
    result = {
        "identity": identity,
        "setup_and_warmup_seconds": warmup_seconds,
        "training_setup_seconds": actual_setup_seconds,
        "training_seconds": training_seconds,
        "peak_torch_allocated_mb": peak_torch_mb,
        "evaluation_seconds": sum(p["evaluation_seconds"] for p in evaluations),
        "history": history,
        "evaluations": evaluations,
    }
    _json(result_path, result)
    return result_path


def main() -> None:
    """Run one or more requested jobs or aggregate a completed experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_suffix(".yaml")
    )
    parser.add_argument(
        "--models",
        type=Path,
        help="Directory containing cartpole/ and ant/ model exports",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/benchmarks/nerd-policy")
    )
    parser.add_argument("--robot", choices=("cartpole", "ant"))
    parser.add_argument("--algorithm", choices=("apg", "ppo"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    cfg = BenchmarkCfg(**yaml.safe_load(args.config.read_text()))
    cfg.check()
    if args.report:
        from ._nerd_policy_report import write_report

        write_report(args.output, cfg)
        return
    if args.models is None:
        parser.error("--models is required for training")
    torch.set_num_threads(cfg.cpu_threads)
    wp.init()
    for robot in ([args.robot] if args.robot else cfg.robots):
        for seed in ([args.seed] if args.seed is not None else cfg.seeds):
            # Alternate paired order between seeds to limit order-related timing bias.
            algorithms = (
                list(cfg.algorithms)
                if seed % 2 == 0
                else list(reversed(cfg.algorithms))
            )
            for algorithm in ([args.algorithm] if args.algorithm else algorithms):
                path = run_one(
                    cfg,
                    robot,
                    algorithm,
                    seed,
                    args.models.resolve(),
                    args.output.resolve(),
                    torch.device(args.device),
                )
                print(f"Completed: {path}", flush=True)


if __name__ == "__main__":
    main()
