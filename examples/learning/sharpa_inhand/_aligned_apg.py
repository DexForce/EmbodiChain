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

"""EmbodiChain APG on the original Sharpa task and SuperMimic world model.

Keep task, HORA teacher, model fitting, replay, Muon, EMA and sampling budgets
fixed. The default ablates the learned terminal value and critic fitting;
``--value-bootstrap`` retains both and routes the original actor objective
through EmbodiChain APG. Task terms remain owned by the pinned upstream runtime.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import sharpa_rl_unilab
import torch
from _aligned_task import _AlignedTask
from omegaconf import OmegaConf
from reference import _source
from sharpa_rl_unilab.algos.supermimic.learner import SharpaSuperMimicLearner
from sharpa_rl_unilab.algos.supermimic.state import unflatten_state
from sharpa_rl_unilab.cli import compose_config
from sharpa_rl_unilab.tasks.sharpa_inhand.protocol import ACTOR_HISTORY
from sharpa_rl_unilab.training import evaluation as evaluator
from sharpa_rl_unilab.training.checkpoints import save_teacher
from sharpa_rl_unilab.training.configuration import configure_threads
from sharpa_rl_unilab.training.logging import (
    append_metrics,
    create_run,
    training_counters,
)
from sharpa_rl_unilab.training.policy import make_actor, make_models
from tensordict import TensorDict

from embodichain.learning.rl.algo.apg import APG, APGCfg
from embodichain.learning.rl.collector import (
    DifferentiableRollout,
    DifferentiableTransition,
)

__all__ = ["main"]


class _APGLearner(SharpaSuperMimicLearner):
    def __init__(
        self, *args: Any, value_bootstrap: bool = False, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.value_bootstrap = value_bootstrap
        # AOTAutograd's donated buffers require one backward per compiled graph.
        # Validate the independent reduction in the eager smoke; preserve the
        # reference compiled training path without retaining a second backward.
        self._verify_actor_gradient = value_bootstrap and not bool(
            self.cfg.algo.compile
        )
        self.apg = APG(
            APGCfg(
                device=str(self.device),
                gamma=float(self.cfg.algo.gamma),
                max_grad_norm=1.0,
            ),
            self.actor,
        )
        # Preserve the reference mixed Muon/Adam optimizer and its parameters.
        self.apg.optimizer = self.opt_pi

    def policy_update(self) -> dict[str, float]:
        start, history = self.buffer.sample_starts(
            int(self.cfg.algo.pi_batch), ACTOR_HISTORY
        )
        state, context = unflatten_state(start["s"]), start["xi"]
        noisy, clean, priv = self._initial_history(history)
        packed = torch.cat((noisy.flatten(-2), priv[:, -1]), -1)
        initial = packed
        survival = torch.ones(packed.shape[0], device=self.device)
        transitions = []
        features = [self._critic_features(clean, priv)]
        rewards, soft_dones = [], []
        flags = torch.zeros_like(survival, dtype=torch.bool)
        for module in (self.wm, self.critic):
            module.requires_grad_(False)
        try:
            for _ in range(int(self.cfg.algo.horizon)):
                before = packed
                state, packed, noisy, clean, priv, feature, reward, done = (
                    self._imagination_step_fn(
                        state, packed, context, noisy, clean, priv
                    )
                )
                # Original differentiable soft survival, including its gradient.
                # No imagined autoreset or hard threshold replaces this owner.
                transitions.append(
                    DifferentiableTransition(
                        observation=before,
                        policy_output=TensorDict({}, batch_size=[packed.shape[0]]),
                        reward=survival * reward,
                        terminated=flags,
                        truncated=flags,
                        next_observation=packed,
                        info={},
                    )
                )
                survival = survival * (1 - done)
                if self.value_bootstrap:
                    features.append(feature)
                    rewards.append(reward.detach())
                    soft_dones.append(done.detach())
            rollout = DifferentiableRollout(initial, tuple(transitions))
            terminal_value = None
            expected_gradients = None
            if self.value_bootstrap:
                terminal_value = survival * self.critic(
                    TensorDict({"critic": features[-1]}, batch_size=[packed.shape[0]])
                ).squeeze(-1)
                if self._verify_actor_gradient:
                    # Independent reduction of the pinned upstream objective:
                    # sum gamma^t * survival_t * reward_t + gamma^H * survival_H * V_H.
                    gamma = float(self.cfg.algo.gamma)
                    objective = torch.zeros_like(survival)
                    for index, transition in enumerate(transitions):
                        objective = objective + gamma**index * transition.reward
                    objective = objective + gamma ** len(transitions) * terminal_value
                    expected_gradients = torch.autograd.grad(
                        -objective.mean(),
                        tuple(self.actor.parameters()),
                        retain_graph=True,
                        allow_unused=True,
                    )
            self.apg.begin_update()
            self.apg.accumulate_complete_rollout(rollout, terminal_value=terminal_value)
            gradient_error = 0.0
            gradient_relative_error = 0.0
            if expected_gradients is not None:
                error_squared = torch.zeros((), device=self.device)
                expected_squared = torch.zeros_like(error_squared)
                for parameter, expected in zip(
                    self.actor.parameters(), expected_gradients, strict=True
                ):
                    if expected is None:
                        assert parameter.grad is None
                    else:
                        difference = parameter.grad - expected
                        gradient_error = max(
                            gradient_error, float(difference.abs().max())
                        )
                        if not bool(self.cfg.algo.use_amp):
                            torch.testing.assert_close(
                                parameter.grad, expected, rtol=1e-5, atol=1e-6
                            )
                        error_squared += difference.float().square().sum()
                        expected_squared += expected.float().square().sum()
                gradient_relative_error = float(
                    (error_squared / expected_squared.clamp_min(1e-30)).sqrt()
                )
                # Equivalent return reductions can traverse BF16 backward
                # accumulation in different orders. Bound the whole-gradient
                # relative L2 error below BF16 epsilon; retain FP32 checks above.
                if bool(self.cfg.algo.use_amp) and (
                    not np.isfinite(gradient_relative_error)
                    or gradient_relative_error > 1e-3
                ):
                    raise AssertionError(
                        f"AMP actor gradient relative error {gradient_relative_error}"
                    )
                self._verify_actor_gradient = False
            # Match both original gradient clipping stages. APG's subsequent
            # norm bound is idempotent after these stricter bounds.
            norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            torch.nn.utils.clip_grad_value_(self.actor.parameters(), 0.1)
            metrics = self.apg.finish_update()
            if self.actor_avg is not None and not metrics["skipped_update"]:
                self.actor_avg.update_parameters(self.actor)
            metrics = {
                **{f"apg/{key}": value for key, value in metrics.items()},
                "loss/actor_grad_norm": float(norm),
                "imagination/survival": float(survival.detach().mean()),
                "parity/actor_gradient_checked": float(expected_gradients is not None),
            }
            if expected_gradients is not None:
                metrics["parity/actor_gradient_max_error"] = gradient_error
                metrics["parity/actor_gradient_relative_l2_error"] = (
                    gradient_relative_error
                )
        except BaseException:
            self.apg.cancel_update()
            raise
        finally:
            for module in (self.wm, self.critic):
                module.requires_grad_(True)
        if self.value_bootstrap:
            metrics["loss/critic"] = self._fit_values(features, rewards, soft_dones)
        return metrics

    def _fit_values(
        self,
        features: list[torch.Tensor],
        rewards: list[torch.Tensor],
        soft_dones: list[torch.Tensor],
    ) -> float:
        """Fit the original imagined lambda-return value targets."""
        inputs = torch.stack(features).detach()
        self.critic.obs_normalizer.update(inputs.flatten(0, 1))
        with torch.no_grad():
            values = self.critic(
                TensorDict(
                    {"critic": inputs.flatten(0, 1)},
                    batch_size=[inputs.shape[0] * inputs.shape[1]],
                )
            ).view(*inputs.shape[:2])
            target = values[-1]
            targets = []
            gamma, lam = float(self.cfg.algo.gamma), float(self.cfg.algo.lam)
            for index in reversed(range(len(rewards))):
                target = rewards[index] + gamma * (1 - soft_dones[index]) * (
                    (1 - lam) * values[index + 1] + lam * target
                )
                targets.append(target)
            targets = torch.stack(targets[::-1])
        flat_inputs, flat_targets = inputs[:-1].flatten(0, 1), targets.flatten()
        count = len(flat_inputs)
        minibatches = int(self.cfg.algo.critic_minibatches)
        size = count // minibatches
        if size == 0:
            raise ValueError("critic_minibatches exceeds imagined transition count")
        for _ in range(int(self.cfg.algo.critic_epochs)):
            order = torch.randperm(count, device=self.device)
            for index in range(minibatches):
                rows = order[index * size : (index + 1) * size]
                prediction = self.critic(
                    TensorDict({"critic": flat_inputs[rows]}, batch_size=[len(rows)])
                ).squeeze(-1)
                loss = (prediction - flat_targets[rows]).square().mean()
                self._optimize(loss, self.opt_v, self.critic)
        return float(loss.detach())


def _load_apg(path: str, device: str | None = "cpu") -> tuple[Any, Any, dict[str, Any]]:
    snapshot = torch.load(path, map_location="cpu", weights_only=False)
    if snapshot["algorithm"] != "embodichain_apg" or snapshot["stage"] != "teacher":
        raise ValueError("Expected an aligned EmbodiChain APG teacher checkpoint")
    cfg = OmegaConf.create(snapshot["config"])
    resolved = device or cfg.training.device
    configure_threads(cfg)
    actor = make_actor(cfg, resolved, stage="teacher")
    actor.load_state_dict(snapshot["actor"], strict=True)
    return actor.eval(), cfg, snapshot


def main() -> None:
    """Run the explicitly labelled APG ablation after reference qualification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", required=True, type=Path)
    parser.add_argument("--parity-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--value-bootstrap", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--release-unused-cuda-cache", action="store_true")
    parser.add_argument("--num-envs", type=int, default=2048)
    args = parser.parse_args()
    reference = args.reference_run.resolve()
    status = json.loads((reference / "status.json").read_text())
    parity = json.loads(args.parity_report.read_text())
    if status["phase"] != "reference_evaluated" or parity["backend"] != "mjwarp":
        raise ValueError("Finish the original reference and MJWarp task parity first")
    if any(parity["max_absolute_errors"].values()):
        raise ValueError("Task/collection parity has unresolved differences")
    required = {"obs/obs", "target", "reward_terms", "physical/s", "physical/xi"}
    if (
        not required.issubset(parity["max_absolute_errors"])
        or parity["steps"] != 401
        or parity["num_envs"] != 8
    ):
        raise ValueError("Expected a complete eight-scale, 401-step task qualification")
    if status["source"]["revision"] != "9816885a3b1fbdb947db71e72c27ed75effcbf9f":
        raise ValueError("Reference revision differs from the qualified baseline")
    root = Path(sharpa_rl_unilab.__file__).resolve().parents[2]
    if _source(root) != status["source"]:
        raise ValueError("Installed task sources differ from the reproduced reference")
    overrides = [
        f"training.log_dir={args.output.resolve()}",
        "training.logger=no_print",
        "training.no_play=true",
        f"algo.seed={args.seed}",
        f"training.num_envs={args.num_envs}",
    ]
    if args.smoke:
        overrides += [
            "training.num_envs=8",
            "algo.max_iterations=2",
            "algo.warmup_steps=4",
            "algo.wm_window=2",
            "algo.wm_batches=1",
            "algo.wm_batch=4",
            "algo.wm_hidden=[64,64]",
            "algo.pi_batch=4",
            "algo.horizon=2",
            "algo.wm_only_iters=0",
            "algo.compile=false",
        ]
    cfg = compose_config("supermimic", "mjwarp", overrides)
    baseline_cfg = OmegaConf.load(reference / "effective_config.yaml")
    baseline_cfg.algo.seed = (
        args.seed
    )  # Independent training replicate, unchanged budget.
    reference_envs = int(baseline_cfg.training.num_envs)
    collection_multiplier = 1
    if not args.smoke:
        if args.num_envs <= 0 or reference_envs % args.num_envs:
            raise ValueError(
                "num_envs must be a positive divisor of reference parallelism"
            )
        collection_multiplier = reference_envs // args.num_envs
        cfg.algo.warmup_steps = (
            int(baseline_cfg.algo.warmup_steps) * collection_multiplier
        )
        cfg.algo.buffer_steps = (
            int(baseline_cfg.algo.buffer_steps) * collection_multiplier
        )
        # Match total warmup/replay transitions and transitions per update;
        # only physical parallelism changes under a shared-GPU memory limit.
        baseline_cfg.training.num_envs = args.num_envs
        baseline_cfg.algo.warmup_steps = cfg.algo.warmup_steps
        baseline_cfg.algo.buffer_steps = cfg.algo.buffer_steps
    for key in ("env", "reward", "model", "protocol"):
        if OmegaConf.to_container(cfg[key], resolve=True) != OmegaConf.to_container(
            baseline_cfg[key], resolve=True
        ):
            raise ValueError(f"Task or policy configuration differs in {key}")
    if not args.smoke and OmegaConf.to_container(
        cfg.algo, resolve=True
    ) != OmegaConf.to_container(baseline_cfg.algo, resolve=True):
        raise ValueError("Learning/replay budgets differ from the reference")
    cfg.algo.algo = "embodichain_apg"
    configure_threads(cfg)
    torch.manual_seed(int(cfg.algo.seed))
    np.random.seed(int(cfg.algo.seed))
    run = create_run(cfg)
    specification = {
        "algorithm": "embodichain_apg",
        "stage": "teacher",
        "smoke": args.smoke,
        "value_bootstrap": args.value_bootstrap,
        "training_seed": args.seed,
        "release_unused_cuda_cache": args.release_unused_cuda_cache,
        "physical_num_envs": int(cfg.training.num_envs),
        "reference_physical_num_envs": reference_envs,
        "collection_steps_multiplier": collection_multiplier,
        "replay_capacity_transitions": int(cfg.algo.buffer_steps)
        * int(cfg.training.num_envs),
        "reference_run": str(reference),
        "asset_cache": os.environ.get("SHARPA_RL_UNILAB_ASSET_CACHE"),
        "task_model_protocol_config_matches_reference": True,
        "parity_report": str(args.parity_report.resolve()),
        "kept": [
            "MJCF/MJWarp task",
            "HORA teacher",
            "replay and collection",
            "world model and fitting",
            "Muon/Adam",
            "EMA",
            "gradient clipping",
            "soft survival",
            "discount",
            "horizon",
        ],
        "changed": (
            [
                "APG core integration of the same terminal value and critic lambda-return objective"
            ]
            if args.value_bootstrap
            else [
                "APG finite-horizon return without terminal value bootstrap",
                "no critic fitting or lambda-return targets",
            ]
        ),
        "evaluation_loader": "APG checkpoint loader; original evaluator and scene manifest",
    }
    (run / "comparison_spec.json").write_text(
        json.dumps(specification, indent=2) + "\n"
    )
    started = time.monotonic()
    counters = training_counters()
    with ExitStack() as resources:
        # Match the reference construction/RNG order exactly.
        reference_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        reference_cfg.algo.algo = "supermimic"
        with torch.random.fork_rng(devices=[0]):
            expected_actor, expected_critic, _ = make_models(
                reference_cfg, cfg.training.device
            )
        actor, critic, _ = make_models(cfg, cfg.training.device)
        torch.testing.assert_close(
            actor.state_dict(), expected_actor.state_dict(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            critic.state_dict(), expected_critic.state_dict(), rtol=0, atol=0
        )
        del expected_actor, expected_critic
        initial_actor = {
            key: value.detach().cpu().clone()
            for key, value in actor.state_dict().items()
        }
        torch.save(initial_actor, run / "initial_actor.pt")
        task = _AlignedTask(cfg)
        resources.callback(task.close)
        learner = _APGLearner(
            task,
            cfg,
            actor,
            critic,
            task.mdp,
            task.capture,
            value_bootstrap=args.value_bootstrap,
        )
        if args.release_unused_cuda_cache:
            torch.cuda.empty_cache()
        (run / "initial_memory.json").write_text(
            json.dumps(
                {
                    "torch_allocated_mb": torch.cuda.memory_allocated() / 2**20,
                    "torch_reserved_mb": torch.cuda.memory_reserved() / 2**20,
                    "global_free_mb": torch.cuda.mem_get_info()[0] / 2**20,
                },
                indent=2,
            )
            + "\n"
        )
        learner.collect(max(int(cfg.algo.warmup_steps), int(cfg.algo.wm_window)))
        accumulator = 0.0
        for iteration in range(int(cfg.algo.max_iterations)):
            accumulator += collection_multiplier / float(cfg.algo.utd)
            count = int(accumulator)
            accumulator -= count
            resource_metrics = {}
            if count and args.release_unused_cuda_cache:
                # Torch and Warp own separate allocators. Release only unused
                # Torch blocks before native simulation needs graph workspace.
                before = torch.cuda.memory_reserved()
                torch.cuda.empty_cache()
                resource_metrics = {
                    "memory/torch_reserved_before_release_mb": before / 2**20,
                    "memory/torch_reserved_after_release_mb": torch.cuda.memory_reserved()
                    / 2**20,
                    "memory/torch_allocated_mb": torch.cuda.memory_allocated() / 2**20,
                    "memory/global_free_mb": torch.cuda.mem_get_info()[0] / 2**20,
                }
            metrics = learner.collect(count)
            metrics.update(resource_metrics)
            for _ in range(int(cfg.algo.wm_batches)):
                loss, error = learner.wm_update()
            metrics.update(
                {"loss/world_model": float(loss), "loss/world_model_last": float(error)}
            )
            counters["training_samples"] += (
                int(cfg.algo.wm_batches)
                * int(cfg.algo.wm_batch)
                * int(cfg.algo.wm_window)
            )
            counters["optimizer_updates"] += int(cfg.algo.wm_batches)
            if iteration >= int(cfg.algo.wm_only_iters):
                metrics.update(learner.policy_update())
                counters["actor_updates"] += 1
                critic_updates = (
                    int(cfg.algo.critic_epochs) * int(cfg.algo.critic_minibatches)
                    if args.value_bootstrap
                    else 0
                )
                counters["critic_updates"] += critic_updates
                counters["optimizer_updates"] += 1 + critic_updates
                counters["training_samples"] += (
                    int(cfg.algo.pi_batch)
                    * int(cfg.algo.horizon)
                    * (1 + critic_updates)
                )
            counters.update(
                collected=learner.collected,
                received=learner.received,
                policy_version=iteration + 1,
            )
            metrics.update(counters)
            metrics.update(learner.episodes.metrics())
            metrics["wall_seconds"] = time.monotonic() - started
            append_metrics(run, metrics)
        path = run / "apg_teacher_final.pt"
        snapshot = save_teacher(
            path,
            cfg,
            learner.checkpoint_actor,
            critic,
            learner,
            counters,
            time.monotonic() - started,
        )
        snapshot.update(
            world_model=learner.wm.state_dict(),
            world_model_optimizer=learner.opt_wm.state_dict(),
            apg=learner.apg.state_dict(),
            critic_optimizer=learner.opt_v.state_dict(),
            comparison=specification,
        )
        torch.save(snapshot, path)
        parameter_change = (
            sum(
                float(
                    (value.detach().cpu() - initial_actor[key]).float().square().sum()
                )
                for key, value in actor.named_parameters()
            )
            ** 0.5
        )
        qualification = {
            "initial_actor_matches_reference_exactly": True,
            "initial_critic_matches_reference_exactly": True,
            "policy_parameter_change_l2": parameter_change,
            "actor_updates": counters["actor_updates"],
            "critic_updates": counters["critic_updates"],
            "measured_transitions": counters["received"],
        }
        if not np.isfinite(parameter_change) or parameter_change == 0:
            raise RuntimeError("APG did not produce a finite nonzero actor update")
        (run / "qualification.json").write_text(
            json.dumps(qualification, indent=2) + "\n"
        )
    # Extend only checkpoint deserialization. Keep episode RNG, task, inference,
    # scoring and all 240 scene records owned by the original evaluator.
    evaluation = {"manifest": str(reference / "scenes.json")}
    if args.smoke:
        print(f"Smoke complete: {path}")
    else:
        with patch.object(evaluator, "load_policy", _load_apg):
            result = evaluator.evaluate_checkpoint(
                path,
                output=run / "evaluation.json",
                device="cuda:0",
                evaluation=evaluation,
            )
        print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
