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

"""Train a G1 motion policy through measured dynamics and shared APG Actor-Critic."""

from __future__ import annotations

import argparse
import fcntl
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any

import torch
import yaml
from embodichain.data import get_data_path
from embodichain.lab.gym.utils.gym_utils import config_to_cfg
from embodichain.learning.dynamics import (
    OnlineModelBasedScheduleCfg,
    OnlineModelBasedSchedule,
)
from embodichain.learning.rl.algo import build_algo
from embodichain.learning.rl.checkpoint import _atomic_write, _capture_rng, _restore_rng
from embodichain.learning.rl.differentiable_trainer import (
    DifferentiableTrainer,
    DifferentiableTrainerCfg,
)
from embodichain.learning.rl.runtime import build_learning_policy_runtime
from embodichain.learning.sl import SupervisedLearner
from embodichain.utils.config_paths import resolve_config_path
from embodichain.utils.utility import load_config
from embodichain_models.dynamics.rwm import (
    KinematicWorldModel,
    KinematicWorldModelCfg,
    TransitionReplay,
    RobotWorldModelObjective,
    replay_supervised_batch,
)
from embodichain_models.dynamics.rwm.model import frozen_world_model
from embodichain_models.dynamics.rwm.robot_state import pack_robot_state
from embodichain_models.policies import motion_tracking  # Register tracking policy.
from embodichain.lab.gym.utils.registration import build_env
from embodichain_tasks.locomotion.tracking import g1_motion  # Register physical task.
from embodichain_tasks.locomotion.tracking.motion_model import replay_motion_starts

from run_metadata import write_run_metadata
from _configuration import normalize_recipe, recipe_identity, validate_recipe


def _optimizer_steps(optimizer: torch.optim.Optimizer) -> dict[str, int]:
    """Read actual Adam step ranges, including restored optimizer history."""
    steps = [
        int(state["step"]) for state in optimizer.state.values() if "step" in state
    ]
    return {
        "minimum": min(steps, default=0),
        "maximum": max(steps, default=0),
        "parameters_with_state": len(steps),
    }


class MotionTraining:
    """Compose G1/SuperMimic training with shared SL and differentiable RL.

    This entry fixes the kinematic world model, objective and replay schema,
    task-derived policy dimensions, and APG Actor-Critic with separate actor and
    critic optimizers. Recipe names are validated before environment allocation.
    """

    def __init__(self, config: dict[str, Any], output: Path) -> None:
        self.invocation_started = time.perf_counter()
        self.invocation_started_at = datetime.now(timezone.utc).isoformat()
        self.sampled_peak_process_gpu_bytes = 0
        self.sampled_peak_device_used_bytes = 0
        self.config, self.output = normalize_recipe(config), output
        validate_recipe(self.config)
        config = self.workflow = self.config["workflow"]
        trainer_block = self.config["trainer"]
        if (
            config["kind"] not in ("test", "convergence")
            or config.get("learning_mode", "online")
            not in ("online", "fixed_world_model")
            or config["wm_batches"] < 0
            or (
                config.get("learning_mode", "online") == "online"
                and config["wm_batches"] == 0
            )
            or (
                config.get("learning_mode") == "fixed_world_model"
                and (config["wm_batches"] or config["warmup_updates"])
            )
            or config["lr_decay_iters"] <= 0
            or config["save_every"] < 0
            or not config["buffer_steps"]
            >= config["prefill_steps"]
            >= max(10, config["wm_horizon"])
        ):
            raise ValueError("Invalid training kind, replay capacity or update budget.")
        torch.manual_seed(config["seed"])
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = config.get("tf32", False)
        torch.backends.cudnn.allow_tf32 = config.get("tf32", False)
        gym = load_config(config["collection_env"])
        self.evaluation_gym_config = load_config(config["evaluation_env"])
        for name, environment in (
            ("collection_env", gym),
            ("evaluation_env", self.evaluation_gym_config),
        ):
            if environment.get("id") != "UnitreeG1MotionRL-v1":
                raise ValueError(
                    f"G1 motion training requires workflow.{name} to select "
                    f"UnitreeG1MotionRL-v1; got {environment.get('id')!r}."
                )
        gym["num_envs"] = config["num_envs"]
        gym["env"]["extensions"]["motion_tracking"]["clips"] = config.get("clips")
        gym["env"]["extensions"]["motion_tracking"].update(
            config.get("task_options", {})
        )
        gym["env"]["extensions"]["motion_tracking"].setdefault(
            "compile_mode", config.get("compile_mode")
        )
        self.gym_config = gym
        evaluation_options = self.evaluation_gym_config["env"]["extensions"][
            "motion_tracking"
        ]
        collection_options = gym["env"]["extensions"]["motion_tracking"]
        if (
            resolve_config_path(config["evaluation_env"]).resolve()
            == resolve_config_path(config["collection_env"]).resolve()
        ):
            evaluation_options.update(deepcopy(collection_options))
        else:
            # Independent evaluation selects its own clips, including all clips
            # when the selection is omitted. Explicit options take precedence.
            for name, value in collection_options.items():
                if name != "clips":
                    evaluation_options.setdefault(name, deepcopy(value))
        cfg = config_to_cfg(gym)
        cfg.episode_context_in_info = config.setdefault(
            "episode_context_in_info", False
        )
        cfg.init_rollout_buffer = False
        self.env = build_env(gym["id"], base_env_cfg=cfg)
        self.env.reset(seed=config["seed"])
        task = self.env.task
        device = self.env.device
        self.model = KinematicWorldModel(
            Path(cfg.robot.fpath).read_text(),
            KinematicWorldModelCfg(
                joint_names=self.env.motion_joint_names,
                resting_position=tuple(task.resting_position.cpu().tolist()),
                action_dim=task.dof,
                context_dim=task.dof + 4,
                hidden_dims=tuple(config["wm_hidden"]),
                amp_bfloat16=config["amp"],
            ),
        ).to(device)
        weights = yaml.safe_load(
            Path(__file__).with_name("world_model_weights.yaml").read_text()
        )
        w = torch.tensor(
            [weights[name] for name in self.env.motion_joint_names], device=device
        ).clamp(0.25, 4)
        w = w / w.mean()
        weights = tuple(torch.cat((w.new_ones(12), w, w)).cpu().tolist())
        self.learner = SupervisedLearner(
            self.model,
            RobotWorldModelObjective(
                None,
                temporal_reduction="sum",
                error_weights=weights,
                compile_mode=config.get("compile_mode"),
            ),
            torch.optim.Adam(
                self.model.parameters(), lr=config["lr"], fused=device.type == "cuda"
            ),
            max_grad_norm=1.0,
        )
        self.replay = TransitionReplay(
            config["buffer_steps"],
            self.env.num_envs,
            self.model.state_dim,
            task.dof,
            context_dim=task.dof + 4,
            device=device,
            seed=config["seed"] + 1,
            extra_fields={
                "frame": (None, torch.int64),
                "previous_action": (task.dof, torch.float32),
                "policy_action": (task.dof, torch.float32),
                "usable": (None, torch.bool),
            },
        )
        policy_block = deepcopy(self.config["policy"])
        policy_dimensions = {
            "history_dim": task.observation_dim * task.cfg.history_steps,
            "goal_dim": task.goal_dim,
            "critic_dim": self.model.feature_dim + task.goal_dim + task.dof + 5,
            "action_dim": task.dof,
        }
        for name, value in policy_dimensions.items():
            configured = policy_block.setdefault("cfg", {}).setdefault(name, value)
            if configured != value:
                raise ValueError(
                    f"Policy {name} differs from the motion task: {configured} != {value}"
                )
        runtime_config = {"trainer": deepcopy(trainer_block), "policy": policy_block}
        runtime_config["trainer"]["learning_env"].setdefault("cfg", {}).setdefault(
            "compile_mode", config.get("compile_mode")
        )
        learning_batch = int(trainer_block["num_envs"])
        self.learning_runtime = build_learning_policy_runtime(
            runtime_config,
            num_envs=learning_batch,
            device=device,
            resources={
                "model": self.model,
                "motion": self.env.motion,
                "task": task,
                "start_provider": lambda: replay_motion_starts(
                    self.replay, learning_batch, task
                ),
            },
        )
        self.imagined, self.policy = (
            self.learning_runtime.env,
            self.learning_runtime.policy,
        )
        if config.get("compile_mode") is not None:
            self.policy.actor.compile(mode=config["compile_mode"], fullgraph=True)
        algorithm_block = self.config["algorithm"]
        algorithm_cfg = deepcopy(algorithm_block.get("cfg", {}))
        for name in ("optimizer", "critic_optimizer"):
            kwargs = algorithm_cfg.get(name, {}).get("kwargs", {})
            if "betas" in kwargs:
                kwargs["betas"] = tuple(kwargs["betas"])
        if device.type != "cuda":
            for name in ("optimizer", "critic_optimizer"):
                algorithm_cfg.get(name, {}).get("kwargs", {}).pop("fused", None)
        self.algorithm = build_algo(
            algorithm_block["name"], algorithm_cfg, self.policy, device
        )
        diff_cfg = {
            key: value
            for key, value in trainer_block.items()
            if key not in ("learning_env", "num_envs")
        }
        self.trainer = DifferentiableTrainer(
            DifferentiableTrainerCfg(**diff_cfg),
            self.imagined,
            self.policy,
            self.algorithm,
        )
        self._optimizer_base_lrs = [
            [group["lr"] for group in optimizer.param_groups]
            for optimizer in (
                self.learner.optimizer,
                self.algorithm.optimizer,
                self.algorithm.critic_optimizer,
            )
        ]
        self.schedule = OnlineModelBasedSchedule(
            OnlineModelBasedScheduleCfg(
                warmup_updates=config["warmup_updates"],
                updates_per_collect=config["utd"],
                model_updates_per_cycle=config["wm_batches"],
            )
        )
        self.collection_steps = 0
        self.first_call_seconds: dict[str, float] = {}
        self.phase_times = dict.fromkeys(
            (
                "collect",
                "wm",
                "apg",
                "policy_rollout_actor",
                "critic",
                "checkpoint",
                "evaluation",
            ),
            0.0,
        )
        self.construction_seconds = time.perf_counter() - self.invocation_started
        self.restore_seconds = 0.0
        self.resume_checkpoint: Path | None = None
        self.environment_metadata: str | None = None
        self.invalid_transitions = 0
        self.latest_collection = {}
        self.output.mkdir(parents=True, exist_ok=True)
        self._stream = (output / "metrics.jsonl").open("a")
        identity_gym = deepcopy(gym)
        identity_gym["env"]["extensions"]["motion_tracking"].pop("compile_mode", None)
        identity_task = self.env.motion_cfg.to_dict()
        identity_task.pop("compile_mode", None)
        self.identity = {
            "position_target_limits": "unclamped_pd",
            "gym": identity_gym,
            "task": identity_task,
            "motion": {
                "names": self.env.motion.names,
                "lengths": self.env.motion.lengths.cpu().tolist(),
                "fps": self.env.motion.fps,
                "content_digest": self.env.motion.content_digest,
            },
            "training": recipe_identity(self.config),
            "wm": self.model.cfg.to_dict(),
            "policy": self.policy.cfg.to_dict(),
        }

    @torch.no_grad()
    def collect(
        self, *, update_normalization: bool = True, reference_control: bool = False
    ) -> int:
        """Collect measured transitions using the policy or reference joint targets.

        Args:
            update_normalization: Refresh actor statistics from measured observations.
            reference_control: Use reference joint targets for model diagnostics.

        Returns:
            Number of measured environment transitions.
        """
        start = time.perf_counter()
        # Physical data are copied into replay before the next learning phase.
        if self.workflow.get("compile_mode") is not None:
            torch.compiler.cudagraph_mark_step_begin()
        before = self.env.capture_transition_start()
        obs = self.env.get_obs()["policy"]
        if update_normalization:
            self.policy.update_actor_normalization(obs)
        if reference_control:
            task = self.env.task
            reference = self.env.motion.future(
                self.env.motion_frame, task.future_offsets.new_ones(1)
            )["q"][:, 0]
            action = (
                (reference - task.resting_position)
                / task.action_scale
                * (1 - task.reference_weight)
            )
        else:
            action = self.policy.actor(obs)
        noise = torch.randn_like(action) * self.workflow["action_noise"]
        noise[: round(self.env.num_envs * self.workflow["collect_det_frac"])] = 0
        _, reward, terminated, truncated, info = self.env.step(action + noise)
        after = info["transition"]
        state = pack_robot_state(before["state"])
        successor = pack_robot_state(after["state"])
        usable = after["usable"] & torch.isfinite(state).all(-1)
        self.replay.append(
            state,
            after["setpoint_action"],
            successor,
            context=before["context"],
            terminated=terminated,
            truncated=truncated,
            extras={
                "frame": before["frame"],
                "previous_action": before["previous_action"],
                "policy_action": after["action"],
                "usable": usable,
            },
        )
        self.invalid_transitions += int((~usable).sum())
        self.collection_steps += 1
        self.latest_collection = {
            "step_reward": float(reward.mean()),
            "terminated_fraction": float(terminated.float().mean()),
            "invalid_transitions": self.invalid_transitions,
        }
        elapsed = time.perf_counter() - start
        self.first_call_seconds.setdefault("collect", elapsed)
        self.phase_times["collect"] += elapsed
        return self.env.num_envs

    def update_model(self) -> dict[str, float]:
        start = time.perf_counter()
        if self.workflow.get("compile_mode") is not None:
            torch.compiler.cudagraph_mark_step_begin()
        for _ in range(4):
            windows = self.replay.sample_windows(
                self.workflow["wm_batch"] * 4, self.workflow["wm_horizon"]
            )
            valid = windows["valid"].all(1) & windows["usable"].all(1)
            if int(valid.sum()) >= self.workflow["wm_batch"]:
                windows = {
                    k: v[valid][: self.workflow["wm_batch"]] for k, v in windows.items()
                }
                break
        else:
            raise RuntimeError("Not enough complete usable model windows.")
        warming = self.schedule.warmup_updates < self.workflow["warmup_updates"]
        refresh = (
            self.learner.num_updates % self.workflow["norm_update_every"] == 0
            if warming
            else (
                self.schedule.policy_updates % self.workflow["norm_update_every"] == 0
                and (self.learner.num_updates - self.workflow["warmup_updates"])
                % self.workflow["wm_batches"]
                == 0
            )
        )
        if refresh:
            self.model.update_normalization(
                windows["state"].flatten(0, 1),
                windows["action"].flatten(0, 1),
                windows["next_state"].flatten(0, 1),
                windows["context"].flatten(0, 1),
            )
        metrics = self.learner.update(
            replay_supervised_batch(windows, complete_windows=True)
        )
        elapsed = time.perf_counter() - start
        self.first_call_seconds.setdefault("wm", elapsed)
        self.phase_times["wm"] += elapsed
        return metrics

    def update_policy(self) -> dict[str, float]:
        start = time.perf_counter()
        if self.workflow.get("compile_mode") is not None:
            torch.compiler.cudagraph_mark_step_begin()
        with frozen_world_model(self.model):
            metrics = self.trainer.update_once()
        elapsed = time.perf_counter() - start
        self.first_call_seconds.setdefault("apg", elapsed)
        self.phase_times["apg"] += elapsed
        self.phase_times["critic"] += metrics["critic_seconds"]
        self.phase_times["policy_rollout_actor"] += elapsed - metrics["critic_seconds"]
        return metrics

    def save(self, path: Path) -> None:
        """Atomically save learning state; measured replay is refilled on resume."""
        started = time.perf_counter()
        state = {
            "schema": 1,
            "identity": self.identity,
            "config": self.config,
            "gym_config": self.gym_config,
            "evaluation_gym_config": self.evaluation_gym_config,
            "model": self.learner.state_dict(),
            "policy": self.policy.state_dict(),
            "algorithm": self.algorithm.state_dict(),
            "schedule": self.schedule.state_dict(),
            "trainer": {
                "global_step": self.trainer.global_step,
                "num_updates": self.trainer.num_updates,
                "rollout_index": self.trainer._rollout_index,
            },
            "rng": _capture_rng(self.env.device),
            "replay_rng": self.replay.generator.get_state(),
            "imagination_rng": self.imagined._generator.get_state(),
            "sampler": self.env.start_sampler.state_dict(),
            "environment_rng": {
                name: gen.get_state()
                for name, gen in self.env._component_generators.items()
            },
            "collection_steps": self.collection_steps,
            "phase_times": self.phase_times,
            "invalid_transitions": self.invalid_transitions,
        }
        _atomic_write(path, lambda stream: torch.save(state, stream))
        self.phase_times["checkpoint"] += time.perf_counter() - started

    def restore(self, path: Path) -> None:
        """Validate recipe and model state before restoring a saved learning boundary."""
        started = time.perf_counter()
        state = torch.load(path, map_location=self.env.device, weights_only=False)
        saved_identity, current_identity = deepcopy(state["identity"]), deepcopy(
            self.identity
        )
        # Normalize saved recipe fields while preserving the resolved task identity.
        saved_identity["training"] = recipe_identity(state["config"])
        current_identity["training"] = recipe_identity(self.config)
        if state["schema"] != 1 or saved_identity != current_identity:
            raise ValueError("Training checkpoint recipe or task identity differs.")
        self.schedule.validate_state_dict(state["schedule"])
        self.env.start_sampler.validate_state_dict(state["sampler"])
        self.algorithm.validate_state_dict(state["algorithm"])
        candidate = deepcopy(self.policy)
        candidate.load_state_dict(state["policy"])
        del candidate
        candidate = deepcopy(self.learner)
        candidate.load_state_dict(state["model"])
        del candidate
        self.learner.load_state_dict(state["model"])
        self.policy.load_state_dict(state["policy"])
        self.algorithm.load_state_dict(state["algorithm"])
        self.schedule.load_state_dict(state["schedule"])
        self.trainer.global_step = state["trainer"]["global_step"]
        self.trainer.num_updates = state["trainer"]["num_updates"]
        self.trainer._rollout_index = state["trainer"]["rollout_index"]
        self.collection_steps = state["collection_steps"]
        self.invalid_transitions = state["invalid_transitions"]
        self.replay.generator.set_state(state["replay_rng"].cpu())
        self.imagined._generator.set_state(state["imagination_rng"].cpu())
        _restore_rng(state["rng"], self.env.device)
        self.env.start_sampler.load_state_dict(state["sampler"])
        for name, rng in state["environment_rng"].items():
            self.env.get_generator(name).set_state(rng.cpu())
        self.env.reset()
        self.restore_seconds = time.perf_counter() - started
        self.resume_checkpoint = path.resolve()

    def save_policy(self, path: Path) -> None:
        """Keep evaluation weights and task metadata for each measured checkpoint."""
        started = time.perf_counter()
        state = {
            "schema": 1,
            "purpose": "evaluation",
            "identity": self.identity,
            "config": self.config,
            "gym_config": self.gym_config,
            "evaluation_gym_config": self.evaluation_gym_config,
            "policy": self.policy.state_dict(),
            "schedule": self.schedule.state_dict(),
        }
        _atomic_write(path, lambda stream: torch.save(state, stream))
        self.phase_times["checkpoint"] += time.perf_counter() - started

    def resource_usage(self) -> dict[str, int | None]:
        """Sample device-wide usage and this process's Torch/native GPU allocation."""
        free, total = torch.cuda.mem_get_info(self.env.device)
        process_bytes = None
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_gpu_memory",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            values = [
                int(memory.strip()) * 1024**2
                for pid, memory in (line.split(",", 1) for line in output.splitlines())
                if int(pid.strip()) == os.getpid() and memory.strip().isdigit()
            ]
            if values:
                process_bytes = sum(values)
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
        self.sampled_peak_process_gpu_bytes = max(
            self.sampled_peak_process_gpu_bytes, process_bytes or 0
        )
        self.sampled_peak_device_used_bytes = max(
            self.sampled_peak_device_used_bytes, total - free
        )
        return {
            "process_gpu_used_bytes": process_bytes,
            "device_used_bytes": total - free,
            "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * 1024,
        }

    def record(self, phase: str, metrics: dict[str, Any]) -> None:
        log_resources = (
            phase != "online"
            or self.schedule.iterations % self.workflow["log_every"] == 0
        )
        record = {
            "phase": phase,
            "kind": self.workflow["kind"],
            "invocation_started_at": self.invocation_started_at,
            "environment_metadata": self.environment_metadata,
            "iteration": self.schedule.iterations,
            "counters": self.schedule.state_dict(),
            "imagined_transitions": self.trainer.global_step,
            "phase_seconds": dict(self.phase_times),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "invocation_seconds": time.perf_counter() - self.invocation_started,
            **metrics,
        }
        if log_resources:
            record["resources"] = self.resource_usage()
        self._stream.write(json.dumps(record) + "\n")
        self._stream.flush()
        if log_resources:
            print(json.dumps(record), flush=True)

    def evaluate(self, label: str) -> None:
        """Evaluate frozen weights in a separate physical process from clip starts."""
        self.save(self.output / "latest.pt")
        checkpoint = self.output / f"policy-{self.schedule.iterations:06d}.pt"
        self.save_policy(checkpoint)
        started = time.perf_counter()
        evaluation_path = self.output / f"evaluation-{label}.json"
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("evaluate_online.py")),
                str(checkpoint),
                "--output",
                str(evaluation_path),
            ],
            check=True,
        )
        elapsed = time.perf_counter() - started
        self.phase_times["evaluation"] += elapsed
        evaluation = json.loads(evaluation_path.read_text())
        self.record(
            "evaluation",
            {
                "seconds": elapsed,
                "macro_normalized_return": evaluation["macro_normalized_return"],
                "failures": evaluation["failures"],
                "path": str(evaluation_path),
            },
        )

    def run(self) -> None:
        self.environment_metadata = write_run_metadata(
            self.output,
            invocation_started_at=self.invocation_started_at,
            config=self.workflow,
            gym_config=self.gym_config,
            motion=self.identity["motion"],
            motion_path=Path(get_data_path(self.env.motion_cfg.motion_path)),
            device=self.env.device,
            resume_checkpoint=self.resume_checkpoint,
        ).name
        config = self.workflow
        run_start = time.perf_counter()
        self.record(
            "constructed",
            {
                "construction_seconds": self.construction_seconds,
                "restore_seconds": self.restore_seconds,
            },
        )
        while self.replay.size < config["prefill_steps"]:
            self.schedule.collect(self.collect, prefill=True)
        self.record(
            "prefill",
            {**self.latest_collection, "seconds": time.perf_counter() - run_start},
        )
        warmup_started = time.perf_counter()
        while self.schedule.warmup_updates < config["warmup_updates"]:
            values = self.schedule.warmup(self.update_model)
            if (
                self.schedule.warmup_updates % 25 == 0
                or self.schedule.warmup_updates == config["warmup_updates"]
            ):
                self.record("model_warmup", values)
        warmup_seconds = time.perf_counter() - warmup_started
        if self.schedule.iterations == 0 and config["eval_every"]:
            self.evaluate("initial")
        start = time.perf_counter()
        while self.schedule.iterations < config["iterations"]:
            i = self.schedule.iterations
            lr_factor = config["lr_final_frac"] + (
                1 - config["lr_final_frac"]
            ) * 0.5 * (1 + math.cos(math.pi * min(i / config["lr_decay_iters"], 1)))
            for optimizer, base_lrs in zip(
                (
                    self.learner.optimizer,
                    self.algorithm.optimizer,
                    self.algorithm.critic_optimizer,
                ),
                self._optimizer_base_lrs,
            ):
                for group, base_lr in zip(optimizer.param_groups, base_lrs):
                    group["lr"] = base_lr * lr_factor
            metrics = self.schedule.step(
                self.collect, self.update_model, self.update_policy
            )
            self.record(
                "online",
                {
                    **metrics,
                    **self.latest_collection,
                    "elapsed_seconds": time.perf_counter() - start,
                    "phase_seconds": dict(self.phase_times),
                    "allocated_bytes": torch.cuda.memory_allocated(),
                    "reserved_bytes": torch.cuda.memory_reserved(),
                },
            )
            i = self.schedule.iterations
            if config["save_every"] and i % config["save_every"] == 0:
                self.save(self.output / "latest.pt")
                self.save_policy(self.output / f"policy-{i:06d}.pt")
            if config["eval_every"] and i % config["eval_every"] == 0:
                self.evaluate(str(i))
        self.save(self.output / "latest.pt")
        self.save_policy(self.output / f"policy-{self.schedule.iterations:06d}.pt")
        if config.get("evaluate_final", bool(config["eval_every"])) and (
            not config["eval_every"] or self.schedule.iterations % config["eval_every"]
        ):
            self.evaluate(str(self.schedule.iterations))
        result = {
            "kind": config["kind"],
            "invocation_started_at": self.invocation_started_at,
            "environment_metadata": self.environment_metadata,
            "counters": self.schedule.state_dict(),
            "imagined_transitions": self.trainer.global_step,
            "phase_seconds": self.phase_times,
            "optimizer_steps": {
                "world_model": _optimizer_steps(self.learner.optimizer),
                "actor": _optimizer_steps(self.algorithm.optimizer),
                "critic": _optimizer_steps(self.algorithm.critic_optimizer),
            },
            "first_call_seconds": self.first_call_seconds,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * 1024,
            "run_seconds": time.perf_counter() - run_start,
            "warmup_seconds": warmup_seconds,
            "construction_seconds": self.construction_seconds,
            "restore_seconds": self.restore_seconds,
            "resources_at_end": self.resource_usage(),
            "sampled_peak_process_gpu_bytes": self.sampled_peak_process_gpu_bytes
            or None,
            "sampled_peak_device_used_bytes": self.sampled_peak_device_used_bytes,
            "timing_scope": "Current invocation's host elapsed wall time; learner scalar metrics synchronize updates. APG contains policy_rollout_actor and critic. Construction and first calls include lazy runtime/kernel initialization. Group metrics by invocation_started_at to combine resumed runs.",
            "memory_scope": "Torch peaks cover this invocation. Process GPU usage includes native allocations from nvidia-smi; device usage includes other processes. Sampled maxima cover phase/log boundaries, not every allocation.",
            "replay_bytes": self.replay.allocated_bytes,
            "invalid_transitions": self.invalid_transitions,
        }
        (self.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        self.record("completed", result)

    def close(self) -> None:
        self._stream.close()
        self.learning_runtime.close()
        self.env.close(exit_process=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--motion-path", type=Path)
    parser.add_argument("--quaternion-order", choices=("xyzw", "wxyz"))
    parser.add_argument(
        "--wm-checkpoint",
        type=Path,
        help="Initialize WM weights from a fitted or online-training checkpoint.",
    )
    parser.add_argument(
        "--evaluate-final", action=argparse.BooleanOptionalAction, default=None
    )
    args = parser.parse_args()
    config_path = resolve_config_path(args.config).resolve()
    recipe = normalize_recipe(load_config(config_path))
    config = recipe["workflow"]
    for key in ("collection_env", "evaluation_env"):
        config[key] = str(
            resolve_config_path(config[key], base_dir=config_path.parent).resolve()
        )
    if args.evaluate_final is not None:
        config["evaluate_final"] = args.evaluate_final
    if args.iterations is not None:
        config["iterations"] = args.iterations
    if args.seed is not None:
        config["seed"] = args.seed
    if args.motion_path is not None:
        config.setdefault("task_options", {})["motion_path"] = str(
            args.motion_path.resolve()
        )
    if args.quaternion_order is not None:
        config.setdefault("task_options", {})[
            "quaternion_order"
        ] = args.quaternion_order
    if args.wm_checkpoint is not None:
        config["wm_checkpoint"] = str(args.wm_checkpoint.resolve())
    if config.get("learning_mode") == "fixed_world_model" and not (
        args.resume or config.get("wm_checkpoint")
    ):
        parser.error("Fixed-model training requires --wm-checkpoint or --resume.")
    if not args.resume and args.output.exists():
        parser.error("Use a new output directory or --resume.")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error(
                "This output directory already has an active training process."
            )
        workflow = MotionTraining(recipe, args.output)
        try:
            if args.resume:
                workflow.restore(args.resume)
            elif config.get("wm_checkpoint"):
                initial = torch.load(
                    config["wm_checkpoint"],
                    map_location=workflow.env.device,
                    weights_only=True,
                )
                learner = (
                    initial["learner"] if "learner" in initial else initial["model"]
                )
                workflow.model.load_state_dict(learner["model"])
            (args.output / "config.yaml").write_text(
                yaml.safe_dump(recipe, sort_keys=False)
            )
            workflow.run()
        finally:
            workflow.close()


if __name__ == "__main__":
    main()
