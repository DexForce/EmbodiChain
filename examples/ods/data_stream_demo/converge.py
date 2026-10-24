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

"""Train the dual-camera PickPlace policy with explicit learning acceptance.

Review the resolved recipe first; preview.py acquires expert images without
training. The train command reuses the existing bounded ODS visual pipeline.
Development evaluation selects a checkpoint before a single held-out test.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any

from embodichain.utils import configclass

__all__ = ["ConvergenceCfg", "load_recipe", "review_recipe"]

_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT = Path(__file__).with_name("configs") / "pick_place_dual_convergence.yaml"


@configclass
class ConvergenceCfg:
    """Fixed useful training work, protected supply and evaluation thresholds."""

    updates: int = 20000
    batch_size: int = 64
    buffer_size: int = 32
    producer_envs: int = 4
    episode_budget: int = 2048
    learning_rate: float = 3e-4
    final_learning_rate: float = 3e-5
    warmup_updates: int = 500
    prediction_horizon: int = 32
    action_horizon: int = 8
    encoder_threads: int = 2
    evaluate_every: int = 2000
    development_scenes: int = 20
    development_success_rate: float = 0.9
    test_scenes: int = 100
    test_success_rate: float = 0.8
    training_seed: int = 20261025
    development_seed: int = 20271025
    test_seed: int = 20281025
    run_timeout: int = 21600
    evaluation_timeout: int = 3600

    def __post_init__(self) -> None:
        positive = (
            "updates",
            "batch_size",
            "buffer_size",
            "producer_envs",
            "episode_budget",
            "warmup_updates",
            "prediction_horizon",
            "action_horizon",
            "encoder_threads",
            "evaluate_every",
            "development_scenes",
            "test_scenes",
            "run_timeout",
            "evaluation_timeout",
        )
        for name in positive:
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            self.buffer_size % self.producer_envs
            or self.batch_size % self.buffer_size
            or self.producer_envs * self.episode_budget < self.batch_size
        ):
            raise ValueError("Require C divisible by P, B divisible by C, and P*K>=B")
        if (
            self.updates % self.evaluate_every
            or self.updates < 2 * self.evaluate_every
            or self.warmup_updates >= self.updates
            or self.action_horizon > self.prediction_horizon
        ):
            raise ValueError(
                "Budget must fund two evaluation intervals and fit warmup/horizon"
            )
        for name in (
            "learning_rate",
            "final_learning_rate",
            "development_success_rate",
            "test_success_rate",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number")
        if not 0 < self.final_learning_rate <= self.learning_rate:
            raise ValueError("Learning rates must be finite and 0<final<=peak")
        for name in ("development_success_rate", "test_success_rate"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0,1]")
        seeds = [self.training_seed, self.development_seed, self.test_seed]
        if (
            any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(set(seeds)) != 3
        ):
            raise ValueError(
                "Training, development and final test seeds must be distinct"
            )

    def rate(self, step: int) -> float:
        """Return the declared linear-warmup/cosine learning rate for an update."""
        if step <= self.warmup_updates:
            return self.learning_rate * max(1, step) / self.warmup_updates
        progress = min(
            1.0, (step - self.warmup_updates) / (self.updates - self.warmup_updates)
        )
        return (
            self.final_learning_rate
            + (self.learning_rate - self.final_learning_rate)
            * (1 + math.cos(math.pi * progress))
            / 2
        )


def load_recipe(path: Path) -> ConvergenceCfg:
    """Load an explicit YAML recipe, rejecting unknown or invalid fields."""
    import yaml

    path = Path(path)
    text = path.read_text()
    values = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
    if not isinstance(values, dict):
        raise ValueError("Convergence YAML must be a mapping")
    cfg = ConvergenceCfg(**values)
    cfg.validate()
    return cfg


def review_recipe(cfg: ConvergenceCfg) -> dict[str, Any]:
    """Describe the complete candidate without creating an engine or optimizer."""
    from .acquisition import camera_profile_version, camera_configs

    cameras = camera_configs("dual", camera_profile="pick_place")
    return {
        "recipe": asdict(cfg),
        "task": "SimpleBCPickPlace-Franka-v1",
        "model": "scratch ResNet10/GroupNorm + spatial embeddings + MLP512/512",
        "policy_input": "two causal RGB frames per camera + two 27D robot states",
        "camera_profile": camera_profile_version("pick_place"),
        "cameras": cameras,
        "history": 2,
        "precision": "FP32",
        "optimizer": "AdamW",
        "weight_decay": 1e-6,
        "gradient_norm_limit": 1.0,
        "normalization": "initial training cache only; frozen",
        "trained_chunks": cfg.updates * cfg.batch_size,
        "valid_action_targets": cfg.updates * cfg.batch_size * cfg.prediction_horizon,
        "minimum_trained_sources_from_cap": math.ceil(
            cfg.updates * cfg.batch_size / cfg.episode_budget
        ),
        "reuse_profile_qualified": False,
        "development_gate": "last two updated checkpoints both meet development threshold",
        "selection": "highest dev success, then lowest mean goal error, before final test",
        "final_test": "one fixed panel, opened only after development gate passes",
        "training_started": False,
    }


def _worker(output: Path, cfg: ConvergenceCfg) -> int:
    from .train import _Experiment, _validate, build_parser
    from torch.utils.tensorboard import SummaryWriter

    args = build_parser().parse_args(
        [
            "--output",
            str(output),
            "--arm",
            "ods_overlap",
            "--vision",
            "dual",
            "--updates",
            str(cfg.updates),
            "--batch-size",
            str(cfg.batch_size),
            "--buffer-size",
            str(cfg.buffer_size),
            "--producer-envs",
            str(cfg.producer_envs),
            "--offline-episodes",
            str(cfg.buffer_size),
            "--episode-budget",
            str(cfg.episode_budget),
            "--prediction-horizon",
            str(cfg.prediction_horizon),
            "--action-horizon",
            str(cfg.action_horizon),
            "--encoder-threads",
            str(cfg.encoder_threads),
            "--seed",
            str(cfg.training_seed),
            "--run-timeout",
            str(cfg.run_timeout),
            "--evaluation-scenes",
            "0",
            "--phase",
            "convergence",
        ]
    )
    args.camera_profile = "pick_place"
    from .acquisition import camera_profile_version

    args.camera_profile_version = camera_profile_version(args.camera_profile)
    args.convergence_recipe = asdict(cfg)
    _validate(args)

    class LearningExperiment(_Experiment):
        def __init__(self) -> None:
            super().__init__(args)
            self.development: list[dict[str, Any]] = []
            self.board = SummaryWriter(str(output / "tensorboard"))

        def save_checkpoint(self, name: str, step: int) -> None:
            if (self.output / name).exists():
                raise FileExistsError("Learning checkpoints are immutable")
            if step == 0:
                for group in self.optimizer.param_groups:
                    group["lr"] = cfg.rate(0)
            super().save_checkpoint(name, step)

        def evaluate(
            self, checkpoint: Path, scenes: int, seed: int, label: str
        ) -> dict[str, Any]:
            receipt = self.output / "evaluation" / f"{label}.json"
            receipt.parent.mkdir(exist_ok=True)
            command = [
                sys.executable,
                "-m",
                "examples.ods.data_stream_demo.training_eval",
                "--checkpoint",
                str(checkpoint),
                "--vision",
                "dual",
                "--scenes",
                str(scenes),
                "--seed",
                str(seed),
                "--output",
                str(receipt),
            ]
            started = self.now()
            with receipt.with_suffix(".log").open("w") as log:
                child = subprocess.Popen(
                    command, cwd=_ROOT, stdout=log, stderr=subprocess.STDOUT
                )
                try:
                    deadline = time.monotonic() + cfg.evaluation_timeout
                    while child.poll() is None:
                        if self.flow is None:
                            self.poll()
                        else:
                            self.flow.poll()
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                f"{label} evaluation exceeded its deadline"
                            )
                        time.sleep(0.1)
                    if child.returncode:
                        raise RuntimeError(
                            f"{label} evaluator failed; see {receipt.with_suffix('.log')}"
                        )
                finally:
                    if child.poll() is None:
                        child.terminate()
                        try:
                            child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait(timeout=10)
            report = json.loads(receipt.read_text())
            if (
                not report.get("passed")
                or not report.get("closed")
                or report.get("expert_initialized")
            ):
                raise RuntimeError(
                    "Independent evaluator failed its student-only/closure contract"
                )
            if self.flow is not None:
                self.metrics.interval(
                    "learning_development_evaluation", started, self.now()
                )
            return report

        def develop(self, checkpoint: Path, step: int) -> None:
            report = self.evaluate(
                checkpoint,
                cfg.development_scenes,
                cfg.development_seed,
                f"dev-{step:08d}",
            )
            row = {"step": step, "checkpoint": str(checkpoint), **report}
            self.development.append(row)
            (output / "development.json").write_text(
                json.dumps(self.development, indent=2)
            )
            for name in ("success_rate", "lift_rate", "goal_error", "episode_length"):
                self.board.add_scalar("development/" + name, report[name], step)
            self.board.flush()

        def bootstrap(self, episodes: list[Any]) -> None:
            super().bootstrap(episodes)
            self.develop(output / "initial.pt", 0)

        def update(self, *positional: Any, **keywords: Any) -> None:
            step = self.completed_updates + 1
            for group in self.optimizer.param_groups:
                group["lr"] = cfg.rate(step)
            super().update(*positional, **keywords)
            if step % 10 == 0:
                for name, value in self.metrics._last_losses.items():
                    self.board.add_scalar("training/" + name, value, step)
                self.board.add_scalar("training/learning_rate", cfg.rate(step), step)
                self.board.add_scalar("ods/generation", self.generation, step)
            if step % cfg.evaluate_every == 0:
                name = f"checkpoint-{step:08d}.pt"
                self.save_checkpoint(name, step)
                self.develop(output / name, step)

    experiment = LearningExperiment()
    result: dict[str, Any] = {}
    try:
        result = experiment.execute()
        candidates = [row for row in experiment.development if row["step"] > 0]
        selected = max(
            candidates, key=lambda row: (row["success_rate"], -row["goal_error"])
        )
        stable = all(
            row["success_rate"] >= cfg.development_success_rate
            for row in candidates[-2:]
        )
        selection = {
            "checkpoint": selected["checkpoint"],
            "step": selected["step"],
            "sha256": selected["checkpoint_sha256"],
            "development_gate_passed": stable,
        }
        (output / "selection.json").write_text(json.dumps(selection, indent=2))
        final = None
        if stable:
            final = experiment.evaluate(
                Path(selected["checkpoint"]),
                cfg.test_scenes,
                cfg.test_seed,
                "final-test",
            )
        met = (
            stable
            and final is not None
            and final["success_rate"] >= cfg.test_success_rate
        )
        result.update(
            learning_target_met=met,
            development_gate_passed=stable,
            selected_checkpoint=selection,
            final_test=final,
            learning_status="qualified" if met else "budget_exhausted_not_converged",
        )
    except BaseException as error:
        result = (
            experiment.failure(error)
            if not experiment.closed
            else {
                **result,
                "status": "failed",
                "error": repr(error),
                "learning_target_met": False,
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        experiment.board.close()
    (output / "summary.json").write_text(json.dumps(result, indent=2))
    return 0 if result.get("learning_target_met") else 2


def _supervise(args: argparse.Namespace, cfg: ConvergenceCfg) -> int:
    import psutil
    from .suite import _gpu_contexts, _partition_contexts, _terminate
    from .training_suite import _devices

    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "review.json").write_text(json.dumps(review_recipe(cfg), indent=2))
    frozen_recipe = args.output / "recipe.json"
    frozen_recipe.write_text(json.dumps(asdict(cfg), indent=2) + "\n")
    pairs, audit = _devices(args)
    if len(pairs) != 1:
        raise ValueError("Convergence training requires exactly two admitted GPUs")
    pair = pairs[0]
    devices = [pair["simulation"], pair["training"]]
    auxiliary = audit["aux_graphics_gpu"]
    allowed = {gpu["uuid"] for gpu in devices}
    affinity = pair["cpu_affinity"]
    environment = os.environ.copy()
    for key in ("DISPLAY", "WAYLAND_DISPLAY", "ODS_AUX_GRAPHICS_GPU_UUID"):
        environment.pop(key, None)
    environment.update(
        CUDA_VISIBLE_DEVICES=",".join(gpu["uuid"] for gpu in devices),
        ODS_GPU_UUID=devices[0]["uuid"],
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        SDL_VIDEODRIVER="dummy",
    )
    if auxiliary is not None:
        environment["ODS_AUX_GRAPHICS_GPU_UUID"] = auxiliary["uuid"]
    command = [
        sys.executable,
        "-m",
        "examples.ods.data_stream_demo.converge",
        "train",
        "--config",
        str(frozen_recipe),
        "--output",
        str(args.output / "training"),
        "--worker",
    ]
    started = time.monotonic()
    failure: BaseException | None = None
    environment["ODS_PROCESS_LAUNCH_MONOTONIC_NS"] = str(time.monotonic_ns())
    with (args.output / "worker.log").open("w") as stream:
        child = subprocess.Popen(
            command,
            cwd=_ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            preexec_fn=lambda: os.sched_setaffinity(0, affinity),
        )
        try:
            while child.poll() is None:
                pids = {child.pid}
                try:
                    pids.update(
                        p.pid
                        for p in psutil.Process(child.pid).children(recursive=True)
                    )
                except psutil.NoSuchProcess:
                    pass
                contexts = []
                for context in _gpu_contexts():
                    try:
                        owned = (
                            context["pid"] in pids
                            or os.getpgid(context["pid"]) == child.pid
                        )
                    except ProcessLookupError:
                        continue
                    if owned:
                        contexts.append(context)
                outside = [c for c in contexts if c["gpu_uuid"] not in allowed]
                _, unexpected = _partition_contexts(
                    outside,
                    "outside-primary-pair",
                    None if auxiliary is None else auxiliary["uuid"],
                )
                if unexpected:
                    raise RuntimeError(
                        f"Owned work used unassigned GPU contexts: {unexpected}"
                    )
                if time.monotonic() - started > cfg.run_timeout + 120:
                    raise TimeoutError("Convergence worker exceeded total runtime")
                time.sleep(1)
        except BaseException as error:
            failure = error
        finally:
            if child.poll() is None:
                _terminate(child)
    members = []
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        members = []
        for process in psutil.process_iter(["pid"]):
            try:
                if os.getpgid(process.pid) == child.pid:
                    members.append(process.pid)
            except ProcessLookupError:
                pass
        if not members:
            break
        time.sleep(0.05)
    (args.output / "process.json").write_text(
        json.dumps(
            {
                "returncode": child.returncode,
                "remaining_group_members": members,
                "gpu_pair": devices,
                "cpu_affinity": affinity,
                "command": command,
                "wall_seconds": time.monotonic() - started,
                "error": None if failure is None else repr(failure),
            },
            indent=2,
        )
    )
    if members:
        _terminate(child)
        raise RuntimeError("Convergence worker left owned processes")
    if failure is not None:
        raise failure
    return child.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("review", "train"))
    parser.add_argument("--config", type=Path, default=_DEFAULT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu-ids", type=int, nargs=2)
    parser.add_argument("--aux-graphics-gpu-id", type=int)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    cfg = load_recipe(args.config)
    if args.command == "review":
        review = review_recipe(cfg)
        review["config_sha256"] = hashlib.sha256(args.config.read_bytes()).hexdigest()
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(review, indent=2) + "\n")
        print(json.dumps(review, indent=2))
        return
    if args.output is None or (not args.worker and args.gpu_ids is None):
        parser.error(
            "train requires --output and an explicit generation/training --gpu-ids pair"
        )
    raise SystemExit(
        _worker(args.output, cfg) if args.worker else _supervise(args, cfg)
    )


if __name__ == "__main__":
    main()
