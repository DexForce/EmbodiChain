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

"""Unified policy evaluation for EmbodiChain training runs."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from embodichain import __version__
from embodichain.lab.gym.utils.registration import (
    discover_task_packages,
    execute_init_hooks,
)
from embodichain.learning.rl.evaluation import evaluate_episodes
from embodichain.learning.rl.runtime import (
    PolicyRuntime,
    build_gym_policy_runtime,
    build_learning_policy_runtime,
)
from embodichain.utils.utility import load_config

from .manifest import RunManifest
from .report import write_evaluation_report

__all__ = ["cli", "parse_args", "run"]


@dataclass(frozen=True)
class EvaluationInput:
    """Checkpoint and configuration selected for one evaluation."""

    checkpoint: Path
    profile: str | None
    configs: Mapping[str, Path]
    run: Path | None
    requested_checkpoint: str
    selected_checkpoint: str


@dataclass(frozen=True)
class NativeRuntime:
    """Reconstructed EmbodiChain task and its runtime choices."""

    runtime: PolicyRuntime
    device: torch.device
    simulation_device: torch.device
    seed: int
    renderer: str
    uses_simulator: bool
    trainer: Mapping[str, Any]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse ``embodichain eval-policy`` arguments."""
    parser = argparse.ArgumentParser(
        prog="embodichain eval-policy",
        description="Evaluate an EmbodiChain or external policy checkpoint.",
    )
    parser.add_argument("run", nargs="?", help="EmbodiChain training run directory.")
    parser.add_argument("--profile", help="Registered external Policy Profile.")
    parser.add_argument(
        "--checkpoint",
        help="latest, best, or a checkpoint path; defaults to latest with RUN.",
    )
    parser.add_argument("--config", help="Training config for an explicit checkpoint.")
    parser.add_argument("--gym-config", help="Task config override.")
    parser.add_argument("--resource-root", help="External Profile resource root.")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--num-envs", type=int)
    count = parser.add_mutually_exclusive_group()
    count.add_argument("--control-steps", type=int)
    count.add_argument("--duration", type=float)
    parser.add_argument("--command", nargs="+", type=float)
    parser.add_argument("--device", help="PyTorch inference device.")
    parser.add_argument("--sim-device", choices=("cpu", "gpu"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--physics-backend")
    parser.add_argument(
        "--renderer",
        choices=("raster", "hybrid", "fastrt", "offlinert"),
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--scene-config")
    parser.add_argument(
        "--termination-behavior",
        choices=("pause", "continue", "auto_reset"),
    )
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--cache-dir")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", help="Evaluation output parent directory.")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Path:
    """Run Headless or Viewer evaluation and write ``evaluation.json``."""
    resolved = _resolve_input(args)
    if resolved.profile is not None:
        return _run_profile(args, resolved)
    discover_task_packages()
    execute_init_hooks()
    _validate_native_options(args)
    if args.viewer:
        return _run_native_viewer(args, resolved)
    return _run_native_headless(args, resolved)


def cli(argv: Sequence[str] | None = None) -> None:
    """Run policy evaluation from the unified EmbodiChain CLI."""
    try:
        report = run(parse_args(argv))
    except (
        FileNotFoundError,
        ImportError,
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        raise SystemExit(f"eval-policy: {error}") from error
    print(f"Evaluation report: {report}")


def _resolve_input(args: argparse.Namespace) -> EvaluationInput:
    if args.run is not None:
        manifest = RunManifest.load(args.run)
        requested = args.checkpoint or "latest"
        if requested in {"best", "latest"}:
            selected, checkpoint = manifest.select_checkpoint(requested)
        else:
            selected = "explicit"
            candidate = Path(requested).expanduser()
            checkpoint = (
                candidate.resolve()
                if candidate.is_absolute()
                else (manifest.root / candidate).resolve()
            )
        configs = dict(manifest.configs)
        if args.config is not None:
            configs["train"] = Path(args.config).expanduser().resolve()
        if args.gym_config is not None:
            configs["gym"] = Path(args.gym_config).expanduser().resolve()
        return EvaluationInput(
            checkpoint=checkpoint,
            profile=args.profile,
            configs=configs,
            run=manifest.root,
            requested_checkpoint=requested,
            selected_checkpoint=selected,
        )
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required without RUN")
    configs = {}
    if args.config is not None:
        configs["train"] = Path(args.config).expanduser().resolve()
    if args.gym_config is not None:
        configs["gym"] = Path(args.gym_config).expanduser().resolve()
    if args.profile is None and "train" not in configs:
        raise ValueError("--config is required for an EmbodiChain checkpoint")
    return EvaluationInput(
        checkpoint=Path(args.checkpoint).expanduser().resolve(),
        profile=args.profile,
        configs=configs,
        run=None,
        requested_checkpoint=args.checkpoint,
        selected_checkpoint="explicit",
    )


def _run_native_headless(
    args: argparse.Namespace,
    resolved: EvaluationInput,
) -> Path:
    native = _build_native_runtime(args, resolved, viewer=False)
    episodes = (
        args.episodes
        if args.episodes is not None
        else int(native.trainer.get("num_eval_episodes", 5))
    )
    try:
        metrics = evaluate_episodes(
            policy=native.runtime.policy,
            env=native.runtime.env,
            num_episodes=episodes,
            device=native.device,
            seed=native.seed,
        )
    finally:
        native.runtime.close()
        _flush_simulator(native.uses_simulator)
    return write_evaluation_report(
        _output_parent(args.output, resolved),
        _headless_report(native, resolved, episodes, metrics),
    )


def _run_native_viewer(
    args: argparse.Namespace,
    resolved: EvaluationInput,
) -> Path:
    from .viewer import evaluate_native_viewer

    native = _build_native_runtime(args, resolved, viewer=True)
    try:
        result = evaluate_native_viewer(
            native.runtime,
            seed=native.seed,
            episodes=args.episodes,
            control_steps=args.control_steps,
            duration=args.duration,
            termination_behavior=args.termination_behavior or "auto_reset",
        )
    finally:
        _flush_simulator(True)
    return write_evaluation_report(
        _output_parent(args.output, resolved),
        _viewer_report(result, native, resolved),
    )


def _run_profile(args: argparse.Namespace, resolved: EvaluationInput) -> Path:
    from .bridge import evaluate_motion_profile
    from .profile import MotionProfileRequest, build_motion_profile

    device = _torch_device(args.device or "cpu")
    renderer = args.renderer or "hybrid"
    profile = build_motion_profile(
        resolved.profile,
        MotionProfileRequest(
            checkpoint=resolved.checkpoint,
            device=device,
            configs=resolved.configs,
            resource_root=(
                None if args.resource_root is None else Path(args.resource_root)
            ),
            renderer=renderer,
        ),
    )
    for warning in profile.warnings:
        print(f"Warning: {warning}", file=sys.stderr)
    result = evaluate_motion_profile(
        profile,
        episodes=args.episodes if args.episodes is not None else 1,
        viewer=args.viewer,
        control_steps=args.control_steps,
        duration=args.duration,
        command=None if args.command is None else tuple(args.command),
        scene_config=args.scene_config or "standard",
        physics_backend=args.physics_backend,
        simulation_device=args.sim_device or "cpu",
        renderer=renderer,
        gpu_id=args.gpu_id,
        termination_behavior=args.termination_behavior,
        cache_dir=args.cache_dir,
        offline=args.offline,
    )
    return write_evaluation_report(
        _output_parent(args.output, resolved),
        _profile_report(result, resolved, device),
    )


def _build_native_runtime(
    args: argparse.Namespace,
    resolved: EvaluationInput,
    *,
    viewer: bool,
) -> NativeRuntime:
    train_config = resolved.configs.get("train")
    if train_config is None:
        raise ValueError("Training config is required for an EmbodiChain checkpoint")
    config = load_config(train_config)
    config["trainer"] = dict(config["trainer"])
    gym_config = resolved.configs.get("gym")
    if gym_config is not None:
        config["trainer"]["gym_config"] = str(gym_config)
    trainer = config["trainer"]
    device = _torch_device(args.device or trainer.get("device", "cpu"))
    simulation_device = _simulation_device(args, device)
    seed = int(
        args.seed
        if args.seed is not None
        else trainer.get("eval_seed", int(trainer.get("seed", 1)) + 10_000)
    )
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    uses_simulator = "gym_config" in trainer
    if viewer and not uses_simulator:
        raise ValueError("--viewer requires a simulator training task")
    renderer = args.renderer or str(trainer.get("renderer", "hybrid"))
    num_envs = (
        1
        if viewer
        else int(
            args.num_envs
            if args.num_envs is not None
            else trainer.get("num_eval_envs", 4)
        )
    )
    if uses_simulator:
        runtime = build_gym_policy_runtime(
            config,
            device=device,
            simulation_device=simulation_device,
            num_envs=num_envs,
            headless=not viewer,
            renderer=renderer,
            gpu_id=args.gpu_id,
            config_dir=train_config.parent,
        )
    else:
        runtime = build_learning_policy_runtime(
            config,
            device=device,
            num_envs=num_envs,
        )
    try:
        runtime.policy.load_state_dict(_load_policy_state_dict(resolved.checkpoint))
    except Exception:
        runtime.close()
        _flush_simulator(uses_simulator)
        raise
    return NativeRuntime(
        runtime=runtime,
        device=device,
        simulation_device=simulation_device,
        seed=seed,
        renderer=renderer,
        uses_simulator=uses_simulator,
        trainer=trainer,
    )


def _validate_native_options(args: argparse.Namespace) -> None:
    profile_options = {
        "--resource-root": args.resource_root,
        "--command": args.command,
        "--physics-backend": args.physics_backend,
        "--scene-config": args.scene_config,
        "--cache-dir": args.cache_dir,
        "--offline": args.offline,
    }
    selected = [
        name for name, value in profile_options.items() if value not in (None, False)
    ]
    if selected:
        raise ValueError(f"{', '.join(selected)} requires --profile")
    if not args.viewer and (
        args.control_steps is not None
        or args.duration is not None
        or args.termination_behavior is not None
    ):
        raise ValueError(
            "--control-steps, --duration, and --termination-behavior require --viewer"
        )


def _load_policy_state_dict(checkpoint: Path) -> Mapping[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or not isinstance(
        payload.get("policy"), Mapping
    ):
        raise TypeError("Checkpoint must contain a 'policy' state mapping")
    return payload["policy"]


def _torch_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda":
        index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        torch.cuda.set_device(index)
        return torch.device(f"cuda:{index}")
    if device.type != "cpu":
        raise ValueError(f"Unsupported device type: {device.type}")
    return device


def _simulation_device(
    args: argparse.Namespace,
    inference_device: torch.device,
) -> torch.device:
    if args.sim_device == "gpu":
        return _torch_device(f"cuda:{args.gpu_id}")
    if args.sim_device == "cpu":
        return torch.device("cpu")
    return inference_device


def _flush_simulator(enabled: bool) -> None:
    if enabled:
        from embodichain.lab.sim.sim_manager import SimulationManager

        SimulationManager.flush_cleanup_queue()


def _output_parent(configured: str | None, resolved: EvaluationInput) -> Path:
    if configured is not None:
        return Path(configured)
    if resolved.run is not None:
        return resolved.run / "evaluations"
    return resolved.checkpoint.parent / "evaluations"


def _checkpoint_inputs(resolved: EvaluationInput) -> dict[str, Any]:
    return {
        "run": resolved.run,
        "checkpoint": {
            "path": resolved.checkpoint,
            "requested": resolved.requested_checkpoint,
            "selected": resolved.selected_checkpoint,
        },
        "configs": resolved.configs,
    }


def _headless_report(
    native: NativeRuntime,
    resolved: EvaluationInput,
    episodes: int,
    metrics: Mapping[str, float],
) -> dict[str, Any]:
    return {
        "mode": "headless",
        "inputs": {
            **_checkpoint_inputs(resolved),
            "task_id": native.runtime.env_id,
            "seed": native.seed,
            "num_envs": int(native.runtime.env.num_envs),
            "device": str(native.device),
            "embodichain_version": __version__,
        },
        "result": {"episodes": episodes, "metrics": metrics},
    }


def _viewer_report(
    result: Any,
    native: NativeRuntime,
    resolved: EvaluationInput,
) -> dict[str, Any]:
    import dexsim

    return {
        "mode": "viewer",
        "inputs": {
            **_checkpoint_inputs(resolved),
            "task_id": result.task_id,
            "seed": native.seed,
            "inference_device": str(native.device),
            "simulation_device": str(native.simulation_device),
            "renderer": native.renderer,
            "embodichain_version": __version__,
            "dexsim_version": getattr(dexsim, "__version__", None),
            "dexsim_commit": getattr(dexsim, "__commit_id__", None),
        },
        "result": {
            "reason": result.reason,
            "simulation_time": result.simulation_time,
            "simulation_steps": result.simulation_steps,
            "control_steps": result.control_steps,
            "requested_duration": result.requested_duration,
            "effective_duration": result.effective_duration,
            "episodes": result.episodes,
            "metrics": result.metrics,
        },
    }


def _profile_report(
    result: Any,
    resolved: EvaluationInput,
    device: torch.device,
) -> dict[str, Any]:
    import dexsim

    return {
        "mode": "viewer" if result.viewer else "headless",
        "inputs": {
            **_checkpoint_inputs(resolved),
            "profile": {
                "id": result.profile.profile_id,
                "provider_version": result.profile.provider_version,
                "provenance": result.profile.provenance,
                "warnings": result.profile.warnings,
            },
            "policy_spec": result.policy_spec,
            "scene_config": result.scene_config,
            "inference_device": str(device),
            "embodichain_version": __version__,
            "dexsim_version": getattr(dexsim, "__version__", None),
            "dexsim_commit": getattr(dexsim, "__commit_id__", None),
        },
        "result": {
            "episodes": result.episodes,
            "summary": result.summary,
        },
    }
