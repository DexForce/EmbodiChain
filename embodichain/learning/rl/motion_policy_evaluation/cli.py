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

"""Command line for visual evaluation of EmbodiChain policy checkpoints."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from embodichain import __version__
from embodichain.lab.gym.utils.registration import (
    discover_task_packages,
    execute_init_hooks,
)
from embodichain.learning.rl.runtime import (
    build_gym_policy_runtime,
    build_learning_policy_runtime,
    resolve_torch_device,
    seed_policy_runtime,
)
from embodichain.utils.utility import load_config

from .bridge import MotionEvaluationResult, evaluate_motion_profile
from .checkpoint import load_policy_state_dict
from .manifest import RunManifest
from .native_task import NativeTaskEvaluationResult, evaluate_native_task
from .profile import MotionProfileRequest, build_motion_profile
from .report import write_evaluation_report

__all__ = ["cli", "parse_args", "run"]


@dataclass(frozen=True)
class MotionInput:
    """Resolved checkpoint, configs, and Profile selection."""

    checkpoint: Path
    profile: str | None
    configs: Mapping[str, Path]
    run: Path | None
    requested_checkpoint: str
    selected_checkpoint: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse ``embodichain eval-motion-policy`` arguments."""
    parser = argparse.ArgumentParser(
        prog="embodichain eval-motion-policy",
        description="Open a DexSim Viewer for an EmbodiChain policy checkpoint.",
    )
    parser.add_argument("run", nargs="?", help="EmbodiChain training run directory.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--profile", help="Registered Motion Profile name.")
    mode.add_argument(
        "--original-task",
        action="store_true",
        help="Use the original EmbodiChain task instead of the manifest Profile.",
    )
    parser.add_argument(
        "--checkpoint",
        help="Checkpoint path, or best/latest when RUN is supplied.",
    )
    parser.add_argument("--config", help="Training config for an explicit checkpoint.")
    parser.add_argument("--gym-config", help="Task config for an explicit checkpoint.")
    parser.add_argument("--resource-root", help="Profile-specific local resource root.")
    parser.add_argument("--episodes", type=int)
    count = parser.add_mutually_exclusive_group()
    count.add_argument("--control-steps", type=int)
    count.add_argument("--duration", type=float)
    parser.add_argument("--command", nargs="+", type=float)
    parser.add_argument("--device", help="PyTorch inference device.")
    parser.add_argument("--sim-device", choices=("cpu", "gpu"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--physics-backend", choices=("default",))
    parser.add_argument(
        "--renderer",
        choices=("raster", "hybrid", "fastrt", "offlinert"),
        default="hybrid",
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
    """Resolve inputs, run the evaluation, and write ``evaluation.json``."""
    resolved = _resolve_input(args)
    discover_task_packages()
    execute_init_hooks()
    if resolved.profile is None:
        return _run_native_task(args, resolved)

    device = torch.device(args.device or "cpu")
    profile = build_motion_profile(
        resolved.profile,
        MotionProfileRequest(
            checkpoint=resolved.checkpoint,
            device=device,
            configs=resolved.configs,
            resource_root=(
                None if args.resource_root is None else Path(args.resource_root)
            ),
            renderer=args.renderer,
        ),
    )
    for warning in profile.warnings:
        print(f"Warning: {warning}", file=sys.stderr)
    result = evaluate_motion_profile(
        profile,
        episodes=args.episodes or 1,
        viewer=args.viewer,
        control_steps=args.control_steps,
        duration=args.duration,
        command=None if args.command is None else tuple(args.command),
        scene_config=args.scene_config or "standard",
        physics_backend=args.physics_backend,
        simulation_device=args.sim_device or "cpu",
        renderer=args.renderer,
        gpu_id=args.gpu_id,
        termination_behavior=args.termination_behavior,
        cache_dir=args.cache_dir,
        offline=args.offline,
    )
    return write_evaluation_report(
        _output_parent(args.output, resolved),
        _profile_report(result, resolved, device),
    )


def cli(argv: Sequence[str] | None = None) -> None:
    """Run motion-policy evaluation from the unified EmbodiChain CLI."""
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
        raise SystemExit(f"eval-motion-policy: {error}") from error
    print(f"Evaluation report: {report}")


def _resolve_input(args: argparse.Namespace) -> MotionInput:
    if args.run is not None:
        manifest = RunManifest.load(args.run)
        requested = args.checkpoint or "best"
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
        profile = (
            None if args.original_task else args.profile or manifest.motion_profile
        )
        return MotionInput(
            checkpoint=checkpoint,
            profile=profile,
            configs=manifest.configs,
            run=manifest.root,
            requested_checkpoint=requested,
            selected_checkpoint=selected,
        )
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required without RUN")
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    configs = {}
    if args.config is not None:
        configs["train"] = Path(args.config).expanduser().resolve()
    if args.gym_config is not None:
        configs["gym"] = Path(args.gym_config).expanduser().resolve()
    if args.profile is None and "train" not in configs:
        raise ValueError("--config is required for a native EmbodiChain checkpoint")
    return MotionInput(
        checkpoint=checkpoint,
        profile=args.profile,
        configs=configs,
        run=None,
        requested_checkpoint=args.checkpoint,
        selected_checkpoint="explicit",
    )


def _run_native_task(args: argparse.Namespace, resolved: MotionInput) -> Path:
    _validate_native_options(args)
    train_config = resolved.configs.get("train")
    if train_config is None:
        raise ValueError("Training config is required for native task evaluation")
    config = load_config(train_config)
    config["trainer"] = dict(config["trainer"])
    gym_config = resolved.configs.get("gym")
    if gym_config is not None:
        config["trainer"]["gym_config"] = str(gym_config)
    trainer = config["trainer"]
    device = resolve_torch_device(args.device or trainer.get("device", "cpu"))
    if args.sim_device == "gpu":
        simulation_device = resolve_torch_device(f"cuda:{args.gpu_id}")
    elif args.sim_device == "cpu":
        simulation_device = torch.device("cpu")
    else:
        simulation_device = device
    seed = int(
        args.seed
        if args.seed is not None
        else trainer.get("eval_seed", int(trainer.get("seed", 1)) + 10_000)
    )
    seed_policy_runtime(seed, device)
    uses_simulator = "learning_env" not in trainer
    if uses_simulator:
        runtime = build_gym_policy_runtime(
            config,
            device=device,
            simulation_device=simulation_device,
            num_envs=1,
            headless=not args.viewer,
            renderer=args.renderer,
            gpu_id=args.gpu_id,
            config_dir=train_config.parent,
        )
    else:
        runtime = build_learning_policy_runtime(
            config,
            device=device,
            num_envs=1,
        )
    try:
        try:
            runtime.policy.load_state_dict(
                load_policy_state_dict(resolved.checkpoint, map_location="cpu")
            )
        except Exception:
            runtime.env.close()
            raise

        episodes = args.episodes
        if (
            episodes is None
            and not args.viewer
            and args.control_steps is None
            and args.duration is None
        ):
            episodes = 1
        result = evaluate_native_task(
            runtime,
            seed=seed,
            viewer=args.viewer,
            episodes=episodes,
            control_steps=args.control_steps,
            duration=args.duration,
            termination_behavior=args.termination_behavior or "auto_reset",
        )
    finally:
        if uses_simulator:
            from embodichain.lab.sim.sim_manager import SimulationManager

            SimulationManager.flush_cleanup_queue()
    return write_evaluation_report(
        _output_parent(args.output, resolved),
        _native_report(
            result,
            resolved,
            device=device,
            simulation_device=simulation_device,
            seed=seed,
            renderer=args.renderer,
        ),
    )


def _validate_native_options(args: argparse.Namespace) -> None:
    """Reject options that belong to external Motion Profiles."""
    values = {
        "--resource-root": args.resource_root,
        "--command": args.command,
        "--physics-backend": args.physics_backend,
        "--scene-config": args.scene_config,
        "--cache-dir": args.cache_dir,
        "--offline": args.offline,
    }
    selected = [name for name, value in values.items() if value not in (None, False)]
    if selected:
        raise ValueError(f"{', '.join(selected)} requires --profile")


def _output_parent(configured: str | None, resolved: MotionInput) -> Path:
    if configured is not None:
        return Path(configured)
    if resolved.run is not None:
        return resolved.run / "evaluations"
    return resolved.checkpoint.parent / "evaluations"


def _profile_report(
    result: MotionEvaluationResult,
    resolved: MotionInput,
    device: torch.device,
) -> dict[str, Any]:
    import dexsim

    return {
        "mode": "viewer" if result.viewer else "headless",
        "inputs": {
            "run": resolved.run,
            "checkpoint": {
                "path": resolved.checkpoint,
                "requested": resolved.requested_checkpoint,
                "selected": resolved.selected_checkpoint,
            },
            "profile": {
                "id": result.profile.profile_id,
                "provider_version": result.profile.provider_version,
                "provenance": result.profile.provenance,
                "warnings": result.profile.warnings,
            },
            "configs": resolved.configs,
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


def _native_report(
    result: NativeTaskEvaluationResult,
    resolved: MotionInput,
    *,
    device: torch.device,
    simulation_device: torch.device,
    seed: int,
    renderer: str,
) -> dict[str, Any]:
    import dexsim

    return {
        "mode": "viewer" if result.viewer else "headless",
        "inputs": {
            "run": resolved.run,
            "checkpoint": {
                "path": resolved.checkpoint,
                "requested": resolved.requested_checkpoint,
                "selected": resolved.selected_checkpoint,
            },
            "configs": resolved.configs,
            "task_id": result.task_id,
            "seed": seed,
            "inference_device": str(device),
            "simulation_device": str(simulation_device),
            "renderer": renderer,
            "embodichain_version": __version__,
            "dexsim_version": getattr(dexsim, "__version__", None),
            "dexsim_commit": getattr(dexsim, "__commit_id__", None),
        },
        "result": {
            "reason": result.reason,
            "simulation_time": result.simulation_time,
            "simulation_steps": result.simulation_steps,
            "control_steps": result.control_steps,
            "physics_backend": result.physics_backend,
            "requested_duration": result.requested_duration,
            "effective_duration": result.effective_duration,
            "episodes": result.episodes,
            "metrics": result.metrics,
        },
    }
