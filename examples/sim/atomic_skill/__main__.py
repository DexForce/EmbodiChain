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

"""Run Atomic Skill sample scenes and trajectory expansion from direct arguments."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import importlib
import json
import math
from pathlib import Path
from typing import Any

from embodichain.cli.sim import add_sim_args_to_parser
from examples.sim.atomic_skill.defaults import (
    PLANNERS,
    ROBOTS,
    SCENES,
    AtomicSkillRunCfg,
    _sample_defaults,
    _scene_args,
)


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="embodichain run-example sim/atomic_skill", description=__doc__
    )
    add_sim_args_to_parser(parser)
    parser.set_defaults(
        num_envs=None,
        device=None,
        renderer=None,
        arena_space=None,
        gpu_id=None,
    )
    parser.add_argument(
        "--skill",
        required=True,
        choices=SCENES,
        help="Atomic Skill sample scene to run.",
    )
    parser.add_argument(
        "--robot",
        choices=ROBOTS,
        default="ur5",
        help="Sample robot preset (default: ur5).",
    )
    parser.add_argument(
        "--planner",
        choices=PLANNERS,
        default="trapezoidal",
        help="Motion planner backend (default: trapezoidal).",
    )
    # The common parser owns --physics; this sample launcher selects it directly.
    for action in parser._actions:
        if action.dest == "physics":
            action.help = "Physics backend for the sample scene (default: default)."
        elif action.dest == "num_envs":
            action.help = (
                "Compatibility check: must equal --variants; capacity is automatic."
            )
        elif action.dest == "renderer":
            action.help = "Renderer backend; omission uses the sample's auto renderer."
    parser.add_argument(
        "--no-expansion",
        action="store_true",
        help="Replay only the nominal grasp and trajectory.",
    )
    parser.add_argument(
        "--variants",
        type=int,
        default=1,
        help="Total executions including the reference; creates N environments.",
    )
    parser.add_argument(
        "--expansion",
        choices=("affordance", "trajectory", "both"),
        default="both",
        help="Enabled factors when --variants is greater than one.",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        type=Path,
        help="Override the trajectory/video/provenance output directory.",
    )
    parser.add_argument(
        "--save-video",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Record each replay as MP4.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--real-time", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--keep-alive",
        action="store_true",
        help="Keep the final scene open after replay until Ctrl+C.",
    )
    return parser


def _resolve_sample(args: argparse.Namespace) -> AtomicSkillRunCfg:
    cfg = _sample_defaults(
        args.skill, physics=args.physics, robot=args.robot, planner=args.planner
    )
    cfg.variants = args.variants
    cfg.expansion = args.expansion
    if cfg.variants < 1:
        raise ValueError("--variants must be positive")
    if args.no_expansion and cfg.variants != 1:
        raise ValueError("--no-expansion requires --variants 1")
    if args.num_envs is not None and args.num_envs != cfg.variants:
        raise ValueError(
            "--num-envs must equal --variants; environment count is automatic"
        )
    if cfg.expansion == "affordance" and cfg.skill in {"move_joint", "push_object"}:
        raise ValueError(
            f"{cfg.skill} has no affordance expansion; use --expansion trajectory"
        )
    for key in (
        "device",
        "headless",
        "renderer",
        "arena_space",
        "gpu_id",
        "real_time",
        "seed",
        "save_video",
    ):
        value = getattr(args, key, None)
        if value is not None:
            setattr(cfg, key, value)
    if cfg.gpu_id < 0:
        raise ValueError("--gpu-id must be non-negative")
    if cfg.arena_space is not None and (
        not math.isfinite(cfg.arena_space) or cfg.arena_space <= 0
    ):
        raise ValueError("--arena-space must be finite and positive")
    if not cfg.device:
        raise ValueError("--device must be a nonempty device string")
    if not 0 <= cfg.seed < 2**32:
        raise ValueError("--seed must be in [0, 2**32)")
    if args.output_dir is not None:
        cfg.output_dir = str(args.output_dir.expanduser().resolve())
    if cfg.save_video and cfg.renderer == "no-render":
        raise ValueError("--save-video requires a renderer")
    cfg.validate()
    return cfg


def _run_sample(cfg: AtomicSkillRunCfg, cli_args: argparse.Namespace, run: Any) -> None:
    from embodichain.lab.sim import SimulationManager
    from embodichain.utils import set_seed
    from examples.sim.atomic_skill.support import run_tutorial

    scene_args = _scene_args(cfg)
    scene_args.run = run
    for key, value in vars(cli_args).items():
        if key == "viser" or key.startswith("viser_"):
            setattr(scene_args, key, value)
    scene = importlib.import_module(f"examples.sim.atomic_skill.{cfg.skill}")
    error = None

    def run_scene() -> None:
        nonlocal error
        try:
            set_seed(cfg.seed)
            scene.run(scene_args)
            run.validate_variants()
            if cli_args.keep_alive:
                sim = SimulationManager.get_instance()
                try:
                    while True:
                        sim.update(step=1)
                except KeyboardInterrupt:
                    pass
        except Exception as exc:
            # Unwind native scene wrappers before run_tutorial destroys the world.
            import traceback

            error = traceback.format_exc()

    run_tutorial(run_scene)
    if error is not None:
        raise RuntimeError(error)


def cli(argv: Sequence[str] | None = None) -> None:
    """Run N fixed-scene Atomic executions with automatic parallel capacity.

    Args:
        argv: Arguments excluding the executable and subcommand names.
    """
    parser = _create_parser()
    args = parser.parse_args(argv)
    try:
        cfg = _resolve_sample(args)
    except ValueError as exc:
        parser.error(str(exc))
    from examples.sim.atomic_skill.host import AtomicSkillRun

    directory = Path(cfg.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    run = AtomicSkillRun(cfg)
    manifest = {
        "schema_version": 2,
        "config": cfg.to_dict(),
        "variants": cfg.variants,
        "num_envs": cfg.variants,
        "status": "running",
        "replays": run.replays,
        "measured_task_success": None,
        "enabled_factors": [
            name
            for name, enabled in (
                ("affordance", cfg.affordance_enabled),
                ("trajectory", cfg.trajectory_enabled),
            )
            if enabled
        ],
    }
    manifest_path = directory / "expansion_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        _run_sample(cfg, args, run)
        manifest["status"] = "executed"
        print(f"Executed {cfg.variants} Atomic variants. Manifest: {manifest_path}")
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    cli()


__all__: list[str] = []
