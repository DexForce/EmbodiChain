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

"""Configured CLI for ten Atomic Skill example scenes and trajectory expansion."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from copy import deepcopy
import importlib
import json
from pathlib import Path
from typing import Any

from embodichain.cli.sim import add_sim_args_to_parser
from embodichain.lab.scripts._atomic_action.config import (
    SCENES,
    _load_config,
    _number,
    _recipes,
    _scene_args,
)


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="embodichain run-atomic-action", description=__doc__
    )
    add_sim_args_to_parser(parser)
    parser.set_defaults(
        num_envs=None,
        device=None,
        headless=None,
        renderer=None,
        physics=None,
        arena_space=None,
        gpu_id=None,
    )
    parser.add_argument(
        "--config",
        "--atomic-config",
        "--atomic_config",
        type=Path,
        help="Atomic scene YAML configuration.",
    )
    parser.add_argument(
        "--list-skills", action="store_true", help="List supported scene names."
    )
    parser.add_argument(
        "--no-expansion",
        action="store_true",
        help="Replay only the nominal grasp and trajectory.",
    )
    parser.add_argument(
        "--recipe-indices",
        "--recipe_indices",
        type=int,
        nargs="+",
        help="Select logical grasp/trajectory recipes independently of physical slots.",
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


def _resolve_config(args: argparse.Namespace) -> tuple[dict[str, Any], tuple[int, ...]]:
    cfg = _load_config(args.config.resolve())
    if args.physics is not None and args.physics != cfg["physics"]:
        raise ValueError(
            "--physics may only confirm the config-owned backend; use a separate config to change it"
        )
    for key in (
        "num_envs",
        "device",
        "headless",
        "renderer",
        "arena_space",
        "gpu_id",
        "real_time",
    ):
        value = getattr(args, key, None)
        if value is not None:
            cfg["runtime"][key] = value
    if type(cfg["runtime"]["num_envs"]) is not int or cfg["runtime"]["num_envs"] < 1:
        raise ValueError("--num-envs must be positive")
    if cfg["runtime"]["gpu_id"] < 0:
        raise ValueError("--gpu-id must be non-negative")
    if cfg["runtime"]["arena_space"] is not None:
        _number(cfg["runtime"]["arena_space"], "--arena-space", positive=True)
    if cfg["runtime"]["device"] == "":
        raise ValueError("--device must be a nonempty device string")
    if args.seed is not None:
        if not 0 <= args.seed < 2**32:
            raise ValueError("--seed must be in [0, 2**32)")
        cfg["seed"] = args.seed
    if args.output_dir is not None:
        cfg["output"]["directory"] = str(args.output_dir.expanduser().resolve())
    if args.save_video is not None:
        cfg["output"]["save_video"] = args.save_video
    if cfg["output"]["save_video"] and cfg["runtime"]["renderer"] == "no-render":
        raise ValueError("--save-video requires a renderer")
    if args.no_expansion:
        cfg["expansion"]["affordance"]["enabled"] = False
        cfg["expansion"]["trajectory"]["variants_per_family"] = 1
        cfg["expansion"]["trajectory"]["spatial"]["enabled"] = False
    recipes = _recipes(cfg)
    selected = (
        tuple(range(len(recipes)))
        if args.recipe_indices is None
        else tuple(args.recipe_indices)
    )
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(index < 0 or index >= len(recipes) for index in selected)
    ):
        raise ValueError(f"recipe indices must be unique and in [0, {len(recipes)})")
    return cfg, selected


def _run_batch(
    cfg: dict[str, Any],
    indices: tuple[int, ...],
    batch_id: int,
    cli_args: argparse.Namespace,
    *,
    last: bool,
) -> list[dict[str, Any]]:
    from embodichain.lab.sim import SimulationManager
    from embodichain.utils import set_seed
    from embodichain.lab.scripts._atomic_action.support import run_tutorial

    scene_args = _scene_args(cfg, indices, batch_id)
    scene_args.num_envs = len(indices)
    # Copy only visualization settings supplied by the shared launcher.
    for key, value in vars(cli_args).items():
        if key == "viser" or key.startswith("viser_"):
            setattr(scene_args, key, value)
    scene = importlib.import_module(
        f"embodichain.lab.scripts._atomic_action.{cfg['skill']}"
    )
    error = None

    def run() -> None:
        nonlocal error
        # Release failure tracebacks before native simulation cleanup. A
        # traceback can otherwise retain borrowed material/scene wrappers.
        try:
            set_seed(cfg["seed"])
            scene.run(scene_args)
            if last and cli_args.keep_alive:
                sim = SimulationManager.get_instance()
                try:
                    while True:
                        sim.update(step=1)
                except KeyboardInterrupt:
                    pass
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

    run_tutorial(run)
    if error is not None:
        raise RuntimeError(error)
    return deepcopy(scene_args._records)


def cli(argv: Sequence[str] | None = None) -> None:
    """Run selected Atomic scene recipes from a YAML configuration.

    Args:
        argv: Arguments excluding the executable and subcommand names.
    """
    parser = _create_parser()
    args = parser.parse_args(argv)
    if args.list_skills:
        print("\n".join(SCENES))
        return
    if args.config is None:
        parser.error("--config is required unless --list-skills is selected")
    try:
        cfg, selected = _resolve_config(args)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    directory = Path(cfg["output"]["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "config": cfg,
        "recipe_indices": selected,
        "status": "running",
        "replays": [],
        "measured_task_success": None,
    }
    manifest_path = directory / "expansion_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        capacity = cfg["runtime"]["num_envs"]
        for batch_id, start in enumerate(range(0, len(selected), capacity)):
            indices = selected[start : start + capacity]
            records = _run_batch(
                cfg, indices, batch_id, args, last=start + capacity >= len(selected)
            )
            manifest["replays"].extend(records)
            manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
        manifest["status"] = "executed"
        print(f"Executed {len(selected)} Atomic recipes. Manifest: {manifest_path}")
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")


if __name__ == "__main__":
    cli()


__all__: list[str] = []
