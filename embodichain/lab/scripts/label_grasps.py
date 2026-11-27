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

"""Write a standalone dataset of physically validated object-local TCPs."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import yaml

from embodichain.data import get_data_path
from embodichain.robot_registry import resolve_robot_cfg
from embodichain.lab.sim.grasp.config import DEFAULT_NUM_ENVS, GraspSamplingCfg
from embodichain.lab.sim.grasp.labeling import label_grasps

__all__ = ["cli"]


def _asset(value: str | None, root: Path) -> Path:
    if not value:
        raise ValueError("Provide an asset path in the YAML or on the CLI")
    raw = Path(value).expanduser()
    local = root / raw
    path = local if raw.is_absolute() or local.exists() else Path(get_data_path(value))
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def cli(argv: Sequence[str] | None = None) -> None:
    """Save physically validated object-local grasp poses and opening widths.

    Args:
        argv: Command-line arguments, or None to read the process arguments.
            Use ``--config`` for the mesh, gripper and sampling settings and
            ``--output`` for the destination PyTorch file.
    """
    parser = argparse.ArgumentParser(
        prog="embodichain affordance label-grasps",
        description=__doc__,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mesh", help="Override the target mesh path in the YAML.")
    parser.add_argument(
        "--gripper-urdf", help="Override the standalone hand URDF in the YAML."
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Destination .pt file."
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        help=f"Override the YAML candidate arena count (default {DEFAULT_NUM_ENVS}).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue the checkpoint next to the output.",
    )
    budgets = (
        "stable_poses",
        "directions",
        "candidates_per_direction",
        "sample_count",
        "seed",
    )
    for name in budgets:
        parser.add_argument("--" + name.replace("_", "-"), type=int)
    args = parser.parse_args(argv)
    if args.output.suffix != ".pt":
        parser.error("Use a .pt output path")
    config = yaml.safe_load(args.config.read_text())
    if not isinstance(config, dict) or not isinstance(config.get("gripper"), dict):
        parser.error("The YAML must contain a gripper settings mapping")
    configured_envs = config.pop("num_envs", DEFAULT_NUM_ENVS)
    num_envs = configured_envs if args.num_envs is None else args.num_envs
    if type(num_envs) is not int or num_envs < 1:
        parser.error("num_envs must be a positive integer")
    unknown = config.keys() - {
        "physics",
        "mesh",
        "gripper_urdf",
        "gripper",
        "rigid",
        "sampling",
        "close_wait_steps",
        "enable_disturbance",
        "collision_threshold",
    }
    if unknown:
        parser.error(f"Unknown configuration fields: {sorted(unknown)}")
    params = dict(config.get("sampling", {}))
    for key in budgets:
        if getattr(args, key) is not None:
            params[key] = getattr(args, key)
    sampling = GraspSamplingCfg(**params)
    base = args.config.resolve().parent
    mesh = _asset(args.mesh or config.get("mesh"), Path.cwd() if args.mesh else base)
    urdf = args.gripper_urdf or config.get("gripper_urdf")
    if urdf:
        hand = _asset(urdf, Path.cwd() if args.gripper_urdf else base)
    elif model := config["gripper"].get("model"):
        preset = resolve_robot_cfg(model, variant=config["gripper"].get("variant"))
        hand = Path(preset.fpath or preset.urdf_cfg.assemble_urdf()).resolve(
            strict=True
        )
    else:
        parser.error(
            "A custom gripper description requires --gripper-urdf or gripper_urdf in YAML"
        )
    checkpoints = args.output.with_suffix(".checkpoints")
    label_grasps(
        config,
        sampling,
        mesh,
        hand,
        args.output,
        checkpoints,
        num_envs,
        args.resume,
    )
