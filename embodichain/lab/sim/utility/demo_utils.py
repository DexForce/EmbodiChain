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

"""Shared utilities for simulation demo scripts."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from collections.abc import Sequence

    from embodichain.lab.sim import SimulationManager


__all__ = [
    "add_demo_args",
    "create_default_sim",
    "shutdown_sim",
    "setup_print_options",
    "format_tensor",
    "maybe_init_gpu_physics",
]


def add_demo_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add common demo arguments to an environment launcher parser.

    Args:
        parser: The parser to extend.

    Returns:
        The same parser with demo flags added.
    """
    from embodichain.lab.gym.utils.gym_utils import add_env_launcher_args_to_parser

    add_env_launcher_args_to_parser(parser)
    parser.add_argument(
        "--auto_play",
        action="store_true",
        help="Skip interactive prompts and run the demo automatically.",
    )
    parser.add_argument(
        "--record_steps",
        type=int,
        default=None,
        help="Number of simulation steps to record. If None, no recording is started.",
    )
    parser.add_argument(
        "--record_fps",
        type=int,
        default=30,
        help="Frames per second for the recorded video.",
    )
    parser.add_argument(
        "--record_save_path",
        type=str,
        default=None,
        help="Directory to save recorded videos. Defaults to ./recordings.",
    )
    parser.add_argument(
        "--no_vis_eef_axis",
        action="store_true",
        help="Disable end-effector axis visualization.",
    )
    return parser


def create_default_sim(
    args: argparse.Namespace,
    *,
    width: int = 1920,
    height: int = 1080,
    physics_dt: float = 1.0 / 100.0,
    arena_space: float = 2.5,
    add_default_light: bool = True,
) -> SimulationManager:
    """Create a SimulationManager with common demo defaults.

    Args:
        args: Parsed command-line arguments. Expected to contain ``headless``,
            ``device``, ``renderer`` and ``arena_space``.
        width: Window/render width.
        height: Window/render height.
        physics_dt: Physics simulation timestep.
        arena_space: Arena space size.
        add_default_light: Whether to add a default point light.

    Returns:
        Configured simulation manager instance.
    """
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import LightCfg, RenderCfg

    cfg = SimulationManagerCfg(
        width=width,
        height=height,
        headless=args.headless,
        sim_device=args.device,
        render_cfg=RenderCfg(renderer=args.renderer),
        physics_dt=physics_dt,
        arena_space=arena_space,
    )
    sim = SimulationManager(cfg)
    if add_default_light:
        sim.add_light(
            cfg=LightCfg(
                uid="main_light",
                color=(0.6, 0.6, 0.6),
                intensity=30.0,
                init_pos=(1.0, 0.0, 3.0),
            )
        )
    return sim


def shutdown_sim(sim: SimulationManager) -> None:
    """Safely destroy a simulation manager.

    Args:
        sim: The simulation manager to destroy.
    """
    sim.destroy()


def setup_print_options() -> None:
    """Set common numpy and torch print options for demos."""
    np.set_printoptions(precision=5, suppress=True)
    torch.set_printoptions(precision=5, sci_mode=False)


def format_tensor(tensor: torch.Tensor) -> str:
    """Return a compact, rounded string representation of a tensor.

    Args:
        tensor: Input tensor.

    Returns:
        Rounded string with 4 decimal places.
    """
    values = tensor.detach().cpu().tolist()
    return "[" + ", ".join(f"{v:.4f}" for v in values) + "]"


def maybe_init_gpu_physics(sim: SimulationManager) -> None:
    """Initialize GPU physics if the simulation is configured to use it.

    Args:
        sim: The simulation manager.
    """
    if sim.is_use_gpu_physics:
        sim.init_gpu_physics()
