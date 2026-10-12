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
"""Animate markers in every sub-environment and update selected environments.

Run with --viser --num_envs 4 to compare selected environment visibility and
clear/repopulation against unaffected environments. The world origin is drawn
once using a separate world-scope group.
"""

from __future__ import annotations

import argparse

from embodichain.cli.sim import add_sim_args_to_parser


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser without initializing simulation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--steps",
        type=int,
        default=900,
        help="Number of bounded animation steps to run.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Maximum host update rate used to pace the demonstration.",
    )
    add_sim_args_to_parser(parser)
    return parser


if __name__ == "__main__":
    _cli_args = build_parser().parse_args()


import math
import time

import numpy as np

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import RenderCfg, physics_cfg_for_backend
from embodichain.lab.visualization import visualization_cfg_from_args
from embodichain.lab.visualization.markers import (
    MarkerGroupCfg,
    MarkerPrototypeCfg,
)

_SHAPES = ("box", "sphere", "cylinder", "capsule", "cone", "arrow", "frame")
_COLORS = (
    (0.95, 0.20, 0.20, 0.85),
    (0.95, 0.55, 0.10, 0.85),
    (0.95, 0.90, 0.15, 0.85),
    (0.25, 0.80, 0.30, 0.85),
    (0.15, 0.65, 0.95, 0.85),
    (0.35, 0.30, 0.95, 0.85),
    (0.85, 0.25, 0.80, 0.85),
)


def _translations(phase: float) -> np.ndarray:
    """Return seven animated positions in one horizontal row."""
    x = np.linspace(-1.5, 1.5, len(_SHAPES), dtype=np.float32)
    z = 0.45 + 0.12 * np.sin(phase + np.arange(len(_SHAPES)))
    return np.column_stack((x, np.zeros_like(x), z)).astype(np.float32)


def main(args: argparse.Namespace | None = None) -> None:
    """Run the bounded marker lifecycle demonstration."""
    if args is None:
        args = build_parser().parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be at least 1.")
    if not math.isfinite(args.fps) or args.fps <= 0.0:
        raise ValueError("--fps must be a positive finite number.")

    sim = SimulationManager(
        SimulationManagerCfg(
            headless=args.headless,
            device=args.device,
            num_envs=args.num_envs,
            arena_space=args.arena_space,
            render_cfg=RenderCfg(renderer=args.renderer),
            physics_cfg=physics_cfg_for_backend(args.physics),
            visualization=visualization_cfg_from_args(args),
        )
    )
    try:
        marker = sim.add_marker_group(
            MarkerGroupCfg(
                name="built_in_markers",
                prototypes={
                    shape: MarkerPrototypeCfg(
                        shape=shape,
                        scale=(0.20, 0.20, 0.20),
                        color=color,
                    )
                    for shape, color in zip(_SHAPES, _COLORS, strict=True)
                },
            )
        )
        sim.prepare()
        if sim.is_use_gpu_physics:
            sim.init_gpu_physics()
        if not args.headless:
            sim.open_window()

        world_frame = sim.add_marker_group(
            MarkerGroupCfg(
                name="world_origin",
                scope="world",
                prototypes={
                    "frame": MarkerPrototypeCfg(shape="frame", scale=(0.4, 0.4, 0.4)),
                },
            )
        )
        world_frame.update(translations=[[0, 0, 0.02]])
        prototype_indices = np.broadcast_to(
            np.arange(len(_SHAPES), dtype=np.int64), (sim.num_envs, len(_SHAPES))
        ).copy()
        colors = np.broadcast_to(
            np.asarray(_COLORS, dtype=np.float32), (sim.num_envs, len(_SHAPES), 4)
        ).copy()

        def positions(phase: float) -> np.ndarray:
            return np.stack(
                [_translations(phase + env * 0.25) for env in range(sim.num_envs)]
            )

        selected_envs = np.arange(0, sim.num_envs, 2)
        other_envs = np.setdiff1d(np.arange(sim.num_envs), selected_envs)
        marker.update(
            translations=positions(0.0),
            prototype_indices=prototype_indices,
            colors=colors,
        )
        lifecycle_duration = max(1, args.steps // 10)
        hide_step = args.steps // 4
        show_step = min(hide_step + lifecycle_duration, args.steps - 1)
        clear_step = args.steps // 2
        repopulate_step = min(clear_step + lifecycle_duration, args.steps - 1)
        populated = True
        frame_period = 1.0 / args.fps
        next_frame = time.perf_counter()
        for step in range(args.steps):
            phase = step * 2.0 * math.pi / max(args.steps, 1)
            if step == hide_step:
                marker.set_visibility(False, env_ids=selected_envs)
            elif step == show_step:
                marker.set_visibility(True, env_ids=selected_envs)
            if step == clear_step:
                marker.clear(env_ids=selected_envs)
                populated = False
            elif step == repopulate_step:
                marker.update(
                    translations=positions(phase)[selected_envs],
                    prototype_indices=prototype_indices[selected_envs],
                    colors=colors[selected_envs],
                    env_ids=selected_envs,
                )
                populated = True
            elif populated:
                marker.update(translations=positions(phase))
            if not populated and len(other_envs):
                marker.update(
                    translations=positions(phase)[other_envs], env_ids=other_envs
                )

            # Marker updates are render-only; physics advances explicitly here.
            sim.update(step=1)
            next_frame += frame_period
            time.sleep(max(0.0, next_frame - time.perf_counter()))
        marker.remove()
        world_frame.remove()
    finally:
        sim.destroy()


if __name__ == "__main__":
    main(_cli_args)
