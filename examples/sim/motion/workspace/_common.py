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

"""Shared setup for the three W1 workspace API examples."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import torch

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import MarkerCfg
from embodichain.lab.sim.objects import Robot
from embodichain.lab.visualization import (
    add_viser_args_to_parser,
    visualization_cfg_from_args,
)
from embodichain_robots import DexforceW1Cfg


def parse_args(description: str, *, samples: int, device: str) -> argparse.Namespace:
    """Parse common viewer and sample controls while preserving mode defaults."""
    parser = argparse.ArgumentParser(description=description)
    add_viser_args_to_parser(parser)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=samples,
        help=f"Number of samples (default: {samples}).",
    )
    parser.add_argument(
        "--sim-device",
        choices=("cpu", "cuda"),
        default=device,
        help=f"Simulation and computation device (default: {device}).",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Exit after analysis instead of keeping the visualization open.",
    )
    args = parser.parse_args()
    if args.num_samples <= 0:
        parser.error("--num-samples must be greater than zero")
    return args


@contextmanager
def workspace_scene(
    args: argparse.Namespace, *, reference_pose: bool = False
) -> Iterator[tuple[SimulationManager, Robot, torch.Tensor | None]]:
    """Own one arms-only W1 scene and its cleanup."""
    np.set_printoptions(precision=5, suppress=True)
    torch.set_printoptions(precision=5, sci_mode=False)
    window = {"width": 1080, "height": 1080} if reference_pose else {}
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=False,
            sim_device=args.sim_device,
            visualization=visualization_cfg_from_args(args),
            **window,
        )
    )
    try:
        robot = sim.add_robot(
            cfg=DexforceW1Cfg.from_dict(
                {
                    "uid": "dexforce_w1",
                    "version": "v021",
                    "with_default_eef": False,
                }
            )
        )
        sim.prepare()
        print("DexforceW1 robot added to the simulation.")
        pose = None
        if reference_pose:
            left_qpos = torch.tensor(
                [0, -np.pi / 4, 0.0, -np.pi / 2, -np.pi / 4, 0.0, 0.0],
                dtype=torch.float32,
                device=robot.device,
            )
            robot.set_qpos(qpos=left_qpos, joint_ids=robot.get_joint_ids("left_arm"))
            robot.set_qpos(qpos=-left_qpos, joint_ids=robot.get_joint_ids("right_arm"))
            pose = robot.compute_fk(qpos=left_qpos, name="left_arm", to_matrix=True)
            sim.draw_marker(
                cfg=MarkerCfg(
                    name="left_arm_pose_axis",
                    marker_type="axis",
                    axis_xpos=pose,
                    axis_size=0.005,
                    axis_len=0.15,
                    arena_index=0,
                )
            )
        yield sim, robot, pose
    finally:
        sim.destroy()
        SimulationManager.flush_cleanup_queue()


def show_results(
    sim: SimulationManager,
    results: dict,
    *,
    title: str,
    count_key: str,
    count_label: str,
    no_wait: bool,
) -> None:
    """Print analysis metrics and keep the selected visualization alive."""
    print(f"\n{title}:")
    print(f"  {count_label}: {results[count_key]} / {results['num_samples']}")
    print(f"  Analysis time: {results['analysis_time']:.2f}s")
    print(f"  Metrics: {results['metrics']}")
    sim.capture_visualization(force=True)
    if not no_wait:
        print("Workspace visualization is ready. Press Ctrl+C to exit.")
        try:
            while True:
                sim.update(step=1)
        except KeyboardInterrupt:
            pass
