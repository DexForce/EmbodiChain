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
import time
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from collections.abc import Sequence

    from embodichain.lab.sim import SimulationManager
    from embodichain.lab.sim.objects import Robot


__all__ = [
    "add_demo_args",
    "create_default_sim",
    "shutdown_sim",
    "setup_print_options",
    "format_tensor",
    "maybe_init_gpu_physics",
    "DemoRecording",
    "maybe_open_window",
    "maybe_wait_for_user",
    "maybe_pause_for_inspection",
    "replay_trajectory",
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


class DemoRecording:
    """Context manager that handles demo video recording.

    Recording is only started when ``args.record_steps`` is not ``None``.
    On exit the window record is stopped and the framework is asked to finish
    saving the video file.

    Args:
        sim: The simulation manager.
        args: Parsed command-line arguments. Expected to contain
            ``record_steps``, ``record_fps`` and ``record_save_path``.
        prefix: Prefix used for the generated video filename.
        look_at: Optional camera look-at tuple for the recording.
    """

    def __init__(
        self,
        sim: SimulationManager,
        args: argparse.Namespace,
        prefix: str = "demo",
        look_at: tuple[Sequence[float], Sequence[float], Sequence[float]] | None = None,
    ):
        self.sim = sim
        self.args = args
        self.prefix = prefix
        self.look_at = look_at
        self.is_active = False

    def __enter__(self) -> DemoRecording:
        """Start recording if requested."""
        if self.args.record_steps is None:
            return self

        import datetime
        import warnings
        from pathlib import Path

        save_dir = Path(self.args.record_save_path or "./recordings")
        save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = str(save_dir / f"{self.prefix}_{timestamp}.mp4")

        original_width = self.sim.sim_config.width
        original_height = self.sim.sim_config.height
        try:
            # Use a smaller resolution for recording to keep files small.
            self.sim.sim_config.width = 640
            self.sim.sim_config.height = 480
            started = self.sim.start_window_record(
                save_path=save_path,
                fps=self.args.record_fps,
                max_memory=2048,
                video_prefix=self.prefix,
                look_at=self.look_at,
                use_sim_time=True,
            )
        finally:
            self.sim.sim_config.width = original_width
            self.sim.sim_config.height = original_height

        if not started:
            warnings.warn(
                f"Failed to start recording for prefix '{self.prefix}'. Continuing without recording.",
                UserWarning,
                stacklevel=2,
            )
            return self

        self.is_active = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Stop recording and wait for the file to be written."""
        if not self.is_active:
            return
        if self.sim.is_window_recording():
            self.sim.stop_window_record()
        self.sim.wait_window_record_saves()


def maybe_open_window(sim: SimulationManager, args: argparse.Namespace) -> None:
    """Open the viewer window unless running headless.

    Args:
        sim: The simulation manager.
        args: Parsed arguments containing ``headless``.
    """
    if not args.headless:
        sim.open_window()


def maybe_wait_for_user(args: argparse.Namespace, prompt: str) -> None:
    """Wait for user input unless auto_play is enabled.

    Args:
        args: Parsed arguments containing ``auto_play``.
        prompt: Message to display when waiting.
    """
    if not args.auto_play:
        input(prompt)


def maybe_pause_for_inspection(args: argparse.Namespace) -> None:
    """Pause at the end of a demo for visual inspection.

    Args:
        args: Parsed arguments containing ``auto_play``.
    """
    maybe_wait_for_user(args, "Demo finished. Press Enter to exit...")


def replay_trajectory(
    sim: SimulationManager,
    robot: Robot,
    traj: torch.Tensor,
    *,
    post_steps: int = 60,
    step_size: int = 4,
    sleep: float = 1e-2,
    arm_name: str | None = None,
) -> None:
    """Replay a joint-space trajectory on a robot.

    ``traj`` may be either a 1-D tensor of shape ``(num_joints,)``, a 2-D
    tensor of shape ``(num_steps, num_joints)`` or a 3-D tensor of shape
    ``(batch, num_steps, num_joints)``. For 1-D input the single
    configuration is held for ``post_steps``. For 2-D/3-D input each step is
    applied sequentially and the final configuration is held.

    Args:
        sim: The simulation manager.
        robot: The robot instance.
        traj: Joint position trajectory tensor.
        post_steps: Number of steps to hold the final configuration.
        step_size: Number of physics steps per ``sim.update()`` call.
        sleep: Sleep duration between steps (seconds).
        arm_name: Optional arm name passed to ``robot.set_qpos``.
    """
    if traj.dim() == 1:
        traj = traj.unsqueeze(0).unsqueeze(0)
    elif traj.dim() == 2:
        traj = traj.unsqueeze(0)

    joint_ids = robot.get_joint_ids(arm_name) if arm_name is not None else None

    for i in range(traj.shape[1]):
        robot.set_qpos(qpos=traj[:, i, :], joint_ids=joint_ids)
        sim.update(step=step_size)
        time.sleep(sleep)

    final_qpos = traj[:, -1, :]
    for _ in range(post_steps):
        robot.set_qpos(qpos=final_qpos, joint_ids=joint_ids)
        sim.update(step=2)
        time.sleep(sleep)
