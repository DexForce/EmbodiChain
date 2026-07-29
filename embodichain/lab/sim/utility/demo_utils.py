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
    from collections.abc import Callable, Sequence

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
    "DEFAULT_DEMO_LOOK_AT",
    "resolve_demo_steps",
    "run_simulation_loop",
    "replay_trajectory",
]

DEFAULT_DEMO_LOOK_AT = (
    (2.6, -2.2, 1.6),
    (0.0, 0.0, 0.45),
    (0.0, 0.0, 1.0),
)


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
        "--auto-play",
        action="store_true",
        help="Skip interactive prompts and run the demo automatically.",
    )
    parser.add_argument(
        "--record_steps",
        "--record-steps",
        type=_positive_int,
        default=None,
        help=(
            "Number of simulation updates to record. Continuous demos also use "
            "this as their run limit. If omitted, recording is disabled."
        ),
    )
    parser.add_argument(
        "--record_fps",
        "--record-fps",
        type=_positive_int,
        default=30,
        help="Frames per second for the recorded video.",
    )
    parser.add_argument(
        "--record_save_path",
        "--record-save-path",
        type=str,
        default=None,
        help=(
            "Output .mp4 path or directory for recorded videos. "
            "Defaults to ./recordings."
        ),
    )
    parser.add_argument(
        "--no_vis_eef_axis",
        action="store_true",
        help="Disable end-effector axis visualization.",
    )
    return parser


def _positive_int(value: str) -> int:
    """Parse a strictly positive integer for an argparse option."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def create_default_sim(
    args: argparse.Namespace,
    *,
    width: int = 1920,
    height: int = 1080,
    physics_dt: float = 1.0 / 100.0,
    arena_space: float = 2.5,
    num_envs: int = 1,
    add_default_light: bool = True,
) -> SimulationManager:
    """Create a SimulationManager with common demo defaults.

    Args:
        args: Parsed command-line arguments. Expected to contain ``headless``,
            ``device`` and ``renderer``.
        width: Window/render width.
        height: Window/render height.
        physics_dt: Physics simulation timestep.
        arena_space: Arena space size.
        num_envs: Number of parallel environments to simulate.
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
        num_envs=num_envs,
        gpu_id=getattr(args, "gpu_id", 0),
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
    # Recording owns renderer resources and must finish before teardown. Use
    # attribute checks so this helper remains useful with lightweight test
    # doubles and older SimulationManager implementations.
    is_recording = getattr(sim, "is_window_recording", None)
    try:
        if callable(is_recording) and is_recording():
            sim.stop_window_record()
            sim.wait_window_record_saves()
    finally:
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
        look_at: Optional camera look-at tuple for the recording. Headless
            recording uses :data:`DEFAULT_DEMO_LOOK_AT` when omitted.
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
        self.look_at = (
            DEFAULT_DEMO_LOOK_AT
            if look_at is None and getattr(args, "headless", False)
            else look_at
        )
        self.is_active = False

    def __enter__(self) -> DemoRecording:
        """Start recording if requested."""
        if self.args.record_steps is None:
            return self

        import datetime
        import warnings
        from pathlib import Path

        requested_path = Path(self.args.record_save_path or "./recordings")
        if requested_path.suffix.lower() == ".mp4":
            requested_path.parent.mkdir(parents=True, exist_ok=True)
            save_path = str(requested_path)
        else:
            requested_path.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = str(requested_path / f"{self.prefix}_{timestamp}.mp4")

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


def resolve_demo_steps(
    args: argparse.Namespace,
    *,
    auto_play_steps: int = 300,
) -> int | None:
    """Resolve the run limit for a continuous demo.

    Explicit ``--record_steps`` takes precedence. ``--auto_play`` uses a
    finite default so non-interactive smoke runs terminate on their own.
    Interactive runs remain open until interrupted.

    Args:
        args: Parsed demo arguments.
        auto_play_steps: Default update count used by ``--auto_play``.

    Returns:
        Maximum number of updates, or ``None`` for an interactive run.

    Raises:
        ValueError: If ``auto_play_steps`` is not positive.
    """
    if auto_play_steps < 1:
        raise ValueError("auto_play_steps must be at least 1")
    record_steps = getattr(args, "record_steps", None)
    if record_steps is not None:
        return record_steps
    return auto_play_steps if getattr(args, "auto_play", False) else None


def run_simulation_loop(
    sim: SimulationManager,
    *,
    max_steps: int | None = None,
    steps_per_update: int = 1,
    sleep: float = 0.0,
    log_interval: int | None = 100,
    on_step: Callable[[int], None] | None = None,
) -> int:
    """Run a standard simulation update loop.

    The function intentionally does not destroy ``sim``; callers should use
    :class:`~embodichain.lab.sim.demo_base.DemoBase` or ``try/finally`` so
    setup failures and loop failures share the same cleanup path.

    Args:
        sim: Simulation manager to update.
        max_steps: Optional number of update calls before returning.
        steps_per_update: Physics steps advanced by each update call.
        sleep: Optional wall-clock delay after each update.
        log_interval: Print aggregate FPS every this many updates. Set to
            ``None`` to disable progress logging.
        on_step: Optional callback receiving the one-based update count.

    Returns:
        Number of completed update calls.

    Raises:
        ValueError: If a numeric loop option is outside its valid range.
    """
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if steps_per_update < 1:
        raise ValueError("steps_per_update must be at least 1")
    if sleep < 0:
        raise ValueError("sleep must be non-negative")
    if log_interval is not None and log_interval < 1:
        raise ValueError("log_interval must be at least 1")

    started_at = time.monotonic()
    last_log_at = started_at
    last_log_step = 0
    step_count = 0

    try:
        while max_steps is None or step_count < max_steps:
            sim.update(step=steps_per_update)
            step_count += 1
            if on_step is not None:
                on_step(step_count)
            if sleep:
                time.sleep(sleep)

            if log_interval is not None and step_count % log_interval == 0:
                now = time.monotonic()
                elapsed = now - last_log_at
                fps = (
                    sim.num_envs
                    * (step_count - last_log_step)
                    * steps_per_update
                    / elapsed
                    if elapsed > 0
                    else 0.0
                )
                print(f"[INFO]: Simulation step: {step_count}, FPS: {fps:.2f}")
                last_log_at = now
                last_log_step = step_count
    except KeyboardInterrupt:
        print("\n[INFO]: Stopping simulation...")

    return step_count


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
