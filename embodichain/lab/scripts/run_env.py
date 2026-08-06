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

from __future__ import annotations

import argparse
import os
import select
import sys
import time

from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import gymnasium
import numpy as np
import torch
import tqdm

from embodichain.lab.gym.envs.wrapper import ReplayWrapper
from embodichain.lab.gym.utils.gym_utils import (
    add_env_launcher_args_to_parser,
    build_env_cfg_from_args,
    load_trajectory,
)
from embodichain.lab.gym.utils.registration import (
    discover_task_packages,
    execute_init_hooks,
)
from embodichain.utils.logger import log_warning, log_info, log_error


def generate_and_execute_action_list(
    env: gymnasium.Env,
    idx: int,
    debug_mode: bool,
    *,
    episode_idx: int = 0,
    **kwargs: object,
) -> bool:
    """Generate and execute one demonstration action list.

    Args:
        env: Environment used to generate and execute the actions.
        idx: Index of the action list within the current episode.
        debug_mode: Whether debug mode is enabled.
        episode_idx: Index of the current episode.
        **kwargs: Additional arguments forwarded to action generation.

    Returns:
        Whether a non-empty action list was generated and executed.
    """

    action_list = env.get_wrapper_attr("create_demo_action_list")(
        action_sentence=idx, **kwargs
    )

    if action_list is None or len(action_list) == 0:
        log_warning("Action is invalid. Skip to next generation.")
        return False

    for action in tqdm.tqdm(
        action_list,
        desc=f"Executing episode #{episode_idx}, action list #{idx}",
        unit="step",
    ):
        # Step the environment with the current action
        # The environment will automatically detect truncation based on action_length
        obs, reward, terminated, truncated, info = env.step(action)

    # TODO: We may assume in export demonstration rollout, there is no truncation from the env.
    # but truncation is useful to improve the generation efficiency.

    return True


def generate_function(
    env,
    num_traj,
    time_id: int = 0,
    save_path: str = "",
    save_video: bool = False,
    debug_mode: bool = False,
    **kwargs,
):
    """Generate and execute a sequence of actions in the environment.

    This function resets the environment, generates and executes action trajectories,
    collects data, and optionally saves videos of the episodes. It supports both online
    and offline data generation modes.

    Args:
        env: The environment instance.
        num_traj (int): Number of trajectories to generate per episode.
        time_id (int, optional): Identifier for the current time step or episode.
        save_path (str, optional): Path to save generated videos.
        save_video (bool, optional): Whether to save episode videos.
        debug_mode (bool, optional): Enable debug mode for visualization and logging.
        **kwargs: Additional keyword arguments for data generation.

    Returns:
        bool: True if data generation is successful, False otherwise.
    """

    valid = True
    _, _ = env.reset()
    while True:
        ret = []
        for trajectory_idx in range(num_traj):
            valid = generate_and_execute_action_list(
                env,
                trajectory_idx,
                debug_mode,
                episode_idx=time_id,
                **kwargs,
            )

            if not valid:
                # Failed execution: reset without saving invalid data
                _, _ = env.reset(options={"save_data": False})
                break

        if valid:
            break
        else:
            log_warning("Reset valid flag to True.")
            valid = True

    return True


def replay(env, trajectory_path: str, mode: str = "kinematic") -> None:
    """Replay a recorded trajectory.

    Args:
        env: The environment built from the same config that recorded the
            trajectory (wrapped via :class:`ReplayWrapper`).
        trajectory_path: Path to the ``.pt`` trajectory file.
        mode: ``"kinematic"`` (exact, physics off), ``"dynamic"`` (feed recorded
            actions, physics on), or ``"control"`` (interactive kinematic scrubber).
    """
    data = load_trajectory(trajectory_path)
    meta = data["meta"]
    lengths = meta.get("lengths", [meta["num_steps"]] * meta["num_envs"])
    log_info(
        f"Replaying trajectory: num_envs={meta['num_envs']}, lengths={lengths}, "
        f"num_steps={meta['num_steps']}, mode={mode}",
        color="green",
    )
    replay_env = ReplayWrapper(env.unwrapped, trajectory_path, mode=mode)
    try:
        if mode == "control":
            replay_control(replay_env)
        else:
            replay_auto(replay_env, mode)
    finally:
        replay_env.close()


def replay_auto(replay_env: ReplayWrapper, mode: str) -> None:
    """Auto-replay the full trajectory with a progress bar."""
    num_steps = int(replay_env._lengths.min().item())
    replay_env.reset()
    max_err = 0.0
    rec_states = replay_env._trajectory["states"]
    for i in tqdm.tqdm(range(num_steps), desc=f"Replaying ({mode})", unit="step"):
        obs, reward, term, trunc, info = replay_env.step(None)
        if mode == "kinematic":
            st = min(i, num_steps - 1)
            err = (
                (replay_env.env.robot.get_qpos() - rec_states["robot"]["qpos"][:, st])
                .abs()
                .max()
                .item()
            )
            max_err = max(max_err, err)
        if bool(trunc.all()):
            break
    if mode == "kinematic":
        log_info(
            f"Replay complete ({num_steps} steps). Max state error vs recorded: {max_err:.6e}",
            color="green",
        )
    else:
        log_info(f"Replay complete ({num_steps} steps).", color="green")


class _ReplayControlInput:
    """Read replay-control commands immediately when stdin is a terminal."""

    def __init__(self):
        self._fd = None
        self._term_attrs = None
        self.single_key = False

    def __enter__(self):
        if not sys.stdin.isatty():
            return self
        self._fd = sys.stdin.fileno()
        if os.name == "nt":
            self.single_key = True
            return self

        import termios
        import tty

        self._term_attrs = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self.single_key = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._term_attrs is not None and self._fd is not None:
            import termios

            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._term_attrs)

    def read_key(self, timeout: float | None = None) -> str | None:
        """Read one key, or return ``None`` when the timeout expires."""
        if os.name == "nt" and self.single_key:
            import msvcrt

            if timeout is None:
                return msvcrt.getwch().lower()
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if msvcrt.kbhit():
                    return msvcrt.getwch().lower()
                time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
            return None

        if timeout is not None:
            ready, _, _ = select.select([sys.stdin], [], [], timeout)
            if not ready:
                return None

        if self.single_key:
            value = sys.stdin.read(1)
        else:
            value = sys.stdin.readline()
        if value == "":
            raise EOFError
        return value.lower() if self.single_key else value.strip().lower()

    @contextmanager
    def suspend_terminal(self) -> Iterator[None]:
        """Restore canonical terminal input while an embedded REPL is active."""
        if self._term_attrs is None or self._fd is None:
            yield
            return

        import termios
        import tty

        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._term_attrs)
        try:
            yield
        finally:
            tty.setcbreak(self._fd)


def _read_replay_control_command(
    control_input: _ReplayControlInput, initial: str | None = None
) -> str:
    """Read a command, collecting multi-digit jump targets until Enter."""
    command = control_input.read_key() if initial is None else initial
    if command is None:
        return ""
    if not control_input.single_key or not command.isdigit():
        return command

    digits = command
    print(digits, end="", flush=True)
    while True:
        key = control_input.read_key()
        if key in ("\r", "\n"):
            print()
            return digits
        if key in ("\b", "\x7f"):
            if digits:
                digits = digits[:-1]
                print("\b \b", end="", flush=True)
            continue
        if key.isdigit():
            digits += key
            print(key, end="", flush=True)


def _run_replay_control_loop(
    replay_env: ReplayWrapper, control_input: _ReplayControlInput
) -> None:
    """Run the interactive replay loop using the provided input source."""
    num_steps = int(replay_env._lengths.min().item())
    max_step = num_steps - 1
    step = 0
    replay_env.go_to_step(step)
    dt = (
        float(replay_env.env.sim_cfg.physics_dt)
        * replay_env.env.cfg.sim_steps_per_control
    )
    pending_command = None
    auto_playing = False

    print(f"Trajectory has {num_steps} steps (0..{max_step}).")
    while True:
        if auto_playing:
            if step >= max_step:
                auto_playing = False
                print(f"\nAuto replay finished at step {step}.")
                continue

            step += 1
            replay_env.go_to_step(step)
            print(
                f"\r[auto step {step}/{max_step}]  press any key to pause",
                end="",
                flush=True,
            )
            try:
                key = control_input.read_key(timeout=dt)
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if key is None:
                continue

            auto_playing = False
            print(f"\nPaused at step {step}.")
            # Pause keys are consumed. Other keys also pause and then execute
            # their normal command, so p/n/q/r remain responsive during auto.
            if key not in ("a", " ", "\r", "\n"):
                pending_command = key
            continue

        print(
            f"[step {step}/{max_step}]  n=next  p=prev  <N>=jump  "
            "a=auto  r=reset  q=quit"
        )
        if control_input.single_key:
            print("> ", end="", flush=True)
        try:
            command = _read_replay_control_command(
                control_input, initial=pending_command
            )
        except (EOFError, KeyboardInterrupt):
            break
        finally:
            pending_command = None
        if control_input.single_key and not command.isdigit():
            print()

        if command in ("q", "quit"):
            break
        if command in ("n", ""):
            step = min(step + 1, max_step)
        elif command in ("p", "b"):
            step = max(step - 1, 0)
        elif command == "r":
            step = 0
        elif command == "a":
            auto_playing = True
            continue
        elif command.isdigit():
            step = max(0, min(int(command), max_step))
        elif command == " ":
            continue
        else:
            print(f"Unknown command: {command!r}")
            continue
        replay_env.go_to_step(step)


def replay_control(replay_env: ReplayWrapper) -> None:
    """Run an interactive, single-key kinematic trajectory scrubber."""
    if replay_env.env.sim_cfg.headless:
        log_warning(
            "control mode with --headless: no window to view the scrub. "
            "Re-run without --headless to see the replay."
        )
    replay_env.reset()
    with _ReplayControlInput() as control_input:
        _run_replay_control_loop(replay_env, control_input)


def main(args, env, gym_config):
    if getattr(args, "replay", False):
        log_info("Replay mode.", color="green")
        replay(
            env,
            args.replay_trajectory,
            getattr(args, "replay_mode", "kinematic"),
        )
        return

    if getattr(args, "preview", False):
        log_info(
            "Preview mode enabled. Launching environment preview...", color="green"
        )
        preview(env)
        return

    log_info("Start offline data generation.", color="green")
    # TODO: Support multiple trajectories per episode generation.
    num_traj = 1
    try:
        for i in range(gym_config.get("max_episodes", 1)):
            generate_function(
                env,
                num_traj,
                i,
                save_path=getattr(args, "save_path", ""),
                save_video=getattr(args, "save_video", False),
                debug_mode=getattr(args, "debug_mode", False),
                regenerate=getattr(args, "regenerate", False),
            )

        # Final reset (saves the last completed episode).
        _, _ = env.reset()

        # Log the trajectory save location BEFORE env.close() (in the finally
        # below) tears down the sim and, by default, os._exit()s the process.
        if getattr(args, "record_trajectory", False):
            save_dir = args.trajectory_save_dir
            if save_dir is None:
                import os

                from embodichain.data.constants import EMBODICHAIN_DEFAULT_DATA_ROOT

                save_dir = os.path.join(
                    EMBODICHAIN_DEFAULT_DATA_ROOT,
                    "trajectories",
                    env.unwrapped._traj_run_id,
                )
            log_info(
                f"Trajectories recorded to: {save_dir} "
                "(replay with --replay --replay_trajectory <path>)",
                color="green",
            )
    finally:
        # Drain the dataset recorder and finalize the LeRobot dataset before
        # the process exits. This is REQUIRED for AsyncLeRobotRecorder: its
        # background worker only *enqueues* episodes during reset, so without
        # close() the worker is killed at exit and no data reaches disk.
        # env.close() -> dataset_manager.finalize() drains the worker + flushes
        # meta/stats, then sim.destroy() tears down the sim. sim.destroy() exits
        # the process without returning, so this MUST be the last thing main()
        # does.
        env.close()


def _enable_preview_ik_gizmos(
    env: gymnasium.Env,
) -> tuple[tuple[str, str], ...]:
    """Create hidden native IK Gizmos for the preview robot.

    Only control parts selected by the environment and backed by an IK solver
    are enabled. Existing Gizmos are reused without changing their visibility.

    Args:
        env: Gymnasium environment being previewed.

    Returns:
        ``(robot_uid, control_part)`` pairs for the available IK Gizmos.
    """
    base_env = env.unwrapped
    sim = getattr(base_env, "sim", None)
    robot = getattr(base_env, "robot", None)
    if sim is None or robot is None:
        log_warning("Preview IK Gizmo is unavailable because no robot was found.")
        return ()
    if not bool(getattr(sim, "is_window_opened", False)):
        log_warning(
            "Preview IK Gizmo requires a native DexSim window and is disabled "
            "for headless or Viser preview."
        )
        return ()

    num_envs = getattr(base_env, "num_envs", None)
    if num_envs is None:
        num_envs = getattr(robot, "num_instances", 1)
    if int(num_envs) != 1:
        log_warning(
            "Preview IK Gizmo supports exactly one environment; "
            f"received num_envs={num_envs}."
        )
        return ()

    robot_uid = getattr(robot, "uid", None)
    control_parts = getattr(robot, "control_parts", None) or {}
    get_solver = getattr(robot, "get_solver", None)
    if not isinstance(robot_uid, str) or not robot_uid or not callable(get_solver):
        log_warning("Preview IK Gizmo requires a named robot with IK control parts.")
        return ()

    configured_parts = getattr(getattr(base_env, "cfg", None), "control_parts", None)
    if configured_parts:
        candidate_parts = tuple(
            dict.fromkeys(part for part in configured_parts if part in control_parts)
        )
    else:
        candidate_parts = tuple(control_parts)
    ik_parts = tuple(part for part in candidate_parts if get_solver(part) is not None)
    if not ik_parts:
        log_warning(
            f"Robot {robot_uid!r} has no active control part with an IK solver; "
            "preview IK Gizmo was not enabled."
        )
        return ()

    gizmo_keys: list[tuple[str, str]] = []
    for control_part in ik_parts:
        if sim.has_gizmo(robot_uid, control_part=control_part):
            gizmo_keys.append((robot_uid, control_part))
            continue
        gizmo = sim.enable_gizmo(
            uid=robot_uid,
            control_part=control_part,
            enable_native=True,
        )
        if gizmo is None:
            continue
        # Preview starts view-only. The native IKGizmoController owns the I
        # hotkey and reveals all newly created targets on the first key press.
        sim.set_gizmo_visibility(
            robot_uid,
            visible=False,
            control_part=control_part,
        )
        gizmo_keys.append((robot_uid, control_part))

    if gizmo_keys:
        part_names = ", ".join(part for _, part in gizmo_keys)
        log_info(
            f"Preview IK Gizmo ready for {robot_uid!r}: {part_names}. "
            "Focus the DexSim window and press I to show or hide it.",
            color="green",
        )
    else:
        log_warning(f"Failed to initialize a preview IK Gizmo for {robot_uid!r}.")
    return tuple(gizmo_keys)


def _toggle_preview_ik_gizmos(
    sim: object,
    gizmo_keys: Sequence[tuple[str, str]],
) -> tuple[bool, ...]:
    """Toggle preview IK Gizmos from the terminal fallback command."""
    states: list[bool] = []
    for robot_uid, control_part in gizmo_keys:
        visible = sim.toggle_gizmo_visibility(
            robot_uid,
            control_part=control_part,
        )
        if visible is not None:
            states.append(bool(visible))
    return tuple(states)


def _run_preview_loop(
    env: gymnasium.Env,
    control_input: _ReplayControlInput,
    gizmo_keys: Sequence[tuple[str, str]],
) -> None:
    """Run terminal commands while servicing interactive Gizmos."""
    sim = env.unwrapped.sim
    physics_dt = float(sim.sim_config.physics_dt)
    visualization = getattr(sim.sim_config, "visualization", None)
    service_interactions = bool(gizmo_keys) or (
        getattr(visualization, "backend", "none") == "viser"
    )

    print("Preview controls:")
    if gizmo_keys:
        print("  DexSim window: I=show/hide IK Gizmo, drag Gizmo=move robot")
        print("  Terminal: i=show/hide IK Gizmo")
    print("  Terminal: p=IPython embed, q=quit")

    while True:
        try:
            command = control_input.read_key(
                timeout=physics_dt if service_interactions else None
            )
        except (EOFError, KeyboardInterrupt):
            break

        if command is not None:
            command = command.strip().lower()
            if command in {"q", "quit"}:
                break
            if command == "p":
                try:
                    from IPython import embed
                except ImportError:
                    log_error(
                        "IPython is not installed. Preview embed mode requires "
                        "IPython. Install it with `pip install ipython`."
                    )
                    continue
                with control_input.suspend_terminal():
                    embed()
            elif command == "i" and gizmo_keys:
                states = _toggle_preview_ik_gizmos(sim, gizmo_keys)
                if states:
                    state = "shown" if all(states) else "hidden"
                    log_info(f"Preview IK Gizmo {state}.", color="green")
            elif command:
                print(f"Unknown preview command: {command!r}")

        if service_interactions:
            # SimulationManager.update() invokes update_gizmos() before the
            # physics step, allowing native or Viser controllers to apply IK
            # drive targets while the terminal remains responsive.
            sim.update(physics_dt, step=1)


def preview(env: gymnasium.Env) -> None:
    """Run an interactive environment preview.

    A native single-environment preview automatically creates hidden IK
    Gizmos for the robot's active solver-backed control parts. Press ``I`` in
    the DexSim window to show or hide the controls, then drag an end-effector
    target to operate the robot. Terminal commands remain available for the
    IPython embed session and shutdown.

    Args:
        env: Gymnasium environment to reset and preview.
    """
    _, _ = env.reset()
    gizmo_keys = _enable_preview_ik_gizmos(env)
    with _ReplayControlInput() as control_input:
        _run_preview_loop(env, control_input, gizmo_keys)


def _create_parser() -> argparse.ArgumentParser:
    """Create the ``run-env`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="embodichain run-env",
        description="Run an environment for data generation or interactive preview.",
    )

    add_env_launcher_args_to_parser(parser, require_gym_config=True)
    parser.set_defaults(viser_image_fps=None)

    parser.add_argument(
        "--replay",
        action="store_true",
        help="Replay a recorded trajectory (--replay_trajectory required).",
    )
    parser.add_argument(
        "--replay_trajectory",
        type=str,
        default=None,
        help="Path to the .pt trajectory file to replay.",
    )
    parser.add_argument(
        "--replay_mode",
        type=str,
        choices=["kinematic", "dynamic", "control"],
        default="kinematic",
        help="Replay mode: kinematic (exact, default), dynamic (physics), "
        "control (interactive scrubber).",
    )
    return parser


def cli(argv: Sequence[str] | None = None) -> None:
    """Command-line interface for environment runner.

    Parses CLI arguments, builds the environment config, and launches
    the data generation, preview, or replay workflow.

    Args:
        argv: Arguments excluding the command name. Uses ``sys.argv`` when
            omitted.
    """
    np.set_printoptions(5, suppress=True)
    torch.set_printoptions(precision=5, sci_mode=False)

    args = _create_parser().parse_args(argv)

    if getattr(args, "replay", False):
        if not args.replay_trajectory:
            log_error("--replay requires --replay_trajectory <path>.")
            return
        if getattr(args, "preview", False):
            log_error("--replay and --preview are mutually exclusive.")
            return

    # Step 1: Discover all task packages via entry_points
    discover_task_packages()

    # Step 2: Execute init hooks (register managers, asset resolvers, etc.)
    execute_init_hooks()

    env_cfg, gym_config, action_config = build_env_cfg_from_args(args)

    if args.replay and args.replay_mode == "control":
        log_info("Dataset saving disabled for control replay mode.", color="green")

    env = gymnasium.make(id=gym_config["id"], cfg=env_cfg, **action_config)

    # Ensure the sim is torn down via env.close() (-> SimulationManager.destroy())
    # before the interpreter shuts down. Without this, C++ resources (dexsim/warp/
    # CUDA) are finalized during Python shutdown in an unpredictable order, which
    # segfaults on exit (exit code 139). ``destroy()`` queues a deferred cleanup
    # task and, by default (EMBODICHAIN_SIM_EXIT_PROCESS=1), calls ``os._exit(0)``
    # to skip the unsafe teardown entirely. When that env var is disabled (e.g.
    # dev/test), ``flush_cleanup_queue`` drains the queue and runs the deferred
    # destruction + GC + scene-barrier so we still exit cleanly.
    try:
        main(args, env, gym_config)
    finally:
        try:
            env.close()
        except Exception as e:
            log_warning(f"Failed to close environment: {e}")
        try:
            from embodichain.lab.sim.sim_manager import SimulationManager

            SimulationManager.flush_cleanup_queue()
        except Exception as e:
            log_warning(f"Failed to flush simulation cleanup queue: {e}")


if __name__ == "__main__":
    cli()


__all__ = [
    "cli",
    "generate_and_execute_action_list",
    "generate_function",
    "main",
    "preview",
]
