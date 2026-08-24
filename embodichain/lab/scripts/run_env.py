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
import json
import os
import select
import sys
import time

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

import gymnasium
import numpy as np
import torch
import tqdm

from embodichain.lab.gym.envs.demo import DemoEpisodeResult, execute_demo_episode
from embodichain.lab.gym.envs.expert_program.loader import (
    load_expert_program as _load_expert_program,
)
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

if TYPE_CHECKING:
    from embodichain.lab.visualization import VisualizationRuntime

_REPLAY_CONTROL_POLL_INTERVAL = 0.05


def _progress_wrapper(actions: Iterable[Any], description: str) -> Iterable[Any]:
    """Wrap a segment action iterable in the run-env progress bar."""
    return tqdm.tqdm(actions, desc=description, unit="step")


def _env_target(env: Any) -> Any:
    """Return the underlying environment used for lifecycle introspection."""
    return getattr(env, "unwrapped", env)


def _normalize_save_env_ids(
    env: Any,
    env_ids: Sequence[int] | torch.Tensor | None,
) -> tuple[int, ...]:
    """Validate environment rows selected for one persisted episode batch."""
    num_envs = int(getattr(_env_target(env), "num_envs", 1))
    if num_envs < 1:
        raise ValueError(f"env.num_envs must be at least 1, got {num_envs}.")

    if env_ids is None:
        normalized = tuple(range(num_envs))
    elif isinstance(env_ids, torch.Tensor):
        normalized = tuple(
            int(env_id) for env_id in env_ids.detach().cpu().reshape(-1).tolist()
        )
    else:
        normalized = tuple(int(env_id) for env_id in env_ids)

    if not normalized:
        raise ValueError("save_env_ids must select at least one environment.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"save_env_ids contains duplicates: {normalized}.")
    invalid = [env_id for env_id in normalized if not 0 <= env_id < num_envs]
    if invalid:
        raise ValueError(
            f"save_env_ids {invalid} are outside the valid range [0, {num_envs})."
        )
    return normalized


def _reset_episode_rows(
    env: Any,
    env_ids: Sequence[int] | torch.Tensor | None,
    *,
    save_data: bool,
) -> None:
    """Reset selected rows, committing or discarding their pending recordings."""
    selected = _normalize_save_env_ids(env, env_ids)
    target = _env_target(env)
    num_envs = int(getattr(target, "num_envs", 1))
    all_env_ids = tuple(range(num_envs))

    if selected == all_env_ids:
        if save_data:
            env.reset()
        else:
            env.reset(options={"save_data": False})
        return

    reset_ids = torch.tensor(
        selected,
        dtype=torch.int32,
        device=getattr(target, "device", None),
    )
    options: dict[str, Any] = {"reset_ids": reset_ids}
    if not save_data:
        options["save_data"] = False
    env.reset(options=options)


def _abort_pending_episode(
    env: Any,
    env_ids: Sequence[int] | torch.Tensor | None = None,
) -> None:
    """Discard buffered data before retrying or closing an environment."""
    _reset_episode_rows(env, env_ids, save_data=False)


def _commit_pending_episode(
    env: Any,
    save_env_ids: Sequence[int] | torch.Tensor | None,
) -> None:
    """Commit selected rows and discard unused rows from the same vector batch."""
    selected = _normalize_save_env_ids(env, save_env_ids)
    num_envs = int(getattr(_env_target(env), "num_envs", 1))
    _reset_episode_rows(env, selected, save_data=True)

    selected_set = set(selected)
    discarded = tuple(
        env_id for env_id in range(num_envs) if env_id not in selected_set
    )
    if discarded:
        _reset_episode_rows(env, discarded, save_data=False)


def _save_failed_episodes_enabled(env: Any) -> bool:
    """Return whether the configured dataset manager keeps failed episodes."""
    dataset_manager = getattr(_env_target(env), "dataset_manager", None)
    return bool(
        dataset_manager is not None
        and getattr(dataset_manager, "save_failed_episodes", False)
    )


def _selected_rows_have_frames(
    result: DemoEpisodeResult,
    save_env_ids: Sequence[int],
) -> bool:
    """Return whether every selected row contains a persistable frame."""
    if result.lengths:
        return all(result.lengths[env_id] > 0 for env_id in save_env_ids)
    return result.length > 0


def generate_and_execute_action_list(
    env: gymnasium.Env,
    idx: int,
    debug_mode: bool,
    *,
    episode_idx: int = 0,
    **kwargs: Any,
) -> bool:
    """Execute one legacy planner result through the common episode executor.

    This compatibility helper now represents one complete task episode. New
    multi-object tasks should implement ``create_demo_segments`` instead of
    calling this function repeatedly.

    Args:
        env: Environment used to generate and execute the actions.
        idx: Index of the legacy action list within the current episode.
        debug_mode: Whether debug mode is enabled.
        episode_idx: Index of the current episode.
        **kwargs: Additional arguments forwarded to action generation.

    Returns:
        Whether a complete, successful episode was executed.
    """
    result = execute_demo_episode(
        env,
        episode_index=episode_idx,
        progress=_progress_wrapper,
        action_sentence=idx,
        **kwargs,
    )
    if not result.completed or not result.all_success:
        log_warning(
            f"Demo episode {episode_idx} is invalid ({result.terminal_reason}); "
            "it will not be saved."
        )
        return False
    return True


def generate_function(
    env: Any,
    num_traj: int | None = None,
    time_id: int = 0,
    save_path: str = "",
    save_video: bool = False,
    debug_mode: bool = False,
    save_env_ids: Sequence[int] | torch.Tensor | None = None,
    **kwargs: Any,
) -> bool:
    """Generate, execute, and transactionally save one task episode batch.

    A task owns its segment count through ``create_demo_segments``. The legacy
    ``num_traj`` parameter is accepted only as ``None`` or ``1`` so callers do
    not accidentally repeat a one-grasp planner inside the same episode. When
    a dataset functor enables ``save_failed_episodes``, a failed result with at
    least one frame in every selected row is committed instead of retried.

    Args:
        env: The environment instance.
        num_traj: Deprecated compatibility value. Must be ``None`` or ``1``.
        time_id (int, optional): Identifier for the current time step or episode.
        save_path (str, optional): Path to save generated videos.
        save_video (bool, optional): Whether to save episode videos.
        debug_mode (bool, optional): Enable debug mode for visualization and logging.
        save_env_ids: Environment rows to persist from this vector batch. Other
            rows are explicitly discarded after the selected rows commit.
        **kwargs: Additional keyword arguments for data generation.

    Returns:
        True if one episode per selected environment row was committed. With
        ``save_failed_episodes`` enabled, committed episodes may be unsuccessful.
    """
    if num_traj not in (None, 1):
        raise ValueError(
            "num_traj no longer controls sub-trajectories. Implement "
            "create_demo_segments() in the task to define multiple segments."
        )

    max_attempts = int(kwargs.pop("max_attempts", 3))
    reset_before = bool(kwargs.pop("reset_before", True))
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be at least 1, got {max_attempts}.")
    normalized_save_env_ids = _normalize_save_env_ids(env, save_env_ids)
    save_failed_episodes = _save_failed_episodes_enabled(env)

    if reset_before:
        _abort_pending_episode(env)

    for attempt in range(1, max_attempts + 1):
        commit_succeeded = False
        try:
            result: DemoEpisodeResult = execute_demo_episode(
                env,
                episode_index=time_id,
                progress=_progress_wrapper,
                **kwargs,
            )
            successful = result.completed and result.all_success
            persistable_failure = (
                not successful
                and save_failed_episodes
                and _selected_rows_have_frames(result, normalized_save_env_ids)
            )
            if successful or persistable_failure:
                # reset() is the commit boundary: dataset functors consume the
                # whole episode once, then buffers and scene state are reset.
                _commit_pending_episode(env, normalized_save_env_ids)
                commit_succeeded = True
                if persistable_failure:
                    log_warning(
                        f"Episode {time_id} failed ({result.terminal_reason}) but "
                        "was saved because save_failed_episodes is enabled."
                    )
                return True
        finally:
            # ``finally`` also covers KeyboardInterrupt, SystemExit, and
            # GeneratorExit. A failed commit is aborted as well, so close()
            # can never implicitly persist the pending partial episode.
            if not commit_succeeded:
                _abort_pending_episode(env)

        log_warning(
            f"Episode {time_id} attempt {attempt}/{max_attempts} failed: "
            f"{result.terminal_reason}. Discarding {result.length} frames."
        )
        if debug_mode:
            log_warning(
                "Failed demo trace: "
                + json.dumps(
                    result.to_metadata(),
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )

    return False


def replay(env, trajectory_path: str, mode: str = "kinematic") -> None:
    """Replay a recorded trajectory.

    The caller retains ownership of ``env``. Wrapper-specific replay state is
    restored before returning, but the environment is not closed here.

    Args:
        env: The environment built from the same config that recorded the
            trajectory (wrapped via :class:`ReplayWrapper`).
        trajectory_path: Path to the ``.pt`` trajectory file.
        mode: ``"kinematic"`` (exact, physics off), ``"dynamic"`` (feed recorded
            actions, physics on), or ``"control"`` (interactive kinematic scrubber).
    """
    data = load_trajectory(trajectory_path)
    meta = data["meta"]
    lengths = meta["lengths"]
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
        # ReplayWrapper.close() also closes its wrapped environment. Restore
        # its local state here and leave the single close() to cli().
        try:
            replay_env.env.sim.enable_physics(True)
        finally:
            replay_env.env._replay_no_auto_reset = False


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
    replay_env: ReplayWrapper,
    control_input: _ReplayControlInput,
    *,
    visualization_runtime: VisualizationRuntime | None = None,
) -> None:
    """Run the interactive replay loop using the provided input source."""
    num_steps = int(replay_env._lengths.min().item())
    max_step = int(getattr(replay_env, "control_max_step", num_steps - 1))
    step = 0
    dt = (
        float(replay_env.env.sim_cfg.physics_dt)
        * replay_env.env.cfg.sim_steps_per_control
    )
    pending_command = None
    auto_playing = False
    prompt_visible = False
    terminal_active = True

    def publish_state(*, visible: bool = True) -> None:
        if visualization_runtime is not None:
            visualization_runtime.publish_replay_control(
                step=step,
                max_step=max_step,
                visible=visible,
            )

    def seek(target: int) -> None:
        nonlocal step
        step = max(0, min(int(target), max_step))
        replay_env.go_to_step(step)
        publish_state()

    def read_key(timeout: float | None) -> str | None:
        nonlocal terminal_active
        if not terminal_active:
            time.sleep(timeout or _REPLAY_CONTROL_POLL_INTERVAL)
            return None
        try:
            return control_input.read_key(timeout=timeout)
        except EOFError:
            if visualization_runtime is None:
                raise
            terminal_active = False
            return None

    seek(0)
    print(f"Trajectory has {num_steps} transitions (state indices 0..{max_step}).")
    try:
        while True:
            if visualization_runtime is not None:
                browser_step = visualization_runtime.drain_replay_control_command()
                if browser_step is not None:
                    if auto_playing:
                        print(f"\nPaused at step {step}.")
                    auto_playing = False
                    pending_command = None
                    seek(browser_step)
                    prompt_visible = False
                    continue

            if auto_playing:
                if step >= max_step:
                    auto_playing = False
                    prompt_visible = False
                    print(f"\nAuto replay finished at step {step}.")
                    continue

                seek(step + 1)
                print(
                    f"\r[auto step {step}/{max_step}]  press any key to pause",
                    end="",
                    flush=True,
                )
                try:
                    key = read_key(dt)
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if key is None:
                    continue

                auto_playing = False
                prompt_visible = False
                print(f"\nPaused at step {step}.")
                if key not in ("a", " ", "\r", "\n"):
                    pending_command = key
                continue

            if not prompt_visible:
                print(
                    f"[step {step}/{max_step}]  n=next  p=prev  <N>=jump  "
                    "a=auto  r=reset  q=quit"
                )
                if control_input.single_key and terminal_active:
                    print("> ", end="", flush=True)
                prompt_visible = True
            try:
                if pending_command is not None:
                    initial = pending_command
                elif visualization_runtime is not None:
                    initial = read_key(_REPLAY_CONTROL_POLL_INTERVAL)
                    if initial is None:
                        continue
                else:
                    initial = None
                command = _read_replay_control_command(
                    control_input,
                    initial=initial,
                )
            except (EOFError, KeyboardInterrupt):
                break
            finally:
                pending_command = None
            if control_input.single_key and not command.isdigit():
                print()
            prompt_visible = False
            if command in ("q", "quit"):
                break
            if command in ("n", ""):
                seek(step + 1)
            elif command in ("p", "b"):
                seek(step - 1)
            elif command == "r":
                seek(0)
            elif command == "a":
                auto_playing = True
            elif command.isdigit():
                seek(int(command))
            elif command == " ":
                continue
            else:
                print(f"Unknown command: {command!r}")
    finally:
        publish_state(visible=False)


def _replay_visualization_runtime(
    replay_env: ReplayWrapper,
) -> VisualizationRuntime | None:
    """Return the running Viser runtime used by a replay environment."""
    runtime = getattr(replay_env.env.sim, "visualization_runtime", None)
    if runtime is None or not runtime.is_running:
        return None
    return runtime


def replay_control(replay_env: ReplayWrapper) -> None:
    """Run an interactive, single-key kinematic trajectory scrubber."""
    visualization_runtime = _replay_visualization_runtime(replay_env)
    if replay_env.env.sim_cfg.headless and visualization_runtime is None:
        log_warning(
            "control mode with --headless: no window to view the scrub. "
            "Re-run without --headless or enable --viser to see the replay."
        )
    replay_env.reset()
    with _ReplayControlInput() as control_input:
        _run_replay_control_loop(
            replay_env,
            control_input,
            visualization_runtime=visualization_runtime,
        )


def main(args: Any, env: Any, gym_config: dict[str, Any]) -> None:
    """Run the selected workflow without taking ownership of ``env``."""
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
    # Prepare one clean scene. max_episodes counts persisted per-environment
    # episodes, not vector batches. Every successful generate_function call
    # commits exactly the selected rows and leaves the next batch ready to plan.
    _abort_pending_episode(env)
    max_episodes = int(gym_config.get("max_episodes", 1))
    if max_episodes < 0:
        raise ValueError(f"max_episodes must be non-negative, got {max_episodes}.")
    num_envs = int(getattr(_env_target(env), "num_envs", 1))
    if num_envs < 1:
        raise ValueError(f"env.num_envs must be at least 1, got {num_envs}.")

    saved_episodes = 0
    generated_batches = 0
    while saved_episodes < max_episodes:
        batch_episode_count = min(num_envs, max_episodes - saved_episodes)
        save_env_ids = tuple(range(batch_episode_count))
        generated = generate_function(
            env,
            time_id=saved_episodes,
            save_path=getattr(args, "save_path", ""),
            save_video=getattr(args, "save_video", False),
            debug_mode=getattr(args, "debug_mode", False),
            save_env_ids=save_env_ids,
            regenerate=getattr(args, "regenerate", False),
            max_attempts=gym_config.get("demo_max_attempts", 3),
            reset_before=False,
        )
        if not generated:
            raise RuntimeError(
                f"Failed to generate episode batch starting at {saved_episodes} after "
                f"{gym_config.get('demo_max_attempts', 3)} attempts."
            )
        saved_episodes += batch_episode_count
        generated_batches += 1

    log_info(
        f"Committed {saved_episodes} episode(s) in {generated_batches} vector batch(es).",
        color="green",
    )

    # Log the trajectory save location before cli() tears down the sim and, by
    # default, os._exit()s the process.
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


def preview(env: gymnasium.Env) -> None:
    """
    Run the following code to create a demonstration and perform env steps.

    ```
    # Demo version of environment rollout
    for i in range(10):
        qpos = env.robot.get_qpos()

        obs, reward, terminated, truncated, info = env.step(qpos)

    # reset the environment
    env.reset()
    ```

    Run the following code to preview the sensor observations.

    ```
    env.preview_sensor_data("camera")
    ```
    """
    _, _ = env.reset()

    end = False
    while end is False:
        print("Press `p` to enter embed mode to interact with the environment.")
        print("Press `q` to quit the simulation.")
        txt = input()
        if txt == "p":
            try:
                from IPython import embed
            except ImportError:
                log_error(
                    "IPython is not installed. Preview mode requires IPython to be "
                    "available. Please install it with `pip install ipython` and try again."
                )
                continue

            embed()
        elif txt == "q":
            end = True

    return


def _create_parser() -> argparse.ArgumentParser:
    """Create the ``run-env`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="embodichain run-env",
        description="Run an environment for data generation or interactive preview.",
    )

    add_env_launcher_args_to_parser(parser, require_gym_config=True)
    parser.set_defaults(viser_image_fps=None)

    parser.add_argument(
        "--expert-program",
        type=str,
        default=None,
        help="Path to a declarative Expert Program (.json, .yaml, or .yml).",
    )
    parser.add_argument(
        "--debug-mode",
        action="store_true",
        help="Log the structured trace for each failed demo attempt.",
    )

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


def _abort_and_close_env(env: Any, *, exit_process: bool | None = None) -> None:
    """Abort pending data, then close the environment exactly once.

    Args:
        env: Gym environment or wrapper.
        exit_process: Optional process-exit policy for ``EmbodiedEnv.close``.
            An abort failure always forces ``False`` so the error can propagate.
    """
    abort_error: BaseException | None = None
    close_error: BaseException | None = None
    try:
        _abort_pending_episode(env)
    except BaseException as error:
        abort_error = error
    try:
        if exit_process is None and abort_error is None:
            env.close()
        else:
            target = getattr(env, "unwrapped", env)
            target.close(
                exit_process=False if abort_error is not None else exit_process
            )
    except BaseException as error:
        close_error = error

    # Recorder finalization is a durability barrier. Never turn a failed flush
    # into a warning that lets an apparently successful data-generation run
    # continue.
    if close_error is not None:
        if abort_error is not None:
            close_error.add_note(
                "Pending episode abort also failed: "
                f"{type(abort_error).__name__}: {abort_error}"
            )
        raise close_error
    if abort_error is not None:
        raise abort_error


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
    expert_program_path = getattr(args, "expert_program", None)
    if expert_program_path is not None:
        if getattr(env_cfg, "expert_program", None) is not None:
            raise ValueError(
                "Expert Program input is ambiguous: choose either the Gym "
                "config's expert_program_path or --expert-program, not both."
            )
        env_cfg.expert_program = _load_expert_program(expert_program_path)
    if (
        getattr(env_cfg, "expert_program", None) is not None
        and getattr(args, "action_config", None) is not None
    ):
        raise ValueError(
            "Declarative Expert Programs and --action_config are mutually "
            "exclusive execution sources."
        )

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
    body_error: BaseException | None = None
    try:
        main(args, env, gym_config)
    except BaseException as error:
        body_error = error
        raise
    finally:
        try:
            # close() may auto-save trajectory state and finalizes asynchronous
            # dataset writers. Resolve the pending transaction as an abort
            # first, including when main() exits via an interrupt or SystemExit.
            _abort_and_close_env(
                env,
                # Successful CLI runs keep the existing fast-exit default.
                # While unwinding, cleanup must return so the original error
                # (including Ctrl-C) remains observable to the caller/shell.
                exit_process=False if body_error is not None else None,
            )
        except BaseException as cleanup_error:
            if body_error is None:
                raise
            if isinstance(body_error, SystemExit) and body_error.code in (None, 0):
                # A nominal zero exit must not hide a failed recorder barrier.
                raise cleanup_error
            body_error.add_note(
                "Environment cleanup also failed: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
        finally:
            try:
                from embodichain.lab.sim.sim_manager import SimulationManager

                SimulationManager.flush_cleanup_queue()
            except Exception as error:
                log_warning(f"Failed to flush simulation cleanup queue: {error}")


if __name__ == "__main__":
    cli()


__all__ = [
    "cli",
    "generate_and_execute_action_list",
    "generate_function",
    "main",
    "preview",
]
