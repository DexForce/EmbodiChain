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

import gymnasium
import numpy as np
import argparse
import os
import time
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


def generate_and_execute_action_list(env, idx, debug_mode, **kwargs):

    action_list = env.get_wrapper_attr("create_demo_action_list")(
        action_sentence=idx, **kwargs
    )

    if action_list is None or len(action_list) == 0:
        log_warning("Action is invalid. Skip to next generation.")
        return False

    for action in tqdm.tqdm(
        action_list, desc=f"Executing action list #{idx}", unit="step"
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
                env, trajectory_idx, debug_mode, **kwargs
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


def replay_control(replay_env: ReplayWrapper) -> None:
    """Interactive kinematic scrubber: terminal input controls progress."""
    num_steps = int(replay_env._lengths.min().item())
    max_step = num_steps - 1
    if replay_env.env.sim_cfg.headless:
        log_warning(
            "control mode with --headless: no window to view the scrub. "
            "Re-run without --headless to see the replay."
        )
    replay_env.reset()
    step = 0
    replay_env.go_to_step(step)
    dt = (
        float(replay_env.env.sim_cfg.physics_dt)
        * replay_env.env.cfg.sim_steps_per_control
    )
    print(f"Trajectory has {num_steps} steps (0..{max_step}).")
    while True:
        print(
            f"[step {step}/{max_step}]  n=next  p=prev  <N>=jump  a=auto  r=reset  q=quit"
        )
        try:
            cmd = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if cmd in ("q", "quit"):
            break
        elif cmd in ("n", ""):
            step = min(step + 1, max_step)
        elif cmd in ("p", "b"):
            step = max(step - 1, 0)
        elif cmd == "r":
            step = 0
        elif cmd == "a":
            for s in range(step + 1, num_steps):
                step = s
                replay_env.go_to_step(step)
                time.sleep(dt)
            continue
        elif cmd.isdigit():
            step = max(0, min(int(cmd), max_step))
        else:
            print(f"Unknown command: {cmd!r}")
            continue
        replay_env.go_to_step(step)


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

    # Final reset.
    _, _ = env.reset()

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

    exit(0)


def cli():
    """Command-line interface for environment runner.

    Parses CLI arguments, builds the environment config, and launches
    the data generation, preview, or replay workflow.
    """
    np.set_printoptions(5, suppress=True)
    torch.set_printoptions(precision=5, sci_mode=False)

    parser = argparse.ArgumentParser()

    add_env_launcher_args_to_parser(parser)

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

    args = parser.parse_args()

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
