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
"""Dynamically replay an accepted Franka RJ45 recording with EmbodiChain."""

from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from franka_rj45_cycle_inputs import (
    JOINT_NAMES,
    TASK_CONFIG,
    load_commands,
    load_scene,
    save_results,
    scene_inputs,
)
from franka_rope_wrap_inputs import file_info

if TYPE_CHECKING:
    from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv

__all__: list[str] = []


def main() -> None:
    """Replay exactly the source commands and save the new state and measurements."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", help="Override the task simulation device")
    args = parser.parse_args()

    import gymnasium
    from tensordict import TensorDict
    from tensordict._reductions import _make_td
    import torch

    from embodichain.lab.gym.envs.types import ControllerAction
    from embodichain.lab.gym.utils.gym_utils import build_env_cfg_from_args
    from embodichain.lab.gym.utils.trajectory_state import capture_trajectory_state
    from embodichain.lab.scripts.run_task import (
        _create_parser,
        discover_task_packages,
        execute_init_hooks,
    )
    from embodichain.lab.sim import SimulationManager
    from embodichain.extensions.robots.franka.rj45_assets import get_rj45_models
    from embodichain.extensions.tasks.manipulation.cable.rj45_measurements import (
        evaluate_rj45,
    )

    recording = args.recording.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    config, environment, _ = load_scene()
    positions, velocities, source = load_commands(recording, config)
    models = get_rj45_models()
    inputs = scene_inputs(recording, models)
    steps = len(positions)
    output.mkdir(parents=True, exist_ok=False)
    cable_cfg = environment["simulation"]["cable_object"][0]
    cable_lengths = np.linalg.norm(np.diff(cable_cfg["points"], axis=0), axis=1)
    dof = len(JOINT_NAMES)
    data = {
        "time_s": np.arange(steps + 1) / config["control_hz"],
        "robot_q": np.empty((steps + 1, dof), dtype=np.float32),
        "robot_qd": np.empty((steps + 1, dof), dtype=np.float32),
        "robot_root_pose": np.empty((steps + 1, 7), dtype=np.float32),
        "plug_pose": np.empty((steps + 1, 7), dtype=np.float32),
        "latch_angle_rad": np.empty(steps + 1, dtype=np.float32),
        "cable_body_pose": np.empty(
            (steps + 1, len(cable_lengths), 7), dtype=np.float32
        ),
        "cable_segment_lengths": cable_lengths,
        "command_q": np.empty_like(positions),
        "command_qd": np.empty_like(velocities),
    }
    order: list[int] = []
    observed = 0
    reset_deltas: dict[str, float] = {}

    def observe(env: EmbodiedEnv, frame: int, total_steps: int) -> None:
        nonlocal observed, order
        if frame != observed or total_steps != steps:
            raise RuntimeError("Unexpected control index in the RJ45 replay.")
        cable = env.sim.get_cable_object("cable")
        plug = env.sim.get_articulation("plug")
        if cable is None or plug is None:
            raise RuntimeError("The configured RJ45 cable or plug is missing.")
        if frame == 0:
            order = [env.robot.joint_names.index(name) for name in JOINT_NAMES]
        state = cable.data.segment_pose_w
        if state.shape != (1, len(cable_lengths), 7):
            raise RuntimeError(
                f"Unexpected RJ45 cable pose shape: {tuple(state.shape)}"
            )
        data["cable_body_pose"][frame] = state[0].cpu().numpy()
        data["robot_q"][frame] = env.robot.get_qpos()[0].cpu().numpy()[order]
        data["robot_qd"][frame] = env.robot.get_qvel()[0].cpu().numpy()[order]
        data["robot_root_pose"][frame] = env.robot.get_local_pose()[0].cpu().numpy()
        data["plug_pose"][frame] = plug.get_link_pose("plug")[0].cpu().numpy()
        latch_id = plug.joint_names.index("latch_hinge")
        data["latch_angle_rad"][frame] = plug.get_qpos()[0, latch_id].item()
        if frame:
            data["command_q"][frame - 1] = (
                env.robot.get_qpos(target=True)[0].cpu().numpy()[order]
            )
            data["command_qd"][frame - 1] = (
                env.robot.get_qvel(target=True)[0].cpu().numpy()[order]
            )
        observed += 1
        if frame != steps:
            return
        if not all(np.isfinite(array).all() for array in data.values()):
            raise FloatingPointError(
                "RJ45 replay generated a nonfinite state or command."
            )
        np.testing.assert_array_equal(data["command_q"], positions)
        np.testing.assert_array_equal(data["command_qd"], velocities)
        rows = env._rj45_monitor.rows
        if len(rows) != steps + 1:
            raise RuntimeError(
                f"Expected {steps + 1} measured RJ45 frames, got {len(rows)}."
            )
        acceptance = evaluate_rj45(rows, config)
        report = {
            "status": "completed",
            "backend": "EmbodiChain DexSim Newton (DexUni)",
            "device": str(state.device),
            "versions": {
                name: version(name)
                for name in ("embodichain", "dexsim_engine", "torch")
            },
            "inputs": inputs,
            "script": file_info(Path(__file__)),
            "source_episode": source,
            "joint_names": JOINT_NAMES,
            "control_hz": config["control_hz"],
            "control_steps": steps,
            "physics_steps_per_control": config["substeps"],
            "pre_roll_s": config["pre_roll_s"],
            "cable_segments": config["cable_segments"],
            "initial_state": {
                "mode": "fresh_scene_pre_roll",
                "max_abs_difference_from_recording": reset_deltas,
            },
            "position_and_velocity_readbacks_equal_recorded_actions": True,
            "task_success": acceptance["passed"],
            "acceptance": acceptance,
        }
        # Save before the native renderer/simulation shutdown can fast-exit.
        save_results(output, data, report, rows)

    launcher_args = [
        "--gym_config",
        str(TASK_CONFIG),
        "--headless",
    ]
    if source["seed"] is not None:
        launcher_args += ["--seed", str(source["seed"])]
    if args.device is not None:
        launcher_args += ["--device", args.device]
    discover_task_packages()
    execute_init_hooks()
    cfg, gym_config, action_config = build_env_cfg_from_args(
        _create_parser().parse_args(launcher_args)
    )
    env = gymnasium.make(gym_config["id"], cfg=cfg, **action_config).unwrapped
    try:
        env.reset(seed=source["seed"])
        with torch.serialization.safe_globals([TensorDict, _make_td]):
            initial = torch.load(
                recording, map_location=env.device, weights_only=True, mmap=True
            )["states"][:, 0]
        for category, lookup in (
            ("cables", env.sim.get_cable_object),
            ("articulations", env.sim.get_articulation),
            ("rigid_objects", env.sim.get_rigid_object),
        ):
            if any(lookup(uid) is None for uid in initial[category].keys()):
                raise ValueError(f"RJ45 replay is missing a recorded {category} UID.")
        env_ids = torch.arange(env.num_envs, device=env.device)
        current = initial.clone()
        capture_trajectory_state(env, current, env_ids)
        # The replay owns a fresh settle. Keep the ordinary cable replay's
        # match check for scene and cable state, while reporting home servo
        # residuals instead of injecting recorded robot state or solver history.
        for key, expected in initial.items(include_nested=True, leaves_only=True):
            value = current[key].to(expected)
            if not torch.isfinite(value).all() or not torch.isfinite(expected).all():
                raise ValueError(f"Nonfinite initial RJ45 replay state at {key!r}.")
            servo_axis = key in (("robot", "qpos"), ("robot", "qvel"))
            if value.shape != expected.shape or (
                not servo_axis
                and not torch.allclose(value, expected, rtol=1e-5, atol=1e-6)
            ):
                raise ValueError(f"RJ45 replay scene state differs at {key!r}.")
            reset_deltas["/".join(key)] = (
                float((value - expected).abs().max()) if value.numel() else 0.0
            )
        observe(env, 0, steps)
        q = torch.as_tensor(positions, device=env.device)
        qd = torch.as_tensor(velocities, device=env.device)
        for frame in range(1, steps + 1):
            command = TensorDict(
                {"qpos": q[frame - 1 : frame], "qvel": qd[frame - 1 : frame]},
                batch_size=[1],
            )
            _, _, terminated, truncated, _ = env.step(ControllerAction(command))
            if frame < steps and (bool(terminated.any()) or bool(truncated.any())):
                raise RuntimeError(
                    f"RJ45 dynamics replay ended after {frame} controls."
                )
            observe(env, frame, steps)
            if frame % 600 == 0 and frame < steps:
                print(f"Completed {frame}/{steps} recorded controls", flush=True)
    except BaseException:
        env.close(exit_process=False)
        SimulationManager.flush_cleanup_queue()
        raise
    else:
        env.close()


if __name__ == "__main__":
    main()
