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
"""Dynamically replay a recorded Franka rope wrap through EmbodiChain.

Use the standard run-task replay and measure cable geometry from its live
state after reset and each control step.
"""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv

__all__: list[str] = []


def main() -> None:
    """Run one headless dynamics replay and write its cable trajectory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recording", required=True, type=Path, help="Trusted run-task .pt artifact"
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="New result directory"
    )
    parser.add_argument("--device", help="Override the task simulation device")
    args = parser.parse_args()

    from franka_rope_wrap_inputs import (
        TASK_CONFIG,
        file_info,
        geometry_summary,
        input_metadata,
        load_commands,
        load_scene,
        sha256,
    )

    from embodichain.data import get_data_path
    from embodichain.lab.scripts.run_task import cli

    recording = args.recording.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    config, _, _, _ = load_scene()
    control_hz = config["control_hz"]
    substeps = config["substeps"]
    segments = config["cable_segments"]
    positions, _, source = load_commands(
        recording, control_hz=control_hz, substeps=substeps
    )
    steps = len(positions)
    output.mkdir(parents=True, exist_ok=False)

    poses = np.empty((steps + 1, segments, 7), dtype=np.float32)
    next_frame = 0

    def observe(env: EmbodiedEnv, frame: int, total_steps: int) -> None:
        nonlocal next_frame
        if total_steps != steps or frame != next_frame or frame > steps:
            raise RuntimeError("Unexpected frame index in the Franka dynamics replay.")
        cable = env.sim.get_cable_object("cable")
        if cable is None:
            raise RuntimeError("No cable object in the Franka replay scene.")
        state = cable.data.segment_pose_w
        if state.shape != (1, segments, 7):
            raise RuntimeError(f"Unexpected cable pose shape {tuple(state.shape)}.")
        poses[frame] = state[0].cpu().numpy()
        if not np.isfinite(poses[frame]).all():
            raise FloatingPointError(f"Nonfinite cable pose at replay frame {frame}.")
        next_frame += 1
        if frame != steps:
            return

        lengths = np.linalg.norm(np.diff(cable.cfg.points, axis=0), axis=1)
        if lengths.shape != (segments,):
            raise RuntimeError("Cable rest lengths do not match the replayed segments.")
        robot_urdf = Path(get_data_path(env.robot.cfg.fpath)).resolve(strict=True)
        report = {
            "status": "completed",
            "backend": "EmbodiChain DexSim Newton (DexUni)",
            "device": str(state.device),
            "versions": {
                name: version(name)
                for name in ("embodichain", "dexsim_engine", "torch")
            },
            "inputs": input_metadata(recording, robot_urdf),
            "script": file_info(Path(__file__)),
            "source_episode": source,
            "control_hz": control_hz,
            "control_steps": steps,
            "physics_steps_per_control": substeps,
            "cable_segments": segments,
            "task_success": None,
            "final_geometry": geometry_summary(
                poses, lengths, config["post_center_xy_m"], control_hz
            ),
        }
        # The CLI may fast-exit while closing the simulator after replay.
        trajectory = output / "trajectory.npz"
        np.savez_compressed(
            trajectory,
            time_s=np.arange(steps + 1) / control_hz,
            cable_body_pose=poses,
            cable_segment_lengths=lengths,
        )
        report["trajectory_sha256"] = sha256(trajectory)
        summary = output / "result.json"
        summary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"Completed {steps} controls. Result: {summary}", flush=True)

    replay_args = [
        "--gym_config",
        str(TASK_CONFIG),
        "--headless",
        "--replay",
        "--replay_mode",
        "dynamic",
        "--replay_trajectory",
        str(recording),
    ]
    if source["seed"] is not None:
        replay_args += ["--seed", str(source["seed"])]
    if args.device is not None:
        replay_args += ["--device", args.device]
    cli(replay_args, replay_observer=observe)
    if next_frame != steps + 1:
        raise RuntimeError(f"Observed only {next_frame} of {steps + 1} replay frames.")


if __name__ == "__main__":
    main()
