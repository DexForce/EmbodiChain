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

"""Blind-fit the fixed-scene UR5/PGI Pickup replay using only its joints/video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from embodichain.data import get_data_path
from examples.sim.atomic_skill.tcp_presets import UR5_PGI_GRASP_TCP


def main() -> None:
    """Prepare the sample-owned robot calibration and invoke the production CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("/tmp/embodichain-skill-fit")
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    profile = {
        "joint_names": [f"joint{i}" for i in range(1, 7)]
        + ["gripper_finger1_joint_1", "gripper_finger2_joint_1"],
        "control_parts": {
            "arm": [f"joint{i}" for i in range(1, 7)],
            "eef": ["gripper_finger1_joint_1"],
        },
        "participants": {"primary": {"motion": "arm", "grasp": "eef"}},
        "solvers": {
            "arm": {
                "class_type": "URSolver",
                "ur_type": "ur5",
                "urdf_path": get_data_path("UniversalRobots/UR5/UR5.urdf"),
                "root_link_name": "base_link",
                "end_link_name": "ee_link",
                "tcp": [list(row) for row in UR5_PGI_GRASP_TCP],
            }
        },
        "base_poses": {"arm": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]},
        "hand_commands": {"eef": {"open": [0.0], "grasp": [0.036]}},
        "lower": [-6.283185307] * 6 + [0, -0.04],
        "upper": [6.283185307] * 6 + [0.04, 0.04],
        "velocity_limits": [3.15] * 6 + [0.5, 0.5],
        "acceleration_limits": [12.0] * 6 + [40.0, 40.0],
        "calibration_revision": "atomic-example-ur5-pgi-grasp-tcp-2026-11",
    }
    path = directory / "robot_profile.json"
    path.write_text(json.dumps(profile, indent=2) + "\n")
    from embodichain.toolkits.skill_fitting.cli import main as fit

    fit(
        [
            "--video",
            str(args.replay_dir / "00_pickup_cube_auto_play.mp4"),
            "--joints",
            str(args.replay_dir / "00_pickup_cube_auto_play.pt"),
            "--robot-profile",
            str(path),
            "--skill",
            "pickup",
            "--time-offset",
            "0",
            "--qpos-semantics",
            "actual_submission",
            "--device",
            args.device,
            "--output",
            str(directory / "pickup_fit.npz"),
        ]
    )


if __name__ == "__main__":
    main()
