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

"""Collect the UR5 reference or one layout strategy with native grasp expansion."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from embodichain.lab.expansion import run_task_program_expansion

HERE = Path(__file__).resolve().parent
TASK = (
    HERE.parents[3]
    / "extensions/embodichain_tasks/configs/tasks/manipulation"
    / "workspace_pose_pick_place/acceptance/scene_first/task.yaml"
)


def main() -> None:
    """Save the resolved configuration and collect one group of measured episodes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("reference", "scene_first", "robot_first"))
    parser.add_argument("--output", type=Path, required=True, help="Result root")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    config = yaml.safe_load((HERE / "expansion.yaml").read_text())
    sampling = config["scene_sampling"]
    if args.mode == "reference":
        config["engine"]["episodes_per_operator"] = 1
        config["trajectory"]["operators"] = ["object_pose"]
        config["trajectory"].pop("affordance", None)
        sampling.update(strategy="scene_first", max_rollouts=1)
        for ranges in config["scene"].values():
            ranges.update(translation_m=[[0, 0], [0, 0], [0, 0]], yaw_rad=[0, 0])
        bounds = sampling["layout"]["policy"]["group_xy_offset_bounds_m"]
        for uid in bounds:
            bounds[uid] = [[0, 0], [0, 0]]
    else:
        sampling["strategy"] = args.mode

    directory = args.output.resolve() / args.mode
    if directory.exists():
        parser.error(f"{directory} already exists. Choose a fresh output directory.")
    directory.mkdir(parents=True)
    snapshot = directory / "expansion.yaml"
    snapshot.write_text(yaml.safe_dump(config, sort_keys=False))
    result = run_task_program_expansion(
        TASK, snapshot, directory / "collection", seed=args.seed
    )
    for name, group in result["operators"].items():
        print(f"{args.mode}/{name}: {group['committed_episodes']} confirmed")
    if not result["complete"]:
        raise SystemExit(
            f"Collection quota incomplete. See {directory / 'collection'}."
        )


if __name__ == "__main__":
    main()
