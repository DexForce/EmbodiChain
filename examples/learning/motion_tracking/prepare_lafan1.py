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

"""Download and prepare LAFAN1 G1 references for the official tracking task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from embodichain.data_pipeline.datasets import MotionLibrary
from embodichain.data.constants import EMBODICHAIN_DEFAULT_DATA_ROOT
from embodichain_robots.locomotion_profiles import load_locomotion_robot_profile

_REPOSITORY = "lvhaidong/LAFAN1_Retargeting_Dataset"
_REVISION = "ce1572906efe6157840e8474d5a0d7aa87481e74"


def _download_source(cache_dir: Path | None) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=_REPOSITORY,
            repo_type="dataset",
            revision=_REVISION,
            allow_patterns=[
                "g1/*.csv",
                "robot_description/g1/**",
                "README*",
                "LICENSE*",
                "license*",
                "NOTICE*",
            ],
            cache_dir=cache_dir,
            token=False,
        )
    )


def _preserve_sources(source: Path, output: Path, downloaded: bool) -> None:
    notices = output / "sources" / "retargeting"
    notices.mkdir(parents=True, exist_ok=True)
    for pattern in ("README*", "LICENSE*", "license*", "NOTICE*"):
        for path in source.glob(pattern):
            if path.is_file():
                shutil.copy2(path, notices / path.name)
    origin = (
        f"The input was downloaded from `{_REPOSITORY}` at revision `{_REVISION}`."
        if downloaded
        else "The input was supplied through `--source`; its notices are retained below."
    )
    (output / "SOURCES.md").write_text(
        f"""# LAFAN1 G1 sources

{origin}

## Motion trajectories

- Original motion capture: [Ubisoft La Forge LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset).
- Original data license: [CC BY-NC-ND 4.0](https://github.com/ubisoft/ubisoft-laforge-animation-dataset/blob/master/license.txt).
- Retargeting project: [Unitree Robotics LAFAN1 Retargeting Dataset](https://www.modelscope.cn/datasets/unitreerobotics/LAFAN1_Retargeting_Dataset).
- Download mirror: [lvhaidong/LAFAN1_Retargeting_Dataset](https://huggingface.co/datasets/{_REPOSITORY}/tree/{_REVISION}).
- Pinned download revision: `{_REVISION}`.
- The retargeting README describes Interaction Mesh and inverse kinematics and identifies the original motion-data license as CC BY-NC-ND 4.0.

Preparation resamples the 30 Hz G1 joint trajectories to 50 Hz, estimates velocities,
and stores the root orientation in XYZW order. Clip names and joint names are recorded
in `motion.npz`.

## Robot description

`robot/` is copied from the input's `robot_description/g1/`. The upstream root
`LICENSE` names Unitree Robotics and contains BSD 3-Clause terms. Available input
README and license files are copied unchanged into `sources/retargeting/`; robot-local
notices remain alongside the copied files. The upstream README identifies the motion-data
license separately from the robot and code notices.
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        help=(
            "Use an existing root containing g1/ and robot_description/g1/. "
            "Otherwise download the pinned public dataset revision."
        ),
    )
    parser.add_argument(
        "--cache-dir", type=Path, help="Hugging Face download cache directory."
    )
    parser.add_argument(
        "--output", type=Path, default=Path(EMBODICHAIN_DEFAULT_DATA_ROOT) / "LAFAN1G1"
    )
    args = parser.parse_args()
    source = (
        args.source.expanduser()
        if args.source is not None
        else _download_source(args.cache_dir)
    )
    args.output = args.output.expanduser()
    names = load_locomotion_robot_profile("UnitreeG1")["robot"]["joint_names"]
    robot = source / "robot_description/g1"
    if not (robot / "g1_29dof_rev_1_0.urdf").is_file():
        parser.error(f"G1 robot description is missing from {robot}")
    library = MotionLibrary.from_csv(sorted((source / "g1").glob("*.csv")), names)
    args.output.mkdir(parents=True, exist_ok=True)
    library.save_npz(args.output / "motion.npz")
    shutil.copytree(robot, args.output / "robot", dirs_exist_ok=True)
    _preserve_sources(source, args.output, args.source is None)
    metadata = {
        "source": (
            {"repo_id": _REPOSITORY, "revision": _REVISION}
            if args.source is None
            else {"kind": "local_directory"}
        ),
        "clips": library.num_clips,
        "frames": library.num_frames,
        "joints": len(names),
        "source_fps": 30.0,
        "fps": library.fps,
        "quaternion_order": "xyzw",
        "motion_file": "motion.npz",
        "robot_file": "robot/g1_29dof_rev_1_0.urdf",
        "source_notices": "sources/retargeting",
    }
    (args.output / "preparation.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                **metadata,
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
