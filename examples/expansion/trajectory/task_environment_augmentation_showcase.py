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

"""Show nine task-space trajectories on parallel PickUp -> Place scenes.

This showcase reuses the existing Atomic Action task scene and runs the nominal
plan plus eight endpoint-preserving task-space deformations through nine parallel
simulation arenas. It writes joint/tool-path plots and ``place_auto_play.mp4``
to a caller-selected directory. It demonstrates physical task execution and
variant diversity; it is not a dataset qualification or confirmed EpisodeSink
run.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

__all__ = ["main"]

SHOWCASE_TRAJECTORY_STRATEGY = "ik_interp"
SHOWCASE_TRAJECTORY_CONTROL_DT = 0.04


def _has_option(arguments: list[str], option: str) -> bool:
    """Return whether a forwarded option uses split or ``--name=value`` form."""
    return any(value == option or value.startswith(f"{option}=") for value in arguments)


def main(argv: list[str] | None = None) -> None:
    """Run the task-environment augmentation showcase.

    Args:
        argv: Showcase arguments followed by optional Place tutorial arguments.
            The only required showcase argument is ``--output-dir``.
    """
    parser = argparse.ArgumentParser(
        description="Run nine parallel task-space PickUp -> Place rollouts."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for variant plots and replay output.",
    )
    args, forwarded = parser.parse_known_args(argv)
    from examples.sim.atomic_action import place

    args.output_dir.mkdir(parents=True, exist_ok=True)
    forwarded = list(forwarded)
    if not _has_option(forwarded, "--variant_plot_dir"):
        forwarded.extend(("--variant_plot_dir", str(args.output_dir)))
    if not _has_option(forwarded, "--trajectory_variants"):
        forwarded.extend(("--trajectory_variants", "9"))
    if not _has_option(forwarded, "--num_envs"):
        forwarded.extend(("--num_envs", "9"))
    if not _has_option(forwarded, "--trajectory_variant_mode"):
        forwarded.extend(("--trajectory_variant_mode", "task_space"))
    if not _has_option(forwarded, "--trajectory_strategy"):
        forwarded.extend(("--trajectory_strategy", SHOWCASE_TRAJECTORY_STRATEGY))
    if not _has_option(forwarded, "--trajectory_control_dt"):
        forwarded.extend(
            ("--trajectory_control_dt", str(SHOWCASE_TRAJECTORY_CONTROL_DT))
        )
    if "--headless" not in forwarded and "--no-headless" not in forwarded:
        forwarded.append("--headless")
    if "--auto_play" not in forwarded:
        forwarded.append("--auto_play")
    if not _has_option(forwarded, "--auto_play_video_path"):
        forwarded.extend(
            (
                "--auto_play_video_path",
                str(args.output_dir / "place_auto_play.mp4"),
            )
        )
    previous = sys.argv
    try:
        sys.argv = ["task_environment_augmentation_showcase", *forwarded]
        place.main()
    finally:
        sys.argv = previous


if __name__ == "__main__":
    main()
