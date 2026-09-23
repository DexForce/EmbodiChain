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

"""Show task-environment trajectory expansion on the PickUp -> Place scene.

This showcase reuses the existing Atomic Action task scene and runs the nominal
plan plus configured trajectory variants through the real simulator. It writes
joint/tool-path plots and the replay filmstrip to a caller-selected directory.
It demonstrates physical task execution and variant diversity; it is not a
dataset qualification or confirmed EpisodeSink run.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

__all__ = ["main"]


def main(argv: list[str] | None = None) -> None:
    """Run the task-environment augmentation showcase.

    Args:
        argv: Showcase arguments followed by optional Place tutorial arguments.
            The only required showcase argument is ``--output-dir``.
    """
    parser = argparse.ArgumentParser(
        description="Run nominal and trajectory-variant PickUp -> Place rollouts."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for variant plots and replay output.",
    )
    args, forwarded = parser.parse_known_args(argv)
    from scripts.tutorials.atomic_action import place

    args.output_dir.mkdir(parents=True, exist_ok=True)
    forwarded = list(forwarded)
    if "--variant_plot_dir" not in forwarded:
        forwarded.extend(("--variant_plot_dir", str(args.output_dir)))
    if "--trajectory_variants" not in forwarded:
        forwarded.extend(("--trajectory_variants", "4"))
    if "--headless_play" not in forwarded:
        forwarded.append("--headless_play")
    previous = sys.argv
    try:
        sys.argv = ["task_environment_augmentation_showcase", *forwarded]
        place.main()
    finally:
        sys.argv = previous


if __name__ == "__main__":
    main()
