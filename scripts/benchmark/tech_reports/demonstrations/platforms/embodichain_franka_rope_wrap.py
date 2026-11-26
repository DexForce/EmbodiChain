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
"""Run the Franka rope-wrap Task Program and optionally record its commands."""

from __future__ import annotations

import argparse
from pathlib import Path

__all__: list[str] = []


def main() -> None:
    """Run one headless episode of the authored Franka rope-wrap task."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=73, help="Task seed (default: 73).")
    parser.add_argument(
        "--record-dir", type=Path, help="Save the accepted episode recording here."
    )
    args = parser.parse_args()

    from franka_rope_wrap_inputs import TASK_CONFIG

    from embodichain.lab.scripts.run_task import cli

    task_args = [
        "--gym_config",
        str(TASK_CONFIG),
        "--headless",
        "--max_episodes",
        "1",
        "--seed",
        str(args.seed),
    ]
    if args.record_dir is not None:
        task_args += [
            "--record_trajectory",
            "--trajectory_save_dir",
            str(args.record_dir.expanduser().resolve()),
        ]
    cli(task_args)


if __name__ == "__main__":
    main()
