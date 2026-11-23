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
"""Run the Franka RJ45 Task Program and optionally record its commands."""

from __future__ import annotations

import argparse
from pathlib import Path

__all__: list[str] = []


def main() -> None:
    """Run one headless episode of the Franka network cable task."""
    from franka_rj45_cycle_inputs import TASK_CONFIG, save_asset_manifest

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--device", help="Override the task simulation device")
    parser.add_argument(
        "--record-dir", type=Path, help="Save the accepted recording here."
    )
    parser.add_argument(
        "--asset-manifest",
        type=Path,
        help="Write the resolved scene assets for Isaac Lab.",
    )
    args = parser.parse_args()

    from embodichain.lab.scripts.run_task import cli

    if args.asset_manifest is not None:
        from embodichain.extensions.robots.franka.rj45_assets import get_rj45_models

        save_asset_manifest(
            args.asset_manifest.expanduser().resolve(), get_rj45_models()
        )

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
    if args.device is not None:
        task_args += ["--device", args.device]
    cli(task_args)


if __name__ == "__main__":
    main()
