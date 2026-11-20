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

"""Clean a dual-arm tabletop reference image with a bottom-center camera base."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path

__all__: list[str] = []


def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> None:
    """Parse the standalone image cleanup command."""
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Source RGB frame")
    parser.add_argument(
        "--output", type=Path, required=True, help="New result directory"
    )
    parser.add_argument(
        "--gripper-prompt", help="Describe the appearance of the grippers"
    )
    parser.add_argument(
        "--gripper-box",
        action="append",
        nargs=4,
        type=int,
        metavar=("X0", "Y0", "X1", "Y1"),
        help="Restrict ambiguous gripper detections to a pixel box. Repeat per gripper.",
    )
    parser.add_argument(
        "--camera-support-prompt",
        help="Describe the appearance of the bottom-center camera support",
    )
    parser.add_argument("--edit-prompt", help="Override the background repair prompt")
    parser.add_argument("--seed", type=int, default=42, help="Image edit seed")
    args = parser.parse_args(argv)

    from embodichain.gen_sim._robot_cleanup import clean_robot_image
    from embodichain.gen_sim.services.runtime import service_scope

    with service_scope() as services:
        report = clean_robot_image(
            args.image,
            args.output,
            services=services,
            gripper_boxes=tuple(tuple(box) for box in (args.gripper_box or ())),
            gripper_prompt=args.gripper_prompt,
            camera_support_prompt=args.camera_support_prompt,
            edit_prompt=args.edit_prompt,
            seed=args.seed,
        )
    print(json.dumps(report, indent=2))
