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

"""Offline preparation and verification of demonstration prototypes."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path

from .contracts import JointSpec
from .io import load_prototype, load_trajectory, prepare_prototype

__all__: list[str] = []


def main(argv: Sequence[str] | None = None) -> None:
    """Prepare or validate a portable source/prototype bundle."""
    parser = argparse.ArgumentParser(prog="embodichain real2sim")
    commands = parser.add_subparsers(dest="operation", required=True)
    prepare = commands.add_parser(
        "prepare", help="Normalize and prepare a motion prototype."
    )
    prepare.add_argument("source", type=Path, help="Source trajectory JSON manifest.")
    prepare.add_argument(
        "--joints",
        type=Path,
        required=True,
        help="JSON array of target JointSpec declarations.",
    )
    prepare.add_argument(
        "--control-dt", type=float, required=True, help="Command period in seconds."
    )
    prepare.add_argument(
        "--output", type=Path, required=True, help="New prototype bundle directory."
    )
    validate = commands.add_parser(
        "validate", help="Verify an existing source or prototype bundle."
    )
    validate.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.operation == "prepare":
            layout = json.loads(args.joints.read_text(encoding="utf-8"))
            if not isinstance(layout, list):
                raise ValueError("joint layout must be a JSON array")
            joints = tuple(JointSpec(**item) for item in layout)
            path = prepare_prototype(
                load_trajectory(args.source),
                joints=joints,
                control_dt=args.control_dt,
                directory=args.output,
            )
            prototype = load_prototype(path)
        elif args.manifest.name == "prototype.json":
            path = args.manifest
            prototype = load_prototype(path)
        else:
            source = load_trajectory(args.manifest)
            print(
                json.dumps(
                    {
                        "status": "valid_source",
                        "source_id": source.source_id,
                        "unit_id": source.unit_id,
                        "origin": source.origin,
                    }
                )
            )
            return
    except (ValueError, TypeError, OSError, json.JSONDecodeError) as exc:
        parser.exit(2, f"{exc}\n")
    print(
        json.dumps(
            {
                "status": "valid_prototype",
                "manifest": str(path),
                "source_id": prototype.unit.source.source_id,
                "unit_id": prototype.unit.source.unit_id,
                "skill_id": prototype.unit.source.skill_id,
                "origin": prototype.unit.source.origin,
                "samples": len(prototype.unit.timestamps),
            }
        )
    )
