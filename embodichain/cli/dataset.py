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

"""Offline dataset inspection, splitting, and recording recovery commands."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

__all__: list[str] = []


def main(argv: Sequence[str] | None = None) -> None:
    """Run offline expert-dataset tools without constructing a simulator.

    Args:
        argv: Arguments excluding the command name. Uses ``sys.argv`` when
            omitted.
    """
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] in {"validate", "split"}:
        from embodichain.data_pipeline.datasets.inspection import main as inspect_main

        status = inspect_main(arguments)
        if status:
            raise SystemExit(status)
        return

    parser = argparse.ArgumentParser(
        prog="embodichain dataset",
        description="Validate, split, and recover recorded expert datasets.",
    )
    subparsers = parser.add_subparsers(dest="operation", required=True)
    subparsers.add_parser("validate", add_help=False, help="Check dataset consistency.")
    subparsers.add_parser(
        "split",
        add_help=False,
        help="Create a grouped quality-filtered split manifest.",
    )
    recover = subparsers.add_parser(
        "recover", help="Diagnose unfinished commits and repair recoverable metadata."
    )
    recover.add_argument("root", type=Path, help="Recorded LeRobot dataset directory.")
    recover.add_argument(
        "--repair",
        action="store_true",
        help="Repair missing sidecars only when committed artifacts verify.",
    )
    args = parser.parse_args(arguments)

    from embodichain.data_pipeline.recording import recover_recording

    try:
        result = recover_recording(args.root, repair=args.repair)
    except (OSError, ValueError) as error:
        parser.exit(2, f"{error}\n")
    payload = result.to_dict() if hasattr(result, "to_dict") else result
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if not payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
