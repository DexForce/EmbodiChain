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

"""CLI for preflight, collection, recovery and verification of expansion jobs."""

from __future__ import annotations
import argparse
import json
from collections.abc import Sequence

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> int:
    """Run a configured expansion operation and print its machine-readable report.

    Args:
        argv: Optional command arguments without the executable name.
    Returns:
        Zero for successful preflight/inspection or a completed collection;
        two when a bounded collection stopped before its target.
    """
    parser = argparse.ArgumentParser(prog="embodichain expansion")
    commands = parser.add_subparsers(dest="operation", required=True)
    for name in ("preflight", "run", "resume"):
        command = commands.add_parser(name)
        command.add_argument("--job", required=True)
    command = commands.add_parser("inspect")
    command.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    from embodichain.lab.expansion.configured import (
        load_expansion_job,
        preflight_expansion_job,
    )
    from embodichain.lab.expansion.workflow import (
        run_expansion_job,
        inspect_expansion_job,
    )

    if args.operation == "inspect":
        result = inspect_expansion_job(args.output)
    else:
        cfg = load_expansion_job(args.job)
        from embodichain.lab.expansion import ExpansionEngine

        engine = ExpansionEngine.from_job(cfg)
        result = getattr(engine, args.operation)()
    printable = {
        key: value for key, value in result.items() if key != "resolved_deployment"
    }
    print(json.dumps(printable, indent=2, sort_keys=True))
    if args.operation in {"run", "resume"} and not result["complete"]:
        raise SystemExit(2)
    return 0
