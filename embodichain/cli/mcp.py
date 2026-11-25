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

"""Serve EmbodiChain robot queries and authorized control operations over stdio."""

from __future__ import annotations

import argparse
from pathlib import Path
from collections.abc import Sequence

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> None:
    """Start the local MCP server for an existing DexSim Runtime session."""
    parser = argparse.ArgumentParser(prog="embodichain mcp", description=__doc__)
    parser.add_argument("--session", help="Existing DexSim Runtime session ID")
    parser.add_argument(
        "--session-directory", help="Optional DexSim discovery directory"
    )
    parser.add_argument(
        "--writer-key-file",
        type=Path,
        help="Enable control tools with this dedicated host's private writer grant",
    )
    parser.add_argument("--run-id", help="Originating Run ID for the DexSim journal")
    parser.add_argument("--turn", type=int, help="Originating turn number")
    parser.add_argument(
        "--policy-catalog",
        type=Path,
        help="Enable policy tools with this local catalog",
    )
    parser.add_argument(
        "--policy-output",
        type=Path,
        help="Directory for policy run receipts and artifacts",
    )
    args = parser.parse_args(argv)
    if args.policy_catalog is not None:
        if args.policy_output is None or not (args.session or args.session_directory):
            parser.error(
                "--policy-catalog requires --policy-output and a Session or discovery directory"
            )
        if args.writer_key_file or args.run_id is not None or args.turn is not None:
            parser.error(
                "Policy runs use their own World scope and journal attribution"
            )
    elif not args.session or args.policy_output is not None:
        parser.error("Supply --session or enable policy tools with --policy-catalog")
    try:
        if args.policy_catalog is not None:
            from embodichain.mcp._policy_runs import serve as serve_policy

            serve_policy(
                args.policy_catalog,
                args.policy_output,
                directory=args.session_directory,
                session_id=args.session,
            )
            return
        from embodichain.mcp.server import serve
    except ModuleNotFoundError as exc:
        if exc.name != "mcp" and not (exc.name or "").startswith("mcp."):
            raise
        parser.exit(
            2, f"Install the MCP extra with pip install 'embodichain[mcp]': {exc}\n"
        )
    serve(
        args.session,
        directory=args.session_directory,
        writer_key_file=args.writer_key_file,
        run_id=args.run_id,
        turn=args.turn,
    )
