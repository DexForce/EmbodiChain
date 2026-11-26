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
    parser.add_argument(
        "--motion-output",
        type=Path,
        help="Receipts directory for a dedicated motion World",
    )
    parser.add_argument(
        "--handoff-key-file",
        type=Path,
        help="Operator grant for a standard handoff-enabled manual World",
    )
    parser.add_argument(
        "--experiment-output",
        type=Path,
        help="Local durable GenSim authoring runs; no pre-existing World required",
    )
    parser.add_argument(
        "--experiment-agent",
        help="Restrict this experiment server to one agent session and one run",
    )
    args = parser.parse_args(argv)
    if args.experiment_agent and not args.experiment_output:
        parser.error("--experiment-agent requires --experiment-output")
    if args.experiment_output is not None:
        if any(
            (
                args.session,
                args.session_directory,
                args.writer_key_file,
                args.handoff_key_file,
                args.motion_output,
                args.policy_catalog,
                args.policy_output,
                args.run_id,
                args.turn,
            )
        ):
            parser.error(
                "Experiment authoring owns its workers; choose one session mode"
            )
        from embodichain.mcp._experiments import serve as serve_experiments

        serve_experiments(args.experiment_output, args.experiment_agent)
        return
    if args.handoff_key_file and (args.writer_key_file or args.policy_catalog):
        parser.error("Select one host control mode")
    if args.motion_output is not None and args.handoff_key_file is None:
        if (
            not args.session
            or args.writer_key_file is None
            or args.policy_catalog is not None
            or args.policy_output is not None
        ):
            parser.error("--motion-output requires --session and --writer-key-file")
        if (args.run_id is None) != (args.turn is None) or (
            args.turn is not None and args.turn < 1
        ):
            parser.error("Supply both a Run ID and a positive turn number")
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
        if args.motion_output is not None and args.handoff_key_file is None:
            from embodichain.mcp._motion import serve as serve_motion

            labels = (
                {}
                if args.run_id is None
                else {"run_id": args.run_id, "turn": args.turn}
            )
            serve_motion(
                args.session,
                args.motion_output,
                args.writer_key_file,
                directory=args.session_directory,
                labels=labels,
            )
            return
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
        parser.exit(2, f"Install EmbodiChain with its default dependencies: {exc}\n")
    serve(
        args.session,
        directory=args.session_directory,
        writer_key_file=args.writer_key_file,
        handoff_key_file=args.handoff_key_file,
        motion_output=args.motion_output,
        run_id=args.run_id,
        turn=args.turn,
    )
