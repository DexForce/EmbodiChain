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

"""Command-line interface for workspace analyzer cache management."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    """Run the workspace analyzer cache CLI.

    Args:
        argv: Arguments excluding the command name. Uses ``sys.argv`` when
            omitted.
    """
    parser = argparse.ArgumentParser(
        prog="embodichain workspace-cache",
        description="Manage workspace analyzer cache sessions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s list                    List all cache sessions
  %(prog)s info session_20241127   Show session details
  %(prog)s clean session_20241127  Clean a specific session
  %(prog)s clean --all             Clean all sessions
  %(prog)s size                    Show total cache size
        """,
    )
    parser.add_argument(
        "command",
        choices=["list", "info", "clean", "size"],
        help="Command to execute.",
    )
    parser.add_argument(
        "session",
        nargs="?",
        help="Session name for the info and clean commands.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Apply the clean command to all sessions.",
    )
    args = parser.parse_args(argv)

    from embodichain.lab.sim.utility.workspace_analyzer.caches.cache_utils import (
        clean_all_sessions,
        clean_session,
        list_sessions,
        show_session_info,
        show_total_size,
    )

    if args.command == "list":
        list_sessions()
    elif args.command == "info":
        if not args.session:
            parser.error("the info command requires a session name")
        show_session_info(args.session)
    elif args.command == "clean":
        if args.all:
            clean_all_sessions()
        elif args.session:
            clean_session(args.session)
        else:
            parser.error("the clean command requires a session name or --all")
    elif args.command == "size":
        show_total_size()


if __name__ == "__main__":
    main()


__all__ = ["main"]
