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

"""Serve EmbodiChain queries over stdio for an explicitly selected session."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> None:
    """Start the local MCP server for an existing DexSim Runtime session."""
    parser = argparse.ArgumentParser(prog="embodichain mcp", description=__doc__)
    parser.add_argument(
        "--session", required=True, help="Existing DexSim Runtime session ID"
    )
    parser.add_argument(
        "--session-directory", help="Optional DexSim discovery directory"
    )
    args = parser.parse_args(argv)
    try:
        from embodichain.mcp.server import serve
    except ModuleNotFoundError as exc:
        if exc.name != "mcp" and not (exc.name or "").startswith("mcp."):
            raise
        parser.exit(
            2, f"Install the MCP extra with pip install 'embodichain[mcp]': {exc}\n"
        )
    serve(args.session, directory=args.session_directory)
