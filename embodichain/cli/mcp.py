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

"""Serve the EmbodiChain catalog or read an existing DexSim Session over stdio."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> None:
    """Start the local EmbodiChain MCP server."""
    parser = argparse.ArgumentParser(prog="embodichain mcp", description=__doc__)
    parser.add_argument("--session", help="Existing DexSim Runtime session ID")
    parser.add_argument(
        "--catalog", action="store_true", help="Expose discovery and catalog resources"
    )
    parser.add_argument(
        "--session-directory", help="Optional DexSim discovery directory"
    )
    args = parser.parse_args(argv)
    if not (args.session or args.catalog):
        parser.error("Supply --session or --catalog")
    if args.session_directory is not None and not args.session:
        parser.error("--session-directory requires --session")
    try:
        from embodichain.mcp.server import serve
    except ModuleNotFoundError as exc:
        if exc.name != "mcp" and not (exc.name or "").startswith("mcp."):
            raise
        parser.exit(
            2, f"Install the MCP extra with pip install 'embodichain[mcp]': {exc}\n"
        )
    serve(args.session, directory=args.session_directory, catalog=args.catalog)
