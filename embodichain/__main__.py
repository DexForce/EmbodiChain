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

"""Unified CLI entry point for ``python -m embodichain``.

Usage examples::

    python -m embodichain preview-asset --asset_path /path/to/asset.usda --preview
    python -m embodichain run-env --env_name my_env
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    """Dispatch to the appropriate sub-command CLI."""
    parser = argparse.ArgumentParser(
        prog="embodichain",
        description="EmbodiChain command-line interface.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # -- preview-asset -------------------------------------------------------
    preview_asset_parser = subparsers.add_parser(
        "preview-asset",
        help="Preview a USD or mesh asset in the simulation.",
    )
    # Import and wire up the existing CLI so argparse is handled by the
    # sub-command module itself.  We pass ``parse_known_args`` style by
    # letting the sub-command parser own its own arguments.
    from embodichain.lab.scripts.preview_asset import cli as preview_asset_cli

    preview_asset_parser.set_defaults(func=preview_asset_cli)

    # Re-add the preview-asset arguments here so ``--help`` works on the
    # sub-command.  We delegate to the existing ``cli()`` which calls
    # ``parse_args()`` internally, so we pass through the raw argv.
    # Instead of duplicating argument definitions, we let the sub-command
    # module handle its own argument parsing by slicing sys.argv.

    # -- run-env -------------------------------------------------------------
    run_env_parser = subparsers.add_parser(
        "run-env",
        help="Run an environment for data generation or preview.",
    )
    from embodichain.lab.scripts.run_env import cli as run_env_cli

    run_env_parser.set_defaults(func=run_env_cli)

    # -- Parse ---------------------------------------------------------------
    # If no sub-command is given, print help and exit.
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        parser.print_help()
        sys.exit(0)

    # Determine which sub-command was selected, then reconstruct argv so
    # that each sub-command's ``cli()`` can call ``parse_args()`` normally.
    known, _ = parser.parse_known_args()

    if hasattr(known, "func"):
        # Rewrite sys.argv so the sub-command's argparse sees only its own args.
        subcommand_argv = [f"embodichain {sys.argv[1]}"] + sys.argv[2:]
        original_argv = sys.argv
        sys.argv = subcommand_argv
        try:
            known.func()
        finally:
            sys.argv = original_argv
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
