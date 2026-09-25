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

"""Command-line entry points for offline diversity analysis and task acceptance."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected analysis operation with lazy optional imports."""
    parser = argparse.ArgumentParser(prog="embodichain analyze-data")
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser(
        "collect-preview", help="Collect the existing Franka repeated pick/place task."
    )
    collect.add_argument("--output", required=True)
    collect.add_argument("--count", type=int, default=12)
    collect.add_argument("--seed", type=int, default=17)
    worker = commands.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("request")
    serve = commands.add_parser(
        "serve", help="Open the Gradio/Viser analysis workbench."
    )
    serve.add_argument("--catalog", required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=7865)
    info = commands.add_parser("summary")
    info.add_argument("--catalog", required=True)
    ingest = commands.add_parser(
        "import", help="Import analysis JSONL or historical LeRobot metadata."
    )
    ingest.add_argument("--catalog", required=True)
    ingest.add_argument("--source", required=True)
    ingest.add_argument("--format", choices=["jsonl", "lerobot"], default="lerobot")
    args = parser.parse_args(argv)
    if args.command == "collect-preview":
        from .collection import collect_preview

        print(
            json.dumps(
                collect_preview(args.output, count=args.count, seed=args.seed), indent=2
            )
        )
    elif args.command == "_worker":
        from .collection import collect_worker

        collect_worker(args.request)
    elif args.command == "serve":
        from .ui import build_app

        build_app(args.catalog).queue().launch(
            server_name=args.host, server_port=args.port, inbrowser=False
        )
    elif args.command == "import":
        from .catalog import Catalog

        with Catalog(args.catalog) as catalog:
            if args.format == "jsonl":
                result = catalog.import_jsonl(args.source)
            else:
                from .importers import import_lerobot_metadata

                result = {"inserted": 0, "updated": 0, "unchanged": 0}
                for record in import_lerobot_metadata(args.source):
                    result[catalog.upsert(record)] += 1
            print(json.dumps(result, indent=2))
    elif args.command == "summary":
        from .catalog import Catalog

        with Catalog(args.catalog) as catalog:
            print(json.dumps(catalog.summary(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
