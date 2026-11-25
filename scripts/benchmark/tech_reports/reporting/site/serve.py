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

"""Serve only the exported site, including conditional MP4 byte-range requests."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from flask import Flask, abort, send_file
from werkzeug.serving import make_server


def create_app(directory: Path) -> Flask:
    """Create a read-only static app with no workspace or directory exposure."""
    root = directory.resolve(strict=True)
    app = Flask(__name__, static_folder=None)

    @app.route("/", defaults={"path": "index.html"}, methods=["GET"])
    @app.route("/<path:path>", methods=["GET"])
    def file(path: str):
        target = (root / path).resolve()
        if target.is_dir():
            target /= "index.html"
        if not target.is_relative_to(root) or not target.is_file():
            abort(404)
        response = send_file(target, conditional=True, max_age=0)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        return response

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    with make_server(
        args.host, args.port, create_app(args.directory), threaded=True
    ) as server:
        print(
            f"Simulation reports: http://{args.host}:{args.port} ({args.directory.resolve()})",
            flush=True,
        )
        server.serve_forever()
