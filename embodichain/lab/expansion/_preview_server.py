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

"""Loopback-only native preview transport. WebRTC integration remains a TODO."""

from __future__ import annotations

import base64
import json
import queue
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from ._preview_data import PreviewData, PreviewState
from ._preview_heatmap import object_heatmaps

__all__: list[str] = []


class PreviewTransport:
    def __init__(self, state: PreviewState, port: int) -> None:
        self.state = state
        self.commands: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=64)
        self.lock = threading.Lock()
        self.status: dict[str, Any] = {}
        self.jpeg = b""
        self.snapshot = b"{}"
        self.deviations = b"{}"
        html = Path(__file__).with_name("_preview.html").read_bytes()
        charts_js = Path(__file__).with_name("_preview_charts.js").read_bytes()
        heatmap_js = Path(__file__).with_name("_preview_heatmap.js").read_bytes()
        heatmaps = json.dumps(
            object_heatmaps(state.data), allow_nan=False, separators=(",", ":")
        ).encode()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def setup(self) -> None:
                super().setup()
                self.connection.settimeout(3)

            def log_message(self, *_: object) -> None:
                pass

            def do_GET(self) -> None:
                if urlsplit("//" + self.headers.get("Host", "")).hostname not in {
                    "127.0.0.1",
                    "localhost",
                    "::1",
                }:
                    self.send_error(403)
                    return
                route = self.path.split("?")[0]
                with owner.lock:
                    if route == "/":
                        data, kind = html, "text/html; charset=utf-8"
                    elif route == "/preview-charts.js":
                        data, kind = charts_js, "text/javascript; charset=utf-8"
                    elif route == "/preview-heatmap.js":
                        data, kind = heatmap_js, "text/javascript; charset=utf-8"
                    elif route == "/object-heatmaps":
                        data, kind = heatmaps, "application/json"
                    elif route == "/state":
                        data, kind = (
                            json.dumps(owner.status, allow_nan=False).encode(),
                            "application/json",
                        )
                    elif route == "/snapshot":
                        data, kind = owner.snapshot, "application/json"
                    elif route == "/deviations":
                        data, kind = owner.deviations, "application/json"
                    elif route == "/frame.jpg":
                        data, kind = owner.jpeg, "image/jpeg"
                    else:
                        self.send_error(404)
                        return
                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                try:
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_POST(self) -> None:
                if self.path != "/control":
                    self.send_error(404)
                    return
                origin = self.headers.get("Origin")
                host = self.headers.get("Host", "")
                # Same-origin loopback controls also work through a different forwarded port.
                if (
                    urlsplit("//" + host).hostname
                    not in {"127.0.0.1", "localhost", "::1"}
                    or self.headers.get("Content-Type") != "application/json"
                    or (origin and origin != "http://" + host)
                ):
                    self.send_error(403)
                    return
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 0 < size <= 4096:
                        raise ValueError("control size")
                    update = owner.state.validate(json.loads(self.rfile.read(size)))
                    owner.commands.put_nowait(update)
                except (ValueError, TypeError, queue.Full, TimeoutError):
                    self.send_error(400)
                    return
                self.send_response(204)
                self.end_headers()

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_port}/"

    def publish(self, status: dict[str, Any], jpeg: bytes) -> None:
        snapshot = json.dumps(
            {
                **status,
                "image": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode(),
            },
            allow_nan=False,
        ).encode()
        with self.lock:
            self.status = status
            self.jpeg = jpeg
            self.snapshot = snapshot


def serve_preview(
    data: PreviewData,
    *,
    port: int = 8769,
    gpu: int = 0,
    open_browser: bool = False,
    duration: float = 0,
) -> None:
    from ._preview_deviations import deviation_curves
    from ._preview_render import native_renderer

    state = PreviewState(data)
    transport = PreviewTransport(state, port)
    thread = None
    try:
        with native_renderer(data, gpu) as renderer:
            transport.deviations = json.dumps(
                deviation_curves(data, renderer.link_names),
                allow_nan=False,
                separators=(",", ":"),
            ).encode()
            frame = 0

            def publish() -> None:
                nonlocal frame
                jpeg = renderer.draw(state)
                transport.publish(
                    {
                        **state.snapshot(),
                        **renderer.stats,
                        "frame": frame,
                        "renderer": "EmbodiChain / DexSim Hybrid + OptiX",
                        "transport": "HTTP JPEG",
                    },
                    jpeg,
                )
                frame += 1

            publish()
            thread = threading.Thread(
                target=transport.server.serve_forever, daemon=True
            )
            thread.start()
            print(
                json.dumps(
                    {
                        "state": "ready",
                        "url": transport.url,
                        "episodes": len(data.clips),
                        "physics_steps": 0,
                        "transport": "HTTP JPEG",
                    }
                ),
                flush=True,
            )
            if open_browser:
                webbrowser.open(transport.url)
            start = last = time.monotonic()
            while not duration or time.monotonic() - start < duration:
                now = time.monotonic()
                elapsed = now - last
                last = now
                dirty = False
                while True:
                    try:
                        state.apply(transport.commands.get_nowait())
                        dirty = True
                    except queue.Empty:
                        break
                if state.values["playing"]:
                    state.values["progress"] = min(
                        1.0, state.values["progress"] + elapsed / state.duration
                    )
                    if state.values["progress"] >= 1:
                        state.values["playing"] = False
                    dirty = True
                if dirty:
                    publish()
                time.sleep(max(0.01, 0.1 - (time.monotonic() - now)))
    finally:
        if thread is not None:
            transport.server.shutdown()
            thread.join(timeout=5)
        transport.server.server_close()
