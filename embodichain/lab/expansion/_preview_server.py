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

"""Serve recorded expansion playback through the native WebRTC viewport."""

from __future__ import annotations

from collections import deque
import json
import socket
import time
import webbrowser
from typing import TYPE_CHECKING
from ._frontend import _preview_assets
from ._preview_data import PreviewData, PreviewState
from ._preview_heatmap import object_heatmaps

if TYPE_CHECKING:
    from embodichain.lab.visualization._web_viewer import _WebViewer

__all__: list[str] = []


def _json(value: object) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"))


class _WebPreviewTransport:
    """Serve replay controls and statistics over the native video listener."""

    def __init__(
        self, state: PreviewState, viewer: _WebViewer, deviations: bytes
    ) -> None:
        self.state = state
        self.commands: deque[dict[str, object]] = deque()
        self.url = viewer.endpoint
        self.status = "{}"
        assets = _preview_assets()
        assets["/object-heatmaps"] = (
            "application/json",
            _json(object_heatmaps(state.data)).encode(),
        )
        assets["/deviations"] = ("application/json", deviations)
        viewer._replace_application(assets, self._request)

    def _request(self, path: str, raw: str) -> tuple[int, str]:
        if path == "/api/state":
            return 200, self.status
        if path != "/api/command":
            return 404, '{"error":"unknown route"}'
        # DexSim has already checked same-origin and the current viewer token.
        if len(self.commands) >= 64:
            return 429, '{"error":"replay control queue is full"}'
        try:
            if len(raw) > 4096:
                raise ValueError("control size")
            update = self.state.validate(json.loads(raw))
        except (ValueError, TypeError):
            return 400, '{"error":"invalid replay control"}'
        self.commands.append(update)
        return 200, "{}"

    def publish(self, status: dict[str, object]) -> None:
        self.status = _json(status)


def serve_preview(
    data: PreviewData,
    *,
    port: int = 8769,
    gpu: int = 0,
    open_browser: bool = False,
    duration: float = 0,
) -> None:
    """Replay confirmed episodes in a browser using the native observer."""
    from ._preview_deviations import deviation_curves
    from ._preview_render import native_renderer

    if port == 0:
        # The native World configuration requires a concrete listening port.
        with socket.socket() as probe:
            probe.bind(("0.0.0.0", 0))
            port = probe.getsockname()[1]

    state = PreviewState(data)
    with native_renderer(data, gpu, web_port=port) as renderer:
        viewer = renderer.web_viewer
        deviations = _json(deviation_curves(data, renderer.link_names)).encode()
        transport = _WebPreviewTransport(state, viewer, deviations)
        frame_period = 1 / viewer.cfg.fps
        frame = 0

        def publish() -> None:
            nonlocal frame
            renderer.draw(state)
            transport.publish(
                {
                    **state.snapshot(),
                    **renderer.stats,
                    "frame": frame,
                    "transport": "WebRTC",
                }
            )
            frame += 1

        publish()
        print(
            json.dumps(
                {
                    "state": "ready",
                    "url": transport.url,
                    "episodes": len(data.clips),
                    "physics_steps": 0,
                    "transport": "WebRTC",
                }
            ),
            flush=True,
        )
        if open_browser:
            webbrowser.open(transport.url)

        start = last = next_frame = time.monotonic()
        next_idle_frame = start + 0.25
        dirty = False
        while not duration or time.monotonic() - start < duration:
            now = time.monotonic()
            elapsed = now - last
            last = now
            while transport.commands:
                state.apply(transport.commands.popleft())
                dirty = True
            if state.values["playing"]:
                state.values["progress"] = min(
                    1.0, state.values["progress"] + elapsed / state.duration
                )
                if state.values["progress"] >= 1:
                    state.values["playing"] = False
                dirty = True
            if now >= next_frame and (dirty or now >= next_idle_frame):
                # Capturing the native observer also services its app requests.
                publish()
                dirty = False
                next_frame = now + frame_period
                next_idle_frame = now + 0.25
            else:
                viewer._process_application_requests()
            wait = min(0.01, next_frame - time.monotonic()) if dirty else 0.01
            time.sleep(max(0.001, wait))
