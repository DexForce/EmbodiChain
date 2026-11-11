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

"""Launch the same annotation application with any image frame source."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import socket
import threading
from typing import Any, TYPE_CHECKING
import webbrowser

if TYPE_CHECKING:
    from fastapi import FastAPI
    from .frames import FrameSource

__all__ = ["create_annotation_app", "launch_annotation"]


def create_annotation_app(
    source_factory: Callable[[], FrameSource],
    annotations: str | Path,
    *,
    fitting_options: dict[str, Any] | None = None,
) -> FastAPI:
    """Create a source-independent FastAPI application for a single process.

    Args:
        source_factory: Create a fresh FrameSource on the app's worker thread.
            The app closes it on shutdown or failed preparation.
        annotations: Destination JSON path; persisted only after an edit/save.
        fitting_options: Initial joint/profile settings, with the same field
            names as the web fitting form. Omit to configure them in the browser.

    Returns:
        ASGI application with lifespan-managed frame input and preview cache.
    """
    from ._server import create_app

    return create_app(
        source_factory, Path(annotations), fitting_options=fitting_options
    )


def launch_annotation(
    source_factory: Callable[[], FrameSource],
    annotations: str | Path,
    *,
    port: int = 0,
    open_browser: bool = True,
    fitting_options: dict[str, Any] | None = None,
) -> None:
    """Run the local website until Ctrl+C, accepting any frame source factory.

    Args:
        source_factory: Create a fresh frame source on its owning worker thread.
        annotations: JSON output file, distinct from any source file.
        port: Loopback port; zero asks the OS to select an available port.
        open_browser: Open the website after the server starts.
        fitting_options: Initial joint/profile settings for offline reconstruction.

    Raises:
        ValueError: Port is outside 0–65535.
        OSError: The requested loopback port cannot be bound.
    """
    import uvicorn

    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError("端口必须在 0–65535 之间。")
    annotations = Path(annotations).resolve()
    server = uvicorn.Server(
        uvicorn.Config(
            create_annotation_app(
                source_factory, annotations, fitting_options=fitting_options
            ),
            host="127.0.0.1",
            port=port,
            access_log=False,
            log_level="warning",
        )
    )
    stopped = threading.Event()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", port))
        url = f"http://127.0.0.1:{listener.getsockname()[1]}"
        print(
            f"原子技能区间标注：{url}\n标注：{annotations}\n按 Ctrl+C 退出。",
            flush=True,
        )

        def browse() -> None:
            while not stopped.wait(0.05):
                if server.started:
                    webbrowser.open(url)
                    return

        if open_browser:
            threading.Thread(
                target=browse, name="annotation-browser", daemon=True
            ).start()
        try:
            server.run(sockets=[listener])
        except KeyboardInterrupt:
            pass
        finally:
            stopped.set()
