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

"""Small real H.264 inputs and a lifecycle-owned local annotation server."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from fractions import Fraction
from pathlib import Path
import socket
import threading
import time

import pytest


@pytest.fixture
def make_video(tmp_path: Path) -> Callable[..., Path]:
    """Encode identifiable frames with B frames and optional VFR/nonzero PTS."""
    av = pytest.importorskip("av")
    import numpy as np
    import cv2

    def make(
        name: str = "episode.mp4",
        count: int = 48,
        variable: bool = False,
        start: int = 0,
        width: int = 160,
        height: int = 96,
    ) -> Path:
        path = tmp_path / name
        with av.open(str(path), "w") as output:
            stream = output.add_stream("libx264", rate=30)
            stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
            stream.options = {"g": "12", "bf": "2", "crf": "18"}
            stream.time_base = stream.codec_context.time_base = Fraction(1, 30)
            for index in range(count):
                image = np.full(
                    (height, width, 3),
                    (index * 31 % 256, index * 13 % 256, index * 7 % 256),
                    dtype=np.uint8,
                )
                cv2.putText(
                    image,
                    str(index),
                    (12, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                frame = av.VideoFrame.from_ndarray(image, format="rgb24")
                frame.pts = start + index + (index // 5 if variable else 0)
                frame.time_base = Fraction(1, 30)
                for packet in stream.encode(frame):
                    output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)
        return path

    return make


@pytest.fixture
def annotation_server(
    make_video: Callable[..., Path], tmp_path: Path
) -> Iterator[tuple[str, Path]]:
    """Serve a real video for browser checks, cleaning up both owning threads."""
    uvicorn = pytest.importorskip("uvicorn")
    pytest.importorskip("fastapi")
    import httpx
    from embodichain.toolkits.trajectory_annotator._cfg import AnnotationCfg
    from embodichain.toolkits.trajectory_annotator._server import create_app

    from embodichain.toolkits.trajectory_annotator.video import VideoFrameSource

    video = make_video(count=90, variable=True, start=90)
    destination = tmp_path / "annotations.json"
    cfg = AnnotationCfg(autosave_ms=150)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        url = f"http://127.0.0.1:{listener.getsockname()[1]}"
        server = uvicorn.Server(
            uvicorn.Config(
                create_app(lambda: VideoFrameSource(video), destination, cfg=cfg),
                log_level="error",
            )
        )
        thread = threading.Thread(
            target=lambda: server.run(sockets=[listener]), daemon=True
        )
        thread.start()
        try:
            deadline = time.monotonic() + 15
            while not server.started:
                if not thread.is_alive() or time.monotonic() > deadline:
                    pytest.fail("Annotation HTTP server did not start.")
                time.sleep(0.01)
            with httpx.Client(base_url=url) as client:
                while True:
                    status = client.get("/api/status").json()
                    if status["phase"] == "ready":
                        break
                    if status["phase"] == "error" or time.monotonic() > deadline:
                        pytest.fail(f"Video initialization failed: {status}")
                    time.sleep(0.01)
            yield url, destination
        finally:
            server.should_exit = True
            thread.join(timeout=15)
            assert not thread.is_alive(), "Annotation server leaked its owning thread."
