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
from typing import Any

import pytest


@pytest.fixture
def fitting_oracle(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Isolate host lifecycle from IK with known accepted/rejected numeric bundles.

    Native reconstruction itself is independently covered by the Cartesian robot
    tests. This oracle protects web timing, background ownership and export.
    """
    import numpy as np
    from embodichain.toolkits.skill_fitting import _pipeline

    control: dict[str, Any] = dict(
        accepted=True, entered=threading.Event(), release=threading.Event(), calls=[]
    )
    control["release"].set()

    def fit(inputs: Any, **kwargs: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        control["calls"].append(kwargs)
        control["entered"].set()
        assert control["release"].wait(
            timeout=10
        ), "Test did not release the fitting worker."
        if control.get("error"):
            raise ValueError("No observable closing event; provide phase hints.")
        # A 3 s recording, 100 Hz, with two known reference columns.
        times = np.arange(300, dtype=np.float64) / 100
        original = np.column_stack((times / 10, -times / 20))
        native = original.copy()
        native[20:84, 0] += 0.02
        proposed = original.copy()
        proposed[23:81, 0] += 0.01
        patched = proposed if control["accepted"] else original.copy()
        report = dict(
            schema_version=1,
            skill_id=kwargs["skill_id"],
            accepted=control["accepted"],
            failures=(
                [] if control["accepted"] else ["patched path exceeds joint limits"]
            ),
            joint_names=["arm_joint", "hand_joint"],
            start=20,
            stop=84,
            phases=[dict(name="approach", start=0, stop=64)],
            parameters=dict(lift_height=0.12),
            native_joint_rmse=0.02,
            tcp_errors={"primary": {"native_tcp_rmse_m": 0.003}},
            time_offset=inputs.time_offset,
            **kwargs["provenance"],
        )
        arrays = dict(timestamps=times, geometry_indices=np.arange(len(times)))
        for name, q in (
            ("original", original),
            ("native", native),
            ("patched", patched),
            ("proposed", proposed),
        ):
            arrays[f"{name}_qpos"] = q
            tcp = np.column_stack((q[:, 0], np.zeros(len(q)), 0.4 + q[:, 1]))
            arrays[f"{name}_tcp"] = tcp[:, None, :]
            arrays[f"{name}_links"] = np.stack(
                (np.zeros_like(tcp), tcp / 2, tcp), axis=1
            )[:, None, :, :]
        return arrays, report

    monkeypatch.setattr(_pipeline, "fit_recording", fit)
    return control


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
