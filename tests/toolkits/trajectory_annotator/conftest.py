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
def fitting_oracle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    """Isolate host lifecycle from IK with known accepted/rejected numeric bundles.

    Native reconstruction itself is independently covered by the Cartesian robot
    tests. This oracle protects web timing, background ownership and export.
    """
    import numpy as np
    from embodichain.toolkits.skill_fitting import _pipeline

    # Real lightweight geometry keeps browser playback tests independent of IK,
    # while still exercising Viser's model, WebSocket and shutdown lifecycle.
    urdf = tmp_path / "viewer.urdf"
    urdf.write_text("""<robot name="comparison">
      <link name="base"><visual><geometry><box size="0.1 0.1 0.1"/></geometry>
        <material name="body"><color rgba="0.8 0.3 0.1 1"/></material></visual></link>
      <link name="arm"><visual><geometry><box size="0.05 0.05 0.4"/></geometry></visual></link>
      <link name="hand"><visual><geometry><box size="0.08 0.05 0.04"/></geometry></visual></link>
      <joint name="arm_joint" type="prismatic"><parent link="base"/><child link="arm"/>
        <origin xyz="0 0 0.2"/><axis xyz="1 0 0"/><limit lower="-1" upper="1" effort="10" velocity="2"/></joint>
      <joint name="hand_joint" type="prismatic"><parent link="arm"/><child link="hand"/>
        <origin xyz="0 0 0.2"/><axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="10" velocity="2"/></joint>
    </robot>""")

    control: dict[str, Any] = dict(
        accepted=True, entered=threading.Event(), release=threading.Event(), calls=[]
    )
    control["release"].set()
    import json

    times = np.arange(300, dtype=np.float64) / 100
    original = np.column_stack((times / 10, -times / 20))
    joints_path, profile_path = tmp_path / "joints.npz", tmp_path / "profile.json"
    np.savez(joints_path, qpos=original, timestamps=times)
    profile = dict(
        joint_names=["arm_joint", "hand_joint"],
        control_parts={"arm": ["arm_joint"], "hand": ["hand_joint"]},
        solvers={
            "arm": dict(urdf_path=str(urdf), root_link_name="base", end_link_name="arm")
        },
        base_poses={},
        participants={},
        hand_commands={"hand": {"open": [0.0], "grasp": [-0.1]}},
        lower=[-1.0, -1.0],
        upper=[1.0, 1.0],
        velocity_limits=[2.0, 2.0],
        acceleration_limits=[20.0, 20.0],
        calibration_revision="known_fixture",
    )
    profile_path.write_text(json.dumps(profile))
    control.update(
        joints=str(joints_path), profile=str(profile_path), original=original
    )

    def fit(inputs: Any, **kwargs: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        control["calls"].append(kwargs)
        control["entered"].set()
        assert control["release"].wait(
            timeout=10
        ), "Test did not release the fitting worker."
        if control.get("error"):
            raise ValueError("No observable closing event; provide phase hints.")
        # A 3 s recording, 100 Hz, with two known reference columns.
        start = int(np.searchsorted(times, kwargs["frame_times"][0] - 1e-7))
        stop = int(
            np.searchsorted(
                times, kwargs["frame_times"][-1] + kwargs["frame_durations"][-1] - 1e-7
            )
        )
        native = original.copy()
        native[start:stop, 0] += 0.02
        proposed = original.copy()
        proposed[start + 3 : stop - 3, 0] += 0.01
        patched = proposed if control["accepted"] else original.copy()
        report = dict(
            schema_version=1,
            skill_id=kwargs["skill_id"],
            accepted=control["accepted"],
            failures=(
                [] if control["accepted"] else ["patched path exceeds joint limits"]
            ),
            joint_names=["arm_joint", "hand_joint"],
            start=start,
            stop=stop,
            phases=[dict(name="approach", start=0, stop=stop - start)],
            parameters=dict(lift_height=0.12),
            native_goal=dict(grasp_xpos=np.eye(4).tolist()),
            native_options=dict(lift_height=0.12),
            native_motion_policy=dict(
                strategy="motion_gen",
                sample_count=80,
                acceleration_limit=0.5,
                control_dt=0.01,
            ),
            native_joint_rmse=0.02,
            tcp_errors={"primary": {"native_tcp_rmse_m": 0.003}},
            selected_participants={"primary": {"motion": "arm", "grasp": "hand"}},
            time_offset=inputs.time_offset,
            robot_profile=profile,
            replay_interpolation_factor=control.get("replay_interpolation_factor", 1),
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
