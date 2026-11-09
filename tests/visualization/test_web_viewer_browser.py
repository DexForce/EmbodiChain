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

"""Opt-in GPU/Chrome acceptance; run serially with --run-gpu.

Requires Playwright and WEBRTC_TEST_CHROME (or google-chrome on PATH).
The native renderer needs the host's existing display/SDL environment.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

import pytest
import numpy as np

pytestmark = [pytest.mark.gpu, pytest.mark.no_sim]


def _wait(read, accept=bool, timeout=30):
    deadline = time.monotonic() + timeout
    value = None
    while time.monotonic() < deadline:
        value = read()
        if accept(value):
            return value
        time.sleep(0.1)
    raise AssertionError(f"Timed out: {value!r}")


@pytest.mark.parametrize("backend,num_envs", [("default", 2), ("newton", 1)])
def test_native_browser_view_and_sensor_isolation(tmp_path, backend, num_envs):
    import dexsim

    if not hasattr(dexsim, "WebApplication"):
        pytest.skip("WebRTC-enabled DexSim with WebApplication is required")
    playwright = pytest.importorskip("playwright.sync_api")
    chrome = os.environ.get("WEBRTC_TEST_CHROME") or shutil.which("google-chrome")
    if not chrome:
        pytest.skip("Chrome is required for native WebRTC acceptance")
    with socket.socket() as candidate:
        candidate.bind(("127.0.0.1", 0))
        port = candidate.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    log_path = tmp_path / "scene.log"
    result_path = tmp_path / "result.json"
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).with_name("_web_scene.py")),
                str(port),
                backend,
                str(result_path),
                str(num_envs),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    try:

        def ready():
            assert process.poll() is None, log_path.read_text(errors="replace")
            try:
                with urllib.request.urlopen(url + "/api/state", timeout=1) as response:
                    return json.load(response)
            except OSError:
                return None

        initial = _wait(
            ready, lambda value: value and "test_sim_step" in value, timeout=45
        )
        with playwright.sync_playwright() as service:
            browser = service.chromium.launch(
                executable_path=chrome,
                headless=True,
                args=["--no-sandbox", "--autoplay-policy=no-user-gesture-required"],
            )
            try:
                page = browser.new_page(viewport={"width": 1280, "height": 800})
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(url)
                _wait(
                    lambda: page.evaluate(
                        "Boolean(window.RTC_ViewPeer?.ViewReady && typeof state !== 'undefined')"
                    )
                )
                _wait(lambda: page.locator("#waiting").is_hidden())
                pixels = page.evaluate("""() => {
                    const v=document.querySelector('#main-view video'), c=document.createElement('canvas');
                    c.width=64;c.height=36;const ctx=c.getContext('2d');ctx.drawImage(v,0,0,64,36);
                    const a=[...ctx.getImageData(0,0,64,36).data].filter((_,i)=>i%4!==3);
                    return Math.max(...a)-Math.min(...a);
                }""")
                assert (
                    pixels > 20
                )  # Nonuniform rendered scene, not the initial black frame.
                assert page.locator("#sensor-views video").count() == num_envs
                assert (
                    page.locator("#sensor-views video").first.get_attribute("tabindex")
                    == "-1"
                )
                before = page.evaluate("state.camera_pose")
                page.locator("#main-view video").click()
                page.mouse.wheel(0, -200)
                _wait(
                    lambda: page.evaluate("state.camera_pose"),
                    lambda pose: pose != before,
                    timeout=10,
                )
                page.get_by_role("button", name="Release control", exact=True).click()
                page.get_by_label("Select and move scene objects").check()
                _wait(lambda: page.evaluate("state.gizmo"))
                page.get_by_role("button", name="Reset view", exact=True).click()
                _wait(
                    lambda: page.evaluate("state.camera_pose"),
                    lambda pose: pose == initial["camera_pose"],
                )

                def projected(point):
                    current = ready()
                    local = (
                        np.linalg.inv(np.asarray(current["camera_pose"]))
                        @ np.r_[point, 1.0]
                    )
                    k = np.asarray(current["test_intrinsic"])
                    return [
                        (k[0, 0] * local[0] / -local[2] + k[0, 2]) / 640,
                        (-k[1, 1] * local[1] / -local[2] + k[1, 2]) / 360,
                    ]

                cube_before = np.asarray(ready()["test_cube_pose"])[0]
                offset = np.asarray(ready()["test_arena_offsets"][0])
                x, y = projected(cube_before[:3, 3] + offset)
                page.evaluate(
                    """([x,y])=>{const p=RTC_ViewPeer;
                    p.InputStream=p.Config.StreamNames.indexOf(state.camera_name);
                    p.SendInput(2,1,x,y);p.SendInput(3,1,x,y)}""",
                    [x, y],
                )
                _wait(ready, lambda value: value["test_gizmo_count"] == 1)
                page.screenshot(path=str(tmp_path / "gizmo.png"))
                x, y = projected(cube_before[:3, 3] + offset + [0.32, 0, 0])
                ex, ey = projected(cube_before[:3, 3] + offset + [0.52, 0, 0])
                page.evaluate("([x,y])=>RTC_ViewPeer.SendInput(2,1,x,y)", [x, y])
                for i in range(1, 6):
                    page.evaluate(
                        "([x,y])=>RTC_ViewPeer.SendInput(1,0,x,y)",
                        [x + (ex - x) * i / 5, y + (ey - y) * i / 5],
                    )
                    time.sleep(0.05)
                page.evaluate("([x,y])=>RTC_ViewPeer.SendInput(3,1,x,y)", [ex, ey])
                _wait(
                    ready,
                    lambda value: not np.allclose(
                        value["test_cube_pose"][0], cube_before
                    ),
                    timeout=5,
                )
                page.evaluate("RTC_ViewPeer.ReleaseInput()")
                page.get_by_label("Select and move scene objects").uncheck()
                _wait(lambda: page.evaluate("state.gizmo"), lambda enabled: not enabled)
                page.get_by_label("Pause viewer rendering (physics continues)").check()
                _wait(lambda: page.evaluate("state.paused"))
                time.sleep(0.6)
                paused_frames = ready()["frames"]
                time.sleep(0.3)
                assert ready()["frames"] == paused_frames
                page.get_by_role("button", name="Render one frame", exact=True).click()
                _wait(ready, lambda state: state["frames"] == paused_frames + 1)
                page.get_by_role("button", name="Reset view", exact=True).click()
                _wait(
                    lambda: page.evaluate("state.camera_pose"),
                    lambda pose: pose == initial["camera_pose"],
                )
                with page.expect_download() as download:
                    page.get_by_role("button", name="Save image", exact=True).click()
                download.value.save_as(tmp_path / "capture.png")
                assert (tmp_path / "capture.png").read_bytes().startswith(b"\x89PNG")
                page.screenshot(path=str(tmp_path / "viewer.png"))
                page.reload()
                _wait(
                    lambda: page.evaluate(
                        "Boolean(window.RTC_ViewPeer?.ViewReady && typeof state !== 'undefined')"
                    )
                )
                current = ready()
                assert current["run_id"] == initial["run_id"]
                assert current["test_sensor_pose"] == initial["test_sensor_pose"]
                assert current["test_sim_step"] == 1
                assert not errors
            finally:
                browser.close()
    finally:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
            pytest.fail("Native WebRTC teardown timed out")
    assert process.returncode == 0, log_path.read_text(errors="replace")
    result = json.loads(result_path.read_text())
    assert result["sensor_unchanged"] and result["physics_steps"] == 1
    assert result["frames"] > 0
    with socket.socket() as released:
        released.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        released.bind(("127.0.0.1", port))


def test_web_does_not_restrict_task_sensor_construction(tmp_path):
    import dexsim

    if not hasattr(dexsim, "WebApplication"):
        pytest.skip("WebRTC-enabled DexSim is required")
    with socket.socket() as candidate:
        candidate.bind(("127.0.0.1", 0))
        port = candidate.getsockname()[1]
    result_path = tmp_path / "sensors.json"
    process = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("_web_scene.py")),
            str(port),
            "default",
            str(result_path),
            "8",
            "sensors",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    assert json.loads(result_path.read_text()) == {
        "num_envs": 8,
        "depth_size": [65, 49],
        "stereo_instances": 8,
    }
