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


@pytest.mark.parametrize(
    "backend,num_envs,renderer",
    [
        ("default", 2, "hybrid"),
        ("newton", 1, "hybrid"),
        ("default", 1, "fast-rt"),
        ("default", 1, "rt"),
    ],
)
def test_native_browser_view_and_sensor_isolation(
    tmp_path, backend, num_envs, renderer
):
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
                "view",
                renderer,
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

                def video_ready():
                    assert not errors, errors
                    return page.evaluate(
                        "Boolean(window.RTC_ViewPeer?.ViewReady && !window.RTC_ViewPeer.Stopped && typeof state !== 'undefined')"
                    )

                _wait(video_ready)
                _wait(
                    lambda: page.evaluate(
                        "Boolean(document.querySelector('#main-view video')?.readyState >= 2)"
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
                assert page.locator("#initialization-summary").is_visible()
                table = page.locator("#initialization-summary").inner_text()
                assert "EmbodiChain" in page.title()
                assert "Physics timestep" in table and "640 × 360" in table
                assert "READY" in table and "task_camera" in table
                assert initial["read_only"] is True
                assert (
                    page.locator("aside button, aside input, aside select").count() == 0
                )
                assert page.evaluate("window.RTC_ViewPeer.Config.Input") is False

                # Observation gestures and forged old commands cannot edit the
                # scene or advance physics through this information page.
                page.locator("#main-view video").click()
                page.mouse.wheel(0, -200)
                page.keyboard.press("KeyR")
                status = page.evaluate("""async () => {
                    const peer = window.RTC_ViewPeer;
                    const response = await fetch('/api/command', {
                        method: 'POST', headers: {'Content-Type':'application/json'},
                        body: JSON.stringify({action:'key',key:21,down:true,
                            run_id:state.run_id,revision:state.revision,
                            room:peer.Config.Room,token:peer.Token,after:peer.After})
                    });
                    return response.status;
                }""")
                assert status == 403
                current = ready()
                assert current["test_camera_pose"] == initial["test_camera_pose"]
                assert current["test_cube_pose"] == initial["test_cube_pose"]
                assert current["frames"] > initial["frames"]
                page.screenshot(path=str(tmp_path / "viewer.png"))
                page.reload()
                _wait(
                    lambda: page.evaluate(
                        "Boolean(window.RTC_ViewPeer?.ViewReady && !window.RTC_ViewPeer.Stopped && typeof state !== 'undefined')"
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
