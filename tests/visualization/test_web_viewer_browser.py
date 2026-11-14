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

Video acceptance requires Playwright and WEBRTC_TEST_CHROME (or google-chrome
on PATH). Native catalog and camera-lifecycle cases do not join RTC media.
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
                description = page.evaluate("RTC_ReadSceneDescription()")
                assert len(description["sources"]) == num_envs + 1
                assert all(
                    source["input"] is False for source in description["sources"]
                )

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
                before_input = ready()
                page.evaluate("""async () => {
                    // Bypass the supplied page's Input:false flag and HTTP
                    // command route. The server must reject native RTC input.
                    const description = await RTC_ReadSceneDescription();
                    const peer = window.RTC_ViewPeer;
                    const stream = description.sources.findIndex(
                        source => source.name === state.camera_name);
                    if (stream < 0 || peer.Channel?.readyState !== 'open')
                        throw new Error('Native observer data channel unavailable');
                    if (!Number.isSafeInteger(peer.InputSequence) || peer.InputSequence < 0)
                        throw new Error('Native input sequence unavailable');
                    const previousStream = peer.InputStream;
                    const previousPermissions = peer.Config.InputStreams;
                    const sequence = peer.InputSequence;
                    try {
                        peer.InputStream = stream;
                        peer.Config.InputStreams = description.sources.map(() => true);
                        // Use the SDK encoder and monotonic sequence. Its wheel
                        // delta occupies X; Y must stay zero for a valid packet.
                        if (!peer.SendInput(4, 0, 4, 0))
                            throw new Error('SDK did not submit native wheel input');
                    } finally {
                        peer.InputStream = previousStream;
                        peer.Config.InputStreams = previousPermissions;
                    }
                    if (peer.InputSequence !== sequence + 1)
                        throw new Error('Native input sequence did not advance');
                }""")
                current = _wait(
                    ready,
                    lambda value: value
                    and value["frames"] > before_input["frames"] + 2,
                )
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


def test_native_web_catalog_disables_input_without_joining_media():
    """Qualify the native policy even when the streaming artifact cannot load."""
    environment = dict(os.environ)
    environment.pop("DEXSIM_WEBRTC_LIBRARY", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json
from pathlib import Path
import resource
import socket
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
import urllib.request
from unittest.mock import patch

import dexsim

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import RenderCfg
from embodichain.lab.visualization import DexSimWebCfg, VisualizationCfg

resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
with socket.socket() as candidate:
    candidate.bind(("127.0.0.1", 0))
    port = candidate.getsockname()[1]
with TemporaryDirectory() as cache:
    manager = SimpleNamespace(
        _material_cache_dir=Path(cache),
        physics=SimpleNamespace(configure_world=lambda *_: None),
    )
    config = SimulationManagerCfg(
        web=True, startup_summary="off", render_cfg=RenderCfg(renderer="hybrid"),
        visualization=VisualizationCfg(web=DexSimWebCfg(port=port)),
    )
    # The native catalog does not load the RTC library until a peer joins.
    # Bypass only the media preflight; retain the production input policy.
    with patch("embodichain.lab.visualization._web_viewer._require_webrtc_runtime"):
        native_config = SimulationManager._convert_sim_config(manager, config)
    world = dexsim.World(native_config)
    world.set_manual_update(True)
    cameras = []
    try:
        arena = world.get_env()
        for name in ("__embodichain_web_viewer", "task_camera_view1"):
            camera = arena.create_camera(name, 64, 64, True, dexsim.render.ViewFlags.COLOR)
            camera.open_camera()
            camera.disable_control()
            camera.render(dexsim.render.ViewFlags.COLOR, dexsim.render.ViewFlags.NULL)
            cameras.append(camera)
        deadline = time.monotonic() + 10
        description = {}
        while time.monotonic() < deadline:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/stream", timeout=1) as response:
                description = json.load(response)
            if len(description.get("sources", [])) == 2:
                break
            time.sleep(0.05)
        sources = description.get("sources", [])
        assert len(sources) == 2, description
        assert all(source["input"] is False for source in sources), description
        print("Native camera catalog denies browser input.", flush=True)
    finally:
        for camera in cameras:
            camera.close_camera()
        cameras.clear()
        world.quit()
""",
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Native camera catalog denies browser input." in result.stdout


def test_native_observer_reopens_after_camera_group_removal():
    """Exercise native camera ownership without loading WebRTC networking."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import resource
from types import SimpleNamespace
from unittest.mock import patch

import dexsim

from embodichain.lab.visualization import DexSimWebCfg
from embodichain.lab.visualization._web_viewer import _WebViewer

resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


class InertApplication:
    def __init__(self, *_):
        pass

    def process(self, *_):
        pass

    def close(self):
        pass


cfg = dexsim.WorldConfig()
cfg.open_windows = False
cfg.webrtc_enabled = False
cfg.renderer = dexsim.types.Renderer.FASTRT
cfg.backend = dexsim.types.Backend.VULKAN
cfg.log_startup_info = False
world = dexsim.World(cfg)
world.set_manual_update(True)
sim = SimpleNamespace(
    sim_config=SimpleNamespace(
        visualization=SimpleNamespace(web=DexSimWebCfg(width=64, height=64)),
    ),
    _web_camera_group=None,
    spawn_result=None,
    get_env=world.get_env,
    get_world=lambda: world,
)
viewer = None
try:
    # Only the application/network facade is inert. Group allocation, camera
    # creation, rendering and removal all use the selected unmodified SDK.
    with patch.object(dexsim, "WebApplication", InertApplication):
        for _ in range(3):
            viewer = _WebViewer(sim)
            assert viewer.capture(force=True)
            assert viewer.frames == 1
            viewer.close()
finally:
    if viewer is not None:
        viewer.close()
    sim._web_camera_group = None
    world.quit()
print("Native observer reopened and rendered three times.", flush=True)
""",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "reopened and rendered three times" in result.stdout
