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

from __future__ import annotations

import json
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from embodichain.lab.sim import SimulationManagerCfg
from embodichain.lab.visualization import DexSimWebCfg
from embodichain.lab.visualization._web_viewer import _WebViewer
from embodichain.lab.visualization._web_viewer import _require_webrtc_runtime


@pytest.fixture
def viewer():
    value = object.__new__(_WebViewer)
    value.sim = SimpleNamespace(
        sim_config=SimulationManagerCfg(web=True),
        device=torch.device("cpu"),
        physics=SimpleNamespace(name="default", step=Mock()),
        _requested_renderer="hybrid",
        is_window_opened=False,
        _visualization_topology_revision=3,
        spawn_result=SimpleNamespace(needs_rebuild=False, topology_revision=1),
        _ready_spawn_topology_revision=1,
        _ensure_render_state=Mock(),
        update_gizmos=Mock(),
        _render_state_published=True,
        num_envs=1,
        _robots={},
        _articulations={},
        _rigid_objects={"cube": object()},
        _rigid_object_groups={},
        _deformable_objects={},
        _sensors={},
        _default_plane=object(),
    )
    value.cfg = DexSimWebCfg(width=640, height=360)
    value.sim._web_viewer = value
    value.owner = threading.get_ident()
    value.run_id = "current-run"
    value.camera = Mock()
    value.camera.get_world_pose.return_value = np.eye(4)
    value.title = "EmbodiChain"
    value.orbit = np.array(value.cfg.target)
    value.group = object()
    value.application = Mock()
    value.arena = Mock()
    value.error = None
    value.frames = 0
    value.render_seconds = 0.0
    value.next_render = float("inf")
    value.dirty = False
    value.closed = False
    return value


def test_information_capture_pumps_requests_without_mutating_the_scene(viewer):
    assert not viewer.capture()
    viewer.application.process.assert_called_once()
    viewer.sim._ensure_render_state.assert_not_called()
    assert viewer.capture(force=True)
    viewer.sim._ensure_render_state.assert_called_once()
    viewer.camera.render.assert_called_once()
    assert viewer.frames == 1
    assert viewer.sim._render_state_published is True
    viewer.sim.physics.step.assert_not_called()
    viewer.sim.update_gizmos.assert_not_called()
    assert not viewer.capture()


def test_render_only_scene_draws_without_preparing_or_stepping(viewer):
    viewer.sim.spawn_result = None
    assert viewer.capture(force=True)
    viewer.sim._ensure_render_state.assert_not_called()
    viewer.camera.render.assert_called_once()


def test_render_failure_keeps_serving_information_without_retrying_renderer(viewer):
    from embodichain.lab.sim.sim_manager import SimulationManager

    sim = object.__new__(SimulationManager)
    sim.sim_config = viewer.sim.sim_config
    sim._web_viewer = viewer
    sim._visualization_error_reported = False
    sim.start_visualization = Mock()
    viewer.camera.render.side_effect = RuntimeError("observer render failed")

    sim.capture_visualization_safely(force=True)

    responses = []
    viewer.application.process.side_effect = lambda handle: responses.append(
        handle("/api/state", "")
    )
    sim.capture_visualization_safely(force=True)

    assert viewer.application.process.call_count == 2
    assert responses[0][0] == 200
    assert json.loads(responses[0][1])["error"] == "observer render failed"
    viewer.camera.render.assert_called_once()
    viewer.sim.physics.step.assert_not_called()


def test_information_table_shares_terminal_rows_and_tracks_scene_readiness(viewer):
    code, body = viewer._request("/api/state", "")
    assert code == 200
    state = json.loads(body)
    assert state["read_only"] is True
    rows = state["summary"]["rows"]
    assert ["Rendering", "Viewer resolution", "640 × 360"] in rows
    assert ["Physics", "Physics timestep", "10 ms (100 Hz)"] in rows
    assert ["Scene / env", "Rigid objects", "1"] in rows
    assert ["Scene", "State", "READY"] in rows
    viewer.sim.spawn_result.needs_rebuild = True
    assert ("Scene", "State", "PENDING (not prepared)") in viewer.state()["summary"][
        "rows"
    ]
    viewer.sim._ensure_render_state.assert_not_called()
    viewer.sim.physics.step.assert_not_called()


def test_information_table_respects_full_and_disabled_summary_modes(viewer):
    viewer.sim.sim_config.startup_summary = "full"
    assert any(
        row[0] == "Rendering detail" for row in viewer.state()["summary"]["rows"]
    )
    viewer.sim.sim_config.startup_summary = "off"
    assert viewer.state()["summary"] is None


def test_environment_initialization_snapshot_is_displayed(viewer):
    snapshot = {
        "title": "Environment initialized: ExampleEnv",
        "rows": [("Environment", "Seed", "42"), ("Managers", "Actions", "1")],
    }
    viewer.sim._web_initialization_summary = snapshot
    assert viewer.state()["summary"] == snapshot


@pytest.mark.parametrize(
    "action", ["target", "joint", "seek", "key", "gizmo", "navigate", "pause", "render"]
)
def test_information_page_rejects_application_commands_even_with_old_permissions(
    viewer, action
):
    viewer.sim.sim_config.visualization.allow_commands = True
    code, body = viewer._request("/api/command", json.dumps({"action": action}))
    assert code == 403
    assert "read-only" in json.loads(body)["error"]
    viewer.camera.look_at.assert_not_called()
    viewer.sim.update_gizmos.assert_not_called()
    assert viewer._request("/unknown", "")[0] == 404


def test_close_releases_owned_observer_without_touching_task_controls(viewer):
    app, camera, arena = viewer.application, viewer.camera, viewer.arena
    viewer.close()
    viewer.close()
    app.close.assert_called_once()
    camera.disable_control.assert_called_once()
    camera.close_camera.assert_called_once()
    arena.remove_camera.assert_called_once()
    assert viewer.closed and viewer.camera is None and viewer.group is None
    assert not viewer.capture(force=True)


def test_wrong_thread_cannot_access_native_viewer_or_read_information(viewer):
    viewer.owner = -1
    with pytest.raises(RuntimeError, match="creator thread"):
        viewer.capture(force=True)
    with pytest.raises(RuntimeError, match="creator thread"):
        viewer._request("/api/state", "")
    viewer.application.process.assert_not_called()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"port": 0},
        {"width": 639},
        {"fps": float("nan")},
        {"eye": (0, 0, 0), "target": (0, 0, 0)},
    ],
)
def test_web_configuration_rejects_unrenderable_inputs(kwargs):
    with pytest.raises(ValueError):
        DexSimWebCfg(**kwargs)


def test_python_viewport_camera_surface_remains_available_without_browser_keys(viewer):
    viewer.set_look_at([2, 3, 4], [0, 0, 0])
    viewer.camera.look_at.assert_called_once_with([2, 3, 4], [0, 0, 0], (0, 0, 1))
    control = Mock()
    viewer.add_input_control(control)
    assert not viewer.native().key_state(19)
    viewer.remove_input_control(control)
    control.on_key_down.assert_not_called()
    assert viewer.dirty


@pytest.mark.parametrize("override", [None, "/selected-sdk/libdexsim_streaming.so"])
def test_web_streaming_preflight_loads_the_selected_native_artifact(
    monkeypatch, override
):
    from pathlib import Path
    import dexsim

    if override is None:
        monkeypatch.delenv("DEXSIM_WEBRTC_LIBRARY", raising=False)
    else:
        monkeypatch.setenv("DEXSIM_WEBRTC_LIBRARY", override)
    load = Mock()
    monkeypatch.setattr("embodichain.lab.visualization._web_viewer.ctypes.CDLL", load)
    _require_webrtc_runtime()
    load.assert_called_once_with(
        override or str(Path(dexsim.__file__).parent / "lib" / "libdexsim_streaming.so")
    )
    load.side_effect = OSError("ELF load command address/offset not page-aligned")
    with pytest.raises(RuntimeError, match="cannot load.*ELF load command"):
        _require_webrtc_runtime()


def test_browser_connection_error_survives_information_polls_until_media_recovers():
    from pathlib import Path
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the browser application script probe")
    script = (
        Path(__file__).resolve().parents[2]
        / "embodichain/lab/visualization/web_assets/app.js"
    )
    result = subprocess.run(
        [node, "-", str(script)],
        input=r"""
const assert = require('node:assert/strict');
const vm = require('node:vm');
const elements = new Map();
const element = id => {
    if (!elements.has(id)) elements.set(id, {
        textContent: '', hidden: false,
        querySelector: () => ({replaceChildren() {}})
    });
    return elements.get(id);
};
let applicationError = null;
const context = vm.createContext({
    document: {getElementById: element},
    window: {}, AbortSignal,
    setTimeout() {},
    RTC_ReadSceneDescription: async () => ({state: 'error', error: 'Camera catalog rejected'}),
    fetch: async () => ({ok: true, json: async () => ({
        title: 'Test', summary: null, error: applicationError, frames: 1
    })})
});
vm.runInContext(require('node:fs').readFileSync(process.argv[2], 'utf8'), context);
(async () => {
    // Finish both automatic startup promises, including the successful state poll.
    await new Promise(resolve => setImmediate(resolve));
    assert.match(element('error').textContent, /Camera catalog rejected/);
    assert.equal(element('error').hidden, false);
    applicationError = 'Observer render failed';
    await vm.runInContext('poll()', context);
    assert.match(element('error').textContent, /Camera catalog rejected/);
    assert.match(element('error').textContent, /Observer render failed/);
    context.window.RTC_ViewPeer = {ViewReady: true, Stopped: false};
    await vm.runInContext('poll()', context);
    assert.equal(element('error').textContent, 'Observer render failed');
    applicationError = null;
    await vm.runInContext('poll()', context);
    assert.equal(element('error').hidden, true);
})().catch(error => { console.error(error); process.exitCode = 1; });
""",
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
