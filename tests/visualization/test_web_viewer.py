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

from embodichain.lab.visualization import DexSimWebCfg, VisualizationCfg
from embodichain.lab.visualization._web_viewer import _WebViewer, _validate_sensor


@pytest.fixture
def viewer():
    value = object.__new__(_WebViewer)
    value.sim = SimpleNamespace(
        sim_config=SimpleNamespace(
            visualization=VisualizationCfg(backend="dexsim-web")
        ),
        _visualization_topology_revision=3,
        spawn_result=object(),
        _ensure_render_state=Mock(),
        asset_uids=["cube"],
        enable_entity_gizmo=Mock(),
        disable_entity_gizmo=Mock(),
    )
    value.cfg = DexSimWebCfg()
    value.owner = threading.get_ident()
    value.run_id = "current-run"
    value.camera = Mock()
    value.camera.get_world_pose.return_value = np.eye(4)
    value.group = object()
    value.application = Mock()
    value.arena = Mock()
    value.gizmo = None
    value.error = None
    value.frames = 0
    value.render_seconds = 0.0
    value.next_render = float("inf")
    value.dirty = False
    value.paused = False
    value.closed = False
    return value


def command(viewer, action, **fields):
    return viewer._request(
        "/api/command",
        json.dumps(
            dict(
                action=action,
                run_id=viewer.run_id,
                revision=viewer.sim._visualization_topology_revision,
                **fields,
            )
        ),
    )


def test_manual_capture_pumps_requests_without_physics_or_sensor_updates(viewer):
    assert not viewer.capture()
    viewer.application.process.assert_called_once()
    viewer.sim._ensure_render_state.assert_not_called()
    assert command(viewer, "render")[0] == 200
    assert viewer.capture()
    viewer.sim._ensure_render_state.assert_called_once()
    viewer.camera.render.assert_called_once()
    assert viewer.frames == 1
    # Capturing a paused scene remains possible without a physics update.
    command(viewer, "pause", enabled=True)
    viewer.capture()
    assert not viewer.capture()
    assert viewer.capture(force=True)


def test_empty_scene_does_not_prepare_or_render(viewer):
    viewer.sim.spawn_result = None
    assert not viewer.capture(force=True)
    viewer.sim._ensure_render_state.assert_not_called()
    viewer.camera.render.assert_not_called()


def test_commands_reject_stale_scene_and_read_only_mutation(viewer):
    code, _ = viewer._request(
        "/api/command",
        json.dumps(dict(action="gizmo", enabled=True, run_id="old-run", revision=3)),
    )
    assert code == 409
    assert command(viewer, "gizmo", enabled=True)[0] == 403
    viewer.sim.enable_entity_gizmo.assert_not_called()
    # View navigation remains available to read-only scene clients.
    assert command(viewer, "reset_view")[0] == 200
    viewer.camera.look_at.assert_called_once_with(viewer.cfg.eye, viewer.cfg.target)
    assert command(viewer, "pause", enabled="false")[0] == 400


def test_gizmo_permission_and_close_release_owned_camera(viewer):
    viewer.sim.sim_config.visualization.allow_commands = True
    assert command(viewer, "gizmo", enabled=True)[0] == 200
    viewer.gizmo.attach_camera.assert_called_once_with(viewer.camera)
    app, camera, arena = viewer.application, viewer.camera, viewer.arena
    viewer.close()
    viewer.close()
    app.close.assert_called_once()
    camera.disable_control.assert_called_once()
    arena.remove_camera.assert_called_once()
    viewer.sim.disable_entity_gizmo.assert_called_once()
    assert viewer.closed and viewer.camera is None and viewer.group is None


def test_wrong_thread_cannot_access_native_viewer(viewer):
    viewer.owner = -1
    with pytest.raises(RuntimeError, match="creator thread"):
        viewer.capture(force=True)
    viewer.application.process.assert_not_called()


@pytest.mark.parametrize(
    "changes",
    [
        {"enable_color": False},
        {"sensor_type": "StereoCamera"},
        {"width": 639},
    ],
)
def test_incompatible_sensor_rejected_before_allocation(changes):
    cfg = SimpleNamespace(
        sensor_type="Camera", enable_color=True, width=640, height=480
    )
    cfg.__dict__.update(changes)
    with pytest.raises(ValueError, match="dexsim-web"):
        _validate_sensor(cfg, [])


def test_native_stream_limit_reserves_one_observer_track():
    cfg = SimpleNamespace(
        sensor_type="Camera", enable_color=True, width=640, height=480
    )
    sensor = SimpleNamespace(cfg=cfg)
    _validate_sensor(cfg, [sensor] * 14)
    with pytest.raises(ValueError, match="15 sensor cameras"):
        _validate_sensor(cfg, [sensor] * 15)


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
