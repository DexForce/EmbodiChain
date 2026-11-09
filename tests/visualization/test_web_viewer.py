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
from embodichain.lab.visualization._web_viewer import _WebViewer


@pytest.fixture
def viewer():
    value = object.__new__(_WebViewer)
    value.sim = SimpleNamespace(
        sim_config=SimpleNamespace(
            visualization=VisualizationCfg(backend="dexsim-web")
        ),
        _visualization_topology_revision=3,
        spawn_result=object(),
        physics=SimpleNamespace(process_interaction=Mock()),
        _ensure_render_state=Mock(),
        update_gizmos=Mock(),
        _enable_default_entity_gizmo=Mock(),
        _render_state_published=True,
        get_gizmo_items=lambda: (),
        is_window_recording=lambda: False,
        asset_uids=["cube"],
        arena_offsets=np.zeros((1, 3)),
        enable_entity_gizmo=Mock(),
        disable_entity_gizmo=Mock(),
    )
    value.cfg = DexSimWebCfg()
    value.owner = threading.get_ident()
    value.run_id = "current-run"
    value.camera = Mock()
    value.camera.get_world_pose.return_value = np.eye(4)
    value.camera.get_intrinsic.return_value = np.eye(3)
    value.title = "EmbodiChain"
    value.home_view = (value.cfg.eye, value.cfg.target, (0, 0, 1))
    value.pan_enabled = True
    value.orbit = np.array(value.cfg.target)
    value.controls = []
    value.keys = set()
    value.key_deadline = 0.0
    value.joint_provider = None
    value.joint_commands = {}
    value.sequence = 0
    value.replay = dict(step=0, max_step=0, visible=False)
    value.replay_command = None
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


def test_render_only_scene_draws_without_preparing_or_stepping(viewer):
    viewer.sim.spawn_result = None
    assert viewer.capture(force=True)
    viewer.sim._ensure_render_state.assert_not_called()
    viewer.sim.update_gizmos.assert_not_called()
    viewer.camera.render.assert_called_once()


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
    viewer.camera.look_at.assert_called_once_with(
        viewer.cfg.eye, viewer.cfg.target, (0, 0, 1)
    )
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


def test_shared_viewer_camera_and_input_control_work_while_paused(viewer):
    viewer.sim.sim_config.visualization.allow_commands = True
    viewer.set_look_at([2, 3, 4], [0, 0, 0])
    viewer.camera.look_at.assert_called_once_with([2, 3, 4], [0, 0, 0], (0, 0, 1))
    control = Mock()
    viewer.add_input_control(control)
    assert command(viewer, "key", key=19, down=True)[0] == 200
    assert viewer.key_state(19)
    control.on_key_down.assert_called_once_with(19)
    assert command(viewer, "key", key=19, down=False)[0] == 200
    assert not viewer.key_state(19)
    control.on_key_up.assert_called_once_with(19)


def test_web_replay_seeks_are_bounded_and_keep_latest(viewer):
    viewer.sim.sim_config.visualization.allow_commands = True
    viewer.publish_replay_control(step=2, max_step=10)
    assert command(viewer, "seek", step=8)[0] == 200
    assert command(viewer, "seek", step=9)[0] == 200
    assert viewer.drain_replay_control_command() == 9
    assert viewer.drain_replay_control_command() is None
    assert command(viewer, "seek", step=11)[0] == 400


def test_web_target_uses_existing_gizmo_contract_and_rejects_bad_rotations(viewer):
    viewer.sim.sim_config.visualization.allow_commands = True
    gizmo = Mock()
    gizmo.get_control_pose.return_value = np.eye(4)[None]
    viewer.sim.get_gizmo_items = lambda: (("robot:arm", gizmo),)
    pose = np.eye(4)
    pose[:3, 3] = [0.4, 0.2, 0.6]
    assert command(viewer, "target", id="robot:arm", pose=pose.tolist())[0] == 200
    np.testing.assert_array_equal(
        gizmo.request_local_pose.call_args.args[0], pose.astype(np.float32)
    )
    pose[0, 0] = 2
    assert command(viewer, "target", id="robot:arm", pose=pose.tolist())[0] == 400


def test_read_only_replay_refuses_seek(viewer):
    viewer.publish_replay_control(step=0, max_step=10)
    assert command(viewer, "seek", step=3)[0] == 403
    assert viewer.drain_replay_control_command() is None


def test_paused_poll_applies_controls_and_invalidates_cached_publication(viewer):
    viewer.joint_provider = Mock()
    viewer.joint_commands["joint"] = object()
    viewer.paused = True
    assert not viewer.capture()
    viewer.joint_provider.update.assert_called_once()
    viewer.sim.update_gizmos.assert_called_once()
    assert viewer.sim._render_state_published is False


def test_read_only_clients_cannot_inject_shared_keyboard_controls(viewer):
    assert command(viewer, "key", key=26, down=True)[0] == 403
    assert not viewer.key_state(26)
    assert not viewer.keys
