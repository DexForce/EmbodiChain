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

"""Keyboard edits publish rendering without advancing the scene."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from embodichain.lab.sim import SimulationManager
from embodichain.lab.sim.utility import keyboard_utils

pytestmark = pytest.mark.no_sim


def _simulation() -> SimpleNamespace:
    marker = Mock()
    return SimpleNamespace(
        draw_marker=Mock(return_value=[marker]),
        remove_marker=Mock(),
        sync_render_state=Mock(),
        update=Mock(side_effect=AssertionError("UI edit stepped physics")),
    )


def test_camera_keyboard_edit_only_publishes(monkeypatch: pytest.MonkeyPatch) -> None:
    sim = _simulation()
    monkeypatch.setattr(SimulationManager, "get_instance", lambda: sim)
    sensor = SimpleNamespace(
        num_instances=1,
        is_attached=False,
        _entities=[Mock()],
        get_local_pose=lambda **_: torch.eye(4)[None],
        get_arena_pose=lambda **_: torch.eye(4)[None],
        get_data=lambda: {"color": torch.zeros((1, 2, 2, 3), dtype=torch.uint8)},
        update=Mock(),
        set_local_pose=Mock(),
    )
    keys = iter([ord("w"), 27])
    monkeypatch.setattr(keyboard_utils.cv2, "waitKey", lambda _: next(keys))
    monkeypatch.setattr(keyboard_utils.cv2, "imshow", lambda *_: None)
    monkeypatch.setattr(keyboard_utils.cv2, "destroyAllWindows", lambda: None)

    keyboard_utils.run_keyboard_control_for_camera(sensor, vis_pose=True)

    assert sensor.set_local_pose.call_args.args[0][0, 2, 3] == pytest.approx(0.01)
    sim.sync_render_state.assert_called_once()
    sim.update.assert_not_called()


def test_light_keyboard_edit_only_publishes(monkeypatch: pytest.MonkeyPatch) -> None:
    sim = _simulation()
    monkeypatch.setattr(SimulationManager, "get_instance", lambda: sim)
    pose = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    light = SimpleNamespace(
        num_instances=1,
        _entities=[Mock()],
        cfg=SimpleNamespace(color=[1.0, 1.0, 1.0], intensity=5.0, radius=1.0),
        get_local_pose=lambda to_matrix=False: (
            torch.eye(4)[None] if to_matrix else pose.clone()
        ),
        set_local_pose=Mock(),
    )
    keys = iter(["w", "\x1b"])
    stdin = SimpleNamespace(fileno=lambda: 0, read=lambda _: next(keys))
    monkeypatch.setattr(keyboard_utils.sys, "stdin", stdin)
    monkeypatch.setattr(keyboard_utils.termios, "tcgetattr", lambda _: [])
    monkeypatch.setattr(keyboard_utils.termios, "tcsetattr", lambda *_: None)
    monkeypatch.setattr(keyboard_utils.tty, "setcbreak", lambda _: None)
    monkeypatch.setattr(keyboard_utils.select, "select", lambda *_: ([stdin], [], []))

    keyboard_utils.run_keyboard_control_for_light(light, vis_pose=True)

    assert light.set_local_pose.call_args.args[0][0, 2] == pytest.approx(0.01)
    sim.sync_render_state.assert_called_once()
    sim.update.assert_not_called()
