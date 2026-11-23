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


"""Verify whiteboard scoring through measured ink and marker withdrawal."""

from __future__ import annotations

from types import SimpleNamespace
import numpy as np
import torch

from embodichain.extensions.tasks.manipulation.whiteboard.draw_table import (
    _LINES,
    _WhiteboardDrawTableEnv,
    _Ink,
)


def test_whiteboard_scores_measured_ink_without_expert_state() -> None:
    env = object.__new__(_WhiteboardDrawTableEnv)
    env.sim = SimpleNamespace(device="cpu")
    env.sim_cfg = SimpleNamespace(physics_dt=0.01)
    env.cfg = SimpleNamespace(sim_steps_per_control=1, episode_context_in_info=False)
    env._ink = _Ink()
    env._ready = True
    env._previous_sample = -1
    env._elapsed_steps = torch.zeros(1, dtype=torch.long)
    tcp = torch.eye(4).unsqueeze(0)
    marker = tcp.clone()
    marker[:, 2, 3] = 0.0355
    tip = torch.tensor([[0.475, 0.0, 1.28]])
    env._tcp = lambda: tcp
    env._pen = SimpleNamespace(get_local_pose=lambda **_: marker)
    env._marker_tip = lambda: tip
    no_contacts = {
        "position": torch.empty(1, 0, 3),
        "is_valid": torch.empty(1, 0, dtype=torch.bool),
        "impulse_valid": torch.empty(1, 0, dtype=torch.bool),
        "impulse": torch.empty(1, 0),
        "normal": torch.empty(1, 0, 3),
    }
    env._sensor = SimpleNamespace(update=lambda: None, get_data=lambda: no_contacts)

    def observe():
        env._elapsed_steps += 1
        return env.get_info()

    def stamp_line(start, end, *, gap=False):
        for index, point in enumerate(np.linspace(start, end, 601)):
            if gap and 260 <= index <= 340:
                env._ink.stamp(None)
            else:
                env._ink.stamp(point)
        env._ink.stamp(None)

    for name, start, end, _ in _LINES:
        stamp_line(start, end, gap=name == "header")
    info = observe()
    assert not info["success"].item()
    assert not info["fail"].item()
    assert info["metrics"]["longest_line_gap"].max().item() > 0.01

    _, start, end, _ = next(line for line in _LINES if line[0] == "header")
    stamp_line(start, end)
    tip[0, 0] = 0.5
    assert not observe()["success"].item()
    tip[0, 0] = 0.475
    assert observe()["success"].item()

    marker[0, 0, 3] = 0.03
    info = observe()
    assert not info["success"].item()
    assert info["fail"].item()
    marker[0, 0, 3] = 0.0

    stamp_line((-0.25, 0.25), (0.25, 0.25))
    info = observe()
    assert not info["success"].item()
    assert info["metrics"]["stray_ink_area"].item() > 0.0001
