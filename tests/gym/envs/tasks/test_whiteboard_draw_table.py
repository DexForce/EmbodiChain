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
import pytest
import torch

from embodichain.extensions.tasks.manipulation.whiteboard.draw_table import (
    _LINES,
    _WhiteboardDrawTableEnv,
    _Ink,
)


def test_whiteboard_scores_measured_ink_without_expert_state() -> None:
    env = object.__new__(_WhiteboardDrawTableEnv)
    env.sim_cfg = SimpleNamespace(physics_dt=0.01)
    env.cfg = SimpleNamespace(sim_steps_per_control=2, episode_context_in_info=False)
    env.sensors = {}
    env._ink = _Ink()
    env._ready = True
    env._elapsed_steps = torch.zeros(1, dtype=torch.long)
    tcp = torch.tensor(
        [
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ]
    )
    marker = tcp.clone()
    marker[:, 0, 3] = 0.0355
    final_tip = torch.tensor([[0.475, 0.0, 1.28]])
    tip = final_tip.clone()
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
    contacts = no_contacts
    transient_contact = None

    def advance(dt, count, *, after_substep, render_final_step):
        nonlocal contacts
        for index in range(count):
            if index == 0 and transient_contact is not None:
                contacts = transient_contact
                tip.copy_(contacts["position"][:, 0])
            else:
                contacts = no_contacts
                tip.copy_(final_tip)
            after_substep(dt)

    env.sim = SimpleNamespace(device="cpu", update=advance)
    env._sensor = SimpleNamespace(
        update=lambda: None,
        get_data=lambda: contacts,
        get_impulse_duration=lambda dt: dt,
    )

    def observe():
        env._advance_physics()
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
    final_tip[0, 0] = 0.5
    assert not observe()["success"].item()
    final_tip[0, 0] = 0.475
    assert observe()["success"].item()

    # The nib touches an out-of-band spot on the first physics step, then
    # withdraws before the end of the control interval.
    transient_contact = {
        "position": torch.tensor([[[0.5, 0.25, 1.53]]]),
        "is_valid": torch.tensor([[True]]),
        "impulse_valid": torch.tensor([[True]]),
        "impulse": torch.tensor([[0.01]]),
        "normal": torch.tensor([[[1.0, 0.0, 0.0]]]),
    }
    info = observe()
    assert info["metrics"]["stray_ink_area"].item() == pytest.approx(5e-6)
    assert info["metrics"]["nib_force"].item() == 0.0
    transient_contact = None

    marker[0, 1, 3] = 0.03
    info = observe()
    assert not info["success"].item()
    assert info["fail"].item()
    marker[0, 1, 3] = 0.0

    stamp_line((-0.25, 0.25), (0.25, 0.25))
    info = observe()
    assert not info["success"].item()
    assert info["metrics"]["stray_ink_area"].item() > 0.0001
