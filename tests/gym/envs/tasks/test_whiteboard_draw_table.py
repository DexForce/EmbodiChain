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


"""Verify that the whiteboard rubric rejects a broken or dirty drawing."""

from __future__ import annotations

from types import SimpleNamespace
import numpy as np

from embodichain.extensions.tasks.manipulation.whiteboard.draw_table import (
    _LINES,
    _WhiteboardDrawTableEnv,
    _Ink,
)


def test_whiteboard_accepts_complete_table_and_rejects_gap_or_stray_ink():
    env = object.__new__(_WhiteboardDrawTableEnv)
    env.sim = SimpleNamespace(device="cpu")
    env._ink = _Ink()

    def stamp_line(start, end, *, gap=False):
        for index, point in enumerate(np.linspace(start, end, 601)):
            if gap and 260 <= index <= 340:
                env._ink.stamp(None)
            else:
                env._ink.stamp(point)
        env._ink.stamp(None)

    for name, start, end, _ in _LINES:
        stamp_line(start, end, gap=name == "header")
    accepted, metrics = env._acceptance()
    assert not accepted
    assert metrics["longest_line_gap"].max().item() > 0.01

    _, start, end, _ = next(line for line in _LINES if line[0] == "header")
    stamp_line(start, end)
    assert env._acceptance()[0]

    stamp_line((-0.25, 0.25), (0.25, 0.25))
    accepted, metrics = env._acceptance()
    assert not accepted
    assert metrics["stray_ink_area"].item() > 0.0001
