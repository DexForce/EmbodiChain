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

"""Measured evidence for sequential Real2Sim call occurrences."""

from __future__ import annotations

import numpy as np
import pytest

from embodichain.lab.task_program.integrations.simulation.real2sim_collection import (
    _held_pose_errors,
    _observed_program_phases,
)


def test_phase_trace_distinguishes_repeated_skills_and_unassociated_holds() -> None:
    skills = {0: "pick_up", 1: "place", 2: "pick_up", 3: "place"}
    pick = (("approach", 0, 1), ("close", 1, 2), ("lift", 2, 3))
    place = (("approach", 0, 1), ("release", 1, 2), ("retract", 2, 3))
    calls = np.array([0, 0, 0, -1, 1, 1, 1, -1, 2, 2, 2, 3, 3, 3])
    samples = np.array([0, 1, 2, -1, 0, 1, 2, -1, 0, 1, 2, 0, 1, 2])
    events = _observed_program_phases(
        skills, {0: pick, 1: place, 2: pick, 3: place}, calls, samples
    )
    assert [(step, index, name) for step, index, _, name in events] == [
        (0, 0, "approach"),
        (1, 0, "close"),
        (2, 0, "lift"),
        (4, 1, "approach"),
        (5, 1, "release"),
        (6, 1, "retract"),
        (8, 2, "approach"),
        (9, 2, "close"),
        (10, 2, "lift"),
        (11, 3, "approach"),
        (12, 3, "release"),
        (13, 3, "retract"),
    ]


def test_phase_trace_rejects_missing_or_reordered_calls() -> None:
    segments = {0: (("approach", 0, 1), ("lift", 1, 2))}
    with pytest.raises(ValueError, match="missing a declared phase"):
        _observed_program_phases({0: "pick_up"}, segments, np.array([0]), np.array([0]))
    with pytest.raises(ValueError, match="out-of-order"):
        _observed_program_phases(
            {0: "pick_up", 1: "pick_up"},
            {0: segments[0], 1: segments[0]},
            np.array([1, 0]),
            np.array([0, 0]),
        )


def test_held_pose_evidence_detects_translation_and_rotation_slip() -> None:
    objects = np.array([[0.0, 0, 0, 0, 0, 0, 1], [0.1, 0, 0, 0, 0, 0, 1]])
    tcp = np.repeat(np.eye(4)[None], 2, axis=0)
    tcp[:, 0, 3] = [0.03, 0.15]  # A 2 cm change in the grasp relation.
    angle = np.deg2rad(10.0)
    tcp[1, :3, :3] = [
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1],
    ]
    translation, rotation = _held_pose_errors(objects, tcp)
    assert translation == pytest.approx(0.02)
    assert rotation == pytest.approx(10.0)
