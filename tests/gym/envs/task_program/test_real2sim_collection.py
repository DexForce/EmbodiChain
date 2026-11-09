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
    _placement_heading_check,
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


@pytest.mark.parametrize(
    "actual,expected,tolerance,accepted,error",
    [
        (-179.0, 179.0, 3.0, True, 2.0),
        (50.0, 45.0, 3.0, False, 5.0),
        (180.0, 0.0, 3.0, False, 180.0),
        (15.0, 15.0, 0.0, True, 0.0),
    ],
)
def test_directed_package_heading_acceptance(
    actual, expected, tolerance, accepted, error
) -> None:
    def yaw(degrees):
        angle = np.deg2rad(degrees)
        pose = np.eye(4)
        pose[:3, :3] = [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
        return pose

    passed, metrics = _placement_heading_check(yaw(actual), yaw(expected), tolerance)
    assert passed is accepted
    assert metrics["placement_yaw_error_deg"] == pytest.approx(error, abs=1e-10)
    # Rebase both poses by the same world yaw: requested relative heading is fixed.
    passed_rebased, rebased = _placement_heading_check(
        yaw(37.0) @ yaw(actual), yaw(37.0) @ yaw(expected), tolerance
    )
    assert passed_rebased is accepted
    assert rebased == pytest.approx(metrics, abs=1e-10)


def test_heading_check_keeps_legacy_profiles_and_rejects_vertical_heading() -> None:
    assert _placement_heading_check(np.eye(4), np.eye(4), None) == (True, {})
    tipped = np.eye(4)
    tipped[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    with pytest.raises(ValueError, match="horizontal projection"):
        _placement_heading_check(tipped, np.eye(4), 3.0)


def test_semantic_release_target_includes_solver_error_in_yaw_acceptance() -> None:
    from types import SimpleNamespace
    import torch
    from embodichain.lab.sim.atomic_actions.primitives.place import PlaceGoal
    from embodichain.lab.task_program.integrations.simulation.real2sim_collision import (
        _semantic_release_pose,
    )

    relation = torch.eye(4, dtype=torch.float64)[None]
    relation[0, 0, 3] = 0.05
    requested = relation.clone()
    request = SimpleNamespace(
        goal=PlaceGoal(requested),
        binding=SimpleNamespace(
            endpoint=lambda *args: SimpleNamespace(task_state_key="hand")
        ),
    )
    context = SimpleNamespace(
        robot=SimpleNamespace(qpos=torch.zeros(1, 1)),
        get_held_object=lambda key: SimpleNamespace(object_to_eef=relation),
    )
    semantic = _semantic_release_pose(request, context)
    np.testing.assert_allclose(semantic, np.eye(4), atol=1e-8)
    observed = np.eye(4)
    angle = np.deg2rad(4.8)
    observed[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    accepted, metrics = _placement_heading_check(observed, semantic, 3.0)
    assert not accepted
    assert metrics["placement_yaw_error_deg"] == pytest.approx(4.8)


def test_explicit_tabletop_evidence_owns_states_and_rejects_incomplete_clock():
    import torch
    from embodichain.lab.task_program.integrations.simulation.real2sim_collection import (
        Real2SimTabletopEpisodeEvidence,
    )

    qpos = torch.zeros(3, 2)
    poses = torch.zeros(3, 7)
    poses[:, 6] = 1
    target = torch.eye(4)
    values = dict(
        qpos=qpos,
        object_pose=poses,
        target_pose=target,
        joint_names=("a", "b"),
        step_dt=1 / 30,
        execution_trace=({}, {}),
        compute_batch_fk=lambda **kw: torch.eye(4)[None, None],
    )
    evidence = Real2SimTabletopEpisodeEvidence(**values)
    qpos.fill_(1)
    poses[:, 0] = 2
    target[0, 3] = 3
    assert evidence.qpos.sum() == 0 and evidence.object_pose[:, 0].sum() == 0
    assert evidence.target_pose[0, 3] == 0
    with pytest.raises(ValueError, match="T acknowledged"):
        Real2SimTabletopEpisodeEvidence(**{**values, "execution_trace": ({},)})
