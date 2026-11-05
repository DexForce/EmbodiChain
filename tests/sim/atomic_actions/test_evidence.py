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

"""Tests for versioned Atomic Action physical-evidence contracts."""

from __future__ import annotations

import pytest
import torch

from embodichain.lab.sim.atomic_actions import (
    PhysicalEvidenceBatch,
    PhysicalEvidenceFrame,
    PhysicalEvidenceRequest,
)


def _frame() -> PhysicalEvidenceFrame:
    env_ids = torch.tensor([3, 7], dtype=torch.long)
    return PhysicalEvidenceFrame(
        timestamp=1.25,
        observation_revision=4,
        env_ids=env_ids,
        batches={
            "contact": PhysicalEvidenceBatch(
                evidence_id="contact",
                values=torch.tensor([True, False]),
                valid=torch.tensor([True, True]),
                acquisition_errors=(None, None),
                timestamp=1.25,
                env_ids=env_ids,
                observation_revision=4,
            )
        },
    )


def test_evidence_frame_owns_aligned_batches() -> None:
    frame = _frame()

    batch = frame.get("contact")
    assert frame.batch_size == 2
    assert torch.equal(batch.env_ids, torch.tensor([3, 7]))
    assert batch.values.tolist() == [True, False]

    with pytest.raises(KeyError, match="missing"):
        frame.get("missing")


@pytest.mark.parametrize("owned", [False, True])
@pytest.mark.parametrize("mismatch", ["env_ids", "timestamp", "observation_revision"])
def test_evidence_frame_rejects_misaligned_channels(owned, mismatch) -> None:
    env_ids = torch.tensor([3, 7], dtype=torch.long)
    create = (
        PhysicalEvidenceFrame._from_owned_batches if owned else PhysicalEvidenceFrame
    )
    with pytest.raises(ValueError, match="share frame"):
        create(
            timestamp=1.25,
            observation_revision=4,
            env_ids=env_ids,
            batches={
                "contact": PhysicalEvidenceBatch(
                    evidence_id="contact",
                    values=torch.tensor([True, False]),
                    valid=torch.ones(2, dtype=torch.bool),
                    acquisition_errors=(None, None),
                    timestamp=1.5 if mismatch == "timestamp" else 1.25,
                    env_ids=torch.tensor(
                        [3, 8] if mismatch == "env_ids" else [3, 7], dtype=torch.long
                    ),
                    observation_revision=5 if mismatch == "observation_revision" else 4,
                )
            },
        )


def test_public_frame_construction_and_snapshot_revalidate_mutable_batches():
    ids = torch.tensor([3, 7])
    values = torch.tensor([1.0, 2.0])
    batch = PhysicalEvidenceBatch(
        "force", values, torch.ones(2, dtype=torch.bool), (None, None), 1.25, ids, 4
    )
    frame = PhysicalEvidenceFrame(1.25, 4, ids, {"force": batch})
    values.zero_()
    ids[0] = 99
    batch.values.fill_(3.0)
    assert frame.get("force").values.tolist() == [1.0, 2.0]
    assert frame.env_ids.tolist() == [3, 7]
    batch.values[0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        PhysicalEvidenceFrame(1.25, 4, torch.tensor([3, 7]), {"force": batch})
    saved = frame.snapshot()
    frame.batches["force"].values[0] = float("nan")
    assert saved.get("force").values.tolist() == [1.0, 2.0]
    with pytest.raises(ValueError, match="finite"):
        frame.snapshot()


def test_evidence_request_rejects_empty_channels() -> None:
    with pytest.raises(ValueError, match="at least one environment"):
        PhysicalEvidenceRequest(
            timestamp=0.0,
            observation_revision=0,
            env_ids=torch.empty(0, dtype=torch.long),
        )


@pytest.mark.parametrize("direction", [1, -1])
def test_press_event_is_contact_backed_shallow_travel_not_activation(direction):
    from embodichain.lab.sim.atomic_actions.articulation_evidence import (
        PressEventTarget,
        JointContactSample,
        evaluate_press_event,
    )

    target = PressEventTarget(0, direction * 0.01)
    result = evaluate_press_event(
        target,
        [
            JointContactSample(0, 0, "prepare", False),
            JointContactSample(0.004, 0, "press", True),
            JointContactSample(0.008, direction * 0.00006, "press", True),
        ],
    )
    assert result == {
        "accepted": True,
        "criterion": "contact_press_0.05mm",
        "minimum_contact_stroke_m": 0.00005,
        "activation_verified": False,
        "reason": "contact_press_event",
        "max_stroke": 0.00006,
        "contact_stroke": 0.00006,
        "contact_peak": 0.00006,
    }


@pytest.mark.parametrize(
    "case,reason",
    [
        ("missing", "missing_samples"),
        ("unavailable", "invalid_observation"),
        ("duplicate_time", "invalid_observation"),
        ("passive", "not_initially_released"),
        ("bounds", "joint_out_of_bounds"),
        ("interrupted", "insufficient_contact_stroke"),
    ],
)
def test_press_event_retains_failure_and_unavailable_meanings(case, reason):
    from embodichain.lab.sim.atomic_actions.articulation_evidence import (
        PressEventTarget,
        JointContactSample,
        evaluate_press_event,
    )

    trace = [
        JointContactSample(0, 0, "prepare", False),
        JointContactSample(0.004, 0, "press", True),
        JointContactSample(0.008, 0.00006, "press", True),
    ]
    if case == "missing":
        trace = []
    elif case == "unavailable":
        trace[1] = JointContactSample(0.004, 0, "press", None, False)
    elif case == "duplicate_time":
        trace[1] = JointContactSample(0, 0, "press", True)
    elif case == "passive":
        trace[0] = JointContactSample(0, 0.002, "prepare", False)
    elif case == "bounds":
        trace[2] = JointContactSample(0.008, 0.02, "press", True)
    else:
        trace[2] = JointContactSample(0.008, 0.00003, "press", True)
        trace += [
            JointContactSample(0.012, 0.00003, "press", False),
            JointContactSample(0.016, 0.00003, "press", True),
            JointContactSample(0.02, 0.00006, "press", True),
        ]
    result = evaluate_press_event(PressEventTarget(0, 0.01), trace)
    assert not result["accepted"]
    assert result["reason"] == reason


def test_twist_event_keeps_coarse_path_separate_from_calibrated_setting():
    from embodichain.lab.sim.atomic_actions.articulation_evidence import (
        JointContactSample,
        TwistEventTarget,
        evaluate_twist_event,
        initial_twist_state_stable,
    )

    samples = [
        JointContactSample(0, 0, "prepare", False),
        JointContactSample(0.004, 0, "twist", True),
        JointContactSample(0.008, 0.3, "twist", True),
    ]
    coarse = evaluate_twist_event(TwistEventTarget(1, (-2, 2)), samples, final_qpos=0.3)
    assert coarse["accepted"] and coarse["coarse_turn_reached"]
    assert not coarse["target_qpos_reached"]
    calibrated = evaluate_twist_event(
        TwistEventTarget(1, (-2, 2), success_criterion="calibrated_setting"),
        samples,
        final_qpos=0.3,
    )
    assert not calibrated["accepted"]
    assert calibrated["contact_path"] == 0.3
    assert not initial_twist_state_stable([0, 0], [0.1, 0.1])
    assert initial_twist_state_stable([0, 0.01], [0.001, 0])


def test_twist_event_subtracts_uncontacted_reversal_and_debounces_jitter():
    from embodichain.lab.sim.atomic_actions.articulation_evidence import (
        JointContactSample,
        TwistEventTarget,
        evaluate_twist_event,
    )

    positions = [0, 0, 0.3, 0, 0, 0.3]
    contacts = [False, True, True, False, True, True]
    samples = [
        JointContactSample(i * 0.004, q, "twist", contact)
        for i, (q, contact) in enumerate(zip(positions, contacts))
    ]
    result = evaluate_twist_event(
        TwistEventTarget(1, (-2, 2), success_criterion="calibrated_setting"),
        samples,
        final_qpos=0.3,
    )
    assert result["contact_travel"] == pytest.approx(0.3)
    assert result["uncontacted_reverse_travel"] == 0.3
    assert result["contact_path"] == 0.6
    jitter = [
        JointContactSample(i * 0.004, q, "twist", i > 0)
        for i, q in enumerate([0, 0, 0.005, 0.003, 0.007])
    ]
    assert (
        evaluate_twist_event(TwistEventTarget(1, (-2, 2)), jitter, final_qpos=0.007)[
            "contact_path"
        ]
        == 0
    )


def _control_source_expected(kind, case):
    import json
    from pathlib import Path

    data = json.loads(
        Path(__file__)
        .with_name("data")
        .joinpath("control_validation_source.json")
        .read_text()
    )
    return next(
        row["receipt"]
        for row in data["cases"]
        if row["kind"] == kind and row["case"] == case
    )


@pytest.mark.parametrize(
    "case", ["good", "incomplete", "overflow", "table", "robot", "velocity"]
)
def test_twist_startup_matches_frozen_source(case):
    from embodichain.lab.sim.atomic_actions.control_validation import (
        TwistStartupSample,
        evaluate_twist_startup,
    )

    rows = [
        TwistStartupSample(
            i * 0.004,
            0,
            0.1 if case == "velocity" else 0.0,
            case == "table",
            case == "robot",
            True,
            0,
        )
        for i in range(4)
    ]
    assert evaluate_twist_startup(
        rows, complete=case != "incomplete", overflow=case == "overflow"
    ) == _control_source_expected("startup", case)


@pytest.mark.parametrize(
    "case", ["retreat", "terminal", "contact", "housing", "no_retreat", "clearance"]
)
def test_press_release_matches_source_safety_boundaries(case):
    from embodichain.lab.sim.atomic_actions.articulation_evidence import (
        JointContactSample,
        PressEventTarget,
    )
    from embodichain.lab.sim.atomic_actions.control_validation import (
        validate_press_release,
    )

    q = float(torch.tensor(0.00006))
    samples = [
        JointContactSample(0, 0, "prepare", False),
        JointContactSample(0.004, 0, "press", True),
        JointContactSample(0.008, q, "press", True),
    ]
    samples += [
        JointContactSample(i * 0.004, q, "cleanup", case == "contact")
        for i in range(3, 133)
    ]
    parent = [False] * len(samples)
    if case == "housing":
        parent[-1] = True
    terminal = case != "retreat"
    result = validate_press_release(
        PressEventTarget(0.0, 0.01),
        samples,
        parent_contacts=parent,
        physics_dt=0.004,
        terminal=terminal,
        retreat_verified=case != "no_retreat",
        retreat_clearance=float(torch.tensor(0.05)),
        tcp_clearance=float(torch.tensor(0.06)),
        terminal_clearance=(
            (0.03 if case == "clearance" else 0.05) if terminal else None
        ),
        observed_qpos=q,
        window_available=True,
    )
    assert result == _control_source_expected("press", case)


@pytest.mark.parametrize(
    "case", ["turned", "moving", "other_joint", "clearance", "intermediate"]
)
def test_twist_terminal_matches_frozen_source(case):
    from embodichain.lab.sim.atomic_actions.articulation_evidence import (
        JointContactSample,
        TwistEventTarget,
    )
    from embodichain.lab.sim.atomic_actions.control_validation import (
        validate_twist_terminal,
    )

    q = float(torch.tensor(0.3))
    samples = [
        JointContactSample(0, 0, "prepare", False),
        JointContactSample(0.004, 0, "twist", True),
        JointContactSample(0.008, q, "twist", True),
    ]
    samples += [JointContactSample(i * 0.004, q, "twist", False) for i in range(3, 133)]
    result = validate_twist_terminal(
        TwistEventTarget(0.3, (-1.0, 1.0)),
        samples,
        parent_contacts=[False] * len(samples),
        positions=[q] * 13,
        velocities=[float(torch.tensor(0.1 if case == "moving" else 0.0))] * 13,
        physics_dt=0.004,
        clearance=0.03 if case == "clearance" else 0.05,
        others_unchanged=case != "other_joint",
        terminal=case != "intermediate",
        early_stop=True,
        window_available=True,
    )
    assert result == _control_source_expected("twist", case)


def test_press_ready_defers_native_limits_until_measured_conditions_pass():
    from embodichain.lab.sim.atomic_actions.articulation_evidence import (
        JointContactSample,
        PressEventTarget,
    )
    from embodichain.lab.sim.atomic_actions.control_validation import (
        validate_press_ready,
    )

    reads = []

    def limits():
        reads.append(True)
        return torch.tensor([0.0, 0.01])

    rejected = validate_press_ready(
        PressEventTarget(0.0, 0.01),
        [0.0] * 13,
        JointContactSample(0, 0, "prepare", True),
        read_limits=limits,
        expected_limits=(0.0, 0.01),
        window_available=True,
    )
    assert not rejected["accepted"] and not reads
    accepted = validate_press_ready(
        PressEventTarget(0.0, 0.01),
        [0.0] * 13,
        JointContactSample(0, 0, "prepare", False),
        read_limits=limits,
        expected_limits=(0.0, 0.01),
        window_available=True,
    )
    assert accepted["accepted"] and reads == [True]


@pytest.mark.parametrize(
    "position, released",
    [(0.0, True), (0.001, True), (0.00101, False), (float("nan"), False)],
)
def test_press_start_retains_source_released_band(position, released):
    from embodichain.lab.sim.atomic_actions.articulation_evidence import (
        PressEventTarget,
    )
    from embodichain.lab.sim.atomic_actions.control_validation import (
        press_start_is_released,
    )

    assert press_start_is_released(position, PressEventTarget(0.0, 0.01)) is released
