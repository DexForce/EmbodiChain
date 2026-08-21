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

"""Tests for verified articulation state and masked symbolic effects."""

from __future__ import annotations

import pytest
import torch

from embodichain.lab.sim.atomic_actions import (
    ArticulationJointState,
    StateDelta,
    TaskState,
)


def test_task_state_normalizes_and_owns_articulation_joint_state() -> None:
    position = torch.tensor([0.35], dtype=torch.float32)
    state = TaskState(
        batch_size=2,
        device="cpu",
        articulation_joints={
            ("drawer", "slide"): ArticulationJointState(position),
        },
    )

    position.fill_(99.0)
    observed = state.get_articulation_joint_state("drawer", "slide")
    assert observed is not None
    assert torch.equal(observed.position, torch.tensor([[0.35], [0.35]]))
    assert torch.equal(observed.env_mask, torch.tensor([True, True]))


def test_state_delta_merges_articulation_rows_without_overwriting_others() -> None:
    state = TaskState(
        batch_size=3,
        device="cpu",
        articulation_joints={
            ("drawer", "slide"): ArticulationJointState(
                torch.tensor([[0.0], [0.1], [0.2]]),
            )
        },
    )
    candidate = ArticulationJointState(
        torch.tensor([[0.5], [0.6], [0.7]]),
        env_mask=torch.tensor([True, True, False]),
    )

    updated = StateDelta(
        articulation_joint_updates={("drawer", "slide"): candidate}
    ).apply(state, torch.tensor([True, False, True]))

    joint = updated.get_articulation_joint_state("drawer", "slide")
    assert joint is not None
    assert torch.equal(joint.position, torch.tensor([[0.5], [0.1], [0.7]]))
    assert torch.equal(joint.env_mask, torch.tensor([True, True, False]))


def test_state_delta_removes_only_selected_articulation_rows() -> None:
    state = TaskState(
        batch_size=2,
        device="cpu",
        articulation_joints={
            ("drawer", "slide"): ArticulationJointState(torch.tensor([0.4]))
        },
    )
    updated = StateDelta(articulation_joint_updates={("drawer", "slide"): None}).apply(
        state, torch.tensor([False, True])
    )

    joint = updated.get_articulation_joint_state("drawer", "slide")
    assert joint is not None
    assert torch.equal(joint.env_mask, torch.tensor([True, False]))

    removed = StateDelta(articulation_joint_updates={("drawer", "slide"): None}).apply(
        updated, torch.tensor([True, False])
    )
    assert removed.get_articulation_joint_state("drawer", "slide") is None


def test_articulation_state_and_delta_validate_strictly() -> None:
    with pytest.raises(TypeError, match="floating"):
        ArticulationJointState(torch.tensor([1], dtype=torch.long))
    with pytest.raises(ValueError, match="finite"):
        ArticulationJointState(torch.tensor([float("nan")]))
    with pytest.raises(ValueError, match="articulation/joint pairs"):
        StateDelta(articulation_joint_updates={("drawer", ""): None})
    with pytest.raises(TypeError, match="ArticulationJointState"):
        StateDelta(
            articulation_joint_updates={("drawer", "slide"): torch.tensor([0.1])}
        )


def test_articulation_state_delta_snapshot_is_independently_owned() -> None:
    source = ArticulationJointState(torch.tensor([[0.2], [0.3]]))
    delta = StateDelta(articulation_joint_updates={("drawer", "slide"): source})
    snapshot = delta.snapshot()
    copied = snapshot.articulation_joint_updates[("drawer", "slide")]

    assert copied is not None
    assert copied is not source
    assert copied.position.data_ptr() != source.position.data_ptr()
    assert torch.equal(copied.position, source.position)


__all__: list[str] = []
