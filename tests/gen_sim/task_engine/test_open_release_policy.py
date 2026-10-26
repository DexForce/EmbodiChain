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

from types import SimpleNamespace

import pytest
import torch

from embodichain.gen_sim.task_engine._task_program.grip_evidence import (
    NativeGripObserver,
    _open_release_reader,
)
from embodichain.gen_sim.task_engine.experiment_policy import (
    experimental_release_enabled,
)
from embodichain.lab.task_program.semantics.effects import HeldObjectRelation


@pytest.mark.parametrize(
    "profile", ["default", "contact_experiment", "composition_experiment"]
)
@pytest.mark.parametrize("override", [None, False, True])
def test_release_profile_preserves_explicit_override(
    profile: str, override: bool | None
) -> None:
    assert experimental_release_enabled(profile, override) is (
        profile != "default" if override is None else override
    )


@pytest.mark.parametrize(
    "enabled,opened,expected,relation,other_pad,dropped,finite,valid,value",
    [
        (False, True, False, HeldObjectRelation.DETACHED, False, 0, True, False, False),
        (True, True, False, HeldObjectRelation.DETACHED, False, 0, True, True, False),
        (True, False, False, HeldObjectRelation.DETACHED, False, 0, True, False, False),
        (True, True, True, HeldObjectRelation.ATTACHED, False, 0, True, False, False),
        (True, True, False, HeldObjectRelation.ATTACHED, False, 0, True, False, False),
        (True, True, False, HeldObjectRelation.DETACHED, True, 0, True, True, True),
        (True, True, False, HeldObjectRelation.DETACHED, False, 1, True, False, False),
        (True, True, False, HeldObjectRelation.DETACHED, False, 0, False, False, False),
    ],
)
def test_open_release_never_certifies_acquisition_or_bilateral_release(
    enabled: bool,
    opened: bool,
    expected: bool,
    relation: HeldObjectRelation,
    other_pad: bool,
    dropped: int,
    finite: bool,
    valid: bool,
    value: bool,
) -> None:
    pairs = [[1, 2], [1, 3]] if other_pad else [[1, 2]]
    normals = [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]][: len(pairs)]
    if not finite:
        normals[0][0] = float("nan")
    sensor = SimpleNamespace(
        dropped_contacts=dropped,
        update=lambda: None,
        get_data=lambda: {
            "is_valid": torch.ones(1, len(pairs), dtype=torch.bool),
            "user_ids": torch.tensor([pairs]),
            "normal": torch.tensor([normals]),
            "impulse": torch.full((1, len(pairs)), 0.1),
        },
    )
    observer = NativeGripObserver(
        sensor, {"subject": torch.tensor([[1]])}, {"hand": torch.tensor([[2, 3]])}
    )
    if enabled:
        observer.release_opening = lambda part, env: {"finite": True, "open": opened}
    query = SimpleNamespace(
        expectation=SimpleNamespace(object_id="subject", relation=relation),
        source=SimpleNamespace(address=SimpleNamespace(control_part="hand")),
        clause=SimpleNamespace(expected=expected),
    )
    result = observer(
        query,
        SimpleNamespace(
            env_ids=torch.tensor([0]), timestamp=1.0, observation_revision=1
        ),
    )
    assert result.valid.tolist() == [valid]
    assert result.values.tolist() == [value]
    assert observer.snapshot_trace()["recorded_query_count"] == 1


@pytest.mark.parametrize("damage", ["zero_normal", "nonopposing", "negative_impulse"])
def test_opening_does_not_resolve_invalid_or_nonopposing_contacts(damage: str) -> None:
    pairs = [[1, 2], [1, 3]] if damage == "nonopposing" else [[1, 2], [4, 5]]
    normals = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    impulses = torch.tensor([[0.1, 0.1]])
    if damage == "zero_normal":
        normals[0, 0].zero_()
    elif damage == "negative_impulse":
        impulses[0, 1] = -0.1
    sensor = SimpleNamespace(
        dropped_contacts=0,
        update=lambda: None,
        get_data=lambda: {
            "is_valid": torch.tensor([[True, True]]),
            "user_ids": torch.tensor([pairs]),
            "normal": normals,
            "impulse": impulses,
        },
    )
    observer = NativeGripObserver(
        sensor, {"subject": torch.tensor([[1]])}, {"hand": torch.tensor([[2, 3]])}
    )
    observer.release_opening = lambda part, env: {"finite": True, "open": True}
    query = SimpleNamespace(
        expectation=SimpleNamespace(
            object_id="subject", relation=HeldObjectRelation.DETACHED
        ),
        source=SimpleNamespace(address=SimpleNamespace(control_part="hand")),
        clause=SimpleNamespace(expected=False),
    )
    result = observer(
        query,
        SimpleNamespace(
            env_ids=torch.tensor([0]), timestamp=1.0, observation_revision=1
        ),
    )
    assert result.valid.tolist() == [False]
    assert result.values.tolist() == [False]


@pytest.mark.parametrize(
    "driver,mimic,opened",
    [
        (0.0, 2.0, True),
        (0.049, 0.0, True),
        (0.05, 0.0, False),
        (0.051, 0.0, False),
        (float("nan"), 0.0, False),
        (0.0, float("nan"), False),
    ],
)
def test_release_reader_uses_measured_driver_and_strict_open_boundary(
    tmp_path, driver: float, mimic: float, opened: bool
) -> None:
    urdf = tmp_path / "hand.urdf"
    urdf.write_text(
        '<robot name="hand"><joint name="drive" type="revolute"/>'
        '<joint name="compliant" type="revolute"><mimic joint="drive"/></joint></robot>'
    )
    robot = SimpleNamespace(
        cfg=SimpleNamespace(fpath=str(urdf)),
        joint_names=["drive", "compliant"],
        get_joint_ids=lambda name: [0, 1],
        get_qpos=lambda name: torch.tensor([[driver, mimic]]),
    )
    source = SimpleNamespace(
        control_parts=["hand"], open_qpos=[0.0, 0.0], minimum_displacement=0.05
    )
    observed = _open_release_reader(robot, source)("hand", 0)
    assert observed["open"] is opened
    assert observed["driving_joint_names"] == ["drive"]
