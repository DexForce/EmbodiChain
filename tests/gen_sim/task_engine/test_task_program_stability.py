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

"""Task-owned upright acceptance migrated from the removed public validator."""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from embodichain.gen_sim.task_engine._task_program.stability import (
    StabilityConstraint,
    TaskStabilityPort,
)
from embodichain.lab.task_program.compiler.program import CompiledPostPolicy
from embodichain.lab.task_program.language.schema import WaitStablePostCfg
from embodichain.lab.task_program.semantics import SceneObjectRef

__all__: list[str] = []


def _local_port(
    cfg: StabilityConstraint,
    pose: torch.Tensor,
    *,
    reference_pose: torch.Tensor | None = None,
):
    preset = "gen_sim.target.stable"
    policy = CompiledPostPolicy(
        cfg=WaitStablePostCfg(entity=cfg.entity, preset=preset),
        entity=SceneObjectRef(cfg.entity),
        source_path=("program", "post", 0),
    )
    segment = SimpleNamespace(post_policies=(policy,))
    robot = SimpleNamespace(
        get_qpos=lambda **kwargs: torch.zeros(pose.shape[0], 2),
        get_joint_ids=lambda **kwargs: [0, 1],
        compute_fk=lambda **kwargs: torch.eye(4).repeat(pose.shape[0], 1, 1),
    )
    entities = {
        cfg.entity: SimpleNamespace(get_local_pose=lambda **kwargs: pose.clone())
    }
    if reference_pose is not None:
        assert cfg.reference is not None
        entities[cfg.reference] = SimpleNamespace(
            get_local_pose=lambda **kwargs: reference_pose.clone()
        )
    port = TaskStabilityPort(
        Mock(),
        SimpleNamespace(get_rigid_object=entities.get),
        robot,
        SimpleNamespace(
            rigid_objects=tuple(
                SimpleNamespace(entity_id=entity, simulation_uid=entity)
                for entity in entities
            )
        ),
        {preset: cfg},
        step_dt=0.04,
    )
    return port, policy, segment


@pytest.mark.parametrize("motion", ["translation", "rotation"])
def test_stack_requires_both_objects_stable_in_each_environment(motion: str) -> None:
    upper = torch.eye(4).repeat(2, 1, 1)
    upper[:, 2, 3] = 0.1
    support = torch.eye(4).repeat(2, 1, 1)
    cfg = StabilityConstraint(
        entity="upper",
        reference="support",
        kind="stack",
        local_axis=(0.0, 0.0, 1.0),
        reference_axis=(0.0, 0.0, 1.0),
        object_bottom=-0.05,
        reference_top=0.05,
        reference_half_extents=(0.05, 0.05),
        duration=3.0,
        timeout=3.2,
    )
    port, policy, segment = _local_port(cfg, upper, reference_pose=support)
    for step, _ in enumerate(
        port.actions(policy, segment=segment, active_mask=torch.tensor([True, True]))
    ):
        if motion == "translation":
            # Stay under the upper object, but exceed the declared drift limit.
            support[1, 0, 3] = 1.5 * cfg.translation_drift if step % 2 == 0 else 0.0
        else:
            angle = 1.5 * cfg.rotation_drift if step % 2 == 0 else 0.0
            support[1, :3, :3] = torch.tensor(
                [
                    [math.cos(angle), -math.sin(angle), 0.0],
                    [math.sin(angle), math.cos(angle), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
    assert port.post_policy_result(policy, segment=segment).tolist() == [True, False]


def test_stack_restarts_its_complete_window_after_support_motion() -> None:
    upper = torch.eye(4).unsqueeze(0)
    upper[:, 2, 3] = 0.1
    support = torch.eye(4).unsqueeze(0)
    duration = 3.0
    port, policy, segment = _local_port(
        StabilityConstraint(
            entity="upper",
            reference="support",
            kind="stack",
            local_axis=(0.0, 0.0, 1.0),
            reference_axis=(0.0, 0.0, 1.0),
            object_bottom=-0.05,
            reference_top=0.05,
            reference_half_extents=(0.05, 0.05),
            duration=duration,
            timeout=4.0,
        ),
        upper,
        reference_pose=support,
    )
    moved_at_step = 5
    steps = 0
    for _ in port.actions(policy, segment=segment, active_mask=torch.tensor([True])):
        steps += 1
        if steps == moved_at_step:
            support[0, 0, 3] = 0.03
    assert port.post_policy_result(policy, segment=segment).tolist() == [True]
    assert steps == moved_at_step + math.ceil(duration / 0.04)


def test_placement_requires_a_target_not_only_a_stationary_object() -> None:
    with pytest.raises(ValueError, match="explicit target"):
        StabilityConstraint(entity="tray", kind="placement")


def test_placement_rejects_a_stable_but_wrong_destination_per_environment() -> None:
    pose = torch.eye(4).repeat(2, 1, 1)
    pose[1, 0, 3] = 0.10
    port, policy, segment = _local_port(
        StabilityConstraint(
            entity="tray",
            kind="placement",
            target_position=(0.0, 0.0, 0.0),
            duration=0.08,
            timeout=0.12,
        ),
        pose,
    )
    commands = list(
        port.actions(policy, segment=segment, active_mask=torch.tensor([True, True]))
    )
    assert len(commands) == 3
    assert port.post_policy_result(policy, segment=segment).tolist() == [True, False]
    metadata = port.post_policy_metadata(policy, segment=segment)
    assert metadata["measurements"]["position_error"] == pytest.approx([0.0, 0.10])


def test_upright_hold_checks_orientation_without_inventing_a_position_goal() -> None:
    pose = torch.eye(4).repeat(2, 1, 1)
    pose[1, :3, :3] = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    port, policy, segment = _local_port(
        StabilityConstraint(
            entity="can",
            kind="hold",
            local_axis=(0.0, 0.0, 1.0),
            motion_parts=("arm",),
            duration=0.08,
            timeout=0.12,
        ),
        pose,
    )
    list(port.actions(policy, segment=segment, active_mask=torch.tensor([True, True])))
    assert port.post_policy_result(policy, segment=segment).tolist() == [True, False]
    assert port.post_policy_metadata(policy, segment=segment)["failed_mask"] == [
        False,
        True,
    ]


@pytest.mark.parametrize(
    "invalid_field",
    [
        {"local_axis": [0.0, 0.0, 0.0]},
        {"entity": None},
        {"local_axis": [True, 0.0, 1.0]},
        {"local_axis": [0.0, 0.0, float("nan")]},
        {"minimum_alignment": 1.1},
        {"absolute_alignment": "false"},
        {"target_axis": [0.0, 0.0, 1.0]},
    ],
)
def test_upright_constraint_rejects_invalid_values(invalid_field: dict) -> None:
    with pytest.raises(ValueError):
        StabilityConstraint.decode(
            {
                "kind": "upright",
                "entity": "cube",
                "local_axis": [1.0, 0.0, 0.0],
                **invalid_field,
            }
        )


def test_upright_constraint_normalizes_declared_local_axis() -> None:
    cfg = StabilityConstraint.decode(
        {"kind": "upright", "entity": "cube", "local_axis": [2.0, 0.0, 0.0]}
    )
    assert cfg.local_axis == (1.0, 0.0, 0.0)


def test_stability_rejects_conflicting_absolute_and_relative_targets() -> None:
    with pytest.raises(ValueError, match="absolute and relative"):
        StabilityConstraint(
            entity="tray",
            kind="placement",
            target_position=(0.0, 0.0, 1.0),
            reference="table",
            displacement=(0.0, 0.0, 0.5),
        )


def test_upright_policy_validates_each_environments_measured_rotation() -> None:
    step_dt = 0.04
    pose = torch.eye(4).repeat(2, 1, 1)
    pose[0, :3, :3] = torch.tensor([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    delegate = Mock()
    robot = SimpleNamespace(get_qpos=lambda **kwargs: torch.zeros(2, 2))
    entity = SimpleNamespace(get_local_pose=lambda **kwargs: pose.clone())
    simulation = SimpleNamespace(get_rigid_object=lambda uid: entity)
    binding = SimpleNamespace(
        rigid_objects=(SimpleNamespace(entity_id="cube", simulation_uid="native_cube"),)
    )
    preset = "gen_sim.cube.upright"
    policy = CompiledPostPolicy(
        cfg=WaitStablePostCfg(entity="cube", preset=preset),
        entity=SceneObjectRef("cube"),
        source_path=("program", "post", 0),
    )
    segment = SimpleNamespace(post_policies=(policy,))
    port = TaskStabilityPort(
        delegate,
        simulation,
        robot,
        binding,
        {
            preset: StabilityConstraint(
                kind="upright",
                entity="cube",
                local_axis=(1.0, 0.0, 0.0),
                minimum_alignment=0.9,
                duration=2 * step_dt,
                timeout=3 * step_dt,
            )
        },
        step_dt=step_dt,
    )

    commands = list(
        port.actions(
            policy, segment=segment, active_mask=torch.ones(2, dtype=torch.bool)
        )
    )
    result = port.post_policy_result(policy, segment=segment)
    metadata = port.post_policy_metadata(policy, segment=segment)

    assert len(commands) == 3
    assert all(command.shape == (2, 2) for command in commands)
    assert result.tolist() == [True, False]
    assert metadata["kind"] == "upright"
    assert metadata["measurements"]["alignment"] == pytest.approx([1.0, 0.0])
    assert metadata["accepted_mask"] == [True, False]
    assert delegate.mock_calls == []
