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


def test_cargo_envelope_tracks_rows_and_excludes_explicit_manipulation(
    monkeypatch,
) -> None:
    import numpy as np
    from embodichain.gen_sim.task_engine import task_program_bundle
    from embodichain.gen_sim.task_engine._task_program.cargo import (
        capture_cargo,
        check_cargo,
    )

    tray = torch.eye(4).repeat(2, 1, 1)
    cargo = tray.clone()
    cargo[:, 2, 3] = 0.02
    poses = {"tray": tray, "knife": cargo, "cup": cargo.clone()}
    objects = {
        uid: SimpleNamespace(get_local_pose=lambda uid=uid, **kw: poses[uid].clone())
        for uid in poses
    }

    def mesh(cfg):
        return (
            np.array([[-0.2, -0.2, 0.0], [0.2, 0.2, 0.05]])
            if cfg["uid"] == "tray"
            else np.array([[-0.01, -0.01, 0.0], [0.01, 0.01, 0.01]])
        )

    monkeypatch.setattr(task_program_bundle, "_mesh_vertices", mesh)
    scene = {"simulation": {"rigid_object": [{"uid": uid} for uid in poses]}}
    graph = {
        "nodes": [
            {"task_type": "E5", "call": {"arguments": {"object": "tray"}}},
            {"task_type": "E1", "call": {"object": "cup"}},
        ]
    }
    guards = capture_cargo(
        SimpleNamespace(sim=SimpleNamespace(get_rigid_object=objects.get)), scene, graph
    )
    assert [g.object_id for g in guards] == ["knife"]
    assert check_cargo(guards, 2)["accepted_mask"] == [True, True]
    cargo[1, 0, 3] = 0.3
    report = check_cargo(guards, 2)
    assert report["accepted_mask"] == [True, False]
    assert report["contents"][0]["initial_mask"] == [True, True]


def test_cargo_envelope_uses_full_mesh_and_carrier_frame() -> None:
    from embodichain.gen_sim.task_engine._task_program.cargo import inside_envelope

    carrier = torch.eye(4).repeat(2, 1, 1)
    carrier[:, :2, :2] = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    carrier[:, 0, 3] = 5.0
    local = torch.eye(4).repeat(2, 1, 1)
    local[:, 2, 3] = 0.02
    local[1, 0, 3] = 0.19
    vertices = torch.tensor([[-0.02, -0.02, 0.0], [0.02, 0.02, 0.01]])
    bounds = torch.tensor([[-0.2, -0.2, 0.0], [0.2, 0.2, 0.05]])
    assert inside_envelope(vertices, carrier @ local, carrier, bounds).tolist() == [
        True,
        False,
    ]


def test_cargo_trajectory_rejects_escape_and_return_but_ignores_unwritten_tail() -> (
    None
):
    from embodichain.gen_sim.task_engine._task_program.cargo import (
        CargoEnvelope,
        check_cargo,
    )

    pose = torch.eye(4).repeat(2, 1, 1)
    cargo_pose = pose.clone()
    cargo_pose[:, 2, 3] = 0.02
    guard = CargoEnvelope(
        "tray",
        "knife",
        SimpleNamespace(get_local_pose=lambda **kw: pose),
        SimpleNamespace(get_local_pose=lambda **kw: cargo_pose),
        torch.tensor([[-0.01, -0.01, 0.0], [0.01, 0.01, 0.01]]),
        torch.tensor([[-0.2, -0.2, 0.0], [0.2, 0.2, 0.05]]),
        torch.tensor([True, True]),
    )
    carrier = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]).repeat(2, 3, 1)
    cargo = carrier.clone()
    cargo[:, :, 2] = 0.02
    cargo[:, 1, 0] = 0.3
    trajectory = {
        "rigid_objects": {"tray": {"pose": carrier}, "knife": {"pose": cargo}}
    }
    report = check_cargo([guard], 2, trajectory=trajectory, step_counts=[3, 1])
    assert report["accepted_mask"] == [False, True]
    assert report["contents"][0]["first_failed_frames"] == [1, None]
    assert check_cargo([guard], 2)["accepted_mask"] == [True, True]
    with pytest.raises(ValueError, match="frame range"):
        check_cargo([guard], 2, trajectory=trajectory, step_counts=[0, 1])


def _local_port(
    cfg: StabilityConstraint,
    pose: torch.Tensor,
    *,
    reference_pose: torch.Tensor | None = None,
    vertices: dict[str, torch.Tensor] | None = None,
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
    for entity, mesh in (vertices or {}).items():
        entities[entity].get_vertices = Mock(return_value=mesh)
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


def _box_vertices(half_extents: tuple[float, float, float], count: int) -> torch.Tensor:
    from itertools import product

    return torch.tensor(list(product(*[(-x, x) for x in half_extents]))).repeat(
        count, 1, 1
    )


def test_supported_placement_uses_current_mesh_rotation_without_locking_it():
    count = 5
    upper = torch.eye(4).repeat(count, 1, 1)
    upper[:, 2, 3] = 0.11
    upper[1:, :3, :3] = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    )
    # Rolling changes the bottom offset from -0.10 m to -0.02 m.
    upper[1:, 2, 3] = torch.tensor([0.03, 0.055, 0.005, 0.03])
    upper[4, 0, 3] = 0.4
    support = torch.eye(4).repeat(count, 1, 1)
    port, policy, segment = _local_port(
        StabilityConstraint(
            entity="upper",
            reference="support",
            kind="supported_placement",
            displacement=(0.0, 0.0, 0.03),
            position_tolerance=0.5,
            duration=0.08,
            timeout=0.12,
        ),
        upper,
        reference_pose=support,
        vertices={
            "upper": _box_vertices((0.01, 0.02, 0.10), count),
            "support": _box_vertices((0.3, 0.3, 0.01), count),
        },
    )
    list(
        port.actions(
            policy, segment=segment, active_mask=torch.ones(count, dtype=torch.bool)
        )
    )
    assert port.post_policy_result(policy, segment=segment).tolist() == [
        True,
        True,
        False,
        False,
        False,
    ]
    measured = port.post_policy_metadata(policy, segment=segment)["measurements"]
    assert "alignment" not in measured
    assert measured["support_gap"] == pytest.approx([0, 0, 0.025, -0.025, 0], abs=1e-7)


@pytest.mark.parametrize("moving", ["upper", "support"])
@pytest.mark.parametrize("motion", ["translation", "rotation"])
def test_supported_placement_still_requires_a_stable_window(moving, motion):
    upper = torch.eye(4)[None]
    upper[:, 2, 3] = 0.11
    support = torch.eye(4)[None]
    port, policy, segment = _local_port(
        StabilityConstraint(
            entity="upper",
            reference="support",
            kind="supported_placement",
            displacement=(0.0, 0.0, 0.11),
            duration=0.12,
            timeout=0.24,
        ),
        upper,
        reference_pose=support,
        vertices={
            "upper": _box_vertices((0.01, 0.02, 0.1), 1),
            "support": _box_vertices((0.3, 0.3, 0.01), 1),
        },
    )
    for step, _ in enumerate(
        port.actions(policy, segment=segment, active_mask=torch.tensor([True]))
    ):
        pose = upper if moving == "upper" else support
        if motion == "translation":
            pose[:, 0, 3] = 0.03 if step % 2 == 0 else 0.0
        else:
            angle = 0.3 if step % 2 == 0 else 0.0
            pose[:, :2, :2] = torch.tensor(
                [
                    [math.cos(angle), -math.sin(angle)],
                    [math.sin(angle), math.cos(angle)],
                ]
            )
    assert port.post_policy_result(policy, segment=segment).tolist() == [False]


def test_supported_placement_requires_a_relative_target():
    with pytest.raises(ValueError, match="relative target"):
        StabilityConstraint(entity="upper", kind="supported_placement")


def test_supported_placement_rotates_reference_geometry_and_footprint():
    upper = torch.eye(4).repeat(2, 1, 1)
    upper[:, 2, 3] = 0.11
    upper[1, 1, 3] = 0.15
    support = torch.eye(4).repeat(2, 1, 1)
    # Turning the support swaps its top offset and its narrow footprint axis.
    support[:, :3, :3] = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    )
    port, policy, segment = _local_port(
        StabilityConstraint(
            entity="upper",
            reference="support",
            kind="supported_placement",
            displacement=(0.0, 0.0, 0.11),
            position_tolerance=0.3,
            duration=0.08,
            timeout=0.12,
        ),
        upper,
        reference_pose=support,
        vertices={
            "upper": _box_vertices((0.01, 0.02, 0.01), 2),
            "support": _box_vertices((0.3, 0.1, 0.01), 2),
        },
    )
    list(
        port.actions(
            policy, segment=segment, active_mask=torch.ones(2, dtype=torch.bool)
        )
    )
    assert port.post_policy_result(policy, segment=segment).tolist() == [True, False]


@pytest.mark.parametrize(
    "mesh",
    [torch.empty(1, 0, 3), torch.full((1, 8, 3), float("nan")), torch.zeros(2, 8, 3)],
)
def test_supported_placement_rejects_invalid_mesh_without_static_fallback(mesh):
    with pytest.raises(ValueError, match="Support geometry"):
        _local_port(
            StabilityConstraint(
                entity="upper",
                reference="support",
                kind="supported_placement",
                displacement=(0.0, 0.0, 0.0),
            ),
            torch.eye(4)[None],
            reference_pose=torch.eye(4)[None],
            vertices={"upper": mesh, "support": _box_vertices((0.3, 0.3, 0.01), 1)},
        )


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


@pytest.mark.parametrize(
    "kind, expected",
    [("placement", [True, True, True]), ("stack", [True, False, False])],
)
def test_support_checks_reject_hovering_or_tipped_objects_near_target(
    kind: str, expected: list[bool]
) -> None:
    upper = torch.eye(4).repeat(3, 1, 1)
    upper[:, 2, 3] = torch.tensor([0.1, 0.125, 0.1])
    upper[2, :3, :3] = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    )
    support = torch.eye(4).repeat(3, 1, 1)
    cfg = StabilityConstraint(
        entity="upper",
        reference="support",
        kind=kind,
        displacement=(0.0, 0.0, 0.11),
        position_tolerance=0.05,
        duration=0.08,
        timeout=0.12,
        **(
            {
                "local_axis": (0.0, 0.0, 1.0),
                "reference_axis": (0.0, 0.0, 1.0),
                "reference_top": 0.1,
                "reference_half_extents": (0.05, 0.05),
            }
            if kind == "stack"
            else {}
        ),
    )
    port, policy, segment = _local_port(cfg, upper, reference_pose=support)
    list(
        port.actions(
            policy, segment=segment, active_mask=torch.ones(3, dtype=torch.bool)
        )
    )
    assert port.post_policy_result(policy, segment=segment).tolist() == expected


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


@pytest.mark.parametrize("relative", [False, True])
def test_placement_rejects_a_stable_but_wrong_destination_per_environment(
    relative: bool,
) -> None:
    pose = torch.eye(4).repeat(2, 1, 1)
    pose[1, 0, 3] = 0.10
    port, policy, segment = _local_port(
        StabilityConstraint(
            entity="tray",
            kind="placement",
            target_position=None if relative else (0.0, 0.0, 0.0),
            reference="block" if relative else None,
            displacement=(0.0, 0.0, 0.0) if relative else None,
            duration=0.08,
            timeout=0.12,
        ),
        pose,
        reference_pose=torch.eye(4).repeat(2, 1, 1) if relative else None,
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
        {"world_axis": [0.0, 0.0, 0.0]},
        {"world_axis": [1.0, 0.0, 0.0]},
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


def test_horizontal_hold_checks_declared_axis_and_rejects_vertical_pose():
    pose = torch.eye(4).repeat(3, 1, 1)
    pose[0, :3, :3] = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    pose[2, :3, :3] = torch.tensor([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    cfg = StabilityConstraint.decode(
        {
            "entity": "cup",
            "kind": "hold",
            "local_axis": [1.0, 0.0, 0.0],
            "world_axis": [0.0, 2.0, 0.0],
            "motion_parts": ["right_arm"],
            "duration": 0.08,
            "timeout": 0.12,
        }
    )
    assert cfg.world_axis == (0.0, 1.0, 0.0)
    port, policy, segment = _local_port(cfg, pose)
    list(
        port.actions(
            policy, segment=segment, active_mask=torch.ones(3, dtype=torch.bool)
        )
    )
    assert port.post_policy_result(policy, segment=segment).tolist() == [
        True,
        False,
        False,
    ]


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
