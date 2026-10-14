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

from dataclasses import dataclass
import itertools
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from embodichain.gen_sim.task_engine._task_program import held_path as gate
from embodichain.gen_sim.task_engine._task_program import motion
from embodichain.gen_sim.task_engine._task_program.state_handoff import StagingContract
from embodichain.lab.sim.atomic_actions import JointPositionTarget, PlannerDiagnostics
from embodichain.lab.sim.atomic_actions.goals import SceneEntityPose
from embodichain.lab.sim.atomic_actions.policies import MotionPolicy
from embodichain.lab.sim.atomic_actions.primitives.move_held_object import (
    HeldObjectPoseGoal,
)
from embodichain.lab.sim.motion.motion_generator import MotionGenerator
from embodichain.lab.sim.motion.planners.utils import PlanResult


@dataclass(frozen=True)
class Plan:
    plan_success: torch.Tensor
    joint_trajectory: object
    diagnostics: PlannerDiagnostics
    commands: object = None


def _box(lower: tuple, upper: tuple) -> torch.Tensor:
    return torch.tensor(
        list(itertools.product(*zip(lower, upper))), dtype=torch.float32
    )[None]


def _fixture(monkeypatch):
    identity = torch.eye(4)[None]
    arm = JointPositionTarget("arm", (0, 1, 2, 3))
    hand = JointPositionTarget("hand", (4,))
    seed = torch.tensor([[-0.2, 0.0, 0.24, 0.0, 0.0]])
    held = SimpleNamespace(
        semantics=SimpleNamespace(entity_id="vessel"), object_to_eef=identity
    )
    endpoints = {
        "motion": SimpleNamespace(task_state_key="held", require_target=lambda _: arm),
        "grasp": SimpleNamespace(
            task_state_key="held",
            require_target=lambda _: hand,
            joint_positions=lambda *a, **kw: torch.zeros(1, 1),
        ),
    }
    receiver = identity.clone()
    receiver[:, :3, 3] = torch.tensor([0.2, 0.0, 0.24])
    request = SimpleNamespace(
        goal=HeldObjectPoseGoal(SceneEntityPose("receiver")),
        invocation_id="transport",
        binding=SimpleNamespace(endpoint=lambda _slot, name: endpoints[name]),
        motion_policy=MotionPolicy(sample_count=12, dynamic_collision_mode="off"),
    )
    entities = {
        name: SimpleNamespace(pose=pose, confidence=1.0)
        for name, pose in (("table", identity), ("receiver", receiver))
    }
    context = SimpleNamespace(
        batch_size=1,
        env_ids=torch.tensor([0]),
        robot=SimpleNamespace(qpos=seed.clone()),
        last_qpos=seed.clone(),
        control_dt=0.04,
        require_control_dt=lambda: 0.04,
        task=SimpleNamespace(exclusive_held_object_mask=lambda _: torch.tensor([True])),
        get_held_object=lambda _: held,
        scene=SimpleNamespace(entities=entities),
    )

    def fk(qpos, **_kw):
        poses = torch.eye(4).expand(*qpos.shape[:-1], 4, 4).clone()
        poses[..., :3, 3] = qpos[..., :3]
        cosine, sine = qpos[..., 3].cos(), qpos[..., 3].sin()
        poses[..., 0, 0], poses[..., 0, 1] = cosine, -sine
        poses[..., 1, 0], poses[..., 1, 1] = sine, cosine
        return poses

    robot = SimpleNamespace(
        device=torch.device("cpu"),
        compute_fk=fk,
        compute_batch_fk=fk,
        get_qpos_limits=lambda **_kw: torch.tensor([[[-2.0, 2.0]] * 4]),
        get_solver=lambda _part: SimpleNamespace(
            get_iteration_params=lambda: {"pos_eps": 0.0001, "rot_eps": 0.0001}
        ),
    )
    action = SimpleNamespace(
        robot=robot,
        motion_generator=SimpleNamespace(robot=robot),
        held_path_scene_vertices={
            "vessel": _box((-0.04,) * 3, (0.04,) * 3),
            "table": _box((-1.0, -1.0, -0.1), (1.0, 1.0, 0.0)),
        },
        staging_contract=StagingContract.empty(),
    )

    def build(selected, observed, **kw):
        assert selected is request and observed is context
        assert isinstance(selected.goal.object_target_pose, SceneEntityPose)
        return Plan(
            kw["success"], kw["trajectory"], PlannerDiagnostics(backend="portable")
        )

    action.build_plan = build
    monkeypatch.setattr(gate, "inspect_staging_plan", lambda *a, **kw: {"valid": True})
    monkeypatch.setattr(
        motion, "_joint_velocity_limits", lambda *a: torch.full((1, 4), 20.0)
    )

    def solve(_generator, targets, options=None):
        # Portable Cartesian robot adapter exercises the real bounded resolver.
        pose = targets[-1].xpos
        joints = torch.cat(
            (pose[:, :3, 3], torch.atan2(pose[:, 1, 0], pose[:, 0, 0])[:, None]), dim=-1
        )
        positions = torch.stack((options.start_qpos, joints), dim=1)
        return PlanResult(
            torch.tensor([True]), positions=positions, dt=torch.tensor([[0.0, 0.04]])
        )

    monkeypatch.setattr(MotionGenerator, "generate", solve)
    positions = seed[:, None].repeat(1, 3, 1)
    positions[:, -1, 0] = 0.2
    positions[:, 1, 2] = -0.050843
    original = Plan(
        torch.tensor([True]),
        SimpleNamespace(positions=positions),
        PlannerDiagnostics(backend="original"),
        SimpleNamespace(payload=positions.clone()),
    )
    return action, request, context, original


def test_safe_endpoints_do_not_hide_recorded_negative_table_sweep(monkeypatch):
    action, request, context, original = _fixture(monkeypatch)
    endpoints = original.joint_trajectory.positions[:, [0, -1]]
    assert gate.inspect_payload(action, request, context, endpoints)["valid"]
    measured = gate.inspect_payload(
        action, request, context, original.joint_trajectory.positions
    )
    assert not measured["valid"]
    # Portable reproduction of the archived -90.843 mm intermediate sweep, not native replay.
    assert measured["minimum_table_gap_m"] == pytest.approx(-0.090843, abs=1e-7)
    assert measured["rejected"] == [{"entity": "table", "first_sample": 1}]


def test_archived_transport_source_vertex_catches_actual_unsafe_command(monkeypatch):
    action, request, context, _ = _fixture(monkeypatch)
    # Frozen CPU FK samples 813/877 and the minimum-Z source vertex from the
    # 20261014_102744 kettle transport. No run artifacts, URDF or native imports
    # are required by this regression; one actual witness suffices to reject it.
    attachment = torch.tensor(
        [
            [
                [0.16900226, 0.35321872, -0.92014933, 0.02724672],
                [0.33048875, -0.89983919, -0.28472194, -0.01794954],
                [-0.92855554, -0.25598036, -0.26880969, 0.08856845],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ]
    )
    poses = torch.tensor(
        [
            [
                [0.34866344, 0.43427409, -0.83056596, -0.17987868],
                [0.41516140, -0.86605497, -0.27854939, -0.37843237],
                [-0.84028256, -0.24769895, -0.48225558, 0.85129399],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [
                [0.89172152, -0.28738801, -0.34962938, 0.03942837],
                [-0.37890853, -0.89653459, -0.22946459, 0.02785319],
                [-0.24750951, 0.33709607, -0.90835303, 0.63572579],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ]
    )
    context.get_held_object("held").object_to_eef = attachment
    action.held_path_scene_vertices = {
        "vessel": torch.tensor([[[-0.03890862, -0.00500814, 0.00994581]]]),
        "table": _box(
            (-0.53227478, -0.81451643, 0.0),
            (0.53227478, 0.81451643, 0.63404536),
        ),
    }
    action.robot.compute_batch_fk = lambda qpos, **_kw: poses[qpos[..., 0].long()]
    positions = torch.zeros(1, 2, 5)
    positions[:, 1, 0] = 1
    assert gate.inspect_payload(action, request, context, positions[:, :1])["valid"]
    measured = gate.inspect_payload(action, request, context, positions)
    assert not measured["valid"]
    assert measured["minimum_table_gap_m"] == pytest.approx(-0.09084328, abs=2e-7)
    assert measured["rejected"] == [{"entity": "table", "first_sample": 1}]


def test_qualified_original_is_returned_without_replanning_or_rng_changes(monkeypatch):
    action, request, context, original = _fixture(monkeypatch)
    original.joint_trajectory.positions[:, 1, 2] = 0.24
    before = gate._RandomState.capture(set())
    monkeypatch.setattr(
        gate, "_cartesian_plan", lambda *a: pytest.fail("qualified original replanned")
    )
    result = gate.qualify_or_recover(action, request, context, original)
    assert result.joint_trajectory is original.joint_trajectory
    assert result.plan_success is original.plan_success
    assert result.commands is original.commands
    assert result.diagnostics.metadata["gen_sim_held_path"]["original"]["valid"]
    assert torch.equal(before.cpu, torch.random.get_rng_state())


def test_scene_entity_exact_target_recovered_by_real_continuous_ik(monkeypatch):
    action, request, context, original = _fixture(monkeypatch)
    before = gate._RandomState.capture(set())
    result = gate.qualify_or_recover(action, request, context, original)
    assert result.plan_success.tolist() == [True]
    assert gate.inspect_payload(
        action, request, context, result.joint_trajectory.positions
    )["valid"]
    assert torch.equal(
        result.joint_trajectory.positions[:, -1, :3],
        context.scene.entities["receiver"].pose[:, :3, 3],
    )
    assert result.diagnostics.metadata["gen_sim_held_path"]["accepted_recovery"]
    assert torch.equal(before.cpu, torch.random.get_rng_state())
    assert before.python == random.getstate()
    assert np.array_equal(before.numpy[1], np.random.get_state()[1])
    assert context.last_qpos.tolist() == [
        [-0.20000000298023224, 0.0, 0.23999999463558197, 0.0, 0.0]
    ]


def test_constructive_lift_recovers_blocked_cartesian_leg_without_changing_endpoint(
    monkeypatch,
):
    action, request, context, original = _fixture(monkeypatch)
    action.held_path_scene_vertices["block"] = _box(
        (-0.05, -0.1, 0.05), (0.05, 0.1, 0.29)
    )
    context.scene.entities["block"] = SimpleNamespace(
        pose=torch.eye(4)[None], confidence=1.0
    )
    result = gate.qualify_or_recover(action, request, context, original)
    assert result.plan_success.all()
    details = result.diagnostics.metadata["gen_sim_held_path"]
    assert not details["attempts"][0]["valid"]
    assert details["attempts"][-1]["variant"] == "lifted_cartesian"
    assert details["attempts"][-1]["valid"]
    assert result.joint_trajectory.positions[0, :, 2].max() == pytest.approx(0.36)
    assert torch.equal(
        result.joint_trajectory.positions[:, -1, :3],
        context.scene.entities["receiver"].pose[:, :3, 3],
    )


@pytest.mark.parametrize(
    "case",
    [
        "ik_infeasible",
        "changed_cartesian_target",
        "command_rejected",
        "explicit_cartesian",
    ],
)
def test_owned_free_space_phase_never_replays_unsafe_success_on_failed_qualification(
    monkeypatch, case
):
    action, request, context, original = _fixture(monkeypatch)
    if case == "ik_infeasible":
        monkeypatch.setattr(
            MotionGenerator,
            "generate",
            lambda *a, **kw: PlanResult(
                torch.tensor([False]), positions=torch.zeros(1, 2, 4)
            ),
        )
    elif case == "changed_cartesian_target":
        generator = MotionGenerator.generate

        def shifted(*a, **kw):
            result = generator(*a, **kw)
            result.positions[:, -1, 0] += 0.0005
            return result

        monkeypatch.setattr(MotionGenerator, "generate", shifted)
    elif case == "command_rejected":
        monkeypatch.setattr(
            gate,
            "inspect_staging_plan",
            lambda *a, **kw: {"valid": False, "reason": "final_velocity_rejected"},
        )
    elif case == "explicit_cartesian":
        monkeypatch.setattr(
            gate,
            "_cartesian_plan",
            lambda *a: pytest.fail("explicit policy overridden"),
        )
    with motion.cartesian_approach_scope(case == "explicit_cartesian"):
        result = gate.qualify_or_recover(action, request, context, original)
    assert not result.plan_success.any()
    assert result.diagnostics.failure.code == "held_transport_path_unqualified"
    assert result.joint_trajectory is original.joint_trajectory


@pytest.mark.parametrize(
    "case",
    [
        "missing_geometry",
        "missing_pose",
        "invalid_mesh",
        "nonexclusive_hold",
        "endpoint_contact",
        "required_intermediate_contact",
        "endpoint_inside_table",
    ],
)
def test_unmigrated_phase_retains_original_commands_and_records_no_certificate(
    monkeypatch, case
):
    action, request, context, original = _fixture(monkeypatch)
    original.joint_trajectory.positions[:, 1, 2] = 0.24
    if case == "missing_geometry":
        action.held_path_scene_vertices = {}
    elif case == "missing_pose":
        del context.scene.entities["table"]
    elif case == "invalid_mesh":
        action.held_path_scene_vertices["vessel"] = torch.full((1, 1, 3), torch.nan)
    elif case == "nonexclusive_hold":
        context.task.exclusive_held_object_mask = lambda _: torch.tensor([False])
    elif case == "endpoint_contact":
        context.scene.entities["receiver"].pose[:, 2, 3] = 0.04
    elif case == "required_intermediate_contact":
        poses = context.scene.entities["receiver"].pose[:, None].repeat(1, 2, 1, 1)
        poses[:, 0, 2, 3] = 0.04
        request.goal = HeldObjectPoseGoal(poses)
    elif case == "endpoint_inside_table":
        context.scene.entities["receiver"].pose[:, 2, 3] = -0.05
    before = gate._RandomState.capture(set())
    monkeypatch.setattr(
        gate, "inspect", lambda *a: pytest.fail("unowned phase guarded")
    )
    monkeypatch.setattr(
        gate, "_cartesian_plan", lambda *a: pytest.fail("unowned phase replanned")
    )
    result = gate.qualify_or_recover(action, request, context, original)
    assert result.joint_trajectory is original.joint_trajectory
    assert result.plan_success is original.plan_success
    assert result.commands is original.commands
    assert result.diagnostics.failure is original.diagnostics.failure
    details = result.diagnostics.metadata["gen_sim_held_path"]
    assert details["not_migrated"]
    assert not details["phase_scope"]["owned"]
    assert not details["accepted_recovery"]
    assert details["attempts"] == []
    assert "original" not in details
    assert torch.equal(before.cpu, torch.random.get_rng_state())


def test_target_owned_scope_is_independent_of_family_target_name_or_pose_representation(
    monkeypatch,
):
    action, request, context, _ = _fixture(monkeypatch)
    assert gate.phase_scope(action, request, context)["owned"]
    request.goal = HeldObjectPoseGoal(context.scene.entities["receiver"].pose)
    request.invocation_id = "unrelated_family/return_target"
    assert gate.phase_scope(action, request, context)["owned"]
    request.goal.object_target_pose[:, 2, 3] = 0.04
    request.invocation_id = "arbitrary_family/free_space_target"
    assert not gate.phase_scope(action, request, context)["owned"]


def test_post_transform_scope_retains_free_space_ownership_and_rejects_unsafe_samples(
    monkeypatch,
):
    action, request, context, original = _fixture(monkeypatch)
    assert gate.phase_scope(action, request, context)["owned"]
    certificate = gate.inspect(action, request, context, original)
    assert not certificate["valid"]
    assert certificate["reason"] == "held_path_scene_envelope_rejected"
    monkeypatch.setattr(
        gate,
        "inspect_staging_plan",
        lambda *a, **kw: {"valid": False, "reason": "initial_velocity_rejected"},
    )
    result = gate.qualify_or_recover(action, request, context, original)
    assert result.diagnostics.metadata["gen_sim_held_path"]["phase_scope"]["owned"]
    assert not result.plan_success.any()
    assert result.diagnostics.failure.code == "held_transport_path_unqualified"


def test_scene_pose_resolution_preserves_canonical_world_orientation_order(monkeypatch):
    action, request, context, _ = _fixture(monkeypatch)
    relative = torch.eye(4)
    relative[0, 3] = 0.1
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    request.goal = HeldObjectPoseGoal(
        SceneEntityPose("receiver", relative_pose=relative, world_orientation=rotation)
    )
    target = gate._targets(action, request, context)
    assert target[0, 0, :3, 3].tolist() == pytest.approx([0.2, 0.1, 0.24])
    assert torch.equal(target[0, 0, :3, :3], rotation)


def test_direct_transport_does_not_override_existing_calibrated_route(monkeypatch):
    from embodichain.gen_sim.task_engine._task_program.actions import (
        GenSimMoveHeldObject,
    )
    from embodichain.lab.sim.atomic_actions.primitives.move_held_object import (
        MoveHeldObject,
    )

    action = GenSimMoveHeldObject()
    action.held_path_scene_vertices = {}
    action._planning_services = SimpleNamespace(
        motion_generator=SimpleNamespace(direct_transport_planning=True)
    )
    sentinel = object()
    monkeypatch.setattr(MoveHeldObject, "_plan", lambda *a: sentinel)
    monkeypatch.setattr(
        gate, "qualify_or_recover", lambda *a: pytest.fail("direct route changed")
    )
    assert action._plan(object(), object()) is sentinel


def _with_articulated_panels(monkeypatch, action, context):
    import trimesh
    from embodichain.gen_sim.task_engine._task_program import (
        drawer_geometry,
        place_recovery,
    )

    panels = {}
    for name, y in (("left", -0.3), ("right", 0.3)):
        panel = trimesh.creation.box(extents=(0.5, 0.04, 0.5))
        panel.apply_translation((0.0, y, 0.25))
        panels[("shell", name)] = panel
    monkeypatch.setattr(drawer_geometry, "collision_meshes", lambda _cfg: panels)
    originals = action.held_path_scene_vertices
    art = SimpleNamespace(
        cfg=SimpleNamespace(fpath="cabinet.usdc", body_scale=(1,) * 3)
    )
    simulation = SimpleNamespace(
        get_rigid_object=lambda uid: SimpleNamespace(
            get_vertices=lambda **_kw: originals[uid]
        ),
        get_articulation=lambda _uid: art,
    )
    bindings = SimpleNamespace(
        rigid_objects=[
            SimpleNamespace(entity_id=uid, simulation_uid=uid) for uid in originals
        ],
        articulations=[SimpleNamespace(entity_id="cabinet", simulation_uid="cabinet")],
        links=[
            SimpleNamespace(
                entity_id="cabinet_shell",
                articulation_id="cabinet",
                native_link_name="shell",
            )
        ],
        rigidized_articulations=(),
    )
    action.held_path_scene_vertices = place_recovery.scene_vertices(
        simulation, bindings
    )
    context.scene.entities["cabinet_shell"] = SimpleNamespace(
        pose=torch.eye(4)[None], confidence=1.0
    )
    return bindings


def test_articulated_scene_preserves_opening_and_uses_observed_link_pose(monkeypatch):
    action, request, context, original = _fixture(monkeypatch)
    original.joint_trajectory.positions[:, 1, 2] = 0.24
    positions = original.joint_trajectory.positions
    before = gate.inspect_payload(action, request, context, positions)
    _with_articulated_panels(monkeypatch, action, context)
    after = gate.inspect_payload(action, request, context, positions)
    assert before["valid"] and after["valid"]
    assert after["minimum_table_gap_m"] == before["minimum_table_gap_m"]
    assert gate.phase_scope(action, request, context)["owned"]
    context.scene.entities["cabinet_shell"].pose[:, 1, 3] = -0.3
    blocked = gate.inspect_payload(action, request, context, positions)
    assert not blocked["valid"]
    assert blocked["rejected"][0]["entity"] == "cabinet"


def test_articulation_link_missing_pose_is_unavailable_not_clearance(monkeypatch):
    action, request, context, original = _fixture(monkeypatch)
    _with_articulated_panels(monkeypatch, action, context)
    del context.scene.entities["cabinet_shell"]
    result = gate.inspect_payload(
        action, request, context, original.joint_trajectory.positions[:, [0, -1]]
    )
    assert not result["valid"]
    assert "cabinet_shell" in result["unavailable"]
