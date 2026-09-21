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

"""GenSim-owned service contracts migrated from shared integration tests."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from embodichain.gen_sim.task_engine._task_program.align_held import _AlignHeldLowerer

from embodichain.gen_sim.task_engine._task_program.configured import (
    decode_task_lowerer,
)
from embodichain.lab.task_program.integrations.configured import _decode_robot_profile
from embodichain.gen_sim.task_engine._task_program.services import (
    _AbsolutePoseTarget,
    _CoordinatedHoldLowerer,
    _CoordinatedTransportRoute,
    _MoveHeldObjectLowerer,
    _MoveHeldObjectRoute,
    _PickLowerer,
    _PickRoute,
)
from embodichain.lab.gym.utils._component_composition import _resolve_gym_components
from embodichain.lab.task_program.integrations._configured_composition import (
    _compose_integration_payload,
    _resolve_task_program_components,
)
from embodichain.lab.sim.atomic_actions import (
    AntipodalAffordance,
    CoordinatedPickmentOptions,
    EntityState,
    GraspGoal,
    HandOverOptions,
    HeldObjectPoseGoal,
    MoveHeldObjectOptions,
    ObjectSemantics,
    PickUpOptions,
    PlanningContext,
    RobotObservation,
    SceneSnapshot,
    TaskState,
)
from embodichain.lab.task_program.semantics import (
    HeldObjectRelation,
    RegisteredSemanticCall,
    SceneObjectRef,
    SemanticEffectKind,
)
from embodichain.utils.utility import load_config

__all__: list[str] = []


def test_motion_velocity_guard_rejects_joint_branch_jumps_per_environment() -> None:
    from embodichain.gen_sim.task_engine._task_program.motion import _velocity_validity

    positions = torch.tensor([[[0.0], [0.02], [0.04]], [[0.0], [3.14], [3.15]]])
    dt = torch.tensor([[0.0, 0.04, 0.04], [0.0, 0.04, 0.04]])
    limits = torch.ones(2, 1)
    assert _velocity_validity(positions, dt, limits).tolist() == [True, False]
    dt[0, 1] = 0
    assert _velocity_validity(positions, dt, limits).tolist() == [False, False]
    with pytest.raises(ValueError, match="matching batched"):
        _velocity_validity(positions, dt, torch.ones(1, 1))


def test_final_plan_velocity_rejects_resampled_and_hand_jumps(monkeypatch) -> None:
    from embodichain.gen_sim.task_engine._task_program import assembly

    monkeypatch.setattr(assembly, "_joint_velocity_limits", lambda *a: torch.ones(1, 2))
    for joint in (0, 1):
        positions = torch.zeros(1, 2, 2)
        positions[0, 1, joint] = 0.2
        plan = SimpleNamespace(
            plan_success=torch.tensor([True]),
            joint_trajectory=SimpleNamespace(
                positions=positions, dt=torch.tensor([[0.0, 0.04]])
            ),
        )
        result = assembly._validate_final_plan_velocity(plan, object())
        assert result["valid_mask"].tolist() == [False]
        assert result["diagnostics"][0]["control_joint_index"] == joint
        assert plan.plan_success.tolist() == [True]


def test_failed_plan_can_expose_grasp_proposals_without_claiming_attachment() -> None:
    from embodichain.gen_sim.task_engine._task_program.assembly import (
        _planned_grasp_candidates,
    )

    transform = torch.eye(4).unsqueeze(0)
    held = SimpleNamespace(
        semantics=SimpleNamespace(entity_id="block"),
        object_to_eef=transform,
        grasp_xpos=transform,
    )
    updates = {"left": held, "right": None}
    plan = SimpleNamespace(
        plan_success=torch.tensor([False]),
        expected_effects=SimpleNamespace(held_object_updates=updates),
    )
    proposals = _planned_grasp_candidates(plan)
    assert len(proposals) == 1
    assert proposals[0]["evidence_scope"] == "proposal_only_not_observed_attachment"
    assert proposals[0]["object_id"] == "block"
    assert proposals[0]["resource"] == "left"
    proposals[0]["object_to_eef"][0][0][0] = 0.0
    assert transform[0, 0, 0] == 1.0
    assert plan.plan_success.tolist() == [False]
    assert updates["left"] is held


def test_initial_success_requires_final_trajectory_evidence() -> None:
    from embodichain.gen_sim.task_engine._task_program.assembly import (
        _validate_final_plan_velocity,
    )

    plan = SimpleNamespace(plan_success=torch.tensor([True]), joint_trajectory=None)
    with pytest.raises(ValueError, match="final joint trajectory"):
        _validate_final_plan_velocity(plan, object())


def test_velocity_diagnostics_locate_peak_and_serialize_zero_duration() -> None:
    import json
    from embodichain.gen_sim.task_engine._task_program.motion import (
        _velocity_diagnostics,
    )

    positions = torch.tensor([[[0.0, 0.0], [0.01, 0.4], [0.02, 0.4]]])
    dt = torch.tensor([[0.0, 0.04, 0.04]])
    limits = torch.tensor([[1.0, 2.0]])
    record = _velocity_diagnostics(positions, dt, limits)[0]
    assert record["frame"] == 1
    assert record["control_joint_index"] == 1
    assert record["speed_ratio"] == pytest.approx(5.0)
    dt[0, 1] = 0
    records = _velocity_diagnostics(positions, dt, limits)
    assert records[0]["speed_ratio"] is None
    json.dumps(records, allow_nan=False)


def test_motion_wrapper_masks_unsafe_rows_without_clipping_commands(
    monkeypatch,
) -> None:
    from embodichain.gen_sim.task_engine._task_program import motion

    generator = object.__new__(motion.ApproachMotionGenerator)
    generator.robot = SimpleNamespace(get_qvel_limits=lambda **kw: torch.ones(2, 1))
    monkeypatch.setattr(motion, "_joint_velocity_limits", lambda *a: torch.ones(2, 1))
    positions = torch.tensor([[[0.0], [0.02]], [[0.0], [3.14]]])
    original = motion.PlanResult(
        success=torch.tensor([True, True]),
        positions=positions,
        dt=torch.tensor([[0.0, 0.04], [0.0, 0.04]]),
    )
    monkeypatch.setattr(motion.MotionGenerator, "generate", lambda *a, **kw: original)
    result = generator.generate(
        [motion.PlanState(move_type=motion.MoveType.EEF_MOVE, xpos=torch.eye(4))],
        motion.MotionGenOptions(control_part="arm"),
    )
    assert result.success.tolist() == [True, False]
    assert original.success.tolist() == [True, True]
    torch.testing.assert_close(result.positions, positions)


def test_single_pose_transport_samples_cartesian_path(monkeypatch) -> None:
    from embodichain.gen_sim.task_engine._task_program import motion

    start = torch.eye(4).unsqueeze(0)
    start[:, 2, 3] = 1.1
    target = start.clone()
    target[:, :3, 3] = torch.tensor([0.2, 0.1, 1.3])
    target[:, :3, :3] = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    generator = object.__new__(motion.ApproachMotionGenerator)
    generator.robot = SimpleNamespace(compute_fk=lambda **kw: start.clone())
    monkeypatch.setattr(motion, "_joint_velocity_limits", lambda *a: torch.ones(1, 1))
    result = motion.PlanResult(
        success=torch.tensor([True]),
        positions=torch.zeros(1, 5, 1),
        dt=torch.full((1, 5), 0.04),
    )
    backend = Mock(return_value=result)
    monkeypatch.setattr(motion.MotionGenerator, "generate", backend)
    options = motion.MotionGenOptions(
        strategy="ik_interp",
        control_part="arm",
        start_qpos=torch.zeros(1, 1),
        sample_count=5,
    )

    generator.generate(
        [motion.PlanState(move_type=motion.MoveType.EEF_MOVE, xpos=target)], options
    )

    samples = backend.call_args.args[0]
    forwarded = backend.call_args.kwargs["options"]
    assert len(samples) == 4
    assert forwarded.preserve_cartesian_samples is True
    assert forwarded.is_linear is True
    assert forwarded.sample_count == 5
    for index, sample in enumerate(samples, start=1):
        expected = start[:, :3, 3].lerp(target[:, :3, 3], index / 4)
        torch.testing.assert_close(sample.xpos[:, :3, 3], expected)
        assert float(sample.xpos[0, 2, 3]) >= 1.1
    torch.testing.assert_close(samples[-1].xpos, target)
    assert options.preserve_cartesian_samples is False


def test_velocity_invalid_e6_retries_with_denser_samples(monkeypatch) -> None:
    from embodichain.gen_sim.task_engine._task_program import motion

    generator = object.__new__(motion.ApproachMotionGenerator)
    start = torch.eye(4).unsqueeze(0)
    start[:, 2, 3] = 1.0
    generator.robot = SimpleNamespace(compute_fk=lambda **kw: start.clone())
    monkeypatch.setattr(motion, "_joint_velocity_limits", lambda *a: torch.ones(1, 1))
    calls = []

    def backend(_self, targets, *, options):
        calls.append((len(targets), options.sample_count))
        success = torch.tensor([len(calls) > 1])
        return motion.PlanResult(
            success=success,
            positions=torch.zeros(1, options.sample_count, 1),
            dt=torch.full((1, options.sample_count), 0.04),
        )

    monkeypatch.setattr(motion.MotionGenerator, "generate", backend)
    options = motion.MotionGenOptions(
        strategy="ik_interp",
        control_part="arm",
        start_qpos=torch.zeros(1, 1),
        sample_count=140,
    )
    result = generator.generate(
        [motion.PlanState(move_type=motion.MoveType.EEF_MOVE, xpos=start)], options
    )
    assert result.success.tolist() == [True]
    assert [sample_count for _, sample_count in calls] == [140, 260]
    assert options.sample_count == 140


def test_stack_place_allows_equivalent_tcp_roll_without_moving_release(
    monkeypatch,
) -> None:
    from embodichain.gen_sim.task_engine._task_program import stack_place
    from embodichain.lab.sim.atomic_actions.primitives.place import PlaceGoal
    from embodichain.lab.task_program.compiler.lowering import SemanticLowering
    from embodichain.lab.task_program.semantics.calls import RegisteredSemanticCall

    pose = torch.eye(4)
    pose[2, 3] = 1.20
    monkeypatch.setattr(
        stack_place._RelativePlaceLowerer,
        "lower",
        lambda *a, **kw: SemanticLowering(goal=PlaceGoal(xpos=pose)),
    )
    lowerer = object.__new__(stack_place._StackPlaceLowerer)
    lowerer._approach = 0.02
    lowerer._release = 0.003
    lowerer._nominal = 0.01

    result = lowerer.lower(
        RegisteredSemanticCall(call_id="gen_sim.stack_place", arguments={}),
        context=SimpleNamespace(batch_size=1),
        bound=object(),
        option_template=object(),
    )

    assert result.goal.tcp_symmetry == "z_roll_180"
    torch.testing.assert_close(
        result.goal.xpos[0, :, 2, 3], torch.tensor([1.21, 1.193])
    )


def test_velocity_limits_use_urdf_names_and_preserve_stricter_runtime_limits(
    tmp_path,
) -> None:
    from embodichain.gen_sim.task_engine._task_program.motion import (
        _joint_velocity_limits,
    )

    path = tmp_path / "robot.urdf"
    path.write_text(
        '<robot><joint name="a"><limit velocity="2"/></joint><joint name="b"><limit velocity="3"/></joint></robot>'
    )
    robot = SimpleNamespace(
        cfg=SimpleNamespace(fpath=path),
        joint_names=["a", "b"],
        get_joint_ids=lambda **kw: [1, 0],
        get_qvel_limits=lambda **kw: torch.tensor([[1e10, 1.0]]),
    )
    torch.testing.assert_close(
        _joint_velocity_limits(robot, "arm"), torch.tensor([[3.0, 1.0]])
    )
    robot.joint_names = ["missing", "b"]
    with pytest.raises(ValueError, match="URDF velocity limit"):
        _joint_velocity_limits(robot, "arm")


def test_curobo_adapter_allocates_only_declared_mesh_cache(monkeypatch) -> None:
    from embodichain.gen_sim.task_engine._task_program import assembly
    from embodichain.lab.sim.motion import motion_generator
    from embodichain.lab.task_program.semantics import SceneCollisionRole

    cfgs = []
    monkeypatch.setattr(
        motion_generator, "MotionGenerator", lambda cfg: cfgs.append(cfg) or object()
    )
    captured = {}

    def factory(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(create_adapter=lambda: "adapter")

    monkeypatch.setattr(assembly, "_TaskFactory", factory)
    monkeypatch.setattr(assembly, "install_grasp_filters", lambda *a: {})
    bindings = [
        SimpleNamespace(
            entity_id=f"object_{i}",
            simulation_uid=f"native_{i}",
            collision_role=SceneCollisionRole.DYNAMIC,
        )
        for i in range(3)
    ]
    registration = SimpleNamespace(
        assert_unchanged=lambda: None,
        robot_profile_binding=SimpleNamespace(
            presets=[
                SimpleNamespace(motion_policy=SimpleNamespace(strategy="motion_gen"))
            ]
        ),
        scene_binding=SimpleNamespace(rigid_objects=bindings),
    )
    objects = {b.simulation_uid: Mock() for b in bindings}
    env = SimpleNamespace(
        robot=SimpleNamespace(uid="robot"),
        sim=SimpleNamespace(get_rigid_object=objects.get),
        step_dt=0.04,
    )
    adapter = assembly.TaskAdapterFactory(registration, "test", (), ())
    assert adapter.create_adapter(env) == "adapter"
    captured["motion_generator_factory"]()
    world = cfgs[0].planner_cfg.world
    assert world.representation == "auto"
    assert set(world.rigid_objects) == {b.entity_id for b in bindings}
    assert world.dynamic_obstacle_names == [b.entity_id for b in bindings]


def test_relative_place_corrects_goal_without_mutating_grasp_state(monkeypatch) -> None:
    from embodichain.gen_sim.task_engine._task_program import services
    from embodichain.lab.sim.atomic_actions.primitives.place import (
        PlaceGoal,
        PlaceOptions,
    )
    from embodichain.lab.task_program.compiler.lowering import SemanticLowering

    object_pose = torch.eye(4).repeat(2, 1, 1)
    object_pose[:, 0, 3] = torch.tensor([0.2, 0.4])
    measured_eef = object_pose.clone()
    measured_eef[:, 2, 3] = torch.tensor([0.1, 0.15])
    nominal = torch.eye(4).repeat(2, 1, 1)
    canonical = SemanticLowering(
        goal=PlaceGoal(
            xpos=services.SceneEntityPose(
                "table",
                relative_pose=nominal,
                world_displacement=torch.tensor([0.0, 0.0, 0.1]),
                world_orientation=object_pose[:, :3, :3].clone(),
            )
        )
    )
    monkeypatch.setattr(
        services._RelativePlaceLowerer, "lower", lambda *a, **kw: canonical
    )
    robot = Mock()
    robot.compute_fk.return_value = measured_eef
    lowerer = services._ObservedRelativePlaceLowerer(
        (
            services._RelativePlaceRoute(
                object_id="cup",
                reference_entity_id="table",
                relation="on",
                world_displacement=(0.0, 0.0, 0.1),
                world_yaw_offset=math.pi / 6.0,
            ),
        ),
        robot,
    )
    context = SimpleNamespace(
        batch_size=2,
        env_ids=torch.tensor([1, 3]),
        robot=SimpleNamespace(qpos=torch.zeros(2, 4)),
        scene=SimpleNamespace(
            entities={"cup": SimpleNamespace(pose=object_pose, confidence=1.0)}
        ),
    )
    bound = SimpleNamespace(
        binding=SimpleNamespace(
            resources={
                "primary": SimpleNamespace(
                    endpoints={
                        "motion": SimpleNamespace(
                            runtime_target=SimpleNamespace(
                                joint_ids=(1, 3), control_part="arm"
                            )
                        )
                    }
                )
            }
        )
    )
    result = lowerer.lower(
        RegisteredSemanticCall(
            call_id="simulation.place_relative",
            arguments={"object": "cup", "reference": "table", "relation": "on"},
        ),
        context=context,
        bound=bound,
        option_template=PlaceOptions(),
    )
    torch.testing.assert_close(
        object_pose @ result.goal.xpos.relative_pose, measured_eef
    )
    angle = torch.tensor(math.pi / 6.0)
    yaw = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle), 0.0],
            [torch.sin(angle), torch.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    torch.testing.assert_close(
        result.goal.xpos.world_orientation,
        yaw @ object_pose[:, :3, :3],
    )
    torch.testing.assert_close(canonical.goal.xpos.relative_pose, nominal)
    torch.testing.assert_close(
        canonical.goal.xpos.world_orientation, object_pose[:, :3, :3]
    )
    assert result.registered_effect is canonical.registered_effect
    assert robot.compute_fk.call_args.kwargs["env_ids"] == [1, 3]


@pytest.mark.parametrize("success", [True, False])
def test_motion_diagnostics_preserve_the_underlying_result(
    monkeypatch, success
) -> None:
    import json
    from embodichain.gen_sim.task_engine._task_program import motion

    generator = object.__new__(motion.ApproachMotionGenerator)
    result = motion.PlanResult(success=torch.tensor([success]))
    backend = Mock(return_value=result)
    log = Mock()
    monkeypatch.setattr(motion.MotionGenerator, "generate", backend)
    monkeypatch.setattr(motion.logger, "log_info", log)
    targets = [motion.PlanState(move_type=motion.MoveType.EEF_MOVE, xpos=torch.eye(4))]
    options = motion.MotionGenOptions(control_part="left_arm")
    assert generator.generate(targets, options) is result
    backend.assert_called_once_with(targets, options=options)
    record = json.loads(log.call_args.args[0].removeprefix("GenSim motion plan: "))
    assert record == {
        "control_part": "left_arm",
        "input_target_count": 1,
        "planned_target_count": 1,
        "input_poses": [torch.eye(4).tolist()],
        "success": [success],
    }


def test_current_pose_upright_binding_preserves_each_environments_position() -> None:
    poses = torch.eye(4).repeat(2, 1, 1)
    poses[:, :3, 3] = torch.tensor([[0.1, -0.2, 1.0], [0.3, 0.2, 1.2]])
    before = poses.clone()
    robot = Mock()
    lowerer = _AlignHeldLowerer(
        (("can", "current_object_pose", True, (1.0, 0.0, 0.0), None),), robot
    )
    context = SimpleNamespace(
        batch_size=2,
        task=SimpleNamespace(
            get_held_object=lambda key: SimpleNamespace(
                semantics=SimpleNamespace(entity_id="can")
            )
        ),
        scene=SimpleNamespace(
            entities={"can": SimpleNamespace(pose=poses, confidence=1.0)}
        ),
    )
    bound = SimpleNamespace(
        binding=SimpleNamespace(
            resources={
                "primary": SimpleNamespace(
                    endpoints={"motion": SimpleNamespace(task_state_key="left")}
                )
            }
        )
    )
    result = lowerer.lower(
        RegisteredSemanticCall(
            call_id="gen_sim.align_held",
            arguments={
                "object": "can",
                "target": "current_object_pose",
                "preserve_yaw": True,
            },
        ),
        context=context,
        bound=bound,
        option_template=MoveHeldObjectOptions(),
    )
    target = result.goal.object_target_pose
    torch.testing.assert_close(target[:, :3, 3], before[:, :3, 3])
    torch.testing.assert_close(
        target[:, :3, 0],
        torch.tensor([[0.0, 0.0, 1.0]]).repeat(2, 1),
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(poses, before)
    assert robot.mock_calls == []


def test_upright_alignment_lifts_before_rotating() -> None:
    pose = torch.eye(4).unsqueeze(0)
    pose[:, :3, :3] = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    pose[:, :3, 3] = torch.tensor([0.1, 0.2, 1.0])
    original = pose.clone()
    robot = SimpleNamespace(
        cfg=SimpleNamespace(solver_cfg={"arm": SimpleNamespace(root_link_name="base")}),
        get_link_pose=lambda **kw: torch.eye(4).unsqueeze(0),
    )
    lowerer = _AlignHeldLowerer(
        (("can", "staging", False, (0.0, 0.0, 1.0), (0.1, 0.2, 1.3)),), robot
    )
    held = SimpleNamespace(
        semantics=SimpleNamespace(entity_id="can"),
        object_to_eef=torch.eye(4).unsqueeze(0),
    )
    context = SimpleNamespace(
        batch_size=1,
        env_ids=torch.tensor([0]),
        task=SimpleNamespace(get_held_object=lambda key: held),
        scene=SimpleNamespace(
            entities={"can": SimpleNamespace(pose=pose, confidence=1.0)}
        ),
    )
    bound = SimpleNamespace(
        binding=SimpleNamespace(
            resources={
                "primary": SimpleNamespace(
                    endpoints={
                        "motion": SimpleNamespace(
                            task_state_key="arm",
                            runtime_target=SimpleNamespace(control_part="arm"),
                        )
                    }
                )
            }
        )
    )
    result = lowerer.lower(
        RegisteredSemanticCall(
            call_id="gen_sim.align_held",
            arguments={
                "object": "can",
                "target": "staging",
                "preserve_yaw": False,
            },
        ),
        context=context,
        bound=bound,
        option_template=MoveHeldObjectOptions(),
    )
    waypoints = result.goal.object_target_pose
    assert waypoints.shape == (1, 2, 4, 4)
    torch.testing.assert_close(waypoints[:, 0, :3, :3], original[:, :3, :3])
    torch.testing.assert_close(
        waypoints[:, :, :3, 3], torch.tensor([[[0.1, 0.2, 1.3]]]).expand(-1, 2, -1)
    )
    torch.testing.assert_close(
        waypoints[:, 1, :3, 2], torch.tensor([[0.0, 0.0, 1.0]]), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(pose, original)


def test_coordinated_hold_lowerer_retains_both_verified_attachments() -> None:
    semantics = ObjectSemantics(
        affordance=AntipodalAffordance(),
        geometry={},
        label="tray",
        entity_id="tray",
    )
    lowerer = _CoordinatedHoldLowerer(
        (
            _CoordinatedTransportRoute(
                object_id="tray",
                target_id="tray_up",
                world_displacement=(0.0, 0.0, 0.14),
            ),
        ),
        (semantics,),
    )

    lowering = lowerer.lower(
        RegisteredSemanticCall(
            call_id="simulation.coordinated_hold",
            arguments={
                "object": "tray",
                "target": "tray_up",
                "world_displacement": [0.0, 0.0, 0.14],
            },
        ),
        context=PlanningContext(
            robot=RobotObservation(
                timestamp=1.0,
                qpos=torch.zeros((1, 1)),
                qvel=torch.zeros((1, 1)),
            ),
            task=TaskState(batch_size=1, device="cpu"),
            scene=SceneSnapshot(
                timestamp=1.0,
                version=1,
                entities={"tray": EntityState(torch.eye(4).unsqueeze(0))},
            ),
            env_ids=torch.tensor([0]),
        ),
        bound=None,  # type: ignore[arg-type]
        option_template=CoordinatedPickmentOptions(release=False),
    )

    assert lowering.registered_effect is not None
    assert lowering.registered_effect.effect_kind is SemanticEffectKind.ATTACH
    assert [effect.relation for effect in lowering.registered_effect.held_objects] == [
        HeldObjectRelation.ATTACHED,
        HeldObjectRelation.ATTACHED,
    ]
    factory = decode_task_lowerer(
        {
            "kind": "coordinated_hold",
            "routes": [
                {
                    "object_id": "tray",
                    "target_id": "tray_up",
                    "world_displacement": [0.0, 0.0, 0.14],
                }
            ],
        },
        path="integration.runtime_services.registered_semantic_lowerers[0]",
    )
    assert factory.call_id == "simulation.coordinated_hold"


def test_configured_handover_uses_baseline_timing_and_rejects_removed_wait_field() -> (
    None
):
    path = (
        Path(__file__).parents[3]
        / "embodichain_tasks/configs/tasks/manipulation/hand_over"
        / "task.dual_ur5_dh_pgi_140_80.yaml"
    )
    physical = _resolve_gym_components(load_config(path), base_dir=path.parent)
    _, task, policy = _resolve_task_program_components(
        physical.config["task_program"], base_dir=path.parent
    )
    payload = _compose_integration_payload(
        task=task,
        policy=policy,
        skill_profile=physical.embodiment_skill_profile,
        scene=task["scene_binding"],
    )["robot_profile"]
    payload["presets"][0]["action_options"]["hand_over"].update(
        retreat_distance=0.12, retreat_steps=28
    )
    before = deepcopy(payload)
    options = (
        _decode_robot_profile(payload).presets[0].action_option_templates["hand_over"]
    )

    assert payload == before
    assert type(options) is HandOverOptions
    assert options.retreat_distance == pytest.approx(0.12)
    assert options.retreat_steps == 28
    assert not hasattr(options, "source_release_settle_steps")
    payload["presets"][0]["action_options"]["hand_over"][
        "source_release_settle_steps"
    ] = 16
    with pytest.raises(ValueError, match="source_release_settle_steps"):
        _decode_robot_profile(payload)


def test_configured_transport_binds_one_baseline_pose_without_planning() -> None:
    pose = _AbsolutePoseTarget((0.1, 0.2, 0.8), (1.0, 0.0, 0.0, 0.0))
    lowerer = _MoveHeldObjectLowerer(
        (_MoveHeldObjectRoute("part", "inspection", pose),)
    )
    forbidden_robot = Mock()
    goal = lowerer.lower(
        RegisteredSemanticCall(
            call_id="simulation.move_held_object",
            arguments={"object": "part", "target": "inspection"},
        ),
        context=forbidden_robot,
        bound=forbidden_robot,
        option_template=MoveHeldObjectOptions(),
    ).goal
    forbidden_robot.assert_not_called()
    assert forbidden_robot.mock_calls == []
    assert type(goal) is HeldObjectPoseGoal
    torch.testing.assert_close(goal.object_target_pose, pose.to_matrix())
    assert not hasattr(goal, "alternative_object_target_poses")
    lookahead = lowerer.pick_lookahead_targets(
        RegisteredSemanticCall(
            call_id="simulation.move_held_object", arguments={"target": "inspection"}
        ),
        picked_object=SceneObjectRef("different_part"),
        bound=None,
        previous_target=None,
    )
    assert lookahead is None


def test_transport_decoder_rejects_old_alternative_pose_declarations() -> None:
    pose = {
        "kind": "pose",
        "position": [0.1, 0.2, 0.8],
        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    with pytest.raises(ValueError, match="alternatives"):
        decode_task_lowerer(
            {
                "kind": "move_held_object",
                "routes": [
                    {
                        "object_id": "part",
                        "target_id": "inspection",
                        "pose": pose,
                        "alternatives": [pose],
                    }
                ],
            },
            path="lowerer",
        )


def test_configured_pick_keeps_baseline_goal_and_preset_ownership() -> None:
    semantics = ObjectSemantics(
        affordance=AntipodalAffordance(), geometry={}, entity_id="part"
    )
    route = _PickRoute("part", "inspection")
    options = PickUpOptions(
        pick_object_part="bottom", approach_direction=torch.tensor([1.0, 0.0, 0.0])
    )
    lowering = _PickLowerer((route,), (semantics,)).lower(
        RegisteredSemanticCall(
            call_id="simulation.pick",
            arguments={"object": "part", "target": "inspection"},
        ),
        context=None,
        bound=None,
        option_template=options,
    )
    assert type(lowering.goal) is GraspGoal
    assert lowering.goal.semantics is semantics
    assert lowering.skill_options is None
    assert options.pick_object_part == "bottom"
    assert options.downstream_object_target_poses == ()
    assert lowering.registered_effect.effect_kind is SemanticEffectKind.ATTACH


def test_pick_decoder_rejects_removed_runtime_option_declarations() -> None:
    with pytest.raises(ValueError, match="required_object_target_poses"):
        decode_task_lowerer(
            {
                "kind": "pick",
                "routes": [
                    {
                        "object_id": "part",
                        "target_id": "inspection",
                        "required_object_target_poses": [],
                    }
                ],
            },
            path="pick",
        )


def test_task_pick_alias_factory_has_stable_immutable_identity() -> None:
    payload = {
        "kind": "pick",
        "call_id": "gen_sim.pick.step_01",
        "routes": [{"object_id": "part", "target_id": "release"}],
    }
    first = decode_task_lowerer(payload, path="pick")
    second = decode_task_lowerer(payload, path="pick")
    assert first.call_id == "gen_sim.pick.step_01"
    assert first.lowerer_type.call_id == first.call_id
    assert type(first).__qualname__ == type(second).__qualname__
    assert first.target_descriptor.skill_id == "pick_up"
    robot = object()
    registry = Mock()
    registry.resolve.return_value = SceneObjectRef("part")
    registry.object_semantics.return_value = ObjectSemantics(
        affordance=AntipodalAffordance(), geometry={}, entity_id="part"
    )
    lowerer = first.create(
        simulation=None,
        robot=robot,
        scene_registry=registry,
        engine=SimpleNamespace(robot=robot, grasp_pose_generators={}),
    )
    assert type(lowerer).call_id == first.call_id
    result = lowerer.lower(
        RegisteredSemanticCall(
            call_id=first.call_id, arguments={"object": "part", "target": "release"}
        ),
        context=None,
        bound=None,
        option_template=PickUpOptions(),
    )
    assert type(result.goal) is GraspGoal
    assert result.skill_options is None
    with pytest.raises(FrozenInstanceError):
        first.routes = ()


@pytest.mark.parametrize(
    "kind",
    [
        "articulation_link_slide",
        "articulation_link_press",
        "articulation_link_twist",
        "release_safe_pick",
        "move_held_object_upright",
    ],
)
def test_task_decoder_rejects_out_of_scope_services(kind: str) -> None:
    with pytest.raises(ValueError, match="unsupported Task Engine service"):
        decode_task_lowerer({"kind": kind}, path="lowerer")
