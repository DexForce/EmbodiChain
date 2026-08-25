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

"""Focused contracts for GenSim's receiver-hold handover action."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch

from embodichain.gen_sim.action_engine.capabilities import (
    HeldObjectHandOver,
    HeldObjectHandOverOptions,
    build_atomic_capability_registry,
)
from embodichain.lab.sim.atomic_actions import (
    ActionInvocation,
    AntipodalAffordance,
    AtomicActionEngine,
    ControlPartCommandProfile,
    GraspGoal,
    HeldObjectState,
    MotionPolicy,
    ObjectSemantics,
    PlanningContext,
    RobotObservation,
    SceneSnapshot,
    TaskState,
)
from embodichain.lab.sim.planners import MotionGenerator
from embodichain.toolkits.graspkit import (
    ParallelJawGraspPoseGenerator,
    ParallelJawGripperModelCfg,
)

_HAND_DOF = 1
_ROBOT_DOF = 6
_CONTROL_DT = 1.0 / 60.0


class _GraspGenerator(ParallelJawGraspPoseGenerator):
    def __init__(self) -> None:
        super().__init__(ParallelJawGripperModelCfg(model_id="test_gripper"))

    def get_valid_grasp_poses(
        self,
        *,
        mesh_vertices: torch.Tensor,
        mesh_triangles: torch.Tensor,
        obj_poses: torch.Tensor,
        approach_direction: torch.Tensor,
        obj_longest_axis: torch.Tensor | None = None,
        is_positive_part: bool | torch.Tensor = True,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        del (
            mesh_vertices,
            mesh_triangles,
            approach_direction,
            obj_longest_axis,
            is_positive_part,
        )
        return [
            (torch.eye(4).unsqueeze(0), torch.zeros(1))
            for _ in range(obj_poses.shape[0])
        ]

    def get_best_grasp_poses(self, **kwargs: object):
        poses = kwargs["obj_poses"]
        assert isinstance(poses, torch.Tensor)
        return (
            torch.ones(poses.shape[0], dtype=torch.bool),
            poses,
            torch.zeros(poses.shape[0]),
        )

    def get_dual_arm_valid_grasp_poses(self, **kwargs: object):
        del kwargs
        raise AssertionError("Receiver-hold handover uses one destination grasp.")


def _motion_generator() -> MotionGenerator:
    robot = Mock()
    robot.device = torch.device("cpu")
    robot.dof = _ROBOT_DOF
    robot.control_parts = {
        "left_arm": object(),
        "left_hand": object(),
        "right_arm": object(),
        "right_hand": object(),
    }
    joint_ids = {
        "left_arm": [0, 1],
        "left_hand": [2],
        "right_arm": [3, 4],
        "right_hand": [5],
    }
    robot.get_joint_ids.side_effect = lambda name: list(joint_ids[name])
    robot.get_qpos.return_value = torch.zeros(1, _ROBOT_DOF)
    robot.compute_fk.side_effect = lambda qpos, **_kwargs: torch.eye(4).repeat(
        qpos.shape[0], 1, 1
    )

    generator = object.__new__(MotionGenerator)
    generator.robot = robot
    generator.device = torch.device("cpu")
    generator.planner = Mock()
    generator.planner.cfg.planner_type = "stub"
    generator.planner.collision_world_info = None
    generator.planner.preserve_plan_samples = False
    return generator


def _engine() -> AtomicActionEngine:
    generator = _motion_generator()
    profiles = {
        hand: ControlPartCommandProfile.joint_positions(
            open=torch.zeros(_HAND_DOF),
            grasp=torch.ones(_HAND_DOF),
        )
        for hand in ("left_hand", "right_hand")
    }
    grasp = _GraspGenerator()
    engine = AtomicActionEngine(
        generator,
        control_profiles=profiles,
        grasp_pose_generators={"left_hand": grasp, "right_hand": grasp},
        load_builtins=False,
    )
    engine.register(HeldObjectHandOver())
    return engine


def _semantics(label: str = "can") -> ObjectSemantics:
    return ObjectSemantics(
        label=label,
        entity_id=label,
        geometry={},
        affordance=AntipodalAffordance(
            object_label=label,
            mesh_vertices=torch.tensor(
                [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]]
            ),
            mesh_triangles=torch.tensor([[0, 1, 2]]),
        ),
    )


def _context(semantics: ObjectSemantics) -> PlanningContext:
    relation = torch.eye(4).unsqueeze(0)
    task = TaskState(
        batch_size=1,
        device="cpu",
        held_objects={
            "left_arm": HeldObjectState(
                semantics=semantics,
                object_to_eef=relation,
                grasp_xpos=relation,
            )
        },
    )
    qpos = torch.zeros(1, _ROBOT_DOF)
    return PlanningContext(
        robot=RobotObservation(0.0, qpos, torch.zeros_like(qpos)),
        task=task,
        scene=SceneSnapshot.empty(),
        env_ids=torch.tensor([0]),
        control_dt=_CONTROL_DT,
    )


def _invocation(
    engine: AtomicActionEngine,
    semantics: ObjectSemantics,
    *,
    final_x: float = 0.0,
) -> ActionInvocation:
    middle = torch.eye(4)
    final = middle.clone()
    final[0, 3] = final_x
    binding = engine.bind_control_parts(
        "hand_over",
        {
            "source": {"motion": "left_arm", "grasp": "left_hand"},
            "destination": {"motion": "right_arm", "grasp": "right_hand"},
        },
    )
    return ActionInvocation(
        skill_id="hand_over",
        goal=GraspGoal(semantics=semantics),
        binding=binding,
        motion_policy=MotionPolicy(sample_count=24),
        skill_options=HeldObjectHandOverOptions(
            middle_object_pose=middle,
            final_object_pose=final,
            hand_interp_steps=2,
            hold_steps=2,
            retreat_steps=4,
        ),
    )


def _install_planner(monkeypatch: pytest.MonkeyPatch) -> None:
    def plan(
        _generator: MotionGenerator,
        _control_part: str,
        start_qpos: torch.Tensor,
        _target_poses: torch.Tensor,
        n_waypoints: int,
        _motion_policy: MotionPolicy,
        _control_dt: float | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.ones(start_qpos.shape[0], dtype=torch.bool),
            start_qpos.unsqueeze(1).repeat(1, n_waypoints, 1),
        )

    monkeypatch.setattr(
        "embodichain.gen_sim.action_engine.capabilities.held_hand_over."
        "plan_named_arm_trajectory",
        plan,
    )


def test_handover_capability_matches_installed_receiver_hold_action() -> None:
    capability = build_atomic_capability_registry().get("HandOver")

    assert capability.action_type is HeldObjectHandOver
    assert capability.config_type is HeldObjectHandOverOptions
    assert capability.action_type.GoalType is GraspGoal


def test_receiver_hold_plan_transfers_ownership_and_preserves_phase_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_planner(monkeypatch)
    engine = _engine()
    semantics = _semantics()
    context = _context(semantics)

    plan = engine.plan(_invocation(engine, semantics), context)
    projected = plan.expected_effects.apply(context.task, plan.plan_success)

    assert plan.plan_success.tolist() == [True]
    assert projected.get_held_object("left_arm") is None
    received = projected.get_held_object("right_arm")
    assert received is not None
    assert received.semantics.entity_id == "can"
    assert [segment.name for segment in plan.segments] == [
        "transfer",
        "approach",
        "close",
        "hold",
        "release",
        "retreat",
    ]
    assert plan.joint_trajectory is not None
    positions = plan.joint_trajectory.positions
    close = plan.segment("close")
    release = plan.segment("release")
    retreat = plan.segment("retreat")
    torch.testing.assert_close(positions[:, close.stop - 1, 5], torch.ones(1))
    torch.testing.assert_close(positions[:, release.stop - 1, 2], torch.zeros(1))
    torch.testing.assert_close(
        positions[:, release.start :, 5],
        torch.ones_like(positions[:, release.start :, 5]),
    )
    torch.testing.assert_close(
        positions[:, retreat.start : retreat.stop, 3:5],
        positions[:, retreat.start : retreat.start + 1, 3:5].expand_as(
            positions[:, retreat.start : retreat.stop, 3:5]
        ),
    )


def test_receiver_hold_rejects_delivery_away_from_exchange(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_planner(monkeypatch)
    engine = _engine()
    semantics = _semantics()

    with pytest.raises(ValueError, match="receiver remains stationary"):
        engine.plan(_invocation(engine, semantics, final_x=0.1), _context(semantics))


def test_receiver_hold_rejects_a_different_requested_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_planner(monkeypatch)
    engine = _engine()

    with pytest.raises(ValueError, match="object held by the source"):
        engine.plan(
            _invocation(engine, _semantics("other")),
            _context(_semantics("held")),
        )
