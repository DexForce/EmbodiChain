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
from typing import ClassVar
from unittest.mock import Mock

import pytest
import torch

from examples.sim.atomic_action.defaults import (
    _sample_defaults,
    _scene_args,
)
from examples.sim.atomic_action.host import (
    AtomicSkillRun,
    _sample_geometry,
    _same_geometry,
)
from examples.sim.atomic_action.support import (
    create_parallel_jaw_grasp_pose_generator,
    create_tutorial_robot_cfg,
)
from embodichain.lab.sim.atomic_actions import (
    ActionOptions,
    Affordance,
    AffordancePoseCandidates,
    AffordanceSamplingContext,
    AtomicAction,
    AtomicActionEngine,
    JointPositionGoal,
    JOINT_POSITION_CAPABILITY,
    PlannerDiagnostics,
    SkillBindingContract,
    SkillEndpointRequirement,
    SkillResourceSlot,
    TimedTrajectory,
)


def _host_with_samples(poses: torch.Tensor, *, valid: torch.Tensor, mode: str):
    count, candidates = valid.shape
    sample = Affordance().sample_candidates(
        AffordancePoseCandidates(
            poses=poses, costs=torch.zeros_like(valid, dtype=torch.float32), valid=valid
        ),
        sampling=AffordanceSamplingContext(
            count=count, branch_overrides=tuple(range(count))
        ),
        env_ids=torch.arange(count),
        key="test:grasp",
        reference_poses=torch.eye(4).repeat(count, 1, 1),
    )

    class SampleAction(AtomicAction[JointPositionGoal, ActionOptions]):
        skill_id: ClassVar[str] = "pick_up"
        GoalType: ClassVar[type] = JointPositionGoal
        binding_contract: ClassVar[SkillBindingContract] = SkillBindingContract(
            slots=(
                SkillResourceSlot(
                    "primary",
                    endpoints=(
                        SkillEndpointRequirement(
                            "motion",
                            capabilities=frozenset({JOINT_POSITION_CAPABILITY}),
                        ),
                    ),
                ),
            )
        )

        def _plan(self, request, context):
            # Distinct IK paths must never substitute for actual affordance diversity.
            target = 1.0 + torch.arange(count, dtype=torch.float32)[:, None] * 0.1
            positions = (
                target[:, None].expand(-1, 21, 3)
                * torch.linspace(0, 1, 21)[None, :, None]
            )
            return self.build_plan(
                request,
                context,
                success=True,
                trajectory=TimedTrajectory.from_uniform_step(
                    positions, env_ids=context.env_ids, step_dt=0.01
                ),
                segment_lengths={"approach": 5, "close": 5, "lift": 11},
                diagnostics=PlannerDiagnostics(
                    backend="stub", metadata={"affordance_sample": sample.metadata}
                ),
            )

    robot = Mock()
    robot.device = torch.device("cpu")
    robot.dof = 3
    robot.joint_names = ("a", "b", "c")
    robot.control_parts = {"custom_motion": object()}
    robot.get_qpos.return_value = torch.zeros(count, 3)
    robot.get_qvel.return_value = torch.zeros(count, 3)
    robot.get_joint_ids.return_value = [0, 1, 2]
    robot.body_data = SimpleNamespace(qpos_limits=torch.tensor([[[-4.0, 4.0]] * 3]))
    generator = Mock()
    generator.robot = robot
    generator.device = torch.device("cpu")
    generator.planner.cfg.planner_type = "stub"
    engine = AtomicActionEngine(generator, load_builtins=False)
    engine.register(SampleAction())
    invocation = engine.make_invocation(
        "pick_up",
        JointPositionGoal(torch.ones(count, 3)),
        control_parts={"primary": {"motion": "custom_motion"}},
    )
    cfg = _sample_defaults("pickup")
    cfg.variants = count
    cfg.expansion = mode
    run = AtomicSkillRun(cfg)
    run.compile(engine, (invocation,), engine.initial_context(control_dt=0.01))
    return run, sample


def test_affordance_shortage_cannot_be_hidden_by_distinct_ik_paths() -> None:
    poses = torch.eye(4).repeat(4, 3, 1, 1)
    poses[:, :, 0, 3] = torch.tensor([0.0, 0.1, 0.2])
    valid = torch.ones(4, 3, dtype=torch.bool)
    valid[3, 1] = False
    run, sample = _host_with_samples(poses, valid=valid, mode="affordance")
    assert sample.metadata["reused"][-1]
    with pytest.raises(ValueError, match="repeats geometry"):
        run.validate_variants()


def test_trajectory_only_mode_rejects_changed_actual_grasp_geometry() -> None:
    poses = torch.eye(4).repeat(2, 1, 1, 1)
    poses[1, 0, 0, 3] = 0.02
    run, _ = _host_with_samples(
        poses, valid=torch.ones(2, 1, dtype=torch.bool), mode="trajectory"
    )
    with pytest.raises(ValueError, match="changed the reference affordance"):
        run.validate_variants()


def test_affordance_identity_uses_local_geometry_instead_of_world_translation() -> None:
    first = torch.eye(4)
    first[0, 3] = 0.1
    reference = torch.eye(4)
    reference[0, 3] = 5.0
    second = first.clone()
    second[0, 3] += 5.0
    geometry_a = _sample_geometry(
        {"selected_poses": first.tolist(), "reference_poses": torch.eye(4).tolist()}
    )
    geometry_b = _sample_geometry(
        {"selected_poses": second.tolist(), "reference_poses": reference.tolist()}
    )
    assert _same_geometry(geometry_a, geometry_b)


def test_cli_candidate_pool_has_no_hidden_approach_direction_draws() -> None:
    args = _scene_args(_sample_defaults("pickup"))
    generator = create_parallel_jaw_grasp_pose_generator(
        n_sample=16,
        force_refresh=False,
        robot=SimpleNamespace(cfg=create_tutorial_robot_cfg("ur5")),
        approach_direction_samples=args.grasp_approach_samples,
    )
    assert generator.algorithm_cfg.approach_direction_samples == 1


def test_recording_cadence_retains_the_complete_long_replay_within_its_budget() -> None:
    from examples.sim.atomic_action.support import (
        _recording_fps,
        AUTO_PLAY_RECORD_FPS,
        AUTO_PLAY_RECORD_MAX_MEMORY,
        RECORD_WIDTH,
        RECORD_HEIGHT,
    )

    duration = 190.0
    fps = _recording_fps(duration)
    assert 0 < fps < AUTO_PLAY_RECORD_FPS
    expected_bytes = (int(duration * fps) + 2) * RECORD_WIDTH * RECORD_HEIGHT * 3
    assert expected_bytes <= AUTO_PLAY_RECORD_MAX_MEMORY * 1024 * 1024
    assert _recording_fps(10.0) == AUTO_PLAY_RECORD_FPS


def test_long_standalone_playback_does_not_validate_an_unused_recording_budget() -> (
    None
):
    from argparse import Namespace
    from examples.sim.atomic_action.support import replay_trajectory

    simulation = SimpleNamespace(
        sim_config=SimpleNamespace(physics_dt=1.0),
        is_newton_backend=False,
        update=Mock(),
    )
    robot = Mock()
    trajectory = TimedTrajectory.from_positions(
        torch.zeros(1, 2, 1),
        dt=torch.tensor([[0.0, 2400.0]]),
        env_ids=torch.tensor([0]),
    )
    replay_trajectory(
        simulation,
        robot,
        trajectory,
        Namespace(auto_play=False),
        video_prefix="unused",
        hold_steps=0,
    )
    simulation.update.assert_called_once_with(1.0, 2400)
    assert robot.set_qpos.call_count == 2
