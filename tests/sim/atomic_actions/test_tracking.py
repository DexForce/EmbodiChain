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
from typing import ClassVar

import pytest
import torch

from embodichain.lab.sim.atomic_actions.bindings import JointPositionTarget
from embodichain.lab.sim.atomic_actions.runtime_commands import (
    EndpointCommand,
    JointPositionPayload,
)
from embodichain.lab.sim.atomic_actions.state import (
    PlanningContext,
    RobotObservation,
    SceneSnapshot,
    TaskState,
)
from embodichain.lab.sim.atomic_actions.tracking import (
    BASE_POSE_CHANNEL,
    JOINT_POSITION_CHANNEL,
    EndpointTrackingChannelBinding,
    EndpointTrackingFeedbackAddress,
    FeedbackTerminalAcceptance,
    InFlightTrackingPolicy,
    JointPositionTrackingMetric,
    JointPositionTrackingState,
    PoseTrackingEvaluator,
    PoseTrackingMetric,
    PoseTrackingState,
    TimedTerminalAcceptance,
    TimedTrackingSequence,
    TrackingFeedbackSourceRef,
    TrackingFrame,
    TrackingMetricCfg,
    TrackingPolicy,
    TrackingProjectorRef,
    TrackingRuntime,
    TrackingSetpoint,
    WholeBodyPoseTrackingEvaluator,
    WholeBodyPoseTrackingMetric,
    WholeBodyPoseTrackingState,
)


@dataclass(frozen=True, slots=True)
class _AlternateJointMetric(TrackingMetricCfg):
    """Different metric identity deliberately sharing the joint channel."""

    metric_id: ClassVar[str] = "joint.alternate"
    channel_id: ClassVar[str] = JOINT_POSITION_CHANNEL


def _joint_binding(target: JointPositionTarget) -> EndpointTrackingChannelBinding:
    return EndpointTrackingChannelBinding(
        channel_id=JOINT_POSITION_CHANNEL,
        source=TrackingFeedbackSourceRef(
            provider_id="planning_context.robot",
            revision="1",
            address=EndpointTrackingFeedbackAddress(
                target=target,
                channel_id=JOINT_POSITION_CHANNEL,
            ),
        ),
        projector=TrackingProjectorRef(
            projector_id="joint_position_payload",
            revision="1",
        ),
    )


def _context(qpos: torch.Tensor) -> PlanningContext:
    batch_size = qpos.shape[0]
    device = qpos.device
    return PlanningContext(
        robot=RobotObservation(
            timestamp=1.0,
            qpos=qpos,
            qvel=torch.zeros_like(qpos),
            root_pose=torch.eye(4, device=device).repeat(batch_size, 1, 1),
        ),
        task=TaskState.empty(batch_size=batch_size, device=device),
        scene=SceneSnapshot.empty(),
        env_ids=torch.arange(batch_size, dtype=torch.long, device=device),
    )


def test_joint_policy_factory_separates_in_flight_and_terminal_contracts() -> None:
    policy = TrackingPolicy.joint_position(
        in_flight_max_abs_error=0.1,
        terminal_max_abs_error=0.08,
        terminal_settle_timeout=0.25,
    )

    assert policy.in_flight is not None
    assert policy.in_flight.metrics == (JointPositionTrackingMetric(0.1),)
    assert isinstance(policy.terminal, FeedbackTerminalAcceptance)
    assert policy.terminal.metrics == (JointPositionTrackingMetric(0.08),)
    assert policy.terminal.settle_timeout == pytest.approx(0.25)


def test_policy_rejects_ambiguous_metric_id_for_a_shared_channel() -> None:
    with pytest.raises(ValueError, match="same exact metric ID"):
        TrackingPolicy(
            in_flight=InFlightTrackingPolicy(
                metrics=(JointPositionTrackingMetric(0.1),)
            ),
            terminal=FeedbackTerminalAcceptance(metrics=(_AlternateJointMetric(),)),
        )


def test_timed_policy_is_an_explicit_no_feedback_contract() -> None:
    policy = TrackingPolicy.timed(settle_duration=0.2)

    assert policy.in_flight is None
    assert isinstance(policy.terminal, TimedTerminalAcceptance)
    assert policy.terminal.settle_duration == pytest.approx(0.2)


def test_tracking_values_and_routes_own_tensor_and_target_snapshots() -> None:
    positions = torch.tensor([[0.1, 0.2]])
    target = JointPositionTarget(control_part="arm", joint_ids=(0, 1))
    setpoint = TrackingSetpoint(
        endpoint_key=("arm", "controller"),
        binding=_joint_binding(target),
        desired=JointPositionTrackingState(positions),
    )

    positions.add_(1.0)

    assert torch.equal(setpoint.desired.positions, torch.tensor([[0.1, 0.2]]))
    assert setpoint.binding.source.address.target is not target
    assert setpoint.key == ("arm", "controller", JOINT_POSITION_CHANNEL)


def test_timed_tracking_sequence_owns_env_ids_and_validates_batches() -> None:
    env_ids = torch.tensor([2, 5], dtype=torch.long)
    target = JointPositionTarget(control_part="arm", joint_ids=(0, 1))
    frame = TrackingFrame(
        (
            TrackingSetpoint(
                endpoint_key=("arm", "controller"),
                binding=_joint_binding(target),
                desired=JointPositionTrackingState(torch.zeros(2, 2)),
            ),
        )
    )
    sequence = TimedTrackingSequence(env_ids=env_ids, frames=(frame,))

    env_ids[0] = 99

    assert sequence.env_ids.tolist() == [2, 5]
    assert sequence.batch_size == 2
    assert sequence.frame_count == 1


def test_timed_tracking_sequence_rejects_mismatched_setpoint_batch() -> None:
    target = JointPositionTarget(control_part="arm", joint_ids=(0, 1))
    frame = TrackingFrame(
        (
            TrackingSetpoint(
                endpoint_key=("arm", "controller"),
                binding=_joint_binding(target),
                desired=JointPositionTrackingState(torch.zeros(1, 2)),
            ),
        )
    )

    with pytest.raises(ValueError, match="setpoint batch"):
        TimedTrackingSequence(
            env_ids=torch.tensor([0, 1], dtype=torch.long),
            frames=(frame,),
        )


def test_builtin_runtime_projects_observes_and_evaluates_joint_positions() -> None:
    target = JointPositionTarget(control_part="arm", joint_ids=(1, 3))
    binding = _joint_binding(target)
    command = EndpointCommand(
        target=target,
        payload=JointPositionPayload(positions=torch.tensor([[0.3, 0.5], [0.1, 0.2]])),
    )
    runtime = TrackingRuntime.with_builtins()
    desired = runtime.project(command, binding)
    setpoint = TrackingSetpoint(("arm", "controller"), binding, desired)
    context = _context(
        torch.tensor(
            [
                [0.0, 0.32, 0.0, 0.49],
                [0.0, 0.25, 0.0, 0.2],
            ]
        )
    )

    feedback = runtime.observe(setpoint, context)
    evaluation = runtime.evaluate(
        setpoint,
        feedback,
        JointPositionTrackingMetric(tolerance=0.05),
    )

    assert torch.equal(evaluation.accepted_mask, torch.tensor([True, False]))
    assert torch.allclose(
        evaluation.component_errors["joint_max_abs"],
        torch.tensor([0.02, 0.15]),
    )


def test_pose_metric_preserves_translation_and_rotation_components() -> None:
    desired = torch.eye(4).repeat(2, 1, 1)
    observed = desired.clone()
    observed[0, 0, 3] = 0.01
    observed[1, :2, :2] = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    evaluator = PoseTrackingEvaluator()

    evaluation = evaluator.evaluate(
        PoseTrackingState(desired),
        PoseTrackingState(observed),
        torch.ones(2, dtype=torch.bool),
        PoseTrackingMetric(translation_tolerance=0.02, rotation_tolerance=0.1),
    )

    assert evaluation.channel_id == BASE_POSE_CHANNEL
    assert evaluation.accepted_mask.tolist() == [True, False]
    assert set(evaluation.component_errors) == {"translation", "rotation"}


def test_whole_body_metric_requires_pose_and_joint_acceptance() -> None:
    root = torch.eye(4).repeat(2, 1, 1)
    desired = WholeBodyPoseTrackingState(root, torch.zeros(2, 2))
    observed = WholeBodyPoseTrackingState(
        root,
        torch.tensor([[0.01, 0.0], [0.0, 0.2]]),
    )

    evaluation = WholeBodyPoseTrackingEvaluator().evaluate(
        desired,
        observed,
        torch.ones(2, dtype=torch.bool),
        WholeBodyPoseTrackingMetric(joint_position_tolerance=0.05),
    )

    assert evaluation.accepted_mask.tolist() == [True, False]
    assert torch.allclose(
        evaluation.component_errors["joint_max_abs"], torch.tensor([0.01, 0.2])
    )
