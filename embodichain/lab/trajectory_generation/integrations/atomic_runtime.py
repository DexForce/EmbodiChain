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

"""Fresh PickUp action plans for physically qualified trajectory candidates."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from collections import Counter
from dataclasses import replace
import math
from typing import TYPE_CHECKING

import torch

from embodichain.lab.sim.atomic_actions import (
    ActionInvocation,
    ActionPlan,
    AtomicActionEngine,
    GraspGoal,
    JointPositionTarget,
    PickUp,
    TimedTrajectory,
    ExecutionRunner,
    ExecutionRunnerCfg,
    RunnerStatus,
    EffectVerificationResult,
    CommandAcknowledgement,
    CommandAckStatus,
    SimulationExecutionAdapter,
)
from embodichain.lab.sim.atomic_actions.goals import resolve_pose_goal
from embodichain.lab.sim.atomic_actions.invocation import ResolvedActionRequest
from embodichain.lab.sim.atomic_actions.state import PlanningContext
from embodichain.lab.sim.atomic_actions.tracking import FeedbackTerminalAcceptance
from embodichain.lab.sim.motion.trajectory_augmentation import (
    CandidateTrajectoryBatch,
    TrajectoryTemplate,
    ValidationResult,
    ValidationCheck,
)

if TYPE_CHECKING:
    from embodichain.lab.trajectory_generation.initial_state import (
        PreparedBatch,
        FixedSceneHost,
    )
    from .contact import PickUpMotionValidator

__all__ = ["PickUpRuntimeSource"]


class PickUpRuntimeSource:
    """Materialize protected PickUp candidates through the registered skill.

    One new invocation and session are required per prepared batch. References
    contain transit/approach/close/lift/hold phases. Only the declared arm transit
    may change; contact targets and timing remain identical to the references.
    The reference's initial observation is omitted from the runtime command
    sequence, preserving the recorder's T commands / T+1 observations convention.

    Args:
        engine: Atomic engine sharing the collection host's robot.
        invocation_factory: Constructs a fresh PickUp invocation for a batch.
        templates: Owned, qualified references in physical environment order.
    """

    def __init__(
        self,
        engine: AtomicActionEngine,
        *,
        invocation_factory: Callable[[PreparedBatch], ActionInvocation],
        templates: Sequence[TrajectoryTemplate],
    ) -> None:
        if not isinstance(engine, AtomicActionEngine) or not callable(
            invocation_factory
        ):
            raise TypeError("Provide an atomic engine and invocation factory")
        if not isinstance(engine.actions.get("pick_up"), PickUp):
            raise ValueError("The engine must register the PickUp implementation")
        self.engine = engine
        self.invocation_factory = invocation_factory
        self.templates = tuple(replace(value) for value in templates)
        if len(self.templates) != engine.robot.num_instances:
            raise ValueError("References must cover the complete physical batch")
        for template in self.templates:
            if (
                template.joint_names != tuple(engine.robot.joint_names)
                or tuple(p.phase_id for p in template.phases)
                != ("transit", "approach", "close", "lift", "hold")
                or len(template.positions) < 3
                or template.phases[0].start_index != 0
                or any(p.allowed_operators for p in template.phases[1:])
            ):
                raise ValueError(
                    "Provide full-joint references with protected PickUp phases"
                )
        if any(t.phases != self.templates[0].phases for t in self.templates[1:]):
            raise ValueError("Runtime PickUp rows must share phase boundaries")

    def materialize(
        self,
        request: ResolvedActionRequest,
        context: PlanningContext,
        candidates: Sequence[CandidateTrajectoryBatch | None],
    ) -> ActionPlan:
        """Build commands and pending effects against the supplied measured state.

        Args:
            request: Fresh engine-resolved PickUp request with arm tracking.
            context: Current physical state, scene revisions and control clock.
            candidates: One candidate per physical row, or None for idle rows.

        Returns:
            A new PickUp plan retaining the selected joint branch and phases.

        Raises:
            ValueError: If candidates differ from qualified contact references,
                initial state, source identity, timing or endpoint layout.
        """
        robot = self.engine.robot
        if request.skill_id != "pick_up" or type(request.goal) is not GraspGoal:
            raise ValueError("Runtime source requires a PickUp GraspGoal")
        if request.goal.grasp_xpos is None or request.goal.semantics.entity_id is None:
            raise ValueError(
                "Runtime PickUp requires an explicit grasp and scene entity"
            )
        if len(candidates) != len(self.templates) or context.batch_size != len(
            candidates
        ):
            raise ValueError("Candidates must cover every physical row")
        if not torch.equal(context.env_ids.cpu(), torch.arange(len(candidates))):
            raise ValueError("Runtime context must preserve physical row order")
        motion = request.binding.endpoint("primary", "motion")
        grasp = request.binding.endpoint("primary", "grasp")
        arm = motion.require_target(JointPositionTarget)
        hand = grasp.require_target(JointPositionTarget)
        if request.tracking_policy.in_flight is None or not isinstance(
            request.tracking_policy.terminal, FeedbackTerminalAcceptance
        ):
            raise ValueError(
                "Runtime collection requires in-flight and terminal feedback"
            )
        if not motion.tracking_channels or grasp.tracking_channels:
            raise ValueError(
                "Track arm feedback; qualify the grasp endpoint by physical contact"
            )
        if set(arm.joint_ids) & set(hand.joint_ids):
            raise ValueError("Arm and hand endpoints must be disjoint")
        period = context.require_control_dt()
        reference = self.templates[0]
        length = len(reference.positions)
        positions = context.robot.qpos[:, None].repeat(1, length, 1)
        active = torch.zeros(
            context.batch_size, dtype=torch.bool, device=positions.device
        )
        for row, (candidate, template) in enumerate(zip(candidates, self.templates)):
            if candidate is None:
                continue
            if (
                len(candidate.identities) != 1
                or candidate.source_row_indices is None
                or candidate.source_row_indices.tolist() != [row]
            ):
                raise ValueError("Candidate does not belong to this physical row")
            identity = candidate.identities[0]
            if (identity.source_id, identity.source_revision, identity.template_id) != (
                template.source_id,
                template.source_revision,
                template.template_id,
            ):
                raise ValueError(
                    "Candidate source differs from its qualified reference"
                )
            if (
                candidate.joint_names != template.joint_names
                or candidate.phases[0] != template.phases
                or int(candidate.valid_length[0]) != length
            ):
                raise ValueError("Candidate joint order or phase boundaries changed")
            qpos = candidate.positions[0, :length].to(positions)
            if not torch.allclose(qpos[0], context.robot.qpos[row], atol=1e-6, rtol=0):
                raise ValueError(
                    "Candidate initial joints differ from current observation"
                )
            if not torch.allclose(
                candidate.dt[0, 1:length].double(),
                torch.full_like(candidate.dt[0, 1:length].double(), period),
                atol=1e-8,
                rtol=0,
            ):
                raise ValueError(
                    "Candidate timing differs from the runtime control clock"
                )
            protected = template.phases[1].start_index
            other = [
                i
                for i in range(robot.dof)
                if i not in arm.joint_ids and i not in robot.mimic_ids
            ]
            baseline = template.positions.to(qpos)
            if (
                not torch.equal(qpos[protected:], baseline[protected:])
                or not torch.equal(qpos[:, other], baseline[:, other])
                or tuple(template.controlled_joint_indices) != arm.joint_ids
            ):
                raise ValueError(
                    "Candidate modified protected contact or non-arm coordinates"
                )
            positions[row] = qpos
            active[row] = True
        if not active.any():
            raise ValueError("At least one candidate must be active")
        grasp_poses = resolve_pose_goal(
            request.goal.grasp_xpos, context, name="grasp_xpos"
        ).to(positions)
        if grasp_poses.shape == (4, 4):
            grasp_poses = grasp_poses.repeat(context.batch_size, 1, 1)
        close = reference.phases[2].start_index
        actual_grasp = robot.compute_fk(
            qpos=positions[:, close - 1, list(arm.joint_ids)],
            name=arm.control_part,
            to_matrix=True,
        )
        if not torch.allclose(
            actual_grasp[active], grasp_poses[active], atol=1e-4, rtol=0
        ):
            raise ValueError(
                "Qualified contact geometry differs from the current grasp goal"
            )
        segment_lengths = {
            p.phase_id: p.stop_index - p.start_index - (1 if i == 0 else 0)
            for i, p in enumerate(reference.phases)
        }
        command_positions = positions[:, 1:]
        # Keep the host's double-precision control clock. Float32 intervals can
        # make the runtime wait past an otherwise exact physical command tick.
        intervals = torch.full(
            command_positions.shape[:2],
            period,
            dtype=torch.float64,
            device=positions.device,
        )
        intervals[:, 0] = 0
        return self.engine.actions["pick_up"].materialize_trajectory(
            request,
            context,
            success=active,
            trajectory=TimedTrajectory(
                positions=command_positions,
                velocities=torch.zeros_like(command_positions),
                accelerations=None,
                dt=intervals,
                env_ids=context.env_ids,
            ),
            grasp_xpos=grasp_poses,
            segment_lengths=segment_lengths,
        )

    def start(
        self,
        host: FixedSceneHost,
        binding: PreparedBatch,
        candidates: Sequence[CandidateTrajectoryBatch | None],
        *,
        contact_validator: PickUpMotionValidator,
        control_dt: float,
    ) -> _PickUpRuntimeRollout:
        """Start a fresh observed session after the host restores its batch.

        Args:
            host: Exclusive pure-simulation collection owner.
            binding: Current prepared batch epoch.
            candidates: Candidates in physical row order; None denotes idle rows.
            contact_validator: The collector's native contact and grasp validator.
            control_dt: Fixed period shared by runtime commands and observations.

        Returns:
            A serialized runtime driver borrowed by QposRolloutExecutor.
        """
        host.assert_current(binding)
        if (
            host.env is not None
            or host.adapter.robot is not self.engine.robot
            or contact_validator.robot is not self.engine.robot
            or contact_validator.sim is not host.adapter.sim
        ):
            raise ValueError(
                "Runtime PickUp requires the same pure-sim host and validator"
            )
        return _PickUpRuntimeRollout(
            self, host, binding, candidates, contact_validator, control_dt
        )


class _PickUpRuntimeRollout:
    """Strict candidate transport around the ordinary observed atomic runner.

    Physics and T/T+1 recording stay with the collection executor. A replan may
    use the registered skill but its commands never cross this transport: the
    entire active batch is cancelled and retained only as failed audit evidence.
    """

    def __init__(self, source, host, binding, candidates, contact, control_dt):
        self.source, self.host, self.binding, self.contact = (
            source,
            host,
            binding,
            contact,
        )
        self.control_dt = control_dt
        self.started_at = float(host.adapter.sim.simulation_time)
        self.count = 0
        self.failure = None
        self._expected = None
        self._sent = False
        self._events = [Counter() for _ in candidates]
        self._effect_translation_error = [None for _ in candidates]
        self._effect_rotation_error = [None for _ in candidates]
        self._effect_passed = torch.zeros(
            len(candidates), dtype=torch.bool, device=source.engine.device
        )
        self._active = torch.tensor(
            [c is not None for c in candidates],
            dtype=torch.bool,
            device=source.engine.device,
        )
        self._transport = SimulationExecutionAdapter(
            host.adapter.sim, source.engine.robot, control_dt=control_dt
        )
        invocation = source.invocation_factory(binding)
        if (
            not isinstance(invocation, ActionInvocation)
            or invocation.skill_id != "pick_up"
            or invocation.goal.semantics.entity_id != contact.profile.object_id
        ):
            raise ValueError(
                "Runtime invocation must target the physically validated PickUp object"
            )
        context = source.engine.initial_context(
            timestamp=self.now(), control_dt=control_dt
        )
        self._arm = invocation.binding.endpoint("primary", "motion").require_target(
            JointPositionTarget
        )
        session = source.engine.start(
            (invocation,),
            context,
            eligible_mask=self._active,
            initial_plan_provider=lambda request, current: source.materialize(
                request, current, candidates
            ),
        )
        self._frame_count = session.active_plan.commands.frame_count
        self.runner = ExecutionRunner(
            session,
            self,
            self,
            clock=self,
            cfg=ExecutionRunnerCfg(
                minimum_cycle_time=0.0,
                hold_on_completion=False,
                hold_during_effect_verification=False,
            ),
        )

    def now(self):
        return float(self.host.adapter.sim.simulation_time)

    def sleep(self, duration):
        raise RuntimeError("Collection executor owns the only physics clock")

    def observe(self, task_state):
        self.host.assert_current(self.binding)
        return self.source.engine.initial_context(
            task=task_state, timestamp=self.now(), control_dt=self.control_dt
        )

    def send(self, command, *, timeout):
        session = self.runner.session
        if session.attempt_generation != 0:
            return CommandAcknowledgement(
                CommandAckStatus.REJECTED,
                "Recovery commands cannot become expert candidate data",
            )
        if (
            self._expected is None
            or self._sent
            or not torch.equal(command.active_mask, self._active)
        ):
            return CommandAcknowledgement(
                CommandAckStatus.REJECTED,
                "Runtime command cohort differs from the candidate",
            )
        if not torch.allclose(
            command.hold_duration[self._active],
            torch.full_like(command.hold_duration[self._active], self.control_dt),
            atol=1e-8,
            rtol=0,
        ):
            return CommandAcknowledgement(
                CommandAckStatus.REJECTED,
                "Runtime command timing differs from the candidate",
            )
        for endpoint in command.commands:
            ids = list(endpoint.target.joint_ids)
            if not torch.allclose(
                endpoint.payload.positions[self._active],
                self._expected[self._active][:, ids].to(endpoint.payload.positions),
                atol=1e-6,
                rtol=0,
            ):
                return CommandAcknowledgement(
                    CommandAckStatus.REJECTED,
                    "Runtime attempted a different candidate command",
                )
        result = self._transport.send(command, timeout=timeout)
        self._sent = result.accepted
        return result

    def hold(self, targets, context, *, timeout):
        return self._transport.hold(targets, context=context, timeout=timeout)

    def cancel(self, targets, *, timeout):
        return self._transport.cancel(targets, timeout=timeout)

    def _remember(self, result):
        if result.tick is not None:
            for event in result.tick.events:
                for row, active in enumerate(event.env_mask.tolist()):
                    if active:
                        self._events[row][event.kind.value] += 1

    def _abort(self, message):
        self.failure = str(message)[:512]
        self._remember(self.runner.cancel(self.failure))
        raise RuntimeError(self.failure)

    def dispatch(self, expected, active):
        self.host.assert_current(self.binding)
        if not math.isclose(
            self.now(),
            self.started_at + self.count * self.control_dt,
            rel_tol=0,
            abs_tol=1e-8,
        ):
            self._abort("Runtime collection clock differs from the candidate")
        if (
            tuple(self._active.tolist()) != tuple(active)
            or self.count >= self._frame_count
        ):
            self._abort("Runtime requires a synchronized fixed candidate cohort")
        self._expected, self._sent = expected.clone(), False
        result = self.runner.step()
        self._remember(result)
        if not self._sent or result.status is not RunnerStatus.RUNNING:
            self._abort(
                "Runtime candidate execution failed: "
                + str(result.message or self._events)
            )
        self.count += 1

    def _verify_effect(self, context, request):
        accepted = torch.zeros_like(request.env_mask)
        observations = self.contact.observations()
        key = self.runner.session.active_plan.expected_effects.held_object_updates
        if (
            set(request.expected_effects.held_object_updates) != set(key)
            or len(key) != 1
        ):
            raise ValueError(
                "Runtime PickUp must verify exactly its declared held-object effect"
            )
        held = next(iter(request.expected_effects.held_object_updates.values()))
        expected = observations["object_pose"] @ held.object_to_eef.to(
            observations["object_pose"]
        )
        # HeldObjectState uses the motion endpoint's solver TCP frame. The
        # contact profile can deliberately declare a different native TCP.
        # Derive pose evidence from this tick's measured joints, in the same
        # frame used to qualify the candidate; native contacts remain mandatory.
        measured = self.source.engine.robot.compute_fk(
            qpos=context.robot.qpos[:, list(self._arm.joint_ids)],
            name=self._arm.control_part,
            to_matrix=True,
        ).to(expected)
        translation = (expected[:, :3, 3] - measured[:, :3, 3]).norm(dim=1)
        rotation = expected[:, :3, :3].transpose(1, 2) @ measured[:, :3, :3]
        angle = torch.acos(
            ((rotation.diagonal(dim1=1, dim2=2).sum(-1) - 1) / 2).clamp(-1, 1)
        )
        p = self.contact.profile
        for row, active in enumerate(request.env_mask.tolist()):
            if active:
                self._effect_translation_error[row] = float(translation[row])
                self._effect_rotation_error[row] = float(angle[row])
            accepted[row] = (
                active
                and self.contact.rollout_validation(row).accepted
                and bool(
                    translation[row] <= p.max_relative_translation
                    and angle[row] <= p.max_relative_rotation
                )
            )
        self._effect_passed |= accepted
        return EffectVerificationResult(
            request.verification_id,
            accepted,
            request.env_mask & ~accepted,
            torch.zeros_like(accepted),
            torch.zeros_like(accepted),
        )

    def finish(self):
        if self.count != self._frame_count:
            self._abort("Runtime candidate did not dispatch every command")
        if not math.isclose(
            self.now(),
            self.started_at + self.count * self.control_dt,
            rel_tol=0,
            abs_tol=1e-8,
        ):
            self._abort("Runtime terminal clock differs from the candidate")
        self._expected = None
        # Two observation-only updates: establish the terminal boundary, then
        # consume freshly measured evidence. They never advance physical time.
        for _ in range(2):
            result = self.runner.step(effect_verifier=self._verify_effect)
            self._remember(result)
            if result.status is not RunnerStatus.RUNNING:
                break
            if self.runner.session.pending_effect is None:
                self._abort("Runtime terminal feedback or phase gate did not complete")
        if self.runner.status is not RunnerStatus.COMPLETED or not bool(
            self._effect_passed[self._active].all()
        ):
            self._abort("Runtime physical effect verification failed")

    def stop(self):
        if self.runner.status is RunnerStatus.RUNNING:
            self.failure = self.failure or "Runtime collection interrupted"
            self._remember(self.runner.cancel(self.failure))

    def validation(self, row):
        passed = (
            self.failure is None
            and self.runner.status is RunnerStatus.COMPLETED
            and self.runner.session.attempt_generation == 0
            and bool(self._effect_passed[row])
        )
        return ValidationResult(
            (
                ValidationCheck(
                    "atomic_runtime",
                    "passed" if passed else "failed",
                    self.failure
                    or "Observed runtime tracking and physical effect verification",
                    {
                        "command_count": self.count,
                        "plan_attempts": self.runner.session.attempt_generation + 1,
                    },
                ),
            )
        )

    def metadata(self, row):
        return {
            "command_count": self.count,
            "plan_attempts": self.runner.session.attempt_generation + 1,
            "events": dict(self._events[row]),
            "status": self.runner.status.value,
            "effect_verified": bool(self._effect_passed[row]),
            "effect_translation_error_m": self._effect_translation_error[row],
            "effect_rotation_error_rad": self._effect_rotation_error[row],
            "effect_tcp_frame": {
                "source": "robot.compute_fk",
                "control_part": self._arm.control_part,
            },
            "contact_tcp_frame": {
                "source": "robot.get_link_pose",
                "link": self.contact.profile.tcp_link,
                "offset": self.contact.profile.tcp_offset,
            },
            "failure": self.failure,
        }
