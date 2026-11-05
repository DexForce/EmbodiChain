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

"""Explicit occurrence-owned assembly for the shared bounded Twist planner."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import math
from typing import TYPE_CHECKING
from uuid import uuid4

from embodichain.lab.sim.atomic_actions.articulation_evidence import (
    TwistEventTarget,
)
from embodichain.lab.sim.atomic_actions.execution import ExecutionPlanAttempt
from embodichain.lab.sim.atomic_actions.primitives.twist import TwistGoal
from embodichain.lab.sim.atomic_actions.primitives.twist_recovery import (
    TwistMeasuredFeedback,
    TwistPlanningOutcome,
    TwistPlanningRecovery,
    TwistRecoveryCursor,
    TwistRecoveryDecision,
    TwistRecoveryScope,
    TwistRecoverySnapshot,
)
from embodichain.lab.sim.atomic_actions.twist_planning import TwistPlanningSession
from embodichain.toolkits.graspkit.pg_grasp.native_contact import NativeQueryScope

if TYPE_CHECKING:
    from embodichain.lab.sim.atomic_actions.invocation import ResolvedActionRequest
    from embodichain.lab.sim.atomic_actions.plans import ActionPlan
    from embodichain.lab.sim.atomic_actions.state import PlanningContext
    from embodichain.lab.sim.motion.revolute_query import RevolutePathQueryFactory
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.atomic_actions import AtomicActionEngine
    from embodichain.lab.task_program.runtime.expansion import TaskProgramPlanRequest
    from .control_observation import ControlObservationWindow

__all__ = ["TwistOperationSession", "TwistOperationRegistry"]


class TwistOperationSession:
    """Bind declared execution occurrences to one scoped shared Twist cursor.

    Preparation only reads the supplied planning observation and creates a
    borrowed geometry query. Logical outcome commit, plan adoption and measured
    feedback are separate explicit boundaries. Failed plans can commit logical
    selection but cannot be adopted. Preview forks have independent cursor IDs.

    Args:
        robot: Exact Robot used by both the Atomic engine and geometry factory.
        geometry: Shared Motion query factory; its lifetime remains caller-owned.
        scope: Existing program/operation/episode/native-environment identity.
        invocation_ids: Complete exact execution occurrences allowed in this scope.
        parent_entity_id: Canonical articulation whose observed root is the parent.
        child_entity_id: Exact registered child/link observed by each Twist goal.
        joint_name: Exact observed joint coordinate shared by these occurrences.
        validate_execution_scope: Host guard rejecting replaced/closed episode owners
            at every planning, adoption and feedback boundary.
        validate_observation: Host guard for the active episode and observation
            clock domain. It must reject stale/replaced observations without
            refreshing state. It does not select policies or plan motion.
    """

    def __init__(
        self,
        robot: Robot,
        geometry: RevolutePathQueryFactory,
        scope: TwistRecoveryScope,
        *,
        invocation_ids: Sequence[str],
        parent_entity_id: str,
        child_entity_id: str,
        joint_name: str,
        validate_observation: Callable[[PlanningContext], None],
        validate_execution_scope: Callable[[], None],
    ) -> None:
        if type(scope) is not TwistRecoveryScope or scope.env_id != 0:
            raise ValueError("Twist operation sessions require an exact env0 scope.")
        if geometry.robot is not robot or robot.num_instances != 1:
            raise ValueError(
                "Twist session and query factory require the same B=1 Robot."
            )
        if isinstance(invocation_ids, str):
            raise ValueError("Twist occurrences must be an explicit sequence.")
        ids = tuple(invocation_ids)
        for value in (*ids, parent_entity_id, child_entity_id, joint_name):
            if type(value) is not str or not value or value != value.strip():
                raise ValueError("Twist operation bindings require exact identifiers.")
        if not ids or len(set(ids)) != len(ids) or parent_entity_id == child_entity_id:
            raise ValueError(
                "Twist occurrences must be unique with distinct parent/child."
            )
        if not callable(validate_observation) or not callable(validate_execution_scope):
            raise TypeError("Twist session requires an explicit observation guard.")
        self._robot, self._geometry = robot, geometry
        self._scope, self._ids = scope, frozenset(ids)
        self._invocation_ids = ids
        self._parent, self._child, self._joint = (
            parent_entity_id,
            child_entity_id,
            joint_name,
        )
        self._guard = validate_observation
        self._execution_guard = validate_execution_scope
        self._cursor = TwistRecoveryCursor(scope)
        self._closed = False
        self._adopted_outcome: TwistPlanningOutcome | None = None
        self._control_window: ControlObservationWindow | None = None
        self._event_target: TwistEventTarget | None = None
        self._physics_dt: float | None = None
        self._chunk_cursor = 0

    @property
    def robot(self) -> Robot:
        """Return the exact borrowed Robot served by this planning provider."""
        return self._robot

    @property
    def scope(self) -> TwistRecoveryScope:
        """Return the declared execution owner; object names never select a scope."""
        return self._scope

    @property
    def invocation_ids(self) -> tuple[str, ...]:
        """Return the exact ordered occurrence declaration owned by this operation."""
        return self._invocation_ids

    def _open(self) -> None:
        if self._closed:
            raise RuntimeError("Twist operation session is closed.")
        self._execution_guard()

    def snapshot(self) -> TwistRecoverySnapshot:
        """Return immutable owned cursor data without planning or sampling."""
        self._open()
        return self._cursor.snapshot()

    def fork_preview(self) -> TwistOperationSession:
        """Fork only logical state, borrowing geometry and the same observation guard."""
        self._open()
        preview = TwistOperationSession(
            self._robot,
            self._geometry,
            self._scope,
            invocation_ids=self._invocation_ids,
            parent_entity_id=self._parent,
            child_entity_id=self._child,
            joint_name=self._joint,
            validate_observation=self._guard,
            validate_execution_scope=self._execution_guard,
        )
        preview._cursor = self._cursor.fork_preview()
        preview._event_target = self._event_target
        return preview

    def prepare(
        self,
        request: ResolvedActionRequest,
        context: PlanningContext,
    ) -> TwistPlanningSession:
        """Prepare a fresh query from declared occurrence and observed pose/joint data."""
        self._open()
        goal = request.goal
        if (
            request.invocation_id not in self._ids
            or request.skill_id != "twist"
            or type(goal) is not TwistGoal
            or goal.joint_target is None
            or goal.semantics.entity_id != self._child
            or goal.joint_target.articulation_id != self._parent
            or goal.joint_target.joint_name != self._joint
        ):
            raise ValueError(
                "Twist request differs from its declared operation binding."
            )
        if (
            context.batch_size != 1
            or context.env_ids.tolist() != [0]
            or context.robot.root_pose is None
            or not math.isclose(
                context.robot.timestamp,
                context.scene.timestamp,
                rel_tol=0,
                abs_tol=1e-9,
            )
        ):
            raise ValueError(
                "Twist preparation requires one complete same-observation root/scene snapshot."
            )
        if (
            self._event_target is not None
            and goal.joint_target.position != self._event_target.target_qpos
        ):
            raise ValueError(
                "Twist goal differs from the declared measured-event target."
            )
        self._guard(context)
        state = self._cursor.snapshot()
        if (
            state.last_feedback_timestamp is not None
            and state.last_feedback_timestamp > context.robot.timestamp + 1e-9
        ):
            raise ValueError("Twist observation precedes its last measured feedback.")
        observed = context.scene.articulation_joints.get((self._parent, self._joint))
        if (
            observed is None
            or observed.position.numel() != 1
            or (observed.valid_mask is not None and not bool(observed.valid_mask.all()))
        ):
            raise ValueError(
                "Twist preparation requires a valid exact joint observation."
            )
        parent, child = (
            context.scene.entities[self._parent],
            context.scene.entities[self._child],
        )
        query = self._geometry.prepare(
            scope=NativeQueryScope(
                uuid4().hex, context.robot.timestamp, context.scene.version
            ),
            robot_root_pose=context.robot.root_pose[0],
            parent_pose=parent.pose if parent.pose.ndim == 2 else parent.pose[0],
            child_pose=child.pose if child.pose.ndim == 2 else child.pose[0],
            measured_joint_position=float(observed.position.reshape(-1)[0]),
        )
        try:
            hold = (
                self._robot.get_qpos(target=True).clone()
                if state.goal_complete
                else None
            )
            self._guard(context)
            self._open()
            return TwistPlanningSession(
                state,
                query,
                self._parent,
                request.invocation_id,
                request.revision,
                hold,
            )
        except BaseException:
            query.close()
            raise

    def commit_planning(self, plan: ActionPlan) -> TwistRecoverySnapshot:
        """Commit the immutable proposal even if final planning qualification failed."""
        self._open()
        if plan.invocation_id not in self._ids:
            raise ValueError(
                "Twist plan occurrence is not declared for this operation."
            )
        outcome = plan.planning_recovery_outcome
        if type(outcome) is not TwistPlanningOutcome or (
            outcome.invocation_id != plan.invocation_id
            or outcome.invocation_revision != plan.invocation_revision
        ):
            raise ValueError("Twist plan lacks its exact logical planning outcome.")
        if self._adopted_outcome is not None:
            raise ValueError("The previous Twist plan still awaits measured feedback.")
        return self._cursor.commit_planning(outcome)

    def adopt_plan(self, plan: ActionPlan) -> None:
        """Authorize physical feedback only for the successfully committed plan."""
        self._open()
        if plan.invocation_id not in self._ids:
            raise ValueError("Twist adoption occurrence is not declared.")
        self._cursor.adopt_plan(plan)
        self._adopted_outcome = plan.planning_recovery_outcome

    def commit_measurement(
        self, feedback: TwistMeasuredFeedback
    ) -> TwistRecoveryDecision:
        """Commit fresh adopted-plan feedback; the caller must honor rejected decisions."""
        self._open()
        if feedback.invocation_id not in self._ids:
            raise ValueError("Twist feedback occurrence is not declared.")
        decision = self._cursor.commit_measurement(feedback)
        self._adopted_outcome = None
        return decision

    def bind_control_observation(
        self,
        window: ControlObservationWindow,
        target: TwistEventTarget,
    ) -> None:
        """Bind the initial measured window after the host's ready qualification.

        The initial prepare sample is retained for whole-operation event checks,
        but excluded from the first chunk slice, matching the source cursor.
        This method does not assert readiness or turn a planning result into it.
        """
        from .control_observation import ControlObservationWindow

        self._open()
        if (
            type(window) is not ControlObservationWindow
            or type(target) is not TwistEventTarget
        ):
            raise TypeError("Twist observation requires typed window and event target.")
        if self._adopted_outcome is not None:
            raise ValueError("Cannot replace a Twist window during an adopted action.")
        if self._event_target is not None and self._event_target != target:
            raise ValueError(
                "Twist event target cannot change within an operation scope."
            )
        snapshot = window.projection()
        if not snapshot.available:
            raise ValueError("Twist initial control observation is unavailable.")
        self._control_window, self._event_target, self._physics_dt = (
            window,
            target,
            window.physics_dt,
        )
        self._chunk_cursor = len(snapshot.samples)

    def observe_chunk(
        self,
        *,
        observed_qpos: float,
        stable_positions: Sequence[float],
        stable_velocities: Sequence[float],
        feedback_timestamp: float,
    ) -> dict[str, object]:
        """Adapt one guarded raw window to shared Atomic chunk validation.

        Control-cadence joint reads and feedback time are supplied by the Lab
        post-policy host; the physical rule and recovery transition belong to
        Atomic. Raw acquisitions keep their separate actual simulation clock.
        """
        from embodichain.lab.sim.atomic_actions.twist_feedback import (
            commit_observed_twist_chunk,
        )

        self._open()
        if (
            self._control_window is None
            or self._event_target is None
            or self._physics_dt is None
        ):
            raise ValueError("Twist has no bound measured control window.")
        outcome = self._adopted_outcome
        if outcome is None:
            raise ValueError("Twist feedback requires an adopted installed plan.")
        observed = self._control_window.projection()
        result = commit_observed_twist_chunk(
            self._cursor,
            outcome,
            self._event_target,
            observed.samples,
            observed.parent_contacts,
            chunk_start=self._chunk_cursor,
            observed_qpos=observed_qpos,
            stable_positions=stable_positions,
            stable_velocities=stable_velocities,
            physics_dt=self._physics_dt,
            feedback_timestamp=feedback_timestamp,
            window_available=observed.available,
        )
        errors = list(observed.errors)
        if observed.samples and observed.samples[-1].qpos != observed_qpos:
            errors.append("Final joint read differs from the current raw acquisition.")
        result["observation_errors"] = errors
        self._adopted_outcome = None
        self._chunk_cursor = len(observed.samples)
        return result

    def on_plan_installed(self, attempt: ExecutionPlanAttempt) -> None:
        """Commit logical selection, adopting only a successful installed B=1 plan."""
        self._open()
        if type(attempt) is not ExecutionPlanAttempt:
            raise TypeError("Twist lifecycle requires exactly ExecutionPlanAttempt.")
        plan = attempt.plan
        if plan.invocation_id not in self._ids:
            raise ValueError("Installed Twist occurrence is not declared.")
        if plan.planning_recovery_outcome is None and not plan.plan_success.any():
            # A pre-candidate failure (for example an explicit exhausted wall
            # budget) has no selection to commit and cannot authorize feedback.
            return
        self.commit_planning(plan)
        if bool(attempt.planned_mask.all()) and bool(plan.plan_success.all()):
            self.adopt_plan(plan)

    def on_execution_aborted(self, reason: str) -> None:
        """Invalidate this operation's adoption after the ordinary safe-stop path."""
        del reason
        self.close()

    def reset_ready(self) -> None:
        """Reset only at the host's measured ready boundary, never on every chunk."""
        self._open()
        self._cursor.reset_ready()
        self._adopted_outcome = None
        self._control_window = None
        self._physics_dt = None
        self._chunk_cursor = 0

    def close(self) -> None:
        """End this cursor scope without closing borrowed geometry or robot resources."""
        if not self._closed:
            self._cursor.close()
            self._adopted_outcome = None
            self._closed = True


class TwistOperationRegistry:
    """Own declared operation sessions and select them only by exact occurrence.

    The same registry implements the shared Twist planning provider and the
    Task Program per-call plan lifecycle factory. Neither object names nor task
    analysis indices select state. It closes cursor scopes, never borrowed native
    Robot/geometry resources. Preview forks replace every cursor identity.
    """

    def __init__(self, sessions: Sequence[TwistOperationSession]) -> None:
        sessions = tuple(sessions)
        if not sessions or any(
            type(session) is not TwistOperationSession for session in sessions
        ):
            raise ValueError("Twist registry requires explicit operation sessions.")
        first = sessions[0]
        owners, invocations = set(), {}
        for session in sessions:
            if (
                session.robot is not first.robot
                or session.scope.program_fingerprint != first.scope.program_fingerprint
                or session.scope.episode_id != first.scope.episode_id
            ):
                raise ValueError(
                    "Twist registry operations must share Robot/program/episode ownership."
                )
            if session.scope in owners:
                raise ValueError("Twist operation scope is declared more than once.")
            owners.add(session.scope)
            for invocation_id in session.invocation_ids:
                if invocation_id in invocations:
                    raise ValueError(
                        "Twist occurrence has more than one operation owner."
                    )
                invocations[invocation_id] = session
        self._sessions, self._invocations = sessions, invocations
        self._robot = first.robot
        self._closed = False

    @property
    def robot(self) -> Robot:
        """Return the one exact borrowed Robot served by these declarations."""
        return self._robot

    def _open(self) -> None:
        if self._closed:
            raise RuntimeError("Twist operation registry is closed.")

    def prepare(
        self, request: ResolvedActionRequest, context: PlanningContext
    ) -> TwistPlanningSession:
        """Resolve only the exact declared occurrence and delegate fresh preparation."""
        self._open()
        session = self._invocations.get(request.invocation_id)
        if session is None:
            raise ValueError(
                "Bounded Twist occurrence has no declared operation owner."
            )
        return session.prepare(request, context)

    def create_plan_lifecycle(
        self,
        request: TaskProgramPlanRequest,
        *,
        engine: AtomicActionEngine,
    ) -> TwistOperationSession | None:
        """Select the same declared owner before ordinary Atomic session execution."""
        self._open()
        invocation = request.invocation
        session = self._invocations.get(invocation.invocation_id)
        bounded = (
            type(invocation.recovery_policy.planning_recovery) is TwistPlanningRecovery
        )
        if session is None:
            if bounded:
                raise ValueError(
                    "Bounded Twist occurrence has no declared operation owner."
                )
            return None
        if (
            engine.robot is not self._robot
            or not bounded
            or invocation.skill_id != "twist"
        ):
            raise ValueError(
                "Twist lifecycle differs from its declared Robot and policy."
            )
        session.snapshot()  # Validate the host execution owner before returning it.
        return session

    def fork_preview(self) -> TwistOperationRegistry:
        """Fork cursor ownership for all operations without mutating live state."""
        self._open()
        return TwistOperationRegistry(
            tuple(session.fork_preview() for session in self._sessions)
        )

    def close(self) -> None:
        """Idempotently invalidate every owned operation cursor scope."""
        if not self._closed:
            for session in self._sessions:
                session.close()
            self._closed = True
