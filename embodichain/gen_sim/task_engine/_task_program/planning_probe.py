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

"""Initial-plan evidence and velocity acceptance without a second execution plan."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from typing import Any

from embodichain.lab.sim.atomic_actions.plans import PlanningFailure

from .motion import _joint_velocity_limits, _velocity_diagnostics, _velocity_validity

__all__: list[str] = []


def _planned_grasp_candidates(plan: Any) -> list[dict[str, Any]]:
    """Expose proposals without claiming an acquired grasp."""
    effects = getattr(plan, "expected_effects", None)
    updates = getattr(effects, "held_object_updates", {})
    return [
        {
            "evidence_scope": "proposal_only_not_observed_attachment",
            "resource": resource,
            "object_id": held.semantics.entity_id,
            "object_to_eef": held.object_to_eef.detach().cpu().tolist(),
            "grasp_xpos": held.grasp_xpos.detach().cpu().tolist(),
        }
        for resource, held in updates.items()
        if held is not None
    ]


def _validate_final_plan_velocity(plan: Any, robot: Any) -> dict[str, Any]:
    """Validate final arm/hand samples, not only intermediate planner paths."""
    trajectory = plan.joint_trajectory
    if trajectory is None:
        if plan.plan_success.any():
            raise ValueError("Initial plan requires final joint trajectory evidence.")
        return {"valid_mask": plan.plan_success.clone(), "diagnostics": []}
    limits = _joint_velocity_limits(robot, None).to(trajectory.positions)
    return {
        "valid_mask": _velocity_validity(trajectory.positions, trajectory.dt, limits),
        "diagnostics": _velocity_diagnostics(
            trajectory.positions, trajectory.dt, limits
        ),
    }


def _plan_evidence(plan: Any, velocity: dict[str, Any]) -> dict[str, Any]:
    return {
        "scope": "initial_call",
        "planned_grasp_candidates": _planned_grasp_candidates(plan),
        "plan_success": (plan.plan_success & velocity["valid_mask"])
        .detach()
        .cpu()
        .tolist(),
        "final_command_velocity": {
            "scope": "initial_call_full_robot_post_resampling",
            "valid_mask": velocity["valid_mask"].detach().cpu().tolist(),
            "diagnostics": velocity["diagnostics"],
        },
        "task_success": None,
    }


class InitialPlanCapture:
    """Observe only the exact initial invocation; retain evidence, never a plan."""

    def __init__(
        self,
        program: Any,
        publish: Callable[[dict[str, Any]], None],
        describe_error: Callable[[BaseException], dict[str, Any]],
    ) -> None:
        segments = tuple(program.iter_segments())
        analyses = program.preflight_analyses()
        if (
            not segments
            or len(segments[0].calls) != 1
            or not analyses
            or analyses[0].kind == "parallel_branch"
        ):
            raise ValueError(
                "Initial capture requires one generated call in its first segment."
            )
        first = segments[0]
        self.invocation_id = f"{program.program_id}/{first.segment_id}:{first.calls[0].segment_call_index}"
        self._metadata = {
            "scope": "initial_call",
            "planning_source": "execution",
            "invocation_id": self.invocation_id,
            "call_index": 0,
            "analysis_call_count": len(analyses[0].calls),
            "unplanned_call_indices": list(range(1, len(analyses[0].calls))),
            "remaining_analysis_count": len(analyses) - 1,
            "task_success": None,
        }
        self._publish = publish
        self._describe_error = describe_error
        self.report: dict[str, Any] | None = None

    def _record(self, evidence: dict[str, Any]) -> None:
        if self.report is None:
            self.report = {**self._metadata, **evidence}
            self._publish(self.report)

    def failed(self, error: BaseException) -> None:
        """Preserve the planning exception even if the shared runtime catches it."""
        self._record({"plan_success": [], "failure": self._describe_error(error)})

    def accept(self, request: Any, plan: Any, robot: Any) -> Any:
        """Keep the former pre-execution velocity gate on the actual initial plan."""
        velocity = _validate_final_plan_velocity(plan, robot)
        evidence = _plan_evidence(plan, velocity)
        evidence["initial_downstream_target_count"] = len(
            getattr(request.skill_options, "downstream_object_target_poses", ())
        )
        self._record(evidence)
        accepted = plan.plan_success & velocity["valid_mask"]
        if accepted.all():
            return plan
        failure = plan.diagnostics.failure
        if failure is None:
            failure = PlanningFailure("initial_plan_velocity_limit", retryable=False)
        elif not accepted.any():
            # The old probe rejected all-failed initial plans before any execution.
            failure = replace(failure, retryable=False)
        return replace(
            plan,
            plan_success=accepted,
            diagnostics=replace(
                plan.diagnostics,
                failure=failure,
                metadata={
                    **plan.diagnostics.metadata,
                    "initial_plan_validation": evidence,
                },
            ),
        )


_INITIAL_PLAN_CAPTURE: ContextVar[InitialPlanCapture | None] = ContextVar(
    "gen_sim_initial_plan_capture", default=None
)


@contextmanager
def capture_initial_plan(
    program: Any,
    publish: Callable[[dict[str, Any]], None],
    describe_error: Callable[[BaseException], dict[str, Any]],
) -> Iterator[InitialPlanCapture]:
    """Scope evidence to one run and restore the observer on failure/cancellation."""
    capture = InitialPlanCapture(program, publish, describe_error)
    token = _INITIAL_PLAN_CAPTURE.set(capture)
    try:
        yield capture
    finally:
        _INITIAL_PLAN_CAPTURE.reset(token)


def initial_plan_capture(invocation_id: str) -> InitialPlanCapture | None:
    capture = _INITIAL_PLAN_CAPTURE.get()
    return (
        capture
        if capture is not None and capture.invocation_id == invocation_id
        else None
    )


def probe_initial_plan(env: Any, deployment: Any, program: Any) -> dict[str, Any]:
    """Explicit probe-only planning, without dispatch or effect commits."""
    from embodichain.lab.sim.atomic_actions.state import TaskState

    unwrapped = getattr(env, "unwrapped", env)
    adapter = deployment.integration.adapter_factory.create_adapter(unwrapped)
    compiled = adapter.compile(program)
    analyses = compiled.preflight_analyses()
    if not analyses or analyses[0].kind == "parallel_branch":
        raise ValueError("Initial probe requires a non-empty sequential workflow.")
    assembly = adapter.assemble_runtime(deployment.selection)
    qpos = unwrapped.robot.get_qpos()
    context = assembly.observation_provider.observe(
        TaskState(batch_size=qpos.shape[0], device=qpos.device)
    )
    first = next(compiled.iter_segments())
    workflow = assembly.compiler.analyze(
        analyses[0].calls, workflow_id=f"{compiled.program_id}/{first.segment_id}"
    )
    grounded = assembly.compiler.ground(workflow, 0, context)
    plan = assembly.engine.plan(grounded.invocation, context)
    return {
        **_plan_evidence(plan, _validate_final_plan_velocity(plan, unwrapped.robot)),
        "planning_source": "probe",
        "call_index": 0,
        "analysis_call_count": len(workflow.calls),
        "initial_downstream_target_count": len(
            workflow.calls[0].downstream_object_targets
        ),
        "unplanned_call_indices": list(range(1, len(workflow.calls))),
        "remaining_analysis_count": len(analyses) - 1,
    }
