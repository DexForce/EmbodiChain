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

"""Explicit configured initial-call planning without simulator execution."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING
from collections.abc import Callable

from embodichain.lab.sim.atomic_actions.state import TaskState
from ..invocation_policy import BoundInvocationPolicies, task_program_fingerprint
from ..environment import TaskProgramEnvironmentAdapter

if TYPE_CHECKING:
    from embodichain.lab.gym.envs.base_env import BaseEnv
    from embodichain.lab.sim.atomic_actions.plans import ActionPlan
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.task_program.compiler import CompiledTaskProgram
    from ..composition import ConfiguredTaskProgramDeployment

__all__ = ["probe_configured_task_program_initial_plan"]


def _initial_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    result = dict(scope="initial_call", **evidence, task_success=None)
    result["final_command_velocity"] = dict(
        scope="initial_call_full_robot_post_resampling",
        **evidence["final_command_velocity"],
    )
    return result


def _evidence(plan: ActionPlan, robot: Robot) -> dict[str, Any]:
    from embodichain.lab.sim.atomic_actions.plan_evidence import (
        action_plan_proposal_evidence,
    )

    return _initial_evidence(action_plan_proposal_evidence(plan, robot))


def _execution_initial_plan_observer(
    program: CompiledTaskProgram,
    callback: Callable[[dict[str, object]], None],
) -> tuple[str, Callable[[dict[str, object]], None]]:
    if not callable(callback):
        raise TypeError("initial_plan_observer must be callable or None.")
    segments = tuple(program.iter_segments())
    analyses = program.preflight_analyses()
    if (
        not segments
        or not analyses
        or analyses[0].kind == "parallel_branch"
        or not segments[0].calls
    ):
        raise ValueError(
            "Initial plan observation requires a non-empty sequential first segment."
        )
    first = segments[0]
    invocation_id = (
        f"{program.program_id}/{first.segment_id}:{first.calls[0].segment_call_index}"
    )

    def observe(evidence: dict[str, object]) -> None:
        if evidence.get("invocation_id") != invocation_id:
            raise ValueError(
                "Initial plan observation belongs to another canonical occurrence."
            )
        values = _initial_evidence(dict(evidence))
        values.update(
            planning_source="execution", remaining_analysis_count=len(analyses) - 1
        )
        callback(values)

    return invocation_id, observe


def probe_configured_task_program_initial_plan(
    environment: BaseEnv,
    deployment: ConfiguredTaskProgramDeployment,
    *,
    compiled_program: CompiledTaskProgram | None = None,
) -> dict[str, Any]:
    """Plan only the first configured occurrence from one current observation.

    Args:
        environment: Initialized simulation environment with public RNG-preservation
            support, before selecting an execution bridge. An existing configured
            host adapter is borrowed after exact registration validation; its
            observers and lifetime remain owned by the environment.
        deployment: Publicly resolved configured Task Program deployment.
        compiled_program: Optional already compiled program from the same deployment.
            Its complete program fingerprint must match fresh provider-free decoding.

    Returns:
        Source-compatible initial-call proposal and final-velocity diagnostics as
        plain data. ``task_success`` remains None. No command is dispatched, effect
        committed, or environment reset/step invoked. Every owned bridge/adapter
        service closes, and process/environment RNG streams restore on all exits.
        Borrowed host adapters stay open; only adapters created by this call close.
        Use only for an explicit probe; normal execution does not call this entry.

    Raises:
        ValueError: For an empty/parallel initial analysis, changed program identity,
            configured nominal sources, or a successful plan lacking final trajectory
            evidence. Ordinary execution's nominal-source support is unchanged.
        TypeError: If the environment lacks its public random-state scope.
    """
    unwrapped = getattr(environment, "unwrapped", environment)
    preserve = getattr(unwrapped, "preserve_random_state", None)
    if not callable(preserve):
        raise TypeError(
            "Initial planning probe requires environment.preserve_random_state()."
        )
    if getattr(unwrapped, "active_task_program_bridge", None) is not None:
        raise ValueError(
            "Initial planning probe requires an unselected execution bridge."
        )
    with preserve():
        try:
            existing_adapter = getattr(unwrapped, "task_program_adapter", None)
        except NotImplementedError:
            existing_adapter = None
        owns_adapter = existing_adapter is None
        if owns_adapter:
            adapter = deployment.integration.adapter_factory.create_adapter(unwrapped)
        else:
            if not isinstance(existing_adapter, TaskProgramEnvironmentAdapter):
                raise TypeError("Configured probe host adapter has another type.")
            expected_registration = deployment.integration.registration
            actual_registration = existing_adapter._registration
            if (
                actual_registration is None
                or actual_registration.fingerprint_payload()
                != expected_registration.fingerprint_payload()
            ):
                raise ValueError(
                    "Configured probe host adapter differs from its deployment registration."
                )
            existing_adapter._require_open()
            adapter = existing_adapter
        bridge = None
        try:
            expected = adapter.compile(deployment.load_program())
            if compiled_program is not None and task_program_fingerprint(
                compiled_program
            ) != task_program_fingerprint(expected):
                raise ValueError(
                    "Initial probe compiled program differs from its deployment."
                )
            compiled = expected if compiled_program is None else compiled_program
            analyses = compiled.preflight_analyses()
            segments = tuple(compiled.iter_segments())
            if (
                not analyses
                or analyses[0].kind == "parallel_branch"
                or not segments
                or not segments[0].calls
            ):
                raise ValueError(
                    "Initial probe requires a non-empty sequential workflow."
                )
            bridge, assembly = adapter._create_bridge_with_assembly(
                compiled, initial_plan_probe=True
            )
            first = segments[0]
            declarations = (
                deployment.integration.registration.invocation_policy_bindings
            )
            policies = BoundInvocationPolicies(compiled, declarations)
            analysis = analyses[0]
            robot = assembly.engine.robot
            qpos = robot.get_qpos()
            context = assembly.observation_provider.observe(
                TaskState(batch_size=qpos.shape[0], device=qpos.device)
            )
            workflow = assembly.compiler.analyze(
                analysis.calls,
                workflow_id=f"{compiled.program_id}/{first.segment_id}",
                preset_ids=policies._presets_for_analysis(analysis),
            )
            grounded = assembly.compiler.ground(
                workflow,
                0,
                context,
                invocation_id=f"{compiled.program_id}/{first.segment_id}:{first.calls[0].segment_call_index}",
            )
            plan = assembly.engine.plan(grounded.invocation, context)
            return dict(
                **_evidence(plan, robot),
                planning_source="probe",
                call_index=0,
                analysis_call_count=len(workflow.calls),
                initial_downstream_target_count=len(
                    workflow.calls[0].downstream_object_targets
                ),
                unplanned_call_indices=list(range(1, len(workflow.calls))),
                remaining_analysis_count=len(analyses) - 1,
            )
        finally:
            try:
                if bridge is not None:
                    bridge.close_runtime_resources()
            finally:
                if owns_adapter:
                    adapter.close()
