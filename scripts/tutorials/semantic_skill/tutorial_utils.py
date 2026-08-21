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

"""Shared construction helpers for semantic-skill tutorials."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

import torch

from embodichain.lab.sim import SimulationManager
from embodichain.lab.sim.atomic_actions import (
    AntipodalAffordance,
    BATCH_INVERSE_KINEMATICS_CAPABILITY,
    CARTESIAN_POSE_CAPABILITY,
    ExecutionEventKind,
    FORWARD_KINEMATICS_CAPABILITY,
    GRASP_CAPABILITY,
    RunnerStep,
    RunnerStepCallback,
    TimedTrajectory,
)
from embodichain.lab.sim.objects import RigidObject, Robot
from embodichain.lab.sim.skills import (
    GRASP_AFFORDANCE_CAPABILITY,
    ControlPartEndpoint,
    RobotResource,
    SceneAffordanceRef,
    SceneEntityRegistration,
    SceneObjectRef,
    SceneRegistry,
    SemanticSkillRuntime,
)
from embodichain.lab.sim.skills.calls import SemanticCallSpec
from embodichain.utils import logger

_MOTION_CAPABILITIES = frozenset(
    {
        BATCH_INVERSE_KINEMATICS_CAPABILITY,
        CARTESIAN_POSE_CAPABILITY,
        FORWARD_KINEMATICS_CAPABILITY,
    }
)


def create_graspable_object_registry(
    simulation: SimulationManager,
    *,
    object_id: str,
    simulation_uid: str,
    semantic_type: str,
    affordance: AntipodalAffordance,
) -> tuple[SceneRegistry, SceneObjectRef]:
    """Register one live object and its default antipodal grasp affordance.

    Args:
        simulation: Simulation containing the selected rigid object.
        object_id: Canonical semantic object identifier.
        simulation_uid: Backend-local rigid-object identifier.
        semantic_type: Human-readable object category.
        affordance: Target-local grasp metadata copied into the registry.

    Returns:
        The immutable registry and its canonical object reference.
    """
    object_ref = SceneObjectRef(object_id)
    grasp_ref = SceneAffordanceRef(f"{object_id}.grasp.antipodal")
    simulation_registry = SceneRegistry.from_simulation(
        simulation,
        rigid_objects={object_id: simulation_uid},
    )
    object_registration = replace(
        simulation_registry.lookup(object_ref, expected_type=SceneObjectRef),
        semantic_type=semantic_type,
        default_affordances={GRASP_AFFORDANCE_CAPABILITY: grasp_ref},
    )
    registry = SceneRegistry(
        (
            object_registration,
            SceneEntityRegistration(
                ref=grasp_ref,
                parent=object_ref,
                native_name="antipodal_grasp",
                affordance=affordance,
                affordance_capabilities=frozenset({GRASP_AFFORDANCE_CAPABILITY}),
                affordance_revision="antipodal-v1",
                relative_pose=torch.eye(4, dtype=torch.float32),
            ),
        )
    )
    return registry, object_ref


def create_manipulator_resource(
    resource_id: str,
    *,
    motion_control_part: str,
    grasp_control_part: str,
) -> RobotResource:
    """Declare one arm-and-gripper resource for semantic skill binding.

    Args:
        resource_id: Stable embodiment-level resource identifier.
        motion_control_part: Robot control part used for Cartesian motion.
        grasp_control_part: Robot control part used for open/grasp commands.

    Returns:
        A resource satisfying the built-in manipulation-skill contracts.
    """
    return RobotResource(
        resource_id=resource_id,
        endpoints={
            "motion": ControlPartEndpoint(
                control_part=motion_control_part,
                capabilities=_MOTION_CAPABILITIES,
            ),
            "grasp": ControlPartEndpoint(
                control_part=grasp_control_part,
                capabilities=frozenset({GRASP_CAPABILITY}),
            ),
        },
    )


def compile_semantic_workflow_for_diagnostics(
    runtime: SemanticSkillRuntime,
    calls: Iterable[SemanticCallSpec],
    *,
    workflow_id: str,
) -> tuple[TimedTrajectory, tuple[str, ...]]:
    """Compile a semantic workflow without physically executing it.

    This helper exists only for the tutorials' ``--diagnose-plan`` path.
    Expected effects are projected hypothetically between calls; normal runs
    must use :class:`SemanticSkillRuntime` and verify physical effects.

    Args:
        runtime: Fully bound semantic runtime.
        calls: Ordered semantic calls to analyze and compile.
        workflow_id: Stable workflow identifier used in diagnostics.

    Returns:
        Concatenated diagnostic trajectory and lowered atomic skill IDs.

    Raises:
        RuntimeError: If any call fails to plan for any environment.
    """
    workflow = runtime.validate(tuple(calls), workflow_id=workflow_id)
    empty_task_state = runtime.engine.initial_context().task
    context = runtime.observation_provider.observe(empty_task_state)
    trajectories: list[TimedTrajectory] = []
    skill_ids: list[str] = []
    for call_index in range(len(workflow.calls)):
        grounded = runtime.compiler.ground(workflow, call_index, context)
        compiled = runtime.engine.compile((grounded.invocation,), context)
        if not compiled.plan_success.all():
            failed_rows = (
                (~compiled.plan_success)
                .nonzero(as_tuple=False)
                .flatten()
                .detach()
                .cpu()
                .tolist()
            )
            raise RuntimeError(
                f"Semantic call {call_index} ({grounded.invocation.skill_id!r}) "
                f"failed to plan for environment rows {failed_rows}."
            )
        trajectories.append(compiled.trajectory)
        skill_ids.append(grounded.invocation.skill_id)
        context = compiled.projected_context
    return TimedTrajectory.concatenate(trajectories), tuple(skill_ids)


def object_to_eef_translation_error(
    obj: RigidObject,
    robot: Robot,
    *,
    motion_control_part: str,
    expected_object_to_eef: torch.Tensor,
) -> torch.Tensor:
    """Compare the observed and expected object-to-EEF translations.

    Args:
        obj: Live object whose relative transform is measured.
        robot: Robot providing current joint state and forward kinematics.
        motion_control_part: Control part identifying the target end effector.
        expected_object_to_eef: Relation declared by the pending symbolic effect.

    Returns:
        Translation error in metres for every environment.
    """
    object_pose = obj.get_local_pose(to_matrix=True)
    eef_pose = robot.compute_fk(
        qpos=robot.get_qpos(name=motion_control_part),
        name=motion_control_part,
        to_matrix=True,
    )
    if (
        not isinstance(object_pose, torch.Tensor)
        or not isinstance(eef_pose, torch.Tensor)
        or object_pose.dim() != 3
        or eef_pose.shape != object_pose.shape
        or object_pose.shape[-2:] != (4, 4)
    ):
        raise ValueError("Object and end-effector poses must share shape (B, 4, 4).")
    expected = torch.as_tensor(
        expected_object_to_eef,
        dtype=object_pose.dtype,
        device=object_pose.device,
    )
    if expected.shape == (4, 4):
        expected = expected.unsqueeze(0).expand(object_pose.shape[0], -1, -1)
    if expected.shape != object_pose.shape:
        raise ValueError(
            "Expected object-to-EEF pose must have shape (4, 4) or (B, 4, 4)."
        )
    observed = torch.bmm(torch.linalg.inv(object_pose), eef_pose)
    return torch.linalg.vector_norm(
        observed[:, :3, 3] - expected[:, :3, 3],
        dim=1,
    )


def joint_target_error(
    robot: Robot,
    *,
    control_part: str,
    target: torch.Tensor,
) -> torch.Tensor:
    """Measure maximum absolute joint error per environment.

    Args:
        robot: Robot providing current control-part positions.
        control_part: Control part whose joints are compared.
        target: One-dimensional target or a full ``(B, D)`` batch.

    Returns:
        Maximum absolute joint error for every environment.
    """
    current = robot.get_qpos(name=control_part)
    if not isinstance(current, torch.Tensor) or current.dim() != 2:
        raise ValueError("Control-part qpos must have shape (B, D).")
    expected = torch.as_tensor(target, dtype=current.dtype, device=current.device)
    if expected.dim() == 1:
        expected = expected.unsqueeze(0).expand(current.shape[0], -1)
    if expected.shape != current.shape:
        raise ValueError("Joint target must have shape (D,) or match qpos (B, D).")
    return torch.amax(torch.abs(current - expected), dim=1)


def create_runtime_step_observer(
    obj: RigidObject,
    robot: Robot,
    *,
    grasp_control_part: str,
    grasp_target: torch.Tensor,
    grasp_tolerance: float = 1.0e-2,
) -> RunnerStepCallback:
    """Create a runner observer that logs recovery and stabilizes one grasp.

    Args:
        obj: Physical object stabilized once the grasp target is reached.
        robot: Robot whose gripper state is observed.
        grasp_control_part: Control part executing the initial grasp.
        grasp_target: Joint target representing a closed grasp.
        grasp_tolerance: Maximum joint error before dynamics are cleared once.

    Returns:
        Callback accepted by ``SemanticSkillRuntime.run(on_step=...)``.
    """
    if grasp_tolerance <= 0.0:
        raise ValueError("grasp_tolerance must be greater than zero.")
    target = grasp_target.clone()
    dynamics_cleared = False
    reported_events = {
        ExecutionEventKind.REPLANNED,
        ExecutionEventKind.TRACKING_ERROR,
        ExecutionEventKind.DYNAMIC_GOAL_CHANGED,
        ExecutionEventKind.COLLISION_WORLD_CHANGED,
        ExecutionEventKind.ACTION_RETRY,
        ExecutionEventKind.RECOVERY_EXHAUSTED,
    }

    def observe(step: RunnerStep) -> None:
        nonlocal dynamics_cleared
        if step.tick is not None:
            for event in step.tick.events:
                if event.kind in reported_events:
                    env_rows = event.env_mask.nonzero(as_tuple=False).flatten().tolist()
                    logger.log_info(
                        f"Runtime event {event.kind.value}: rows={env_rows}; "
                        f"{event.message}"
                    )
        if dynamics_cleared:
            return
        error = joint_target_error(
            robot,
            control_part=grasp_control_part,
            target=target,
        )
        if torch.all(error <= grasp_tolerance):
            obj.clear_dynamics()
            dynamics_cleared = True

    return observe


__all__ = [
    "compile_semantic_workflow_for_diagnostics",
    "create_graspable_object_registry",
    "create_manipulator_resource",
    "create_runtime_step_observer",
    "joint_target_error",
    "object_to_eef_translation_error",
]
