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

"""Real semantic-runtime recovery gate for a dynamic cuRobo collision world."""

from __future__ import annotations

from typing import ClassVar

import pytest
import torch

# Module-level guards must precede cuRobo-only imports.
pytest.importorskip("curobo")
if not torch.cuda.is_available():
    pytest.skip("cuRobo V2 requires CUDA", allow_module_level=True)

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg  # noqa: E402
from embodichain.lab.sim.atomic_actions import (  # noqa: E402
    CARTESIAN_POSE_CAPABILITY,
    AtomicActionEngine,
    CommandAcknowledgement,
    DynamicCollisionMode,
    EndEffectorPoseGoal,
    ExecutionEventKind,
    ExecutionRunnerCfg,
    MotionPolicy,
    MoveEndEffector,
    MoveEndEffectorOptions,
    PlanningContext,
    RecoveryPolicy,
    RuntimeCommandFrame,
    RuntimeEndpointTarget,
    SimulationExecutionAdapter,
    SkillDescriptor,
)
from embodichain.lab.sim.atomic_actions.tracking import TrackingPolicy  # noqa: E402
from embodichain.lab.sim.cfg import RigidBodyPhysicsCfg  # noqa: E402
from embodichain.lab.sim.objects import RigidObjectCfg  # noqa: E402
from embodichain.lab.sim.planners import MotionGenCfg, MotionGenerator  # noqa: E402
from embodichain.lab.sim.planners.curobo.curobo_planner import (  # noqa: E402
    CuroboAutoGenCfg,
    CuroboPlannerCfg,
    CuroboWorldCfg,
)
from embodichain.lab.sim.robots import FrankaPandaCfg  # noqa: E402
from embodichain.lab.sim.shapes import CubeCfg  # noqa: E402
from embodichain.lab.task_program.semantics import (  # noqa: E402
    BoundSemanticCall,
    ControlPartEndpoint,
    EffectAssurance,
    EffectEvidenceCollector,
    EffectEvidenceProviderRegistry,
    RegisteredSemanticCall,
    RobotResource,
    RobotSkillProfile,
    SceneCollisionRole,
    SceneCollisionWorldMode,
    SceneManifest,
    SceneRegistry,
    SemanticCallDescriptor,
    SemanticIntegrationManifest,
    SkillPolicyPreset,
    builtin_semantic_call_catalog,
)
from embodichain.lab.task_program.compiler.lowering import (  # noqa: E402
    RegisteredSemanticLowerer,
    SemanticCallCompiler,
    SemanticLowering,
)
from embodichain.lab.task_program.runtime.executor import (  # noqa: E402
    SemanticCallExecutor,
)
from embodichain.lab.task_program.runtime.results import (  # noqa: E402
    SemanticExecutionStatus,
)

pytestmark = [
    pytest.mark.requires_sim,
    pytest.mark.gpu,
    pytest.mark.slow,
]

ROBOT_UID = "semantic_dynamic_scene_franka"
OBSTACLE_UID = "semantic_dynamic_obstacle"
CONTROL_PART = "arm"
CALL_ID = "test.move_end_effector"
SAMPLE_COUNT = 80
COMMAND_CYCLE_TIME = 0.1
MOVE_AFTER_COMMAND = 12
OBSTACLE_SIZE = [0.10, 0.10, 0.12]
OBSTACLE_START_POSITION = [0.59, -0.20, 0.455]
MAXIMUM_FINAL_EEF_ERROR = 0.04

_MOVE_TARGET = MoveEndEffector.descriptor()
assert _MOVE_TARGET.binding_contract is not None


class _MoveEndEffectorLowerer(RegisteredSemanticLowerer):
    """Lower a declarative matrix into the built-in Cartesian motion goal."""

    call_id: ClassVar[str] = CALL_ID
    target_descriptor: ClassVar[SkillDescriptor] = _MOVE_TARGET

    def lower(
        self,
        call: RegisteredSemanticCall,
        *,
        context: PlanningContext,
        bound: BoundSemanticCall,
        option_template: MoveEndEffectorOptions,
    ) -> SemanticLowering:
        del bound, option_template
        values = call.arguments.get("xpos")
        if type(values) is not tuple or len(values) != 16:
            raise ValueError("xpos must contain one flattened 4x4 pose matrix.")
        pose = torch.tensor(
            values,
            dtype=context.robot.qpos.dtype,
            device=context.robot.qpos.device,
        ).reshape(4, 4)
        return SemanticLowering(goal=EndEffectorPoseGoal(xpos=pose))


class _CountingCommandSink:
    """Count accepted real-simulation command frames while delegating transport."""

    def __init__(self, delegate: SimulationExecutionAdapter) -> None:
        self.delegate = delegate
        self.command_count = 0

    def send(
        self,
        command: RuntimeCommandFrame,
        *,
        timeout: float,
    ) -> CommandAcknowledgement:
        acknowledgement = self.delegate.send(command, timeout=timeout)
        if acknowledgement.accepted:
            self.command_count += 1
        return acknowledgement

    def hold(
        self,
        targets: tuple[RuntimeEndpointTarget, ...],
        context: PlanningContext,
        *,
        timeout: float,
    ) -> CommandAcknowledgement:
        return self.delegate.hold(targets, context, timeout=timeout)

    def cancel(
        self,
        targets: tuple[RuntimeEndpointTarget, ...],
        *,
        timeout: float,
    ) -> CommandAcknowledgement:
        return self.delegate.cancel(targets, timeout=timeout)


def _profile() -> RobotSkillProfile:
    """Declare the exact robot resource and bounded safe recovery policy."""
    return RobotSkillProfile(
        profile_id="semantic_dynamic_scene_franka",
        resources={
            "manipulator": RobotResource(
                resource_id="manipulator",
                endpoints={
                    "motion": ControlPartEndpoint(
                        control_part=CONTROL_PART,
                        capabilities=frozenset({CARTESIAN_POSE_CAPABILITY}),
                    )
                },
            )
        },
        presets={
            "safe": SkillPolicyPreset(
                "safe",
                effect_assurance=EffectAssurance.PROJECTED,
                action_option_templates={
                    CALL_ID: MoveEndEffectorOptions(),
                },
                motion_policy=MotionPolicy(
                    strategy="motion_gen",
                    sample_count=SAMPLE_COUNT,
                ),
                tracking_policy=TrackingPolicy.joint_position(
                    in_flight_max_abs_error=0.1,
                    terminal_max_abs_error=0.1,
                ),
                recovery_policy=RecoveryPolicy(
                    max_replans=2,
                    action_timeout=30.0,
                ),
                runner_cfg=ExecutionRunnerCfg(minimum_cycle_time=COMMAND_CYCLE_TIME),
            )
        },
        default_preset="safe",
    )


def _compiler(
    registry: SceneRegistry,
    engine: AtomicActionEngine,
) -> SemanticCallCompiler:
    """Bind the test semantic extension to the real engine and scene registry."""
    catalog = builtin_semantic_call_catalog().with_descriptor(
        SemanticCallDescriptor(
            call_id=CALL_ID,
            spec_type=RegisteredSemanticCall,
            target_descriptor=_MOVE_TARGET,
        )
    )
    integration = SemanticIntegrationManifest(
        scene=SceneManifest.from_registry(registry),
        robot_profile=_profile(),
        call_catalog=catalog,
    ).bind(registry, engine)
    return SemanticCallCompiler(
        integration,
        registered_lowerers=(_MoveEndEffectorLowerer(),),
    )


def test_semantic_runtime_replans_after_dynamic_curobo_world_change() -> None:
    """Run semantic lowering, real cuRobo planning, world update, and recovery."""
    sim = SimulationManager(
        SimulationManagerCfg(headless=True, sim_device="cuda", num_envs=1)
    )
    planner = None
    try:
        robot = sim.add_robot(
            cfg=FrankaPandaCfg.from_dict({"uid": ROBOT_UID, "robot_type": "panda"})
        )
        obstacle = sim.add_rigid_object(
            cfg=RigidObjectCfg(
                uid=OBSTACLE_UID,
                shape=CubeCfg(size=OBSTACLE_SIZE),
                attrs=RigidBodyPhysicsCfg(),
                body_type="kinematic",
                init_pos=OBSTACLE_START_POSITION,
                init_rot=[0.0, 0.0, 0.0],
            )
        )
        sim.update(step=10)

        motion_generator = MotionGenerator(
            MotionGenCfg(
                planner_cfg=CuroboPlannerCfg(
                    robot_uid=ROBOT_UID,
                    auto_gen=CuroboAutoGenCfg(
                        fit_type="morphit",
                        sphere_density=0.3,
                        collision_sphere_buffer=0.005,
                    ),
                    world=CuroboWorldCfg(
                        rigid_objects=[obstacle],
                        obstacle_representation="cuboid",
                        dynamic_obstacle_names=[OBSTACLE_UID],
                        multi_env=False,
                    ),
                    warmup_iterations=0,
                )
            )
        )
        planner = motion_generator.planner
        registry = SceneRegistry.from_simulation(
            sim,
            rigid_objects={OBSTACLE_UID: OBSTACLE_UID},
            collision_roles={OBSTACLE_UID: SceneCollisionRole.DYNAMIC},
            collision_world_mode=SceneCollisionWorldMode.SHARED,
        )
        scene_provider = registry.make_planning_scene_provider(
            motion_generator,
            batch_size=1,
        )
        adapter = SimulationExecutionAdapter(
            sim,
            robot,
            control_dt=COMMAND_CYCLE_TIME,
            scene_provider=scene_provider,
        )
        sink = _CountingCommandSink(adapter)
        engine = AtomicActionEngine(motion_generator)
        runtime = SemanticCallExecutor(
            _compiler(registry, engine),
            adapter,
            sink,
            EffectEvidenceCollector(EffectEvidenceProviderRegistry()),
            clock=adapter,
        )

        start_pose = robot.compute_fk(
            qpos=robot.get_qpos(name=CONTROL_PART),
            name=CONTROL_PART,
            to_matrix=True,
        )
        target_pose = start_pose.clone()
        target_pose[:, :3, 3] += torch.tensor(
            [0.22, 0.24, 0.12],
            dtype=target_pose.dtype,
            device=target_pose.device,
        )
        call = RegisteredSemanticCall(
            call_id=CALL_ID,
            arguments={
                "xpos": tuple(
                    float(value)
                    for value in target_pose[0].detach().cpu().reshape(-1).tolist()
                )
            },
            resources={"primary": "manipulator"},
        )

        result = runtime.start(call, workflow_id="dynamic_curobo_recovery")
        assert result.status is SemanticExecutionStatus.RUNNING
        obstacle_moved = False
        for _ in range(2_000):
            if result.terminal:
                break
            if result.wait_duration > 0.0:
                adapter.sleep(result.wait_duration)
            result = runtime.step()
            if not obstacle_moved and sink.command_count >= MOVE_AFTER_COMMAND:
                blocking_pose = obstacle.get_local_pose(to_matrix=True).clone()
                blocking_pose[:, :3, 3] = 0.5 * (
                    start_pose[:, :3, 3] + target_pose[:, :3, 3]
                )
                obstacle.set_local_pose(blocking_pose)
                adapter.sleep(adapter.physics_dt)
                obstacle_moved = True

        assert obstacle_moved
        assert result.status is SemanticExecutionStatus.COMPLETED, result.message
        assert result.success_mask.tolist() == [True]
        assert len(result.calls) == 1
        trace = result.calls[0]
        assert trace.semantic_id == CALL_ID
        assert trace.skill_id == MoveEndEffector.skill_id
        assert len(trace.plan_attempts) >= 2

        event_kinds = tuple(event.kind for event in result.events)
        assert ExecutionEventKind.COLLISION_WORLD_CHANGED in event_kinds
        assert ExecutionEventKind.REPLANNED in event_kinds

        initial_attempt = trace.plan_attempts[0]
        changed_attempts = tuple(
            attempt
            for attempt in trace.plan_attempts[1:]
            if attempt.planned_collision_world_revision[0]
            > initial_attempt.planned_collision_world_revision[0]
        )
        assert changed_attempts
        assert changed_attempts[0].trigger == ExecutionEventKind.REPLANNED.value
        assert changed_attempts[0].planner_backend == "curobo"
        assert initial_attempt.collision_world_sensitive
        assert (
            initial_attempt.resolved_core_policy.motion_policy.dynamic_collision_mode
            is DynamicCollisionMode.REQUIRED
        )

        final_pose = robot.compute_fk(
            qpos=robot.get_qpos(name=CONTROL_PART),
            name=CONTROL_PART,
            to_matrix=True,
        )
        final_error = torch.linalg.vector_norm(
            final_pose[:, :3, 3] - target_pose[:, :3, 3],
            dim=1,
        )
        assert bool((final_error < MAXIMUM_FINAL_EEF_ERROR).all().item())
    finally:
        if planner is not None:
            planner.close()
        sim.destroy()
        SimulationManager.flush_cleanup_queue()
