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

"""Move the UR10 end effector through a small Cartesian loop using Task Program.

Run from the repository root::

    python scripts/tutorials/gym/simple_task_program.py --headless --seed 7

The scene contains only the robot. Four targets relative to the initial TCP move
its end effector around a 50 mm square while retaining its orientation. A trusted
tutorial-local Semantic Call lowers to the existing MoveEndEffector Atomic Skill;
the ordinary Gym bridge owns every simulation step. The UR10 keeps its normal
physics and drive defaults; the final measured return tolerance is 20 mm.
No official task is registered.
"""

from __future__ import annotations

import argparse
from typing import ClassVar

from embodichain.cli.sim import (
    add_seed_arg_to_parser,
    add_sim_args_to_parser,
    resolve_seed,
)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI options before loading optional simulation dependencies."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_sim_args_to_parser(parser)
    add_seed_arg_to_parser(parser, scope="task environment")
    return parser


if __name__ == "__main__":
    _cli_args = build_parser().parse_args()


import torch

from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.envs.demo import execute_demo_episode
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.atomic_skills import (
    SkillOptions,
    AtomicSkillEngine,
    EndEffectorPoseGoal,
    JointPositionTarget,
    MotionPolicy,
    MoveEndEffector,
    MoveEndEffectorOptions,
    PlanningContext,
    SkillDescriptor,
    TrackingPolicy,
)
from embodichain.lab.sim.cfg import RenderCfg, physics_cfg_for_backend
from embodichain.lab.sim.objects import Robot
from embodichain.lab.task_program import (
    InvokeCfg,
    RegisteredSemanticCallCfg,
    SegmentCfg,
    SequenceCfg,
    TaskProgramCfg,
    TaskProgramIntegrationCfg,
)
from embodichain.lab.task_program.compiler.lowering import (
    RegisteredSemanticLowerer,
    SemanticLowering,
)
from embodichain.lab.task_program.integrations import (
    ControlPartEndpointBinding,
    ControlPartResourceBinding,
    SimulationRobotSkillProfileBinding,
    SimulationSceneBinding,
    SimulationTaskProgramAdapterFactory,
    SimulationTaskProgramRegistration,
)
from embodichain.lab.task_program.semantics import (
    BoundSemanticCall,
    EffectAssurance,
    RegisteredSemanticCall,
    SceneRegistry,
    SemanticCallCatalog,
    SemanticCallDescriptor,
    SkillPolicyPreset,
)
from embodichain.lab.visualization import visualization_cfg_from_args
from embodichain_robots import URRobotCfg

_CALL_ID = "tutorial.move_end_effector"
_RETURN_TOLERANCE = 0.02  # Metres; this tutorial does not tune the arm's controller.


class _MoveEndEffectorLowerer(RegisteredSemanticLowerer):
    """Translate a declared world offset to a pose goal using the bound robot's FK."""

    call_id: ClassVar[str] = _CALL_ID
    target_descriptor: ClassVar[SkillDescriptor] = MoveEndEffector.descriptor()

    def __init__(self, robot: Robot) -> None:
        self._robot = robot
        self._origin: torch.Tensor | None = None

    def lower(
        self,
        call: RegisteredSemanticCall,
        *,
        context: PlanningContext,
        bound: BoundSemanticCall,
        option_template: SkillOptions,
    ) -> SemanticLowering:
        """Resolve a relative Cartesian target without stepping the environment."""
        if set(call.arguments) != {"position_offset"}:
            raise ValueError("MoveEndEffector requires only position_offset.")
        offset = call.arguments["position_offset"]
        if (
            type(offset) is not tuple
            or len(offset) != 3
            or any(type(value) not in (int, float) for value in offset)
        ):
            raise ValueError("position_offset must contain three finite numbers.")
        if type(option_template) is not MoveEndEffectorOptions:
            raise TypeError("MoveEndEffector requires its typed action options.")
        target = bound.binding.skill_binding.endpoint(
            "primary", "motion"
        ).require_target(JointPositionTarget)
        if self._origin is None:
            self._origin = self._robot.compute_fk(
                qpos=context.robot.qpos[:, list(target.joint_ids)],
                name=target.control_part,
                to_matrix=True,
            ).clone()
        # Anchor every waypoint to the same observation so tracking errors do
        # not accumulate when the final waypoint returns to the initial pose.
        pose = self._origin.clone()
        delta = pose.new_tensor(offset)
        if not torch.isfinite(delta).all():
            raise ValueError("position_offset must contain three finite numbers.")
        pose[:, :3, 3] += delta
        return SemanticLowering(goal=EndEffectorPoseGoal(pose))


class _MoveEndEffectorFactory:
    """Declare a trusted extension and create a fresh lowerer for each Gym adapter."""

    call_id: ClassVar[str] = _CALL_ID
    revision: ClassVar[str] = "1"
    target_descriptor: ClassVar[SkillDescriptor] = MoveEndEffector.descriptor()

    def create(
        self,
        *,
        simulation: object,
        robot: object,
        scene_registry: SceneRegistry,
        engine: AtomicSkillEngine,
    ) -> RegisteredSemanticLowerer:
        """Create episode-local motion grounding for the adapter's exact robot."""
        del simulation, scene_registry
        if not isinstance(robot, Robot) or engine.robot is not robot:
            raise ValueError("The lowerer must use the adapter's exact robot.")
        return _MoveEndEffectorLowerer(robot)


def build_registration() -> SimulationTaskProgramRegistration:
    """Bind the free-space motion call to the UR10 arm without scene objects."""
    profile = SimulationRobotSkillProfileBinding(
        profile_id="tutorial_ur10",
        resources=(
            ControlPartResourceBinding(
                resource_id="manipulator",
                endpoints=(
                    ControlPartEndpointBinding(
                        endpoint_id="motion",
                        control_part="arm",
                        capabilities=frozenset({"motion.cartesian_pose"}),
                    ),
                ),
            ),
        ),
        presets=(
            SkillPolicyPreset(
                "trajectory",
                skill_option_templates={_CALL_ID: MoveEndEffectorOptions()},
                effect_assurance=EffectAssurance.PROJECTED,
                motion_policy=MotionPolicy(sample_count=80),
                tracking_policy=TrackingPolicy.joint_position(
                    in_flight_max_abs_error=0.1,
                    terminal_max_abs_error=0.01,
                    terminal_settle_timeout=2.0,
                ),
            ),
        ),
        default_preset="trajectory",
    )
    return SimulationTaskProgramRegistration(
        scene_binding=SimulationSceneBinding(registry_id="tutorial_empty_scene"),
        robot_profile_binding=profile,
        call_catalog=SemanticCallCatalog(
            (
                SemanticCallDescriptor(
                    call_id=_CALL_ID,
                    spec_type=RegisteredSemanticCall,
                    target_descriptor=MoveEndEffector.descriptor(),
                ),
            )
        ),
        registered_semantic_lowerer_factories=(_MoveEndEffectorFactory(),),
    )


def build_program() -> TaskProgramCfg:
    """Compose four Cartesian moves that return the end effector to its start."""
    offsets = (
        ("move_x", (0.05, 0.0, 0.0)),
        ("move_y", (0.05, 0.05, 0.0)),
        ("return_x", (0.0, 0.05, 0.0)),
        ("return_y", (0.0, 0.0, 0.0)),
    )
    return TaskProgramCfg(
        program_id="simple_end_effector_motion",
        integration=TaskProgramIntegrationCfg(
            robot_profile="tutorial_ur10",
            scene_registry="tutorial_empty_scene",
            runtime_preset="trajectory",
        ),
        instruction="Move the end effector around a small square and return.",
        targets={},
        program=SequenceCfg(
            items=tuple(
                SegmentCfg(
                    name=name,
                    instruction=f"Move to world-axis offset {offset} metres from the initial TCP.",
                    steps=InvokeCfg(
                        call=RegisteredSemanticCallCfg(
                            call_id=_CALL_ID,
                            arguments={"position_offset": offset},
                            resources={"primary": "manipulator"},
                        )
                    ),
                )
                for name, offset in offsets
            )
        ),
    )


def main(args: argparse.Namespace) -> None:
    """Run the program through Gym and measure the final end-effector return error.

    Args:
        args: Simulation and seed options returned by the tutorial parser.
    """
    registration = build_registration()
    program = build_program()
    registration.catalog.preflight(program)
    cfg = EmbodiedEnvCfg(
        seed=resolve_seed(args.seed),
        num_envs=args.num_envs,
        max_episode_steps=600,
        sim_cfg=SimulationManagerCfg(
            num_envs=args.num_envs,
            headless=args.headless,
            device=args.device,
            arena_space=args.arena_space,
            gpu_id=args.gpu_id,
            physics_cfg=physics_cfg_for_backend(args.physics),
            render_cfg=RenderCfg(renderer=args.renderer),
            visualization=visualization_cfg_from_args(args),
        ),
        robot=URRobotCfg.from_dict(
            {
                "robot_type": "ur10",
                "uid": "UR10",
                "init_pos": [0.0, 0.0, 0.7775],
                "init_qpos": [1.57079, -1.57079, 1.57079, -1.57079, -1.57079, -3.14159],
            }
        ),
        enable_sensor=False,
        init_rollout_buffer=False,
        filter_dataset_saving=True,
        task_program=program,
    )
    env = EmbodiedEnv(
        cfg,
        task_program_adapter_factory=SimulationTaskProgramAdapterFactory(registration),
    )
    try:
        env.reset()
        arm_ids = env.robot.get_joint_ids(name="arm")
        start = env.robot.compute_fk(
            qpos=env.robot.get_qpos()[:, arm_ids], name="arm", to_matrix=True
        ).clone()
        result = execute_demo_episode(env)
        final = env.robot.compute_fk(
            qpos=env.robot.get_qpos()[:, arm_ids], name="arm", to_matrix=True
        )
        error = torch.linalg.vector_norm(final[:, :3, 3] - start[:, :3, 3], dim=-1)
        print(
            f"Completed: {result.completed}; accepted: {result.all_success}; "
            f"return error (m): {error.tolist()}",
            flush=True,
        )
        if (
            not result.completed
            or not result.all_success
            or not bool((error < _RETURN_TOLERANCE).all())
        ):
            raise RuntimeError("The end-effector motion did not complete its loop.")
    finally:
        env.close(exit_process=False)
        SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    main(_cli_args)
