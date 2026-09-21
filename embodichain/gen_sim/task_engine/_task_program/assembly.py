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

"""Task-owned composition around the unchanged simulation runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from embodichain.lab.task_program.integrations.environment import (
    TaskProgramEnvironmentAdapter,
)
from embodichain.lab.task_program.integrations.simulation.environment import (
    SimulationTaskProgramFactory,
)
from embodichain.utils.utility import load_config

from ..contracts import canonical_hash
from .stability import StabilityConstraint, TaskStabilityPort
from .configured import compose_deployment
from .grasp_filter import (
    GRASP_FILTER_REVISION,
    geometry_key,
    install_e6_approach_filters,
    install_grasp_filters,
)
from .motion import (
    MOTION_VALIDATION_REVISION,
    _joint_velocity_limits,
    _velocity_diagnostics,
    _velocity_validity,
)

__all__: list[str] = []

ADAPTER_CONTRACT = "gen_sim.task_program/2620929c/v3"


def probe_initial_plan(env: Any, deployment: Any, program: Any) -> dict[str, Any]:
    """Plan the initial call without dispatching commands or committing effects."""
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
    workflow = assembly.compiler.analyze(analyses[0].calls)
    grounded = assembly.compiler.ground(workflow, 0, context)
    plan = assembly.engine.plan(grounded.invocation, context)
    final_velocity = _validate_final_plan_velocity(plan, unwrapped.robot)
    return {
        "scope": "initial_call",
        "call_index": 0,
        "analysis_call_count": len(workflow.calls),
        "initial_downstream_target_count": len(
            workflow.calls[0].downstream_object_targets
        ),
        "unplanned_call_indices": list(range(1, len(workflow.calls))),
        "remaining_analysis_count": len(analyses) - 1,
        "planned_grasp_candidates": _planned_grasp_candidates(plan),
        "plan_success": (plan.plan_success & final_velocity["valid_mask"])
        .detach()
        .cpu()
        .tolist(),
        "final_command_velocity": {
            "scope": "initial_call_full_robot_post_resampling",
            "valid_mask": final_velocity["valid_mask"].detach().cpu().tolist(),
            "diagnostics": final_velocity["diagnostics"],
        },
        "task_success": None,
    }


def _planned_grasp_candidates(plan: Any) -> list[dict[str, Any]]:
    """Expose immutable planning proposals without claiming an acquired grasp."""
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
    """Validate the final arm/hand samples, not only the planner's intermediate path."""
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


class _TaskFactory(SimulationTaskProgramFactory):
    """Select task-owned observation services, not a different executor."""

    def __init__(
        self,
        *args: Any,
        constraints: dict[str, StabilityConstraint],
        articulation_bindings: tuple = (),
        drawer_routes: tuple = (),
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._task_post_port = TaskStabilityPort(
            self.segment_policy_port,
            self._simulation,
            self._robot,
            self.task_program_registration.scene_binding,
            constraints,
            step_dt=self.step_dt,
        )
        if articulation_bindings:
            from .articulation_slide import (
                ArticulationStabilityPort,
                synchronize_joint_limits,
            )

            for binding in articulation_bindings:
                synchronize_joint_limits(
                    binding, self._simulation.get_articulation(binding.object_id)
                )
            self._task_post_port = ArticulationStabilityPort(
                self._task_post_port,
                self._simulation,
                self._robot,
                articulation_bindings,
                self.step_dt,
            )
        self._drawers = ()
        if drawer_routes:
            from .drawer_runtime import DrawerObservation

            self._drawers = tuple(
                DrawerObservation(route, self._simulation) for route in drawer_routes
            )

    def create_atomic_action_engine(self, profile: Any) -> Any:
        if not self._drawers:
            return super().create_atomic_action_engine(profile)
        from .drawer_runtime import DrawerPlacementEngine

        engine = DrawerPlacementEngine(
            self._create_motion_generator(),
            control_profiles=profile.action_control_profiles(),
            grasp_pose_generators=self._grasp_pose_generators,
            drawer_observations=self._drawers,
        )
        self.task_program_registration.validate_engine(engine)
        return engine

    def registration_owned_segment_policy_ports(self) -> tuple[Any, Any]:
        return self._task_post_port, self.segment_policy_port


@dataclass(frozen=True, slots=True)
class TaskAdapterFactory:
    """Immutable identity and lazy service construction for a GenSim bundle."""

    registration: Any
    integration_fingerprint: str
    constraints: tuple[tuple[str, StabilityConstraint], ...]
    grasp_factories: tuple[tuple[str, Any], ...]
    cartesian_approaches: bool = False
    articulation_bindings: tuple = ()
    drawer_routes: tuple = ()

    def create_adapter(self, environment: Any) -> TaskProgramEnvironmentAdapter:
        """Return the exact shared adapter; no Session or Bridge is overridden."""
        self.registration.assert_unchanged()
        motion_factory = None
        if any(
            preset.motion_policy.strategy == "motion_gen"
            for preset in self.registration.robot_profile_binding.presets
        ):
            from embodichain.lab.sim.motion.motion_generator import (
                MotionGenCfg,
                MotionGenerator,
            )
            from embodichain.lab.sim.motion.planners.curobo.curobo_planner import (
                CuroboPlannerCfg,
                CuroboWorldCfg,
            )
            from embodichain.lab.task_program.semantics import SceneCollisionRole

            bindings = self.registration.scene_binding.rigid_objects
            obstacles = {
                item.entity_id: environment.sim.get_rigid_object(item.simulation_uid)
                for item in bindings
                if item.collision_role is not SceneCollisionRole.NONE
            }
            dynamic = [
                item.entity_id
                for item in bindings
                if item.collision_role is SceneCollisionRole.DYNAMIC
            ]
            if not obstacles or any(value is None for value in obstacles.values()):
                raise ValueError(
                    "GenSim motion_gen requires bound scene collision objects."
                )
            motion_factory = lambda: MotionGenerator(
                MotionGenCfg(
                    planner_cfg=CuroboPlannerCfg(
                        robot_uid=environment.robot.uid,
                        use_cuda_graph=False,
                        world=CuroboWorldCfg(
                            rigid_objects=obstacles,
                            dynamic_obstacle_names=dynamic,
                        ),
                    )
                )
            )
        else:
            from embodichain.lab.sim.motion.motion_generator import MotionGenCfg
            from embodichain.lab.sim.motion.planners import ToppraPlannerCfg
            from .motion import ApproachMotionGenerator, CheckedMotionGenerator

            generator_type = (
                ApproachMotionGenerator
                if self.cartesian_approaches
                else CheckedMotionGenerator
            )
            motion_factory = lambda: generator_type(
                MotionGenCfg(
                    planner_cfg=ToppraPlannerCfg(robot_uid=environment.robot.uid)
                )
            )
            if self.drawer_routes:
                from .drawer_curobo import DrawerMotionGenerator

                motion_factory = lambda: DrawerMotionGenerator(
                    MotionGenCfg(
                        planner_cfg=ToppraPlannerCfg(
                            robot_uid=environment.robot.uid,
                            sim_instance_id=environment.sim.instance_id,
                        )
                    ),
                    simulation=environment.sim,
                )
        grasp_generators = {name: create() for name, create in self.grasp_factories}
        grasp_generators = install_grasp_filters(
            self.registration,
            environment.sim,
            grasp_generators,
        )
        if self.articulation_bindings:
            import torch

            from .articulation_binding import handle_mesh
            from .e6_clearance import profile_clearances

            geometry_keys = set()
            for binding in self.articulation_bindings:
                art = environment.sim.get_articulation(binding.object_id)
                if art is None:
                    raise ValueError(
                        f"Declared articulation {binding.object_id!r} is absent."
                    )
                vertices, faces = handle_mesh(binding, art.cfg.fpath)
                geometry_keys.add(
                    geometry_key(
                        torch.as_tensor(vertices, dtype=torch.float32),
                        torch.as_tensor(faces, dtype=torch.int64),
                    )
                )
            grasp_generators = install_e6_approach_filters(
                grasp_generators,
                frozenset(geometry_keys),
                clearances=profile_clearances(
                    self.registration,
                    environment.robot,
                    environment.sim.get_rigid_object("table"),
                ),
            )
        factory = _TaskFactory(
            environment.sim,
            environment.robot,
            self.registration,
            step_dt=environment.step_dt,
            motion_generator_factory=motion_factory,
            grasp_pose_generators=grasp_generators,
            constraints=dict(self.constraints),
            articulation_bindings=self.articulation_bindings,
            drawer_routes=self.drawer_routes,
        )
        return factory.create_adapter()


def load_deployment(
    *, task_program: object, skill_profile: object, base_dir: str | Path
) -> Any:
    """Compose the current integration plus an explicitly versioned local contract."""
    base = compose_deployment(
        task_program=task_program,
        skill_profile=skill_profile,
        base_dir=base_dir,
    )
    path = Path(base_dir) / "task_program" / "constraints.json"
    if not path.is_file():
        raise ValueError(
            "GenSim bundle has no task constraints; regenerate the bundle."
        )
    payload = load_config(path)
    if (
        type(payload) is not dict
        or not {"schema_version", "presets"}.issubset(payload)
        or set(payload) - {"schema_version", "presets", "drawers"}
        or payload["schema_version"] != "gen_sim_task_constraints/v1"
    ):
        raise ValueError(
            "Unsupported GenSim task constraint format; regenerate the bundle."
        )
    presets = payload["presets"]
    if type(presets) is not dict or any(
        type(name) is not str or not name.startswith("gen_sim.") or name != name.strip()
        for name in presets
    ):
        raise ValueError("Task stability presets must have exact gen_sim.* names.")
    constraints = {
        name: StabilityConstraint.decode(cfg) for name, cfg in presets.items()
    }
    from .drawer_binding import DrawerRoute

    if type(payload.get("drawers", [])) is not list:
        raise ValueError("Drawer routes must be a list.")
    drawers = tuple(DrawerRoute.decode(value) for value in payload.get("drawers", []))
    if len({route.affordance for route in drawers}) != len(drawers):
        raise ValueError("Drawer routes must have unique affordance identities.")
    settle_presets = dict(base.integration.registration.settle_presets)
    if set(settle_presets) & set(constraints):
        raise ValueError("Task stability presets cannot replace core settling presets.")
    for name in constraints:
        settle_presets[name] = settle_presets["rigid_object"].snapshot()
    from .articulation_slide import (
        ArticulationSlideFactory,
        ArticulationWithdrawFactory,
        preset_id,
    )

    slides = [
        f
        for f in base.integration.registration.registered_semantic_lowerer_factories
        if type(f) is ArticulationSlideFactory
    ]
    withdrawals = [
        f
        for f in base.integration.registration.registered_semantic_lowerer_factories
        if type(f) is ArticulationWithdrawFactory
    ]
    articulation_bindings = ()
    if slides or withdrawals:
        if (
            len(slides) != 1
            or len(withdrawals) != 1
            or slides[0].bindings != withdrawals[0].bindings
        ):
            raise ValueError(
                "E6 Slide and withdrawal require identical declared bindings."
            )
        articulation_bindings = slides[0].bindings
        for binding in articulation_bindings:
            for state in ("open", "closed"):
                name = preset_id(binding, state)
                if name in settle_presets:
                    raise ValueError(
                        "Articulation policies cannot replace existing presets."
                    )
                settle_presets[name] = settle_presets["rigid_object"].snapshot()
    for route in drawers:
        if route.binding not in articulation_bindings:
            raise ValueError(
                "Drawer placement and E6 must share the exact part binding."
            )
        containers = base.integration.registration.scene_binding.containers
        if not any(
            c.entity_id == route.affordance
            and c.parent_id == route.binding.link_id
            and c.release_clearance == 0
            for c in containers
        ):
            raise ValueError(
                "Drawer container must belong to the declared live E6 link."
            )
    # Registration materializes built-in grounders during construction; feeding
    # them back through replace would duplicate placement routes.
    registration = replace(
        base.integration.registration,
        settle_presets=settle_presets,
        relation_grounders=(),
    )
    program = load_config(base.program_path)
    from .align_held import with_held_alignment

    registration = with_held_alignment(
        registration, program=program, constraints=constraints
    )
    from .release_clearance import CLEAR_RELEASED_CALL, with_release_clearance

    registration = with_release_clearance(registration, program=program)
    if any(cfg.kind == "stack" for cfg in constraints.values()):
        from .stack_place import with_stack_placement

        registration = with_stack_placement(
            registration,
            targets=frozenset(
                (cfg.entity, cfg.reference)
                for cfg in constraints.values()
                if cfg.kind == "stack"
            ),
        )
    grasp_factories = base.integration.grasp_factories
    # The pad envelope is needed for thin-object coordinated grasps. Retain
    # the conservative full-finger envelope for inclined single-arm grasps:
    # this three-box model couples palm placement to the finger length.
    # The configured Robotiq grasp command closes its pads to zero gap.
    # Its reference 1 cm sampling cutoff excludes the original tray's
    # measured 5-8 mm contacts. The URDF pad collision mesh spans 65 mm
    # about the configured 0.2 m TCP, not the proxy's symmetric 130 mm.
    # Keep palm/width/thickness checks and all physical commands unchanged.
    coordinated = any(
        item.get("steps", {}).get("call", {}).get("call_id")
        in {"simulation.coordinated_hold", "simulation.coordinated_transport"}
        for item in program["program"]["items"]
    )
    grasp_factories = tuple(
        (
            name,
            (
                replace(
                    factory,
                    min_opening_width=(
                        min(factory.min_opening_width, 0.005)
                        if coordinated
                        else factory.min_opening_width
                    ),
                    finger_length=0.065 if coordinated else factory.finger_length,
                )
                if factory.model_id == "robotiq_arg2f_140"
                else factory
            ),
        )
        for name, factory in grasp_factories
    )
    cartesian_approaches = any(
        item.get("steps", {}).get("call", {}).get("kind") == "hand_over"
        or item.get("steps", {}).get("call", {}).get("call_id")
        in {CLEAR_RELEASED_CALL, "gen_sim.articulation_withdraw"}
        for item in program["program"]["items"]
    )
    fingerprint = canonical_hash(
        {
            **(
                {"drawer_curobo_revision": 1}
                if any(
                    item.get("steps", {})
                    .get("call", {})
                    .get("arguments", {})
                    .get("target")
                    in {route.affordance for route in drawers}
                    for item in program["program"]["items"]
                )
                else {}
            ),
            "adapter_contract": ADAPTER_CONTRACT,
            "grasp_filter_revision": GRASP_FILTER_REVISION,
            "motion_validation_revision": MOTION_VALIDATION_REVISION,
            "core_integration": base.integration.integration_fingerprint,
            "registration": registration.fingerprint,
            "task_constraints": payload,
            "program": program,
            "cartesian_approaches": cartesian_approaches,
            "grasp_pose_generators": {
                name: asdict(factory) for name, factory in grasp_factories
            },
        }
    )
    adapter = TaskAdapterFactory(
        registration,
        fingerprint,
        tuple(constraints.items()),
        grasp_factories,
        cartesian_approaches,
        articulation_bindings,
        drawers,
    )
    integration = replace(
        base.integration,
        registration=registration,
        adapter_factory=adapter,
        integration_fingerprint=fingerprint,
    )
    return replace(base, integration=integration)


def register_deployment(
    deployment: Any, *, environment_id: str, max_episode_steps: int
) -> None:
    """Use the common environment and registry with a task-owned adapter factory."""
    from embodichain.lab.gym.envs import EmbodiedEnv
    from embodichain.lab.gym.utils.registration import (
        REGISTERED_ENVS,
        get_env_spec,
        register_env_function,
    )

    if environment_id in REGISTERED_ENVS:
        previous = get_env_spec(environment_id)
        if (
            previous.cls is not EmbodiedEnv
            or previous.task_program_adapter_factory.integration_fingerprint
            != deployment.integration.integration_fingerprint
        ):
            raise ValueError(
                "GenSim environment ID is already bound to another integration."
            )
        return
    register_env_function(
        EmbodiedEnv,
        environment_id,
        max_episode_steps=max_episode_steps,
        task_program_adapter_factory=deployment.integration.adapter_factory,
    )
