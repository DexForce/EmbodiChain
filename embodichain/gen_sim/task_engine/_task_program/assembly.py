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
import hashlib
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
from .identity import bundle_registration_fingerprint
from .grasp_filter import (
    GRASP_FILTER_REVISION,
    geometry_key,
    install_e6_approach_filters,
    install_grasp_filters,
)
from .motion import MOTION_VALIDATION_REVISION
from .state_handoff import StagingContract

__all__: list[str] = []

from .adaptive_grasp import ADAPTIVE_GRASP_REVISION
from .coordinated_grasp import COORDINATED_GRASP_REVISION
from .coordinated_motion import COORDINATED_MOTION_REVISION, GenSimCoordinatedPickment
from .invocation_policy import (
    INVOCATION_POLICY_REVISION,
    GenSimActionEngine,
    bind_cartesian_calls,
    bind_motion_samples,
    bind_pour_receivers,
    bind_articulation_calls,
)

ADAPTER_CONTRACT = "gen_sim.task_program/2620929c/v8"


def _press_scene_identity(base_dir: str | Path) -> dict[str, Any]:
    """Hash physical values and asset contents, never publication locations."""
    scene_path = Path(base_dir) / "components/scene.yaml"

    def normalize(value: Any) -> Any:
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                if key == "fpath" and isinstance(item, str):
                    asset = Path(item)
                    if not asset.is_absolute():
                        asset = scene_path.parent / asset
                    result[key] = {
                        "sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
                        "format": asset.suffix.lower(),
                    }
                else:
                    result[key] = normalize(item)
            return result
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return value

    return normalize(load_config(scene_path))


class _TaskFactory(SimulationTaskProgramFactory):
    """Select task-owned observation services, not a different executor."""

    _pick_continuations: tuple = ()

    def __init__(
        self,
        *args: Any,
        constraints: dict[str, StabilityConstraint],
        articulation_bindings: tuple = (),
        pour_receivers: dict[str, str] | None = None,
        drawer_routes: tuple = (),
        adaptive_pick: bool = False,
        pick_purposes: tuple[tuple[str, str], ...] = (),
        coordinated_motion: bool = False,
        motion_samples: tuple[tuple[str, int], ...] = (),
        cartesian_calls: tuple[str, ...] = (),
        articulation_calls: tuple[tuple[str, str], ...] = (),
        clearance_routes: tuple = (),
        alignment_continuation_routes: tuple = (),
        staging_contract: StagingContract | None = None,
        runtime_preset: str | None = None,
        press_routes: tuple = (),
        pick_continuations: tuple = (),
        twist_routes: tuple = (),
        pick_contact_revision: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._pour_receivers = dict(pour_receivers or {})
        self._adaptive_pick = adaptive_pick
        self._pick_purposes = pick_purposes
        self._coordinated_motion = coordinated_motion
        self._motion_samples = motion_samples
        self._cartesian_calls = cartesian_calls
        self._articulation_calls = articulation_calls
        self._clearance_routes = clearance_routes
        self._alignment_continuation_routes = alignment_continuation_routes
        self._staging_contract = staging_contract or StagingContract.empty()
        self._runtime_preset = runtime_preset
        self._press_routes = press_routes
        self._pick_continuations = pick_continuations
        self._twist_routes = twist_routes
        self._pick_contact_revision = pick_contact_revision
        self._task_post_port = TaskStabilityPort(
            self.segment_policy_port,
            self._simulation,
            self._robot,
            self.task_program_registration.scene_binding,
            constraints,
            step_dt=self.step_dt,
        )
        self._support_post_port = self._task_post_port
        self._strong_support = any(
            c.support_contract_revision == 1 for c in constraints.values()
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
        self._twist_ports: dict[str, Any] = {}
        for route in press_routes:
            from .press_runtime import PressAcceptancePort

            self._task_post_port = PressAcceptancePort(
                self._task_post_port,
                self._simulation,
                self._robot,
                route,
                self.step_dt,
            )

        for route in twist_routes:
            from .twist_runtime import TwistAcceptancePort

            self._task_post_port = TwistAcceptancePort(
                self._task_post_port,
                self._simulation,
                self._robot,
                route,
                self.step_dt,
            )
            self._twist_ports[route.operation_id] = self._task_post_port

    def create_atomic_action_engine(self, profile: Any) -> Any:
        """Keep drawer-owned transport separate from ordinary GenSim wrappers."""
        from .actions import (
            GenSimHandOver,
            GenSimMoveHeldObject,
            GenSimPickUp,
            GenSimPlace,
            GenSimPour,
        )

        from embodichain.lab.task_program.semantics import RobotSkillProfile

        if (
            not isinstance(profile, RobotSkillProfile)
            or profile.profile_id != self.robot_profile_id
        ):
            raise ValueError(
                "Invocation policies require the registered robot profile."
            )
        motion = self._create_motion_generator()
        if motion.robot is not self._robot:
            raise ValueError(
                "The invocation policy engine must own the selected robot."
            )
        if self._drawers:
            from .drawer_runtime import DrawerPlacementEngine

            engine = DrawerPlacementEngine(
                motion,
                control_profiles=profile.action_control_profiles(),
                grasp_pose_generators=self._grasp_pose_generators,
                drawer_observations=self._drawers,
                motion_samples=self._motion_samples,
                cartesian_calls=self._cartesian_calls,
                articulation_calls=self._articulation_calls,
            )
        else:
            engine = GenSimActionEngine(
                motion,
                control_profiles=profile.action_control_profiles(),
                grasp_pose_generators=self._grasp_pose_generators,
                motion_samples=self._motion_samples,
                cartesian_calls=self._cartesian_calls,
                articulation_calls=self._articulation_calls,
            )
        pick = GenSimPickUp()
        pick.adaptive_unconstrained = self._adaptive_pick
        pick.pick_purposes = dict(self._pick_purposes)
        pick.path_simulation = getattr(self, "_simulation", None)
        pick.path_excluded_invocations = frozenset(self._cartesian_calls)
        if self._pick_contact_revision in (1, 2):
            from .pick_contact import build_closing_contact_screen

            pick.closing_geometry_factory = (
                lambda request, context: build_closing_contact_screen(
                    self._robot,
                    self._simulation,
                    self.task_program_registration.scene_binding,
                    request,
                    revision=self._pick_contact_revision,
                )
            )
        engine.register(pick, replace=True)
        engine.register(GenSimHandOver(), replace=True)
        transport = GenSimMoveHeldObject(staging_contract=self._staging_contract)
        from .place_recovery import scene_vertices

        transport.held_path_scene_vertices = scene_vertices(
            getattr(self, "_simulation", None),
            getattr(self.task_program_registration, "scene_binding", None),
        )
        engine.register(transport, replace=True)
        place = GenSimPlace()
        place.recovery_scene_vertices = transport.held_path_scene_vertices
        engine.register(place, replace=True)
        engine.register(GenSimPour(self._pour_receivers), replace=True)
        if self._coordinated_motion:
            engine.register(GenSimCoordinatedPickment(), replace=True)
        if self._press_routes:
            from .press_runtime import GenSimPress

            engine.register(GenSimPress(), replace=True)
        if self._twist_routes:
            from .twist_runtime import GenSimTwist
            from .twist_geometry_guard import E8GeometryGuard

            action = GenSimTwist()
            for route in self._twist_routes:
                operation = getattr(route, "operation_id", "")
                port = getattr(self, "_twist_ports", {}).get(
                    operation, self._task_post_port
                )
                guard = E8GeometryGuard(
                    route, self._simulation, self._robot, port.sensor
                )
                action._operation_guards[operation] = guard
                action._operation_states[operation] = getattr(
                    port, "_candidate_state", None
                )
                if getattr(route, "feedback_enabled", False):
                    from .twist_feedback import TwistFeedbackState

                    feedback = TwistFeedbackState(
                        route, self._simulation, self._robot, port.sensor, guard
                    )
                    port.sensor._feedback_state = feedback
                    action._operation_feedback[operation] = feedback
            engine.register(action, replace=True)
            for feedback in action._operation_feedback.values():
                engine.configure_twist_feedback(feedback)
        self.task_program_registration.validate_engine(engine)
        if (
            self._clearance_routes
            or self._alignment_continuation_routes
            or self._pick_continuations
        ):
            engine._clearance_profile = profile
        return engine

    def create_planning_observation_provider(
        self, *, scene_registry: Any, engine: Any, clock: Any
    ) -> Any:
        provider = super().create_planning_observation_provider(
            scene_registry=scene_registry, engine=engine, clock=clock
        )
        # GenSim snapshot collision qualification needs the same-tick base frame.
        provider._capture_root_pose = True
        if self._strong_support:
            self._support_post_port.bind_clock(clock)
        return provider

    def create_registered_semantic_lowerers(
        self, *, scene_registry: Any, engine: Any
    ) -> tuple[Any, ...]:
        """Bind preview calls through canonical assembly, without another executor."""
        profile = None
        if (
            self._clearance_routes
            or self._alignment_continuation_routes
            or self._pick_continuations
        ):
            profile = engine._clearance_profile
            del engine._clearance_profile
        lowerers = super().create_registered_semantic_lowerers(
            scene_registry=scene_registry, engine=engine
        )
        if (
            self._clearance_routes
            or self._alignment_continuation_routes
            or self._pick_continuations
        ):
            from embodichain.lab.task_program.language import TaskProgramIntegrationCfg
            from embodichain.lab.task_program.runtime.assembly import (
                assemble_semantic_executor_components,
            )
            from .clearance_recovery import bind_park_requests

            registration = self.task_program_registration
            semantic = assemble_semantic_executor_components(
                scene_registry,
                profile,
                engine,
                TaskProgramIntegrationCfg(
                    robot_profile=self.robot_profile_id,
                    scene_registry=self.scene_registry_id,
                    runtime_preset=self._runtime_preset,
                ),
                call_catalog=registration.call_catalog,
                endpoint_adapters=registration.endpoint_adapter_map,
                registered_lowerers=lowerers,
                relation_grounders=registration.relation_grounders,
                handover_pose_providers=registration.handover_pose_providers,
            )
            if self._clearance_routes:
                engine.bind_release_clearance(
                    bind_park_requests(semantic.compiler, self._clearance_routes),
                    simulation=self._simulation,
                )
            if self._alignment_continuation_routes:
                from .alignment_continuation import bind_previews

                engine.bind_alignment_continuation(
                    bind_previews(
                        semantic.compiler, self._alignment_continuation_routes
                    )
                )
            if self._pick_continuations:
                from .pick_continuation import bind_previews

                engine.actions["pick_up"].continuation_previews = bind_previews(
                    semantic.compiler, self._pick_continuations
                )
        return lowerers

    def registration_owned_segment_policy_ports(self) -> tuple[Any, Any]:
        return self._task_post_port, self.segment_policy_port


@dataclass(frozen=True, slots=True)
class TaskAdapterFactory:
    """Immutable identity and lazy service construction for a GenSim bundle."""

    registration: Any
    integration_fingerprint: str
    constraints: tuple[tuple[str, StabilityConstraint], ...]
    grasp_factories: tuple[tuple[str, Any], ...]
    articulation_bindings: tuple = ()
    pour_receivers: tuple[tuple[str, str], ...] = ()
    drawer_routes: tuple = ()
    adaptive_pick: bool = False
    pick_purposes: tuple[tuple[str, str], ...] = ()
    coordinated_grasp: bool = False
    motion_samples: tuple[tuple[str, int], ...] = ()
    cartesian_calls: tuple[str, ...] = ()
    articulation_calls: tuple[tuple[str, str], ...] = ()
    clearance_routes: tuple = ()
    alignment_continuation_routes: tuple = ()
    staging_contract: StagingContract = StagingContract.empty()
    runtime_preset: str | None = None
    press_routes: tuple = ()
    twist_routes: tuple = ()
    support_contract_revision: int | None = None
    pick_contact_revision: int | None = None

    def create_adapter(self, environment: Any) -> TaskProgramEnvironmentAdapter:
        """Bind task-owned planning scopes without overriding Session or Bridge."""
        self.registration.assert_unchanged()
        for route in self.twist_routes:
            from .twist_runtime import (
                SENSOR_UID as TWIST_SENSOR_UID,
                ensure_sensor as ensure_twist_sensor,
            )

            environment.sensors[route.sensor_uid] = ensure_twist_sensor(
                environment.sim, environment.robot, route
            )
        for route in self.press_routes:
            from .press_runtime import SENSOR_UID, ensure_sensor

            environment.sensors[route.sensor_uid] = ensure_sensor(
                environment.sim, environment.robot, route
            )
        motion_factory = None
        if any(
            preset.motion_policy.strategy == "motion_gen"
            and (self.runtime_preset is None or preset.preset_id == self.runtime_preset)
            for preset in self.registration.robot_profile_binding.presets
        ):
            from embodichain.lab.sim.motion.motion_generator import (
                MotionGenCfg,
            )
            from .motion import CheckedMotionGenerator
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
            motion_factory = lambda: CheckedMotionGenerator(
                MotionGenCfg(
                    planner_cfg=CuroboPlannerCfg(
                        robot_uid=environment.robot.uid,
                        sim_instance_id=environment.sim.instance_id,
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
            from .motion import CheckedMotionGenerator

            motion_factory = lambda: CheckedMotionGenerator(
                MotionGenCfg(
                    planner_cfg=ToppraPlannerCfg(
                        robot_uid=environment.robot.uid,
                        sim_instance_id=environment.sim.instance_id,
                    )
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
        if self.coordinated_grasp:
            from .coordinated_grasp import install_coordinated_grasps

            grasp_generators = install_coordinated_grasps(
                self.registration, environment.robot, grasp_generators
            )
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
        pick_continuations = ()
        if self.adaptive_pick:
            from embodichain.lab.task_program.language import TaskProgramCfg
            from .pick_continuation import bind_calls

            program = getattr(getattr(environment, "cfg", None), "task_program", None)
            if type(program) is TaskProgramCfg:
                preset = next(
                    (
                        p
                        for p in self.registration.robot_profile_binding.presets
                        if p.preset_id == self.runtime_preset
                    ),
                    None,
                )
                pick_continuations = bind_calls(
                    self.registration.catalog.preflight(program), preset
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
            pour_receivers=dict(self.pour_receivers),
            drawer_routes=self.drawer_routes,
            adaptive_pick=self.adaptive_pick,
            pick_purposes=self.pick_purposes,
            coordinated_motion=self.coordinated_grasp,
            motion_samples=self.motion_samples,
            cartesian_calls=self.cartesian_calls,
            articulation_calls=self.articulation_calls,
            clearance_routes=self.clearance_routes,
            alignment_continuation_routes=self.alignment_continuation_routes,
            staging_contract=self.staging_contract,
            runtime_preset=self.runtime_preset,
            press_routes=self.press_routes,
            pick_continuations=pick_continuations,
            twist_routes=self.twist_routes,
            pick_contact_revision=self.pick_contact_revision,
        )
        support = [
            cfg
            for _, cfg in self.constraints
            if cfg.kind in {"stack", "supported_placement", "supported_region"}
        ]
        if support:
            native_ids = {
                b.entity_id: b.simulation_uid
                for b in self.registration.scene_binding.rigid_objects
            }
            for cfg in support:
                uid = f"gen_sim_support__{native_ids[cfg.entity]}__{native_ids[cfg.reference]}"
                sensor = environment.sim.get_sensor(uid)
                if sensor is None:
                    raise ValueError(f"Declared support sensor {uid!r} is absent.")
                environment.sensors[uid] = sensor
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
        or not {
            "schema_version",
            "presets",
            "motion_samples",
            "cartesian_calls",
            "pour_receivers",
            "staging_calls",
            "staging_retention_sources",
        }.issubset(payload)
        or set(payload)
        - {
            "schema_version",
            "presets",
            "drawers",
            "adaptive_pick",
            "pick_purposes",
            "motion_samples",
            "cartesian_calls",
            "pour_receivers",
            "staging_calls",
            "staging_retention_sources",
            "support_contract_revision",
            "pick_contact_revision",
        }
        or payload["schema_version"] != "gen_sim_task_constraints/v1"
    ):
        raise ValueError(
            "Unsupported GenSim task constraint format; regenerate the bundle."
        )
    presets = payload["presets"]
    support_revision = payload.get("support_contract_revision")
    pick_contact_revision = payload.get("pick_contact_revision")
    if pick_contact_revision is not None and (
        type(pick_contact_revision) is not int or pick_contact_revision not in (1, 2)
    ):
        raise ValueError("Unsupported GenSim Pick contact revision.")
    if support_revision is not None and (
        type(support_revision) is not int or support_revision != 1
    ):
        raise ValueError("Unsupported GenSim support contract revision.")
    adaptive_pick = payload.get("adaptive_pick", False)
    if type(adaptive_pick) is not bool:
        raise ValueError("GenSim adaptive_pick must be an explicit boolean.")
    if type(presets) is not dict or any(
        type(name) is not str or not name.startswith("gen_sim.") or name != name.strip()
        for name in presets
    ):
        raise ValueError("Task stability presets must have exact gen_sim.* names.")
    constraints = {
        name: StabilityConstraint.decode(cfg) for name, cfg in presets.items()
    }
    if support_revision is None and any(
        cfg.support_contract_revision == 1 for cfg in constraints.values()
    ):
        raise ValueError("Strong support presets require the bundle contract marker.")
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
    from .twist_runtime import TwistFactory, TwistPrepareFactory

    turns = [
        f
        for f in base.integration.registration.registered_semantic_lowerer_factories
        if isinstance(f, TwistFactory)
    ]
    turn_preparations = [
        f
        for f in base.integration.registration.registered_semantic_lowerer_factories
        if isinstance(f, TwistPrepareFactory) and not isinstance(f, TwistFactory)
    ]
    twist_routes = ()
    if turns or turn_preparations:
        if [f.route for f in turns] != [f.route for f in turn_preparations] or len(
            {f.route.operation_id for f in turns}
        ) != len(turns):
            raise ValueError(
                "E8 prepare and Twist must share one exact calibrated route."
            )
        twist_routes = tuple(f.route for f in turns)
        for route in twist_routes:
            for phase in ("ready", "chunk", "turned"):
                name = route.preset(phase)
                if name in settle_presets:
                    raise ValueError("E8 policies cannot replace existing presets.")
                settle_presets[name] = settle_presets["rigid_object"].snapshot()
    from .press_runtime import PressFactory, PressPrepareFactory

    presses = [
        f
        for f in base.integration.registration.registered_semantic_lowerer_factories
        if isinstance(f, PressFactory)
    ]
    preparations = [
        f
        for f in base.integration.registration.registered_semantic_lowerer_factories
        if isinstance(f, PressPrepareFactory) and not isinstance(f, PressFactory)
    ]
    press_routes = ()
    if presses or preparations:
        if [f.route for f in presses] != [f.route for f in preparations] or len(
            {f.route.operation_id for f in presses}
        ) != len(presses):
            raise ValueError(
                "E9 prepare and Press must share exactly one source-qualified route."
            )
        press_routes = tuple(f.route for f in presses)
        for route in press_routes:
            for phase in ("ready", "pressed"):
                name = route.preset(phase)
                if name in settle_presets:
                    raise ValueError(
                        "E9 post-policies cannot replace existing presets."
                    )
                settle_presets[name] = settle_presets["rigid_object"].snapshot()
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
    from embodichain.lab.task_program.language import load_task_program
    from .adaptive_grasp import bind_pick_purposes

    if press_routes:
        from .press_binding import validate_program

        for route in press_routes:
            validate_program(program, route)
    if twist_routes:
        from .twist_binding import validate_program as validate_twist_program

        for route in twist_routes:
            validate_twist_program(program, route)
    from .align_held import _AlignHeldFactory, with_held_alignment

    registration = with_held_alignment(
        registration,
        program=program,
        constraints=constraints,
        verify_retention=True,
    )
    from .release_clearance import with_release_clearance

    registration = with_release_clearance(registration, program=program)
    from .grip_evidence import with_native_grip_evidence

    registration = with_native_grip_evidence(registration)
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
    compiled = registration.catalog.preflight(
        load_task_program(
            base.program_path,
            integration=base.selection,
            validation_context=registration.catalog,
        )
    )
    pick_purposes = bind_pick_purposes(payload.get("pick_purposes", {}), compiled)
    motion_samples = bind_motion_samples(payload["motion_samples"], compiled)
    cartesian_calls = bind_cartesian_calls(payload["cartesian_calls"], compiled)
    articulation_calls = bind_articulation_calls(compiled)
    from .staging_recovery import STAGING_RECOVERY_REVISION

    # Owned drawer transports opt out at the call boundary; unrelated routes
    # must not revoke verified ordinary-call retention authority.
    staging_calls = tuple(str(value) for value in payload.get("staging_calls", ()))
    staging_retention_sources = tuple(
        tuple(str(value) for value in source)
        for source in payload.get("staging_retention_sources", ())
    )
    if any(len(source) != 5 for source in staging_retention_sources):
        raise ValueError("Staging retention sources require five declared fields.")
    from .staging_recovery import verified_retention_sources

    verified_sources = verified_retention_sources(
        registration.robot_profile_binding,
        base.selection.runtime_preset,
        staging_retention_sources,
    )
    staging_contract = StagingContract(
        rigid_entity_ids=frozenset(
            item.entity_id for item in registration.scene_binding.rigid_objects
        ),
        invocation_ids=frozenset(verified_sources),
        retention_sources=verified_sources,
    )
    from .clearance_recovery import CLEARANCE_RECOVERY_REVISION, bind_clearance_calls

    clearance_routes = bind_clearance_calls(compiled)
    from .alignment_continuation import ALIGNMENT_CONTINUATION_REVISION, bind_calls

    continuation_preset = next(
        (
            preset
            for preset in registration.robot_profile_binding.presets
            if preset.preset_id == base.selection.runtime_preset
        ),
        None,
    )
    alignment_continuation_routes = bind_calls(
        compiled,
        tuple(item[0] for item in staging_retention_sources),
        continuation_preset,
    )
    grasp_factories = base.integration.grasp_factories
    coordinated = any(
        item.get("steps", {}).get("call", {}).get("call_id")
        in {"simulation.coordinated_hold", "simulation.coordinated_transport"}
        for item in program["program"]["items"]
    )
    pour_receivers = bind_pour_receivers(payload["pour_receivers"], compiled)
    from .articulation_recovery import ARTICULATION_RECOVERY_REVISION
    from .park_recovery import PARK_RECOVERY_REVISION
    from .pick_safety import PICK_OPENING_RECOVERY_REVISION
    from .services import _PickLowererFactory

    fingerprint = canonical_hash(
        {
            "adaptive_grasp_revision": ADAPTIVE_GRASP_REVISION,
            "invocation_policy_revision": INVOCATION_POLICY_REVISION,
            **(
                {"pick_opening_recovery_revision": PICK_OPENING_RECOVERY_REVISION}
                if any(
                    isinstance(factory, _PickLowererFactory)
                    and any(
                        route.grasp_region is not None
                        or route.release_clearance_object_pose is not None
                        for route in factory.routes
                    )
                    for factory in registration.registered_semantic_lowerer_factories
                )
                else {}
            ),
            **(
                {"staging_recovery_revision": STAGING_RECOVERY_REVISION}
                if staging_calls or (drawers and verified_sources)
                else {}
            ),
            **(
                {"clearance_recovery_revision": CLEARANCE_RECOVERY_REVISION}
                if clearance_routes
                else {}
            ),
            **(
                {"park_recovery_revision": PARK_RECOVERY_REVISION}
                if clearance_routes
                else {}
            ),
            **(
                {"alignment_continuation_revision": ALIGNMENT_CONTINUATION_REVISION}
                if alignment_continuation_routes
                else {}
            ),
            **(
                {"articulation_recovery_revision": ARTICULATION_RECOVERY_REVISION}
                if articulation_calls
                else {}
            ),
            **(
                {
                    "coordinated_grasp_revision": COORDINATED_GRASP_REVISION,
                    "coordinated_motion_revision": COORDINATED_MOTION_REVISION,
                }
                if coordinated
                else {}
            ),
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
            **(
                {
                    "twist_runtime_revision": 1,
                    "twist_scene_configuration": _press_scene_identity(base_dir),
                }
                if twist_routes
                else {}
            ),
            **({"press_runtime_revision": 2} if press_routes else {}),
            **(
                {"press_scene_configuration": _press_scene_identity(base_dir)}
                if press_routes
                else {}
            ),
            "grasp_filter_revision": GRASP_FILTER_REVISION,
            "motion_validation_revision": MOTION_VALIDATION_REVISION,
            "core_integration": base.integration.integration_fingerprint,
            "registration": bundle_registration_fingerprint(registration),
            "task_constraints": payload,
            "program": program,
            "cartesian_calls": cartesian_calls,
            "pour_receivers": pour_receivers,
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
        articulation_bindings=articulation_bindings,
        pour_receivers=pour_receivers,
        drawer_routes=drawers,
        adaptive_pick=adaptive_pick,
        pick_purposes=pick_purposes,
        coordinated_grasp=coordinated,
        motion_samples=motion_samples,
        cartesian_calls=cartesian_calls,
        articulation_calls=articulation_calls,
        clearance_routes=clearance_routes,
        alignment_continuation_routes=alignment_continuation_routes,
        staging_contract=staging_contract,
        runtime_preset=base.selection.runtime_preset,
        press_routes=press_routes,
        twist_routes=twist_routes,
        support_contract_revision=support_revision,
        pick_contact_revision=pick_contact_revision,
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
