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
from .motion import (
    MOTION_VALIDATION_REVISION,
    SHARED_MOTION_CONTRACT,
    validate_motion_contract,
)
from .state_handoff import StagingContract
from .held_command import bind_held_command_sources, bind_observed_alignment_calls

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
    bind_pour_return_calls,
    POUR_RETURN_POLICY_REVISION,
    bind_articulation_calls,
    bind_articulation_followups,
)

ADAPTER_CONTRACT = "gen_sim.task_program/2620929c/v16"


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
    _pick_contact_revisions: tuple[tuple[str, int], ...] = ()
    _world_yaw_free_places: tuple[str, ...] = ()
    _post_handover_places: tuple = ()

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
        handover_scene_clearance: bool = False,
        motion_samples: tuple[tuple[str, int], ...] = (),
        cartesian_calls: tuple[str, ...] = (),
        articulation_calls: tuple[tuple[str, str], ...] = (),
        articulation_followups: tuple[tuple[str, str], ...] = (),
        pour_return_calls: tuple[str, ...] = (),
        clearance_routes: tuple = (),
        alignment_continuation_routes: tuple = (),
        staging_contract: StagingContract | None = None,
        runtime_preset: str | None = None,
        press_routes: tuple = (),
        pick_continuations: tuple = (),
        pour_continuations: tuple = (),
        twist_routes: tuple = (),
        pick_contact_revision: int | None = None,
        pick_contact_revisions: tuple[tuple[str, int], ...] = (),
        world_yaw_free_places: tuple[str, ...] = (),
        measured_rigid_calls: tuple = (),
        post_handover_places: tuple = (),
        idle_invocations: tuple[tuple[str, str], ...] = (),
        observed_transport_calls: tuple[str, ...] = (),
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._pour_receivers = dict(pour_receivers or {})
        self._adaptive_pick = adaptive_pick
        self._pick_purposes = pick_purposes
        self._coordinated_motion = coordinated_motion
        self._handover_scene_clearance = handover_scene_clearance
        self._motion_samples = motion_samples
        self._cartesian_calls = cartesian_calls
        self._articulation_calls = articulation_calls
        self._articulation_followups = articulation_followups
        self._pour_return_calls = pour_return_calls
        self._clearance_routes = clearance_routes
        self._alignment_continuation_routes = alignment_continuation_routes
        self._staging_contract = staging_contract or StagingContract.empty()
        self._runtime_preset = runtime_preset
        self._press_routes = press_routes
        self._pick_continuations = pick_continuations
        self._pour_continuations = pour_continuations
        self._twist_routes = twist_routes
        self._pick_contact_revision = pick_contact_revision
        self._pick_contact_revisions = pick_contact_revisions
        self._world_yaw_free_places = world_yaw_free_places
        self._measured_rigid_calls = measured_rigid_calls
        self._post_handover_places = post_handover_places
        self._idle_invocations = idle_invocations
        self._observed_transport_calls = observed_transport_calls
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
                articulation_followups=self._articulation_followups,
                pour_return_calls=self._pour_return_calls,
            )
        else:
            engine = GenSimActionEngine(
                motion,
                control_profiles=profile.action_control_profiles(),
                grasp_pose_generators=self._grasp_pose_generators,
                motion_samples=self._motion_samples,
                cartesian_calls=self._cartesian_calls,
                articulation_calls=self._articulation_calls,
                articulation_followups=self._articulation_followups,
                pour_return_calls=self._pour_return_calls,
            )
        if self._idle_invocations:
            from .idle_clearance import IdleGuardMoveJoints

            idle = IdleGuardMoveJoints()
            idle.idle_invocations = dict(self._idle_invocations)
            idle.idle_simulation = self._simulation
            engine.register(idle, replace=True)
        pick = GenSimPickUp()
        pick.adaptive_unconstrained = self._adaptive_pick
        pick.pick_purposes = dict(self._pick_purposes)
        pick.path_simulation = getattr(self, "_simulation", None)
        pick.path_excluded_invocations = frozenset(self._cartesian_calls)
        contact_revisions = dict(self._pick_contact_revisions)
        if (
            self._pick_contact_revision in (1, 2, 3, 4, 5, 6, 7, 8, 9)
            or contact_revisions
        ):
            from .pick_contact import build_closing_contact_screen

            def closing_geometry(request: Any, context: Any) -> Any:
                revision = contact_revisions.get(
                    request.invocation_id, self._pick_contact_revision
                )
                if revision is None:
                    return None
                opening = {}
                if revision in (3, 4, 5, 6, 7, 8, 9):
                    opening = {
                        "context": context,
                        "preview": (pick.continuation_previews or {}).get(
                            request.invocation_id
                        ),
                        "purpose": (pick.pick_purposes or {}).get(
                            request.invocation_id
                        ),
                    }
                screen = build_closing_contact_screen(
                    self._robot,
                    self._simulation,
                    self.task_program_registration.scene_binding,
                    request,
                    revision=revision,
                    **opening,
                )
                return screen

            pick.closing_geometry_factory = closing_geometry
        engine.register(pick, replace=True)
        handover = GenSimHandOver()
        handover.post_place_exits = {
            route.handover_id: (route.subject, route.source, route.receiver)
            for route in self._post_handover_places
        }
        if self._handover_scene_clearance:
            from .handover_safety import build_handover_environment_screen

            handover.environment_screen_factory = (
                lambda request, context, resources: build_handover_environment_screen(
                    self._robot,
                    self._simulation,
                    self.task_program_registration.scene_binding,
                    request,
                    context,
                    resources,
                )
            )
        engine.register(handover, replace=True)
        transport = GenSimMoveHeldObject(
            staging_contract=self._staging_contract,
        )
        transport.observed_hand_invocations = frozenset(self._observed_transport_calls)
        from .place_recovery import scene_vertices

        meshes = scene_vertices(
            getattr(self, "_simulation", None),
            getattr(self.task_program_registration, "scene_binding", None),
        )
        pour = GenSimPour(self._pour_receivers)
        if self._pour_continuations:
            from .pour_services import gravity_from_simulation

            gravity = gravity_from_simulation(self._simulation)
            pick.pour_continuations = {r.pick_id: r for r in self._pour_continuations}
            pick.pour_scene_vertices = meshes
            pick.pour_gravity_world = gravity
            pour.continuations = {r.pour_id: r for r in self._pour_continuations}
            pour.scene_vertices = meshes
            pour.gravity_world = gravity
        engine.register(transport, replace=True)
        place = GenSimPlace()
        place.recovery_scene_vertices = meshes
        place.post_handover_places = {
            route.place_id: (route.subject, route.receiver, route.reference)
            for route in self._post_handover_places
        }
        from .place_branch import bind_on_invocations, bind_lateral_invocations

        place.ordinary_on_invocations = bind_on_invocations(
            self._pick_continuations,
            self._pick_purposes,
            self.task_program_registration.scene_binding,
        )
        place.support_surface_owners = {
            item.entity_id: item.parent_id
            for item in self.task_program_registration.scene_binding.support_surfaces
        }
        place.ordinary_lateral_invocations = bind_lateral_invocations(
            self._pick_continuations,
            self._pick_purposes,
            self.task_program_registration.scene_binding,
        )
        if self._world_yaw_free_places:
            from .place_yaw import bind_routes

            place.world_yaw_free_routes = bind_routes(self)
            place.world_yaw_simulation = self._simulation
        engine.register(place, replace=True)
        if self._measured_rigid_calls:
            from .measured_rigid import MeasuredRigidPolicy

            engine._measured_rigid_policy = MeasuredRigidPolicy(
                self._simulation,
                self.task_program_registration.scene_binding,
                self._measured_rigid_calls,
            )
        engine.register(pour, replace=True)
        if self._coordinated_motion:
            coordinated = GenSimCoordinatedPickment()
            coordinated.cargo_scene_vertices = meshes
            engine.register(coordinated, replace=True)
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
            engine.register(action, replace=True)
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
        from .pour_services import _PourRegionTransportLowerer

        region_lowerers = [
            lowerer
            for lowerer in lowerers
            if type(lowerer) is _PourRegionTransportLowerer
        ]
        if region_lowerers:
            if len(region_lowerers) != 1:
                raise ValueError("Pour region requires one exact geometry owner.")
            geometry = region_lowerers[0].region_geometry
            engine.actions["pick_up"].pour_region_geometries = {
                route.pick_id: geometry
                for route in self._pour_continuations
                if route.proxy_position_tolerance_m is not None
            }
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
    handover_scene_clearance: bool = False
    motion_samples: tuple[tuple[str, int], ...] = ()
    cartesian_calls: tuple[str, ...] = ()
    articulation_calls: tuple[tuple[str, str], ...] = ()
    articulation_followups: tuple[tuple[str, str], ...] = ()
    clearance_routes: tuple = ()
    alignment_continuation_routes: tuple = ()
    staging_contract: StagingContract = StagingContract.empty()
    runtime_preset: str | None = None
    press_routes: tuple = ()
    twist_routes: tuple = ()
    support_contract_revision: int | None = None
    pick_contact_revision: int | None = None
    pick_contact_revisions: tuple[tuple[str, int], ...] = ()
    world_yaw_free_places: tuple[str, ...] = ()
    measured_rigid_calls: tuple = ()
    post_handover_places: tuple = ()
    idle_invocations: tuple[tuple[str, str], ...] = ()
    pour_return_calls: tuple[str, ...] = ()
    held_command_sources: tuple = ()
    motion_contract: str = SHARED_MOTION_CONTRACT
    observed_transport_calls: tuple[str, ...] = ()

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
                ),
                motion_contract=self.motion_contract,
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
                ),
                motion_contract=self.motion_contract,
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
                    motion_contract=self.motion_contract,
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
                clearance_factory=lambda hand, fraction: profile_clearances(
                    self.registration,
                    environment.robot,
                    environment.sim.get_rigid_object("table"),
                    preshape_fraction=fraction,
                )[hand],
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
        from .pour_services import (
            PourTransportFactory,
            PourRegionTransportFactory,
            RetainedPourFactory,
        )

        pour_factory = next(
            (
                f
                for f in getattr(
                    self.registration, "registered_semantic_lowerer_factories", ()
                )
                if type(f) in (PourTransportFactory, PourRegionTransportFactory)
            ),
            None,
        )
        retained_factory = next(
            (
                f
                for f in getattr(
                    self.registration, "registered_semantic_lowerer_factories", ()
                )
                if type(f) is RetainedPourFactory
            ),
            None,
        )
        pour_continuations = ()
        if (pour_factory is None) != (retained_factory is None):
            raise ValueError(
                "Canonical Pour requires both transport and retention contracts."
            )
        if pour_factory is not None:
            from embodichain.lab.task_program.language import TaskProgramCfg
            from .pour_continuation import bind_calls as bind_pour_calls

            program = getattr(getattr(environment, "cfg", None), "task_program", None)
            if type(program) is not TaskProgramCfg:
                raise ValueError(
                    "Canonical Pour requires its compiled configured program."
                )
            preset = next(
                p
                for p in self.registration.robot_profile_binding.presets
                if p.preset_id == self.runtime_preset
            )
            if any(
                r.object_id != retained_factory.object_id
                for r in pour_factory.staging_routes
            ):
                raise ValueError(
                    "Canonical Pour transport and retained object identities disagree."
                )
            pour_continuations = bind_pour_calls(
                self.registration.catalog.preflight(program),
                preset,
                pour_factory,
                self.motion_samples,
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
            handover_scene_clearance=self.handover_scene_clearance,
            motion_samples=self.motion_samples,
            cartesian_calls=self.cartesian_calls,
            articulation_calls=self.articulation_calls,
            articulation_followups=self.articulation_followups,
            pour_return_calls=self.pour_return_calls,
            clearance_routes=self.clearance_routes,
            alignment_continuation_routes=self.alignment_continuation_routes,
            staging_contract=self.staging_contract,
            observed_transport_calls=self.observed_transport_calls,
            runtime_preset=self.runtime_preset,
            press_routes=self.press_routes,
            pick_continuations=pick_continuations,
            pour_continuations=pour_continuations,
            twist_routes=self.twist_routes,
            pick_contact_revision=self.pick_contact_revision,
            pick_contact_revisions=self.pick_contact_revisions,
            world_yaw_free_places=self.world_yaw_free_places,
            measured_rigid_calls=self.measured_rigid_calls,
            post_handover_places=self.post_handover_places,
            idle_invocations=self.idle_invocations,
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
            "upright_container_picks",
            "motion_samples",
            "motion_contract",
            "cartesian_calls",
            "pour_receivers",
            "pour_return_calls",
            "staging_calls",
            "staging_retention_sources",
            "support_contract_revision",
            "held_command_sources",
            "held_command_revision",
            "pick_contact_revision",
            "pick_contact_revisions",
            "world_yaw_free_places",
            "measured_rigid_calls",
            "post_handover_places",
            "runtime_profile",
            "experimental_open_release",
        }
        or payload["schema_version"] != "gen_sim_task_constraints/v1"
    ):
        raise ValueError(
            "Unsupported GenSim task constraint format; regenerate the bundle."
        )
    legacy_quality = payload.get("upright_container_picks", {})
    if not isinstance(legacy_quality, dict) or legacy_quality:
        raise ValueError("upright_container_picks is unsupported in this deployment.")
    presets = payload["presets"]
    motion_contract = validate_motion_contract(
        payload.get("motion_contract", SHARED_MOTION_CONTRACT)
    )
    if motion_contract == "gen_sim.sampled_velocity/v1":
        from embodichain.lab.task_program.integrations._configured_composition import (
            _resolve_task_program_components,
        )

        _, _, authored_policy = _resolve_task_program_components(
            task_program, base_dir=Path(base_dir)
        )
        if {"velocity_limit", "acceleration_limit"} & authored_policy["motion"].keys():
            raise ValueError(
                "Sampled GenSim cannot discard explicit motion limits; select lab.motion_limits/v1."
            )
    from ..experiment_policy import validate_profile, experimental_release_enabled

    validate_profile(payload.get("runtime_profile", "default"))
    if payload.get("runtime_profile") == "auto":
        raise ValueError("Bundle constraints require a concrete runtime profile.")
    if (
        "experimental_open_release" in payload
        and type(payload["experimental_open_release"]) is not bool
    ):
        raise ValueError("Bundle experimental_open_release must be a boolean.")
    release_experiment = experimental_release_enabled(
        payload.get("runtime_profile", "default"),
        payload.get("experimental_open_release"),
    )
    support_revision = payload.get("support_contract_revision")
    pick_contact_revision = payload.get("pick_contact_revision")
    if pick_contact_revision is not None and (
        type(pick_contact_revision) is not int
        or pick_contact_revision not in (1, 2, 3, 4, 5, 6, 7, 8, 9)
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

    registration = with_native_grip_evidence(
        registration,
        experimental_open_release=release_experiment,
    )
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
    from .idle_clearance import IdleFactory, bind_invocations

    idle_invocations = bind_invocations(
        compiled,
        tuple(
            f
            for f in registration.registered_semantic_lowerer_factories
            if isinstance(f, IdleFactory)
        ),
    )
    from .adaptive_grasp import bind_contact_revisions

    pick_contact_revisions = bind_contact_revisions(
        payload.get("pick_contact_revisions", {}), compiled
    )
    from .place_yaw import bind_calls as bind_yaw_calls

    world_yaw_free_places = bind_yaw_calls(
        payload.get("world_yaw_free_places", []), compiled
    )
    from .place_branch import bind_handover_places

    post_handover_places = bind_handover_places(
        payload.get("post_handover_places", []), compiled, registration.scene_binding
    )
    if world_yaw_free_places and not adaptive_pick:
        raise ValueError("World yaw freedom requires ordinary Pick/Place binding.")
    from .measured_rigid import (
        MEASURED_RIGID_REVISION,
        bind_calls as bind_measured_calls,
    )

    measured_rigid_calls = bind_measured_calls(
        payload.get("measured_rigid_calls", []), compiled, dict(pick_purposes)
    )
    if measured_rigid_calls and not adaptive_pick:
        raise ValueError("Measured rigid policy requires ordinary Pick/Place binding.")
    for identity, _, container in measured_rigid_calls:
        if not container and dict(pick_contact_revisions).get(
            identity, pick_contact_revision
        ) not in (None, 4):
            raise ValueError(
                "Measured preshape requires the qualified revision4 closing contract."
            )
    motion_samples = bind_motion_samples(payload["motion_samples"], compiled)
    cartesian_calls = bind_cartesian_calls(payload["cartesian_calls"], compiled)
    from .drawer_binding import bind_drawer_routes

    drawers = bind_drawer_routes(drawers, compiled)
    pour_return_calls = bind_pour_return_calls(
        payload.get("pour_return_calls", []), compiled
    )
    articulation_calls = bind_articulation_calls(compiled)
    articulation_followups = bind_articulation_followups(compiled)
    observed_transport_calls = bind_observed_alignment_calls(compiled, motion_contract)
    held_command_revision = payload.get("held_command_revision", 1)
    held_command_sources = bind_held_command_sources(
        payload.get("held_command_sources", {}),
        compiled,
        registration.robot_profile_binding,
        base.selection.runtime_preset,
        frozenset(item.entity_id for item in registration.scene_binding.rigid_objects),
        revision=held_command_revision,
    )
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
    from .pick_continuation import bind_calls as bind_pick_continuations
    from .place_branch import (
        PRESERVED_LATERAL_RECOVERY_REVISION,
        PRESERVED_ON_RECOVERY_REVISION,
        bind_lateral_invocations,
        bind_on_invocations,
    )

    lateral_recovery = (
        adaptive_pick
        and continuation_preset is not None
        and bool(
            bind_lateral_invocations(
                bind_pick_continuations(compiled, continuation_preset),
                pick_purposes,
                registration.scene_binding,
            )
        )
    )
    on_recovery = (
        adaptive_pick
        and continuation_preset is not None
        and bool(
            bind_on_invocations(
                bind_pick_continuations(compiled, continuation_preset),
                pick_purposes,
                registration.scene_binding,
            )
        )
    )
    grasp_factories = base.integration.grasp_factories
    coordinated = any(
        item.get("steps", {}).get("call", {}).get("call_id")
        in {"simulation.coordinated_hold", "simulation.coordinated_transport"}
        for item in program["program"]["items"]
    )
    handover_scene_clearance = any(
        item.get("steps", {}).get("call", {}).get("kind") == "hand_over"
        for item in program["program"]["items"]
    )
    if handover_scene_clearance:
        from .adaptive_grasp import HANDOVER_END_REGION_REVISION
        from .handover_safety import HANDOVER_SCENE_CLEARANCE_REVISION
        from .handover_exchange import HANDOVER_EXCHANGE_REVISION

    pour_receivers = bind_pour_receivers(payload["pour_receivers"], compiled)
    from .articulation_recovery import ARTICULATION_RECOVERY_REVISION
    from .e6_preshape import E6_PRESHAPE_REVISION
    from .park_recovery import PARK_RECOVERY_REVISION
    from .pick_safety import PICK_OPENING_RECOVERY_REVISION
    from .services import _PickLowererFactory

    if drawers:
        from .drawer_runtime import DRAWER_RELEASE_RECOVERY_REVISION

    fingerprint = canonical_hash(
        {
            "adaptive_grasp_revision": ADAPTIVE_GRASP_REVISION,
            "invocation_policy_revision": INVOCATION_POLICY_REVISION,
            **(
                {
                    "preserved_lateral_recovery_revision": PRESERVED_LATERAL_RECOVERY_REVISION
                }
                if lateral_recovery
                else {}
            ),
            **(
                {"preserved_on_recovery_revision": PRESERVED_ON_RECOVERY_REVISION}
                if on_recovery
                else {}
            ),
            **(
                {"measured_rigid_revision": MEASURED_RIGID_REVISION}
                if measured_rigid_calls
                else {}
            ),
            **(
                {
                    "handover_scene_clearance_revision": HANDOVER_SCENE_CLEARANCE_REVISION,
                    "handover_exchange_revision": HANDOVER_EXCHANGE_REVISION,
                    "handover_end_region_revision": HANDOVER_END_REGION_REVISION,
                }
                if handover_scene_clearance
                else {}
            ),
            **(
                {"held_command_revision": held_command_revision}
                if payload.get("held_command_sources")
                else {}
            ),
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
                {
                    "articulation_recovery_revision": ARTICULATION_RECOVERY_REVISION,
                    "e6_preshape_revision": E6_PRESHAPE_REVISION,
                }
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
            "observed_transport_calls": observed_transport_calls,
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
            **(
                {"drawer_release_recovery_revision": DRAWER_RELEASE_RECOVERY_REVISION}
                if drawers
                else {}
            ),
            "core_integration": base.integration.integration_fingerprint,
            "registration": bundle_registration_fingerprint(registration),
            "task_constraints": payload,
            "program": program,
            "cartesian_calls": cartesian_calls,
            "pour_receivers": pour_receivers,
            **(
                {"pour_return_policy_revision": POUR_RETURN_POLICY_REVISION}
                if pour_return_calls
                else {}
            ),
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
        handover_scene_clearance=handover_scene_clearance,
        motion_samples=motion_samples,
        motion_contract=motion_contract,
        observed_transport_calls=observed_transport_calls,
        cartesian_calls=cartesian_calls,
        articulation_calls=articulation_calls,
        articulation_followups=articulation_followups,
        pour_return_calls=pour_return_calls,
        clearance_routes=clearance_routes,
        alignment_continuation_routes=alignment_continuation_routes,
        staging_contract=staging_contract,
        held_command_sources=tuple(held_command_sources.items()),
        runtime_preset=base.selection.runtime_preset,
        press_routes=press_routes,
        twist_routes=twist_routes,
        support_contract_revision=support_revision,
        pick_contact_revision=pick_contact_revision,
        pick_contact_revisions=pick_contact_revisions,
        world_yaw_free_places=world_yaw_free_places,
        measured_rigid_calls=measured_rigid_calls,
        post_handover_places=post_handover_places,
        idle_invocations=idle_invocations,
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
