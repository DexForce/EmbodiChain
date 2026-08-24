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

"""Production simulation assembly for Gym-backed Expert Programs.

This module owns the reusable live wiring between declarative simulation
bindings and :class:`ExpertProgramEnvironmentAdapter`.  A task declares a
scene binding and a robot profile binding; this factory constructs the motion
generator, atomic-action engine, planning observation port, effect-evidence
providers, and segment-policy port without task-local motion code.

The resulting runtime is intentionally Gym-only.  Its buffered command sink
must remain attached to :class:`AtomicDemoBridge`, which advances the shared
clock only after an ordinary ``env.step()`` consumes a yielded command.  It is
therefore not a ``SkillRuntimeProvider`` for synchronous ``AtomicSkills`` use.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
import math
from typing import Protocol, TYPE_CHECKING

import torch

from embodichain.lab.gym.envs.settling import DynamicSettleMonitorCfg
from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    EntityState,
    ObservedArticulationJointState,
    PlanningContext,
    RobotObservation,
    SceneProvider,
    SceneSnapshot,
    TaskState,
)
from embodichain.lab.sim.atomic_actions.runner import ExecutionRunnerCfg
from embodichain.lab.sim.planners import (
    BasePlannerCfg,
    MotionGenCfg,
    MotionGenerator,
    ToppraPlannerCfg,
)
from embodichain.lab.sim.skills.calls import SemanticCallCatalog
from embodichain.lab.sim.skills.compiler import (
    HandOverPoseProvider,
    RegisteredSemanticLowerer,
    RelationTargetGrounder,
)
from embodichain.lab.sim.skills.effects import EffectMonitorRegistry
from embodichain.lab.sim.skills.evidence import (
    BinaryObservationCallback,
    ControlPartRobotEvidenceSource,
    ControlPartSimulationEvidenceProvider,
    EffectEvidenceProvider,
    ScalarObservationCallback,
    SceneArticulationEvidenceProvider,
)
from embodichain.lab.sim.skills.parallel_runtime import (
    ParallelCommandSafetyValidator,
)
from embodichain.lab.sim.skills.profiles import (
    ResourceEndpoint,
    ResourceEndpointAdapter,
    RobotSkillProfile,
)
from embodichain.lab.sim.skills.scene import RegistrySceneProvider, SceneRegistry

from .bridge import (
    EnvironmentStepClock,
    GymPlanningObservationProvider,
    RuntimeTransportActionEncoder,
)
from .environment import (
    ExpertProgramEnvironmentAdapter,
    ExpertProgramEnvironmentFactory,
    PlanningObservationPort,
)
from .simulation import (
    SimulationRobotSkillProfileBinding,
    SimulationSceneBinding,
)
from .simulation_policies import SimulationSegmentPolicyPort

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.sim_manager import SimulationManager
    from embodichain.toolkits.graspkit import GraspPoseGenerator


MotionGeneratorFactory = Callable[[], MotionGenerator]
"""Zero-argument factory that must return one fresh motion generator."""


class SimulationExpertProgramEnvironment(Protocol):
    """Minimal Gym environment surface used by the simulation factory."""

    sim: SimulationManager
    robot: Robot

    @property
    def step_dt(self) -> float:
        """Return the authoritative Gym control cadence in seconds."""


def _positive_finite(value: float, *, field_name: str) -> float:
    """Validate one positive finite real number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a real number.")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{field_name} must be finite and positive.")
    return normalized


def _non_negative_finite(value: float, *, field_name: str) -> float:
    """Validate one non-negative finite real number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a real number.")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{field_name} must be finite and non-negative.")
    return normalized


def _robot_uid(robot: Robot) -> str:
    """Return one strict live robot UID."""
    uid = getattr(robot, "uid", None)
    if type(uid) is not str or not uid or uid != uid.strip():
        raise ValueError(
            "robot.uid must be a non-empty string without outer whitespace."
        )
    return uid


def _full_robot_tensor(
    robot: Robot,
    getter_name: str,
    *,
    required: bool,
    reference: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Read and validate one full-robot floating state tensor."""
    getter = getattr(robot, getter_name, None)
    if not callable(getter):
        if required:
            raise TypeError(f"robot must provide {getter_name}().")
        return None
    value = getter()
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"robot.{getter_name}() must return a torch.Tensor.")
    if not value.is_floating_point() or value.dim() != 2:
        raise ValueError(
            f"robot.{getter_name}() must return floating shape (B, robot_dof)."
        )
    if value.shape[0] == 0 or value.shape[1] == 0:
        raise ValueError(f"robot.{getter_name}() dimensions must be non-zero.")
    if reference is not None and (
        value.shape != reference.shape or value.device != reference.device
    ):
        raise ValueError(
            f"robot.{getter_name}() must match robot.get_qpos() shape and device."
        )
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"robot.{getter_name}() must contain only finite values.")
    return value.clone()


class SharedTickSceneProvider(SceneProvider):
    """Share one immutable scene snapshot across consumers in the same tick.

    ``RegistrySceneProvider`` is stateful: every call observes native entities
    and updates material-change baselines.  Planning observations and multiple
    evidence providers can legitimately request the same timestamp.  This
    wrapper always delegates one full-batch request per tick, then returns the
    exact snapshot or an owned ordered-row projection to later consumers.
    """

    def __init__(
        self,
        delegate: RegistrySceneProvider,
        full_env_ids: torch.Tensor,
    ) -> None:
        if type(delegate) is not RegistrySceneProvider:
            raise TypeError("delegate must be exactly RegistrySceneProvider.")
        if (
            not isinstance(full_env_ids, torch.Tensor)
            or full_env_ids.dtype != torch.long
            or full_env_ids.dim() != 1
            or full_env_ids.numel() == 0
        ):
            raise ValueError("full_env_ids must be a non-empty 1D int64 tensor.")
        if torch.unique(full_env_ids).numel() != full_env_ids.numel():
            raise ValueError("full_env_ids must be unique.")
        self._delegate = delegate
        self._full_env_ids = full_env_ids.clone()
        self._row_by_env_id = {
            int(env_id): row
            for row, env_id in enumerate(full_env_ids.detach().cpu().tolist())
        }
        self._timestamp: float | None = None
        self._snapshot: SceneSnapshot | None = None

    @property
    def collision_entity_ids(self) -> tuple[str, ...]:
        """Return canonical dynamic collision IDs from the delegate."""
        return self._delegate.collision_entity_ids

    def snapshot(
        self,
        *,
        timestamp: float,
        env_ids: torch.Tensor,
    ) -> SceneSnapshot:
        """Return the single shared snapshot for ``timestamp`` and ``env_ids``."""
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
            raise TypeError("timestamp must be a real number.")
        normalized_timestamp = float(timestamp)
        if not math.isfinite(normalized_timestamp) or normalized_timestamp < 0.0:
            raise ValueError("timestamp must be finite and non-negative.")
        if (
            not isinstance(env_ids, torch.Tensor)
            or env_ids.dtype != torch.long
            or env_ids.dim() != 1
            or env_ids.numel() == 0
        ):
            raise ValueError("env_ids must be a non-empty 1D int64 tensor.")
        if torch.unique(env_ids).numel() != env_ids.numel():
            raise ValueError("env_ids must be unique.")
        if env_ids.device != self._full_env_ids.device:
            raise ValueError("env_ids must share the full simulation batch device.")
        try:
            rows = tuple(
                self._row_by_env_id[int(env_id)]
                for env_id in env_ids.detach().cpu().tolist()
            )
        except KeyError as exc:
            raise ValueError(
                f"Environment ID {int(exc.args[0])} is absent from full_env_ids."
            ) from exc

        if self._timestamp is not None:
            if normalized_timestamp < self._timestamp:
                raise ValueError("Shared scene snapshot timestamps must be monotonic.")
            if normalized_timestamp == self._timestamp:
                assert self._snapshot is not None
                return self._select_rows(self._snapshot, rows)

        snapshot = self._delegate.snapshot(
            timestamp=normalized_timestamp,
            env_ids=self._full_env_ids.clone(),
        )
        if not isinstance(snapshot, SceneSnapshot):
            raise TypeError(
                "RegistrySceneProvider.snapshot() must return SceneSnapshot."
            )
        if snapshot.timestamp != normalized_timestamp:
            raise ValueError("Scene snapshot timestamp must match the requested tick.")
        self._timestamp = normalized_timestamp
        self._snapshot = snapshot
        return self._select_rows(snapshot, rows)

    def _select_rows(
        self,
        snapshot: SceneSnapshot,
        rows: tuple[int, ...],
    ) -> SceneSnapshot:
        """Project one cached full-batch snapshot to an ordered row subset."""
        full_size = int(self._full_env_ids.numel())
        if rows == tuple(range(full_size)):
            return snapshot
        entities: dict[str, EntityState] = {}
        for entity_id, state in snapshot.entities.items():
            pose = state.pose
            if pose.dim() == 3:
                if pose.shape[0] != full_size:
                    raise ValueError(
                        f"Scene entity {entity_id!r} batch does not match "
                        "full_env_ids."
                    )
                index = torch.tensor(rows, dtype=torch.long, device=pose.device)
                pose = pose.index_select(0, index)
            entities[entity_id] = EntityState(pose, confidence=state.confidence)

        articulation_joints: dict[tuple[str, str], ObservedArticulationJointState] = {}
        for address, state in snapshot.articulation_joints.items():
            position = state.position
            valid = state.valid_mask
            if position.dim() == 2:
                if position.shape[0] != full_size:
                    raise ValueError(
                        f"Scene articulation joint {address!r} batch does not "
                        "match full_env_ids."
                    )
                index = torch.tensor(rows, dtype=torch.long, device=position.device)
                position = position.index_select(0, index)
                if valid is not None:
                    valid = valid.index_select(0, index.to(valid.device))
            articulation_joints[address] = ObservedArticulationJointState(
                position,
                valid,
            )

        revisions = snapshot.collision_world_revisions(full_size)
        return SceneSnapshot(
            timestamp=snapshot.timestamp,
            version=snapshot.version,
            entities=entities,
            collision_world_revision=tuple(revisions[row] for row in rows),
            collision_entity_ids=snapshot.collision_entity_ids,
            articulation_joints=articulation_joints,
        )


class SimulationPlanningObservationProvider(GymPlanningObservationProvider):
    """Gym planning observations backed by live robot and shared scene state."""

    def __init__(
        self,
        robot: Robot,
        scene_provider: SharedTickSceneProvider,
        clock: EnvironmentStepClock,
        env_ids: torch.Tensor,
        *,
        owner_token: object,
    ) -> None:
        if type(scene_provider) is not SharedTickSceneProvider:
            raise TypeError("scene_provider must be exactly SharedTickSceneProvider.")
        if type(clock) is not EnvironmentStepClock:
            raise TypeError("clock must be exactly EnvironmentStepClock.")
        qpos = _full_robot_tensor(robot, "get_qpos", required=True)
        assert qpos is not None
        if (
            not isinstance(env_ids, torch.Tensor)
            or env_ids.dtype != torch.long
            or env_ids.shape != (qpos.shape[0],)
        ):
            raise ValueError("env_ids must be int64 with one ID per robot row.")
        if env_ids.device != qpos.device:
            raise ValueError("env_ids and robot qpos must share a device.")
        if torch.unique(env_ids).numel() != env_ids.numel():
            raise ValueError("env_ids must be unique.")
        self._robot = robot
        self._scene_provider = scene_provider
        self._clock = clock
        self._env_ids = env_ids.clone()
        self._owner_token = owner_token
        super().__init__(self._capture)

    @property
    def scene_provider(self) -> SharedTickSceneProvider:
        """Return the snapshot-sharing scene provider used by evidence ports."""
        return self._scene_provider

    @property
    def env_ids(self) -> torch.Tensor:
        """Return stable ordered simulation row IDs."""
        return self._env_ids.clone()

    def is_owned_by(self, owner_token: object) -> bool:
        """Return whether this provider belongs to one factory instance."""
        return self._owner_token is owner_token

    def _capture(self, task_state: TaskState) -> PlanningContext:
        """Capture one synchronized robot and scene observation."""
        qpos = _full_robot_tensor(self._robot, "get_qpos", required=True)
        assert qpos is not None
        if (
            qpos.shape[0] != self._env_ids.numel()
            or qpos.device != self._env_ids.device
        ):
            raise ValueError("Robot batch shape or device changed after assembly.")
        qvel = _full_robot_tensor(
            self._robot,
            "get_qvel",
            required=False,
            reference=qpos,
        )
        if qvel is None:
            qvel = torch.zeros_like(qpos)
        qeffort = _full_robot_tensor(
            self._robot,
            "get_qf",
            required=False,
            reference=qpos,
        )
        timestamp = self._clock.now()
        scene = self._scene_provider.snapshot(
            timestamp=timestamp,
            env_ids=self._env_ids.clone(),
        )
        return PlanningContext(
            robot=RobotObservation(
                timestamp=timestamp,
                qpos=qpos,
                qvel=qvel,
                qeffort=qeffort,
            ),
            task=task_state,
            scene=scene,
            env_ids=self._env_ids,
            control_dt=self._clock.step_dt,
        )


class SimulationExpertProgramFactory(ExpertProgramEnvironmentFactory):
    """Build every live Expert Program component from explicit declarations.

    Args:
        simulation: Exact live simulation that owns ``robot`` and scene UIDs.
        robot: Exact robot selected for planning and evidence acquisition.
        scene_binding: Canonical-to-native scene declaration.
        robot_profile_binding: Typed robot resource and policy declaration.
        step_dt: Authoritative Gym control cadence.
        planner_cfg: Explicit planner configuration.  ``None`` selects TOPPRA
            for ``robot.uid``.
        motion_generator_factory: Optional fresh-generator factory.  It is
            mutually exclusive with ``planner_cfg`` and intended for custom
            planners and isolated tests.
        grasp_pose_generators: Standalone grasp-pose services keyed by grasp
            endpoint target ID, normally a robot control-part name.
        endpoint_adapters: Explicit adapters for non-built-in resource endpoint
            types.
        settle_presets: Optional named segment settling policies.
        translation_threshold: Material scene translation threshold.
        rotation_threshold: Material scene rotation threshold.
        contact_observer: Optional raw contact evidence callback.
        constraint_observer: Optional raw constraint evidence callback.
        force_observer: Optional raw force evidence callback.
        wrench_observer: Optional raw wrench evidence callback.

    The Gym cadence is attached to each live :class:`PlanningContext`.  Motion
    policy presets describe behavior; they do not own the environment's
    command period.
    """

    def __init__(
        self,
        simulation: SimulationManager,
        robot: Robot,
        scene_binding: SimulationSceneBinding,
        robot_profile_binding: SimulationRobotSkillProfileBinding,
        *,
        step_dt: float,
        planner_cfg: BasePlannerCfg | None = None,
        motion_generator_factory: MotionGeneratorFactory | None = None,
        grasp_pose_generators: Mapping[str, GraspPoseGenerator] | None = None,
        endpoint_adapters: (
            Mapping[type[ResourceEndpoint], ResourceEndpointAdapter] | None
        ) = None,
        settle_presets: Mapping[str, DynamicSettleMonitorCfg] | None = None,
        translation_threshold: float = 1.0e-4,
        rotation_threshold: float = 1.0e-3,
        contact_observer: BinaryObservationCallback | None = None,
        constraint_observer: BinaryObservationCallback | None = None,
        force_observer: ScalarObservationCallback | None = None,
        wrench_observer: ScalarObservationCallback | None = None,
    ) -> None:
        if type(scene_binding) is not SimulationSceneBinding:
            raise TypeError("scene_binding must be exactly SimulationSceneBinding.")
        if type(robot_profile_binding) is not SimulationRobotSkillProfileBinding:
            raise TypeError(
                "robot_profile_binding must be exactly "
                "SimulationRobotSkillProfileBinding."
            )
        if planner_cfg is not None and motion_generator_factory is not None:
            raise ValueError(
                "planner_cfg and motion_generator_factory are mutually exclusive."
            )
        if planner_cfg is not None and not isinstance(planner_cfg, BasePlannerCfg):
            raise TypeError("planner_cfg must be a BasePlannerCfg or None.")
        if motion_generator_factory is not None and not callable(
            motion_generator_factory
        ):
            raise TypeError("motion_generator_factory must be callable or None.")
        if grasp_pose_generators is not None and not isinstance(
            grasp_pose_generators, Mapping
        ):
            raise TypeError("grasp_pose_generators must be a mapping or None.")
        if endpoint_adapters is not None and not isinstance(endpoint_adapters, Mapping):
            raise TypeError("endpoint_adapters must be a mapping or None.")
        for name, callback in (
            ("contact_observer", contact_observer),
            ("constraint_observer", constraint_observer),
            ("force_observer", force_observer),
            ("wrench_observer", wrench_observer),
        ):
            if callback is not None and not callable(callback):
                raise TypeError(f"{name} must be callable or None.")

        robot_uid = _robot_uid(robot)
        get_robot = getattr(simulation, "get_robot", None)
        if not callable(get_robot):
            raise TypeError("simulation must provide get_robot().")
        if get_robot(robot_uid) is not robot:
            raise ValueError(
                f"simulation.get_robot({robot_uid!r}) must return the exact "
                "selected robot."
            )
        selected_planner_cfg = deepcopy(planner_cfg)
        if (
            selected_planner_cfg is not None
            and selected_planner_cfg.robot_uid != robot_uid
        ):
            raise ValueError(
                f"planner_cfg.robot_uid must equal selected robot UID {robot_uid!r}."
            )

        self._simulation = simulation
        self._robot = robot
        self._scene_binding = scene_binding
        self._robot_profile_binding = robot_profile_binding
        self._step_dt = _positive_finite(step_dt, field_name="step_dt")
        self._planner_cfg = selected_planner_cfg
        self._motion_generator_factory = motion_generator_factory
        self._grasp_pose_generators = (
            {} if grasp_pose_generators is None else dict(grasp_pose_generators)
        )
        self._endpoint_adapters = (
            None if endpoint_adapters is None else dict(endpoint_adapters)
        )
        self._translation_threshold = _non_negative_finite(
            translation_threshold,
            field_name="translation_threshold",
        )
        self._rotation_threshold = _non_negative_finite(
            rotation_threshold,
            field_name="rotation_threshold",
        )
        self._contact_observer = contact_observer
        self._constraint_observer = constraint_observer
        self._force_observer = force_observer
        self._wrench_observer = wrench_observer
        self._owner_token = object()

        qpos = _full_robot_tensor(robot, "get_qpos", required=True)
        assert qpos is not None
        self._env_ids = torch.arange(
            qpos.shape[0],
            dtype=torch.long,
            device=qpos.device,
        )
        self._segment_policy_port = SimulationSegmentPolicyPort(
            simulation,
            robot,
            scene_binding,
            settle_presets=settle_presets,
            env_ids=self._env_ids,
        )

    @classmethod
    def from_environment(
        cls,
        environment: SimulationExpertProgramEnvironment,
        *,
        scene_binding: SimulationSceneBinding,
        robot_profile_binding: SimulationRobotSkillProfileBinding,
        planner_cfg: BasePlannerCfg | None = None,
        motion_generator_factory: MotionGeneratorFactory | None = None,
        grasp_pose_generators: Mapping[str, GraspPoseGenerator] | None = None,
        endpoint_adapters: (
            Mapping[type[ResourceEndpoint], ResourceEndpointAdapter] | None
        ) = None,
        settle_presets: Mapping[str, DynamicSettleMonitorCfg] | None = None,
        translation_threshold: float = 1.0e-4,
        rotation_threshold: float = 1.0e-3,
        contact_observer: BinaryObservationCallback | None = None,
        constraint_observer: BinaryObservationCallback | None = None,
        force_observer: ScalarObservationCallback | None = None,
        wrench_observer: ScalarObservationCallback | None = None,
    ) -> SimulationExpertProgramFactory:
        """Create a factory from the explicit standard Gym environment surface."""
        simulation = getattr(environment, "sim", None)
        robot = getattr(environment, "robot", None)
        try:
            step_dt = environment.step_dt
        except AttributeError as exc:
            raise TypeError("environment must expose step_dt.") from exc
        if simulation is None or robot is None:
            raise TypeError("environment must expose non-None sim and robot values.")
        return cls(
            simulation,
            robot,
            scene_binding,
            robot_profile_binding,
            step_dt=step_dt,
            planner_cfg=planner_cfg,
            motion_generator_factory=motion_generator_factory,
            grasp_pose_generators=grasp_pose_generators,
            endpoint_adapters=endpoint_adapters,
            settle_presets=settle_presets,
            translation_threshold=translation_threshold,
            rotation_threshold=rotation_threshold,
            contact_observer=contact_observer,
            constraint_observer=constraint_observer,
            force_observer=force_observer,
            wrench_observer=wrench_observer,
        )

    @property
    def scene_registry_id(self) -> str:
        """Return the exact configured scene-registry ID."""
        return self._scene_binding.registry_id

    @property
    def robot_profile_id(self) -> str:
        """Return the exact configured robot-profile ID."""
        return self._robot_profile_binding.profile_id

    @property
    def step_dt(self) -> float:
        """Return the authoritative Gym control cadence."""
        return self._step_dt

    @property
    def segment_policy_port(self) -> SimulationSegmentPolicyPort:
        """Return the shared simulation post-policy and validator port."""
        return self._segment_policy_port

    def create_scene_registry(self) -> SceneRegistry:
        """Build one fresh authoritative registry from explicit bindings."""
        return self._scene_binding.build(self._simulation)

    def create_robot_skill_profile(self) -> RobotSkillProfile:
        """Build the robot's declared semantic-skill profile."""
        return self._robot_profile_binding.build(self._robot)

    def create_atomic_action_engine(
        self,
        profile: RobotSkillProfile,
    ) -> AtomicActionEngine:
        """Create a fresh engine around the selected planner and exact profile."""
        if not isinstance(profile, RobotSkillProfile):
            raise TypeError("profile must be a RobotSkillProfile.")
        if profile.profile_id != self.robot_profile_id:
            raise ValueError(
                f"profile ID must be {self.robot_profile_id!r}, got "
                f"{profile.profile_id!r}."
            )
        motion_generator = self._create_motion_generator()
        if motion_generator.robot is not self._robot:
            raise ValueError(
                "Motion generator must own the exact robot selected by the factory."
            )
        return AtomicActionEngine(
            motion_generator,
            grasp_pose_generators=self._grasp_pose_generators,
            skill_profile=profile,
            endpoint_adapters=self._endpoint_adapters,
        )

    def create_planning_observation_provider(
        self,
        *,
        scene_registry: SceneRegistry,
        engine: AtomicActionEngine,
        clock: EnvironmentStepClock,
    ) -> PlanningObservationPort:
        """Create one planning port and planner-validated shared scene provider."""
        if type(scene_registry) is not SceneRegistry:
            raise TypeError("scene_registry must be exactly SceneRegistry.")
        if not isinstance(engine, AtomicActionEngine):
            raise TypeError("engine must be an AtomicActionEngine.")
        if engine.robot is not self._robot:
            raise ValueError("engine must own the exact factory robot.")
        if type(clock) is not EnvironmentStepClock:
            raise TypeError("clock must be exactly EnvironmentStepClock.")
        if clock.step_dt != self._step_dt:
            raise ValueError("clock.step_dt must equal the factory Gym cadence.")
        provider = scene_registry.make_planning_scene_provider(
            engine.motion_generator,
            batch_size=int(self._env_ids.numel()),
            translation_threshold=self._translation_threshold,
            rotation_threshold=self._rotation_threshold,
        )
        shared = SharedTickSceneProvider(provider, self._env_ids)
        return SimulationPlanningObservationProvider(
            self._robot,
            shared,
            clock,
            self._env_ids,
            owner_token=self._owner_token,
        )

    def create_effect_evidence_providers(
        self,
        *,
        scene_registry: SceneRegistry,
        engine: AtomicActionEngine,
        observation_provider: PlanningObservationPort,
    ) -> Iterable[EffectEvidenceProvider]:
        """Create built-in control-part and articulation evidence providers."""
        if type(scene_registry) is not SceneRegistry:
            raise TypeError("scene_registry must be exactly SceneRegistry.")
        if not isinstance(engine, AtomicActionEngine):
            raise TypeError("engine must be an AtomicActionEngine.")
        if engine.robot is not self._robot:
            raise ValueError("engine must own the exact factory robot.")
        if type(observation_provider) is not SimulationPlanningObservationProvider:
            raise TypeError(
                "observation_provider must be exactly "
                "SimulationPlanningObservationProvider."
            )
        if not observation_provider.is_owned_by(self._owner_token):
            raise ValueError("observation_provider belongs to another factory.")
        scene_provider = observation_provider.scene_provider
        providers: list[EffectEvidenceProvider] = []
        if isinstance(self._robot, ControlPartRobotEvidenceSource):
            providers.append(
                ControlPartSimulationEvidenceProvider(
                    self._robot,
                    scene_provider=scene_provider,
                    contact_observer=self._contact_observer,
                    constraint_observer=self._constraint_observer,
                    force_observer=self._force_observer,
                    wrench_observer=self._wrench_observer,
                )
            )
        providers.append(
            SceneArticulationEvidenceProvider(scene_provider=scene_provider)
        )
        return tuple(providers)

    def create_adapter(
        self,
        *,
        call_catalog: SemanticCallCatalog | None = None,
        registered_lowerers: Iterable[RegisteredSemanticLowerer] = (),
        relation_grounders: Iterable[RelationTargetGrounder] = (),
        handover_pose_providers: Iterable[HandOverPoseProvider] = (),
        effect_monitor_registry: EffectMonitorRegistry | None = None,
        runtime_transports: Iterable[RuntimeTransportActionEncoder] = (),
        runner_cfg: ExecutionRunnerCfg | None = None,
        parallel_safety_validator: ParallelCommandSafetyValidator | None = None,
    ) -> ExpertProgramEnvironmentAdapter:
        """Create the exact Gym adapter with shared simulation policy ports."""
        return ExpertProgramEnvironmentAdapter(
            self,
            step_dt=self._step_dt,
            call_catalog=call_catalog,
            endpoint_adapters=self._endpoint_adapters,
            registered_lowerers=registered_lowerers,
            relation_grounders=relation_grounders,
            handover_pose_providers=handover_pose_providers,
            effect_monitor_registry=effect_monitor_registry,
            runtime_transports=runtime_transports,
            runner_cfg=runner_cfg,
            post_policy_port=self._segment_policy_port,
            validator_port=self._segment_policy_port,
            parallel_safety_validator=parallel_safety_validator,
        )

    def _create_motion_generator(self) -> MotionGenerator:
        """Create and validate one exact motion generator."""
        if self._motion_generator_factory is not None:
            generator = self._motion_generator_factory()
        else:
            planner_cfg = (
                ToppraPlannerCfg(robot_uid=_robot_uid(self._robot))
                if self._planner_cfg is None
                else deepcopy(self._planner_cfg)
            )
            generator = MotionGenerator(MotionGenCfg(planner_cfg=planner_cfg))
        if not isinstance(generator, MotionGenerator):
            raise TypeError(
                "motion_generator_factory must return a MotionGenerator instance."
            )
        return generator


def create_simulation_expert_program_adapter(
    environment: SimulationExpertProgramEnvironment,
    *,
    scene_binding: SimulationSceneBinding,
    robot_profile_binding: SimulationRobotSkillProfileBinding,
    planner_cfg: BasePlannerCfg | None = None,
    motion_generator_factory: MotionGeneratorFactory | None = None,
    grasp_pose_generators: Mapping[str, GraspPoseGenerator] | None = None,
    endpoint_adapters: (
        Mapping[type[ResourceEndpoint], ResourceEndpointAdapter] | None
    ) = None,
    relation_grounders: Iterable[RelationTargetGrounder] = (),
    handover_pose_providers: Iterable[HandOverPoseProvider] = (),
    runtime_transports: Iterable[RuntimeTransportActionEncoder] = (),
    settle_presets: Mapping[str, DynamicSettleMonitorCfg] | None = None,
    translation_threshold: float = 1.0e-4,
    rotation_threshold: float = 1.0e-3,
    contact_observer: BinaryObservationCallback | None = None,
    constraint_observer: BinaryObservationCallback | None = None,
    force_observer: ScalarObservationCallback | None = None,
    wrench_observer: ScalarObservationCallback | None = None,
    parallel_safety_validator: ParallelCommandSafetyValidator | None = None,
) -> ExpertProgramEnvironmentAdapter:
    """Create a complete production adapter from one standard Gym environment.

    This is the intended task-side one-line integration. Relation-target
    grounders and embodiment-owned handover pose providers are explicit and
    default to empty collections, so calls that require an uninstalled provider
    remain fail-closed during program preflight. Advanced callers can retain
    :class:`SimulationExpertProgramFactory` and call ``create_adapter`` directly
    to install registered semantic lowerers or custom monitors. Custom endpoint
    adapters and their matching Gym runtime transports are accepted here so a
    non-joint endpoint remains executable through the one-line path.

    Args:
        environment: Standard Gym simulation environment exposing ``sim``,
            ``robot``, and ``step_dt``.
        scene_binding: Authoritative typed scene declaration.
        robot_profile_binding: Typed robot resource and policy declaration.
        planner_cfg: Optional planner configuration owned by the factory.
        motion_generator_factory: Optional factory for one fresh motion generator.
        grasp_pose_generators: Standalone grasp-pose services keyed by grasp
            endpoint target ID.
        endpoint_adapters: Optional exact-type custom endpoint adapters.
        relation_grounders: Explicit typed relation-target grounders.
        handover_pose_providers: Explicit embodiment-owned handover pose providers.
        runtime_transports: Additional runtime-command-to-Gym encoders.
        settle_presets: Optional named dynamic-settling policies.
        translation_threshold: Scene translation revision threshold.
        rotation_threshold: Scene rotation revision threshold.
        contact_observer: Optional raw contact evidence callback.
        constraint_observer: Optional raw constraint evidence callback.
        force_observer: Optional raw force evidence callback.
        wrench_observer: Optional raw wrench evidence callback.
        parallel_safety_validator: Optional authoritative parallel-command gate.

    Returns:
        Complete production Expert Program environment adapter.
    """
    factory = SimulationExpertProgramFactory.from_environment(
        environment,
        scene_binding=scene_binding,
        robot_profile_binding=robot_profile_binding,
        planner_cfg=planner_cfg,
        motion_generator_factory=motion_generator_factory,
        grasp_pose_generators=grasp_pose_generators,
        endpoint_adapters=endpoint_adapters,
        settle_presets=settle_presets,
        translation_threshold=translation_threshold,
        rotation_threshold=rotation_threshold,
        contact_observer=contact_observer,
        constraint_observer=constraint_observer,
        force_observer=force_observer,
        wrench_observer=wrench_observer,
    )
    return factory.create_adapter(
        relation_grounders=relation_grounders,
        handover_pose_providers=handover_pose_providers,
        runtime_transports=runtime_transports,
        parallel_safety_validator=parallel_safety_validator,
    )


__all__: list[str] = []
