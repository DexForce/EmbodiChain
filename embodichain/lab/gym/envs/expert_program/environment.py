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

"""Explicit production assembly for environment-backed Expert Programs.

The adapter in this module is deliberately strict.  It does not scan a
simulation, infer robot resources, or manufacture task semantics from naming
conventions.  An environment supplies one typed factory that owns all live
provider choices; the adapter validates those declarations and wires the
shared semantic compiler, runtime, and Gym bridge.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Protocol, runtime_checkable

from embodichain.lab.sim.atomic_actions.engine import AtomicActionEngine
from embodichain.lab.sim.atomic_actions.runner import (
    ExecutionRunnerCfg,
    ObservationProvider,
)
from embodichain.lab.sim.skills.calls import (
    SemanticCallCatalog,
    SemanticCallSpec,
    builtin_semantic_call_catalog,
)
from embodichain.lab.sim.skills.compiler import (
    HandOverPoseProvider,
    RegisteredSemanticLowerer,
    RelationTargetGrounder,
    SemanticSkillCompiler,
)
from embodichain.lab.sim.skills.effects import EffectMonitorRegistry
from embodichain.lab.sim.skills.evidence import (
    EffectEvidenceCollector,
    EffectEvidenceProvider,
    EffectEvidenceProviderRegistry,
)
from embodichain.lab.sim.skills.integration import (
    SceneManifest,
    SemanticIntegrationManifest,
)
from embodichain.lab.sim.skills.parallel_runtime import (
    ParallelCommandSafetyValidator,
    analyze_parallel_branches,
)
from embodichain.lab.sim.skills.profiles import (
    BoundRobotSkillProfile,
    ResourceEndpoint,
    ResourceEndpointAdapter,
    RobotSkillProfile,
)
from embodichain.lab.sim.skills.runtime import SkillRuntime
from embodichain.lab.sim.skills.scene import SceneRegistry

from .bridge import (
    AtomicDemoBridge,
    BufferedGymCommandSink,
    CurrentQposProvider,
    DemoBridgeError,
    EnvironmentStepClock,
    JointPositionGymTransportEncoder,
    RuntimeCommandFrameEncoder,
    RuntimeTransportActionEncoder,
    SegmentPostPolicyPort,
    SegmentValidatorPort,
)
from .catalog import (
    ExpertProgramIntegrationCatalog,
    IntegrationFingerprintMismatch,
    SimulationExpertProgramRegistration,
)
from .cfg import ExpertProgramCfg, ExpertProgramIntegrationCfg
from .compiler import (
    CompiledProgram,
    ExpertProgramCompiler,
)


def _validate_identifier(value: str, *, field_name: str) -> str:
    """Validate one stable integration identifier."""
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(
            f"{field_name} must be a non-empty string without outer whitespace."
        )
    return value


@runtime_checkable
class PlanningObservationPort(
    ObservationProvider,
    CurrentQposProvider,
    Protocol,
):
    """Combined observation and full-qpos port required by the Gym runtime."""


@runtime_checkable
class ExpertProgramEnvironmentFactory(Protocol):
    """Environment-owned factories for one explicit semantic integration.

    Implementations normally live in reusable robot/task integration modules,
    not in individual task motion planners.  Every method is passed the exact
    objects selected earlier in the assembly so a factory cannot silently bind
    a different scene, robot profile, or engine.
    """

    @property
    def scene_registry_id(self) -> str:
        """Return the configuration ID selecting this scene declaration.

        Returns:
            Stable scene-registry identifier.
        """

    @property
    def robot_profile_id(self) -> str:
        """Return the configuration ID selecting this robot profile.

        Returns:
            Stable robot-profile identifier.
        """

    def create_scene_registry(self) -> SceneRegistry:
        """Create the authoritative explicitly registered live scene.

        Returns:
            Fresh registry containing only explicitly selected entities.
        """

    def create_robot_skill_profile(self) -> RobotSkillProfile:
        """Create the authoritative declarative robot skill profile.

        Returns:
            Profile whose ID matches :attr:`robot_profile_id`.
        """

    def create_atomic_action_engine(
        self,
        profile: RobotSkillProfile,
    ) -> AtomicActionEngine:
        """Create an engine for exactly ``profile`` and its motion backend.

        Args:
            profile: Profile selected and validated by the adapter.

        Returns:
            Atomic engine connected to the environment's robot and planner.
        """

    def create_planning_observation_provider(
        self,
        *,
        scene_registry: SceneRegistry,
        engine: AtomicActionEngine,
        clock: EnvironmentStepClock,
    ) -> PlanningObservationPort:
        """Create fresh planning observations and aligned full-qpos reads.

        Args:
            scene_registry: Exact registry selected for this runtime.
            engine: Exact atomic engine selected for this runtime.
            clock: Shared environment-step execution clock.

        Returns:
            Combined planning-observation and qpos provider.
        """

    def create_effect_evidence_providers(
        self,
        *,
        scene_registry: SceneRegistry,
        engine: AtomicActionEngine,
        observation_provider: PlanningObservationPort,
    ) -> Iterable[EffectEvidenceProvider]:
        """Create exact-version providers used by semantic effect monitors.

        Args:
            scene_registry: Exact registry selected for this runtime.
            engine: Exact atomic engine selected for this runtime.
            observation_provider: Shared planning observation provider.

        Returns:
            Explicit provider set; an empty iterable is permitted.
        """


@runtime_checkable
class ParallelCommandSafetyValidatorProvider(Protocol):
    """Runtime-factory capability for a fresh registration-owned safety gate."""

    def create_parallel_command_safety_validator(
        self,
        *,
        scene_registry: SceneRegistry,
        engine: AtomicActionEngine,
        observation_provider: PlanningObservationPort,
    ) -> ParallelCommandSafetyValidator:
        """Create the live gate for the exact assembled runtime components."""


@runtime_checkable
class _RegistrationOwningExpertProgramFactory(Protocol):
    """Internal capability exposing one exact standard registration owner."""

    @property
    def expert_program_registration(self) -> SimulationExpertProgramRegistration:
        """Return the exact registration owned by this live factory."""

    def registration_owned_segment_policy_ports(
        self,
    ) -> tuple[SegmentPostPolicyPort | None, SegmentValidatorPort | None]:
        """Return factory-owned post-policy and validator ports."""


@dataclass(frozen=True, slots=True)
class ExpertProgramRuntimeAssembly:
    """Auditable result of one fresh environment runtime assembly.

    Attributes:
        integration: Owned integration-selection snapshot.
        scene_registry: Authoritative live scene registry.
        robot_profile: Declarative robot resource profile.
        manifest: Static scene/profile/call integration manifest.
        engine: Bound atomic action engine.
        compiler: Bound semantic skill compiler.
        observation_provider: Shared planning and full-qpos provider.
        evidence_collector: Exact-version semantic evidence collector.
        clock: Shared environment-step clock.
        command_encoder: Runtime-frame to Gym-action encoder.
        command_sink: Buffered Gym command sink.
        runner_cfg: Runner policy selected by the integration runtime preset.
        parallel_safety_validator: Optional fresh registration-owned safety gate.
        runtime: Nonblocking semantic skill runtime.
    """

    integration: ExpertProgramIntegrationCfg
    scene_registry: SceneRegistry
    robot_profile: RobotSkillProfile
    manifest: SemanticIntegrationManifest
    engine: AtomicActionEngine
    compiler: SemanticSkillCompiler
    observation_provider: PlanningObservationPort
    evidence_collector: EffectEvidenceCollector
    clock: EnvironmentStepClock
    command_encoder: RuntimeCommandFrameEncoder
    command_sink: BufferedGymCommandSink
    runner_cfg: ExecutionRunnerCfg
    parallel_safety_validator: ParallelCommandSafetyValidator | None
    runtime: SkillRuntime


@dataclass(frozen=True, slots=True)
class _ExpertProgramSemanticAssembly:
    """Observation-free semantic components prepared for program preflight."""

    integration: ExpertProgramIntegrationCfg
    scene_registry: SceneRegistry
    robot_profile: RobotSkillProfile
    manifest: SemanticIntegrationManifest
    engine: AtomicActionEngine
    compiler: SemanticSkillCompiler


class ExpertProgramEnvironmentAdapter:
    """Compile and run Expert Programs through explicit environment factories.

    Args:
        factory: Environment-owned live-provider and engine factory.
        step_dt: Authoritative Gym control cadence in seconds.
        integration_catalog: Optional immutable task-registration catalog used
            for provider-free compilation.
        registration: Optional exact standard task registration. When present,
            every compiler/runtime extension comes exclusively from it.
        call_catalog: Optional immutable semantic call catalog.  The built-in
            catalog is used when omitted.
        endpoint_adapters: Optional custom robot endpoint adapters.
        registered_lowerers: Explicit lowerers for registered semantic calls.
        relation_grounders: Explicit relation-target grounding providers.
        handover_pose_providers: Explicit embodiment hand-over providers.
        effect_monitor_registry: Optional exact-version monitor registry.
        runtime_transports: Additional runtime-command-to-Gym encoders.
        runner_cfg: Optional execution-runner policy.
        post_policy_port: Optional environment post-policy executor.
        validator_port: Optional environment segment validator.
        parallel_safety_validator: Optional authoritative parallel safety gate.

    A call to :meth:`compile` snapshots only scene identities.  A call to
    :meth:`assemble_runtime` creates a fresh live runtime, which makes reset and
    episode ownership explicit and avoids retaining providers in compiled data.
    """

    def __init__(
        self,
        factory: ExpertProgramEnvironmentFactory,
        *,
        step_dt: float,
        integration_catalog: ExpertProgramIntegrationCatalog | None = None,
        registration: SimulationExpertProgramRegistration | None = None,
        call_catalog: SemanticCallCatalog | None = None,
        endpoint_adapters: (
            Mapping[type[ResourceEndpoint], ResourceEndpointAdapter] | None
        ) = None,
        registered_lowerers: Iterable[RegisteredSemanticLowerer] = (),
        relation_grounders: Iterable[RelationTargetGrounder] = (),
        handover_pose_providers: Iterable[HandOverPoseProvider] = (),
        effect_monitor_registry: EffectMonitorRegistry | None = None,
        runtime_transports: Iterable[RuntimeTransportActionEncoder] = (),
        runner_cfg: ExecutionRunnerCfg | None = None,
        post_policy_port: SegmentPostPolicyPort | None = None,
        validator_port: SegmentValidatorPort | None = None,
        parallel_safety_validator: ParallelCommandSafetyValidator | None = None,
    ) -> None:
        if not isinstance(factory, ExpertProgramEnvironmentFactory):
            raise TypeError("factory must implement ExpertProgramEnvironmentFactory.")
        if not isinstance(step_dt, (int, float)) or isinstance(step_dt, bool):
            raise TypeError("step_dt must be a real number.")
        if not math.isfinite(float(step_dt)) or float(step_dt) <= 0.0:
            raise ValueError("step_dt must be finite and positive.")
        scene_registry_id = _validate_identifier(
            factory.scene_registry_id,
            field_name="factory.scene_registry_id",
        )
        robot_profile_id = _validate_identifier(
            factory.robot_profile_id,
            field_name="factory.robot_profile_id",
        )
        if (
            integration_catalog is not None
            and type(integration_catalog) is not ExpertProgramIntegrationCatalog
        ):
            raise TypeError(
                "integration_catalog must be exactly "
                "ExpertProgramIntegrationCatalog or None."
            )
        if (
            registration is not None
            and type(registration) is not SimulationExpertProgramRegistration
        ):
            raise TypeError(
                "registration must be exactly "
                "SimulationExpertProgramRegistration or None."
            )
        registration_owner = None
        if registration is not None:
            if not isinstance(factory, _RegistrationOwningExpertProgramFactory):
                raise TypeError(
                    "registration requires a factory that exposes exact "
                    "registration ownership and factory-owned segment policy ports."
                )
            registration_owner = factory
            owned_registration = registration_owner.expert_program_registration
            if type(owned_registration) is not SimulationExpertProgramRegistration:
                raise TypeError(
                    "A registration-owning factory must expose exactly "
                    "SimulationExpertProgramRegistration."
                )
            if registration is not owned_registration:
                raise ValueError(
                    "registration must be the exact object owned by the factory."
                )
        registered_lowerer_values = tuple(registered_lowerers)
        relation_grounder_values = tuple(relation_grounders)
        handover_pose_provider_values = tuple(handover_pose_providers)
        runtime_transport_values = tuple(runtime_transports)
        if registration is not None:
            if integration_catalog is not None:
                raise ValueError(
                    "integration_catalog cannot override an exact task registration."
                )
            forbidden = {
                "call_catalog": call_catalog is not None,
                "endpoint_adapters": endpoint_adapters is not None,
                "registered_lowerers": bool(registered_lowerer_values),
                "relation_grounders": bool(relation_grounder_values),
                "handover_pose_providers": bool(handover_pose_provider_values),
                "effect_monitor_registry": effect_monitor_registry is not None,
                "runtime_transports": bool(runtime_transport_values),
                "runner_cfg": runner_cfg is not None,
                "post_policy_port": post_policy_port is not None,
                "validator_port": validator_port is not None,
                "parallel_safety_validator": parallel_safety_validator is not None,
            }
            supplied = tuple(name for name, present in forbidden.items() if present)
            if supplied:
                raise ValueError(
                    "Standard task registration owns all semantic and runtime "
                    f"extensions; external overrides are forbidden: {supplied}."
                )
            registration.assert_unchanged()
            integration_catalog = registration.catalog
            endpoint_adapters = dict(registration.endpoint_adapter_map)
            registered_lowerer_values = ()
            relation_grounder_values = registration.relation_grounders
            handover_pose_provider_values = registration.handover_pose_providers
            effect_monitor_registry = None
            runtime_transport_values = registration.runtime_transports
            runner_cfg = None
            parallel_safety_validator = None
            assert registration_owner is not None
            owned_ports = registration_owner.registration_owned_segment_policy_ports()
            if type(owned_ports) is not tuple or len(owned_ports) != 2:
                raise TypeError(
                    "registration_owned_segment_policy_ports() must return an "
                    "exact 2-tuple."
                )
            post_policy_port, validator_port = owned_ports
        if integration_catalog is not None:
            if integration_catalog.scene_registry_id != scene_registry_id:
                raise ValueError(
                    "integration_catalog scene_registry_id does not match factory."
                )
            if integration_catalog.robot_profile_id != robot_profile_id:
                raise ValueError(
                    "integration_catalog robot_profile_id does not match factory."
                )
            if call_catalog is not None and (
                call_catalog is not integration_catalog.call_catalog
            ):
                raise ValueError(
                    "call_catalog cannot override the task registration catalog."
                )
            selected_catalog = integration_catalog.call_catalog
        else:
            selected_catalog = call_catalog or builtin_semantic_call_catalog()
        if type(selected_catalog) is not SemanticCallCatalog:
            raise TypeError("call_catalog must be exactly SemanticCallCatalog or None.")
        if endpoint_adapters is not None and not isinstance(endpoint_adapters, Mapping):
            raise TypeError("endpoint_adapters must be a mapping or None.")
        if runner_cfg is not None and not isinstance(runner_cfg, ExecutionRunnerCfg):
            raise TypeError("runner_cfg must be an ExecutionRunnerCfg or None.")
        if post_policy_port is not None and not isinstance(
            post_policy_port,
            SegmentPostPolicyPort,
        ):
            raise TypeError(
                "post_policy_port must implement SegmentPostPolicyPort or be None."
            )
        if validator_port is not None and not isinstance(
            validator_port,
            SegmentValidatorPort,
        ):
            raise TypeError(
                "validator_port must implement SegmentValidatorPort or be None."
            )
        if parallel_safety_validator is not None and not isinstance(
            parallel_safety_validator,
            ParallelCommandSafetyValidator,
        ):
            raise TypeError(
                "parallel_safety_validator must implement "
                "ParallelCommandSafetyValidator or be None."
            )

        self._factory = factory
        self._scene_registry_id = scene_registry_id
        self._robot_profile_id = robot_profile_id
        self._step_dt = float(step_dt)
        self._registration = registration
        self._integration_catalog = integration_catalog
        self._call_catalog = selected_catalog
        self._endpoint_adapters = (
            None if endpoint_adapters is None else dict(endpoint_adapters)
        )
        self._registered_lowerers = registered_lowerer_values
        self._relation_grounders = relation_grounder_values
        self._handover_pose_providers = handover_pose_provider_values
        self._effect_monitor_registry = effect_monitor_registry
        self._runtime_transports = runtime_transport_values
        self._runner_cfg = runner_cfg
        self._post_policy_port = post_policy_port
        self._validator_port = validator_port
        self._parallel_safety_validator = parallel_safety_validator

    @property
    def scene_registry_id(self) -> str:
        """Return the exact scene integration ID accepted by this adapter.

        Returns:
            Stable scene-registry identifier.
        """
        return self._scene_registry_id

    @property
    def robot_profile_id(self) -> str:
        """Return the exact robot profile ID accepted by this adapter.

        Returns:
            Stable robot-profile identifier.
        """
        return self._robot_profile_id

    @property
    def step_dt(self) -> float:
        """Return the authoritative environment-step cadence.

        Returns:
            Positive control step duration in seconds.
        """
        return self._step_dt

    def compile(self, program: ExpertProgramCfg) -> CompiledProgram:
        """Compile one program after exact integration-selection validation.

        Args:
            program: Strict declarative program configuration.

        Returns:
            Provider-free lazily expanded compiled program.
        """
        if type(program) is not ExpertProgramCfg:
            raise TypeError("program must be exactly ExpertProgramCfg.")
        self._validate_selection(program.integration)
        if self._integration_catalog is not None:
            return self._integration_catalog.preflight(program)
        registry = self._create_scene_registry()
        return ExpertProgramCompiler.from_scene_registry(registry).compile(program)

    def assemble_runtime(
        self,
        integration: ExpertProgramIntegrationCfg,
    ) -> ExpertProgramRuntimeAssembly:
        """Create a fresh fully connected semantic runtime.

        Args:
            integration: Exact scene, profile, and runtime-preset selection.

        Returns:
            Owned assembly containing every validated runtime boundary.
        """
        semantic = self._assemble_semantic_components(integration)
        return self._assemble_execution_runtime(semantic)

    def _assemble_semantic_components(
        self,
        integration: ExpertProgramIntegrationCfg,
    ) -> _ExpertProgramSemanticAssembly:
        """Bind compiler dependencies without observation or evidence ports."""
        self._validate_selection(integration)
        registry = self._create_scene_registry()
        current_profile_id = _validate_identifier(
            self._factory.robot_profile_id,
            field_name="factory.robot_profile_id",
        )
        if current_profile_id != self._robot_profile_id:
            raise ValueError(
                "Factory robot profile declaration drifted: expected "
                f"{self._robot_profile_id!r}, got {current_profile_id!r}."
            )
        profile = self._factory.create_robot_skill_profile()
        self._validate_registration_ownership()
        if type(profile) is not RobotSkillProfile:
            raise TypeError(
                "create_robot_skill_profile() must return exactly RobotSkillProfile."
            )
        if profile.profile_id != self._robot_profile_id:
            raise ValueError(
                "Factory robot profile declaration drifted: expected "
                f"{self._robot_profile_id!r}, got {profile.profile_id!r}."
            )
        if self._registration is not None:
            self._registration.validate_robot_profile(profile)

        engine = self._factory.create_atomic_action_engine(profile)
        self._validate_registration_ownership()
        if not isinstance(engine, AtomicActionEngine):
            raise TypeError(
                "create_atomic_action_engine() must return an AtomicActionEngine."
            )
        if self._registration is not None:
            bound_profile = engine.skill_profile
            if type(bound_profile) is not BoundRobotSkillProfile:
                raise IntegrationFingerprintMismatch(
                    "The standard factory engine must own one exact bound robot "
                    "profile."
                )
            if bound_profile.source_profile is not profile:
                raise IntegrationFingerprintMismatch(
                    "The standard factory engine is bound to a different robot "
                    "profile object than the adapter validated."
                )
            self._registration.validate_engine(engine)

        manifest = self._create_manifest(
            registry,
            profile,
            runtime_preset=integration.runtime_preset,
        )
        bound = manifest.bind(
            registry,
            engine,
            endpoint_adapters=self._endpoint_adapters,
        )
        if self._registration is not None:
            self._validate_registration_ownership()
            # ``manifest.bind`` resolves endpoints again and replaces the
            # engine-owned bound profile. Revalidate that second live result so a
            # provider cannot pass factory construction and drift before compile.
            self._registration.validate_engine(engine)
        compiler = SemanticSkillCompiler(
            bound,
            registered_lowerers=self._registered_lowerers,
            relation_grounders=self._relation_grounders,
            handover_pose_providers=self._handover_pose_providers,
            effect_monitor_registry=self._effect_monitor_registry,
        )

        selection = ExpertProgramIntegrationCfg(
            robot_profile=integration.robot_profile,
            scene_registry=integration.scene_registry,
            runtime_preset=integration.runtime_preset,
        )
        return _ExpertProgramSemanticAssembly(
            integration=selection,
            scene_registry=registry,
            robot_profile=profile,
            manifest=manifest,
            engine=engine,
            compiler=compiler,
        )

    def _assemble_execution_runtime(
        self,
        semantic: _ExpertProgramSemanticAssembly,
    ) -> ExpertProgramRuntimeAssembly:
        """Attach live observation, evidence, command, and runtime boundaries."""
        if type(semantic) is not _ExpertProgramSemanticAssembly:
            raise TypeError("semantic must be exactly _ExpertProgramSemanticAssembly.")
        self._validate_registration_ownership()

        clock = EnvironmentStepClock(self._step_dt)
        observation_provider = self._factory.create_planning_observation_provider(
            scene_registry=semantic.scene_registry,
            engine=semantic.engine,
            clock=clock,
        )
        self._validate_registration_ownership()
        if not isinstance(observation_provider, PlanningObservationPort):
            raise TypeError(
                "create_planning_observation_provider() must return a port "
                "implementing both ObservationProvider and CurrentQposProvider."
            )
        providers = self._factory.create_effect_evidence_providers(
            scene_registry=semantic.scene_registry,
            engine=semantic.engine,
            observation_provider=observation_provider,
        )
        self._validate_registration_ownership()
        if isinstance(providers, (str, bytes)):
            raise TypeError(
                "create_effect_evidence_providers() must return an iterable of "
                "EffectEvidenceProvider values."
            )
        try:
            provider_values = tuple(providers)
        except TypeError as exc:
            raise TypeError(
                "create_effect_evidence_providers() must return an iterable of "
                "EffectEvidenceProvider values."
            ) from exc
        evidence_collector = EffectEvidenceCollector(
            EffectEvidenceProviderRegistry(provider_values)
        )
        expected_transport_ids: tuple[str, ...] | None = None
        include_joint_position = True
        if self._registration is not None:
            expected_transport_ids = tuple(
                declaration.transport_id
                for declaration in self._registration.catalog.extensions.runtime_transports
            )
            include_joint_position = (
                JointPositionGymTransportEncoder.transport_id in expected_transport_ids
            )
        command_encoder = RuntimeCommandFrameEncoder(
            observation_provider,
            transports=self._runtime_transports,
            include_joint_position=include_joint_position,
        )
        if expected_transport_ids is not None:
            if command_encoder.transport_ids != expected_transport_ids:
                raise IntegrationFingerprintMismatch(
                    "Live command encoder transport order differs from the exact "
                    "registration catalog."
                )
            command_encoder.freeze()
        command_sink = BufferedGymCommandSink(command_encoder, clock)
        try:
            selected_preset = semantic.robot_profile.presets[
                semantic.integration.runtime_preset
            ]
        except KeyError as exc:
            raise ValueError(
                "The selected runtime preset is absent from the assembled robot "
                "profile."
            ) from exc
        selected_runner_cfg = selected_preset.runner_cfg
        if self._registration is None and self._runner_cfg is not None:
            selected_runner_cfg = deepcopy(self._runner_cfg)
        runtime = SkillRuntime.from_components(
            semantic.compiler,
            observation_provider,
            command_sink,
            evidence_collector,
            clock=clock,
            runner_cfg=deepcopy(selected_runner_cfg),
        )
        parallel_safety_validator = self._parallel_safety_validator
        if (
            self._registration is not None
            and self._registration.parallel_safety_factory is not None
        ):
            if not isinstance(
                self._factory,
                ParallelCommandSafetyValidatorProvider,
            ):
                raise TypeError(
                    "A registration-owned parallel_safety_factory requires the "
                    "environment factory to implement "
                    "ParallelCommandSafetyValidatorProvider."
                )
            parallel_safety_validator = (
                self._factory.create_parallel_command_safety_validator(
                    scene_registry=semantic.scene_registry,
                    engine=semantic.engine,
                    observation_provider=observation_provider,
                )
            )
            self._validate_registration_ownership()
            if not isinstance(
                parallel_safety_validator,
                ParallelCommandSafetyValidator,
            ):
                raise TypeError(
                    "create_parallel_command_safety_validator() must return a "
                    "ParallelCommandSafetyValidator."
                )
        return ExpertProgramRuntimeAssembly(
            integration=semantic.integration,
            scene_registry=semantic.scene_registry,
            robot_profile=semantic.robot_profile,
            manifest=semantic.manifest,
            engine=semantic.engine,
            compiler=semantic.compiler,
            observation_provider=observation_provider,
            evidence_collector=evidence_collector,
            clock=clock,
            command_encoder=command_encoder,
            command_sink=command_sink,
            runner_cfg=selected_runner_cfg,
            parallel_safety_validator=parallel_safety_validator,
            runtime=runtime,
        )

    def create_bridge(self, program: CompiledProgram) -> AtomicDemoBridge:
        """Create a fresh Gym bridge for one provider-free compiled program.

        Args:
            program: Program compiled for this adapter's exact integration IDs.

        Returns:
            Lazy bridge sharing one newly assembled runtime, clock, and sink.
        """
        if type(program) is not CompiledProgram:
            raise TypeError("program must be exactly CompiledProgram.")
        self._validate_selection(program.integration)
        self._preflight_program_surfaces(program)
        semantic = self._assemble_semantic_components(program.integration)
        self._preflight_program(program, semantic.compiler)
        assembly = self._assemble_execution_runtime(semantic)
        return AtomicDemoBridge(
            program,
            assembly.runtime,
            assembly.command_sink,
            assembly.clock,
            post_policy_port=self._post_policy_port,
            validator_port=self._validator_port,
            runner_cfg=assembly.runner_cfg,
            parallel_safety_validator=assembly.parallel_safety_validator,
        )

    def _preflight_program_surfaces(
        self,
        program: CompiledProgram,
    ) -> None:
        """Validate every segment hook without live observation or action."""
        if type(program) is not CompiledProgram:
            raise TypeError("program must be exactly CompiledProgram.")
        for segment in program.iter_segments():
            if segment.post_policies and self._post_policy_port is None:
                raise DemoBridgeError(
                    f"Segment {segment.segment_id!r} declares post-policies, but no "
                    "SegmentPostPolicyPort was installed."
                )
            for policy in segment.post_policies:
                assert self._post_policy_port is not None
                self._post_policy_port.validate_policy(policy, segment=segment)
            if segment.validators and self._validator_port is None:
                raise DemoBridgeError(
                    f"Segment {segment.segment_id!r} declares validators, but no "
                    "SegmentValidatorPort was installed."
                )
            for validator in segment.validators:
                assert self._validator_port is not None
                self._validator_port.validate_validator(
                    validator,
                    segment=segment,
                )

    def _preflight_program(
        self,
        program: CompiledProgram,
        compiler: SemanticSkillCompiler,
    ) -> None:
        """Analyze every program workflow before any physical action can run.

        Sequential stretches retain cross-segment state flow and target
        look-ahead.  A parallel barrier cuts that flow; each branch is checked
        independently through the same canonical semantic compiler used by the
        runtime.  This boundary materializes no observations and starts no
        execution session.
        """
        if type(program) is not CompiledProgram:
            raise TypeError("program must be exactly CompiledProgram.")
        if not isinstance(compiler, SemanticSkillCompiler):
            raise TypeError("compiler must be a SemanticSkillCompiler.")
        analyses = program.preflight_analyses()
        if any(analysis.kind == "parallel_branch" for analysis in analyses) and (
            not self._parallel_safety_is_registered
        ):
            raise ValueError(
                "Expert Programs containing parallel blocks require an explicit "
                "ParallelCommandSafetyValidator before bridge creation."
            )

        index = 0
        while index < len(analyses):
            analysis = analyses[index]
            if analysis.kind != "parallel_branch":
                compiler.analyze(
                    analysis.calls,
                    workflow_id=analysis.analysis_id,
                    path=analysis.source_path,
                )
                index += 1
                continue
            segment_index = analysis.segment_indices[0]
            branches: dict[str, tuple[SemanticCallSpec, ...]] = {}
            branch_paths: dict[str, tuple[str | int, ...]] = {}
            while index < len(analyses):
                branch = analyses[index]
                if branch.kind != "parallel_branch" or branch.segment_indices != (
                    segment_index,
                ):
                    break
                branch_id = f"branch_{len(branches)}"
                branches[branch_id] = branch.calls
                branch_paths[branch_id] = branch.source_path
                index += 1
            analyze_parallel_branches(
                compiler,
                branches,
                workflow_id=(
                    f"{program.program_id}:preflight:parallel:{segment_index}"
                ),
                branch_paths=branch_paths,
            )

    @property
    def _parallel_safety_is_registered(self) -> bool:
        """Whether static assembly owns an authoritative parallel safety gate."""
        if self._registration is not None:
            return self._registration.parallel_safety_factory is not None
        return self._parallel_safety_validator is not None

    def _validate_selection(
        self,
        integration: ExpertProgramIntegrationCfg,
    ) -> None:
        """Reject an integration selection owned by another adapter."""
        if type(integration) is not ExpertProgramIntegrationCfg:
            raise TypeError("integration must be exactly ExpertProgramIntegrationCfg.")
        self._validate_registration_ownership()
        current_scene_id = _validate_identifier(
            self._factory.scene_registry_id,
            field_name="factory.scene_registry_id",
        )
        current_profile_id = _validate_identifier(
            self._factory.robot_profile_id,
            field_name="factory.robot_profile_id",
        )
        if current_scene_id != self._scene_registry_id:
            raise ValueError(
                "Factory scene registry declaration drifted: expected "
                f"{self._scene_registry_id!r}, got {current_scene_id!r}."
            )
        if current_profile_id != self._robot_profile_id:
            raise ValueError(
                "Factory robot profile declaration drifted: expected "
                f"{self._robot_profile_id!r}, got {current_profile_id!r}."
            )
        if integration.scene_registry != self._scene_registry_id:
            raise ValueError(
                f"Expert Program selects scene_registry "
                f"{integration.scene_registry!r}, but this environment exposes "
                f"only {self._scene_registry_id!r}."
            )
        if integration.robot_profile != self._robot_profile_id:
            raise ValueError(
                f"Expert Program selects robot_profile "
                f"{integration.robot_profile!r}, but this environment exposes "
                f"only {self._robot_profile_id!r}."
            )

    def _validate_registration_ownership(self) -> None:
        """Reject a standard factory whose exact registration owner drifted."""
        registration = self._registration
        if registration is None:
            return
        if not isinstance(self._factory, _RegistrationOwningExpertProgramFactory):
            raise IntegrationFingerprintMismatch(
                "The standard environment factory no longer exposes registration "
                "ownership."
            )
        current = self._factory.expert_program_registration
        if type(current) is not SimulationExpertProgramRegistration:
            raise IntegrationFingerprintMismatch(
                "The standard environment factory no longer exposes an exact "
                "SimulationExpertProgramRegistration."
            )
        if current is not registration:
            raise IntegrationFingerprintMismatch(
                "The standard environment factory registration ownership changed "
                "after adapter construction."
            )

    def _create_scene_registry(self) -> SceneRegistry:
        """Create and validate one exact live scene registry."""
        current_id = _validate_identifier(
            self._factory.scene_registry_id,
            field_name="factory.scene_registry_id",
        )
        if current_id != self._scene_registry_id:
            raise ValueError(
                "Factory scene registry declaration drifted: expected "
                f"{self._scene_registry_id!r}, got {current_id!r}."
            )
        registry = self._factory.create_scene_registry()
        self._validate_registration_ownership()
        if type(registry) is not SceneRegistry:
            raise TypeError(
                "create_scene_registry() must return exactly SceneRegistry."
            )
        if self._registration is not None:
            self._registration.validate_scene_registry(registry)
        return registry

    def _create_manifest(
        self,
        registry: SceneRegistry,
        profile: RobotSkillProfile,
        *,
        runtime_preset: str,
    ) -> SemanticIntegrationManifest:
        """Create one static manifest from exact selected declarations."""
        return SemanticIntegrationManifest(
            scene=SceneManifest.from_registry(registry),
            robot_profile=profile,
            call_catalog=self._call_catalog,
            runtime_preset=runtime_preset,
        )


__all__: list[str] = []
