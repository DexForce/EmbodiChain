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

"""Versioned declarative Expert Program schema, compiler, and runtime types."""

from __future__ import annotations

from .cfg import (
    EXPERT_PROGRAM_SCHEMA_VERSION,
    ArticulationJointPositionValidatorCfg,
    BarrierCfg,
    CyclicPoseTargetCfg,
    ExpertProgramCfg,
    ExpertProgramIntegrationCfg,
    HandOverCfg,
    InvokeCfg,
    ObjectNearTargetValidatorCfg,
    ParallelCfg,
    PickCfg,
    PlaceCfg,
    PoseCfg,
    RegisteredSemanticCallCfg,
    RepeatCfg,
    SegmentCfg,
    SequenceCfg,
    TargetRefCfg,
    WaitStablePostCfg,
)
from .decoder import (
    ConfigPath,
    ConfigPathPart,
    ExpertProgramConfigError,
    ExpertProgramDecodeError,
    ExpertProgramValidationContext,
    ExpertProgramValidationError,
    SceneReferenceRole,
    decode_expert_program,
    render_config_path,
    validate_expert_program,
)
from .loader import (
    load_expert_program,
    loads_expert_program_json,
    parse_expert_program_json,
)
from .bridge import (
    AtomicDemoBridge,
    BufferedGymCommandSink,
    DemoBridgeError,
    EnvironmentStepClock,
    EnvironmentStepTimingError,
    GymPlanningObservationProvider,
    RuntimeCommandFrameEncoder,
    RuntimeTransportActionEncoder,
    SegmentPostPolicyPort,
    SegmentValidatorPort,
    UnsupportedRuntimeTransportError,
)
from .compiler import (
    CompiledProgram,
    ExpertProgramCompileError,
    ExpertProgramCompiler,
)
from .environment import (
    ExpertProgramEnvironmentAdapter,
    ExpertProgramEnvironmentFactory,
    ExpertProgramRuntimeAssembly,
    PlanningObservationPort,
)
from .simulation import (
    AntipodalGraspAffordanceBinding,
    ContainerAffordanceBinding,
    ControlPartCommandPreset,
    ControlPartEndpointBinding,
    ControlPartResourceBinding,
    SimulationArticulationBinding,
    SimulationArticulationLinkBinding,
    SimulationRigidObjectBinding,
    SimulationRobotSkillProfileBinding,
    SimulationSceneBinding,
    SupportSurfaceAffordanceBinding,
)
from .catalog import (
    ExpertProgramIntegrationCatalog,
    IntegrationFingerprintMismatch,
    SimulationExpertProgramRegistration,
)
from .extensions import (
    ControlPartEvidenceProviderDeclaration,
    ControlPartEvidenceProviderFactory,
    EndpointAdapterDeclaration,
    ParallelCommandSafetyValidatorFactory,
    ParallelSafetyDeclaration,
    RuntimeTransportDeclaration,
    StandardExtensionDeclarations,
    VersionedKey,
)
from .simulation_environment import (
    SimulationExpertProgramFactory,
    create_simulation_expert_program_adapter,
)
from .simulation_handover import ConfiguredHandOverPoseProvider
from .simulation_parallel_safety import (
    CuroboParallelCommandSafetyValidator,
    CuroboParallelSafetyValidatorFactory,
)
from .simulation_policies import (
    SimulationSegmentPolicyPort,
    default_simulation_settle_presets,
)

__all__ = [
    "AntipodalGraspAffordanceBinding",
    "ArticulationJointPositionValidatorCfg",
    "AtomicDemoBridge",
    "BarrierCfg",
    "BufferedGymCommandSink",
    "ConfigPath",
    "ConfigPathPart",
    "CompiledProgram",
    "ContainerAffordanceBinding",
    "ControlPartCommandPreset",
    "ControlPartEvidenceProviderDeclaration",
    "ControlPartEvidenceProviderFactory",
    "ControlPartEndpointBinding",
    "ControlPartResourceBinding",
    "ConfiguredHandOverPoseProvider",
    "CyclicPoseTargetCfg",
    "CuroboParallelCommandSafetyValidator",
    "CuroboParallelSafetyValidatorFactory",
    "DemoBridgeError",
    "EXPERT_PROGRAM_SCHEMA_VERSION",
    "EnvironmentStepClock",
    "EnvironmentStepTimingError",
    "EndpointAdapterDeclaration",
    "ExpertProgramCfg",
    "ExpertProgramCompileError",
    "ExpertProgramCompiler",
    "ExpertProgramConfigError",
    "ExpertProgramDecodeError",
    "ExpertProgramEnvironmentAdapter",
    "ExpertProgramEnvironmentFactory",
    "ExpertProgramIntegrationCfg",
    "ExpertProgramIntegrationCatalog",
    "ExpertProgramRuntimeAssembly",
    "ExpertProgramValidationContext",
    "ExpertProgramValidationError",
    "HandOverCfg",
    "GymPlanningObservationProvider",
    "InvokeCfg",
    "IntegrationFingerprintMismatch",
    "ObjectNearTargetValidatorCfg",
    "ParallelCfg",
    "ParallelCommandSafetyValidatorFactory",
    "ParallelSafetyDeclaration",
    "PickCfg",
    "PlaceCfg",
    "PlanningObservationPort",
    "PoseCfg",
    "RegisteredSemanticCallCfg",
    "RepeatCfg",
    "RuntimeCommandFrameEncoder",
    "RuntimeTransportDeclaration",
    "RuntimeTransportActionEncoder",
    "SceneReferenceRole",
    "SegmentPostPolicyPort",
    "SegmentCfg",
    "SegmentValidatorPort",
    "SequenceCfg",
    "SimulationArticulationBinding",
    "SimulationArticulationLinkBinding",
    "SimulationExpertProgramFactory",
    "SimulationExpertProgramRegistration",
    "SimulationRigidObjectBinding",
    "SimulationRobotSkillProfileBinding",
    "SimulationSceneBinding",
    "SimulationSegmentPolicyPort",
    "StandardExtensionDeclarations",
    "SupportSurfaceAffordanceBinding",
    "TargetRefCfg",
    "UnsupportedRuntimeTransportError",
    "VersionedKey",
    "WaitStablePostCfg",
    "create_simulation_expert_program_adapter",
    "default_simulation_settle_presets",
    "decode_expert_program",
    "load_expert_program",
    "loads_expert_program_json",
    "parse_expert_program_json",
    "render_config_path",
    "validate_expert_program",
]
