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

"""Gym and simulation adapters for Expert Program execution."""

from __future__ import annotations

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
from .environment import (
    ExpertProgramAdapterFactory,
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
    RegisteredSemanticLowererDeclaration,
    RegisteredSemanticLowererFactory,
    RuntimeTransportDeclaration,
    StandardExtensionDeclarations,
    VersionedKey,
)
from .simulation_environment import (
    SimulationExpertProgramAdapterFactory,
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
    "AtomicDemoBridge",
    "BufferedGymCommandSink",
    "ContainerAffordanceBinding",
    "ControlPartCommandPreset",
    "ControlPartEvidenceProviderDeclaration",
    "ControlPartEvidenceProviderFactory",
    "ControlPartEndpointBinding",
    "ControlPartResourceBinding",
    "ConfiguredHandOverPoseProvider",
    "CuroboParallelCommandSafetyValidator",
    "CuroboParallelSafetyValidatorFactory",
    "DemoBridgeError",
    "EnvironmentStepClock",
    "EnvironmentStepTimingError",
    "EndpointAdapterDeclaration",
    "ExpertProgramAdapterFactory",
    "ExpertProgramEnvironmentAdapter",
    "ExpertProgramEnvironmentFactory",
    "ExpertProgramIntegrationCatalog",
    "ExpertProgramRuntimeAssembly",
    "GymPlanningObservationProvider",
    "IntegrationFingerprintMismatch",
    "ParallelCommandSafetyValidatorFactory",
    "ParallelSafetyDeclaration",
    "PlanningObservationPort",
    "RegisteredSemanticLowererDeclaration",
    "RegisteredSemanticLowererFactory",
    "RuntimeCommandFrameEncoder",
    "RuntimeTransportDeclaration",
    "RuntimeTransportActionEncoder",
    "SegmentPostPolicyPort",
    "SegmentValidatorPort",
    "SimulationArticulationBinding",
    "SimulationArticulationLinkBinding",
    "SimulationExpertProgramAdapterFactory",
    "SimulationExpertProgramFactory",
    "SimulationExpertProgramRegistration",
    "SimulationRigidObjectBinding",
    "SimulationRobotSkillProfileBinding",
    "SimulationSceneBinding",
    "SimulationSegmentPolicyPort",
    "StandardExtensionDeclarations",
    "SupportSurfaceAffordanceBinding",
    "UnsupportedRuntimeTransportError",
    "VersionedKey",
    "create_simulation_expert_program_adapter",
    "default_simulation_settle_presets",
]
