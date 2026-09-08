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

"""Explicit provider, environment, and simulation Task Program integrations."""

from __future__ import annotations

from .environment import (
    TaskProgramAdapterFactory,
    TaskProgramEnvironmentAdapter,
    TaskProgramEnvironmentFactory,
    TaskProgramRuntimeAssembly,
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
    TaskProgramIntegrationCatalog,
    IntegrationFingerprintMismatch,
    SimulationTaskProgramRegistration,
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
from .simulation.environment import (
    SimulationTaskProgramAdapterFactory,
    SimulationTaskProgramFactory,
    create_simulation_task_program_adapter,
)
from .simulation.handover import ConfiguredHandOverPoseProvider
from .simulation.parallel_safety import (
    CuroboParallelCommandSafetyValidator,
    CuroboParallelSafetyValidatorFactory,
)
from .simulation.policies import (
    SimulationSegmentPolicyPort,
    default_simulation_settle_presets,
)

__all__ = [
    "AntipodalGraspAffordanceBinding",
    "ContainerAffordanceBinding",
    "ControlPartCommandPreset",
    "ControlPartEvidenceProviderDeclaration",
    "ControlPartEvidenceProviderFactory",
    "ControlPartEndpointBinding",
    "ControlPartResourceBinding",
    "ConfiguredHandOverPoseProvider",
    "CuroboParallelCommandSafetyValidator",
    "CuroboParallelSafetyValidatorFactory",
    "EndpointAdapterDeclaration",
    "TaskProgramAdapterFactory",
    "TaskProgramEnvironmentAdapter",
    "TaskProgramEnvironmentFactory",
    "TaskProgramIntegrationCatalog",
    "TaskProgramRuntimeAssembly",
    "IntegrationFingerprintMismatch",
    "ParallelCommandSafetyValidatorFactory",
    "ParallelSafetyDeclaration",
    "PlanningObservationPort",
    "RegisteredSemanticLowererDeclaration",
    "RegisteredSemanticLowererFactory",
    "RuntimeTransportDeclaration",
    "SimulationArticulationBinding",
    "SimulationArticulationLinkBinding",
    "SimulationTaskProgramAdapterFactory",
    "SimulationTaskProgramFactory",
    "SimulationTaskProgramRegistration",
    "SimulationRigidObjectBinding",
    "SimulationRobotSkillProfileBinding",
    "SimulationSceneBinding",
    "SimulationSegmentPolicyPort",
    "StandardExtensionDeclarations",
    "SupportSurfaceAffordanceBinding",
    "VersionedKey",
    "create_simulation_task_program_adapter",
    "default_simulation_settle_presets",
]
