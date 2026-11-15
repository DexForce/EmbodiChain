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

"""Explicit provider, environment, and simulation Task Program integrations.

Public owners load on access; data-only evidence imports create no runtime.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "AntipodalGraspAffordanceBinding",
    "BoundInvocationPolicies",
    "InvocationPolicyBinding",
    "task_program_fingerprint",
    "ContainerAffordanceBinding",
    "ControlPartCommandPreset",
    "ControlPartEvidenceProviderDeclaration",
    "ControlPartEvidenceProviderFactory",
    "ControlPartEndpointBinding",
    "ControlPartResourceBinding",
    "ConfiguredHandOverPoseProvider",
    "ConfiguredTaskProgramDeployment",
    "ConfiguredTaskProgramIntegration",
    "CuroboParallelCommandSafetyValidator",
    "CuroboParallelSafetyValidatorFactory",
    "EndpointAdapterDeclaration",
    "TaskProgramAdapterFactory",
    "TaskProgramProgress",
    "TaskProgramEnvironmentAdapter",
    "TaskProgramEnvironmentFactory",
    "TaskProgramIntegrationCatalog",
    "TaskProgramSourceAdapter",
    "CombinedTaskProgramExpansionFactory",
    "TaskProgramExpansionRecord",
    "TaskProgramMappedRetimeFactory",
    "TaskProgramMappedRetimeRecord",
    "TaskProgramSameGridExpansionFactory",
    "TaskProgramSameGridExpansionRecord",
    "TaskProgramRuntimeAssembly",
    "IntegrationFingerprintMismatch",
    "ParallelCommandSafetyValidatorFactory",
    "ParallelSafetyDeclaration",
    "PlanningObservationPort",
    "RegisteredSemanticLowererDeclaration",
    "RegisteredSemanticLowererFactory",
    "RegisteredSemanticLowererSceneValidator",
    "RigidizedArticulationAntipodalGraspBinding",
    "RuntimeTransportDeclaration",
    "SimulationArticulationBinding",
    "SimulationArticulationLinkBinding",
    "SimulationRigidizedArticulationObjectBinding",
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
    "load_configured_task_program_deployment",
]

_EXPORTS = {
    "TaskProgramProgress": (".progress", "TaskProgramProgress"),
    "AntipodalGraspAffordanceBinding": (
        ".simulation",
        "AntipodalGraspAffordanceBinding",
    ),
    "BoundInvocationPolicies": (".invocation_policy", "BoundInvocationPolicies"),
    "InvocationPolicyBinding": (".invocation_policy", "InvocationPolicyBinding"),
    "task_program_fingerprint": (".invocation_policy", "task_program_fingerprint"),
    "ContainerAffordanceBinding": (".simulation", "ContainerAffordanceBinding"),
    "ControlPartCommandPreset": (".simulation", "ControlPartCommandPreset"),
    "ControlPartEvidenceProviderDeclaration": (
        ".extensions",
        "ControlPartEvidenceProviderDeclaration",
    ),
    "ControlPartEvidenceProviderFactory": (
        ".extensions",
        "ControlPartEvidenceProviderFactory",
    ),
    "ControlPartEndpointBinding": (".simulation", "ControlPartEndpointBinding"),
    "ControlPartResourceBinding": (".simulation", "ControlPartResourceBinding"),
    "ConfiguredHandOverPoseProvider": (
        ".simulation.handover",
        "ConfiguredHandOverPoseProvider",
    ),
    "ConfiguredTaskProgramDeployment": (
        ".composition",
        "ConfiguredTaskProgramDeployment",
    ),
    "ConfiguredTaskProgramIntegration": (
        ".composition",
        "ConfiguredTaskProgramIntegration",
    ),
    "CuroboParallelCommandSafetyValidator": (
        ".simulation.parallel_safety",
        "CuroboParallelCommandSafetyValidator",
    ),
    "CuroboParallelSafetyValidatorFactory": (
        ".simulation.parallel_safety",
        "CuroboParallelSafetyValidatorFactory",
    ),
    "EndpointAdapterDeclaration": (".extensions", "EndpointAdapterDeclaration"),
    "TaskProgramAdapterFactory": (".environment", "TaskProgramAdapterFactory"),
    "TaskProgramEnvironmentAdapter": (".environment", "TaskProgramEnvironmentAdapter"),
    "TaskProgramEnvironmentFactory": (".environment", "TaskProgramEnvironmentFactory"),
    "TaskProgramIntegrationCatalog": (".catalog", "TaskProgramIntegrationCatalog"),
    "TaskProgramSourceAdapter": (".expansion", "TaskProgramSourceAdapter"),
    "CombinedTaskProgramExpansionFactory": (
        ".expansion",
        "CombinedTaskProgramExpansionFactory",
    ),
    "TaskProgramExpansionRecord": (".expansion", "TaskProgramExpansionRecord"),
    "TaskProgramMappedRetimeFactory": (".expansion", "TaskProgramMappedRetimeFactory"),
    "TaskProgramMappedRetimeRecord": (".expansion", "TaskProgramMappedRetimeRecord"),
    "TaskProgramSameGridExpansionFactory": (
        ".expansion",
        "TaskProgramSameGridExpansionFactory",
    ),
    "TaskProgramSameGridExpansionRecord": (
        ".expansion",
        "TaskProgramSameGridExpansionRecord",
    ),
    "TaskProgramRuntimeAssembly": (".environment", "TaskProgramRuntimeAssembly"),
    "IntegrationFingerprintMismatch": (".catalog", "IntegrationFingerprintMismatch"),
    "ParallelCommandSafetyValidatorFactory": (
        ".extensions",
        "ParallelCommandSafetyValidatorFactory",
    ),
    "ParallelSafetyDeclaration": (".extensions", "ParallelSafetyDeclaration"),
    "PlanningObservationPort": (".environment", "PlanningObservationPort"),
    "RegisteredSemanticLowererDeclaration": (
        ".extensions",
        "RegisteredSemanticLowererDeclaration",
    ),
    "RegisteredSemanticLowererFactory": (
        ".extensions",
        "RegisteredSemanticLowererFactory",
    ),
    "RegisteredSemanticLowererSceneValidator": (
        ".extensions",
        "RegisteredSemanticLowererSceneValidator",
    ),
    "RigidizedArticulationAntipodalGraspBinding": (
        ".simulation",
        "RigidizedArticulationAntipodalGraspBinding",
    ),
    "RuntimeTransportDeclaration": (".extensions", "RuntimeTransportDeclaration"),
    "SimulationArticulationBinding": (".simulation", "SimulationArticulationBinding"),
    "SimulationArticulationLinkBinding": (
        ".simulation",
        "SimulationArticulationLinkBinding",
    ),
    "SimulationRigidizedArticulationObjectBinding": (
        ".simulation",
        "SimulationRigidizedArticulationObjectBinding",
    ),
    "SimulationTaskProgramAdapterFactory": (
        ".simulation.environment",
        "SimulationTaskProgramAdapterFactory",
    ),
    "SimulationTaskProgramFactory": (
        ".simulation.environment",
        "SimulationTaskProgramFactory",
    ),
    "SimulationTaskProgramRegistration": (
        ".catalog",
        "SimulationTaskProgramRegistration",
    ),
    "SimulationRigidObjectBinding": (".simulation", "SimulationRigidObjectBinding"),
    "SimulationRobotSkillProfileBinding": (
        ".simulation",
        "SimulationRobotSkillProfileBinding",
    ),
    "SimulationSceneBinding": (".simulation", "SimulationSceneBinding"),
    "SimulationSegmentPolicyPort": (
        ".simulation.policies",
        "SimulationSegmentPolicyPort",
    ),
    "StandardExtensionDeclarations": (".extensions", "StandardExtensionDeclarations"),
    "SupportSurfaceAffordanceBinding": (
        ".simulation",
        "SupportSurfaceAffordanceBinding",
    ),
    "VersionedKey": (".extensions", "VersionedKey"),
    "create_simulation_task_program_adapter": (
        ".simulation.environment",
        "create_simulation_task_program_adapter",
    ),
    "default_simulation_settle_presets": (
        ".simulation.policies",
        "default_simulation_settle_presets",
    ),
    "load_configured_task_program_deployment": (
        ".composition",
        "load_configured_task_program_deployment",
    ),
}


def __getattr__(name: str) -> Any:
    """Resolve the existing public object only when its capability is requested."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
