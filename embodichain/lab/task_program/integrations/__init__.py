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
    "PostPolicyPresetBinding",
    "BoundPostPolicyPresets",
    "task_program_fingerprint",
    "ContainerAffordanceBinding",
    "ControlPartCommandPreset",
    "ControlPartEvidenceProviderDeclaration",
    "ControlPartEvidenceProviderFactory",
    "DeclaredAssetFileProvider",
    "ControlPartEndpointBinding",
    "ControlPartResourceBinding",
    "ConfiguredHandOverPoseProvider",
    "ConfiguredTaskProgramDeployment",
    "ConfiguredTaskProgramIntegration",
    "CuroboParallelCommandSafetyValidator",
    "CuroboParallelSafetyValidatorFactory",
    "EndpointAdapterDeclaration",
    "TaskProgramAdapterFactory",
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
    "compile_configured_task_program",
]

_EXPORTS = {
    "TaskProgramAdapterFactory": (".environment", "TaskProgramAdapterFactory"),
    "TaskProgramEnvironmentAdapter": (".environment", "TaskProgramEnvironmentAdapter"),
    "TaskProgramEnvironmentFactory": (".environment", "TaskProgramEnvironmentFactory"),
    "TaskProgramRuntimeAssembly": (".environment", "TaskProgramRuntimeAssembly"),
    "PlanningObservationPort": (".environment", "PlanningObservationPort"),
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
    "TaskProgramSourceAdapter": (".expansion", "TaskProgramSourceAdapter"),
    "AntipodalGraspAffordanceBinding": (
        ".simulation",
        "AntipodalGraspAffordanceBinding",
    ),
    "ContainerAffordanceBinding": (".simulation", "ContainerAffordanceBinding"),
    "ControlPartCommandPreset": (".simulation", "ControlPartCommandPreset"),
    "ControlPartEndpointBinding": (".simulation", "ControlPartEndpointBinding"),
    "ControlPartResourceBinding": (".simulation", "ControlPartResourceBinding"),
    "RigidizedArticulationAntipodalGraspBinding": (
        ".simulation",
        "RigidizedArticulationAntipodalGraspBinding",
    ),
    "SimulationArticulationBinding": (".simulation", "SimulationArticulationBinding"),
    "SimulationArticulationLinkBinding": (
        ".simulation",
        "SimulationArticulationLinkBinding",
    ),
    "SimulationRigidizedArticulationObjectBinding": (
        ".simulation",
        "SimulationRigidizedArticulationObjectBinding",
    ),
    "SimulationRigidObjectBinding": (".simulation", "SimulationRigidObjectBinding"),
    "SimulationRobotSkillProfileBinding": (
        ".simulation",
        "SimulationRobotSkillProfileBinding",
    ),
    "SimulationSceneBinding": (".simulation", "SimulationSceneBinding"),
    "SupportSurfaceAffordanceBinding": (
        ".simulation",
        "SupportSurfaceAffordanceBinding",
    ),
    "TaskProgramIntegrationCatalog": (".catalog", "TaskProgramIntegrationCatalog"),
    "IntegrationFingerprintMismatch": (".catalog", "IntegrationFingerprintMismatch"),
    "SimulationTaskProgramRegistration": (
        ".catalog",
        "SimulationTaskProgramRegistration",
    ),
    "ConfiguredTaskProgramDeployment": (
        ".composition",
        "ConfiguredTaskProgramDeployment",
    ),
    "ConfiguredTaskProgramIntegration": (
        ".composition",
        "ConfiguredTaskProgramIntegration",
    ),
    "load_configured_task_program_deployment": (
        ".composition",
        "load_configured_task_program_deployment",
    ),
    "compile_configured_task_program": (
        ".composition",
        "compile_configured_task_program",
    ),
    "ControlPartEvidenceProviderDeclaration": (
        ".extensions",
        "ControlPartEvidenceProviderDeclaration",
    ),
    "ControlPartEvidenceProviderFactory": (
        ".extensions",
        "ControlPartEvidenceProviderFactory",
    ),
    "DeclaredAssetFileProvider": (".extensions", "DeclaredAssetFileProvider"),
    "EndpointAdapterDeclaration": (".extensions", "EndpointAdapterDeclaration"),
    "ParallelCommandSafetyValidatorFactory": (
        ".extensions",
        "ParallelCommandSafetyValidatorFactory",
    ),
    "ParallelSafetyDeclaration": (".extensions", "ParallelSafetyDeclaration"),
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
    "RuntimeTransportDeclaration": (".extensions", "RuntimeTransportDeclaration"),
    "StandardExtensionDeclarations": (".extensions", "StandardExtensionDeclarations"),
    "VersionedKey": (".extensions", "VersionedKey"),
    "SimulationTaskProgramAdapterFactory": (
        ".simulation.environment",
        "SimulationTaskProgramAdapterFactory",
    ),
    "SimulationTaskProgramFactory": (
        ".simulation.environment",
        "SimulationTaskProgramFactory",
    ),
    "create_simulation_task_program_adapter": (
        ".simulation.environment",
        "create_simulation_task_program_adapter",
    ),
    "ConfiguredHandOverPoseProvider": (
        ".simulation.handover",
        "ConfiguredHandOverPoseProvider",
    ),
    "CuroboParallelCommandSafetyValidator": (
        ".simulation.parallel_safety",
        "CuroboParallelCommandSafetyValidator",
    ),
    "CuroboParallelSafetyValidatorFactory": (
        ".simulation.parallel_safety",
        "CuroboParallelSafetyValidatorFactory",
    ),
    "SimulationSegmentPolicyPort": (
        ".simulation.policies",
        "SimulationSegmentPolicyPort",
    ),
    "default_simulation_settle_presets": (
        ".simulation.policies",
        "default_simulation_settle_presets",
    ),
    "BoundInvocationPolicies": (".invocation_policy", "BoundInvocationPolicies"),
    "InvocationPolicyBinding": (".invocation_policy", "InvocationPolicyBinding"),
    "PostPolicyPresetBinding": (".invocation_policy", "PostPolicyPresetBinding"),
    "BoundPostPolicyPresets": (".invocation_policy", "BoundPostPolicyPresets"),
    "task_program_fingerprint": (".invocation_policy", "task_program_fingerprint"),
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
