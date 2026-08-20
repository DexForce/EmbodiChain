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

"""Semantic-skill integration contracts built on the atomic-action core."""

from __future__ import annotations

from .calls import (
    HandOver,
    Pick,
    Place,
    RegisteredSemanticCall,
    SemanticCallCatalog,
    SemanticCallDescriptor,
    SemanticPose,
    builtin_semantic_call_catalog,
)
from .compiler import (
    GroundedSemanticCall,
    HandOverPoseProvider,
    HandOverPoseTargets,
    RegisteredSemanticLowerer,
    RelationTargetGrounder,
    SemanticLowering,
    SemanticObjectTarget,
    SemanticRelationTarget,
    SemanticSkillCompiler,
    SemanticWorkflow,
)
from .integration import (
    SceneEntityManifest,
    SceneManifest,
    SemanticDiagnostic,
    SemanticIntegrationManifest,
    SemanticValidationError,
)
from .profiles import (
    AmbiguousSkillBindingError,
    BoundRobotSkillProfile,
    ControlPartEndpoint,
    ControlPartEndpointAdapter,
    EndpointResolution,
    ProfileValidationError,
    ResolvedResourceEndpoint,
    ResolvedRobotResource,
    ResolvedSkillBinding,
    ResourceBinding,
    ResourceClaim,
    ResourceEndpoint,
    ResourceEndpointAdapter,
    RobotResource,
    RobotSkillProfile,
    SkillPolicyPreset,
    UnsupportedSkillError,
)
from .runtime import (
    SemanticCallRecord,
    SemanticEffectVerifier,
    SemanticExecution,
    SemanticExecutionStatus,
    SemanticExecutionStep,
    SemanticSegmentResult,
    SemanticSkillRuntime,
    SemanticTask,
    SemanticTaskResult,
    SemanticTaskStatus,
)
from .scene import (
    AmbiguousSceneAffordanceError,
    GRASP_AFFORDANCE_CAPABILITY,
    PLACE_IN_AFFORDANCE_CAPABILITY,
    PLACE_ON_AFFORDANCE_CAPABILITY,
    RegistrySceneProvider,
    SceneAffordanceRef,
    SceneArticulationRef,
    SceneCollisionRole,
    SceneCollisionWorldMode,
    SceneDynamics,
    SceneEntityRef,
    SceneEntityMetadata,
    SceneEntityRegistration,
    SceneEntityStateProvider,
    SceneGeometryProvider,
    SceneLinkRef,
    SceneObjectRef,
    SceneRegistry,
    UnsupportedSceneAffordanceError,
)

__all__ = [
    "AmbiguousSceneAffordanceError",
    "AmbiguousSkillBindingError",
    "BoundRobotSkillProfile",
    "ControlPartEndpoint",
    "ControlPartEndpointAdapter",
    "EndpointResolution",
    "GRASP_AFFORDANCE_CAPABILITY",
    "GroundedSemanticCall",
    "HandOver",
    "HandOverPoseProvider",
    "HandOverPoseTargets",
    "PLACE_IN_AFFORDANCE_CAPABILITY",
    "PLACE_ON_AFFORDANCE_CAPABILITY",
    "Pick",
    "Place",
    "ProfileValidationError",
    "RegistrySceneProvider",
    "ResolvedRobotResource",
    "ResolvedResourceEndpoint",
    "ResolvedSkillBinding",
    "ResourceBinding",
    "ResourceClaim",
    "ResourceEndpoint",
    "ResourceEndpointAdapter",
    "RegisteredSemanticCall",
    "RegisteredSemanticLowerer",
    "RelationTargetGrounder",
    "RobotResource",
    "RobotSkillProfile",
    "SceneAffordanceRef",
    "SceneArticulationRef",
    "SceneCollisionRole",
    "SceneCollisionWorldMode",
    "SceneDynamics",
    "SceneEntityRef",
    "SceneEntityMetadata",
    "SceneEntityRegistration",
    "SceneEntityManifest",
    "SceneEntityStateProvider",
    "SceneGeometryProvider",
    "SceneLinkRef",
    "SceneObjectRef",
    "SceneRegistry",
    "SceneManifest",
    "SemanticCallCatalog",
    "SemanticCallDescriptor",
    "SemanticCallRecord",
    "SemanticDiagnostic",
    "SemanticEffectVerifier",
    "SemanticExecution",
    "SemanticExecutionStatus",
    "SemanticExecutionStep",
    "SemanticIntegrationManifest",
    "SemanticLowering",
    "SemanticObjectTarget",
    "SemanticPose",
    "SemanticRelationTarget",
    "SemanticSegmentResult",
    "SemanticSkillCompiler",
    "SemanticSkillRuntime",
    "SemanticTask",
    "SemanticTaskResult",
    "SemanticTaskStatus",
    "SemanticValidationError",
    "SemanticWorkflow",
    "SkillPolicyPreset",
    "UnsupportedSkillError",
    "UnsupportedSceneAffordanceError",
    "builtin_semantic_call_catalog",
]
