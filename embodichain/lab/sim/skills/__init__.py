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
    DeclarativeValue,
    HandOver,
    Pick,
    Place,
    PlaceRelationTarget,
    RegisteredSemanticCall,
    SemanticCallCatalog,
    SemanticCallDescriptor,
    SemanticCallSpec,
    SemanticPose,
    builtin_semantic_call_catalog,
)
from .integration import (
    BoundSemanticCall,
    BoundSemanticIntegration,
    LinkedSemanticCall,
    PathPart,
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
    "BoundSemanticCall",
    "BoundSemanticIntegration",
    "BoundRobotSkillProfile",
    "ControlPartEndpoint",
    "ControlPartEndpointAdapter",
    "DeclarativeValue",
    "EndpointResolution",
    "GRASP_AFFORDANCE_CAPABILITY",
    "HandOver",
    "LinkedSemanticCall",
    "PLACE_IN_AFFORDANCE_CAPABILITY",
    "PLACE_ON_AFFORDANCE_CAPABILITY",
    "PathPart",
    "Pick",
    "Place",
    "PlaceRelationTarget",
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
    "SemanticCallSpec",
    "SemanticDiagnostic",
    "SemanticIntegrationManifest",
    "SemanticPose",
    "SemanticValidationError",
    "SkillPolicyPreset",
    "UnsupportedSkillError",
    "UnsupportedSceneAffordanceError",
    "builtin_semantic_call_catalog",
]
