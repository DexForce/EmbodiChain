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

"""Shared provider-free semantic runtime assembly."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from embodichain.lab.sim.atomic_actions.engine import AtomicActionEngine

from .calls import SemanticCallCatalog, builtin_semantic_call_catalog
from .compiler import (
    HandOverPoseProvider,
    RegisteredSemanticLowerer,
    RelationTargetGrounder,
    SemanticSkillCompiler,
)
from .effects import EffectMonitorRegistry
from .integration import SceneManifest, SemanticIntegrationManifest
from .profiles import ResourceEndpoint, ResourceEndpointAdapter, RobotSkillProfile
from .scene import SceneRegistry


@dataclass(frozen=True, slots=True)
class SemanticRuntimeComponents:
    """Shared scene, embodiment, engine, and compiler assembly result."""

    scene_registry: SceneRegistry
    robot_profile: RobotSkillProfile
    manifest: SemanticIntegrationManifest
    engine: AtomicActionEngine
    compiler: SemanticSkillCompiler


def assemble_semantic_runtime_components(
    scene_registry: SceneRegistry,
    robot_profile: RobotSkillProfile,
    engine: AtomicActionEngine,
    *,
    call_catalog: SemanticCallCatalog | None = None,
    runtime_preset: str | None = None,
    endpoint_adapters: (
        Mapping[type[ResourceEndpoint], ResourceEndpointAdapter] | None
    ) = None,
    registered_lowerers: Iterable[RegisteredSemanticLowerer] = (),
    relation_grounders: Iterable[RelationTargetGrounder] = (),
    handover_pose_providers: Iterable[HandOverPoseProvider] = (),
    effect_monitor_registry: EffectMonitorRegistry | None = None,
) -> SemanticRuntimeComponents:
    """Bind the canonical semantic compiler components once for every frontend."""
    if type(scene_registry) is not SceneRegistry:
        raise TypeError("scene_registry must be exactly SceneRegistry.")
    if type(robot_profile) is not RobotSkillProfile:
        raise TypeError("robot_profile must be exactly RobotSkillProfile.")
    if not isinstance(engine, AtomicActionEngine):
        raise TypeError("engine must be an AtomicActionEngine.")
    selected_catalog = (
        builtin_semantic_call_catalog() if call_catalog is None else call_catalog
    )
    if type(selected_catalog) is not SemanticCallCatalog:
        raise TypeError("call_catalog must be exactly SemanticCallCatalog or None.")

    manifest = SemanticIntegrationManifest(
        scene=SceneManifest.from_registry(scene_registry),
        robot_profile=robot_profile,
        call_catalog=selected_catalog,
        runtime_preset=runtime_preset,
    )
    integration = manifest.bind(
        scene_registry,
        engine,
        endpoint_adapters=endpoint_adapters,
    )
    compiler = SemanticSkillCompiler(
        integration,
        registered_lowerers=registered_lowerers,
        relation_grounders=relation_grounders,
        handover_pose_providers=handover_pose_providers,
        effect_monitor_registry=effect_monitor_registry,
    )
    return SemanticRuntimeComponents(
        scene_registry=scene_registry,
        robot_profile=robot_profile,
        manifest=manifest,
        engine=engine,
        compiler=compiler,
    )
