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

__all__: list[str] = []

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from embodichain.lab.sim.atomic_actions.engine import AtomicActionEngine

from embodichain.lab.task_program.semantics.calls import (
    SemanticCallCatalog,
    builtin_semantic_call_catalog,
)
from embodichain.lab.task_program.semantics.effects import EffectMonitorRegistry
from embodichain.lab.task_program.semantics.integration import (
    SceneManifest,
    SemanticIntegrationManifest,
)
from embodichain.lab.task_program.semantics.profiles import (
    ResourceEndpoint,
    ResourceEndpointAdapter,
    RobotSkillProfile,
)
from embodichain.lab.task_program.semantics.scene import SceneRegistry

from ..compiler.lowering import (
    HandOverPoseProvider,
    RegisteredSemanticLowerer,
    RelationTargetGrounder,
    SemanticCallCompiler,
)
from ..language.schema import TaskProgramIntegrationCfg


@dataclass(frozen=True, slots=True)
class SemanticExecutorComponents:
    """Shared scene, embodiment, engine, and compiler assembly result."""

    integration: TaskProgramIntegrationCfg
    scene_registry: SceneRegistry
    robot_profile: RobotSkillProfile
    manifest: SemanticIntegrationManifest
    engine: AtomicActionEngine
    compiler: SemanticCallCompiler


def assemble_semantic_executor_components(
    scene_registry: SceneRegistry,
    robot_profile: RobotSkillProfile,
    engine: AtomicActionEngine,
    integration: TaskProgramIntegrationCfg,
    *,
    call_catalog: SemanticCallCatalog | None = None,
    endpoint_adapters: (
        Mapping[type[ResourceEndpoint], ResourceEndpointAdapter] | None
    ) = None,
    registered_lowerers: Iterable[RegisteredSemanticLowerer] = (),
    relation_grounders: Iterable[RelationTargetGrounder] = (),
    handover_pose_providers: Iterable[HandOverPoseProvider] = (),
    effect_monitor_registry: EffectMonitorRegistry | None = None,
) -> SemanticExecutorComponents:
    """Bind the canonical semantic compiler components once for every frontend."""
    if type(scene_registry) is not SceneRegistry:
        raise TypeError("scene_registry must be exactly SceneRegistry.")
    if type(robot_profile) is not RobotSkillProfile:
        raise TypeError("robot_profile must be exactly RobotSkillProfile.")
    if not isinstance(engine, AtomicActionEngine):
        raise TypeError("engine must be an AtomicActionEngine.")
    if type(integration) is not TaskProgramIntegrationCfg:
        raise TypeError("integration must be exactly TaskProgramIntegrationCfg.")
    if integration.robot_profile != robot_profile.profile_id:
        raise ValueError(
            "integration.robot_profile must match robot_profile.profile_id."
        )
    selected_integration = TaskProgramIntegrationCfg(
        robot_profile=integration.robot_profile,
        scene_registry=integration.scene_registry,
        runtime_preset=integration.runtime_preset,
    )
    selected_catalog = (
        builtin_semantic_call_catalog() if call_catalog is None else call_catalog
    )
    if type(selected_catalog) is not SemanticCallCatalog:
        raise TypeError("call_catalog must be exactly SemanticCallCatalog or None.")

    manifest = SemanticIntegrationManifest(
        scene=SceneManifest.from_registry(scene_registry),
        robot_profile=robot_profile,
        call_catalog=selected_catalog,
        runtime_preset=selected_integration.runtime_preset,
    )
    integration = manifest.bind(
        scene_registry,
        engine,
        endpoint_adapters=endpoint_adapters,
    )
    compiler = SemanticCallCompiler(
        integration,
        registered_lowerers=registered_lowerers,
        relation_grounders=relation_grounders,
        handover_pose_providers=handover_pose_providers,
        effect_monitor_registry=effect_monitor_registry,
    )
    return SemanticExecutorComponents(
        integration=selected_integration,
        scene_registry=scene_registry,
        robot_profile=robot_profile,
        manifest=manifest,
        engine=engine,
        compiler=compiler,
    )
