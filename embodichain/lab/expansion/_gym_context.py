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

"""Explicit native-host context shared by its non-owning components."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING
from collections.abc import Mapping, Callable
from contextlib import ExitStack
from pathlib import Path
import torch
from .configured import ExpansionRunCfg
from .journal import ExpansionJournal, RecipeAttempt
from .recipes import BatchCompatibility, OperatorCapability, ExpansionRecipe
from .contracts import CandidateIdentity, ValidationResult
from .host_extensions import GymExpansionHostExtensions

if TYPE_CHECKING:
    from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv, EmbodiedEnvCfg
    from embodichain.lab.gym.envs.task_program.bridge import TaskProgramDemoBridge
    from embodichain.lab.sim.scene import SceneView, SceneState
    from embodichain.lab.sim.scene.physics import RigidPhysicsState
    from embodichain.lab.sim.motion.expansion.combined import (
        CombinedExpansionProfile,
        CandidateRecipe,
    )
    from embodichain.lab.task_program.compiler.program import CompiledTaskProgram
    from embodichain.lab.task_program.integrations.simulation.real2sim_collection import (
        Real2SimTabletopCollectionCfg,
    )
    from .collision import _MeasuredCollision
    from .scene import JointSceneSource
    from .sources import _EpisodeSource

__all__: list[str] = []

# Positional order is retained for existing host instrumentation.
PreparedRow = tuple[CandidateIdentity, str, str, Path, dict[str, Any]]


@dataclass
class GymHostContext:
    """World handles and cohort evidence; lifecycle ownership stays with GymExpansionHost."""

    cfg: ExpansionRunCfg
    report: Mapping[str, object]
    root: Path
    journal: ExpansionJournal
    source_artifacts: Mapping[str, object]
    capacity: int
    env: EmbodiedEnv | None = None
    collision: _MeasuredCollision | None = None
    appearance: ExitStack | None = None
    environment_cfg: EmbodiedEnvCfg | None = None
    view: SceneView | None = None
    managed: tuple[str, ...] = ()
    baseline: RigidPhysicsState | None = None
    auxiliary_baseline: Mapping[str, torch.Tensor] | None = None
    profile: Real2SimTabletopCollectionCfg | None = None
    scene_source: JointSceneSource | None = None
    motion_profile: CombinedExpansionProfile | None = None
    motion_recipes: tuple[CandidateRecipe, ...] = ()
    names: tuple[str, ...] = ()
    limits: torch.Tensor | None = None
    visual_registry: Mapping[str, dict[str, Any]] | None = None
    timings: dict[str, float] = field(default_factory=dict)
    started: float = 0.0
    compatibility: BatchCompatibility | None = None
    supported_operations: dict[str, OperatorCapability] = field(default_factory=dict)
    initial: SceneState | None = None
    initial_obs: object = None
    physics_before: Mapping[str, torch.Tensor] | None = None
    requests: dict[int, RecipeAttempt] = field(default_factory=dict)
    prepared: dict[int, PreparedRow] = field(default_factory=dict)
    preparation_checks: dict[int, ValidationResult] = field(default_factory=dict)
    program: CompiledTaskProgram | None = None
    services: dict[int, _EpisodeSource] = field(default_factory=dict)
    bridges: dict[int, TaskProgramDemoBridge] = field(default_factory=dict)
    recipe: Callable[[int], ExpansionRecipe] | None = None
    extensions: GymExpansionHostExtensions = field(
        default_factory=GymExpansionHostExtensions
    )
