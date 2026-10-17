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

"""Registered nominal expert sources shared by Task Program integrations.

Factories provide plans; Task Program retains execution, effects, recovery and
persistence. Declarations contain stable IDs and immutable parameters only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Protocol, runtime_checkable, TYPE_CHECKING

from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    ActionPlan,
    PlanningContext,
    ResolvedActionRequest,
    SceneProvider,
    NominalPlanProvider,
)
from embodichain.lab.task_program.runtime.expansion import (
    TaskProgramPlanRequest,
    TaskProgramNominalPlanFactory,
)
from embodichain.lab.task_program.semantics.profiles import (
    RobotSkillProfile,
    SkillPolicyPreset,
)
from .extensions import validate_immutable_extension_declaration

if TYPE_CHECKING:
    from embodichain.lab.task_program.compiler.program import CompiledTaskProgram

__all__ = [
    "ExpertSourceProvider",
    "ExpertSourceProviderFactory",
    "ExpertSourceSelection",
    "ExpertSourceRegistry",
]


def _id(value: str) -> None:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError("Expert source IDs must be nonempty stripped strings")


@runtime_checkable
class ExpertSourceProvider(Protocol):
    """Call-local source of a complete, side-effect-free nominal plan."""

    def plan(
        self, request: ResolvedActionRequest, context: PlanningContext
    ) -> ActionPlan:
        """Return a complete plan or raise; source failure never falls back."""


@runtime_checkable
class ExpertSourceProviderFactory(Protocol):
    """Immutable, catalog-owned declaration of a fresh source provider."""

    provider_id: ClassVar[str]
    revision: ClassVar[str]
    capabilities: ClassVar[frozenset[str]]

    def create(
        self,
        *,
        call_id: str,
        scene_provider: SceneProvider,
        robot_profile: RobotSkillProfile,
        control_dt: float,
        engine: AtomicActionEngine,
    ) -> ExpertSourceProvider:
        """Construct a fresh provider for the selected call and host clock."""


@dataclass(frozen=True)
class ExpertSourceSelection:
    """Bind one expanded call occurrence to an exact registered source version.

    Args:
        call_id: Semantic Call kind, for example pick.
        provider_id: Stable registered provider ID.
        revision: Exact provider implementation revision.
        call_index: Global expanded program call index; never a kind-wide wildcard.
        required_capabilities: Explicit generation capabilities required by this binding.
    """

    call_id: str
    provider_id: str
    revision: str
    call_index: int = 0
    required_capabilities: tuple[str, ...] = ("replay", "phase_annotation")

    def __post_init__(self) -> None:
        for value in (self.call_id, self.provider_id, self.revision):
            _id(value)
        if type(self.call_index) is not int or self.call_index < 0:
            raise ValueError("Source call_index must be a nonnegative integer")
        for value in self.required_capabilities:
            _id(value)
        if type(self.required_capabilities) is not tuple or len(
            set(self.required_capabilities)
        ) != len(self.required_capabilities):
            raise ValueError("Source capabilities must be a unique immutable tuple")


@dataclass(frozen=True)
class ExpertSourceRegistry:
    """Exact immutable factories and source selections in one task catalog.

    Args:
        factories: Immutable factory instances, keyed by type-declared ID/revision.
        selections: Exact semantic call occurrences using these factories.
    """

    factories: tuple[ExpertSourceProviderFactory, ...] = ()
    selections: tuple[ExpertSourceSelection, ...] = ()

    def __post_init__(self) -> None:
        if type(self.factories) is not tuple or type(self.selections) is not tuple:
            raise TypeError("Expert source registration requires immutable tuples")
        keys = []
        for factory in self.factories:
            if not isinstance(factory, ExpertSourceProviderFactory):
                raise TypeError("Expert source factory does not implement the protocol")
            validate_immutable_extension_declaration(
                factory, field_name="expert source factory"
            )
            for name in ("provider_id", "revision"):
                _id(getattr(type(factory), name, None))
            caps = getattr(type(factory), "capabilities", None)
            if type(caps) is not frozenset:
                raise TypeError(
                    "Factory capabilities must be a type-declared frozenset"
                )
            for value in caps:
                _id(value)
            keys.append((factory.provider_id, factory.revision))
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate expert source provider version")
        indices = []
        for selected in self.selections:
            if type(selected) is not ExpertSourceSelection:
                raise TypeError("Invalid expert source selection")
            key = selected.provider_id, selected.revision
            if key not in keys:
                raise ValueError(f"Unregistered expert source {key}")
            factory = self.factories[keys.index(key)]
            missing = set(selected.required_capabilities) - factory.capabilities
            if missing:
                raise ValueError(
                    f"Expert source lacks required capabilities: {sorted(missing)}"
                )
            indices.append(selected.call_index)
        if len(set(indices)) != len(indices):
            raise ValueError("Duplicate source call occurrence")
        if set(keys) != {(s.provider_id, s.revision) for s in self.selections}:
            raise ValueError("Every registered expert source must be selected")

    def validate_policy(self, preset: SkillPolicyPreset) -> None:
        """Reject replay-incompatible recovery policies before simulator construction.

        Args:
            preset: Selected Task Program execution preset.
        """
        for selected in self.selections:
            factory = self._factory(selected)
            recovery = preset.recovery_policy
            if "replanning" not in factory.capabilities and (
                recovery.max_replans or recovery.max_action_retries
            ):
                raise ValueError(
                    "Replay-only expert sources require max_replans=0 and max_action_retries=0"
                )
            workflow = getattr(preset, "workflow_recovery_policy", None)
            if (
                "replanning" not in factory.capabilities
                and workflow is not None
                and workflow.max_recovery_attempts
            ):
                raise ValueError(
                    "Replay-only expert sources require workflow recovery to be disabled"
                )
            if (
                "replanning" not in factory.capabilities
                and preset.motion_policy.dynamic_collision_mode.value == "required"
            ):
                raise ValueError(
                    "Replay-only sources cannot satisfy dynamic collision replanning"
                )

    def validate_program(self, program: CompiledTaskProgram) -> None:
        """Validate exact call coverage and supported sequencing before assembly.

        Args:
            program: Immutable compiled Task Program to execute.
        """
        occurrences = {
            call.call_index: call.call.semantic_id
            for segment in program.iter_segments()
            for call in segment.calls
        }
        for selected in self.selections:
            if occurrences.get(selected.call_index) != selected.call_id:
                raise ValueError(
                    "Expert source selection does not match an expanded program call"
                )
        if self.selections and any(
            segment.parallel_block is not None for segment in program.iter_segments()
        ):
            raise ValueError(
                "External expert sources currently require sequential program segments"
            )

    def _factory(self, selected: ExpertSourceSelection) -> ExpertSourceProviderFactory:
        return next(
            f
            for f in self.factories
            if (f.provider_id, f.revision) == (selected.provider_id, selected.revision)
        )

    def create_runtime(
        self,
        *,
        scene_provider: SceneProvider,
        robot_profile: RobotSkillProfile,
        control_dt: float,
    ) -> TaskProgramNominalPlanFactory:
        """Create a fresh source selector for one assembled Task Program runtime.

        Args:
            scene_provider: Existing scene snapshot boundary.
            robot_profile: Existing registered embodiment profile.
            control_dt: Host-owned command period.

        Returns:
            Call-scoped nominal-plan factory using this exact registration.
        """
        return _RegisteredNominalSources(
            self, scene_provider, robot_profile, control_dt
        )


class _RegisteredNominalSources:
    def __init__(self, registry, scene_provider, robot_profile, control_dt):
        self.registry, self.scene_provider, self.robot_profile, self.control_dt = (
            registry,
            scene_provider,
            robot_profile,
            control_dt,
        )
        self._providers = []

    def create_nominal_plan_provider(
        self, request: TaskProgramPlanRequest, *, engine: AtomicActionEngine
    ) -> NominalPlanProvider | None:
        selected = next(
            (
                s
                for s in self.registry.selections
                if s.call_index == request.analysis_call_index
            ),
            None,
        )
        if selected is None:
            return None
        if selected.call_id != request.call.semantic_id:
            raise ValueError(
                "Source call kind does not match selected program occurrence"
            )
        factory = self.registry._factory(selected)
        provider = factory.create(
            call_id=selected.call_id,
            scene_provider=self.scene_provider,
            robot_profile=self.robot_profile,
            control_dt=self.control_dt,
            engine=engine,
        )
        if not isinstance(provider, ExpertSourceProvider):
            raise TypeError("Factory returned an invalid expert source provider")
        if any(provider is previous for previous in self._providers):
            raise ValueError("Factory reused live expert source state")
        self._providers.append(provider)
        return provider.plan
