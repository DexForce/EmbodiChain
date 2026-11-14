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

"""Non-owning public extension ports for the native Gym expansion host."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol
from .contracts import CandidateIdentity, SceneCase, ValidationResult
from .recipes import ExpansionRecipe

__all__ = ["ExpansionValidator", "ExpansionRowContext", "GymExpansionHostExtensions"]


class ExpansionValidator(Protocol):
    """Measured validation port shared by geometry and native-contact checks.

    Unavailable evidence must reject. Implementations never step/reset the World;
    the host calls prepare after settling and sample on every measured boundary.
    """

    initial: Mapping[int, bool]
    terminal: Mapping[int, ValidationResult]
    results: Mapping[int, Any]

    def prepare(self, snapshot: Any = None) -> None:
        """Reset measured evidence after the host has settled its cohort."""
        ...

    def sample(self, ids: tuple[int, ...], action: Any, snapshot: Any = None) -> None:
        """Observe the original control boundary and its physical substeps."""
        ...

    def close(self) -> None:
        """Release only owned observers/resources; the host retains its World."""
        ...


@dataclass(frozen=True)
class ExpansionRowContext:
    """Immutable identity supplied to a per-row plan-transform factory."""

    env_id: int
    identity: CandidateIdentity
    ordinal: int
    attempt_id: int
    directory: Path
    scene_case: SceneCase
    recipe: ExpansionRecipe


@dataclass(frozen=True)
class GymExpansionHostExtensions:
    """Runtime callbacks; construction, collection and persistence stay native.

    Args:
        collection_renderer: Explicit headless collection renderer, or None for
            the native sensor/media default.
        scene_source_factory: Receives the live environment and returns a source
            exposing candidate(ordinal); it cannot own another World.
        validator_factory: Receives environment and artifact root; returns the
            same measured port for either physical geometry or native contacts.
        recipe_transform: Receives the native immutable recipe and returns its
            task-owned operations. Preparation verifies it again before mutation.
        plan_transform_factory: Receives environment and immutable row identity;
            returns an ordinary Task Program transform factory. Its record metadata
            and hashed plan attachments are archived by the native episode owner.
        require_plan_transform: Reject completed programs without transform records.
    """

    collection_renderer: str | None = None
    scene_source_factory: Callable[[Any], Any] | None = None
    validator_factory: Callable[[Any, Path], ExpansionValidator] | None = None
    recipe_transform: Callable[[ExpansionRecipe], ExpansionRecipe] | None = None
    plan_transform_factory: Callable[[Any, ExpansionRowContext], Any] | None = None
    require_plan_transform: bool = False

    def __post_init__(self) -> None:
        if self.collection_renderer is not None and (
            type(self.collection_renderer) is not str
            or not self.collection_renderer.strip()
        ):
            raise ValueError("collection_renderer must be a nonempty renderer name")
        for name in (
            "scene_source_factory",
            "validator_factory",
            "recipe_transform",
            "plan_transform_factory",
        ):
            value = getattr(self, name)
            if value is not None and not callable(value):
                raise TypeError(f"{name} must be callable")
        if type(self.require_plan_transform) is not bool or (
            self.require_plan_transform and self.plan_transform_factory is None
        ):
            raise ValueError("required plan transforms need an explicit factory")
