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

"""Public, execution-free composition of physical environment declarations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

__all__ = ["ResolvedGymComponents", "resolve_gym_components"]


@dataclass(frozen=True, slots=True)
class ResolvedGymComponents:
    """One detached declaration after physical component expansion.

    Attributes:
        config: Expanded environment/robot/sensor/scene values and untouched
            Task Program selectors. These are not constructed runtime configs.
        embodiment_selected: Whether an embodiment component supplied robot/sensors.
        scene_selected: Whether a component or inline fields supplied scene values.
        embodiment_skill_profile: Optional detached profile for the public Task
            Program deployment composer.
        scene_config: Optional physical scene projection for semantic binding checks.
        usd_scene_inventory: Optional preflighted USD source/resource identity.

    Resolver calls own their nested data independently. The mappings remain
    mutable for explicit declaration assembly; this is not a sealed identity.
    """

    config: dict[str, object]
    embodiment_selected: bool
    scene_selected: bool
    embodiment_skill_profile: dict[str, object] | None
    scene_config: dict[str, object] | None
    usd_scene_inventory: dict[str, object] | None = None

    def validate_scene_bindings(self, binding: Mapping[str, object]) -> None:
        """Check semantic native roots against the resolved physical scene.

        Args:
            binding: Semantic rigid/articulation declarations naming native roots.

        Raises:
            ValueError: No physical scene was selected or a native root is absent.
        """
        from ._component_composition import _validate_scene_binding_targets

        if self.scene_config is None:
            raise ValueError("Scene binding validation requires a physical scene.")
        _validate_scene_binding_targets(
            binding,
            simulation=self.scene_config,
            usd_scene_inventory=self.usd_scene_inventory,
        )


def resolve_gym_components(
    config: Mapping[str, object],
    *,
    base_dir: str | Path,
    component_sources: Mapping[str, Mapping[str, object]] | None = None,
) -> ResolvedGymComponents:
    """Expand physical components without registering or constructing an environment.

    The existing strict component decoders retain field ownership, local path
    resolution, physical backend declarations and duplicate-field rejection.
    Task Program selectors are retained without decoding their runtime contract.
    Use ``load_configured_task_program_deployment`` with the returned skill
    profile for semantic composition, then the normal environment loader for
    complete runtime configuration validation.

    Args:
        config: Inline or component-based runnable environment declaration.
        base_dir: Owning deployment directory for relative component references.
        component_sources: Optional complete selected-component snapshot, keyed
            by owner-relative or absolute component paths. When provided, all
            selected components must be present; component files are never read.
            Relative resource assets still resolve from each component owner.

    Returns:
        Independently owned expanded mappings and component-selection metadata.

    Raises:
        TypeError: A component selector or field has an invalid type.
        ValueError: Component schema, ownership or physical declarations conflict.
        FileNotFoundError: A selected component file cannot be resolved.
    """
    from ._component_composition import _resolve_gym_components

    return _resolve_gym_components(
        config, base_dir=base_dir, component_sources=component_sources
    )
