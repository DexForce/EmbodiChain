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

"""Thin EmbodiChain coordination around DexSim Spawn."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

__all__ = ["SpawnScene"]

_AssetKind = Literal[
    "rigid_object",
    "rigid_object_group",
    "articulation",
    "soft_object",
    "cloth_object",
]


@dataclass(slots=True)
class _AssetDeclaration:
    kind: _AssetKind
    descriptor: Any
    facade: Any | None
    source_configurator: Callable[[Any], None] | None = None


class SpawnScene:
    """Map EmbodiChain asset declarations onto one DexSim Spawn scene.

    DexSim's ``SceneBuilder`` and ``SpawnResult`` own lifecycle state and
    revisions.  This class only remembers how logical asset ids map to Spawn
    paths and how the resulting handles bind back into EmbodiChain facades.
    """

    def __init__(
        self,
        world: Any,
        *,
        num_envs: int,
        spacing: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        from dexsim.spawn import SceneBuilder

        self.builder = SceneBuilder(world)
        self.builder.replicate(
            count=num_envs,
            spacing=spacing,
            name_format="arena_{i}",
        )
        self._assets: dict[str, _AssetDeclaration] = {}

    @property
    def arena_names(self) -> tuple[str, ...]:
        """Names of the replicated per-environment Arenas."""
        return tuple(self.builder.replicate_plan.env_names())

    def __contains__(self, uid: str) -> bool:
        return uid in self._assets

    def declare(
        self,
        kind: _AssetKind,
        uid: str,
        descriptor: Any,
        *,
        facade: Any | None = None,
        configure_source: Callable[[Any], None] | None = None,
    ) -> None:
        """Add a descriptor and associate it with an EmbodiChain facade."""
        if uid in self._assets:
            raise ValueError(f"Spawn asset uid is already declared: {uid!r}.")
        declaration = _AssetDeclaration(
            kind=kind,
            descriptor=descriptor,
            facade=facade,
            source_configurator=configure_source,
        )

        if kind == "rigid_object_group":
            declaration.descriptor = tuple(
                self.builder.add_object(member) for member in descriptor
            )
        else:
            if (
                kind == "articulation"
                and configure_source is not None
                and self.builder.is_finalized
            ):
                self.builder.resolve_articulation_source(descriptor)
                configure_source(descriptor)
                declaration.source_configurator = None
            add_name = {
                "rigid_object": "add_object",
                "articulation": "add_articulation",
                "soft_object": "add_soft_object",
                "cloth_object": "add_cloth_object",
            }[kind]
            declaration.descriptor = getattr(self.builder, add_name)(descriptor)
        self._assets[uid] = declaration
        handles = self.handles(uid)
        if facade is not None and handles:
            facade.attach_spawn_handles(handles)

    def resolve_sources(self) -> None:
        """Resolve and configure declarations before backend materialization."""
        if self.builder.is_finalized:
            return
        self.builder.resolve_sources()
        for declaration in self._assets.values():
            configure = declaration.source_configurator
            if configure is None:
                continue
            configure(declaration.descriptor)
            declaration.source_configurator = None

    def track(
        self,
        kind: _AssetKind,
        uid: str,
        descriptor: Any,
        *,
        facade: Any | None = None,
    ) -> None:
        """Track a descriptor that was already added to ``SceneBuilder``."""
        if uid in self._assets:
            raise ValueError(f"Spawn asset uid is already declared: {uid!r}.")
        declaration = _AssetDeclaration(kind, descriptor, facade)
        self._assets[uid] = declaration
        handles = self.handles(uid)
        if facade is not None and handles:
            facade.attach_spawn_handles(handles)

    def remove(self, uid: str) -> None:
        """Remove a declared asset from its DexSim owner."""
        declaration = self._assets[uid]
        if declaration.kind in {"soft_object", "cloth_object"}:
            raise NotImplementedError(
                "DexSim Spawn does not yet expose pending removal for "
                f"{declaration.kind.replace('_', ' ')}."
            )
        if declaration.kind == "rigid_object_group":
            for member in declaration.descriptor:
                self.builder.remove_object(member.name)
        else:
            remove_name = {
                "rigid_object": "remove_object",
                "articulation": "remove_articulation",
            }[declaration.kind]
            removed = getattr(self.builder, remove_name)(declaration.descriptor.name)
            if removed is None:
                raise KeyError(f"Spawn asset is absent from SceneBuilder: {uid!r}.")
        del self._assets[uid]

    def commit(self) -> Any:
        """Finalize once or let ``SpawnResult`` consume pending changes."""
        if not self.builder.is_finalized:
            self.resolve_sources()
            return self.builder.finalize()
        result = self.builder.result
        assert result is not None
        if self.builder.has_pending_changes or result.needs_rebuild:
            self.builder.result = result.rebuild(self.builder)
        return self.builder.result

    def bind(self) -> None:
        """Complete post-finalize runtime binding for declared facades.

        Native entity creation belongs to ``SceneBuilder`` and its backend
        adapter. This method only attaches handles that were unavailable during
        declaration, then lets each facade create its result-dependent
        Batch/Data state through ``bind_spawn()``. Eager Default handles may
        already be attached; deferred Newton handles are resolved here.
        """
        result = self.builder.result
        if result is None or not self.builder.is_finalized:
            raise RuntimeError("Spawn scene must be materialized before binding.")

        for uid, declaration in self._assets.items():
            facade = declaration.facade
            if facade is None or not facade.is_declared:
                continue
            if not facade._entities:
                facade.attach_spawn_handles(self.handles(uid))
            facade.bind_spawn(result)

    def close(self) -> None:
        """Release Spawn resources and facade references."""
        result = self.builder.result
        if result is not None:
            result.close()
        self.builder.result = None
        self._assets.clear()

    def handles(self, uid: str) -> tuple[Any, ...]:
        """Return currently materialized handles for one logical asset."""
        result = self.builder.result
        if result is None:
            return ()
        declaration = self._assets[uid]
        if declaration.kind == "rigid_object_group":
            paths = tuple(
                f"{arena}/{member.name}"
                for arena in self.arena_names
                for member in declaration.descriptor
            )
        elif declaration.descriptor.per_env:
            paths = tuple(
                f"{arena}/{declaration.descriptor.name}" for arena in self.arena_names
            )
        else:
            paths = (declaration.descriptor.name,)
        if any(path not in result.handles for path in paths):
            return ()
        return tuple(result.handles[path] for path in paths)
