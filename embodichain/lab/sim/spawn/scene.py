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

AssetBindCallback = Callable[[Any, tuple[Any, ...]], None]
_AssetKind = Literal[
    "rigid_object",
    "articulation",
    "soft_object",
    "cloth_object",
    "light",
]


@dataclass(slots=True)
class _AssetDeclaration:
    kind: _AssetKind
    descriptor: Any
    on_bind: AssetBindCallback | None


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
        self.result: Any | None = None
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
        on_bind: AssetBindCallback | None = None,
    ) -> None:
        """Add a descriptor to the Builder and remember its facade binding."""
        if uid in self._assets:
            raise ValueError(f"Spawn asset uid is already declared: {uid!r}.")
        declaration = _AssetDeclaration(
            kind=kind,
            descriptor=descriptor,
            on_bind=on_bind,
        )

        if kind == "light" and self.result is not None:
            arenas = self.arena_names if descriptor.per_env else ("default",)
            handles = tuple(
                self.result.add_light(descriptor, arena_name=arena) for arena in arenas
            )
            if on_bind is not None:
                on_bind(self.result, handles)
        else:
            add_name = {
                "rigid_object": "add_object",
                "articulation": "add_articulation",
                "soft_object": "add_soft_object",
                "cloth_object": "add_cloth_object",
                "light": "add_light",
            }[kind]
            declaration.descriptor = getattr(self.builder, add_name)(descriptor)
        self._assets[uid] = declaration

    def track(
        self,
        kind: _AssetKind,
        uid: str,
        descriptor: Any,
        *,
        on_bind: AssetBindCallback | None = None,
    ) -> None:
        """Track a descriptor that was already added to ``SceneBuilder``."""
        if uid in self._assets:
            raise ValueError(f"Spawn asset uid is already declared: {uid!r}.")
        self._assets[uid] = _AssetDeclaration(kind, descriptor, on_bind)

    def remove(self, uid: str) -> None:
        """Remove a declared asset from its DexSim owner."""
        declaration = self._assets[uid]
        if declaration.kind in {"soft_object", "cloth_object"}:
            raise NotImplementedError(
                "DexSim Spawn does not yet expose pending removal for "
                f"{declaration.kind.replace('_', ' ')}."
            )
        if declaration.kind == "light" and self.result is not None:
            for path in self._paths(declaration):
                self.result.remove_light(path)
        else:
            remove_name = {
                "rigid_object": "remove_object",
                "articulation": "remove_articulation",
                "light": "remove_light",
            }[declaration.kind]
            removed = getattr(self.builder, remove_name)(declaration.descriptor.name)
            if removed is None:
                raise KeyError(f"Spawn asset is absent from SceneBuilder: {uid!r}.")
        del self._assets[uid]

    def materialize(self) -> Any:
        """Finalize once or let ``SpawnResult`` consume pending changes."""
        if self.result is None:
            self.result = self.builder.finalize()
        elif self.builder.has_pending_changes or self.result.needs_rebuild:
            self.result = self.result.rebuild(self.builder)
        return self.result

    def bind(self) -> None:
        """Resolve current Spawn handles and bind every declared facade."""
        if self.result is None:
            raise RuntimeError("Spawn scene must be materialized before binding.")

        for declaration in self._assets.values():
            if declaration.on_bind is None:
                continue
            paths = self._paths(declaration)
            handles = tuple(self.result.handles[path] for path in paths)
            declaration.on_bind(self.result, handles)

    def close(self) -> None:
        """Release Spawn resources and facade callback references."""
        if self.result is not None:
            self.result.close()
        self.result = None
        self._assets.clear()

    def _paths(self, declaration: _AssetDeclaration) -> tuple[str, ...]:
        name = declaration.descriptor.name
        if not declaration.descriptor.per_env:
            return (name,)
        return tuple(f"{arena}/{name}" for arena in self.arena_names)
