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

from ._usd_assets import _UsdTextureFiles
from .._material_presets import (
    _AssetMaterialDesc,
    _MaterialResources,
    _validate_backend,
)

__all__ = ["SpawnScene"]

_AssetKind = Literal[
    "visual_object",
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
    declaration_only: bool = False


class SpawnScene:
    """Map EmbodiChain asset declarations onto one DexSim Spawn scene.

    DexSim owns declaration materialization, stable handles, and topology
    revisions. EmbodiChain resolves and configures source metadata before the
    first backend build so Newton does not materialize an articulation twice.
    """

    def __init__(
        self,
        world: Any,
        *,
        num_envs: int,
        spacing: tuple[float, float, float] = (0.0, 0.0, 0.0),
        renderer: str | None = None,
    ) -> None:
        from dexsim.spawn import SceneBuilder

        self.builder = SceneBuilder(world)
        self._num_envs = num_envs
        self.builder.replicate(
            count=num_envs,
            spacing=spacing,
            name_format="arena_{i}",
            collision_policy="isolated",
        )
        self._assets: dict[str, _AssetDeclaration] = {}
        self._usd_texture_files = _UsdTextureFiles()
        if renderer is None:
            from dexsim.types import Renderer

            renderer = {
                Renderer.NORENDER: "no-render",
                Renderer.HYBRID: "hybrid",
                Renderer.FASTRT: "fast-rt",
                Renderer.OFFLINERT: "rt",
            }[world.get_renderer_type()]
        self._material_resources = _MaterialResources(world, renderer)
        self._resource_material_definitions: dict[str, _AssetMaterialDesc] = {}
        self._pending_material_assignments: (
            dict[tuple[str, int, str | None], list[Any]] | None
        ) = None

    def _resolve_materials(self, materials: dict[str, Any]) -> dict[str, Any]:
        """Resolve package images before DexSim creates native materials."""
        resolved = self._usd_texture_files.resolve(materials)
        for uid, descriptor in resolved.items():
            existing = self._material_definition(uid)
            if existing is not None and existing != descriptor:
                raise ValueError(f"Conflicting visual material definition for {uid!r}.")
        resources = {
            uid: desc
            for uid, desc in resolved.items()
            if isinstance(desc, _AssetMaterialDesc)
        }
        for desc in resources.values():
            _validate_backend(desc, self._material_resources.renderer)
        for uid, desc in resources.items():
            if self._material_resources.renderer != "no-render":
                self._material_resources.resolve(desc)
            self._resource_material_definitions[uid] = desc
        # Native Spawn's simplified MaterialDesc path cannot express these
        # definitions. Bind the exact loaded child directly on each RenderDesc.
        # NoRender retains only metadata and never invokes a native loader.
        return {uid: desc for uid, desc in resolved.items() if uid not in resources}

    def _material_definition(self, uid: str) -> Any | None:
        return getattr(self, "_resource_material_definitions", {}).get(
            uid, self.builder.materials.get(uid)
        )

    def _bind_resource_materials(self, descriptor: Any) -> None:
        definitions = getattr(self, "_resource_material_definitions", {})
        if not definitions or self._material_resources.renderer == "no-render":
            return
        if isinstance(descriptor, (tuple, list)):
            for member in descriptor:
                self._bind_resource_materials(member)
            return
        reference = getattr(descriptor, "material_ref", None)
        if reference in definitions:
            _, descriptor.material = self._material_resources.resolve(
                definitions[reference]
            )
        for name in ("renders", "links", "mesh", "visual_mesh"):
            value = getattr(descriptor, name, None)
            if value is not None:
                self._bind_resource_materials(value)
        for variant in (getattr(descriptor, "env_variants", None) or {}).values():
            self._bind_resource_materials(variant)

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
        variants = getattr(descriptor, "env_variants", None) or {}
        if variants:
            if getattr(self.builder, "backend", None) != "newton":
                raise NotImplementedError(
                    "Per-environment rigid variants require the Newton backend."
                )
            if max(variants) >= self._num_envs:
                raise ValueError(
                    "env_variants contains an index outside the configured environment count."
                )
        self._bind_resource_materials(descriptor)
        self._initialize_facade_declaration(facade)
        declaration = _AssetDeclaration(
            kind=kind,
            descriptor=descriptor,
            facade=facade,
            source_configurator=configure_source,
        )
        if (
            kind == "visual_object"
            and getattr(getattr(self, "_material_resources", None), "renderer", None)
            == "no-render"
        ):
            declaration.declaration_only = True
            self._assets[uid] = declaration
            return

        if kind == "rigid_object_group":
            declaration.descriptor = tuple(
                self.builder.add_object(member) for member in descriptor
            )
        else:
            if (
                kind == "articulation"
                and configure_source is not None
                and (self.builder.is_finalized or self.builder.result is not None)
                and self._can_resolve_before_materialization()
            ):
                self._resolve_articulation_source(descriptor)
                configure_source(descriptor)
                self._bind_resource_materials(descriptor)
                declaration.source_configurator = None
            add_name = {
                "visual_object": "add_object",
                "rigid_object": "add_object",
                "articulation": "add_articulation",
                "soft_object": "add_soft_object",
                "cloth_object": "add_cloth_object",
            }[kind]
            declaration.descriptor = getattr(self.builder, add_name)(descriptor)
        self._assets[uid] = declaration
        self._configure_materialized_source(uid)
        # A Newton replacement retains the old same-path handles until commit.
        # Bind the new facade only after those handles have been replaced.
        pending_newton = (
            getattr(self.builder, "backend", None) == "newton"
            and self.builder.is_finalized
            and self.builder.has_pending_changes
        )
        handles = () if pending_newton else self.handles(uid)
        if facade is not None and handles:
            facade.attach_spawn_handles(handles)

    def resolve_sources(self) -> None:
        """Resolve and configure declarations before backend materialization."""
        if self.builder.is_finalized:
            return

        builder_resolver = getattr(self.builder, "resolve_sources", None)
        if builder_resolver is not None:
            builder_resolver()
        elif getattr(self.builder, "backend", None) == "newton":
            for declaration in self._assets.values():
                if (
                    declaration.kind == "articulation"
                    and declaration.source_configurator is not None
                ):
                    self._resolve_articulation_source(declaration.descriptor)
        else:
            return

        for declaration in self._assets.values():
            configure = declaration.source_configurator
            if configure is None:
                continue
            configure(declaration.descriptor)
            self._bind_resource_materials(declaration.descriptor)
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
        self._initialize_facade_declaration(facade)
        declaration = _AssetDeclaration(kind, descriptor, facade)
        self._assets[uid] = declaration
        handles = self.handles(uid)
        if facade is not None and handles:
            facade.attach_spawn_handles(handles)

    def remove(self, uid: str) -> None:
        """Remove a declared asset from its DexSim owner."""
        declaration = self._assets[uid]
        if getattr(declaration, "declaration_only", False):
            del self._assets[uid]
            return
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
                "visual_object": "remove_object",
                "rigid_object": "remove_object",
                "articulation": "remove_articulation",
            }[declaration.kind]
            removed = getattr(self.builder, remove_name)(declaration.descriptor.name)
            if removed is None:
                raise KeyError(f"Spawn asset is absent from SceneBuilder: {uid!r}.")
        del self._assets[uid]
        pending = getattr(self, "_pending_material_assignments", None)
        if pending is not None:
            self._pending_material_assignments = {
                key: materials for key, materials in pending.items() if key[0] != uid
            }

    def commit(self) -> Any:
        """Finalize once or let the current ``Scene`` consume pending changes."""
        if not self.builder.is_finalized:
            self.resolve_sources()
            result = self.builder.finalize()
        else:
            result = self.builder.result
            assert result is not None
            if self.builder.has_pending_changes or result.needs_rebuild:
                if getattr(self, "_pending_material_assignments", None) is None:
                    self._pending_material_assignments = (
                        self._capture_material_assignments()
                    )
                result = result.rebuild(self.builder)

        for uid in self._assets:
            self._configure_materialized_source(uid)
        self.builder.result = result
        return result

    def _capture_material_assignments(
        self,
    ) -> dict[tuple[str, int, str | None], list[Any]]:
        """Snapshot current per-segment assignments before native resources rebuild."""
        from embodichain.lab.sim.material import _capture_render_materials

        assignments = {}
        for uid, declaration in self._assets.items():
            facade = declaration.facade
            if facade is None or not getattr(facade, "is_spawn_bound", False):
                continue
            links = facade.link_names if declaration.kind == "articulation" else [None]
            for instance_idx, entity in enumerate(self.handles(uid)):
                for link in links:
                    render_body = (
                        entity.get_render_body()
                        if link is None
                        else entity.get_render_body(link)
                    )
                    if render_body is not None:
                        assignments[(uid, instance_idx, link)] = (
                            _capture_render_materials(render_body)
                        )
        return assignments

    def _restore_material_assignments(self) -> None:
        """Replay current assignments; retain the snapshot when preparation must retry."""
        from embodichain.lab.sim.material import _restore_render_materials

        assignments = getattr(self, "_pending_material_assignments", None)
        if assignments is None:
            return
        for (uid, instance_idx, link), materials in assignments.items():
            if uid not in self._assets:
                continue
            entity = self.handles(uid)[instance_idx]
            render_body = (
                entity.get_render_body()
                if link is None
                else entity.get_render_body(link)
            )
            if render_body is None or render_body.get_mesh_count() != len(materials):
                raise RuntimeError(f"Render material topology changed for {uid!r}.")
            _restore_render_materials(render_body, materials)
        self._pending_material_assignments = None

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

        self._restore_material_assignments()
        for uid, declaration in self._assets.items():
            facade = declaration.facade
            if facade is None or not facade.is_declared:
                continue
            if (
                declaration.kind == "visual_object"
                and getattr(
                    getattr(self, "_material_resources", None), "renderer", None
                )
                == "no-render"
            ):
                # NoRender has no transform batch for render-only actors.
                # Keep their geometry/material declarations without binding.
                continue
            if not facade._entities:
                facade.attach_spawn_handles(self.handles(uid))
            facade.bind_spawn(result)

    def prepare_runtime_config(self, result: Any) -> None:
        """Apply facade configuration required before backend initialization.

        Default Direct GPU simulation snapshots some native articulation
        properties during initialization. Articulation facades therefore get
        a narrow pre-bind hook after materialization but before the manager
        initializes backend runtime buffers.
        """
        if result is not self.builder.result or not self.builder.is_finalized:
            raise RuntimeError("Spawn scene must be materialized before runtime setup.")

        for uid, declaration in self._assets.items():
            facade = declaration.facade
            if facade is None or declaration.kind != "articulation":
                continue
            if not facade._entities:
                facade.attach_spawn_handles(self.handles(uid))
            facade._prepare_spawn_runtime_config(result)

    def close(self) -> None:
        """Release Spawn resources and facade references."""
        result = self.builder.result
        self._pending_material_assignments = None
        try:
            if result is not None:
                result.close()
        finally:
            self.builder.result = None
            self._assets.clear()
            self._usd_texture_files.close()
            if hasattr(self, "_material_resources"):
                self._material_resources.close()
                self._resource_material_definitions.clear()

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

    def _initialize_facade_declaration(self, facade: Any | None) -> None:
        """Give a Spawn facade the instance count owned by this scene.

        The duck-typed fallback keeps ``SpawnScene`` usable by lightweight
        callers that only need descriptor tracking and do not expose an
        EmbodiChain object facade.
        """
        if facade is None:
            return
        initialize = getattr(facade, "_initialize_spawn_declaration", None)
        if initialize is not None:
            initialize(self._num_envs)

    def _resolve_articulation_source(self, descriptor: Any) -> None:
        """Resolve one descriptor through the available DexSim boundary."""
        builder_resolver = getattr(
            self.builder,
            "resolve_articulation_source",
            None,
        )
        if builder_resolver is not None:
            builder_resolver(descriptor)
            # Keep the invalid-source COM policy identical when a newer
            # SceneBuilder supplies its own resolver implementation.
            from embodichain.lab.sim.spawn.source import _clear_invalid_source_com

            _clear_invalid_source_com(descriptor)
            return

        from embodichain.lab.sim.spawn.source import resolve_articulation_source

        resolve_articulation_source(self.builder, descriptor)

    def _can_resolve_before_materialization(self) -> bool:
        """Return whether exact source metadata is available before add."""
        return (
            getattr(self.builder, "resolve_articulation_source", None) is not None
            or getattr(self.builder, "backend", None) == "newton"
        )

    def _configure_materialized_source(self, uid: str) -> None:
        """Apply a pending source config to an eager Default articulation."""
        declaration = self._assets[uid]
        configure = declaration.source_configurator
        if configure is None or declaration.kind != "articulation":
            return

        handles = self.handles(uid)
        if not handles:
            return

        result = self.builder.result
        assert result is not None
        if result.backend != "dexsim":
            raise RuntimeError(
                "Newton articulation source configuration must run before "
                "SceneBuilder.finalize()."
            )

        prototype = declaration.descriptor
        source = (
            prototype
            if getattr(prototype, "links", None) or getattr(prototype, "joints", None)
            else handles[0].articulation_desc
        )
        # The Default URDF loader owns the native source mass properties but
        # does not copy them into ``LinkDesc``. Capture them before compiling
        # the sparse overlay, then apply only explicitly configured link
        # physics. This keeps the source tensor intact by default while still
        # presenting both backends with one resolved descriptor contract.
        from embodichain.lab.sim.spawn.source import (
            _apply_dexsim_source_overlay,
            _capture_dexsim_source_physics,
            _retain_dexsim_source_descriptor,
        )

        _capture_dexsim_source_physics(handles[0], source)
        configure(source)
        if getattr(source, "_embodichain_preserve_source_physics", False):
            for handle in handles:
                _retain_dexsim_source_descriptor(handle, source)
        else:
            for handle in handles:
                _apply_dexsim_source_overlay(handle, source)
        declaration.source_configurator = None
