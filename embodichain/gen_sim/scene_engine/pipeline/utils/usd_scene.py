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

"""EmbodiChain entity metadata and indexing for a single USD scene stage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

USD_SCENE_SCHEMA = "embodichain.scene/v2"

__all__ = [
    "USD_SCENE_SCHEMA",
    "UsdEntityDesc",
    "UsdSceneIndex",
    "UsdEntityBinding",
    "UsdSceneBinding",
]


@dataclass(frozen=True, slots=True)
class UsdEntityDesc:
    """One EmbodiChain entity authored on a USD prim.

    Args:
        uid: Stable logical entity identifier used by Gym scene bindings.
        prim_path: Absolute USD prim path containing the entity metadata.
        kind: Entity kind such as ``rigid`` or ``articulation``.
        runtime_name: Optional native runtime name retained for diagnostics.
        body_type: Optional rigid-body mode from the source scene.
        fixed_base: Optional articulation root fixation flag.
        joint_names: Stable articulation joint names in public order.
    """

    uid: str
    prim_path: str
    kind: str
    runtime_name: str | None = None
    body_type: str | None = None
    fixed_base: bool | None = None
    joint_names: tuple[str, ...] = ()


class UsdSceneIndex:
    """Index EmbodiChain entity metadata from one USD stage.

    The index deliberately reads entity identity from USD prim custom data. A
    manifest can remain as a derived compatibility cache, but runtime callers
    do not need a second source of entity names or paths.
    """

    def __init__(
        self,
        *,
        stage_path: Path,
        schema_version: str | None,
        entities: tuple[UsdEntityDesc, ...],
    ) -> None:
        self.stage_path = stage_path
        self.schema_version = schema_version
        self._entities = entities
        self._by_uid = {entity.uid: entity for entity in entities}

    @classmethod
    def load(
        cls,
        stage_path: str | Path,
        *,
        require_schema: bool = False,
    ) -> "UsdSceneIndex":
        """Open a USD stage and index its EmbodiChain entity prims.

        Args:
            stage_path: USD file to open.
            require_schema: Require ``embodichain.scene/v2`` metadata on
                ``/World``. Set this for newly generated runtime scenes;
                leave it false for legacy USD snapshots.

        Returns:
            A deterministic entity index sorted by USD prim path.

        Raises:
            FileNotFoundError: If the stage does not exist.
            ValueError: If the stage is malformed or contains duplicate UIDs.
        """
        resolved_path = Path(stage_path).expanduser().resolve()
        if not resolved_path.is_file():
            raise FileNotFoundError(f"USD scene does not exist: {resolved_path}")

        from pxr import Usd

        stage = Usd.Stage.Open(str(resolved_path))
        if stage is None:
            raise ValueError(f"Could not open USD scene: {resolved_path}")

        world = stage.GetPrimAtPath("/World")
        schema_version = (
            world.GetCustomDataByKey("embodichain:scene_schema")
            if world.IsValid()
            else None
        )
        if schema_version is not None and not isinstance(schema_version, str):
            raise ValueError("USD scene schema metadata must be a string.")
        if require_schema and schema_version != USD_SCENE_SCHEMA:
            raise ValueError(
                f"Expected {USD_SCENE_SCHEMA!r} on /World, got {schema_version!r}."
            )

        entities: list[UsdEntityDesc] = []
        seen_uids: set[str] = set()
        for prim in stage.Traverse():
            uid = prim.GetCustomDataByKey("embodichain:uid")
            if uid is None:
                continue
            if not isinstance(uid, str) or not uid:
                raise ValueError(f"USD entity at {prim.GetPath()} has an invalid UID.")
            if uid in seen_uids:
                raise ValueError(f"USD scene contains duplicate entity UID {uid!r}.")
            kind = prim.GetCustomDataByKey("embodichain:kind")
            if not isinstance(kind, str) or not kind:
                raise ValueError(
                    f"USD entity {uid!r} is missing embodichain:kind metadata."
                )
            runtime_name = prim.GetCustomDataByKey("embodichain:runtime_name")
            body_type = prim.GetCustomDataByKey("embodichain:body_type")
            fixed_base = prim.GetCustomDataByKey("embodichain:fixed_base")
            joint_names = prim.GetCustomDataByKey("embodichain:joint_names")
            if joint_names is None:
                joint_names = ()
            if runtime_name is not None and not isinstance(runtime_name, str):
                raise ValueError(
                    f"USD entity {uid!r} has invalid runtime_name metadata."
                )
            if body_type is not None and not isinstance(body_type, str):
                raise ValueError(f"USD entity {uid!r} has invalid body_type metadata.")
            if fixed_base is not None and not isinstance(fixed_base, bool):
                raise ValueError(f"USD entity {uid!r} has invalid fixed_base metadata.")
            if isinstance(joint_names, (str, bytes)) or not all(
                isinstance(name, str) for name in joint_names
            ):
                raise ValueError(
                    f"USD entity {uid!r} has invalid joint_names metadata."
                )
            seen_uids.add(uid)
            entities.append(
                UsdEntityDesc(
                    uid=uid,
                    prim_path=str(prim.GetPath()),
                    kind=kind,
                    runtime_name=runtime_name,
                    body_type=body_type,
                    fixed_base=fixed_base,
                    joint_names=tuple(joint_names),
                )
            )

        entities.sort(key=lambda entity: entity.prim_path)
        return cls(
            stage_path=resolved_path,
            schema_version=schema_version,
            entities=tuple(entities),
        )

    @property
    def entities(self) -> tuple[UsdEntityDesc, ...]:
        """Return all indexed entities in deterministic prim-path order."""
        return self._entities

    def get(self, uid: str) -> UsdEntityDesc:
        """Return one entity by stable UID."""
        try:
            return self._by_uid[uid]
        except KeyError as exc:
            raise KeyError(f"USD scene entity {uid!r} is not registered.") from exc


@dataclass(frozen=True, slots=True)
class UsdEntityBinding:
    """One USD entity paired with its simulator facade."""

    desc: UsdEntityDesc
    runtime: object


class UsdSceneBinding:
    """Map one USD entity index to DexSim/EmbodiChain runtime objects.

    ``SceneEntityCfg(uid=...)`` objects can be created from this binding, so
    Gym managers resolve the same stable UID that the USD stage uses.
    """

    def __init__(
        self,
        *,
        index: UsdSceneIndex,
        bindings: tuple[UsdEntityBinding, ...],
    ) -> None:
        self.index = index
        self._bindings = bindings
        self._by_uid = {binding.desc.uid: binding for binding in bindings}

    @classmethod
    def load_into(cls, sim: object, stage_path: str | Path) -> "UsdSceneBinding":
        """Import a schema-v2 USD stage and bind every indexed entity."""
        path = Path(stage_path).expanduser().resolve()
        index = UsdSceneIndex.load(path, require_schema=True)
        assets = sim.add_usd(name=path.stem, file_path=str(path))
        bindings: list[UsdEntityBinding] = []
        for desc in index.entities:
            runtime = assets.get(desc.prim_path)
            if runtime is None:
                getter = (
                    sim.get_articulation
                    if desc.kind == "articulation"
                    else sim.get_rigid_object
                )
                runtime = getter(desc.uid)
            if runtime is None:
                raise RuntimeError(
                    f"USD entity {desc.uid!r} has no simulator runtime binding."
                )
            bindings.append(UsdEntityBinding(desc=desc, runtime=runtime))
        return cls(index=index, bindings=tuple(bindings))

    @property
    def entities(self) -> tuple[UsdEntityBinding, ...]:
        """Return all UID-to-runtime bindings in USD prim order."""
        return self._bindings

    def get(self, uid: str) -> UsdEntityBinding:
        """Return the runtime binding for one stable entity UID."""
        try:
            return self._by_uid[uid]
        except KeyError as exc:
            raise KeyError(f"USD scene entity {uid!r} is not bound.") from exc

    def scene_entity_cfg(self, uid: str) -> object:
        """Create the Gym ``SceneEntityCfg`` corresponding to ``uid``."""
        self.get(uid)
        from embodichain.lab.gym.envs.managers.cfg import SceneEntityCfg

        return SceneEntityCfg(uid=uid)
