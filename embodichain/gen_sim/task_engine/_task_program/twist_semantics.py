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
"""Explicit source-owned knob roles, independent of asset names and identity."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

__all__: list[str] = []

_SCHEMA = "gen_sim.twist-control/v1"
_EMBEDDED_KEY = "gen_sim:twistControl"
_ROLE_KEYS = {"schema", "joint", "grip", "pointer", "settings"}


class MissingTwistSemanticsError(ValueError):
    """The asset has no explicit twist-control annotation."""


class InvalidTwistSemanticsError(ValueError):
    """An authored annotation or its USD ownership cannot be qualified."""


@dataclass(frozen=True, slots=True)
class TwSemantics:
    source_sha256: str
    semantics_sha256: str
    joint_path: str
    grip_path: str
    pointer_path: str
    settings: tuple[tuple[int, tuple[str, ...]], ...]
    sidecar_sha256: str | None = None


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InvalidTwistSemanticsError(
                f"E8 annotation contains duplicate {key!r}."
            )
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise InvalidTwistSemanticsError(f"E8 annotation contains non-finite {value}.")


def _path(value: Any) -> str:
    from pxr import Sdf

    if type(value) is not str or not value or value.strip() != value:
        raise InvalidTwistSemanticsError("E8 roles require exact absolute prim paths.")
    valid, _ = Sdf.Path.IsValidPathString(value)
    if not valid:
        raise InvalidTwistSemanticsError("E8 role path is not a valid USD prim path.")
    path = Sdf.Path(value)
    if (
        not path.IsAbsolutePath()
        or not path.IsPrimPath()
        or path == Sdf.Path.absoluteRootPath
        or str(path) != value
        or path.ContainsPrimVariantSelection()
    ):
        raise InvalidTwistSemanticsError(
            "E8 roles require canonical absolute prim paths."
        )
    return value


def _normalise(value: Any, source_digest: str, *, sidecar: bool) -> dict[str, Any]:
    from pxr import Vt

    expected = _ROLE_KEYS | ({"source_sha256"} if sidecar else set())
    if type(value) is not dict or set(value) != expected:
        raise InvalidTwistSemanticsError(
            "E8 annotation has missing or unexpected fields."
        )
    if value["schema"] != _SCHEMA:
        raise InvalidTwistSemanticsError("E8 annotation schema is not supported.")
    if sidecar and (
        type(value["source_sha256"]) is not str
        or len(value["source_sha256"]) != 64
        or any(
            character not in "0123456789abcdef" for character in value["source_sha256"]
        )
        or value["source_sha256"] != source_digest
    ):
        raise InvalidTwistSemanticsError(
            "E8 annotation source digest is stale or invalid."
        )
    settings = value["settings"]
    if type(settings) is not dict or not settings:
        raise InvalidTwistSemanticsError(
            "E8 settings require a nonempty label mapping."
        )
    selected: list[tuple[int, list[str]]] = []
    used: set[str] = set()
    for key, parts in settings.items():
        try:
            number = int(key)
        except (TypeError, ValueError) as error:
            raise InvalidTwistSemanticsError(
                "E8 setting keys must be canonical integers."
            ) from error
        if type(key) is not str or str(number) != key:
            raise InvalidTwistSemanticsError(
                "E8 setting keys must be canonical integers."
            )
        allowed = type(parts) is list or (
            not sidecar and isinstance(parts, (tuple, Vt.StringArray))
        )
        if not allowed or not parts:
            raise InvalidTwistSemanticsError(
                "E8 setting labels require nonempty mesh lists."
            )
        paths = [_path(part) for part in parts]
        if len(set(paths)) != len(paths) or used.intersection(paths):
            raise InvalidTwistSemanticsError(
                "E8 label parts must be unique across settings."
            )
        used.update(paths)
        selected.append((number, sorted(paths)))
    return {
        "schema": _SCHEMA,
        "joint": _path(value["joint"]),
        "grip": _path(value["grip"]),
        "pointer": _path(value["pointer"]),
        "settings": {str(number): parts for number, parts in sorted(selected)},
    }


def _owner(prim: Any) -> Any:
    from pxr import UsdPhysics

    current = prim.GetParent()
    while current and not current.IsPseudoRoot():
        if current.HasAPI(UsdPhysics.RigidBodyAPI):
            return current
        current = current.GetParent()
    return None


def _validate_mesh(stage: Any, path: str, owner: Any, cache: Any, role: str) -> Any:
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(path)
    if (
        not prim
        or not prim.IsActive()
        or not prim.IsDefined()
        or not prim.IsA(UsdGeom.Mesh)
        or _owner(prim) != owner
    ):
        raise InvalidTwistSemanticsError(
            f"E8 {role} must be a mesh owned by its selected body."
        )
    points_attr = UsdGeom.Mesh(prim).GetPointsAttr()
    points = points_attr.Get()
    matrix = cache.GetLocalToWorldTransform(prim)
    if (
        points is None
        or len(points) < 3
        or any(not math.isfinite(float(value)) for point in points for value in point)
        or any(not math.isfinite(float(value)) for row in matrix for value in row)
        or not math.isfinite(matrix.GetDeterminant())
        or abs(matrix.GetDeterminant()) < 1e-15
        or any(
            not math.isfinite(float(value))
            for point in points
            for value in matrix.Transform(Gf.Vec3d(point))
        )
    ):
        raise InvalidTwistSemanticsError(
            f"E8 {role} geometry must contain finite points and transforms."
        )
    current = prim
    while current and not current.IsPseudoRoot():
        if UsdGeom.Xformable(current).GetTimeSamples():
            raise InvalidTwistSemanticsError(
                "E8 roles require static source transforms."
            )
        current = current.GetParent()
    if points_attr.GetNumTimeSamples():
        raise InvalidTwistSemanticsError("E8 roles require static source mesh points.")
    return prim


def _validate_roles(stage: Any, roles: dict[str, Any]) -> None:
    from pxr import UsdGeom, UsdPhysics

    joint_prim = stage.GetPrimAtPath(roles["joint"])
    if (
        not joint_prim
        or not joint_prim.IsActive()
        or not joint_prim.IsDefined()
        or not joint_prim.IsA(UsdPhysics.RevoluteJoint)
    ):
        raise InvalidTwistSemanticsError(
            "E8 joint path must select one revolute joint."
        )
    joint = UsdPhysics.RevoluteJoint(joint_prim)
    parent_paths, child_paths = (
        joint.GetBody0Rel().GetTargets(),
        joint.GetBody1Rel().GetTargets(),
    )
    if (
        joint.GetJointEnabledAttr().Get() is not True
        or len(parent_paths) != 1
        or len(child_paths) != 1
        or parent_paths[0] == child_paths[0]
    ):
        raise InvalidTwistSemanticsError("E8 requires one enabled parent/child joint.")
    parent, child = stage.GetPrimAtPath(parent_paths[0]), stage.GetPrimAtPath(
        child_paths[0]
    )
    for body in (parent, child):
        if (
            not body
            or not body.IsActive()
            or not body.HasAPI(UsdPhysics.RigidBodyAPI)
            or UsdPhysics.RigidBodyAPI(body).GetRigidBodyEnabledAttr().Get() is not True
        ):
            raise InvalidTwistSemanticsError(
                "E8 joint bodies must be enabled rigid bodies."
            )
    roots = [
        prim
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        and parent.GetPath().HasPrefix(prim.GetPath())
        and child.GetPath().HasPrefix(prim.GetPath())
        and joint_prim.GetPath().HasPrefix(prim.GetPath())
    ]
    default = stage.GetDefaultPrim()
    if len(roots) != 1 or not roots[0].GetPath().HasPrefix(default.GetPath()):
        raise InvalidTwistSemanticsError(
            "E8 roles require one owned articulation root."
        )
    if any(
        prim.IsA(UsdPhysics.Joint)
        and prim != joint_prim
        and any(
            target in (parent_paths[0], child_paths[0])
            for target in UsdPhysics.Joint(prim).GetBody1Rel().GetTargets()
        )
        for prim in stage.Traverse()
    ):
        raise InvalidTwistSemanticsError(
            "E8 requires a root parent and a directly owned knob."
        )
    cache = UsdGeom.XformCache()
    grip = _validate_mesh(stage, roles["grip"], child, cache, "grip")
    if (
        not grip.HasAPI(UsdPhysics.CollisionAPI)
        or UsdPhysics.CollisionAPI(grip).GetCollisionEnabledAttr().Get() is not True
    ):
        raise InvalidTwistSemanticsError("E8 grip must be an enabled target collider.")
    if roles["grip"] == roles["pointer"]:
        raise InvalidTwistSemanticsError(
            "E8 grip and pointer must have distinct explicit roles."
        )
    _validate_mesh(stage, roles["pointer"], child, cache, "pointer")
    for paths in roles["settings"].values():
        for path in paths:
            _validate_mesh(stage, path, parent, cache, "label")


def resolve_twist_semantics(path: str | Path) -> TwSemantics:
    """Qualify explicit source roles without choosing from names or a registry.

    A source-side ``.twist.json`` declares schema, source SHA256, exact joint,
    grip, pointer and setting-label mesh paths. Alternatively, the USD default
    prim may author ``gen_sim:twistControl`` with the same fields except the
    self-referential source digest. Two present annotations must agree. The
    semantic digest binds normalized roles, while the separate source digest
    covers all self-contained USD geometry and physics bytes. The optional
    sidecar digest records the exact qualified bytes for later freshness checks,
    rather than rereading a potentially replaced annotation as a new baseline.
    """
    from pxr import Sdf, Usd, UsdGeom

    path = Path(path)
    sidecar_path = path.with_suffix(".twist.json")
    if path.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        if not sidecar_path.exists():
            raise MissingTwistSemanticsError(
                "E8 non-USD source has no twist-control semantics."
            )
        raise InvalidTwistSemanticsError(
            "E8 semantics require a self-contained USD source."
        )
    try:
        source_bytes = path.read_bytes()
        source_digest = hashlib.sha256(source_bytes).hexdigest()
        # Fresh anonymous layers prevent a cached identifier from serving old
        # geometry after source bytes and their integrity digest have changed.
        layer = Sdf.Layer.OpenAsAnonymous(str(path))
        stage = Usd.Stage.Open(layer) if layer is not None else None
    except Exception as error:
        raise InvalidTwistSemanticsError("E8 source USD cannot be read.") from error
    if stage is None:
        raise InvalidTwistSemanticsError("E8 source USD cannot be read.")
    default = stage.GetDefaultPrim()
    embedded = default.GetCustomDataByKey(_EMBEDDED_KEY) if default else None
    sidecar_bytes: bytes | None = None
    annotations: list[dict[str, Any]] = []
    if sidecar_path.exists():
        try:
            sidecar_bytes = sidecar_path.read_bytes()
            payload = json.loads(
                sidecar_bytes,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        except (OSError, ValueError, UnicodeError) as error:
            if isinstance(error, InvalidTwistSemanticsError):
                raise
            raise InvalidTwistSemanticsError(
                "E8 sidecar must contain valid strict JSON."
            ) from error
        annotations.append(_normalise(payload, source_digest, sidecar=True))
    if embedded is not None:
        annotations.append(_normalise(embedded, source_digest, sidecar=False))
    if not annotations:
        raise MissingTwistSemanticsError(
            "E8 source has no explicit twist-control semantics."
        )
    if UsdGeom.GetStageMetersPerUnit(stage) != 1.0:
        raise InvalidTwistSemanticsError("E8 source must be metre-authored USD.")
    if (
        layer.subLayerPaths
        or layer.GetExternalReferences()
        or any(
            item not in (layer, stage.GetSessionLayer())
            for item in stage.GetUsedLayers()
        )
    ):
        raise InvalidTwistSemanticsError(
            "E8 source must be self-contained and hash-bound."
        )
    if not default:
        raise InvalidTwistSemanticsError("E8 source requires an explicit default prim.")
    if any(item != annotations[0] for item in annotations[1:]):
        raise InvalidTwistSemanticsError("E8 sidecar and embedded roles conflict.")
    roles = annotations[0]
    _validate_roles(stage, roles)
    try:
        unchanged = path.read_bytes() == source_bytes and (
            sidecar_path.read_bytes() == sidecar_bytes
            if sidecar_bytes is not None
            else not sidecar_path.exists()
        )
    except OSError as error:
        raise InvalidTwistSemanticsError(
            "E8 annotation changed during qualification."
        ) from error
    if not unchanged:
        raise InvalidTwistSemanticsError(
            "E8 source or annotation changed during qualification."
        )
    canonical = json.dumps(
        roles, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return TwSemantics(
        source_digest,
        hashlib.sha256(canonical).hexdigest(),
        roles["joint"],
        roles["grip"],
        roles["pointer"],
        tuple((int(key), tuple(parts)) for key, parts in roles["settings"].items()),
        sidecar_sha256=(
            hashlib.sha256(sidecar_bytes).hexdigest()
            if sidecar_bytes is not None
            else None
        ),
    )
