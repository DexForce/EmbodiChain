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
"""Qualified, bundle-owned Default mass compensation without source edits."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .twist_semantics import TwSemantics, resolve_twist_semantics

__all__: list[str] = []

_SCHEMA = "gen_sim.twist-mass-source/v1"
_POLICY = "default_com_s_principal_inertia_s2/v1"
_FIELDS = {
    "schema",
    "policy",
    "backend",
    "key_sha256",
    "source_file",
    "runtime_file",
    "source_sha256",
    "runtime_sha256",
    "source_semantics_sha256",
    "runtime_semantics_sha256",
    "source_sidecar_sha256",
    "runtime_sidecar_sha256",
    "scale",
}


@dataclass(frozen=True, slots=True)
class MassSourceCopy:
    source: Path
    runtime: Path
    manifest: Path
    source_sha256: str
    runtime_sha256: str
    source_semantics_sha256: str
    runtime_semantics_sha256: str
    manifest_sha256: str
    scale: float
    source_sidecar_sha256: str | None
    runtime_sidecar_sha256: str | None


@dataclass(frozen=True, slots=True)
class _Snapshot:
    raw: bytes
    sidecar: bytes | None
    semantics: TwSemantics
    stage: Any


def adjacent_manifest_path(runtime: Path) -> Path:
    """Return the deployment's adjacent, independently pinned lineage file."""
    return Path(runtime).with_suffix(".mass.json")


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("E8 mass manifest contains duplicate fields.")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"E8 mass manifest contains non-finite {value}.")


def _manifest(path: Path) -> tuple[bytes, dict[str, Any]]:
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise ValueError("E8 mass manifest cannot be read as strict JSON.") from error
    if type(value) is not dict or set(value) != _FIELDS:
        raise ValueError("E8 mass manifest has missing or unexpected fields.")
    if (
        value["schema"] != _SCHEMA
        or value["policy"] != _POLICY
        or value["backend"] != "default"
    ):
        raise ValueError("E8 mass manifest requires the qualified Default-only policy.")
    for field in _FIELDS:
        if field.endswith("_sha256"):
            digest = value[field]
            if digest is None and field in {
                "source_sidecar_sha256",
                "runtime_sidecar_sha256",
            }:
                continue
            if (
                type(digest) is not str
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError("E8 mass manifest has invalid SHA-256 identities.")
    if (
        type(value["scale"]) not in (int, float)
        or not math.isfinite(value["scale"])
        or value["scale"] <= 0
    ):
        raise ValueError("E8 mass manifest requires a positive finite uniform scale.")
    return raw, value


def _uniform_scale(body_scale: tuple[float, ...]) -> float:
    if not isinstance(body_scale, (tuple, list, np.ndarray)):
        raise ValueError("E8 mass copy requires a positive finite uniform scale.")
    values = np.asarray(body_scale)
    if values.shape != (3,) or any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        for value in body_scale
    ):
        raise ValueError("E8 mass copy requires a positive finite uniform scale.")
    values = np.asarray(values, dtype=float)
    if (
        values.shape != (3,)
        or not np.isfinite(values).all()
        or (values <= 0).any()
        or not np.all(values == values[0])
    ):
        raise ValueError("E8 mass copy requires a positive finite uniform scale.")
    return float(values[0])


def _reject_asset_paths(layer: Any) -> None:
    from pxr import Sdf

    def inspect(value: Any) -> None:
        if isinstance(value, Sdf.AssetPath):
            if value.path or value.resolvedPath:
                raise ValueError("E8 mass source cannot contain nonempty asset paths.")
        elif isinstance(value, dict):
            for child in value.values():
                inspect(child)
        elif isinstance(value, (list, tuple, Sdf.AssetPathArray)):
            for child in value:
                inspect(child)

    def inspect_spec(path: Any) -> None:
        spec = layer.GetObjectAtPath(path)
        if spec is None:
            return
        for field in spec.ListInfoKeys():
            inspect(spec.GetInfo(field))

    # Layer specs include inactive prims, unselected variants and time samples.
    layer.Traverse(Sdf.Path.absoluteRootPath, inspect_spec)


def _validate_stage(stage: Any) -> None:
    from pxr import UsdGeom, UsdPhysics

    root = stage.GetDefaultPrim()
    if (
        not root
        or not root.HasAPI(UsdPhysics.ArticulationRootAPI)
        or UsdGeom.GetStageUpAxis(stage) != "Z"
        or UsdGeom.GetStageMetersPerUnit(stage) != 1.0
    ):
        raise ValueError(
            "E8 mass source requires a metre Z-up default articulation root."
        )
    layer = stage.GetRootLayer()
    if (
        layer.subLayerPaths
        or layer.GetExternalReferences()
        or any(
            item not in (layer, stage.GetSessionLayer())
            for item in stage.GetUsedLayers()
        )
    ):
        raise ValueError("E8 mass source must be self-contained.")
    _reject_asset_paths(layer)
    bodies = []
    for prim in stage.Traverse():
        if (
            prim.HasAuthoredReferences()
            or prim.HasAuthoredPayloads()
            or prim.IsInstance()
        ):
            raise ValueError(
                "E8 mass source references, payloads or instances are unsupported."
            )
        if any(attr.GetNumTimeSamples() for attr in prim.GetAttributes()):
            raise ValueError("E8 mass source requires static source properties.")
        is_body = prim.HasAPI(UsdPhysics.RigidBodyAPI)
        if prim.HasAPI(UsdPhysics.MassAPI) and not is_body:
            raise ValueError("Inherited or non-body MassAPI is unsupported.")
        if not is_body:
            continue
        bodies.append(prim)
        if not prim.HasAPI(UsdPhysics.MassAPI):
            raise ValueError("Every E8 rigid body requires its own authored MassAPI.")
        api = UsdPhysics.MassAPI(prim)
        attrs = (
            api.GetMassAttr(),
            api.GetCenterOfMassAttr(),
            api.GetDiagonalInertiaAttr(),
            api.GetPrincipalAxesAttr(),
        )
        if any(not attr.HasAuthoredValue() for attr in attrs):
            raise ValueError(
                "E8 mass, COM, principal inertia and axes must be authored."
            )
        if api.GetDensityAttr().HasAuthoredValue():
            raise ValueError("E8 mass copy cannot combine authored mass and density.")
        mass = float(attrs[0].Get())
        com = np.asarray(attrs[1].Get())
        inertia = np.asarray(attrs[2].Get())
        axes = attrs[3].Get()
        quat = np.r_[np.asarray(axes.GetImaginary()), axes.GetReal()]
        if (
            not math.isfinite(mass)
            or mass <= 0
            or not np.isfinite(com).all()
            or not np.isfinite(inertia).all()
            or (inertia <= 0).any()
            or not np.isfinite(quat).all()
            or not np.isclose(np.linalg.norm(quat), 1.0, atol=1e-6)
        ):
            raise ValueError("E8 authored mass properties must be positive and finite.")
    if not bodies:
        raise ValueError("E8 mass source contains no authored physical links.")


def _snapshot(path: Path) -> _Snapshot:
    from pxr import Sdf, Usd

    sidecar_path = path.with_suffix(".twist.json")
    try:
        if path.is_symlink() or sidecar_path.is_symlink():
            raise ValueError("E8 mass-source files cannot be symbolic links.")
        raw = path.read_bytes()
        sidecar = sidecar_path.read_bytes() if sidecar_path.exists() else None
        semantics = resolve_twist_semantics(path)
        layer = Sdf.Layer.OpenAsAnonymous(str(path))
        stage = Usd.Stage.Open(layer) if layer is not None else None
        if stage is None:
            raise ValueError("E8 mass source USD cannot be read.")
        _validate_stage(stage)
        result = _Snapshot(raw, sidecar, semantics, stage)
        _fresh(path, result)
    except OSError as error:
        raise ValueError("E8 mass source cannot be read.") from error
    if semantics.source_sha256 != _digest(raw) or semantics.sidecar_sha256 != (
        None if sidecar is None else _digest(sidecar)
    ):
        raise ValueError("E8 mass source differs from the qualified semantic snapshot.")
    return result


def _fresh(path: Path, snapshot: _Snapshot) -> None:
    sidecar = path.with_suffix(".twist.json")
    try:
        same = path.read_bytes() == snapshot.raw and (
            sidecar.read_bytes() == snapshot.sidecar
            if snapshot.sidecar is not None
            else not sidecar.exists()
        )
    except OSError as error:
        raise ValueError("E8 mass source changed during qualification.") from error
    if not same:
        raise ValueError("E8 mass source changed during qualification.")


def _scaled_layer(stage: Any, factor: float) -> Any:
    from pxr import Gf, Sdf, Usd, UsdPhysics

    layer = Sdf.Layer.CreateAnonymous("mass-source-copy.usda")
    if not layer.ImportFromString(stage.GetRootLayer().ExportToString()):
        raise ValueError("E8 source layer cannot be cloned.")
    clone = Usd.Stage.Open(layer)
    try:
        squared = factor**2
        if not math.isfinite(squared):
            raise ValueError("E8 mass scale produced non-finite inertia.")
        for prim in clone.Traverse():
            if not prim.HasAPI(UsdPhysics.MassAPI):
                continue
            api = UsdPhysics.MassAPI(prim)
            for attr, multiplier in (
                (api.GetCenterOfMassAttr(), factor),
                (api.GetDiagonalInertiaAttr(), squared),
            ):
                # Preserve the qualified probe's scalar multiplication order.
                attr.Set(Gf.Vec3f(*(float(value) * multiplier for value in attr.Get())))
    except (OverflowError, TypeError) as error:
        raise ValueError("E8 mass scale produced invalid mass properties.") from error
    _validate_stage(clone)
    return layer


def _key(
    source_sha256: str, semantics_sha256: str, sidecar_sha256: str | None, scale: float
) -> str:
    return _digest(
        _json_bytes(
            {
                "policy": _POLICY,
                "source_sha256": source_sha256,
                "source_semantics_sha256": semantics_sha256,
                "source_sidecar_sha256": sidecar_sha256,
                "scale": scale,
            }
        )
    )


def _roles(semantics: TwSemantics, source_sha256: str) -> dict[str, Any]:
    return {
        "schema": "gen_sim.twist-control/v1",
        "source_sha256": source_sha256,
        "joint": semantics.joint_path,
        "grip": semantics.grip_path,
        "pointer": semantics.pointer_path,
        "settings": {str(key): list(paths) for key, paths in semantics.settings},
    }


def qualify_mass_source_copy(
    runtime: Path, *, manifest_sha256: str | None = None
) -> MassSourceCopy:
    """Recheck byte identities, roles and every authored layer field.

    The expected layer is rebuilt from the pristine source with exactly the
    qualified two-attribute operation. An origin marker is not evidence.
    """
    runtime = Path(runtime).expanduser().absolute()
    manifest = adjacent_manifest_path(runtime)
    if runtime.is_symlink() or manifest.is_symlink():
        raise ValueError("E8 mass-source files cannot be symbolic links.")
    manifest_bytes, data = _manifest(manifest)
    digest = _digest(manifest_bytes)
    if manifest_sha256 is not None and manifest_sha256 != digest:
        raise ValueError("E8 mass manifest differs from its pinned SHA-256 identity.")
    if (
        runtime.suffix not in {".usd", ".usda", ".usdc"}
        or runtime.name != "r" + runtime.suffix
        or data["runtime_file"] != runtime.name
        or data["source_file"] != "s" + runtime.suffix
    ):
        raise ValueError(
            "E8 mass manifest requires exact adjacent source/runtime filenames."
        )
    source = runtime.with_name(data["source_file"])
    original, derived = _snapshot(source), _snapshot(runtime)
    scale = float(data["scale"])
    key = _key(
        original.semantics.source_sha256,
        original.semantics.semantics_sha256,
        original.semantics.sidecar_sha256,
        scale,
    )
    if data["key_sha256"] != key or runtime.parent.name != key[:16]:
        raise ValueError("E8 mass-source content-address identity is stale.")
    for prefix, snapshot in (("source", original), ("runtime", derived)):
        semantics = snapshot.semantics
        if (
            data[prefix + "_sha256"] != semantics.source_sha256
            or data[prefix + "_semantics_sha256"] != semantics.semantics_sha256
            or data[prefix + "_sidecar_sha256"] != semantics.sidecar_sha256
        ):
            raise ValueError(
                "E8 mass-source bytes or semantics differ from the manifest."
            )
    if _roles(original.semantics, "") != _roles(derived.semantics, ""):
        raise ValueError("E8 mass-source original and deployment roles differ.")
    expected = _scaled_layer(original.stage, scale).ExportToString()
    if expected != derived.stage.GetRootLayer().ExportToString():
        raise ValueError("E8 runtime USD differs beyond the allowed MassAPI operation.")
    _fresh(source, original)
    _fresh(runtime, derived)
    if manifest.read_bytes() != manifest_bytes:
        raise ValueError("E8 mass manifest changed during qualification.")
    return MassSourceCopy(
        source,
        runtime,
        manifest,
        original.semantics.source_sha256,
        derived.semantics.source_sha256,
        original.semantics.semantics_sha256,
        derived.semantics.semantics_sha256,
        digest,
        scale,
        original.semantics.sidecar_sha256,
        derived.semantics.sidecar_sha256,
    )


def create_mass_source_copy(
    source: Path,
    directory: Path,
    body_scale: tuple[float, ...],
    *,
    backend: str = "default",
) -> MassSourceCopy:
    """Create or fully qualify a content-addressed deployment mass copy.

    Only Default's existing raw-mass-property import is supported. Geometry
    continues to use the original body scale. Failed artifacts are retained.
    """
    if backend != "default":
        raise ValueError("E8 deployment mass compensation is Default-only.")
    scale = _uniform_scale(body_scale)
    source = Path(source).expanduser().absolute()
    if adjacent_manifest_path(source).exists():
        raise ValueError("E8 deployment mass source is already compensated.")
    original = _snapshot(source)
    semantics = original.semantics
    key = _key(
        semantics.source_sha256,
        semantics.semantics_sha256,
        semantics.sidecar_sha256,
        scale,
    )
    destination = Path(directory).expanduser().resolve() / key[:16]
    pristine = destination / ("s" + source.suffix)
    runtime = destination / ("r" + source.suffix)
    if destination.exists():
        manifest_bytes, data = _manifest(adjacent_manifest_path(runtime))
        # Cache reuse pins the canonical receipt originally written by this API.
        if manifest_bytes != _json_bytes(data):
            raise ValueError("E8 cached mass manifest bytes changed.")
        result = qualify_mass_source_copy(
            runtime, manifest_sha256=_digest(manifest_bytes)
        )
        if (
            result.source_sha256 != semantics.source_sha256
            or result.source_semantics_sha256 != semantics.semantics_sha256
            or result.source_sidecar_sha256 != semantics.sidecar_sha256
            or result.scale != scale
            or pristine.read_bytes() != original.raw
        ):
            raise ValueError("E8 mass-source content-address collision.")
        _fresh(source, original)
        return result
    layer = _scaled_layer(original.stage, scale)
    destination.mkdir(parents=True, exist_ok=False)
    with pristine.open("xb") as output:
        output.write(original.raw)
    if original.sidecar is not None:
        with pristine.with_suffix(".twist.json").open("xb") as output:
            output.write(original.sidecar)
    if not layer.Export(str(runtime)):
        raise ValueError("E8 deployment USD export failed.")
    with runtime.with_suffix(".twist.json").open("xb") as output:
        output.write(_json_bytes(_roles(semantics, _digest(runtime.read_bytes()))))
    runtime_semantics = resolve_twist_semantics(runtime)
    data = {
        "schema": _SCHEMA,
        "policy": _POLICY,
        "backend": "default",
        "key_sha256": key,
        "source_file": pristine.name,
        "runtime_file": runtime.name,
        "source_sha256": semantics.source_sha256,
        "runtime_sha256": runtime_semantics.source_sha256,
        "source_semantics_sha256": semantics.semantics_sha256,
        "runtime_semantics_sha256": runtime_semantics.semantics_sha256,
        "source_sidecar_sha256": semantics.sidecar_sha256,
        "runtime_sidecar_sha256": runtime_semantics.sidecar_sha256,
        "scale": scale,
    }
    manifest_bytes = _json_bytes(data)
    with adjacent_manifest_path(runtime).open("xb") as output:
        output.write(manifest_bytes)
    result = qualify_mass_source_copy(runtime, manifest_sha256=_digest(manifest_bytes))
    _fresh(source, original)
    return result
