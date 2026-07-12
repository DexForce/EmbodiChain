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

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import hashlib
import json
import math

__all__ = [
    "GLB_GEOMETRY_BAKE_POLICY_VERSION",
    "GLB_GEOMETRY_NORMALIZATION_POLICY_VERSION",
    "GlbGeometryNormalizer",
    "bake_body_scale_into_glbs",
    "bake_glb_geometry",
]


GLB_GEOMETRY_BAKE_POLICY_VERSION = "action_agent_glb_geometry_bake_v1"
GLB_GEOMETRY_NORMALIZATION_POLICY_VERSION = "action_agent_glb_geometry_normalize_v1"

_IDENTITY_SCALE = [1.0, 1.0, 1.0]
_IDENTITY_TRANSFORM = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]
_FLOAT_ABS_TOL = 1e-12


@dataclass(frozen=True)
class NormalizedGlbResult:
    """A normalized GLB asset and its provenance metadata."""

    source_path: Path
    normalized_path: Path
    source_sha256: str
    status: str
    transform: list[list[float]]

    def to_summary(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path.as_posix(),
            "normalized_path": self.normalized_path.as_posix(),
            "source_sha256": self.source_sha256,
            "status": self.status,
            "policy_version": GLB_GEOMETRY_NORMALIZATION_POLICY_VERSION,
            "transform": self.transform,
        }


class GlbGeometryNormalizer:
    """Flatten source scene transforms into deterministic GLB runtime assets."""

    def __init__(
        self,
        *,
        output_dir: str | Path,
        transform: list[list[float]] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.transform = _matrix4(transform or _IDENTITY_TRANSFORM).tolist()
        self._results_by_source: dict[Path, NormalizedGlbResult] = {}
        self._reports: list[dict[str, Any]] = []

    @property
    def reports(self) -> list[dict[str, Any]]:
        """Return summaries for GLB assets normalized by this instance."""
        return list(self._reports)

    def normalize_path(self, mesh_path: str | Path) -> Path:
        """Return a flattened GLB path for a GLB or GLTF source mesh."""
        source_path = Path(mesh_path).expanduser().resolve()
        if source_path.suffix.lower() not in {".glb", ".gltf"}:
            raise ValueError(
                "GLB-only runtime assets require a GLB or GLTF source mesh: "
                f"{source_path.as_posix()}"
            )

        cached = self._results_by_source.get(source_path)
        if cached is not None:
            return cached.normalized_path

        source_sha256 = _file_sha256(source_path)
        normalized_path = _normalized_glb_path(
            source_path,
            self.output_dir,
            source_sha256,
            self.transform,
        )
        status = "reused" if normalized_path.is_file() else "generated"
        if status == "generated":
            bake_glb_geometry(
                source_path,
                normalized_path,
                transform=self.transform,
            )

        result = NormalizedGlbResult(
            source_path=source_path,
            normalized_path=normalized_path,
            source_sha256=source_sha256,
            status=status,
            transform=self.transform,
        )
        self._results_by_source[source_path] = result
        self._reports.append(result.to_summary())
        return normalized_path


def bake_body_scale_into_glbs(
    gym_config: MutableMapping[str, Any],
    *,
    output_dir: str | Path,
) -> list[dict[str, Any]]:
    """Bake mesh ``body_scale`` values into runtime GLB files in-place.

    The generated runtime config is normalized to identity ``body_scale`` so
    rendering, physics, grasping, and geometry-derived pose logic consume the
    same GLB vertices.
    """

    output_dir = Path(output_dir).expanduser().resolve()
    reports: list[dict[str, Any]] = []
    for section, obj_config in _iter_mesh_object_configs(gym_config):
        body_scale = _vector3(obj_config.get("body_scale", _IDENTITY_SCALE))
        if _is_identity_scale(body_scale):
            obj_config["body_scale"] = list(_IDENTITY_SCALE)
            continue

        shape = obj_config["shape"]
        source_path = Path(str(shape["fpath"])).expanduser().resolve()
        if source_path.suffix.lower() != ".glb":
            raise ValueError(
                "GLB geometry baking requires GLB mesh assets: "
                f"{source_path.as_posix()}"
            )
        if not source_path.is_file():
            raise FileNotFoundError(
                f"Mesh path for GLB geometry baking not found: {source_path}"
            )

        source_sha256 = _file_sha256(source_path)
        baked_path = _baked_glb_path(
            source_path,
            output_dir,
            source_sha256,
            body_scale,
            _IDENTITY_TRANSFORM,
        )
        status = "reused" if baked_path.is_file() else "generated"
        if status == "generated":
            bake_glb_geometry(
                source_path,
                baked_path,
                body_scale=body_scale,
            )

        shape["fpath"] = baked_path.as_posix()
        obj_config["body_scale"] = list(_IDENTITY_SCALE)
        reports.append(
            {
                "uid": str(obj_config.get("uid", "")),
                "section": section,
                "source_path": source_path.as_posix(),
                "baked_path": baked_path.as_posix(),
                "source_sha256": source_sha256,
                "body_scale": body_scale,
                "status": status,
                "policy_version": GLB_GEOMETRY_BAKE_POLICY_VERSION,
            }
        )
    return reports


def bake_glb_geometry(
    source_path: str | Path,
    output_path: str | Path,
    *,
    body_scale: list[float] | tuple[float, float, float] = _IDENTITY_SCALE,
    transform: list[list[float]] | None = None,
) -> Path:
    """Write a GLB with scene-node transforms and ``body_scale`` baked into vertices.

    Args:
        source_path: Source GLB asset.
        output_path: Destination GLB asset.
        body_scale: Object-local xyz scale to bake.
        transform: Optional 4x4 transform applied after each source node's
            world transform. It is intended for explicit frame conversion.

    Returns:
        The resolved destination path.
    """

    source_path = Path(source_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if source_path.suffix.lower() not in {".glb", ".gltf"}:
        raise ValueError(f"Expected a GLB or GLTF source mesh, got: {source_path}")

    scale = _vector3(body_scale)
    bake_transform = _matrix4(transform or _IDENTITY_TRANSFORM)
    bake_transform = bake_transform @ _scale_matrix(scale)

    trimesh = _require_trimesh()
    source_scene = trimesh.load(source_path.as_posix(), force="scene")
    baked_scene = trimesh.Scene()
    for node_name in source_scene.graph.nodes_geometry:
        node_transform, geometry_name = source_scene.graph.get(node_name)
        source_mesh = source_scene.geometry[geometry_name]
        baked_mesh = source_mesh.copy()
        baked_mesh.apply_transform(bake_transform @ node_transform)
        baked_scene.add_geometry(
            baked_mesh,
            node_name=str(node_name),
            geom_name=f"geometry_{len(baked_scene.geometry)}",
        )

    if not baked_scene.geometry:
        raise ValueError(f"GLB contains no mesh geometry: {source_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    baked_scene.export(output_path.as_posix(), file_type="glb")
    return output_path


def _iter_mesh_object_configs(
    gym_config: Mapping[str, Any],
) -> list[tuple[str, MutableMapping[str, Any]]]:
    objects: list[tuple[str, MutableMapping[str, Any]]] = []
    for section in ("background", "rigid_object"):
        value = gym_config.get(section, [])
        if isinstance(value, Mapping):
            value = [value]
        if not isinstance(value, list):
            continue
        for obj_config in value:
            if not isinstance(obj_config, MutableMapping):
                continue
            shape = obj_config.get("shape", {})
            if (
                isinstance(shape, MutableMapping)
                and shape.get("shape_type") == "Mesh"
                and shape.get("fpath")
            ):
                objects.append((section, obj_config))
    return objects


def _baked_glb_path(
    source_path: Path,
    output_dir: Path,
    source_sha256: str,
    body_scale: list[float],
    transform: list[list[float]],
) -> Path:
    stem = source_path.stem[:32].strip("._") or "mesh"
    runtime_hash = hashlib.sha256(
        json.dumps(
            {
                "source_sha256": source_sha256,
                "body_scale": body_scale,
                "transform": transform,
                "policy_version": GLB_GEOMETRY_BAKE_POLICY_VERSION,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return output_dir / f"{stem}_{runtime_hash[:16]}.glb"


def _normalized_glb_path(
    source_path: Path,
    output_dir: Path,
    source_sha256: str,
    transform: list[list[float]],
) -> Path:
    stem = source_path.stem[:32].strip("._") or "mesh"
    runtime_hash = hashlib.sha256(
        json.dumps(
            {
                "source_sha256": source_sha256,
                "transform": transform,
                "policy_version": GLB_GEOMETRY_NORMALIZATION_POLICY_VERSION,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return output_dir / f"{stem}_{runtime_hash[:16]}.glb"


def _vector3(value: Any) -> list[float]:
    if isinstance(value, (int, float)):
        values = [float(value), float(value), float(value)]
    else:
        values = [float(item) for item in value]
    if len(values) != 3 or not all(math.isfinite(item) for item in values):
        raise ValueError(f"Expected finite xyz body_scale, got: {value!r}")
    return values


def _matrix4(value: list[list[float]]) -> Any:
    numpy = _require_numpy()
    matrix = numpy.asarray(value, dtype=numpy.float64)
    if matrix.shape != (4, 4) or not numpy.all(numpy.isfinite(matrix)):
        raise ValueError("Expected a finite 4x4 bake transform.")
    return matrix


def _scale_matrix(scale: list[float]) -> Any:
    numpy = _require_numpy()
    matrix = numpy.eye(4, dtype=numpy.float64)
    matrix[0, 0], matrix[1, 1], matrix[2, 2] = scale
    return matrix


def _is_identity_scale(body_scale: list[float]) -> bool:
    return all(
        math.isclose(value, 1.0, rel_tol=0.0, abs_tol=_FLOAT_ABS_TOL)
        for value in body_scale
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_numpy() -> Any:
    try:
        import numpy
    except ImportError as exc:
        raise ImportError("GLB geometry baking requires numpy.") from exc
    return numpy


def _require_trimesh() -> Any:
    try:
        import trimesh
    except ImportError as exc:
        raise ImportError("GLB geometry baking requires trimesh.") from exc
    return trimesh
