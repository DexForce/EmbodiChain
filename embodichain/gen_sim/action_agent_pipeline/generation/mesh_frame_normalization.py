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

from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any
import hashlib
import json
import math
import re

__all__ = [
    "GLB_LOCAL_X_CORRECTION_DEGREES",
    "MESH_FRAME_NORMALIZATION_POLICY_VERSION",
    "MeshFrameNormalizer",
    "NormalizedMeshResult",
]


MESH_FRAME_NORMALIZATION_POLICY_VERSION = "action_agent_glb_rx_minus_90_obj_v1"
GLB_LOCAL_X_CORRECTION_DEGREES = -90.0

_SAFE_STEM_RE = re.compile(r"[^0-9a-zA-Z_.-]+")


@dataclass(frozen=True)
class NormalizedMeshResult:
    """A normalized mesh path and metadata for generation summaries."""

    source_path: Path
    normalized_path: Path
    source_sha256: str
    status: str
    transform: list[list[float]]
    dexsim_engine_version: str

    def to_summary(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path.as_posix(),
            "normalized_path": self.normalized_path.as_posix(),
            "source_sha256": self.source_sha256,
            "status": self.status,
            "policy_version": MESH_FRAME_NORMALIZATION_POLICY_VERSION,
            "dexsim_engine_version": self.dexsim_engine_version,
            "transform": self.transform,
        }


class MeshFrameNormalizer:
    """Normalize GLB meshes to OBJ so visual and collision share one frame."""

    def __init__(
        self,
        *,
        output_dir: str | Path,
        local_x_correction_degrees: float = GLB_LOCAL_X_CORRECTION_DEGREES,
    ) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.local_x_correction_degrees = float(local_x_correction_degrees)
        self.transform = _rotation_x_matrix4(self.local_x_correction_degrees)
        self.dexsim_engine_version = _dexsim_engine_version()
        self._results_by_source: dict[Path, NormalizedMeshResult] = {}
        self._reports: list[dict[str, Any]] = []

    @property
    def reports(self) -> list[dict[str, Any]]:
        return list(self._reports)

    def normalize_path(self, mesh_path: str | Path) -> Path:
        """Return a runtime mesh path, normalizing GLB/GLTF inputs to OBJ."""

        path = Path(mesh_path).expanduser().resolve()
        if path.suffix.lower() not in {".glb", ".gltf"}:
            return path

        cached = self._results_by_source.get(path)
        if cached is not None:
            return cached.normalized_path

        source_sha256 = _file_sha256(path)
        normalized_path = self._normalized_path_for(path, source_sha256)
        status = "reused" if normalized_path.is_file() else "generated"
        if status == "generated":
            self._write_normalized_obj(path, normalized_path, source_sha256)

        result = NormalizedMeshResult(
            source_path=path,
            normalized_path=normalized_path,
            source_sha256=source_sha256,
            status=status,
            transform=self.transform,
            dexsim_engine_version=self.dexsim_engine_version,
        )
        self._results_by_source[path] = result
        self._reports.append(result.to_summary())
        return normalized_path

    def _normalized_path_for(self, mesh_path: Path, source_sha256: str) -> Path:
        stem = _SAFE_STEM_RE.sub("_", mesh_path.stem).strip("._") or "mesh"
        filename = (
            f"{stem}_{source_sha256[:12]}_"
            f"{MESH_FRAME_NORMALIZATION_POLICY_VERSION}.obj"
        )
        return self.output_dir / filename

    def _write_normalized_obj(
        self,
        source_path: Path,
        normalized_path: Path,
        source_sha256: str,
    ) -> None:
        trimesh = _require_trimesh()
        scene = trimesh.load(str(source_path), force="scene")
        mesh = _scene_to_world_mesh(scene)
        mesh.apply_transform(self.transform)

        normalized_path.parent.mkdir(parents=True, exist_ok=True)
        obj_payload = mesh.export(file_type="obj")
        if isinstance(obj_payload, bytes):
            obj_text = obj_payload.decode("utf-8")
        else:
            obj_text = str(obj_payload)

        header = "\n".join(
            [
                "# EmbodiChain action-agent normalized mesh",
                f"# policy_version: {MESH_FRAME_NORMALIZATION_POLICY_VERSION}",
                f"# dexsim_engine_version: {self.dexsim_engine_version}",
                f"# source_path: {source_path.as_posix()}",
                f"# source_sha256: {source_sha256}",
                f"# transform: {json.dumps(self.transform, separators=(',', ':'))}",
                "",
            ]
        )
        normalized_path.write_text(header + obj_text, encoding="utf-8")


def _scene_to_world_mesh(scene: Any) -> Any:
    try:
        mesh = scene.dump(concatenate=True)
    except AttributeError:
        mesh = scene
    if not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
        raise ValueError("Mesh contains no vertices.")
    return mesh


def _rotation_x_matrix4(degrees: float) -> list[list[float]]:
    radians = math.radians(degrees)
    cos_value = math.cos(radians)
    sin_value = math.sin(radians)
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, cos_value, -sin_value, 0.0],
        [0.0, sin_value, cos_value, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dexsim_engine_version() -> str:
    for package_name in ("dexsim-engine", "dexsim_engine"):
        try:
            return metadata.version(package_name)
        except metadata.PackageNotFoundError:
            continue
    return "unknown"


def _require_trimesh() -> Any:
    try:
        import trimesh
    except ImportError as exc:
        raise ImportError("trimesh is required to normalize GLB meshes.") from exc
    return trimesh
