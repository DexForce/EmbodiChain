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

from embodichain.gen_sim.action_agent_pipeline.generation.glb_io import read_glb

__all__ = [
    "GLB_TO_OBJ_BAKED_X_ROTATION_DEGREES",
    "GLB_TO_OBJ_BAKED_Z_ROTATION_DEGREES",
    "GLB_LOCAL_X_CORRECTION_DEGREES",
    "GLB_LOCAL_Z_CORRECTION_DEGREES",
    "MESH_FRAME_NORMALIZATION_POLICY_VERSION",
    "MeshFrameNormalizer",
    "NormalizedMeshResult",
]


MESH_FRAME_NORMALIZATION_POLICY_VERSION = "action_agent_glb_scene_texture_obj_v9"
# Match the legacy action-agent GLB->OBJ path: do not bake extra frame rotations.
GLB_TO_OBJ_BAKED_X_ROTATION_DEGREES = 0.0
GLB_TO_OBJ_BAKED_Z_ROTATION_DEGREES = 0.0
GLB_LOCAL_X_CORRECTION_DEGREES = GLB_TO_OBJ_BAKED_X_ROTATION_DEGREES
GLB_LOCAL_Z_CORRECTION_DEGREES = GLB_TO_OBJ_BAKED_Z_ROTATION_DEGREES

_SAFE_STEM_RE = re.compile(r"[^0-9a-zA-Z_.-]+")
_TEXTURE_EXTENSION_BY_MIME_TYPE = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}


@dataclass(frozen=True)
class _MaterialSpec:
    name: str
    texture_path: str | None = None


@dataclass(frozen=True)
class _TextureAsset:
    data: bytes
    extension: str


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
        local_x_correction_degrees: float = GLB_TO_OBJ_BAKED_X_ROTATION_DEGREES,
        local_z_correction_degrees: float = GLB_TO_OBJ_BAKED_Z_ROTATION_DEGREES,
    ) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.local_x_correction_degrees = float(local_x_correction_degrees)
        self.local_z_correction_degrees = float(local_z_correction_degrees)
        self.transform = _matrix_multiply(
            _rotation_z_matrix4(self.local_z_correction_degrees),
            _rotation_x_matrix4(self.local_x_correction_degrees),
        )
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
            if cached.normalized_path.is_file():
                material_spec = self._material_spec_for(
                    path,
                    cached.normalized_path,
                    cached.source_sha256,
                )
                _repair_obj_material_reference(
                    cached.normalized_path,
                    material_spec.name,
                )
                self._ensure_material_library({material_spec.name: material_spec})
            return cached.normalized_path

        source_sha256 = _file_sha256(path)
        normalized_path = self._normalized_path_for(path, source_sha256)
        material_spec = self._material_spec_for(path, normalized_path, source_sha256)
        status = "reused" if normalized_path.is_file() else "generated"
        if status == "generated":
            self._write_normalized_obj(
                path,
                normalized_path,
                source_sha256,
                material_spec,
            )
        else:
            _repair_obj_material_reference(normalized_path, material_spec.name)
            self._ensure_material_library({material_spec.name: material_spec})

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
        stem = stem[:32].strip("._") or "mesh"
        runtime_hash = hashlib.sha256(
            json.dumps(
                {
                    "source_sha256": source_sha256,
                    "policy_version": MESH_FRAME_NORMALIZATION_POLICY_VERSION,
                    "dexsim_engine_version": self.dexsim_engine_version,
                    "transform": self.transform,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return self.output_dir / f"{stem}_{runtime_hash[:16]}.obj"

    def _material_path(self) -> Path:
        return self.output_dir / "material.mtl"

    def _texture_dir(self) -> Path:
        return self.output_dir / "textures"

    def _material_spec_for(
        self,
        source_path: Path,
        normalized_path: Path,
        source_sha256: str,
    ) -> _MaterialSpec:
        material_hash = _material_hash_for(normalized_path)
        material_name = f"material_{material_hash}"
        texture_path = self._write_base_color_texture(
            source_path,
            material_hash,
            source_sha256,
        )
        return _MaterialSpec(name=material_name, texture_path=texture_path)

    def _write_base_color_texture(
        self,
        source_path: Path,
        material_hash: str,
        source_sha256: str,
    ) -> str | None:
        try:
            texture = _extract_glb_base_color_texture(source_path)
        except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None
        if texture is None:
            return None

        texture_dir = self._texture_dir()
        texture_dir.mkdir(parents=True, exist_ok=True)
        texture_name = (
            f"{material_hash}_{source_sha256[:12]}_basecolor{texture.extension}"
        )
        texture_path = texture_dir / texture_name
        texture_path.write_bytes(texture.data)
        return f"textures/{texture_name}"

    def _ensure_material_library(
        self, material_specs: dict[str, _MaterialSpec]
    ) -> None:
        if not material_specs:
            return

        material_path = self._material_path()
        all_specs = {
            **_read_material_specs(material_path),
            **material_specs,
        }
        material_path.write_text(
            "\n".join(
                [
                    "# EmbodiChain action-agent normalized mesh materials",
                    *[
                        _format_material_spec(spec)
                        for spec in sorted(
                            all_specs.values(), key=lambda item: item.name
                        )
                    ],
                    "",
                ]
            ),
            encoding="utf-8",
        )

    def _write_normalized_obj(
        self,
        source_path: Path,
        normalized_path: Path,
        source_sha256: str,
        material_spec: _MaterialSpec,
    ) -> None:
        trimesh = _require_trimesh()
        scene = trimesh.load(str(source_path), force="scene")
        mesh = _scene_to_world_mesh(scene)
        if self.local_x_correction_degrees or self.local_z_correction_degrees:
            mesh.apply_transform(self.transform)

        normalized_path.parent.mkdir(parents=True, exist_ok=True)
        obj_payload = mesh.export(file_type="obj")
        if isinstance(obj_payload, bytes):
            obj_text = obj_payload.decode("utf-8")
        else:
            obj_text = str(obj_payload)
        obj_text = _ensure_obj_material_reference(obj_text, material_spec.name)

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
        self._ensure_material_library({material_spec.name: material_spec})


def _scene_to_world_mesh(scene: Any) -> Any:
    if hasattr(scene, "to_geometry"):
        mesh = scene.to_geometry()
    elif hasattr(scene, "dump"):
        mesh = scene.dump(concatenate=True)
    else:
        mesh = scene
    if not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
        raise ValueError("Mesh contains no vertices.")
    return mesh


def _material_hash_for(normalized_path: Path) -> str:
    hash_part = normalized_path.stem.rsplit("_", maxsplit=1)[-1]
    if re.fullmatch(r"[0-9a-fA-F]{8,}", hash_part):
        return hash_part.lower()
    return hashlib.sha256(normalized_path.stem.encode("utf-8")).hexdigest()[:16]


def _repair_obj_material_reference(obj_path: Path, material_name: str) -> str:
    obj_text = obj_path.read_text(encoding="utf-8")
    repaired = _ensure_obj_material_reference(obj_text, material_name)
    if repaired != obj_text:
        obj_path.write_text(repaired, encoding="utf-8")
    return repaired


def _ensure_obj_material_reference(obj_text: str, material_name: str) -> str:
    lines = obj_text.splitlines()
    header_lines: list[str] = []
    body_start = 0
    for line in lines:
        if not line.startswith("#"):
            break
        header_lines.append(line)
        body_start += 1

    body_lines: list[str] = []
    has_usemtl = False
    for line in lines[body_start:]:
        if line.startswith("mtllib "):
            continue
        if line.startswith("usemtl "):
            body_lines.append(f"usemtl {material_name}")
            has_usemtl = True
            continue
        body_lines.append(line)

    prefix = ["mtllib material.mtl"]
    if not has_usemtl:
        prefix.append(f"usemtl {material_name}")
    return "\n".join(header_lines + prefix + body_lines) + "\n"


def _read_material_specs(material_path: Path) -> dict[str, _MaterialSpec]:
    if not material_path.is_file():
        return {}

    specs: dict[str, _MaterialSpec] = {}
    current_name: str | None = None
    current_texture_path: str | None = None
    for line in material_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("newmtl "):
            if current_name is not None:
                specs[current_name] = _MaterialSpec(
                    name=current_name,
                    texture_path=current_texture_path,
                )
            current_name = line.split(maxsplit=1)[1].strip()
            current_texture_path = None
            continue
        if current_name is not None and line.startswith("map_Kd "):
            current_texture_path = line.split(maxsplit=1)[1].strip()
    if current_name is not None:
        specs[current_name] = _MaterialSpec(
            name=current_name,
            texture_path=current_texture_path,
        )
    return specs


def _format_material_spec(spec: _MaterialSpec) -> str:
    ambient = "1.0 1.0 1.0" if spec.texture_path else "0.8 0.8 0.8"
    diffuse = "1.0 1.0 1.0" if spec.texture_path else "0.8 0.8 0.8"
    lines = [
        f"newmtl {spec.name}",
        f"Ka {ambient}",
        f"Kd {diffuse}",
        "Ks 0.0 0.0 0.0",
        "Ns 1.0",
        "d 1.0",
        "illum 2",
    ]
    if spec.texture_path:
        lines.append(f"map_Kd {spec.texture_path}")
    return "\n".join(lines)


def _extract_glb_base_color_texture(source_path: Path) -> _TextureAsset | None:
    if source_path.suffix.lower() != ".glb":
        return None

    doc, binary_chunk = read_glb(source_path)
    material = _first_textured_material(doc)
    if material is None:
        return None

    texture_index = int(material["pbrMetallicRoughness"]["baseColorTexture"]["index"])
    textures = doc.get("textures", [])
    if not isinstance(textures, list) or texture_index >= len(textures):
        return None

    texture = textures[texture_index]
    if not isinstance(texture, dict):
        return None
    image_index = texture.get("source")
    if image_index is None:
        return None

    images = doc.get("images", [])
    if not isinstance(images, list) or int(image_index) >= len(images):
        return None

    image = images[int(image_index)]
    if not isinstance(image, dict):
        return None

    mime_type = str(image.get("mimeType", ""))
    extension = _TEXTURE_EXTENSION_BY_MIME_TYPE.get(mime_type)
    if extension is None:
        return None

    buffer_view_index = image.get("bufferView")
    if buffer_view_index is None:
        return None

    image_data = _buffer_view_bytes(doc, binary_chunk, int(buffer_view_index))
    if not image_data:
        return None
    return _TextureAsset(data=image_data, extension=extension)


def _first_textured_material(doc: dict[str, Any]) -> dict[str, Any] | None:
    materials = doc.get("materials", [])
    if not isinstance(materials, list):
        return None
    for material in materials:
        if not isinstance(material, dict):
            continue
        pbr = material.get("pbrMetallicRoughness", {})
        if not isinstance(pbr, dict):
            continue
        base_color_texture = pbr.get("baseColorTexture", {})
        if not isinstance(base_color_texture, dict):
            continue
        if "index" in base_color_texture:
            return material
    return None


def _buffer_view_bytes(
    doc: dict[str, Any],
    binary_chunk: bytes,
    buffer_view_index: int,
) -> bytes:
    buffer_views = doc.get("bufferViews", [])
    if not isinstance(buffer_views, list) or buffer_view_index >= len(buffer_views):
        return b""
    buffer_view = buffer_views[buffer_view_index]
    if not isinstance(buffer_view, dict):
        return b""
    if int(buffer_view.get("buffer", 0)) != 0:
        return b""
    byte_offset = int(buffer_view.get("byteOffset", 0))
    byte_length = int(buffer_view.get("byteLength", 0))
    if byte_length <= 0:
        return b""
    return binary_chunk[byte_offset : byte_offset + byte_length]


def _rotation_x_matrix4(degrees: float) -> list[list[float]]:
    if degrees == 0.0:
        return _identity_matrix4()
    radians = math.radians(degrees)
    cos_value = math.cos(radians)
    sin_value = math.sin(radians)
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, cos_value, -sin_value, 0.0],
        [0.0, sin_value, cos_value, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _rotation_z_matrix4(degrees: float) -> list[list[float]]:
    if degrees == 0.0:
        return _identity_matrix4()
    radians = math.radians(degrees)
    cos_value = math.cos(radians)
    sin_value = math.sin(radians)
    return [
        [cos_value, -sin_value, 0.0, 0.0],
        [sin_value, cos_value, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _identity_matrix4() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _matrix_multiply(
    left: list[list[float]],
    right: list[list[float]],
) -> list[list[float]]:
    return [
        [
            sum(left[row][index] * right[index][column] for index in range(4))
            for column in range(4)
        ]
        for row in range(4)
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
