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
from pathlib import Path
from typing import Any
import hashlib
import json
import math
import shutil

__all__ = [
    "BODY_SCALE_BAKE_POLICY_VERSION",
    "bake_body_scale_into_meshes",
]

BODY_SCALE_BAKE_POLICY_VERSION = "action_agent_body_scale_bake_v1"

_IDENTITY_SCALE = [1.0, 1.0, 1.0]
_FLOAT_ABS_TOL = 1e-12


def bake_body_scale_into_meshes(
    gym_config: MutableMapping[str, Any],
    *,
    output_dir: str | Path,
) -> list[dict[str, Any]]:
    """Bake mesh ``body_scale`` values into runtime OBJ files in-place.

    The action-agent runtime computes both environment CoACD and grasp collision
    data from mesh vertices. Baking the final configured scale into the mesh
    makes that scaled mesh the single source of truth and lets both users share
    the same convex decomposition cache.
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
        if source_path.suffix.lower() != ".obj":
            raise ValueError(
                "body_scale baking expects OBJ meshes after frame normalization: "
                f"{source_path.as_posix()}"
            )
        if not source_path.is_file():
            raise FileNotFoundError(
                f"Mesh path for body_scale baking not found: {source_path}"
            )

        source_sha256 = _file_sha256(source_path)
        scaled_path = _scaled_mesh_path(
            source_path,
            output_dir,
            source_sha256,
            body_scale,
        )
        status = "reused" if scaled_path.is_file() else "generated"
        if status == "generated":
            _write_scaled_obj(
                source_path,
                scaled_path,
                body_scale,
                source_sha256,
            )
        _copy_obj_material_assets(source_path, scaled_path.parent)

        shape["fpath"] = scaled_path.as_posix()
        obj_config["body_scale"] = list(_IDENTITY_SCALE)
        reports.append(
            {
                "uid": str(obj_config.get("uid", "")),
                "section": section,
                "source_path": source_path.as_posix(),
                "scaled_path": scaled_path.as_posix(),
                "source_sha256": source_sha256,
                "body_scale": body_scale,
                "status": status,
                "policy_version": BODY_SCALE_BAKE_POLICY_VERSION,
            }
        )
    return reports


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


def _scaled_mesh_path(
    source_path: Path,
    output_dir: Path,
    source_sha256: str,
    body_scale: list[float],
) -> Path:
    stem = source_path.stem[:32].strip("._") or "mesh"
    runtime_hash = hashlib.sha256(
        json.dumps(
            {
                "source_sha256": source_sha256,
                "body_scale": body_scale,
                "policy_version": BODY_SCALE_BAKE_POLICY_VERSION,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return output_dir / f"{stem}_{runtime_hash[:16]}.obj"


def _write_scaled_obj(
    source_path: Path,
    scaled_path: Path,
    body_scale: list[float],
    source_sha256: str,
) -> None:
    scaled_path.parent.mkdir(parents=True, exist_ok=True)
    source_text = source_path.read_text(encoding="utf-8")
    scaled_lines = []
    for line in source_text.splitlines():
        parts = line.split()
        if parts and parts[0] == "v":
            if len(parts) < 4:
                raise ValueError(f"Malformed OBJ vertex line in {source_path}: {line}")
            xyz = [
                _format_float(float(parts[index]) * body_scale[index - 1])
                for index in range(1, 4)
            ]
            parts = ["v", *xyz, *parts[4:]]
            line = " ".join(parts)
        scaled_lines.append(line)

    header = "\n".join(
        [
            "# EmbodiChain action-agent body-scale baked mesh",
            f"# policy_version: {BODY_SCALE_BAKE_POLICY_VERSION}",
            f"# source_path: {source_path.as_posix()}",
            f"# source_sha256: {source_sha256}",
            f"# body_scale: {json.dumps(body_scale, separators=(',', ':'))}",
            "",
        ]
    )
    scaled_path.write_text(header + "\n".join(scaled_lines) + "\n", encoding="utf-8")


def _copy_obj_material_assets(source_path: Path, output_dir: Path) -> None:
    for material_path in _obj_material_paths(source_path):
        if not material_path.is_file():
            continue
        relative_material_path = material_path.relative_to(source_path.parent)
        target_material_path = output_dir / relative_material_path
        target_material_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(material_path, target_material_path)

    texture_dir = source_path.parent / "textures"
    if texture_dir.is_dir():
        shutil.copytree(texture_dir, output_dir / "textures", dirs_exist_ok=True)


def _obj_material_paths(source_path: Path) -> list[Path]:
    material_paths = []
    for line in source_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts and parts[0] == "mtllib" and len(parts) >= 2:
            material_paths.append((source_path.parent / parts[1]).resolve())
    return material_paths


def _vector3(value: Any) -> list[float]:
    if isinstance(value, (int, float)):
        values = [float(value), float(value), float(value)]
    else:
        values = [float(item) for item in value]
    if len(values) != 3 or not all(math.isfinite(item) for item in values):
        raise ValueError(f"Expected finite xyz body_scale, got: {value!r}")
    return values


def _is_identity_scale(body_scale: list[float]) -> bool:
    return all(
        math.isclose(value, 1.0, rel_tol=0.0, abs_tol=_FLOAT_ABS_TOL)
        for value in body_scale
    )


def _format_float(value: float) -> str:
    formatted = f"{value:.12g}"
    return "0" if formatted == "-0" else formatted


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
