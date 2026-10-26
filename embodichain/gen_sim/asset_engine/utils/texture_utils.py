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

from typing import Any, Dict, List, Optional, Tuple
import trimesh

PBR_TEXTURE_FIELDS = (
    "baseColorTexture",
    "metallicRoughnessTexture",
    "normalTexture",
    "occlusionTexture",
    "emissiveTexture",
)

PBR_SCALAR_FIELDS = (
    "baseColorFactor",
    "metallicFactor",
    "roughnessFactor",
    "emissiveFactor",
    "alphaMode",
    "alphaCutoff",
    "doubleSided",
)

SIMPLE_SCALAR_FIELDS = (
    "diffuse",
    "ambient",
    "specular",
    "glossiness",
)


def _shape(x: Any) -> Optional[Tuple[int, ...]]:
    try:
        return tuple(x.shape)  # numpy / array-like
    except Exception:
        return None


def _to_jsonable(x: Any) -> Any:

    if x is None:
        return None

    if hasattr(x, "tolist"):
        try:
            return x.tolist()
        except Exception:
            pass

    if hasattr(x, "size") and hasattr(x, "mode"):
        try:
            return {
                "type": type(x).__name__,
                "size": list(x.size),
                "mode": x.mode,
            }
        except Exception:
            return {"type": type(x).__name__}

    if isinstance(x, (str, int, float, bool)):
        return x

    return str(x)


def _describe_texture_value(value: Any) -> Dict[str, Any]:

    info: Dict[str, Any] = {
        "present": value is not None,
        "type": None,
        "meta": None,
    }

    if value is None:
        return info

    info["type"] = type(value).__name__
    info["meta"] = _to_jsonable(value)
    return info


def _inspect_material(material: Any) -> Dict[str, Any]:
    """
    Recursively inspect trimesh materials.
    """
    out: Dict[str, Any] = {
        "material_class": type(material).__name__ if material is not None else None,
        "material_kind": None,
        "name": getattr(material, "name", None) if material is not None else None,
        "main_color": None,
        "texture_count": 0,
        "textures": {},
        "scalars": {},
        "children": None,
    }

    if material is None:
        return out

    out["main_color"] = _to_jsonable(getattr(material, "main_color", None))

    # MultiMaterial: wrapper around a list of Materials
    if isinstance(material, trimesh.visual.material.MultiMaterial):
        out["material_kind"] = "multi"
        children: List[Dict[str, Any]] = []
        total = 0

        mats = getattr(material, "materials", None) or []
        for idx, child in enumerate(mats):
            child_info = _inspect_material(child)
            child_info["index"] = idx
            children.append(child_info)
            total += int(child_info.get("texture_count", 0))

        out["children"] = children
        out["texture_count"] = total
        return out

    # PBRMaterial
    if isinstance(material, trimesh.visual.material.PBRMaterial):
        out["material_kind"] = "pbr"
        for field in PBR_SCALAR_FIELDS:
            out["scalars"][field] = _to_jsonable(getattr(material, field, None))

        texture_count = 0
        for field in PBR_TEXTURE_FIELDS:
            tex_value = getattr(material, field, None)
            out["textures"][field] = _describe_texture_value(tex_value)
            if tex_value is not None:
                texture_count += 1

        out["texture_count"] = texture_count
        return out

    # SimpleMaterial
    if isinstance(material, trimesh.visual.material.SimpleMaterial):
        out["material_kind"] = "simple"
        for field in SIMPLE_SCALAR_FIELDS:
            out["scalars"][field] = _to_jsonable(getattr(material, field, None))

        image = getattr(material, "image", None)
        out["textures"]["image"] = _describe_texture_value(image)
        out["texture_count"] = 1 if image is not None else 0
        return out

    # Generic Material or unknown subclass
    out["material_kind"] = "generic_or_unknown"
    # Collect anything that looks texture-like or important
    for key, value in getattr(material, "__dict__", {}).items():
        if "texture" in key.lower() or key.lower() in {"image", "name"}:
            out["textures"][key] = _describe_texture_value(value)

    return out


def classify_visual(mesh: trimesh.Trimesh) -> Dict[str, Any]:
    """
    Returns a nested dict with:
      - top-level visual category
      - color mode / texture mode
      - uv presence
      - material type
      - material texture slots
      - total texture count
      - completeness flags
    """
    vis = getattr(mesh, "visual", None)

    result: Dict[str, Any] = {
        "visual_class": type(vis).__name__ if vis is not None else None,
        "visual_category": "none",
        "visual_kind": None,
        "visual_defined": False,
        "is_color_visual": False,
        "is_texture_visual": False,
        "uv_present": False,
        "uv_shape": None,
        "material": None,
        "material_type": None,
        "material_kind": None,
        "texture_count_total": 0,
        "texture_state": "none",
        "face_materials_present": False,
        "face_materials_shape_or_len": None,
        "color_mode": None,
        "face_colors_shape": None,
        "vertex_colors_shape": None,
        "has_transparency": None,
        "main_color": None,
        "notes": [],
    }

    if vis is None:
        result["notes"].append("mesh.visual is None")
        return result

    result["visual_kind"] = getattr(vis, "kind", None)
    result["visual_defined"] = bool(getattr(vis, "defined", False))

    # -------- TextureVisuals --------
    if isinstance(vis, trimesh.visual.texture.TextureVisuals):
        result["visual_category"] = "texture"
        result["is_texture_visual"] = True

        uv = getattr(vis, "uv", None)
        result["uv_present"] = uv is not None
        result["uv_shape"] = _shape(uv)

        # face_materials is an optional constructor arg; inspect defensively
        face_materials = getattr(vis, "face_materials", None)
        result["face_materials_present"] = face_materials is not None
        if face_materials is not None:
            try:
                result["face_materials_shape_or_len"] = len(face_materials)
            except Exception:
                result["face_materials_shape_or_len"] = _shape(face_materials)

        material = getattr(vis, "material", None)
        result["material"] = (
            _inspect_material(material) if material is not None else None
        )
        if material is not None:
            result["material_type"] = type(material).__name__
            result["material_kind"] = result["material"]["material_kind"]
            result["main_color"] = result["material"]["main_color"]
            result["texture_count_total"] = int(result["material"]["texture_count"])

        # TextureVisuals is only really usable when UV exists.
        if not result["uv_present"]:
            result["texture_state"] = "texture_visual_missing_uv"
            result["notes"].append("TextureVisuals exists, but uv is missing.")
        elif material is None:
            result["texture_state"] = "texture_visual_missing_material"
            result["notes"].append("TextureVisuals has uv, but material is missing.")
        elif result["texture_count_total"] == 0:
            result["texture_state"] = "texture_visual_material_no_textures"
            result["notes"].append(
                "TextureVisuals has uv and material, but material contains no texture slots/images."
            )
        else:
            result["texture_state"] = "texture_visual_complete_or_partially_complete"

        # If the visual has alpha/transparency info through material, expose it.
        if material is not None and hasattr(material, "alphaMode"):
            result["notes"].append(f"alphaMode={getattr(material, 'alphaMode', None)}")
        return result

    # -------- ColorVisuals --------
    if isinstance(vis, trimesh.visual.color.ColorVisuals):
        result["visual_category"] = "color"
        result["is_color_visual"] = True
        result["color_mode"] = getattr(vis, "kind", None)

        result["face_colors_shape"] = _shape(getattr(vis, "face_colors", None))
        result["vertex_colors_shape"] = _shape(getattr(vis, "vertex_colors", None))
        result["has_transparency"] = bool(getattr(vis, "transparency", False))
        result["main_color"] = _to_jsonable(getattr(vis, "main_color", None))

        if result["color_mode"] == "face":
            result["texture_state"] = "color_face"
        elif result["color_mode"] == "vertex":
            result["texture_state"] = "color_vertex"
        else:
            result["texture_state"] = "color_unset_or_default"

        return result

    # -------- Unknown visual subclass --------
    result["visual_category"] = "unknown"
    result["notes"].append(
        f"Unhandled visual type: {type(vis).__name__}. Inspect __dict__ for custom extension."
    )

    # Best-effort generic dump for custom visuals
    if hasattr(vis, "__dict__"):
        result["material"] = {
            "raw_attributes": {k: _to_jsonable(v) for k, v in vis.__dict__.items()}
        }

    return result
