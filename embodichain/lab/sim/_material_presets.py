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

"""Validated native material definitions and World-owned resource lifetimes."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from numbers import Real
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, TYPE_CHECKING
from uuid import uuid4

from embodichain.data.default_materials import get_default_material_info

if TYPE_CHECKING:
    from .material import VisualMaterialCfg

__all__: list[str] = []

_FLAT_PARAMETERS = {
    "base_color": "base_color",
    "metallic": "metallic",
    "roughness": "roughness",
    "emissive": "emissive",
    "emissive_intensity": "emission_intensity",
    "ior": "ior",
}
_TEXTURE_FIELDS = {
    "base_color_texture": "base_color",
    "metallic_texture": "metallic",
    "roughness_texture": "roughness",
    "normal_texture": "normal",
    "ao_texture": "ambient_occlusion",
}
_FRACTIONS = {
    "metallic",
    "roughness",
    "reflectance",
    "ambient_occlusion",
    "transmission",
    "anisotropy_strength",
    "clearcoat",
    "clearcoat_roughness",
    "sheen_roughness",
    "specular_strength",
    "diffuse_transmission_strength",
    "iridescence_factor",
    "iridescence_min_reflection",
}
_POSITIVE = {"ior", "volume_attenuation_distance", "iridescence_ior"}
_NONNEGATIVE = {
    "emission_intensity",
    "volume_thickness",
    "iridescence_reflection_boost",
}
_VECTORS = {
    "base_color": (3, 4),
    "emissive": (3,),
    "sheen_color": (3,),
    "specular_color": (3,),
    "diffuse_transmission_color": (3,),
    "volume_attenuation_color": (3,),
    "iridescence_thickness": (2,),
}
_BOOLEANS = {"thin_walled", "double_sided"}
_BASIC_PARAMETERS = {
    "base_color",
    "metallic",
    "roughness",
    "reflectance",
    "ambient_occlusion",
    "ior",
    "emissive",
    "emission_intensity",
    "double_sided",
}
_TEXTURE_KEYS = {
    "base_color",
    "ambient_occlusion",
    "roughness",
    "metallic",
    "metallic_roughness",
    "normal",
    "emissive",
    "anisotropy",
    "transmission",
    "clearcoat",
    "clearcoat_roughness",
    "clearcoat_normal",
    "sheen_color",
    "sheen_roughness",
    "iridescence",
    "iridescence_thickness",
    "specular_color",
    "specular_strength",
    "diffuse_transmission_color",
    "diffuse_transmission_strength",
    "volume_attenuation_strength",
}


def _number(value: Any, key: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"Visual material parameter {key!r} must be a finite number.")
    return float(value)


def _normalize_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(parameters, dict):
        raise ValueError("Visual material parameters must be a dictionary.")
    normalized = {}
    for supplied_key, value in parameters.items():
        key = (
            "emission_intensity"
            if supplied_key == "emissive_intensity"
            else supplied_key
        )
        if key in normalized:
            raise ValueError(f"Duplicate visual material parameter {key!r}.")
        if key in _BOOLEANS:
            if not isinstance(value, bool):
                raise ValueError(
                    f"Visual material parameter {key!r} must be a boolean."
                )
            normalized[key] = value
        elif key in _VECTORS:
            if (
                not isinstance(value, Sequence)
                or isinstance(value, (str, bytes))
                or len(value) not in _VECTORS[key]
            ):
                raise ValueError(f"Invalid visual material vector {key!r}.")
            vector = [_number(component, key) for component in value]
            if any(component < 0 for component in vector) or (
                key == "base_color" and len(vector) == 4 and vector[3] > 1
            ):
                raise ValueError(f"Invalid visual material vector {key!r}.")
            if key == "iridescence_thickness" and vector[0] > vector[1]:
                raise ValueError(
                    "iridescence_thickness must be ordered from minimum to maximum."
                )
            normalized[key] = vector
        elif key in _FRACTIONS | _POSITIVE | _NONNEGATIVE | {"anisotropy_rotation"}:
            scalar = _number(value, key)
            if (
                (key in _FRACTIONS and not 0 <= scalar <= 1)
                or (key in _POSITIVE and scalar <= 0)
                or (key in _NONNEGATIVE and scalar < 0)
            ):
                raise ValueError(
                    f"Invalid visual material parameter {key!r}: {value!r}."
                )
            normalized[key] = scalar
        else:
            raise ValueError(f"Unknown visual material parameter {supplied_key!r}.")
    return normalized


def _cfg_parameters(cfg: VisualMaterialCfg) -> dict[str, Any]:
    parameters = _normalize_parameters(cfg.parameters)
    for field_name, key in _FLAT_PARAMETERS.items():
        value = getattr(cfg, field_name)
        if value is not None:
            if key in parameters:
                raise ValueError(
                    f"Duplicate visual material parameter {key!r} in flat fields and parameters."
                )
            parameters[key] = value
    return _normalize_parameters(parameters)


@dataclass
class _AssetMaterialDesc:
    # Keep identity independent of download roots, temporary paths and native
    # handles. Names are mutable only for anonymous Spawn declaration naming.
    name: str
    preset: str | None
    parameters_json: str
    revision: str
    textures: tuple[tuple[str, str], ...] = ()
    material_type: str | None = None

    def template_key(self) -> str:
        definition = (
            self.revision,
            self.preset,
            self.parameters_json,
            self.textures,
            self.material_type,
        )
        return sha256(repr(definition).encode()).hexdigest()


def _asset_desc_from_cfg(cfg: VisualMaterialCfg, uid: str) -> _AssetMaterialDesc:
    if cfg.preset is not None:
        info = get_default_material_info(cfg.preset)
        if cfg.material_type is not None:
            raise ValueError(
                "Preset materials cannot specify material_type; native parameters select scattering."
            )
        if any(getattr(cfg, key) is not None for key in _TEXTURE_FIELDS):
            raise ValueError("Preset texture replacement or removal is not supported.")
        revision = info["revision"]
    else:
        revision = "native-manifest-v1"
    parameters = _cfg_parameters(cfg)
    return _AssetMaterialDesc(
        name=uid,
        preset=cfg.preset,
        parameters_json=json.dumps(
            parameters, sort_keys=True, separators=(",", ":"), allow_nan=False
        ),
        revision=revision,
        textures=tuple(
            (key, str(getattr(cfg, field_name)))
            for field_name, key in _TEXTURE_FIELDS.items()
            if getattr(cfg, field_name) is not None
        ),
        material_type=cfg.material_type if cfg.preset is None else None,
    )


def _validate_backend(desc: _AssetMaterialDesc, renderer: str) -> None:
    # NoRender stores the full declaration; no native resource is requested.
    if renderer == "no-render":
        return
    supported = (
        get_default_material_info(desc.preset)["renderers"]
        if desc.preset
        else ["rt", "hybrid", "fast-rt"]
    )
    advanced = set(json.loads(desc.parameters_json)) - _BASIC_PARAMETERS
    if renderer not in supported or (advanced and renderer != "rt"):
        raise ValueError(
            f"Visual material {desc.preset or desc.name!r} requires renderer='rt' (OfflineRT); current renderer is {renderer!r}."
        )


def _legacy_manifest(directory: Path) -> dict[str, Any]:
    # Match the native suffix discovery, including its alphabetical precedence
    # for Metalness over Metallic. Displacement remains authoring-only.
    suffixes = {
        "AmbientOcclusion": "ambient_occlusion",
        "Color": "base_color",
        "Metallic": "metallic",
        "MetallicRoughness": "metallic_roughness",
        "Metalness": "metallic",
        "Normal": "normal",
        "Roughness": "roughness",
    }
    textures = {}
    for suffix, key in suffixes.items():
        matches = sorted(directory.glob(f"*_{suffix}.png"))
        if matches:
            textures[key] = str(matches[0].resolve())
    return {"schema_version": 1, "parameters": {}, "textures": textures}


class _MaterialResources:
    """One World owns staged manifests, immutable templates and UID handles.

    Cost scales with distinct definitions and UIDs, never with frames or repeated
    lookups. Only JSON is staged; source images stay in the shared data cache.
    """

    def __init__(self, world: Any, renderer: str) -> None:
        self._world = world
        self.renderer = renderer
        self._directory: TemporaryDirectory[str] | None = None
        self._dataset: Any | None = None
        self._templates: dict[str, Any] = {}
        self._instances: dict[str, tuple[Any, Any]] = {}

    def _manifest(self, desc: _AssetMaterialDesc) -> dict[str, Any]:
        if desc.preset:
            if self._dataset is None:
                from embodichain.data.assets.materials import DefaultMaterials

                self._dataset = DefaultMaterials()
            directory = Path(self._dataset.get_material_path(desc.preset))
            source = directory / "material.json"
            manifest = (
                json.loads(source.read_text(encoding="utf-8"))
                if source.is_file()
                else _legacy_manifest(directory)
            )
            pack_root = directory.parent.resolve()
        else:
            directory = Path.cwd()
            pack_root = None
            manifest = {
                "schema_version": 1,
                "parameters": {},
                "textures": dict(desc.textures),
            }
        if manifest.get("schema_version") != 1:
            raise ValueError(
                "Only native material manifest schema_version=1 is supported."
            )
        manifest = copy.deepcopy(manifest)
        parameters = _normalize_parameters(manifest.get("parameters", {}))
        parameters.update(json.loads(desc.parameters_json))
        manifest["parameters"] = _normalize_parameters(parameters)
        textures = manifest.get("textures", {})
        if not isinstance(textures, dict) or set(textures) - _TEXTURE_KEYS:
            raise ValueError("Invalid native material texture definitions.")
        for key, value in textures.items():
            if not isinstance(value, str) or not value:
                raise ValueError(f"Invalid native material texture {key!r}.")
            path = (directory / value).resolve()
            if pack_root is not None and not path.is_relative_to(pack_root):
                raise ValueError(
                    f"Material texture escapes its resource pack: {value!r}."
                )
            if not path.is_file():
                raise FileNotFoundError(f"Missing material texture: {path}.")
            textures[key] = str(path)
        return manifest

    def resolve(self, desc: _AssetMaterialDesc) -> tuple[Any, Any]:
        _validate_backend(desc, self.renderer)
        if self.renderer == "no-render":
            raise RuntimeError("Native material creation requires a renderer.")
        if desc.name in self._instances:
            return self._instances[desc.name]
        key = desc.template_key()
        if key not in self._templates:
            manifest = self._manifest(desc)
            if self._directory is None:
                self._directory = TemporaryDirectory(prefix="ec-materials-")
            directory = Path(self._directory.name) / key
            directory.mkdir(exist_ok=True)
            (directory / "material.json").write_text(
                json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
            )
            if len(str(directory).encode()) > 255:
                raise ValueError(
                    "Native material directory exceeds the 255-byte path limit."
                )
            template = self._world.get_env().load_material(
                str(directory), f"ec_material_{uuid4().hex}"
            )
            if template is None:
                raise RuntimeError(
                    f"DexSim could not load material {desc.preset or desc.name!r}."
                )
            if desc.material_type is not None:
                from .material import _MATERIAL_TYPES

                template.get_inst().update_pbr_material_type(
                    _MATERIAL_TYPES[desc.material_type]
                )
            self._templates[key] = template
        template = self._templates[key]
        instance = template.get_inst(f"ec_instance_{uuid4().hex}")
        result = (template, instance)
        self._instances[desc.name] = result
        return result

    def close(self) -> None:
        self._instances.clear()
        self._templates.clear()
        self._dataset = None
        if self._directory is not None:
            self._directory.cleanup()
            self._directory = None
        self._world = None
