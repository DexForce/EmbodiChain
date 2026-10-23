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

import copy
import math
import torch
import dexsim
import numpy as np

from dataclasses import dataclass
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from dexsim.engine import MaterialInst, Material
from embodichain.utils import configclass, logger

if TYPE_CHECKING:
    from dexsim.spawn import MaterialDesc
    from ._material_presets import _AssetMaterialDesc

__all__ = [
    "VisualMaterialCfg",
    "VisualMaterial",
    "VisualMaterialInst",
    "ReuseSegmentState",
]


@configclass
class VisualMaterialCfg:
    """Manual PBR properties or explicit overrides of a default material preset.

    With ``preset=None``, unspecified common fields receive the historical
    defaults at construction. With a preset, ``None`` inherits the authored
    resource value. Serialization retains that distinction. Use
    :meth:`from_preset` to construct a new preset configuration; ``replace``
    preserves existing fields rather than switching configuration sources.
    """

    uid: str | None = None
    """Explicit shared identity, or None for an automatically named material."""

    # Basic PBR properties
    base_color: list[float] | None = None
    """Base color/diffuse color (RGBA)"""

    metallic: float | None = None
    """Metallic factor (0.0 = dielectric, 1.0 = metallic)"""

    roughness: float | None = None
    """Surface roughness (0.0 = smooth, 1.0 = rough)"""

    # Additional PBR properties
    emissive: list[float] | None = None  # Emissive color (RGB)
    emissive_intensity: float | None = None  # Emissive intensity multiplier

    # Texture maps
    base_color_texture: str | None = None
    """Base color texture map"""

    metallic_texture: str | None = None
    """Metallic map"""

    roughness_texture: str | None = None
    """Roughness map"""

    normal_texture: str | None = None
    """Normal map"""

    ao_texture: str | None = None
    """Ambient occlusion map"""

    # Ray tracing specific properties
    ior: float | None = None
    """Index of refraction for PBR materials, only used in ray tracing."""

    material_type: str | None = None
    """Ray tracing material type. Options: 'BRDF', 'BTDF', 'BSDF'"""

    preset: str | None = None
    """Default material name. Presets select scattering from native parameters."""

    parameters: dict[str, Any] = {}
    """Native schema-v1 advanced parameters; cannot duplicate common fields."""

    def __post_init__(self) -> None:
        if self.preset is None:
            defaults = {
                "base_color": [0.5, 0.5, 0.5, 1.0],
                "metallic": 0.0,
                "roughness": 0.7,
                "emissive": [0.0, 0.0, 0.0],
                "emissive_intensity": 1.0,
                "ior": 1.5,
                "material_type": "BRDF",
            }
            for key, value in defaults.items():
                if getattr(self, key) is None:
                    setattr(self, key, copy.deepcopy(value))
        else:
            from ._material_presets import _asset_desc_from_cfg

            _asset_desc_from_cfg(self, self.uid or "preset")

    @classmethod
    def from_preset(
        cls,
        name: str,
        *,
        uid: str | None = None,
        parameters: dict[str, Any] | None = None,
        **properties: Any,
    ) -> VisualMaterialCfg:
        """Construct explicit overrides of an authored default material.

        Args:
            name: Case-sensitive catalog name, or ``SoapBubbleInterior``.
            uid: Optional shared identity within the simulation World.
            parameters: Advanced native parameters, such as volume absorption.
            **properties: Common fields such as ``ior`` or ``roughness``.

        Returns:
            A configuration retaining ``None`` for inherited common fields.

        Raises:
            ValueError: A preset, override, duplicate property or texture edit
                is invalid. ``material_type`` is unavailable for presets.
        """
        from embodichain.data.default_materials import get_default_material_info

        get_default_material_info(name)
        return cls(
            preset=name,
            uid=uid,
            parameters={} if parameters is None else parameters,
            **properties,
        )

    @classmethod
    def from_dict(cls, cfg_dict: dict[str, Any]) -> VisualMaterialCfg:
        """Construct manual defaults or preset overrides from Python/YAML data.

        Args:
            cfg_dict: Serialized configuration, including optional ``preset``.

        Returns:
            Configuration with inheritance preserved before defaults are filled.
        """
        from dataclasses import fields

        names = {item.name for item in fields(cls)}
        values = {}
        for k, v in cfg_dict.items():
            if k in names:
                values[k] = v
            else:
                if cfg_dict.get("preset") is not None:
                    raise ValueError(
                        f"Unknown field {k!r} in preset VisualMaterialCfg."
                    )
                logger.log_warning(f"Unknown field '{k}' in VisualMaterialCfg.")
        return cls(**values)


_MATERIAL_TYPES = {
    "BRDF": "BRDF_GGX_SMITH",
    "BTDF": "BTDF_GGX_SMITH",
    "BSDF": "BSDF_GGX_SMITH",
}


def _material_scalar(
    value: float, name: str, *, lower: float = 0.0, upper: float | None = None
) -> float:
    value = float(value)
    if (
        not math.isfinite(value)
        or value < lower
        or (upper is not None and value > upper)
    ):
        raise ValueError(f"Visual material {name} has an invalid value: {value}.")
    return value


def _material_color(value: Sequence[float], name: str, size: int) -> tuple[float, ...]:
    if len(value) != size:
        raise ValueError(f"Visual material {name} must have {size} components.")
    return tuple(_material_scalar(v, name) for v in value)


def _material_desc_from_cfg(
    cfg: VisualMaterialCfg, uid: str
) -> MaterialDesc | _AssetMaterialDesc:
    """Normalize both declaration and runtime material creation at one boundary."""
    from dexsim.spawn import MaterialDesc

    if not isinstance(uid, str) or not uid.strip():
        raise ValueError("Visual material uid must be a nonempty string.")
    if not isinstance(cfg.parameters, dict):
        raise ValueError("Visual material parameters must be a dictionary.")
    if cfg.preset is not None:
        from ._material_presets import _asset_desc_from_cfg

        return _asset_desc_from_cfg(cfg, uid)
    if cfg.material_type not in _MATERIAL_TYPES:
        raise ValueError(f"Invalid visual material_type {cfg.material_type!r}.")
    if cfg.parameters:
        from ._material_presets import _asset_desc_from_cfg

        return _asset_desc_from_cfg(cfg, uid)
    color = _material_color(cfg.base_color, "base_color (RGBA)", 4)
    _material_scalar(color[3], "base_color alpha", upper=1.0)
    emissive = _material_color(cfg.emissive, "emissive (RGB)", 3)
    intensity = _material_scalar(cfg.emissive_intensity, "emissive_intensity")
    ior = _material_scalar(cfg.ior, "ior")
    if ior == 0:
        raise ValueError("Visual material ior must be positive.")
    return MaterialDesc(
        name=uid,
        base_color=color,
        base_color_map=cfg.base_color_texture,
        metallic=_material_scalar(cfg.metallic, "metallic", upper=1.0),
        metallic_map=cfg.metallic_texture,
        roughness=_material_scalar(cfg.roughness, "roughness", upper=1.0),
        roughness_map=cfg.roughness_texture,
        normal_map=cfg.normal_texture,
        ao_map=cfg.ao_texture,
        emissive=(
            *(_material_scalar(v * intensity, "emissive") for v in emissive),
            1.0,
        ),
        ior=ior,
        material_type=_MATERIAL_TYPES[cfg.material_type],
        has_alpha=True,
    )


class VisualMaterial:
    """Visual material definition in the simulation environment.

    A visual material is actually a material template from which material instances can be created.
    It holds multiple material instances, which is used to assign to different objects in the environment.
    """

    RT_MATERIAL_TYPES = [
        "BRDF",
        "BTDF",
        "BSDF",
    ]

    MAT_TYPE_MAPPING: dict[str, str] = _MATERIAL_TYPES

    def __init__(
        self,
        cfg: VisualMaterialCfg,
        mat: Material,
        *,
        initialize_default: bool = True,
        mat_inst: MaterialInst | None = None,
    ) -> None:
        """Create a template wrapper or borrow an already configured native one.

        Args:
            cfg: Defaults for newly created material instances.
            mat: Native template owned by the simulation.
            initialize_default: Apply configuration to the native default instance.
                Disable for a known existing template to preserve runtime edits.
            mat_inst: Exact child instance of an immutable native resource template.
        """
        self.uid = cfg.uid or f"visual_material_{uuid4().hex}"
        self._definition = _material_desc_from_cfg(cfg, self.uid)
        self._uses_manifest = cfg.preset is not None or bool(cfg.parameters)
        self.cfg = copy.deepcopy(cfg)
        self.cfg.uid = self.uid
        self._mat = mat
        self._instances: dict[str, VisualMaterialInst] = {}
        self._native_instance_aliases: dict[str, VisualMaterialInst] = {}

        # Spawn resolves a named template through its native default instance.
        # Keep that entry configured too when runtime and declarations share a UID.
        if mat_inst is None and self._uses_manifest:
            mat_inst = self._mat.get_inst(f"ec_instance_{uuid4().hex}")
        self._default_mat_inst = VisualMaterialInst(
            self.uid,
            self._mat,
            mat_inst=mat_inst if mat_inst is not None else self._mat.get_inst(),
        )
        if initialize_default and not self._uses_manifest:
            self.set_default_properties(self._default_mat_inst, self.cfg)
        self._instances[self.uid] = self._default_mat_inst
        if self._uses_manifest:
            self._native_instance_aliases[self._default_mat_inst.mat.get_name()] = (
                self._default_mat_inst
            )

    @property
    def mat(self) -> Material:
        return self._mat

    @property
    def inst(self) -> VisualMaterialInst:
        return self._default_mat_inst

    def set_default_properties(
        self, mat_inst: VisualMaterialInst, cfg: VisualMaterialCfg
    ) -> None:
        """Explicitly apply validated configuration to an existing instance.

        Args:
            mat_inst: Instance whose properties will be replaced.
            cfg: Material properties to apply.
        """
        desc = _material_desc_from_cfg(cfg, self.uid)
        if self._uses_manifest:
            if desc != self._definition:
                raise ValueError(
                    "Resource material reset requires its original definition; create a variant before binding."
                )
            self._mat.reset_inst(mat_inst.mat)
            mat_inst._emissive_rgb = None
            mat_inst._emissive_intensity = 1.0
            mat_inst.base_color_texture = mat_inst.mat.get_base_color_map()
            mat_inst.metallic_texture = None
            mat_inst.roughness_texture = None
            mat_inst.normal_texture = None
            mat_inst.ao_texture = None
            return
        mat_inst.set_base_color(desc.base_color)
        mat_inst.set_metallic(desc.metallic)
        mat_inst.set_roughness(desc.roughness)
        mat_inst._set_emissive(cfg.emissive, cfg.emissive_intensity)

        mat_inst.set_base_color_texture(cfg.base_color_texture)
        mat_inst.set_metallic_texture(cfg.metallic_texture)
        mat_inst.set_roughness_texture(cfg.roughness_texture)
        mat_inst.set_normal_texture(cfg.normal_texture)
        mat_inst.set_ao_texture(cfg.ao_texture)

        mat_inst.set_ior(desc.ior)
        mat_inst.mat.update_pbr_material_type(desc.material_type)

    def create_instance(self, uid: str) -> VisualMaterialInst:
        """Create a new material instance from this material template.

        Note:
            - If the uid already exists, the existing instance will be returned.

        Args:
            uid (str): Unique identifier for the material instance.

        Returns:
            VisualMaterialInst: The created material instance.
        """
        if uid in self._instances:
            return self._instances[uid]
        if self._uses_manifest:
            # Distinct public templates may share a native immutable template.
            # Keep child names private so their user-provided UIDs cannot alias.
            inst = VisualMaterialInst(
                uid,
                self._mat,
                mat_inst=self._mat.get_inst(f"ec_instance_{uuid4().hex}"),
            )
        else:
            inst = VisualMaterialInst(uid, self._mat)
            self.set_default_properties(inst, self.cfg)
        self._instances[uid] = inst
        if self._uses_manifest:
            self._native_instance_aliases[inst.mat.get_name()] = inst
        return inst

    def get_default_instance(self) -> VisualMaterialInst:
        """Get the default material instance created with the same uid as the material template.

        Returns:
            VisualMaterialInst: The default material instance.
        """
        return self._default_mat_inst

    def get_instance(self, uid: str) -> VisualMaterialInst:
        """Get an existing material instance by its uid.

        Args:
            uid (str): Unique identifier for the material instance.

        Returns:
            VisualMaterialInst: The material instance.
        """
        if uid in self._instances:
            return self._instances[uid]
        if uid in self._native_instance_aliases:
            return self._native_instance_aliases[uid]
        if uid not in self._instances:
            logger.log_error(f"Material instance with uid '{uid}' does not exist.")
        return self._instances[uid]


class VisualMaterialInst:
    """Instance of a visual material in the simulation environment."""

    def __init__(
        self,
        uid: str,
        mat: Material,
        mat_inst: MaterialInst | None = None,
    ) -> None:
        self.uid = uid
        self._mat = mat
        self._mat_inst = mat_inst if mat_inst is not None else mat.get_inst(uid)
        self._emissive_rgb: tuple[float, ...] | None = None
        self._emissive_intensity = 1.0
        self.base_color_texture = self.mat.get_base_color_map()
        self.metallic_texture = None
        self.roughness_texture = None
        self.normal_texture = None
        self.ao_texture = None

    @classmethod
    def from_existing(cls, mat_inst: MaterialInst) -> VisualMaterialInst:
        """Wrap an existing dexsim material instance without copying it.

        Args:
            mat_inst: Existing dexsim material instance parsed from an asset.

        Returns:
            The EmbodiChain wrapper for the existing instance.

        Raises:
            ValueError: If the material instance has no template.
        """
        mat = mat_inst.get_template()
        if mat is None:
            raise ValueError("Cannot wrap a material instance without a template.")
        return cls(uid=mat_inst.get_name(), mat=mat, mat_inst=mat_inst)

    @property
    def mat(self) -> MaterialInst:
        return self._mat_inst

    @property
    def base_color(self) -> list[float]:
        """Current native display-sRGB color (RGB); DexSim exposes no alpha getter."""
        return [float(v) for v in self.mat.get_base_color()[:3]]

    @property
    def metallic(self) -> float:
        """Current native metallic factor."""
        return float(self.mat.get_metallic())

    @property
    def roughness(self) -> float:
        """Current native roughness."""
        return float(self.mat.get_roughness())

    @property
    def ior(self) -> float:
        """Current native index of refraction."""
        return float(self.mat.get_ior())

    @property
    def emissive(self) -> list[float]:
        """Unscaled emissive RGB, including when intensity is zero."""
        if self._emissive_intensity == 0 and self._emissive_rgb is not None:
            return list(self._emissive_rgb)
        return [
            float(v) / self._emissive_intensity for v in self.mat.get_emissive()[:3]
        ]

    @property
    def emissive_intensity(self) -> float:
        """Multiplier applied to emissive RGB by this wrapper."""
        return self._emissive_intensity

    def set_base_color(self, color: Sequence[float]) -> None:
        """Set base color/diffuse color."""
        color = _material_color(color, "base_color (RGBA)", 4)
        _material_scalar(color[3], "base_color alpha", upper=1.0)
        self.mat.set_base_color(list(color))

    def set_metallic(self, metallic: float) -> None:
        """Set metallic factor."""
        self.mat.set_metallic(_material_scalar(metallic, "metallic", upper=1.0))

    def set_roughness(self, roughness: float) -> None:
        """Set surface roughness."""
        self.mat.set_roughness(_material_scalar(roughness, "roughness", upper=1.0))

    def set_emissive(self, emissive: Sequence[float]) -> None:
        """Set emissive color."""
        self._set_emissive(emissive, self._emissive_intensity)

    def _set_emissive(self, emissive: Sequence[float], intensity: float) -> None:
        rgb = _material_color(emissive, "emissive (RGB)", 3)
        intensity = _material_scalar(intensity, "emissive_intensity")
        value = [_material_scalar(v * intensity, "emissive") for v in rgb]
        self.mat.set_emissive(np.asarray([*value, 1.0]))
        self._emissive_rgb = rgb
        self._emissive_intensity = intensity

    def set_emissive_intensity(self, intensity: float) -> None:
        """Set emissive intensity multiplier."""
        self._set_emissive(self.emissive, intensity)

    def set_base_color_texture(
        self,
        texture_path: str = None,
        texture_data: torch.Tensor | None = None,
        texture_obj=None,
    ) -> None:
        """Set base color texture from file path, tensor data, or a pre-created Texture.

        Args:
            texture_path: Path to texture file.
            texture_data: Texture data as a torch.Tensor (uploaded each call).
            texture_obj: A pre-created dexsim ``Texture`` object. When provided, it is
                bound directly without re-uploading (priority over ``texture_data``).
        """
        if texture_path is not None and (
            texture_data is not None or texture_obj is not None
        ):
            logger.log_warning(
                "Both texture_path and another texture source are provided. Using texture_path."
            )

        if texture_path is not None:
            self.base_color_texture = texture_path
            inst = self.mat
            inst.set_base_color_map(texture_path)
        elif texture_obj is not None:
            self.base_color_texture = texture_obj
            inst = self.mat
            inst.set_base_color_map(texture_obj)
        elif texture_data is not None:
            self.base_color_texture = texture_data
            inst = self.mat

            # TODO: Optimize texture creation method.
            world = dexsim.default_world()
            env = world.get_env()
            color_texture = env.create_color_texture(
                texture_data.cpu().numpy(), has_alpha=True
            )
            inst.set_base_color_map(color_texture)

    def set_metallic_texture(
        self, texture_path: str = None, texture_data: torch.Tensor | None = None
    ) -> None:
        """Set metallic texture from file path or texture data.

        Args:
            texture_path: Path to texture file
            texture_data: Texture data as a torch.Tensor
        """
        if texture_path is not None and texture_data is not None:
            logger.log_warning(
                "Both texture_path and texture_data are provided. Using texture_path."
            )

        if texture_path is not None:
            self.metallic_texture = texture_path
            inst = self.mat
            inst.set_metallic_map(texture_path)
        elif texture_data is not None:
            self.metallic_texture = texture_data
            inst = self.mat

            # TODO: Optimize texture creation method.
            world = dexsim.default_world()
            env = world.get_env()
            metallic_texture = env.create_color_texture(
                texture_data.cpu().numpy(), has_alpha=False
            )
            inst.set_metallic_map(metallic_texture)

    def set_roughness_texture(
        self, texture_path: str = None, texture_data: torch.Tensor | None = None
    ) -> None:
        """Set roughness texture from file path or texture data.

        Args:
            texture_path: Path to texture file
            texture_data: Texture data as a torch.Tensor
        """
        if texture_path is not None and texture_data is not None:
            logger.log_warning(
                "Both texture_path and texture_data are provided. Using texture_path."
            )

        if texture_path is not None:
            self.roughness_texture = texture_path
            inst = self.mat
            inst.set_roughness_map(texture_path)
        elif texture_data is not None:
            self.roughness_texture = texture_data
            inst = self.mat

            # TODO: Optimize texture creation method.
            world = dexsim.default_world()
            env = world.get_env()
            roughness_texture = env.create_color_texture(
                texture_data.cpu().numpy(), has_alpha=False
            )
            inst.set_roughness_map(roughness_texture)

    def set_normal_texture(
        self, texture_path: str = None, texture_data: torch.Tensor | None = None
    ) -> None:
        """Set normal texture from file path or texture data.

        Args:
            texture_path: Path to texture file
            texture_data: Texture data as a torch.Tensor
        """
        if texture_path is not None and texture_data is not None:
            logger.log_warning(
                "Both texture_path and texture_data are provided. Using texture_path."
            )

        if texture_path is not None:
            self.normal_texture = texture_path
            inst = self.mat
            inst.set_normal_map(texture_path)
        elif texture_data is not None:
            self.normal_texture = texture_data
            inst = self.mat

            # TODO: Optimize texture creation method.
            world = dexsim.default_world()
            env = world.get_env()
            normal_texture = env.create_color_texture(
                texture_data.cpu().numpy(), has_alpha=False
            )
            inst.set_normal_map(normal_texture)

    def set_ao_texture(
        self, texture_path: str = None, texture_data: torch.Tensor | None = None
    ) -> None:
        """Set ambient occlusion texture from file path or texture data.

        Args:
            texture_path: Path to texture file
            texture_data: Texture data as a torch.Tensor
        """
        if texture_path is not None and texture_data is not None:
            logger.log_warning(
                "Both texture_path and texture_data are provided. Using texture_path."
            )

        if texture_path is not None:
            self.ao_texture = texture_path
            inst = self.mat
            inst.set_ao_map(texture_path)
        elif texture_data is not None:
            self.ao_texture = texture_data
            inst = self.mat

            # TODO: Optimize texture creation method.
            world = dexsim.default_world()
            env = world.get_env()
            ao_texture = env.create_color_texture(
                texture_data.cpu().numpy(), has_alpha=False
            )
            inst.set_ao_map(ao_texture)

    def set_ior(self, ior: float) -> None:
        """Set index of refraction."""

        ior = _material_scalar(ior, "ior")
        if ior == 0:
            raise ValueError("Visual material ior must be positive.")
        self.mat.set_pbr_param("ior", ior)


@dataclass
class ReuseSegmentState:
    """Reuse state for one render-body segment of a parsed object.

    Used by ``randomize_visual_material`` to randomize on top of the material dexsim
    parsed from the asset. It creates a working material instance from the existing
    template, but does not create a new material template.

    Attributes:
        mesh_id: The render-body segment index.
        original_inst: The dexsim ``MaterialInst`` parsed from the asset. Kept immutable
            and swapped back onto the render body for the "original" tier.
        working_inst: A ``VisualMaterialInst`` created from the first segment's
            template and shared by sibling segments; mutated in place for the
            "library"/"solid" tiers.
    """

    mesh_id: int
    original_inst: MaterialInst
    working_inst: VisualMaterialInst


def _capture_render_materials(render_body) -> list[MaterialInst | None]:
    """Capture the material attached to every render-body mesh segment."""
    return [
        render_body.get_material(mesh_id)
        for mesh_id in range(render_body.get_mesh_count())
    ]


def _wrap_first_render_material(
    materials: list[MaterialInst | None],
) -> VisualMaterialInst | None:
    """Wrap the first material instance that has a template."""
    for mat_inst in materials:
        if mat_inst is not None and mat_inst.get_template() is not None:
            return VisualMaterialInst.from_existing(mat_inst)
    return None


def _is_same_material_inst(
    current_inst: MaterialInst | None, original_inst: MaterialInst | None
) -> bool:
    """Return whether two handles refer to the same material instance."""
    if current_inst is original_inst:
        return True
    if current_inst is None or original_inst is None:
        return False
    return current_inst.get_material_id() == original_inst.get_material_id()


def _set_render_material(render_body, mesh_id: int, mat_inst: MaterialInst) -> None:
    """Set a segment material only when it is not already attached."""
    if _is_same_material_inst(render_body.get_material(mesh_id), mat_inst):
        return
    render_body.set_material(mesh_id, mat_inst)


def _restore_render_materials(
    render_body, original_materials: list[MaterialInst | None]
) -> None:
    """Restore a render body's captured per-segment material assignments."""
    # A null original can only be recovered through the render body's default
    # material. Cleaning is unnecessary when every null segment is already null.
    if any(
        original_inst is None and render_body.get_material(mesh_id) is not None
        for mesh_id, original_inst in enumerate(original_materials)
    ):
        render_body.clean_material()

    for mesh_id, original_inst in enumerate(original_materials):
        if original_inst is not None:
            _set_render_material(render_body, mesh_id, original_inst)
