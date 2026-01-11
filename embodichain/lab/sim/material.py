# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
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
import torch
import dexsim
import numpy as np

from typing import Dict, Union
from functools import cached_property

from dexsim.engine import MaterialInst, Material
from embodichain.lab.sim.utility import is_rt_enabled
from embodichain.utils import configclass, logger


@configclass
class VisualMaterialCfg:
    """Configuration for visual material with PBR properties for rasterization and ray tracing."""

    uid: str = "default_mat"

    # Basic PBR properties
    base_color: list = [0.5, 0.5, 0.5, 1.0]
    """Base color/diffuse color (RGBA)"""

    metallic: float = 0.0
    """Metallic factor (0.0 = dielectric, 1.0 = metallic)"""

    roughness: float = 0.5
    """Surface roughness (0.0 = smooth, 1.0 = rough)"""

    # Additional PBR properties
    emissive: list = [0.0, 0.0, 0.0]  # Emissive color (RGB)
    emissive_intensity: float = 1.0  # Emissive intensity multiplier

    # Texture maps
    base_color_texture: str = None
    """Base color texture map"""

    metallic_texture: str = None
    """Metallic map"""

    roughness_texture: str = None
    """Roughness map"""

    normal_texture: str = None
    """Normal map"""

    ao_texture: str = None
    """Ambient occlusion map"""

    # Ray tracing specific properties
    ior: float = 1.5
    """Index of refraction for PBR materials, only used in ray tracing."""

    material_type: str = "BRDF"
    """Ray tracing material type. Options: 'BRDF', 'BTDF', 'BSDF'"""

    # Currently disabled properties
    # subsurface: float = 0.0  # Subsurface scattering factor
    # subsurface_color: list = [1.0, 1.0, 1.0]  # Subsurface scattering color

    @classmethod
    def from_dict(cls, cfg_dict: dict) -> VisualMaterialCfg:
        base = cls()
        for k, v in cfg_dict.items():
            if hasattr(base, k):
                setattr(base, k, v)
            else:
                logger.log_warning(f"Unknown field '{k}' in VisualMaterialCfg.")
        return base


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

    MAT_TYPE_MAPPING: Dict[str, str] = {
        "BRDF": "BRDF_GGX_SMITH",
        "BTDF": "BTDF_GGX_SMITH",
        "BSDF": "BSDF_GGX_SMITH",
    }

    def __init__(self, cfg: VisualMaterialCfg, mat: Material):
        if cfg.material_type not in self.RT_MATERIAL_TYPES:
            logger.log_error(
                f"Invalid material_type '{cfg.material_type}'. "
                f"Supported types: {self.RT_MATERIAL_TYPES}"
            )

        self.uid = cfg.uid
        self.cfg = copy.deepcopy(cfg)
        self._mat = mat
        self._mat_inst_list: list[str] = []

        self._default_mat_inst = self.create_instance(self.uid)

    @cached_property
    def is_rt_enabled(self) -> bool:
        return is_rt_enabled()

    @property
    def mat(self) -> Material:
        return self._mat

    @property
    def inst(self) -> VisualMaterialInst:
        return self._default_mat_inst

    def set_default_properties(
        self, mat_inst: VisualMaterialInst, cfg: VisualMaterialCfg
    ) -> None:
        mat_inst.set_base_color(cfg.base_color)
        mat_inst.set_metallic(cfg.metallic)
        mat_inst.set_roughness(cfg.roughness)
        mat_inst.set_emissive(cfg.emissive)
        # mat_inst.set_emissive_intensity(self.cfg.emissive_intensity)  # Unimplemented

        mat_inst.set_base_color_texture(cfg.base_color_texture)
        mat_inst.set_metallic_texture(cfg.metallic_texture)
        mat_inst.set_roughness_texture(cfg.roughness_texture)
        mat_inst.set_normal_texture(cfg.normal_texture)
        mat_inst.set_ao_texture(cfg.ao_texture)

        if self.is_rt_enabled:
            mat_inst.set_ior(cfg.ior)
            mat_inst.mat.update_pbr_material_type(
                self.MAT_TYPE_MAPPING[cfg.material_type]
            )

    def create_instance(self, uid: str) -> VisualMaterialInst:
        """Create a new material instance from this material template.

        Note:
            - If the uid already exists, the existing instance will be returned.

        Args:
            uid (str): Unique identifier for the material instance.

        Returns:
            VisualMaterialInst: The created material instance.
        """
        inst = VisualMaterialInst(uid, self._mat)
        # TODO: Support change default properties for material.
        # This will improve the instance creation efficiency.
        self.set_default_properties(inst, self.cfg)
        self._mat_inst_list.append(uid)
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
        if uid not in self._mat_inst_list:
            logger.log_error(f"Material instance with uid '{uid}' does not exist.")
        return VisualMaterialInst(uid, self._mat)


class VisualMaterialInst:
    """Instance of a visual material in the simulation environment."""

    def __init__(self, uid: str, mat: Material):
        self.uid = uid
        self._mat = mat

        # Init properties with default values
        self.base_color = [0.5, 0.5, 0.5, 1.0]
        self.metallic = 0.0
        self.roughness = 0.5
        self.emissive = [0.0, 0.0, 0.0]
        self.emissive_intensity = 1.0
        self.base_color_texture = None
        self.metallic_texture = None
        self.roughness_texture = None
        self.normal_texture = None
        self.ao_texture = None
        self.ior = 1.5
        # self.subsurface = 0.0

    @property
    def mat(self) -> MaterialInst:
        return self._mat.get_inst(self.uid)

    def set_base_color(self, color: list) -> None:
        """Set base color/diffuse color."""
        self.base_color = color
        self.mat.set_base_color(color)

    def set_metallic(self, metallic: float) -> None:
        """Set metallic factor."""
        self.metallic = metallic
        inst = self._mat.get_inst(self.uid)
        inst.set_metallic(metallic)

    def set_roughness(self, roughness: float) -> None:
        """Set surface roughness."""
        self.roughness = roughness
        inst = self._mat.get_inst(self.uid)
        inst.set_roughness(roughness)

    def set_emissive(self, emissive: list) -> None:
        """Set emissive color."""
        self.emissive = emissive
        value = np.zeros(4)
        value[0:3] = emissive
        inst = self._mat.get_inst(self.uid)
        inst.set_emissive(value)

    def set_emissive_intensity(self, intensity: float) -> None:
        """Set emissive intensity multiplier."""
        logger.log_error("Unimplemented: set_emissive_intensity")

    def set_base_color_texture(
        self, texture_path: str = None, texture_data: torch.Tensor | None = None
    ) -> None:
        """Set base color texture from file path or texture data.

        Args:
            texture_path: Path to texture file
            texture_data: Texture data as a torch.Tensor
        """
        if texture_path is not None and texture_data is not None:
            logger.log_warning(
                "Both texture_path and texture_data are provided. Using texture_path."
            )

        if texture_path is not None:
            self.base_color_texture = texture_path
            inst = self._mat.get_inst(self.uid)
            inst.set_base_color_map(texture_path)
        elif texture_data is not None:
            self.base_color_texture = texture_data
            inst = self._mat.get_inst(self.uid)

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
            inst = self._mat.get_inst(self.uid)
            inst.set_metallic_map(texture_path)
        elif texture_data is not None:
            self.metallic_texture = texture_data
            inst = self._mat.get_inst(self.uid)

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
            inst = self._mat.get_inst(self.uid)
            inst.set_roughness_map(texture_path)
        elif texture_data is not None:
            self.roughness_texture = texture_data
            inst = self._mat.get_inst(self.uid)

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
            inst = self._mat.get_inst(self.uid)
            inst.set_normal_map(texture_path)
        elif texture_data is not None:
            self.normal_texture = texture_data
            inst = self._mat.get_inst(self.uid)

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
            inst = self._mat.get_inst(self.uid)
            inst.set_ao_map(texture_path)
        elif texture_data is not None:
            self.ao_texture = texture_data
            inst = self._mat.get_inst(self.uid)

            # TODO: Optimize texture creation method.
            world = dexsim.default_world()
            env = world.get_env()
            ao_texture = env.create_color_texture(
                texture_data.cpu().numpy(), has_alpha=False
            )
            inst.set_ao_map(ao_texture)

    def set_ior(self, ior: float) -> None:
        """Set index of refraction."""
        if is_rt_enabled() is False:
            logger.log_debug("Ray Tracing rendering not enabled, ignoring IOR setting.")
            return
        self.ior = ior
        inst = self._mat.get_inst(self.uid)
        inst.set_rt_param("ior", ior)
