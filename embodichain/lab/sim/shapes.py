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

from typing import Optional, List, Dict, Union, TYPE_CHECKING, Any
from dataclasses import MISSING
from embodichain.utils import configclass, is_configclass, logger

if TYPE_CHECKING:
    from embodichain.lab.sim.material import VisualMaterialCfg


@configclass
class LoadOption:

    rebuild_normals: bool = False
    """Whether to rebuild normals for the shape. Defaults to False."""

    rebuild_tangent: bool = False
    """Whether to rebuild tangents for the shape. Defaults to False."""

    rebuild_3rdnormal: bool = False
    """Whether to rebuild the normal for the shape using 3rd party library. Defaults to False."""

    rebuild_3rdtangent: bool = False
    """Whether to rebuild the tangent for the shape using 3rd party library. Defaults to False."""

    smooth: float = -1.0
    """Angle threshold (in degrees) for smoothing normals. Defaults to -1.0 (no smoothing)."""

    @classmethod
    def from_dict(cls, init_dict: Dict[str, Any]) -> LoadOption:
        """Initialize the configuration from a dictionary."""
        cfg = cls()
        for key, value in init_dict.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
            else:
                logger.log_warning(
                    f"Key '{key}' not found in {cfg.__class__.__name__}."
                )
        return cfg


@configclass
class ShapeCfg:

    shape_type: str = MISSING
    """Type of the shape. Must be specified in subclasses."""

    visual_material: Optional[VisualMaterialCfg] = None
    """Configuration parameters for the visual material of the shape. Defaults to None."""

    @classmethod
    def from_dict(cls, init_dict: Dict[str, Any]) -> ShapeCfg:
        """Initialize the configuration from a dictionary."""
        from embodichain.utils.utility import get_class_instance

        if "shape_type" not in init_dict:
            logger.log_error("shape type must be specified in the configuration.")

        cfg = get_class_instance(
            "embodichain.lab.sim.shapes", init_dict["shape_type"] + "Cfg"
        )()
        for key, value in init_dict.items():
            if hasattr(cfg, key):
                attr = getattr(cfg, key)
                if key == "visual_material" and isinstance(value, dict):
                    setattr(
                        cfg,
                        key,
                        VisualMaterialCfg.from_dict(value),
                    )
                elif is_configclass(attr):
                    setattr(cfg, key, attr.from_dict(value))
                else:
                    setattr(cfg, key, value)
            else:
                logger.log_warning(
                    f"Key '{key}' not found in {cfg.__class__.__name__}."
                )
        return cfg


@configclass
class MeshCfg(ShapeCfg):
    """Configuration parameters for a triangle mesh shape."""

    shape_type: str = "Mesh"

    fpath: str = MISSING
    """File path to the shape mesh file."""

    load_option: LoadOption = LoadOption()
    """Options for loading and processing the shape."""

    compute_uv: bool = False
    """Whether to compute UV coordinates for the shape. Defaults to False.
    
    If the shape already has UV coordinates, setting this to True will recompute and overwrite them.
    """

    project_direction: List[float] = [1.0, 1.0, 1.0]
    """Direction to project the UV coordinates. Defaults to [1.0, 1.0, 1.0]."""


@configclass
class CubeCfg(ShapeCfg):
    """Configuration parameters for a cube shape."""

    shape_type: str = "Cube"

    size: List[float] = [1.0, 1.0, 1.0]
    """Size of the cube (in m) as [length, width, height]."""


@configclass
class SphereCfg(ShapeCfg):
    """Configuration parameters for a sphere shape."""

    shape_type: str = "Sphere"

    radius: float = 1.0
    """Radius of the sphere (in m)."""

    resolution: int = 20
    """Resolution of the sphere mesh. Defaults to 20."""
