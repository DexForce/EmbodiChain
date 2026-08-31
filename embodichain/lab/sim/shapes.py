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

import math
import warnings
from dataclasses import MISSING
from numbers import Integral
from typing import Any, Dict, List, Literal, TYPE_CHECKING

from embodichain.utils import configclass, is_configclass, logger

if TYPE_CHECKING:
    from embodichain.lab.sim.material import VisualMaterialCfg

__all__ = [
    "MeshCollisionApproximation",
    "MeshCollisionCfg",
    "LoadOption",
    "ShapeCfg",
    "MeshCfg",
    "CubeCfg",
    "SphereCfg",
]


MeshCollisionApproximation = Literal[
    "convex_hull",
    "convex_decomposition",
    "triangle_mesh",
    "sdf",
]
"""Supported collision representations for a triangle mesh."""


@configclass
class MeshCollisionCfg:
    """Collision-geometry construction for :class:`MeshCfg`.

    The approximation is explicit. Strategy-specific fields are rejected when
    they do not apply, so changing a numerical cooking value cannot silently
    select a different collision representation.
    """

    approximation: MeshCollisionApproximation = "convex_hull"
    """Collision representation built from the source triangle mesh."""

    max_hulls: int | None = None
    """Maximum hull count for ``convex_decomposition``; must be at least two."""

    acd_method: Literal["coacd", "vhacd"] | None = None
    """Approximate-convex-decomposition implementation."""

    sdf_resolution: int | None = None
    """Maximum SDF grid resolution; valid only for the ``sdf`` strategy."""

    is_hydroelastic: bool | None = None
    """Whether Newton uses the generated SDF for hydroelastic contact."""

    sdf_narrow_band_range: tuple[float, float] | None = None
    """Inner and outer signed-distance limits of the Newton SDF band [m]."""

    sdf_target_voxel_size: float | None = None
    """Target Newton sparse-SDF voxel size [m], alternative to resolution."""

    sdf_texture_format: Literal["uint16", "float32", "uint8"] | None = None
    """Newton SDF voxel storage format."""

    sdf_padding: float | None = None
    """Extra padding used while Newton builds the mesh SDF [m]."""

    @property
    def max_convex_hull_num(self) -> int:
        """Deprecated compatibility view of :attr:`max_hulls`."""
        return self.max_hulls or 1

    def __post_init__(self) -> None:
        """Validate strategy-specific mesh-cooking fields."""
        supported = {
            "convex_hull",
            "convex_decomposition",
            "triangle_mesh",
            "sdf",
        }
        if self.approximation not in supported:
            raise ValueError(
                "MeshCollisionCfg.approximation must be one of "
                f"{sorted(supported)}, got {self.approximation!r}."
            )

        if self.approximation == "convex_decomposition":
            if (
                not isinstance(self.max_hulls, Integral)
                or isinstance(self.max_hulls, bool)
                or self.max_hulls < 2
            ):
                raise ValueError(
                    "convex_decomposition requires max_hulls to be an integer "
                    "of at least 2."
                )
            if self.acd_method not in (None, "coacd", "vhacd"):
                raise ValueError("acd_method must be 'coacd' or 'vhacd'.")
        elif self.max_hulls is not None or self.acd_method is not None:
            raise ValueError(
                "max_hulls and acd_method are valid only for convex_decomposition."
            )

        sdf_values = {
            "sdf_resolution": self.sdf_resolution,
            "is_hydroelastic": self.is_hydroelastic,
            "sdf_narrow_band_range": self.sdf_narrow_band_range,
            "sdf_target_voxel_size": self.sdf_target_voxel_size,
            "sdf_texture_format": self.sdf_texture_format,
            "sdf_padding": self.sdf_padding,
        }
        configured_sdf_fields = [
            name for name, value in sdf_values.items() if value is not None
        ]
        if self.approximation != "sdf" and configured_sdf_fields:
            raise ValueError(
                f"{configured_sdf_fields} are valid only for the sdf approximation."
            )
        if self.sdf_resolution is not None and (
            not isinstance(self.sdf_resolution, Integral)
            or isinstance(self.sdf_resolution, bool)
            or self.sdf_resolution <= 0
        ):
            raise ValueError("sdf_resolution must be a positive integer.")
        if self.sdf_target_voxel_size is not None and (
            not math.isfinite(self.sdf_target_voxel_size)
            or self.sdf_target_voxel_size <= 0.0
        ):
            raise ValueError("sdf_target_voxel_size must be finite and positive.")
        if self.sdf_resolution is not None and self.sdf_target_voxel_size is not None:
            raise ValueError(
                "Configure only one of sdf_resolution and sdf_target_voxel_size."
            )
        if self.sdf_padding is not None and (
            not math.isfinite(self.sdf_padding) or self.sdf_padding < 0.0
        ):
            raise ValueError("sdf_padding must be finite and non-negative.")
        if self.is_hydroelastic is not None and not isinstance(
            self.is_hydroelastic, bool
        ):
            raise TypeError("is_hydroelastic must be a boolean when configured.")
        if self.sdf_texture_format not in (None, "uint16", "float32", "uint8"):
            raise ValueError(
                "sdf_texture_format must be 'uint16', 'float32', or 'uint8'."
            )
        if self.sdf_narrow_band_range is not None:
            if len(self.sdf_narrow_band_range) != 2:
                raise ValueError("sdf_narrow_band_range must contain two values.")
            inner, outer = (float(value) for value in self.sdf_narrow_band_range)
            if not math.isfinite(inner) or not math.isfinite(outer):
                raise ValueError("sdf_narrow_band_range values must be finite.")
            if inner > outer:
                raise ValueError(
                    "sdf_narrow_band_range inner value cannot exceed the outer value."
                )

    @classmethod
    def from_dict(cls, init_dict: Dict[str, Any]) -> MeshCollisionCfg:
        """Parse a mesh-collision mapping, including deprecated field names."""
        data = dict(init_dict)
        legacy_fields = {
            "max_convex_hull_num",
            "force_sdf",
            "sdf_max_resolution",
        }
        has_legacy_fields = bool(legacy_fields.intersection(data))
        if has_legacy_fields:
            warnings.warn(
                "Legacy mesh collision fields are deprecated; use an explicit "
                "approximation with max_hulls or sdf_resolution.",
                DeprecationWarning,
                stacklevel=2,
            )

        legacy_max_hulls = data.pop("max_convex_hull_num", None)
        legacy_force_sdf = data.pop("force_sdf", None)
        legacy_sdf_resolution = data.pop("sdf_max_resolution", None)
        if legacy_sdf_resolution is not None:
            if "sdf_resolution" in data:
                raise ValueError(
                    "sdf_max_resolution and sdf_resolution cannot both be configured."
                )
            data["sdf_resolution"] = legacy_sdf_resolution

        if "approximation" not in data and has_legacy_fields:
            sdf_requested = bool(legacy_force_sdf) or (
                data.get("sdf_resolution") is not None
                and int(data["sdf_resolution"]) > 0
            )
            if sdf_requested:
                data["approximation"] = "sdf"
                data.pop("max_hulls", None)
                data.pop("acd_method", None)
            elif legacy_max_hulls is not None and int(legacy_max_hulls) > 1:
                data["approximation"] = "convex_decomposition"
                data["max_hulls"] = int(legacy_max_hulls)
            else:
                data["approximation"] = "convex_hull"
                data.pop("acd_method", None)
        elif legacy_max_hulls is not None:
            if "max_hulls" in data:
                raise ValueError(
                    "max_convex_hull_num and max_hulls cannot both be configured."
                )
            data["max_hulls"] = int(legacy_max_hulls)

        if data.get("sdf_resolution") == 0:
            data.pop("sdf_resolution")
        return cls(**data)


_mesh_collision_cfg_init = MeshCollisionCfg.__init__


def _mesh_collision_cfg_init_with_legacy_max_hulls(
    self: MeshCollisionCfg,
    approximation: MeshCollisionApproximation | None = None,
    max_hulls: int | None = None,
    acd_method: Literal["coacd", "vhacd"] | None = None,
    sdf_resolution: int | None = None,
    is_hydroelastic: bool | None = None,
    sdf_narrow_band_range: tuple[float, float] | None = None,
    sdf_target_voxel_size: float | None = None,
    sdf_texture_format: Literal["uint16", "float32", "uint8"] | None = None,
    sdf_padding: float | None = None,
    *,
    max_convex_hull_num: int | None = None,
) -> None:
    """Initialize with the deprecated hull-count spelling at the API boundary."""
    if max_convex_hull_num is not None:
        warnings.warn(
            "max_convex_hull_num is deprecated; use max_hulls with an explicit "
            "approximation.",
            DeprecationWarning,
            stacklevel=2,
        )
        if max_hulls is not None:
            raise ValueError(
                "max_convex_hull_num and max_hulls cannot both be configured."
            )
        if (
            not isinstance(max_convex_hull_num, Integral)
            or isinstance(max_convex_hull_num, bool)
            or max_convex_hull_num < 1
        ):
            raise ValueError("max_convex_hull_num must be a positive integer.")
        if approximation is None:
            approximation = (
                "convex_decomposition" if max_convex_hull_num > 1 else "convex_hull"
            )
        max_hulls = (
            None
            if approximation == "convex_hull" and max_convex_hull_num == 1
            else max_convex_hull_num
        )

    _mesh_collision_cfg_init(
        self,
        approximation="convex_hull" if approximation is None else approximation,
        max_hulls=max_hulls,
        acd_method=acd_method,
        sdf_resolution=sdf_resolution,
        is_hydroelastic=is_hydroelastic,
        sdf_narrow_band_range=sdf_narrow_band_range,
        sdf_target_voxel_size=sdf_target_voxel_size,
        sdf_texture_format=sdf_texture_format,
        sdf_padding=sdf_padding,
    )


MeshCollisionCfg.__init__ = _mesh_collision_cfg_init_with_legacy_max_hulls


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

    visual_material: VisualMaterialCfg | None = None
    """Configuration parameters for the visual material of the shape. Defaults to None."""

    @classmethod
    def from_dict(cls, init_dict: Dict[str, Any]) -> ShapeCfg:
        """Initialize the configuration from a dictionary."""
        from embodichain.utils.utility import get_class_instance

        data = dict(init_dict)
        if "shape_type" not in data:
            logger.log_error("shape type must be specified in the configuration.")

        cfg = get_class_instance(
            "embodichain.lab.sim.shapes", data["shape_type"] + "Cfg"
        )()
        legacy_mesh_fields = {
            "max_convex_hull_num",
            "acd_method",
            "sdf_resolution",
        }
        if isinstance(cfg, MeshCfg):
            configured_legacy = legacy_mesh_fields.intersection(data)
            if configured_legacy:
                if data.get("collision") is not None:
                    raise ValueError(
                        "MeshCfg collision cannot be combined with deprecated flat "
                        f"mesh fields {sorted(configured_legacy)}."
                    )
                legacy_collision = {
                    key: data.pop(key) for key in tuple(configured_legacy)
                }
                # Route through the legacy normalizer. Presence of this old hull
                # name also makes the deprecation warning deterministic.
                legacy_collision.setdefault("max_convex_hull_num", 1)
                data["collision"] = legacy_collision

        for key, value in data.items():
            if hasattr(cfg, key):
                attr = getattr(cfg, key)
                if key == "visual_material" and isinstance(value, dict):
                    from embodichain.lab.sim.material import VisualMaterialCfg

                    setattr(
                        cfg,
                        key,
                        VisualMaterialCfg.from_dict(value),
                    )
                elif key == "collision" and isinstance(cfg, MeshCfg):
                    if value is not None and not isinstance(value, MeshCollisionCfg):
                        if not isinstance(value, dict):
                            raise TypeError(
                                "MeshCfg.collision must be a mapping, "
                                "MeshCollisionCfg, or None."
                            )
                        value = MeshCollisionCfg.from_dict(value)
                    setattr(cfg, key, value)
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

    collision: MeshCollisionCfg | None = None
    """Optional collision representation and cooking parameters.

    ``None`` uses a single convex hull. Mesh collision construction belongs to
    the geometry because it cannot be applied meaningfully to primitive shapes
    or to articulation links without a named source-shape overlay.
    """


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
