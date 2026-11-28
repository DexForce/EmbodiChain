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

"""Deformable-body physical and object configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import MISSING, field
from typing import Any, ClassVar, Literal, Sequence

import numpy as np

from embodichain.utils import configclass
from embodichain.utils.configclass import class_to_dict

from ..shapes import MeshCfg
from .asset import ObjectBaseCfg
from .source import UsdSourceCfg, _usd_source_from_value

__all__: list[str] = []


@configclass
class VolumeDeformableMeshingCfg:
    """Newton tetrahedralization and render-volume binding parameters."""

    triangle_remesh_resolution: int = 8
    """Resolution used to remesh the source surface before tetrahedralization."""

    triangle_simplify_target: int = 0
    """Target face count for the proxy surface; zero disables simplification."""

    simulation_mesh_resolution: int = 8
    """Voxel resolution used to build the tetrahedral simulation mesh."""

    voxel_num_relaxation_iters: int = 5
    """Number of tetrahedral-mesh relaxation iterations."""

    voxel_rel_min_tet_volume: float = 0.05
    """Minimum tetrahedron volume relative to the voxel volume."""

    voxel_surface_dist_ratio: float = 0.2
    """Maximum surface distance expressed as a voxel-size ratio."""

    embedding_impl: str = "dexsim_exact_cpu"
    """DexSim implementation used to bind render vertices to tetrahedra."""


@configclass
class SurfaceElementPropertiesCfg:
    """Newton surface triangle, bending, and aerodynamic properties.

    ``None`` preserves Newton's cloth defaults; volume objects resolve it to
    zero, matching their disabled-by-default surface forces.
    """

    tri_ke: float | None = None
    """Triangle elastic stiffness."""

    tri_ka: float | None = None
    """Triangle area stiffness."""

    tri_kd: float | None = None
    """Triangle damping."""

    tri_drag: float | None = None
    """Aerodynamic drag coefficient."""

    tri_lift: float | None = None
    """Aerodynamic lift coefficient."""

    edge_ke: float | None = None
    """Bending-edge stiffness."""

    edge_kd: float | None = None
    """Bending-edge damping."""


_SURFACE_FIELDS = (
    "tri_ke",
    "tri_ka",
    "tri_kd",
    "tri_drag",
    "tri_lift",
    "edge_ke",
    "edge_kd",
)


@configclass
class VolumeDeformablePhysicsCfg:
    """Volume density, elasticity, and optional Newton surface constraints."""

    youngs: float = 1e6
    """Young's modulus [Pa]; higher values make the volume stiffer."""

    poissons: float = 0.45
    """Poisson's ratio; values approaching 0.5 resist volume change."""

    elasticity_damping: float = 0.0
    """Volumetric damping coefficient forwarded as Newton ``k_damp``."""

    density: float = 1000.0
    """Volume density [kg/m³]."""

    surface_props: SurfaceElementPropertiesCfg = SurfaceElementPropertiesCfg(
        **dict.fromkeys(_SURFACE_FIELDS, 0.0)
    )
    """Optional surface forces; all coefficients default to zero."""

    add_surface_edges: bool = True
    """Whether Newton creates surface bending-edge constraints."""

    contact_material: Mapping[str, float] | None = None
    """Optional DexUni object contact material, separate from element elasticity.

    Keys are ``stiffness`` [N/m], ``damping`` [N s/m], and
    ``dynamic_friction``. Omitted keys use the native material defaults.
    Values must be finite, nonnegative float32 values. ``None`` preserves the
    backend defaults; explicit materials require a supporting DexSim build.
    """

    def __post_init__(self) -> None:
        if isinstance(self.surface_props, Mapping):
            self.surface_props = SurfaceElementPropertiesCfg(**self.surface_props)


@configclass
class SurfaceDeformablePhysicsCfg:
    """Surface density, Newton surface elements, and optional mesh springs."""

    density: float = 1.0
    """Surface density [kg/m²]."""

    surface_props: SurfaceElementPropertiesCfg = SurfaceElementPropertiesCfg()
    """Triangle, bending, and aerodynamic overrides; None uses Newton defaults."""

    add_springs: bool = False
    """Whether Newton creates explicit mesh-edge springs."""

    spring_ke: float | None = None
    """Spring stiffness; ``None`` uses the Newton default."""

    spring_kd: float | None = None
    """Spring damping; ``None`` uses the Newton default."""

    contact_material: Mapping[str, float] | None = None
    """Optional DexUni object contact material, separate from element elasticity.

    Keys are ``stiffness`` [N/m], ``damping`` [N s/m], and
    ``dynamic_friction``. Omitted keys use the native material defaults.
    Values must be finite, nonnegative float32 values. ``None`` preserves the
    backend defaults; explicit materials require a supporting DexSim build.
    """

    edge_rest_angle: float | None = None
    """Constant signed rest dihedral angle [rad] for every bending edge.

    ``None`` preserves angles inferred from the authored mesh. Set ``0.0``
    for a garment that should relax toward a flat surface. This changes the
    rest geometry independently of ``surface_props.edge_ke`` and requires
    a DexSim build supporting the cloth ``edge_rest_angle`` parameter.
    """

    pneumatic: Mapping[str, Any] | None = None
    """Optional DexUni closed-cavity pressure parameters for this surface.

    Keys follow DexSim ``PneumaticPhysicsDesc``: ``mode`` (``isothermal``,
    ``adiabatic``, ``target_volume`` or ``prescribed_gauge_pressure``),
    ``reference_absolute_pressure`` and ``ambient_pressure`` [Pa],
    ``heat_capacity_ratio``, ``target_volume`` [m³], ``volume_stiffness``,
    ``bulk_damping``, ``min_volume_ratio``, ``max_absolute_pressure`` [Pa],
    and ``prescribed_gauge_pressure`` [Pa]. Native validation owns their
    defaults and pressure-law constraints. The authored surface must be
    closed; native model construction validates cavity topology and solver
    compatibility. ``None`` preserves ordinary cloth and supports older
    wheels; an explicit block requires a supporting DexSim build.
    """

    def __post_init__(self) -> None:
        if isinstance(self.surface_props, Mapping):
            self.surface_props = SurfaceElementPropertiesCfg(**self.surface_props)


def _mesh_cfg_from_dict(data: Mapping[str, Any]) -> MeshCfg:
    cfg = MeshCfg.from_dict({"shape_type": "Mesh", **data})
    if not isinstance(cfg, MeshCfg):
        raise TypeError("Deformable shape must be a MeshCfg.")
    return cfg


def _normalize_tetrahedra(values: Sequence[Sequence[int]] | np.ndarray) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype.kind not in "iu":
        raise TypeError("tetrahedra must contain integer indices.")
    if raw.ndim != 2 or raw.shape[1:] != (4,) or len(raw) == 0:
        raise ValueError("tetrahedra must have non-empty shape (M, 4).")
    if np.any(raw < 0) or np.any(raw > np.iinfo(np.int32).max):
        raise ValueError("tetrahedra indices must be nonnegative and fit int32.")
    return raw.astype(np.int32)


@configclass
class TetrahedralModelCfg:
    """Tetrahedral simulation geometry and elasticity for a deformable volume."""

    kind: Literal["tetrahedral"] = "tetrahedral"

    shape: MeshCfg = MeshCfg()
    """Source mesh or explicit simulation vertices and optional boundary faces."""

    tetrahedra: Sequence[Sequence[int]] | np.ndarray | None = None
    """Explicit ``(M, 4)`` indices into ``shape.vertices``. Omit to voxelize ``shape``."""

    meshing: VolumeDeformableMeshingCfg = VolumeDeformableMeshingCfg()
    """Meshing settings applied when ``tetrahedra`` is omitted."""

    attrs: VolumeDeformablePhysicsCfg = VolumeDeformablePhysicsCfg()
    """Volumetric material properties and optional surface forces."""

    def __post_init__(self) -> None:
        if self.kind != "tetrahedral":
            raise ValueError(f"Unknown tetrahedral model kind {self.kind!r}.")
        if isinstance(self.shape, Mapping):
            self.shape = _mesh_cfg_from_dict(self.shape)
        if isinstance(self.meshing, Mapping):
            self.meshing = VolumeDeformableMeshingCfg(**self.meshing)
        if isinstance(self.attrs, Mapping):
            self.attrs = VolumeDeformablePhysicsCfg(**self.attrs)
        if self.tetrahedra is not None:
            self.tetrahedra = _normalize_tetrahedra(self.tetrahedra)


@configclass
class TriangleSurfaceModelCfg:
    """Simulation triangle mesh, surface physics and optional render embedding."""

    kind: Literal["triangle_surface"] = "triangle_surface"

    shape: MeshCfg = MeshCfg()
    """Triangle mesh whose vertices and faces participate in simulation."""

    visual_shape: MeshCfg | None = None
    """Separately indexed render mesh. Omission renders the simulation mesh."""

    visual_binding_mode: Literal["auto", "nearest_vertex"] = "auto"
    """Use native surface embedding or bind each render vertex to a particle."""

    attrs: SurfaceDeformablePhysicsCfg = SurfaceDeformablePhysicsCfg()
    """Triangle, bending and contact properties."""

    def __post_init__(self) -> None:
        if self.kind != "triangle_surface":
            raise ValueError(f"Unknown triangle surface model kind {self.kind!r}.")
        if isinstance(self.shape, Mapping):
            self.shape = _mesh_cfg_from_dict(self.shape)
        if isinstance(self.visual_shape, Mapping):
            self.visual_shape = _mesh_cfg_from_dict(self.visual_shape)
        if isinstance(self.attrs, Mapping):
            self.attrs = SurfaceDeformablePhysicsCfg(**self.attrs)


@configclass
class DeformableObjectCfg(ObjectBaseCfg):
    """Common configuration contract for one deformable asset.

    The object family determines its public identity. Its model determines how
    physics represents the geometry.
    """

    _legacy_model_fields: ClassVar[tuple[str, ...]] = ("shape",)

    deformable_type: Literal["volume", "surface"] = MISSING
    """Deformable object family."""

    shape: MeshCfg | None = None
    """Legacy flat mesh configuration. New configurations put it in ``model``."""

    source: UsdSourceCfg | None = field(default=None, kw_only=True)
    """Optional authored USD particle-set source; source physics is retained."""

    particle_radius: float | None = None
    """Newton particle radius; ``None`` uses the active solver default."""

    particle_flags: int | Sequence[int] | np.ndarray | None = None
    """Newton particle flags, provided as one broadcast value or one value per node.

    Clear the Newton ``ACTIVE`` bit for nodes that will be driven kinematically.
    Per-node arrays must follow the resolved simulation-particle order. For a
    surface deformable, an array-backed
    :class:`~embodichain.lab.sim.shapes.MeshCfg` preserves this order. A volume
    deformable is voxelized into a separate tetrahedral simulation mesh, so its
    particle indices do not correspond to source-mesh vertex indices.
    """

    validate_mesh: bool = False
    """Whether Newton reports source-mesh quality validation warnings."""

    def __post_init__(self) -> None:
        self.source = _usd_source_from_value(self.source)
        if isinstance(self.shape, Mapping):
            self.shape = _mesh_cfg_from_dict(self.shape)
        if getattr(self, "model", None) is None and self.shape is None:
            self.shape = MeshCfg()

    def _require_exclusive_model(self) -> None:
        if self.source is not None or any(
            getattr(self, name) is not None for name in self._legacy_model_fields
        ):
            raise ValueError(
                "Explicit deformable models own sampling, geometry and physics."
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the authored model or the legacy flat representation."""
        explicit_model = getattr(self, "model", None) is not None
        if explicit_model:
            self._require_exclusive_model()
        result = class_to_dict(self)
        if explicit_model:
            for name in (*self._legacy_model_fields, "source"):
                result.pop(name, None)
        else:
            result.pop("model", None)
        return result

    @classmethod
    def from_dict(cls, init_dict: Mapping[str, Any]) -> DeformableObjectCfg:
        """Parse nested deformable configs using the current schema.

        Args:
            init_dict: Configuration fields from a dictionary or YAML loader.

        Returns:
            A typed configuration with the base asset pose conventions applied.
        """
        if init_dict.get("model") is not None and any(
            name in init_dict for name in cls._legacy_model_fields
        ):
            raise ValueError(
                "Explicit deformable models own sampling, geometry and physics."
            )
        cfg = cls(**dict(init_dict))
        pose = ObjectBaseCfg.from_dict(
            {
                "init_pos": cfg.init_pos,
                "init_rot": cfg.init_rot,
                "init_local_pose": cfg.init_local_pose,
            }
        )
        cfg.init_pos, cfg.init_rot = pose.init_pos, pose.init_rot
        cfg.init_local_pose = pose.init_local_pose
        if cfg.source is not None and (
            ("init_local_pose" in init_dict and init_dict["init_local_pose"] is None)
            or not {"init_pos", "init_rot", "init_local_pose"}.intersection(init_dict)
        ):
            cfg.init_local_pose = None
        return cfg


@configclass
class MaterialPointModelCfg:
    """Initial material points and constitutive properties for a volume."""

    sampling: Mapping[str, Any] = MISSING
    physics: Mapping[str, Any] = MISSING
    kind: Literal["material_point"] = "material_point"

    def __post_init__(self) -> None:
        if self.kind != "material_point":
            raise ValueError(f"Unknown material point model {self.kind!r}")
        if (
            not isinstance(self.physics, Mapping)
            or "particle_masses" not in self.physics
        ):
            raise ValueError("Material-point physics requires particle_masses.")
        if not isinstance(self.sampling, Mapping):
            raise ValueError("Material-point sampling requires a mapping.")


@configclass
class VolumeDeformableObjectCfg(DeformableObjectCfg):
    """Volume deformable backed by a tetrahedral or material-point model."""

    _legacy_model_fields: ClassVar[tuple[str, ...]] = (
        "shape",
        "tetrahedra",
        "meshing",
        "attrs",
    )

    deformable_type: Literal["volume"] = "volume"

    model: TetrahedralModelCfg | MaterialPointModelCfg | None = field(
        default=None, kw_only=True
    )
    """Physical representation. Omission accepts legacy flat tetrahedral fields."""

    meshing: VolumeDeformableMeshingCfg | None = None
    """Legacy flat meshing settings for a tetrahedral model."""

    tetrahedra: Sequence[Sequence[int]] | np.ndarray | None = None
    """Explicit ``(M, 4)`` tetrahedra indexing ``shape.vertices``.

    When set, :attr:`shape` must be array-backed: its vertices become the
    simulation particles in the same order, so particle indices equal vertex
    indices, and :attr:`meshing` is not used. ``shape.triangles`` may list the
    boundary surface over those vertex indices; leave it ``None`` to derive
    the surface from the tetrahedra. ``None`` voxelizes :attr:`shape` instead.
    """

    attrs: VolumeDeformablePhysicsCfg | None = None
    """Legacy flat tetrahedral physical attributes."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if isinstance(self.model, Mapping):
            kind = self.model.get(
                "kind", "material_point" if "sampling" in self.model else "tetrahedral"
            )
            if kind == "material_point":
                self.model = MaterialPointModelCfg(**self.model)
            elif kind == "tetrahedral":
                self.model = TetrahedralModelCfg(**self.model)
            else:
                raise ValueError(f"Unknown volume model kind {kind!r}.")
        if self.model is not None:
            if not isinstance(self.model, (TetrahedralModelCfg, MaterialPointModelCfg)):
                raise TypeError("A volume model must be tetrahedral or material-point.")
            self._require_exclusive_model()
            return
        if self.attrs is None:
            self.attrs = VolumeDeformablePhysicsCfg()
        elif isinstance(self.attrs, Mapping):
            self.attrs = VolumeDeformablePhysicsCfg(**self.attrs)
        if self.meshing is None:
            self.meshing = VolumeDeformableMeshingCfg()
        elif isinstance(self.meshing, Mapping):
            self.meshing = VolumeDeformableMeshingCfg(**self.meshing)
        if self.tetrahedra is not None:
            self.tetrahedra = _normalize_tetrahedra(self.tetrahedra)


@configclass
class SurfaceDeformableObjectCfg(DeformableObjectCfg):
    """Surface deformable backed by a triangle-surface model."""

    _legacy_model_fields: ClassVar[tuple[str, ...]] = (
        "shape",
        "visual_shape",
        "visual_binding_mode",
        "attrs",
    )

    deformable_type: Literal["surface"] = "surface"

    model: TriangleSurfaceModelCfg | None = field(default=None, kw_only=True)
    """Physical representation. Omission accepts legacy flat surface fields."""

    visual_shape: MeshCfg | None = None
    """Legacy flat render-mesh override."""

    visual_binding_mode: Literal["auto", "nearest_vertex"] | None = None
    """Legacy flat binding override. Omission uses native surface embedding."""

    attrs: SurfaceDeformablePhysicsCfg | None = None
    """Legacy flat triangle-surface physical attributes."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if isinstance(self.model, Mapping):
            self.model = TriangleSurfaceModelCfg(**self.model)
        if self.model is not None:
            if not isinstance(self.model, TriangleSurfaceModelCfg):
                raise TypeError("A surface model must be triangle-surface.")
            self._require_exclusive_model()
            return
        if isinstance(self.visual_shape, Mapping):
            self.visual_shape = _mesh_cfg_from_dict(self.visual_shape)
        if self.visual_binding_mode is None:
            self.visual_binding_mode = "auto"
        if self.attrs is None:
            self.attrs = SurfaceDeformablePhysicsCfg()
        elif isinstance(self.attrs, Mapping):
            self.attrs = SurfaceDeformablePhysicsCfg(**self.attrs)
