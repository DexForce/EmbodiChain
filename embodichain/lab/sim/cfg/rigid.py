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

"""Rigid-body mass, collision, material, and backend property schemas."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from typing import Any, Sequence

import numpy as np
from dexsim.types import PhysicalAttr

from embodichain.utils import configclass

from .._legacy_cfg import RigidBodyAttributesCfg, RigidBodyAttributesOverrideCfg


@configclass
class MassPropertiesCfg:
    """Backend-neutral rigid-body mass properties.

    ``None`` means that the source asset or selected backend keeps ownership of
    that value.  For a non-static body, explicit inertia requires a positive
    mass; otherwise a positive mass rescales geometry-derived inertia, while
    density derives mass, center of mass, and inertia from collision geometry.
    Static bodies omit all mass properties during Spawn compilation.
    """

    mass: float | None = None
    """Rigid-body mass [kg].

    A positive value takes precedence over :attr:`density`.  Zero explicitly
    selects density-based derivation and therefore requires a positive density.
    Negative values are invalid.
    """

    density: float | None = None
    """Uniform density used to derive mass properties from collision shapes [kg/m^3].

    The value must be positive and is ignored when :attr:`mass` is positive.
    """

    inertia: Sequence[float] | np.ndarray | None = None
    """Inertia about the center of mass [kg*m^2].

    Supply either three positive principal moments or a symmetric,
    positive-definite 3-by-3 tensor in the body frame.  Explicit inertia is
    accepted only together with a positive :attr:`mass`.  For one definition
    shared by both backends, prefer principal moments plus
    :attr:`com_quaternion`; the current Default adapter consumes the principal-
    moment representation, while Newton can retain a full tensor.
    """

    com_position: Sequence[float] | np.ndarray | None = None
    """Center-of-mass position expressed in the rigid body's local frame [m]."""

    com_quaternion: Sequence[float] | np.ndarray | None = None
    """Orientation of the center-of-mass/inertia frame in ``xyzw`` order.

    Spawn normalizes the quaternion and converts it to the backend descriptor's
    ``wxyz`` convention.  A zero quaternion is invalid.
    """


@configclass
class RigidBodyPropertiesCfg:
    """Common root for backend-specific rigid-body properties.

    Actor type and mass properties already live in backend-neutral descriptors,
    and no additional body-level field currently has identical semantics in
    both backends.  The root is therefore intentionally empty and serves as the
    typed extension/serialization boundary.
    """


@configclass
class DefaultRigidBodyPropertiesCfg(RigidBodyPropertiesCfg):
    """Rigid-body properties consumed only by the Default backend.

    Every field defaults to ``None`` so a partial overlay preserves an authored
    USD/URDF value or the backend default.
    """

    linear_damping: float | None = None
    """Non-negative damping coefficient applied to linear velocity."""

    angular_damping: float | None = None
    """Non-negative damping coefficient applied to angular velocity."""

    has_gravity: bool | None = None
    """Whether world gravity accelerates this body."""

    max_linear_velocity: float | None = None
    """Maximum rigid-body linear speed [m/s]."""

    max_angular_velocity: float | None = None
    """Maximum rigid-body angular speed [rad/s]."""

    max_depenetration_velocity: float | None = None
    """Maximum separation speed introduced to resolve penetration [m/s]."""

    retain_acceleration: bool | None = None
    """Whether accumulated acceleration is retained across simulation steps."""

    enable_ccd: bool | None = None
    """Whether continuous collision detection is enabled for this body.

    Scene-level CCD must also be enabled through :attr:`PhysicsCfg.enable_ccd`.
    """

    min_position_iters: int | None = None
    """Minimum number of position-solver iterations for this body (1 to 255)."""

    min_velocity_iters: int | None = None
    """Minimum number of velocity-solver iterations for this body (0 to 255)."""

    sleep_threshold: float | None = None
    """Mass-normalized kinetic-energy threshold below which the body may sleep."""


@configclass
class NewtonRigidBodyPropertiesCfg(RigidBodyPropertiesCfg):
    """Newton rigid-body extension point.

    Newton currently consumes common mass properties and per-shape settings,
    but DexSim Spawn exposes no additional Newton-native body-level field.  The
    class remains as a stable extension and serialization point.
    """


@configclass
class CollisionPropertiesCfg:
    """Collision-shape properties with identical intent across both backends.

    ``None`` leaves the corresponding source/backend value unchanged.  The
    contact envelope is expressed once with Default-backend terminology and is
    compiled to Newton's ``margin``/``gap`` representation at the Spawn
    boundary. Backend-native filtering lives in the Newton extension, while
    mesh SDF settings use :class:`NewtonMeshCollisionPropertiesCfg`.
    """

    collision_enabled: bool | None = None
    """Whether the shape participates in rigid shape-shape collision.

    On Newton this maps to ``ShapeConfig.has_shape_collision``;
    :attr:`NewtonCollisionPropertiesCfg.has_particle_collision` remains an
    independent flag.  ``None`` preserves the source/backend value.
    """

    contact_offset: float | None = None
    """Per-shape distance at which contact generation starts [m].

    The pair threshold is the sum of both shapes' contact offsets.  This value
    must be non-negative and no smaller than :attr:`rest_offset`.  Default
    consumes it directly; Newton compiles it together with :attr:`rest_offset`
    to ``gap = contact_offset - rest_offset``.
    """

    rest_offset: float | None = None
    """Per-shape target separation at rest [m].

    Pairwise rest separation is the sum of both shapes' values.  Positive
    values leave an air gap, zero targets touching surfaces, and negative
    values permit limited penetration.  Default consumes it directly; Newton
    maps it to ``margin``.
    """


@configclass
class DefaultCollisionPropertiesCfg(CollisionPropertiesCfg):
    """Default-native collision-property extension point.

    ``contact_offset`` and ``rest_offset`` now live on
    :class:`CollisionPropertiesCfg` because both backends consume their intent.
    """


@configclass
class NewtonCollisionPropertiesCfg(CollisionPropertiesCfg):
    """Newton-native shape geometry, filtering, and visibility properties.

    Fields map by name to ``newton.ModelBuilder.ShapeConfig`` through DexSim
    Spawn.  They are shape-level settings; scene-wide pair generation belongs
    to :class:`NewtonCollisionPipelineCfg`, and contact coefficients belong to
    :class:`NewtonRigidBodyMaterialCfg`.

    The SDF/hydroelastic fields remain here as compatibility aliases. New
    configurations should use :class:`NewtonMeshCollisionPropertiesCfg` in
    ``newton_props.mesh_collision_props``; that explicit block takes
    precedence when both forms are present.

    See `Newton Shape Configuration
    <https://newton-physics.github.io/newton/latest/concepts/collisions.html#shape-configuration>`_.
    """

    margin: float | None = None
    """Outward collision-surface offset [m].

    Margins from both shapes are added.  They determine where contact is placed
    and also affect inertia/SDF handling for hollow shapes.
    """

    gap: float | None = None
    """Additional contact-detection distance outside :attr:`margin` [m].

    Gaps from both shapes are added.  Broad phase expands each shape by
    ``margin + gap``; increasing the gap detects approaching contact earlier.
    """

    is_solid: bool | None = None
    """Whether the shape represents a solid volume rather than a hollow shell."""

    collision_group: int | None = None
    """Newton collision-group identifier.

    Group ``0`` disables collisions.  Equal positive groups collide; a negative
    group collides with positive and different negative groups.  Spawn may
    replace this value when replicated arenas use isolated collision groups.
    """

    collision_filter_parent: bool | None = None
    """Whether to filter collision with the adjacent parent body of a joint."""

    has_particle_collision: bool | None = None
    """Whether this shape collides with Newton particles/soft bodies."""

    is_visible: bool | None = None
    """Whether Newton exposes the shape to its render/sensor visibility path.

    This flag does not enable or disable physical collision.
    """

    is_site: bool | None = None
    """Whether Newton treats the shape as a reference site.

    This is an expert pass-through.  Setting it does not automatically reconcile
    ``collision_enabled``, particle collision, density, or collision group in
    EmbodiChain; those values must be configured consistently.
    """

    is_hydroelastic: bool | None = None
    """Whether the shape opts into SDF-based hydroelastic contact.

    Both shapes in a pair must opt in and have SDF data.  Plane, heightfield,
    and other non-volumetric shapes cannot use hydroelastic contact.
    """

    sdf_narrow_band_range: tuple[float, float] | None = None
    """Inner and outer signed-distance limits of the generated SDF band [m]."""

    sdf_target_voxel_size: float | None = None
    """Target sparse-SDF voxel size [m].

    This enables SDF generation, requires CUDA, and takes precedence over
    :attr:`sdf_max_resolution`; configure only one resolution policy.
    """

    sdf_max_resolution: int | None = None
    """Maximum sparse-SDF grid dimension.

    The value must be divisible by eight, requires CUDA, and is used only when
    :attr:`sdf_target_voxel_size` is ``None``.
    """

    sdf_texture_format: str | None = None
    """SDF voxel storage format: ``"uint16"``, ``"float32"``, or ``"uint8"``."""

    force_sdf: bool | None = None
    """Whether to build an SDF at Newton's default resolution when none is set."""

    sdf_padding: float | None = None
    """Extra construction padding used while building a mesh SDF [m].

    Hydroelastic SDF coverage must include at least the configured contact
    envelope.  When omitted, the DexSim adapter chooses its fallback padding.

    This field is a compatibility alias. New configurations should place it in
    :class:`NewtonMeshCollisionPropertiesCfg`.
    """


@configclass
class MeshCollisionPropertiesCfg:
    """Backend-neutral mesh collision approximation and cooking settings.

    These values describe collision geometry, not render geometry. ``None``
    falls back to the deprecated fields on :class:`~embodichain.lab.sim.shapes.MeshCfg`.
    """

    max_convex_hull_num: int | None = None
    """Maximum number of convex hulls produced for convex decomposition."""

    acd_method: str | None = None
    """Approximate-convex-decomposition method, currently ``coacd`` or ``vhacd``."""

    sdf_resolution: int | None = None
    """Uniform SDF cooking resolution; zero disables SDF approximation."""


@configclass
class NewtonMeshCollisionPropertiesCfg:
    """Newton-native mesh SDF and hydroelastic collision properties."""

    is_hydroelastic: bool | None = None
    """Whether the mesh opts into SDF-based hydroelastic contact."""

    sdf_narrow_band_range: tuple[float, float] | None = None
    """Inner and outer signed-distance limits of the generated SDF band [m]."""

    sdf_target_voxel_size: float | None = None
    """Target sparse-SDF voxel size [m]."""

    sdf_max_resolution: int | None = None
    """Maximum sparse-SDF grid dimension."""

    sdf_texture_format: str | None = None
    """SDF voxel storage format."""

    force_sdf: bool | None = None
    """Whether to build an SDF when no explicit resolution is configured."""

    sdf_padding: float | None = None
    """Extra construction padding used while building the mesh SDF [m]."""


@configclass
class RigidBodyMaterialCfg:
    """Common rigid-contact material intent.

    All fields use sparse-overlay semantics: ``None`` preserves the source or
    backend default.  The Default backend consumes all three values.  Newton
    has one Coulomb friction coefficient, so it maps :attr:`dynamic_friction`
    to ``ShapeConfig.mu`` and currently has no separate static-friction input;
    restitution is consumed only by Newton solvers that support it.
    """

    static_friction: float | None = None
    """Static friction coefficient used before tangential slip begins.

    This is currently consumed only by the Default backend.
    """

    dynamic_friction: float | None = None
    """Sliding friction coefficient.

    The Default backend uses it as dynamic friction; Newton uses it as its
    single Coulomb friction coefficient ``mu``.
    """

    restitution: float | None = None
    """Coefficient of restitution, where zero is inelastic and one is elastic.

    The active backend/solver may further restrict or ignore restitution.
    """


@configclass
class DefaultRigidBodyMaterialCfg(RigidBodyMaterialCfg):
    """Contact-material extensions consumed only by the Default backend."""

    torsional_patch_radius: float | None = None
    """Contact-patch radius used to approximate torsional friction [m].

    Zero disables the approximation.
    """

    min_torsional_patch_radius: float | None = None
    """Minimum contact-patch radius used for torsional friction [m]."""

    disable_strong_friction: bool | None = None
    """Whether to disable Default-backend strong-friction contact anchoring."""


@configclass
class NewtonRigidBodyMaterialCfg(RigidBodyMaterialCfg):
    """Newton contact-material extensions.

    Solver support differs by field.  Semi-implicit and Featherstone consume
    ``ke``, ``kd``, ``kf``, ``ka``, ``mu``, and ``kh``; MuJoCo Warp consumes
    ``ke``, ``kd``, ``mu``, ``kh``, and the torsional/rolling coefficients;
    XPBD consumes ``mu``, restitution, and torsional/rolling friction.  DexSim
    warns when an explicitly changed contact field is ignored by the selected
    solver.
    """

    ke: float | None = None
    """Elastic contact stiffness coefficient."""

    kd: float | None = None
    """Normal contact damping coefficient."""

    kf: float | None = None
    """Tangential/friction damping coefficient."""

    ka: float | None = None
    """Contact adhesion distance [m]."""

    kh: float | None = None
    """Hydroelastic contact stiffness used when hydroelastic contact is enabled."""

    torsional_friction: float | None = None
    """Torsional friction coefficient resisting spin at a contact point."""

    rolling_friction: float | None = None
    """Rolling friction coefficient resisting rolling motion."""


def _nested_cfg_from_dict(
    value: Mapping[str, Any] | object | None,
    *,
    config_type: type,
    field_name: str,
) -> object | None:
    """Parse one optional, statically typed nested config."""
    if value is None or isinstance(value, config_type):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping or {config_type.__name__}.")
    try:
        return config_type(**dict(value))
    except TypeError as exc:
        raise TypeError(f"Invalid {field_name} configuration: {exc}") from exc


@configclass
class DefaultRigidBodyPhysicsCfg:
    """Default-only extension block for one rigid-body configuration.

    Portable inherited fields must remain in the common slots on
    :class:`RigidBodyPhysicsCfg`; this block is reserved for native fields.
    """

    rigid_props: DefaultRigidBodyPropertiesCfg | None = None
    collision_props: DefaultCollisionPropertiesCfg | None = None
    material_props: DefaultRigidBodyMaterialCfg | None = None

    @classmethod
    def from_dict(cls, init_dict: Mapping[str, Any]) -> DefaultRigidBodyPhysicsCfg:
        """Parse a Default backend extension block."""
        unknown = set(init_dict) - {
            "rigid_props",
            "collision_props",
            "material_props",
        }
        if unknown:
            raise KeyError(
                f"Unknown DefaultRigidBodyPhysicsCfg fields: {sorted(unknown)}"
            )
        return cls(
            rigid_props=_nested_cfg_from_dict(
                init_dict.get("rigid_props"),
                config_type=DefaultRigidBodyPropertiesCfg,
                field_name="default_props.rigid_props",
            ),
            collision_props=_nested_cfg_from_dict(
                init_dict.get("collision_props"),
                config_type=DefaultCollisionPropertiesCfg,
                field_name="default_props.collision_props",
            ),
            material_props=_nested_cfg_from_dict(
                init_dict.get("material_props"),
                config_type=DefaultRigidBodyMaterialCfg,
                field_name="default_props.material_props",
            ),
        )


@configclass
class NewtonRigidBodyPhysicsCfg:
    """Newton-only extension block for one rigid-body configuration."""

    rigid_props: NewtonRigidBodyPropertiesCfg | None = None
    collision_props: NewtonCollisionPropertiesCfg | None = None
    mesh_collision_props: NewtonMeshCollisionPropertiesCfg | None = None
    material_props: NewtonRigidBodyMaterialCfg | None = None

    @classmethod
    def from_dict(cls, init_dict: Mapping[str, Any]) -> NewtonRigidBodyPhysicsCfg:
        """Parse a Newton backend extension block."""
        unknown = set(init_dict) - {
            "rigid_props",
            "collision_props",
            "mesh_collision_props",
            "material_props",
        }
        if unknown:
            raise KeyError(
                f"Unknown NewtonRigidBodyPhysicsCfg fields: {sorted(unknown)}"
            )
        return cls(
            rigid_props=_nested_cfg_from_dict(
                init_dict.get("rigid_props"),
                config_type=NewtonRigidBodyPropertiesCfg,
                field_name="newton_props.rigid_props",
            ),
            collision_props=_nested_cfg_from_dict(
                init_dict.get("collision_props"),
                config_type=NewtonCollisionPropertiesCfg,
                field_name="newton_props.collision_props",
            ),
            mesh_collision_props=_nested_cfg_from_dict(
                init_dict.get("mesh_collision_props"),
                config_type=NewtonMeshCollisionPropertiesCfg,
                field_name="newton_props.mesh_collision_props",
            ),
            material_props=_nested_cfg_from_dict(
                init_dict.get("material_props"),
                config_type=NewtonRigidBodyMaterialCfg,
                field_name="newton_props.material_props",
            ),
        )


_RIGID_PHYSICS_LEGACY_FIELD_GROUPS = {
    "mass": "mass_props",
    "density": "mass_props",
    "inertia": "mass_props",
    "com_position": "mass_props",
    "com_quaternion": "mass_props",
    "linear_damping": "rigid_props",
    "angular_damping": "rigid_props",
    "max_linear_velocity": "rigid_props",
    "max_angular_velocity": "rigid_props",
    "max_depenetration_velocity": "rigid_props",
    "enable_ccd": "rigid_props",
    "min_position_iters": "rigid_props",
    "min_velocity_iters": "rigid_props",
    "sleep_threshold": "rigid_props",
    "contact_offset": "collision_props",
    "rest_offset": "collision_props",
    "static_friction": "material_props",
    "dynamic_friction": "material_props",
    "restitution": "material_props",
}

_RIGID_PHYSICS_GROUP_FIELDS = frozenset(
    {
        "mass_props",
        "rigid_props",
        "collision_props",
        "mesh_collision_props",
        "material_props",
        "default_props",
        "newton_props",
    }
)


def _physics_property_cfg_from_dict(
    value: Mapping[str, Any] | object | None,
    *,
    common_type: type,
    default_type: type,
    newton_type: type,
    field_name: str,
) -> object | None:
    """Parse one polymorphic rigid-physics property slot."""
    if value is None:
        return None
    if isinstance(value, common_type):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping or {common_type.__name__}.")
    data = dict(value)
    configured_backend = data.pop("backend", None)
    if configured_backend is None:
        common_fields = {item.name for item in fields(common_type)}
        default_fields = {item.name for item in fields(default_type)} - common_fields
        newton_fields = {item.name for item in fields(newton_type)} - common_fields
        has_default_fields = bool(default_fields.intersection(data))
        has_newton_fields = bool(newton_fields.intersection(data))
        if has_default_fields and has_newton_fields:
            raise ValueError(
                f"{field_name} mixes Default and Newton-only fields; select one "
                "backend-specific property config."
            )
        backend = (
            "default"
            if has_default_fields
            else "newton" if has_newton_fields else "common"
        )
    else:
        backend = str(configured_backend).replace("-", "_").lower()
    config_type = {
        "common": common_type,
        "default": default_type,
        "newton": newton_type,
    }.get(backend)
    if config_type is None:
        raise ValueError(
            f"{field_name}.backend must be 'common', 'default', or 'newton', "
            f"got {backend!r}."
        )
    try:
        return config_type(**data)
    except TypeError as exc:
        raise TypeError(f"Invalid {field_name} configuration: {exc}") from exc


def _physics_property_cfg_to_dict(
    value: object | None,
    *,
    common_type: type,
    default_type: type,
    newton_type: type,
    field_name: str,
) -> dict[str, Any] | None:
    """Serialize one polymorphic property slot with a stable discriminator."""
    if value is None:
        return None
    if isinstance(value, newton_type):
        backend = "newton"
    elif isinstance(value, default_type):
        backend = "default"
    elif type(value) is common_type:
        backend = None
    else:
        raise TypeError(
            f"Unsupported {field_name} config type {type(value).__name__!r}."
        )
    data = dict(value.to_dict())
    if backend is not None:
        data["backend"] = backend
    return data


@configclass
class RigidBodyPhysicsCfg:
    """Grouped rigid-body physics configuration used by Spawn.

    Common slots carry backend-neutral values. :attr:`default_props` and
    :attr:`newton_props` carry native extensions and may be configured at the
    same time. The older polymorphic subclasses in the common slots remain
    accepted as compatibility input; an explicit backend block takes
    precedence for duplicate native fields.

    Every nested field defaults to ``None``.  With
    ``asset_physics_mode="overlay"``, Spawn therefore changes only explicitly
    configured values and preserves all other USD/URDF or backend defaults.
    Dict/YAML input for compatibility slots selects a subclass with a local
    ``backend: common|default|newton`` discriminator; a unique native field may
    also infer the subclass. New definitions should keep those slots common
    and place backend-native values in the explicit backend blocks.

    .. attention::
        Portable fields inherited by a backend subtype still belong in the
        common slot. Explicit backend blocks accept native fields only.
    """

    mass_props: MassPropertiesCfg | None = None
    """Backend-neutral mass, inertia, and center-of-mass overrides."""

    rigid_props: RigidBodyPropertiesCfg | None = None
    """Optional body-level backend properties.

    Use :class:`DefaultRigidBodyPropertiesCfg` for Default-backend fields or the
    currently empty :class:`NewtonRigidBodyPropertiesCfg` extension point.
    """

    collision_props: CollisionPropertiesCfg | None = None
    """Portable collision envelope plus optional backend-native shape properties."""

    mesh_collision_props: MeshCollisionPropertiesCfg | None = None
    """Mesh collision approximation/cooking settings independent of render geometry."""

    material_props: RigidBodyMaterialCfg | None = None
    """Portable contact material values plus optional backend-native coefficients."""

    default_props: DefaultRigidBodyPhysicsCfg | None = None
    """Default-only native property extensions."""

    newton_props: NewtonRigidBodyPhysicsCfg | None = None
    """Newton-only native property extensions, including mesh SDF settings."""

    @classmethod
    def from_dict(cls, init_dict: Mapping[str, Any]) -> RigidBodyPhysicsCfg:
        """Parse grouped physics properties from a YAML/JSON-style mapping."""
        unknown = set(init_dict) - _RIGID_PHYSICS_GROUP_FIELDS
        if unknown:
            raise KeyError(f"Unknown RigidBodyPhysicsCfg fields: {sorted(unknown)}")
        cfg = cls()
        if "mass_props" in init_dict:
            value = init_dict["mass_props"]
            if value is not None:
                if not isinstance(value, (MassPropertiesCfg, Mapping)):
                    raise TypeError(
                        "mass_props must be a mapping or MassPropertiesCfg."
                    )
                cfg.mass_props = (
                    value
                    if isinstance(value, MassPropertiesCfg)
                    else MassPropertiesCfg(**value)
                )
        if "rigid_props" in init_dict:
            cfg.rigid_props = _physics_property_cfg_from_dict(
                init_dict["rigid_props"],
                common_type=RigidBodyPropertiesCfg,
                default_type=DefaultRigidBodyPropertiesCfg,
                newton_type=NewtonRigidBodyPropertiesCfg,
                field_name="rigid_props",
            )
        if "collision_props" in init_dict:
            cfg.collision_props = _physics_property_cfg_from_dict(
                init_dict["collision_props"],
                common_type=CollisionPropertiesCfg,
                default_type=DefaultCollisionPropertiesCfg,
                newton_type=NewtonCollisionPropertiesCfg,
                field_name="collision_props",
            )
        if "mesh_collision_props" in init_dict:
            cfg.mesh_collision_props = _nested_cfg_from_dict(
                init_dict["mesh_collision_props"],
                config_type=MeshCollisionPropertiesCfg,
                field_name="mesh_collision_props",
            )
        if "material_props" in init_dict:
            cfg.material_props = _physics_property_cfg_from_dict(
                init_dict["material_props"],
                common_type=RigidBodyMaterialCfg,
                default_type=DefaultRigidBodyMaterialCfg,
                newton_type=NewtonRigidBodyMaterialCfg,
                field_name="material_props",
            )
        if "default_props" in init_dict:
            value = init_dict["default_props"]
            if value is not None:
                if not isinstance(value, (DefaultRigidBodyPhysicsCfg, Mapping)):
                    raise TypeError(
                        "default_props must be a mapping or "
                        "DefaultRigidBodyPhysicsCfg."
                    )
                cfg.default_props = (
                    value
                    if isinstance(value, DefaultRigidBodyPhysicsCfg)
                    else DefaultRigidBodyPhysicsCfg.from_dict(value)
                )
        if "newton_props" in init_dict:
            value = init_dict["newton_props"]
            if value is not None:
                if not isinstance(value, (NewtonRigidBodyPhysicsCfg, Mapping)):
                    raise TypeError(
                        "newton_props must be a mapping or "
                        "NewtonRigidBodyPhysicsCfg."
                    )
                cfg.newton_props = (
                    value
                    if isinstance(value, NewtonRigidBodyPhysicsCfg)
                    else NewtonRigidBodyPhysicsCfg.from_dict(value)
                )
        return cfg

    def to_dict(self) -> dict[str, Any]:
        """Serialize grouped properties without losing backend subclasses."""
        return {
            "mass_props": (
                None if self.mass_props is None else self.mass_props.to_dict()
            ),
            "rigid_props": _physics_property_cfg_to_dict(
                self.rigid_props,
                common_type=RigidBodyPropertiesCfg,
                default_type=DefaultRigidBodyPropertiesCfg,
                newton_type=NewtonRigidBodyPropertiesCfg,
                field_name="rigid_props",
            ),
            "collision_props": _physics_property_cfg_to_dict(
                self.collision_props,
                common_type=CollisionPropertiesCfg,
                default_type=DefaultCollisionPropertiesCfg,
                newton_type=NewtonCollisionPropertiesCfg,
                field_name="collision_props",
            ),
            "mesh_collision_props": (
                None
                if self.mesh_collision_props is None
                else self.mesh_collision_props.to_dict()
            ),
            "material_props": _physics_property_cfg_to_dict(
                self.material_props,
                common_type=RigidBodyMaterialCfg,
                default_type=DefaultRigidBodyMaterialCfg,
                newton_type=NewtonRigidBodyMaterialCfg,
                field_name="material_props",
            ),
            "default_props": (
                None if self.default_props is None else self.default_props.to_dict()
            ),
            "newton_props": (
                None if self.newton_props is None else self.newton_props.to_dict()
            ),
        }

    @property
    def enable_collision(self) -> bool:
        """Compatibility view used by legacy object initialization."""
        value = (
            None
            if self.collision_props is None
            else self.collision_props.collision_enabled
        )
        return True if value is None else bool(value)

    def attr(self) -> PhysicalAttr:
        """Project Default-compatible values to the legacy ``PhysicalAttr``.

        Newton-native fields have no representation in ``PhysicalAttr`` and are
        intentionally omitted.  New Spawn code should consume the grouped
        configuration directly instead of calling this compatibility method.
        """
        attr = PhysicalAttr()
        for cfg in (
            self.mass_props,
            (
                self.rigid_props
                if isinstance(self.rigid_props, DefaultRigidBodyPropertiesCfg)
                else None
            ),
            (
                self.collision_props
                if isinstance(self.collision_props, CollisionPropertiesCfg)
                else None
            ),
            self.material_props,
            *(
                (
                    self.default_props.rigid_props,
                    self.default_props.collision_props,
                    self.default_props.material_props,
                )
                if self.default_props is not None
                else ()
            ),
        ):
            if cfg is None:
                continue
            for item in fields(cfg):
                value = getattr(cfg, item.name)
                if value is not None and hasattr(attr, item.name):
                    setattr(attr, item.name, value)
        return attr

    def __getattr__(self, name: str) -> Any:
        """Provide read-only compatibility for legacy flat property access."""
        group_name = _RIGID_PHYSICS_LEGACY_FIELD_GROUPS.get(name)
        if group_name is None:
            raise AttributeError(name)
        group = object.__getattribute__(self, group_name)
        if group is not None and hasattr(group, name):
            value = getattr(group, name)
            if value is not None:
                return value
        default_props = object.__getattribute__(self, "default_props")
        if default_props is not None:
            backend_group = getattr(default_props, group_name, None)
            if backend_group is not None and hasattr(backend_group, name):
                value = getattr(backend_group, name)
                if value is not None:
                    return value
        legacy_defaults = PhysicalAttr()
        return getattr(legacy_defaults, name, None)


def _rigid_body_attrs_from_dict(
    value: Mapping[str, Any],
    *,
    override: bool = False,
) -> RigidBodyPhysicsCfg | RigidBodyAttributesCfg | RigidBodyAttributesOverrideCfg:
    """Parse grouped physics or the deprecated Default-only flat schema."""
    grouped_fields = _RIGID_PHYSICS_GROUP_FIELDS.intersection(value)
    if grouped_fields:
        flat_fields = set(value) - _RIGID_PHYSICS_GROUP_FIELDS
        if flat_fields:
            raise ValueError(
                "Do not mix deprecated flat rigid-body fields with grouped "
                f"RigidBodyPhysicsCfg fields: {sorted(flat_fields)}"
            )
        return RigidBodyPhysicsCfg.from_dict(value)
    legacy_type = RigidBodyAttributesOverrideCfg if override else RigidBodyAttributesCfg
    return legacy_type.from_dict(dict(value))
