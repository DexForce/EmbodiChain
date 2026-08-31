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
from embodichain.utils.math import convert_quat


@configclass
class MassPropertiesCfg:
    """Backend-neutral rigid-body mass properties.

    ``None`` means that the source asset or selected backend keeps ownership of
    that value. For a non-static body, explicit inertia requires a positive
    mass. A source-backed body retains authored inertia unless
    :attr:`recompute_inertia` is enabled; procedural or recomputed bodies derive
    inertia from collision geometry and the effective mass or density. Static
    bodies omit all mass properties during Spawn compilation.
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

    recompute_inertia: bool | None = None
    """Whether collision geometry should replace source-authored inertia.

    ``True`` discards source inertia so the backend recomputes it from the
    collision geometry and effective mass or density. ``False`` preserves the
    source inertia. ``None`` inherits an outer rigid-body overlay and otherwise
    behaves like ``False``. Explicit :attr:`inertia` cannot be combined with
    recomputation.
    """

    com_position: Sequence[float] | np.ndarray | None = None
    """Center-of-mass position expressed in the rigid body's local frame [m]."""

    com_quaternion: Sequence[float] | np.ndarray | None = None
    """Orientation of the center-of-mass/inertia frame in ``xyzw`` order.

    Spawn normalizes the quaternion and converts it to the backend descriptor's
    ``wxyz`` convention.  A zero quaternion is invalid.
    """


@configclass
class DefaultRigidBodyPropertiesCfg:
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
class CollisionPropertiesCfg:
    """Collision-shape properties with identical intent across both backends.

    ``None`` leaves the corresponding source/backend value unchanged.  The
    contact envelope is expressed once with Default-backend terminology and is
    compiled to Newton's ``margin``/``gap`` representation at the Spawn
    boundary. Mesh approximation and SDF cooking belong to
    :class:`~embodichain.lab.sim.shapes.MeshCollisionCfg`.
    """

    collision_enabled: bool | None = None
    """Whether the shape participates in rigid shape-shape collision.

    On Newton this maps to ``ShapeConfig.has_shape_collision``. ``None``
    preserves the source/backend value.
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
    """Collision-solver properties consumed only by the Default backend.

    ``contact_offset`` and ``rest_offset`` now live on
    :class:`CollisionPropertiesCfg` because both backends consume their intent.
    """

    torsional_patch_radius: float | None = None
    """Contact-patch radius used to approximate torsional friction [m]."""

    min_torsional_patch_radius: float | None = None
    """Minimum contact-patch radius used for torsional friction [m]."""

    disable_strong_friction: bool | None = None
    """Whether to disable Default-backend strong-friction contact anchoring."""


@configclass
class NewtonCollisionPropertiesCfg(CollisionPropertiesCfg):
    """Newton-native contact-envelope properties.

    Mesh construction belongs to ``MeshCfg.collision``; filtering, visual, and
    semantic-site policies are deliberately not part of rigid-body physics.

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


_RIGID_PHYSICS_GROUP_FIELDS = frozenset(
    {
        "mass_props",
        "rigid_props",
        "collision_props",
        "material_props",
    }
)

_REMOVED_RIGID_PHYSICS_GROUP_FIELDS = {
    "default_props": "the corresponding polymorphic property slot",
    "newton_props": "the corresponding polymorphic property slot",
    "mesh_collision_props": "MeshCfg.collision",
}


def _default_rigid_props_from_dict(
    value: Mapping[str, Any] | object | None,
) -> DefaultRigidBodyPropertiesCfg | None:
    """Parse the currently Default-only rigid-body property slot."""
    if value is None or isinstance(value, DefaultRigidBodyPropertiesCfg):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(
            "rigid_props must be a mapping or DefaultRigidBodyPropertiesCfg."
        )
    data = dict(value)
    backend = str(data.pop("backend", "default")).replace("-", "_").lower()
    if backend != "default":
        raise ValueError(
            "rigid_props.backend must be 'default'; Newton currently exposes no "
            "body-level property config."
        )
    try:
        return DefaultRigidBodyPropertiesCfg(**data)
    except TypeError as exc:
        raise TypeError(f"Invalid rigid_props configuration: {exc}") from exc


def _physics_property_cfg_from_dict(
    value: Mapping[str, Any] | object | None,
    *,
    common_type: type,
    backend_types: Mapping[str, type],
    field_name: str,
) -> object | None:
    """Parse one polymorphic rigid-physics property slot."""
    if value is None:
        return None
    supported_types = (common_type, *backend_types.values())
    if isinstance(value, supported_types):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping or {common_type.__name__}.")
    data = dict(value)
    configured_backend = data.pop("backend", None)
    if configured_backend is None:
        common_fields = {item.name for item in fields(common_type)}
        matching_backends = [
            backend
            for backend, config_type in backend_types.items()
            if (
                {item.name for item in fields(config_type)} - common_fields
            ).intersection(data)
        ]
        if len(matching_backends) > 1:
            raise ValueError(
                f"{field_name} mixes Default and Newton-only fields; select one "
                "backend-specific property config."
            )
        backend = matching_backends[0] if matching_backends else "common"
    else:
        backend = str(configured_backend).replace("-", "_").lower()
    config_type = common_type if backend == "common" else backend_types.get(backend)
    if config_type is None:
        supported_backends = ("common", *backend_types)
        raise ValueError(
            f"{field_name}.backend must be one of {supported_backends}, got "
            f"{backend!r}."
        )
    try:
        return config_type(**data)
    except TypeError as exc:
        raise TypeError(f"Invalid {field_name} configuration: {exc}") from exc


def _physics_property_cfg_to_dict(
    value: object | None,
    *,
    common_type: type,
    backend_types: Mapping[str, type],
    field_name: str,
) -> dict[str, Any] | None:
    """Serialize one polymorphic property slot with a stable discriminator."""
    if value is None:
        return None
    backend = next(
        (
            name
            for name, config_type in backend_types.items()
            if isinstance(value, config_type)
        ),
        None,
    )
    if backend is None and type(value) is not common_type:
        raise TypeError(
            f"Unsupported {field_name} config type {type(value).__name__!r}."
        )
    data = dict(value.to_dict())
    if backend is not None:
        data["backend"] = backend
    return data


def _copy_dexsim_physical_attr(source: PhysicalAttr) -> PhysicalAttr:
    """Copy a native ``PhysicalAttr`` without relying on pickle support.

    DexSim exposes ``PhysicalAttr`` through a pybind extension object, so
    :func:`copy.deepcopy` cannot clone it.  Copy scalar fields from the native
    mapping and clone its array-valued mass/COM fields explicitly before a
    sparse grouped overlay is applied.
    """
    copied = PhysicalAttr()
    for field_name, value in source.as_dict().items():
        setattr(copied, field_name, value)
    for field_name in ("inertia", "com_position", "com_quaternion"):
        value = getattr(source, field_name, None)
        if value is not None:
            setattr(copied, field_name, np.array(value, dtype=np.float32, copy=True))
    return copied


@configclass
class RigidBodyPhysicsCfg:
    """Grouped rigid-body physics configuration used by Spawn.

    Every nested field defaults to ``None``.  With
    ``asset_physics_mode="overlay"``, Spawn therefore changes only explicitly
    configured values and preserves all other USD/URDF or backend defaults.
    Each physical concept has exactly one slot. Dict/YAML input selects a
    backend subclass with a local discriminator, while a unique native field
    may infer that subclass. Mesh collision construction belongs to
    :class:`~embodichain.lab.sim.shapes.MeshCfg`, not this body-physics schema.
    """

    mass_props: MassPropertiesCfg | None = None
    """Backend-neutral mass, inertia, COM, and recomputation overrides."""

    rigid_props: DefaultRigidBodyPropertiesCfg | None = None
    """Optional Default-native body properties.

    Newton currently exposes no body-level property group beyond common mass
    properties, so there is no empty Newton marker config.
    """

    collision_props: CollisionPropertiesCfg | None = None
    """Portable collision envelope plus one optional backend-specific subtype."""

    material_props: RigidBodyMaterialCfg | None = None
    """Portable contact material values plus optional backend-native coefficients."""

    @classmethod
    def from_dict(cls, init_dict: Mapping[str, Any]) -> RigidBodyPhysicsCfg:
        """Parse grouped physics properties from a YAML/JSON-style mapping."""
        removed = _REMOVED_RIGID_PHYSICS_GROUP_FIELDS.keys() & init_dict.keys()
        if removed:
            replacements = ", ".join(
                f"{name} -> {_REMOVED_RIGID_PHYSICS_GROUP_FIELDS[name]}"
                for name in sorted(removed)
            )
            raise ValueError(f"Removed RigidBodyPhysicsCfg fields: {replacements}.")
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
            cfg.rigid_props = _default_rigid_props_from_dict(init_dict["rigid_props"])
        if "collision_props" in init_dict:
            cfg.collision_props = _physics_property_cfg_from_dict(
                init_dict["collision_props"],
                common_type=CollisionPropertiesCfg,
                backend_types={
                    "default": DefaultCollisionPropertiesCfg,
                    "newton": NewtonCollisionPropertiesCfg,
                },
                field_name="collision_props",
            )
        if "material_props" in init_dict:
            cfg.material_props = _physics_property_cfg_from_dict(
                init_dict["material_props"],
                common_type=RigidBodyMaterialCfg,
                backend_types={"newton": NewtonRigidBodyMaterialCfg},
                field_name="material_props",
            )
        return cfg

    def to_dict(self) -> dict[str, Any]:
        """Serialize grouped properties without losing backend subclasses."""
        return {
            "mass_props": (
                None if self.mass_props is None else self.mass_props.to_dict()
            ),
            "rigid_props": (
                None
                if self.rigid_props is None
                else {**self.rigid_props.to_dict(), "backend": "default"}
            ),
            "collision_props": _physics_property_cfg_to_dict(
                self.collision_props,
                common_type=CollisionPropertiesCfg,
                backend_types={
                    "default": DefaultCollisionPropertiesCfg,
                    "newton": NewtonCollisionPropertiesCfg,
                },
                field_name="collision_props",
            ),
            "material_props": _physics_property_cfg_to_dict(
                self.material_props,
                common_type=RigidBodyMaterialCfg,
                backend_types={"newton": NewtonRigidBodyMaterialCfg},
                field_name="material_props",
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

    def to_dexsim_physical_attr(
        self,
        *,
        base: PhysicalAttr | None = None,
    ) -> PhysicalAttr:
        """Translate configured Default-compatible values to ``PhysicalAttr``.

        Args:
            base: Optional native attributes to overlay.  This is used by the
                retained raw Default articulation path for sparse per-link
                updates.

        Returns:
            A DexSim physical-attribute object using its defaults for every
            unconfigured grouped field.
        """
        attr = PhysicalAttr() if base is None else _copy_dexsim_physical_attr(base)
        configs = (
            (self.mass_props, {"recompute_inertia": None}),
            (self.rigid_props, {}),
            (self.collision_props, {"collision_enabled": "enable_collision"}),
            (self.material_props, {}),
        )
        for cfg, field_map in configs:
            if cfg is None:
                continue
            for item in fields(cfg):
                value = getattr(cfg, item.name)
                target_name = field_map.get(item.name, item.name)
                if (
                    value is None
                    or target_name is None
                    or not hasattr(attr, target_name)
                ):
                    continue
                if target_name in {"inertia", "com_position"}:
                    value = np.asarray(value, dtype=np.float32)
                elif target_name == "com_quaternion":
                    value = convert_quat(np.asarray(value, dtype=np.float32), to="wxyz")
                setattr(attr, target_name, value)
        return attr

    @classmethod
    def from_dexsim_physical_attr(
        cls,
        attr: PhysicalAttr,
    ) -> RigidBodyPhysicsCfg:
        """Capture native Default attributes in the grouped configuration."""

        def _array(name: str) -> np.ndarray | None:
            value = getattr(attr, name, None)
            return None if value is None else np.asarray(value, dtype=np.float32)

        com_quaternion = _array("com_quaternion")
        if com_quaternion is not None:
            com_quaternion = convert_quat(com_quaternion, to="xyzw")
        return cls(
            mass_props=MassPropertiesCfg(
                mass=getattr(attr, "mass", None),
                density=getattr(attr, "density", None),
                inertia=_array("inertia"),
                com_position=_array("com_position"),
                com_quaternion=com_quaternion,
            ),
            rigid_props=DefaultRigidBodyPropertiesCfg(
                angular_damping=getattr(attr, "angular_damping", None),
                linear_damping=getattr(attr, "linear_damping", None),
                max_depenetration_velocity=getattr(
                    attr, "max_depenetration_velocity", None
                ),
                sleep_threshold=getattr(attr, "sleep_threshold", None),
                min_position_iters=getattr(attr, "min_position_iters", None),
                min_velocity_iters=getattr(attr, "min_velocity_iters", None),
                max_linear_velocity=getattr(attr, "max_linear_velocity", None),
                max_angular_velocity=getattr(attr, "max_angular_velocity", None),
                enable_ccd=getattr(attr, "enable_ccd", None),
            ),
            collision_props=DefaultCollisionPropertiesCfg(
                collision_enabled=getattr(attr, "enable_collision", None),
                contact_offset=getattr(attr, "contact_offset", None),
                rest_offset=getattr(attr, "rest_offset", None),
                torsional_patch_radius=getattr(attr, "torsional_patch_radius", None),
                min_torsional_patch_radius=getattr(
                    attr, "min_torsional_patch_radius", None
                ),
                disable_strong_friction=getattr(attr, "disable_strong_friction", None),
            ),
            material_props=RigidBodyMaterialCfg(
                restitution=getattr(attr, "restitution", None),
                dynamic_friction=getattr(attr, "dynamic_friction", None),
                static_friction=getattr(attr, "static_friction", None),
            ),
        )


_REMOVED_FLAT_RIGID_BODY_FIELDS = frozenset(
    {
        "mass",
        "density",
        "inertia",
        "com_position",
        "com_quaternion",
        "angular_damping",
        "linear_damping",
        "max_depenetration_velocity",
        "sleep_threshold",
        "min_position_iters",
        "min_velocity_iters",
        "max_linear_velocity",
        "max_angular_velocity",
        "enable_ccd",
        "contact_offset",
        "rest_offset",
        "enable_collision",
        "restitution",
        "dynamic_friction",
        "static_friction",
    }
)


def _rigid_body_physics_from_dict(value: Mapping[str, Any]) -> RigidBodyPhysicsCfg:
    """Parse the grouped rigid-body physics schema.

    Flat ``attrs`` fields and their compatibility configuration types were
    removed. Reject them at the config boundary so no input silently changes
    physical meaning.
    """
    flat_fields = _REMOVED_FLAT_RIGID_BODY_FIELDS.intersection(value)
    if flat_fields:
        raise ValueError(
            "Removed flat rigid-body attrs fields: "
            f"{sorted(flat_fields)}. Use grouped mass_props, rigid_props, "
            "collision_props, and material_props."
        )
    return RigidBodyPhysicsCfg.from_dict(value)
