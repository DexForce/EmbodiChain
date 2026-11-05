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

"""Source-specific physical object defaults and declaration adaptation."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

__all__ = ["source_scene_object_declaration"]

_BACKGROUND_POLICY = {
    "mass": 10.0,
    "static_friction": 0.95,
    "dynamic_friction": 0.9,
    "restitution": 0.01,
    "max_convex_hull_num": 1,
}
_RIGID_POLICY = {
    "mass": 0.1,
    "static_friction": 0.95,
    "dynamic_friction": 0.9,
    "linear_damping": 0.9,
    "angular_damping": 0.9,
    "contact_offset": 0.003,
    "rest_offset": 0.001,
    "restitution": 0.05,
    "max_depenetration_velocity": 0.8,
    "max_linear_velocity": 5.0,
    "max_angular_velocity": 5.0,
    "min_position_iters": 32,
    "min_velocity_iters": 8,
    "max_convex_hull_num": 16,
    "acd_method": "visacd",
}
_BACKGROUND_ATTRS = {
    key: value
    for key, value in _BACKGROUND_POLICY.items()
    if key != "max_convex_hull_num"
}
_RIGID_ATTRS = {
    key: value
    for key, value in _RIGID_POLICY.items()
    if key not in {"max_convex_hull_num", "acd_method"}
}
# Prompt2Scene and older Scene Engine exports store rigid-body properties in a
# single flat mapping.  ``RigidBodyPhysicsCfg`` now has four typed property
# slots, so normalize the legacy spelling once at the source-scene boundary.
# Keeping this table here also makes the migration explicit: a field cannot be
# accidentally emitted in a group that a selected backend does not understand.
_PHYSICS_FIELD_GROUPS = {
    "mass": "mass_props",
    "density": "mass_props",
    "inertia": "mass_props",
    "recompute_inertia": "mass_props",
    "com_position": "mass_props",
    "com_quaternion": "mass_props",
    "linear_damping": "rigid_props",
    "angular_damping": "rigid_props",
    "has_gravity": "rigid_props",
    "max_linear_velocity": "rigid_props",
    "max_angular_velocity": "rigid_props",
    "max_depenetration_velocity": "rigid_props",
    "retain_acceleration": "rigid_props",
    "enable_ccd": "rigid_props",
    "min_position_iters": "rigid_props",
    "min_velocity_iters": "rigid_props",
    "sleep_threshold": "rigid_props",
    "collision_enabled": "collision_props",
    # ``enable_collision`` was the old public spelling.
    "enable_collision": "collision_props",
    "contact_offset": "collision_props",
    "rest_offset": "collision_props",
    "static_friction": "material_props",
    "dynamic_friction": "material_props",
    "restitution": "material_props",
}
_PHYSICS_GROUP_NAMES = (
    "mass_props",
    "rigid_props",
    "collision_props",
    "material_props",
)
_LEGACY_PHYSICS_GROUP_ALIASES = frozenset({"default_props", "newton_props"})
_PHYSICS_NATIVE_FIELD_GROUPS = {
    "torsional_patch_radius": "collision_props",
    "min_torsional_patch_radius": "collision_props",
    "disable_strong_friction": "collision_props",
    "condim": "collision_props",
    "has_particle_collision": "collision_props",
    "margin": "collision_props",
    "gap": "collision_props",
    "ke": "material_props",
    "kd": "material_props",
    "kf": "material_props",
    "ka": "material_props",
    "kh": "material_props",
    "torsional_friction": "material_props",
    "rolling_friction": "material_props",
}
_ALL_PHYSICS_FIELD_GROUPS = {
    **_PHYSICS_FIELD_GROUPS,
    **_PHYSICS_NATIVE_FIELD_GROUPS,
}


def _grouped_physics_attrs(
    source_attrs: Mapping[str, Any],
    defaults: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Convert legacy rigid-body attributes to the grouped config schema.

    Prompt2Scene exports predate :class:`RigidBodyPhysicsCfg` and place values
    such as ``mass`` and ``dynamic_friction`` directly under ``attrs``.  New
    simulation configs reject that spelling, so the conversion belongs at this
    ingestion boundary.  Canonical grouped blocks are accepted as-is and take
    precedence over a legacy flat value when both spellings are present.

    Args:
        source_attrs: Physics mapping from the source-scene object.
        defaults: Flat policy values used only when a source value is absent.

    Returns:
        A deep-copied mapping containing only the four grouped physics slots.

    Raises:
        ValueError: If a grouped block is not a mapping or a source field is
            not a recognized rigid-body property.
    """
    if not isinstance(source_attrs, Mapping):
        raise ValueError("Source rigid-body attrs must be a mapping.")

    grouped: dict[str, dict[str, Any]] = {group: {} for group in _PHYSICS_GROUP_NAMES}

    def add_flat(name: str, value: Any, *, from_default: bool = False) -> None:
        canonical_name = {
            "enable_collision": "collision_enabled",
            # Newton's native Coulomb coefficient has the same source intent
            # as the portable dynamic-friction field.
            "mu": "dynamic_friction",
        }.get(name, name)
        group = _ALL_PHYSICS_FIELD_GROUPS.get(canonical_name)
        if group is None:
            if from_default:
                return
            raise ValueError(
                "Unsupported source rigid-body attrs field "
                f"{name!r}; use grouped mass_props, rigid_props, "
                "collision_props, or material_props."
            )
        grouped[group][canonical_name] = deepcopy(value)

    # Defaults are deliberately applied first; every explicit source spelling
    # below then overrides them.
    for name, value in defaults.items():
        if name in _ALL_PHYSICS_FIELD_GROUPS:
            add_flat(name, value, from_default=True)

    legacy_alias_values: list[tuple[str, Mapping[str, Any]]] = []
    flat_values: list[tuple[str, Any]] = []
    canonical_values: list[tuple[str, Mapping[str, Any]]] = []
    for name, value in source_attrs.items():
        if name in _PHYSICS_GROUP_NAMES:
            if value is None:
                continue
            if not isinstance(value, Mapping):
                raise ValueError(f"Source attrs.{name} must be a mapping.")
            canonical_values.append((name, value))
        elif name in _LEGACY_PHYSICS_GROUP_ALIASES:
            if value is None:
                continue
            if not isinstance(value, Mapping):
                raise ValueError(f"Source attrs.{name} must be a mapping.")
            legacy_alias_values.append((name, value))
        else:
            flat_values.append((name, value))

    for name, value in flat_values:
        add_flat(name, value)

    # Intermediate exports used ``default_props``/``newton_props`` as flat
    # containers.  Route their recognized fields through the same map; the
    # current grouped parser infers the backend from native fields.
    for alias_name, values in legacy_alias_values:
        for name, value in values.items():
            if name == "backend":
                continue
            add_flat(name, value)

    # A canonical block is the most explicit representation and therefore
    # wins over any legacy value that may have been emitted alongside it.
    for group, values in canonical_values:
        grouped[group].update(deepcopy(dict(values)))

    return {group: values for group, values in grouped.items() if values}


def source_scene_object_declaration(
    config: Mapping[str, Any], *, role: str
) -> dict[str, Any]:
    """Build an owned physical object declaration from a normalized source mapping.

    Args:
        config: Source object with already selected UID, resolved paths and poses.
        role: Source section: ``background``, ``rigid_object`` or ``articulation``.

    Returns:
        Detached declaration retaining authored physical values, source property
        precedence, legacy collider handling and bounded source mesh cooking.
        Articulations retain their authored fields except the description. This
        explicit source adapter leaves shared simulation configuration defaults
        unchanged and opens no files or native resources.

    Raises:
        ValueError: For unsupported source physics fields, malformed property
            groups, or conflicting legacy/canonical collider declarations.
    """
    if role == "articulation":
        # Articulation schemas vary by asset; preserve their source fields after
        # path and pose normalization instead of guessing a reduced schema.
        result = deepcopy(dict(config))
        result.pop("description", None)
        return result

    result = {
        key: deepcopy(config[key])
        for key in (
            "uid",
            "shape",
            "init_pos",
            "init_rot",
            "init_local_pose",
            "body_scale",
        )
        if key in config
    }
    result.setdefault("body_scale", [1.0, 1.0, 1.0])
    source_attrs = config.get("attrs", {})
    if not isinstance(source_attrs, Mapping):
        source_attrs = {}
    source_attrs = dict(source_attrs)
    shape = result.get("shape")
    legacy_mesh_collision = source_attrs.pop("mesh_collision_props", None)
    if legacy_mesh_collision is not None:
        if not isinstance(shape, Mapping):
            raise ValueError("Legacy attrs.mesh_collision_props requires a mesh shape.")
        if shape.get("collision") is not None:
            raise ValueError(
                "Legacy attrs.mesh_collision_props cannot be combined with "
                "shape.collision."
            )
        shape = dict(shape)
        shape["collision"] = deepcopy(legacy_mesh_collision)
        result["shape"] = shape
    if role == "background":
        result["attrs"] = _grouped_physics_attrs(source_attrs, _BACKGROUND_ATTRS)
        result["body_type"] = "kinematic"
        result["max_convex_hull_num"] = int(_BACKGROUND_POLICY["max_convex_hull_num"])
    else:
        # Imported physical properties are authoritative; defaults only fill gaps.
        result["attrs"] = _grouped_physics_attrs(source_attrs, _RIGID_ATTRS)
        result["body_type"] = "dynamic"
        hull_limit = int(_RIGID_POLICY["max_convex_hull_num"])
        max_hulls = max(
            1,
            min(int(config.get("max_convex_hull_num", hull_limit)), hull_limit),
        )
        result["max_convex_hull_num"] = max_hulls
        result["acd_method"] = str(_RIGID_POLICY["acd_method"])
        shape = result.get("shape")
        if isinstance(shape, dict):
            if shape.get("collision") is None:
                # Legacy exports carry mesh cooking at the shape's top level;
                # retain that compatibility spelling until the final config
                # loader performs its deprecation migration.
                shape["acd_method"] = str(_RIGID_POLICY["acd_method"])
                shape["max_convex_hull_num"] = max_hulls
            else:
                # Canonical scene exports already own collision cooking in a
                # nested MeshCfg block.  Do not reintroduce deprecated sibling
                # fields, which the strict shape parser rejects when both
                # representations are present.
                shape.pop("acd_method", None)
                shape.pop("max_convex_hull_num", None)
    return result
