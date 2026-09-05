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

"""Normalize source-backed articulation physics for both Spawn backends."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any
from xml.etree import ElementTree

import numpy as np
from dexsim.spawn import (
    ArticulationDesc,
    CollisionDesc,
    DexsimCollisionDesc,
    DexsimPhysicsDesc,
    RigidBodyPhysicsDesc,
)

if TYPE_CHECKING:
    from dexsim.spawn import SceneBuilder

__all__ = ["resolve_articulation_source"]


@dataclass(frozen=True)
class _UrdfInertialState:
    """Validity of one source ``<inertial>`` block.

    The native Default loader replaces an all-zero URDF inertia with a small
    epsilon tensor.  That replacement is useful as a native fallback, but it
    must not be mistaken for an authored source inertia when EmbodiChain
    applies a sparse overlay later.
    """

    has_inertial: bool
    mass_valid: bool
    inertia_valid: bool
    has_collision_geometry: bool


def resolve_articulation_source(
    builder: SceneBuilder,
    desc: ArticulationDesc,
) -> ArticulationDesc:
    """Populate exact URDF metadata without building a Newton model.

    DexSim 0.4.3 removed its public source-resolution phase while retaining
    the same URDF-to-descriptor translator inside the Newton adapter. This
    compatibility boundary invokes that translator with a disposable
    render-only skeleton, allowing name-dependent EmbodiChain overlays to be
    authored before :meth:`SceneBuilder.finalize`.

    Args:
        builder: Scene builder that owns the target arena layout.
        desc: Articulation descriptor to resolve in place.

    Returns:
        The resolved descriptor.
    """
    signature = _source_signature(desc)
    previous = getattr(desc, "_embodichain_source_signature", None)
    if previous == signature:
        return desc

    if desc.urdf_path is None:
        setattr(desc, "_embodichain_source_signature", signature)
        return desc

    had_retained_links = bool(desc.links)
    if previous is not None:
        desc.links = []
        desc.joints = []
        desc.root_link_name = None

    arena = _source_arena(builder, desc)
    temp_name = f"__embodichain_resolve__{desc.name.replace('/', '__')}__{id(desc)}"
    skeleton = arena.create_skeleton("skeleton")
    if skeleton is None:
        raise RuntimeError(f"Failed to create a source resolver for {desc.name!r}.")
    skeleton.set_name(temp_name)
    skeleton.detach_parent()
    try:
        scale = np.asarray(desc.body_scale, dtype=np.float32).reshape(3)
        load_result = skeleton.load_urdf(os.path.abspath(desc.urdf_path), scale)
        if load_result != 0:
            raise RuntimeError(
                f"Skeleton.load_urdf({desc.urdf_path!r}) failed: {load_result}"
            )

        # DexSim currently exposes no public metadata-only resolver. Reuse the
        # adapter's source translator so its retained descriptor semantics stay
        # identical to the subsequent Newton build.
        from dexsim.spawn.adapters.newton_articulation_adapter import (
            _translate_urdf_articulation,
        )

        collision_link_names = set()
        for link_name in skeleton.get_link_names(True):
            try:
                if skeleton.get_collision_shapes(link_name):
                    collision_link_names.add(link_name)
            except (KeyError, RuntimeError, TypeError, AttributeError):
                continue
        _translate_urdf_articulation(skeleton, desc)
        source_states = _read_urdf_inertial_states(desc.urdf_path)
        _annotate_urdf_source_physics(
            desc,
            source_states,
            collision_link_names,
        )
        # A zero/invalid source tensor falls back to geometry in both backends.
        # The Newton translator currently carries the URDF origin through even
        # after rejecting that tensor; clear it before configuration can retain
        # the value as an explicit COM override.  Do not touch descriptors that
        # already contain caller-authored links (those are not source defaults).
        if previous is not None or not had_retained_links:
            _clear_invalid_source_com(desc)
    finally:
        # Drop the wrapper before deleting its Arena-owned native object.
        skeleton = None
        arena.remove_skeleton(temp_name)

    setattr(desc, "_embodichain_source_signature", signature)
    return desc


def _capture_dexsim_source_physics(
    handle: Any,
    desc: ArticulationDesc,
) -> ArticulationDesc:
    """Copy source physics into a materialized Default descriptor.

    DexSim's URDF adapter populates topology but intentionally leaves
    ``LinkDesc.rigid_body`` empty.  A later sparse overlay therefore calls the
    native setter with an empty mass-property block, which derives geometry
    inertia and silently replaces valid URDF inertia.  Capture the complete
    native body/contact snapshot before the config is merged, then let the
    Default application boundary write only links with an explicit overlay.

    Invalid/all-zero source inertia is represented as ``inertia=None`` and
    ``com_* = None``.  Both backends then use collision geometry, while a
    valid source mass remains an explicit mass override.
    """
    if getattr(desc, "_embodichain_source_physics_captured", False):
        return desc

    getter = getattr(handle, "get_physical_attr", None)
    if getter is None or desc.urdf_path is None:
        return desc

    source_states = _read_urdf_inertial_states(desc.urdf_path)
    for link in desc.links:
        try:
            attrib = getter(link.name)
        except (KeyError, RuntimeError, TypeError, AttributeError):
            # A backend may expose a topology link without a physical body.
            continue

        state = source_states.get(link.name) if source_states is not None else None
        if state is None:
            state = _infer_native_inertial_state(attrib)
        mass = _finite_positive_scalar(getattr(attrib, "mass", None))
        inertia = _finite_vector(getattr(attrib, "inertia", None), 3)
        com_position = _finite_vector(getattr(attrib, "com_position", None), 3)
        com_quaternion = _finite_vector(
            getattr(attrib, "com_quaternion", None),
            4,
        )
        source_inertia_valid = bool(
            state.has_inertial
            and state.mass_valid
            and state.inertia_valid
            and mass is not None
            and inertia is not None
            and np.any(inertia > 0.0)
            and com_position is not None
            and com_quaternion is not None
            and float(np.linalg.norm(com_quaternion)) > 1.0e-8
        )
        _set_source_link_markers(
            link,
            has_collision_geometry=state.has_collision_geometry,
            inertia_valid=source_inertia_valid,
        )

        # Keep a descriptor snapshot even when the source has no usable
        # inertial block. It preserves native damping/material values for a
        # later partial overlay without claiming native fallback inertia as an
        # authored asset value.
        body = _native_source_rigid_body(
            attrib,
            mass=(
                mass
                if source_inertia_valid
                or (state.has_inertial and state.mass_valid and mass is not None)
                else None
            ),
            density=(
                None
                if state.has_inertial and state.mass_valid and mass is not None
                else _finite_positive_scalar(getattr(attrib, "density", None))
            ),
            inertia=inertia if source_inertia_valid else None,
            com_position=com_position if source_inertia_valid else None,
            com_quaternion=com_quaternion if source_inertia_valid else None,
        )
        link.rigid_body = body
        link.replace_inertial = False
        link._inertia_from_source = source_inertia_valid
        if state.has_collision_geometry:
            link.collisions = [_native_source_collision(attrib)]

    setattr(desc, "_embodichain_source_physics_captured", True)
    return desc


def _clear_invalid_source_com(desc: ArticulationDesc) -> None:
    """Drop invalid source inertia/COM before geometric fallback is built."""
    for link in desc.links:
        body = getattr(link, "rigid_body", None)
        if (
            body is None
            or getattr(link, "_embodichain_source_inertia_valid", None) is not False
            or not getattr(link, "_embodichain_has_collision_geometry", False)
        ):
            continue
        body.inertia = None
        body.com_position = None
        body.com_quaternion = None
        link._inertia_from_source = False


def _annotate_urdf_source_physics(
    desc: ArticulationDesc,
    source_states: dict[str, _UrdfInertialState] | None,
    collision_link_names: set[str],
) -> None:
    """Attach source ownership markers to a Newton-resolved descriptor."""
    if source_states is None:
        for link in desc.links:
            setattr(
                link,
                "_embodichain_has_collision_geometry",
                link.name in collision_link_names,
            )
        return

    for link in desc.links:
        state = source_states.get(link.name)
        if state is None:
            continue
        inertia_valid = bool(
            state.has_inertial and state.mass_valid and state.inertia_valid
        )
        _set_source_link_markers(
            link,
            has_collision_geometry=link.name in collision_link_names,
            inertia_valid=inertia_valid,
        )
        if not inertia_valid:
            link._inertia_from_source = False


def _set_source_link_markers(
    link: Any,
    *,
    has_collision_geometry: bool,
    inertia_valid: bool,
) -> None:
    """Record source provenance without changing public Spawn descriptors."""
    setattr(
        link,
        "_embodichain_has_collision_geometry",
        bool(has_collision_geometry),
    )
    setattr(link, "_embodichain_source_inertia_valid", bool(inertia_valid))


def _native_source_rigid_body(
    attrib: Any,
    *,
    mass: float | None,
    density: float | None,
    inertia: np.ndarray | None,
    com_position: np.ndarray | None,
    com_quaternion: np.ndarray | None,
) -> RigidBodyPhysicsDesc:
    """Convert one native ``PhysicalAttr`` into a sparse Spawn body snapshot."""
    dexsim_values = {
        item.name: getattr(attrib, item.name, None)
        for item in fields(DexsimPhysicsDesc)
    }
    return RigidBodyPhysicsDesc.dynamic(
        mass=mass,
        density=density,
        inertia=inertia,
        com_position=com_position,
        com_quaternion=com_quaternion,
        dexsim=DexsimPhysicsDesc(**dexsim_values),
    )


def _native_source_collision(attrib: Any) -> CollisionDesc:
    """Convert native contact attributes into an attribute-only collision desc."""
    dexsim_values = {
        item.name: getattr(attrib, item.name, None)
        for item in fields(DexsimCollisionDesc)
    }
    return CollisionDesc(
        enable_collision=bool(getattr(attrib, "enable_collision", True)),
        dexsim=DexsimCollisionDesc(**dexsim_values),
    )


def _retain_dexsim_source_descriptor(
    handle: Any,
    desc: ArticulationDesc,
) -> ArticulationDesc | None:
    """Retain the normalized source descriptor without issuing native writes."""
    binding = getattr(handle, "_physics_binding", None)
    if binding is None or not hasattr(handle, "articulation_desc"):
        return None

    from dexsim.spawn._copy import copy_articulation_desc

    previous = handle.articulation_desc
    collision_filters = {
        link.name: link.rigid_body.collision_filter_data.copy()
        for link in getattr(previous, "links", ())
        if link.rigid_body is not None
        and link.rigid_body.collision_filter_data is not None
    }
    effective = copy_articulation_desc(desc)
    for link in effective.links:
        collision_filter = collision_filters.get(link.name)
        if collision_filter is not None and link.rigid_body is not None:
            link.rigid_body.collision_filter_data = collision_filter
    handle.articulation_desc = effective
    if hasattr(handle, "_desc_shared"):
        handle._desc_shared = False
    return effective


def _apply_dexsim_source_overlay(
    handle: Any,
    desc: ArticulationDesc,
) -> None:
    """Apply only explicit Default physics overlays to a loaded articulation.

    The public DexSim helper applies every non-empty ``LinkDesc``. EmbodiChain
    intentionally retains a full source snapshot in those descriptors so that
    source properties are visible and mergeable just like Newton's pre-build
    descriptor. Applying that snapshot wholesale would nevertheless invoke
    the Default native setter for every link and trigger unwanted geometric
    inertia derivation. This narrow compatibility boundary keeps its native
    calls limited to links marked by :func:`configure_articulation_desc`.
    """
    binding = getattr(handle, "_physics_binding", None)
    if binding is None:
        # Keep light-weight Scene tests and third-party Spawn handles on the
        # public DexSim path. Real Default SpawnedArticulations always expose
        # the binding used below.
        apply = getattr(handle, "apply_dexsim_properties", None)
        if apply is not None:
            apply(desc)
        return

    effective = _retain_dexsim_source_descriptor(handle, desc)
    if effective is None:
        apply = getattr(handle, "apply_dexsim_properties", None)
        if apply is not None:
            apply(desc)
        return

    from dexsim.spawn.adapters.common import physical_attr_for_dexsim
    from dexsim.spawn.adapters.dexsim_adapter import (
        _apply_dexsim_joint_properties,
        _apply_rigid_body_mass_properties,
    )

    get_body = getattr(binding, "get_physical_body", None)
    set_attr = getattr(binding, "set_physical_attr", None)
    if get_body is None or set_attr is None:
        raise RuntimeError(
            "Default Spawn articulation binding does not expose source "
            "physical-property APIs."
        )

    for link in effective.links:
        if not getattr(link, "_embodichain_apply_physics", True):
            continue
        physics = link.rigid_body
        if physics is None:
            continue
        rigid_body = get_body(link.name)
        if rigid_body is None:
            continue
        attr = physical_attr_for_dexsim(physics, link.collisions)
        set_attr(attr, link.name, link.replace_inertial)
        _apply_rigid_body_mass_properties(
            rigid_body,
            physics,
            apply_inertia=(
                effective.urdf_read_inertia or not link._inertia_from_source
            ),
        )
        _restore_dexsim_mass_override(rigid_body, link)

    if effective.joints:
        _apply_dexsim_joint_properties(binding, effective.joints)


def _restore_dexsim_mass_override(rigid_body: Any, link: Any) -> None:
    """Correct Default's non-replacing source-mass behavior.

    ``DFArticulationX::DX_SetPhysicAttrib(..., replace_inertial=False)`` keeps
    the currently loaded mass even when the descriptor authors another one.
    Newton honors that mass before its model is built. Apply it afterwards at
    the raw body boundary and restore retained/explicit inertia and COM, which
    makes the two backends follow the same descriptor contract.
    """
    physics = getattr(link, "rigid_body", None)
    if (
        physics is None
        or link.replace_inertial
        or not getattr(link, "_embodichain_mass_override", False)
        or physics.mass is None
    ):
        return

    set_mass = getattr(rigid_body, "set_mass", None)
    if set_mass is None:
        return
    set_mass(float(physics.mass))

    if physics.inertia is not None:
        set_inertia = getattr(rigid_body, "set_mass_space_inertia_tensor", None)
        if set_inertia is not None:
            set_inertia(np.asarray(physics.inertia, dtype=np.float32).reshape(-1)[:3])
    if physics.com_position is not None or physics.com_quaternion is not None:
        get_com = getattr(rigid_body, "get_cmass_local_pose", None)
        set_com = getattr(rigid_body, "set_cmass_local_pose", None)
        if get_com is not None and set_com is not None:
            position, quaternion = get_com()
            if physics.com_position is not None:
                position = np.asarray(
                    physics.com_position,
                    dtype=np.float32,
                ).reshape(
                    -1
                )[:3]
            if physics.com_quaternion is not None:
                quaternion = np.asarray(
                    physics.com_quaternion,
                    dtype=np.float32,
                ).reshape(-1)[:4]
            set_com(position, quaternion)


def _read_urdf_inertial_states(
    path: str,
) -> dict[str, _UrdfInertialState] | None:
    """Read source inertial validity without changing the native asset."""
    try:
        root = ElementTree.parse(path).getroot()
    except (OSError, ElementTree.ParseError):
        return None

    states: dict[str, _UrdfInertialState] = {}
    for link_node in root.iter():
        if _xml_local_name(link_node.tag) != "link":
            continue
        link_name = link_node.attrib.get("name")
        if not link_name:
            continue
        inertial_node = next(
            (child for child in link_node if _xml_local_name(child.tag) == "inertial"),
            None,
        )
        if inertial_node is None:
            states[link_name] = _UrdfInertialState(
                False,
                False,
                False,
                any(_xml_local_name(child.tag) == "collision" for child in link_node),
            )
            continue

        mass_node = next(
            (child for child in inertial_node if _xml_local_name(child.tag) == "mass"),
            None,
        )
        inertia_node = next(
            (
                child
                for child in inertial_node
                if _xml_local_name(child.tag) == "inertia"
            ),
            None,
        )
        mass = _parse_float(
            None if mass_node is None else mass_node.attrib.get("value")
        )
        values = (
            None
            if inertia_node is None
            else [
                _parse_float(inertia_node.attrib.get(name))
                for name in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")
            ]
        )
        inertia_valid = False
        if values is not None and all(value is not None for value in values):
            matrix = np.asarray(
                [
                    [values[0], values[1], values[2]],
                    [values[1], values[3], values[4]],
                    [values[2], values[4], values[5]],
                ],
                dtype=np.float64,
            )
            try:
                inertia_valid = bool(
                    np.all(np.isfinite(matrix))
                    and not np.allclose(matrix, 0.0)
                    and np.all(np.linalg.eigvalsh(matrix) >= 0.0)
                )
            except np.linalg.LinAlgError:
                inertia_valid = False
        states[link_name] = _UrdfInertialState(
            True,
            mass is not None and bool(np.isfinite(mass) and mass > 0.0),
            inertia_valid,
            any(_xml_local_name(child.tag) == "collision" for child in link_node),
        )
    return states


def _infer_native_inertial_state(attrib: Any) -> _UrdfInertialState:
    """Conservative validity fallback when a source file cannot be parsed."""
    mass = _finite_positive_scalar(getattr(attrib, "mass", None))
    inertia = _finite_vector(getattr(attrib, "inertia", None), 3)
    valid = inertia is not None and np.any(inertia > 0.0)
    return _UrdfInertialState(True, mass is not None, bool(valid), True)


def _finite_positive_scalar(value: Any) -> float | None:
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    return scalar if np.isfinite(scalar) and scalar > 0.0 else None


def _finite_vector(value: Any, size: int) -> np.ndarray | None:
    try:
        array = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if array.size != size or not np.all(np.isfinite(array)):
        return None
    return array.copy()


def _parse_float(value: str | None) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _xml_local_name(tag: str) -> str:
    return str(tag).rsplit("}", 1)[-1]


def _source_signature(desc: ArticulationDesc) -> tuple[object, ...]:
    if desc.urdf_path is None:
        return "explicit", id(desc)
    return (
        "urdf",
        os.path.abspath(desc.urdf_path),
        tuple(float(value) for value in np.asarray(desc.body_scale).reshape(3)),
    )


def _source_arena(builder: SceneBuilder, desc: ArticulationDesc) -> Any:
    if desc.per_env and builder.replicate_plan is not None:
        arenas = builder.prepare_arenas()
        if not arenas:
            raise RuntimeError(
                f"No replicated Arena is available to resolve {desc.name!r}."
            )
        return arenas[0]
    return builder.world.get_env()
