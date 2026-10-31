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

"""Light and inter-object constraint configuration."""

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal, Sequence

import numpy as np

from embodichain.utils import configclass

from .asset import ObjectBaseCfg


@configclass
class LightCfg(ObjectBaseCfg):
    """Configuration for a light asset in the simulation.

    Supports six light types matching the dexsim rendering backend:

    - ``"point"``: Per-environment omnidirectional point light with position
      and falloff radius. Created as a batched light (one per environment).
    - ``"sun"``: Global directional sun light (infinite distance). Created as
      a single scene-level instance. Uses direction only; position is ignored.
      Sun-specific fields (``angular_radius``, ``halo_size``, ``halo_falloff``)
      are reserved for future backend support.
    - ``"direction"``: Global pure directional light at infinite distance.
      Created as a single scene-level instance. Direction only; no position.
    - ``"spot"``: Per-environment spotlight with position, direction, and
      inner/outer cone angles. Created as a batched light.
    - ``"rect"``: Per-environment rectangular area light with position,
      direction, width, and height. Created as a batched light.
    - ``"mesh"``: Per-environment mesh-based emissive light. Requires a
      :class:`~dexsim.models.MeshObject` via
      :meth:`embodichain.lab.sim.objects.light.Light.set_mesh`
      (not tensor-batched). Created as a batched light.

    .. attention::
        The ``angular_radius``, ``halo_size``, and ``halo_falloff`` fields are
        reserved for future use. The dexsim Python bindings do not yet expose
        setters for these sun-specific properties.
    """

    light_type: Literal["point", "sun", "direction", "spot", "rect", "mesh"] = "point"
    """Light type. Supported: ``"point"``, ``"sun"``, ``"direction"``, ``"spot"``, ``"rect"``, ``"mesh"``."""

    # ------------------------------------------------------------------
    # Universal properties (apply to all light types)
    # ------------------------------------------------------------------

    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """RGB color of the light source. Defaults to white ``(1.0, 1.0, 1.0)``."""

    intensity: float = 30.0
    """Intensity of the light source in watts/m^2. Defaults to ``30.0``."""

    enable_shadow: bool = True
    """Whether the light casts shadows. Defaults to ``True``."""

    # ------------------------------------------------------------------
    # Point light
    # ------------------------------------------------------------------

    radius: float = 10.0
    """Falloff radius for point lights. Only used when ``light_type="point"``. Defaults to ``10.0``."""

    # ------------------------------------------------------------------
    # Directional properties (sun, direction, spot, rect, mesh)
    # ------------------------------------------------------------------

    direction: tuple[float, float, float] = (0.0, 0.0, -1.0)
    """Direction vector for directional, spot, rect, and mesh lights.
    Defaults to ``(0.0, 0.0, -1.0)`` (pointing down along -Z)."""

    # ------------------------------------------------------------------
    # Sun light (reserved — Python bindings not yet available)
    # ------------------------------------------------------------------

    angular_radius: float = 0.5
    """Angular radius of the sun disc in degrees. Reserved for future use."""

    halo_size: float = 10.0
    """Halo size for sun light. Reserved for future use."""

    halo_falloff: float = 3.0
    """Halo falloff for sun light. Reserved for future use."""

    # ------------------------------------------------------------------
    # Spot light
    # ------------------------------------------------------------------

    spot_angle_inner: float = 30.0
    """Inner cone angle of the spotlight in degrees. Only used when ``light_type="spot"``.
    Defaults to ``30.0``."""

    spot_angle_outer: float = 45.0
    """Outer cone angle of the spotlight in degrees. Only used when ``light_type="spot"``.
    Defaults to ``45.0``."""

    # ------------------------------------------------------------------
    # Rect light
    # ------------------------------------------------------------------

    rect_width: float = 1.0
    """Width of the rectangular area light. Only used when ``light_type="rect"``.
    Defaults to ``1.0``."""

    rect_height: float = 1.0
    """Height of the rectangular area light. Only used when ``light_type="rect"``.
    Defaults to ``1.0``."""

    # ------------------------------------------------------------------
    # Mesh light
    # ------------------------------------------------------------------

    mesh_path: str = ""
    """Asset path for mesh-based emissive lights. Only used when ``light_type="mesh"``.
    The actual mesh assignment is done via
    :meth:`embodichain.lab.sim.objects.light.Light.set_mesh` which accepts a
    :class:`dexsim.models.MeshObject`. This field stores the path for reference."""


@configclass
class RigidConstraintCfg:
    """Configuration for a rigid constraint between object or link endpoints.

    The constraint binds rigid_object_a's entity[i] to rigid_object_b's entity[i]
    within arena[i] (one constraint per arena).

    Args:
        name: Base constraint name. Per-arena names are derived as ``f"{name}"``
            (single env) or ``f"{name}_{i}"`` (multi env).
        rigid_object_a_uid: UID of the first object, or articulation when
            ``articulation_link_a`` is provided.
        rigid_object_b_uid: UID of the second object, or articulation when
            ``articulation_link_b`` is provided.
        articulation_link_a: Physical link name on the first articulation.
        articulation_link_b: Physical link name on the second articulation.
        local_frame_a: 4x4 joint frame in object A's local coordinates.
            ``None`` -> identity (object A's origin). Accepts a single
            ``(4, 4)`` matrix (shared by all envs) or an ``(N, 4, 4)`` array
            (one frame per env). Defaults to None.
        local_frame_b: 4x4 joint frame in object B's local coordinates.
            ``None`` -> the frame is computed per env as ``inv(pose_B) @ pose_A``
            from the objects' current poses, so the constraint welds the objects
            at their *current* relative pose (rather than pulling their origins
            together). An explicit ``(4, 4)`` or ``(N, 4, 4)`` value is used
            verbatim. Defaults to None.
        constraint_type: ``fixed`` locks all relative motion; ``spherical``
            connects the two joint-frame origins with free rotation. Spherical
            constraints use the SDK's D6 rigid-body endpoint factory, including
            in Direct GPU scenes.

    .. attention::
        Endpoints must share the same number of arenas and the Default backend.
    """

    name: str = MISSING
    """Base name of the constraint (per-arena names are derived from this)."""

    rigid_object_a_uid: str = MISSING
    """UID of the first RigidObject."""

    rigid_object_b_uid: str = MISSING
    """UID of the second RigidObject."""

    articulation_link_a: str | None = None
    """Physical link on the first endpoint, or None for a rigid object."""

    articulation_link_b: str | None = None
    """Physical link on the second endpoint, or None for a rigid object."""

    local_frame_a: np.ndarray | None = None
    """Local joint frame on object A. None -> identity (object A's origin)."""

    local_frame_b: np.ndarray | None = None
    """Local joint frame on object B. None -> ``inv(pose_B) @ pose_A`` per env
    (weld at the objects' current relative pose)."""

    constraint_type: Literal["fixed", "spherical"] = "fixed"
    """Fixed attachment or point connection with free rotation."""

    linear_drive_stiffness: float | None = None
    """Optional positive D6 linear spring gain for a compliant point connection."""

    linear_drive_damping: float | None = None
    """Nonnegative D6 linear damping, required with linear_drive_stiffness."""

    linear_drive_acceleration: bool = False
    """Use native mass-normalized gains rather than N/m and N*s/m."""

    linear_drive_velocity: Sequence[float] | None = None
    """Optional compliant-point velocity target [m/s], with three components.

    Describes endpoint B relative to A in A's joint frame. Requires the linear
    spring/damping pair and SDK velocity-target support. ``None`` retains the
    existing zero-target drive behavior.
    """

    linear_drive_point_anchors: bool = False
    """Apply compliant-point moments at each endpoint's own joint origin.

    Requires linear spring/damping gains and a matching Default GPU SDK.
    ``False`` keeps ordinary D6 geometry, which uses endpoint B's joint origin
    for both moments. Runtime gain updates retain the selected geometry.
    """
