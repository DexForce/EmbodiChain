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

"""Geometry-derived support clearance for fixed-base E8 deployments."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import fields
import math
from typing import Any, TYPE_CHECKING

import numpy as np

from ..scene.articulation_geometry import (
    _fixed_base,
    _root_pose,
    read_articulation_geometry,
)

if TYPE_CHECKING:
    from .twist_binding import KnobBinding

__all__: list[str] = []

SUPPORT_NUMERICAL_CLEARANCE = 1e-4


class TwistSupportPenetrationError(ValueError):
    """A bounded lift candidate leaves moving collision geometry below support."""


def revolute_swept_min_z(
    points: np.ndarray,
    rotation: np.ndarray,
    origin: np.ndarray,
    axis: np.ndarray,
    limits: tuple[float, float],
    scale: float,
) -> float:
    """Minimize world Z over all vertices and the complete revolute interval.

    Points, origin, and axis use the unscaled native root frame. ``rotation``
    maps that frame to world; translation is deliberately excluded. Rodrigues'
    formula reduces each vertex height to ``C + A*cos(q) + B*sin(q)``. Both
    interval endpoints and every applicable sinusoid minimum are considered,
    so tilted deployments cannot hide collisions between sampled angles.
    """
    points, rotation, origin, axis = (
        np.asarray(value, dtype=float) for value in (points, rotation, origin, axis)
    )
    bounds = np.asarray(limits, dtype=float)
    if (
        points.ndim != 2
        or points.shape[1:] != (3,)
        or not len(points)
        or not np.isfinite(points).all()
    ):
        raise ValueError("E8 support requires finite moving collision vertices.")
    if (
        rotation.shape != (3, 3)
        or not np.isfinite(rotation).all()
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6, rtol=0)
    ):
        raise ValueError("E8 support requires a rigid root rotation.")
    if (
        origin.shape != (3,)
        or axis.shape != (3,)
        or not np.isfinite(origin).all()
        or not np.isfinite(axis).all()
        or np.linalg.norm(axis) <= 0
    ):
        raise ValueError("E8 support requires a finite pivot and nonzero axis.")
    if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] > bounds[1]:
        raise ValueError("E8 support requires finite ordered joint limits.")
    if isinstance(scale, bool) or not math.isfinite(float(scale)) or scale <= 0:
        raise ValueError("E8 support requires a positive uniform scale.")

    unit_axis = axis / np.linalg.norm(axis)
    offsets = points - origin
    parallel = (offsets @ unit_axis)[:, None] * unit_axis
    world_up = rotation[2]
    constant = (parallel + origin) @ world_up
    cosine = (offsets - parallel) @ world_up
    sine = np.cross(unit_axis, offsets) @ world_up
    low, high = bounds
    values = np.minimum(
        cosine * np.cos(low) + sine * np.sin(low),
        cosine * np.cos(high) + sine * np.sin(high),
    )
    critical = np.arctan2(sine, cosine) + math.pi
    # Select the first periodic minimum at or above the interval's lower bound.
    first_minimum = critical + 2 * math.pi * np.ceil((low - critical) / (2 * math.pi))
    reaches_minimum = (high - low >= 2 * math.pi) | (first_minimum <= high)
    values = np.where(reaches_minimum, -np.hypot(cosine, sine), values)
    return float(np.min(constant + values) * scale)


def _validate_envelope(contact: float, rest: float) -> tuple[float, float]:
    contact, rest = float(contact), float(rest)
    if not math.isfinite(contact) or not math.isfinite(rest) or contact < max(0, rest):
        raise ValueError("E8 support requires finite valid contact/rest offsets.")
    return contact, rest


def _moving_collision_envelopes(
    config: Mapping[str, Any], link_name: str
) -> list[dict[str, Any]]:
    from dexsim.kit.usd.parser import parse_usd
    from dexsim.types import PhysicalAttr

    from embodichain.lab.sim.cfg.articulation import ArticulationCfg
    from embodichain.lab.sim.spawn.descriptors import configure_articulation_desc

    source = parse_usd(str(config["fpath"]))
    if len(source.articulations) != 1:
        raise ValueError("E8 support requires one source articulation.")
    supported = {item.name for item in fields(ArticulationCfg)}
    canonical = {
        key: deepcopy(value) for key, value in config.items() if key in supported
    }
    if "fix_base" in config:
        canonical["root_props"] = {
            **canonical.get("root_props", {}),
            "fixed_base": config["fix_base"],
        }
    desc = configure_articulation_desc(
        source.articulations[0], ArticulationCfg.from_dict(canonical)
    )
    links = [link for link in desc.links if link.name == link_name]
    if len(links) != 1:
        raise ValueError("E8 support cannot resolve the moving link.")
    defaults = PhysicalAttr()
    envelopes = []
    for index, collision in enumerate(links[0].collisions):
        if collision.enable_collision is False:
            continue
        native = collision.dexsim
        contact, rest = _validate_envelope(
            (
                defaults.contact_offset
                if native is None or native.contact_offset is None
                else native.contact_offset
            ),
            (
                defaults.rest_offset
                if native is None or native.rest_offset is None
                else native.rest_offset
            ),
        )
        envelopes.append(
            {"collision_index": index, "contact_offset": contact, "rest_offset": rest}
        )
    if not envelopes:
        raise ValueError("E8 support requires enabled moving collision shapes.")
    return envelopes


def _table_collision_envelope(table: Mapping[str, Any]) -> tuple[float, float]:
    from embodichain.lab.sim.cfg.rigid import RigidBodyPhysicsCfg

    attrs = RigidBodyPhysicsCfg.from_dict(dict(table.get("attrs") or {}))
    if not attrs.enable_collision:
        raise ValueError("E8 support requires a collision-enabled table.")
    native = attrs.to_dexsim_physical_attr()
    return _validate_envelope(native.contact_offset, native.rest_offset)


def adjust_twist_support(
    config: Mapping[str, Any],
    binding: KnobBinding,
    table: Mapping[str, Any],
    *,
    table_top_z: float,
    requested_total_lift_m: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Lift a deployment copy enough to keep its moving link off the support.

    The caller supplies the already measured tabletop height. Only root Z is
    changed; source assets, rotation, scale, joint limits and physics remain
    unchanged. Contact offsets bound a no-contact placement, not penetration
    depth or the pair's equilibrium rest separation. An explicit 1--4 mm total
    lift is a geometrically screened runtime candidate, not a no-contact
    certificate; its conservative reference and actual gap are both recorded.
    """
    if not _fixed_base(config):
        raise ValueError("E8 support adjustment requires a fixed base.")
    if config.get("uid") != binding.object_id:
        raise ValueError("E8 support binding does not match the articulation.")
    if isinstance(table_top_z, bool) or not math.isfinite(float(table_top_z)):
        raise ValueError("E8 support requires a measured finite tabletop height.")
    if requested_total_lift_m is not None and (
        type(requested_total_lift_m) not in (int, float)
        or not math.isfinite(requested_total_lift_m)
        or not any(
            math.isclose(requested_total_lift_m, value / 1000, abs_tol=1e-10)
            for value in (1, 2, 3, 4)
        )
    ):
        raise ValueError("E8 requested total lift must be 1, 2, 3 or 4 mm.")
    scale = np.asarray(config.get("body_scale", [1.0] * 3), dtype=float)
    if (
        scale.shape != (3,)
        or not np.isfinite(scale).all()
        or (scale <= 0).any()
        or not np.allclose(scale, scale[0])
        or not math.isfinite(binding.scale)
        or binding.scale <= 0
        or binding.axis_sign not in (-1.0, 1.0)
    ):
        raise ValueError("E8 support requires positive uniform calibrated scale.")
    geometry = read_articulation_geometry(config["fpath"])
    points = geometry.link_vertices.get(binding.link)
    link_pose = geometry.link_poses.get(binding.link)
    if points is None or link_pose is None:
        raise ValueError("E8 support has no measured moving collision geometry.")
    # Binding coordinates are scaled child-link-local; geometry is unscaled root-local.
    origin = np.asarray(binding.origin) / binding.scale
    origin = link_pose[:3, :3] @ origin + link_pose[:3, 3]
    axis = link_pose[:3, :3] @ (np.asarray(binding.axis) * binding.axis_sign)
    pose = _root_pose(config).copy()
    minimum_z = revolute_swept_min_z(
        points, pose[:3, :3], origin, axis, binding.limits, float(scale[0])
    ) + float(pose[2, 3])
    # E8's generated runtime uses overlay even when PreparedScene has no mode.
    envelopes = _moving_collision_envelopes(
        {**config, "asset_physics_mode": "overlay"}, binding.link
    )
    moving_offset = max(item["contact_offset"] for item in envelopes)
    table_offset, table_rest = _table_collision_envelope(table)
    required_gap = table_offset + moving_offset + SUPPORT_NUMERICAL_CLEARANCE
    gap_before = minimum_z - float(table_top_z)
    conservative_lift = max(0.0, required_gap - gap_before)
    lift = (
        conservative_lift
        if requested_total_lift_m is None
        else float(requested_total_lift_m)
    )
    if gap_before + lift < -1e-8:
        raise TwistSupportPenetrationError(
            "E8 requested total lift leaves swept geometry below the table."
        )
    before = pose[:3, 3].copy()
    pose[2, 3] += lift
    adjusted = deepcopy(dict(config))
    adjusted["init_pos"] = pose[:3, 3].tolist()
    adjusted["init_local_pose"] = pose.tolist()
    audit = {
        "policy": (
            "moving_link_swept_no_support_contact"
            if requested_total_lift_m is None
            else "bounded_total_lift_geometric_candidate"
        ),
        "object_id": binding.object_id,
        "joint": binding.joint,
        "moving_link": binding.link,
        "source_sha256": binding.source_sha256,
        "source_edited": False,
        "geometry_frame": "unscaled_native_root_link",
        "height_frame": "runtime_world_z_up_meters",
        "body_scale": scale.tolist(),
        "joint_limits": list(binding.limits),
        "pivot_root": origin.tolist(),
        "joint_axis_root": axis.tolist(),
        "moving_vertices_root_bounds": np.stack(
            (points.min(0), points.max(0))
        ).tolist(),
        "moving_collision_envelopes": envelopes,
        "runtime_asset_physics_mode": "overlay",
        "table_id": table.get("uid"),
        "table_top_z": float(table_top_z),
        "table_contact_offset": table_offset,
        "table_rest_offset": table_rest,
        "moving_max_contact_offset": moving_offset,
        "numerical_clearance": SUPPORT_NUMERICAL_CLEARANCE,
        "required_no_contact_gap": required_gap,
        "conservative_translation_z": conservative_lift,
        "requested_total_lift_m": requested_total_lift_m,
        "geometric_penetration_free": True,
        "no_contact_envelope_satisfied": gap_before + lift >= required_gap - 1e-8,
        "qualification": "runtime_initial_stability_and_terminal_checks_required",
        "moving_swept_min_z_before": minimum_z,
        "moving_swept_min_z_after": minimum_z + lift,
        "support_gap_before": gap_before,
        "support_gap_after": gap_before + lift,
        "translation_z": lift,
        "original_init_pos": before.tolist(),
        "adapted_init_pos": pose[:3, 3].tolist(),
    }
    return adjusted, audit
