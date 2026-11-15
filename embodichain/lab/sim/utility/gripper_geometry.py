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

"""Build geometry from an explicit physical description, without model lookup."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from embodichain.toolkits.graspkit.pg_grasp.contact_fit import SampledHandGeometry
    from embodichain.toolkits.graspkit.pg_grasp.fixed_geometry import (
        FixedGripperCollisionGeometry,
    )

__all__ = [
    "build_gripper_collision_geometry",
    "build_gripper_closing_geometry",
    "grasp_generator_parameters",
]


def build_gripper_closing_geometry(
    defaults: Mapping[str, object],
    *,
    sample_count: int,
    point_spacing: float = 0.012,
) -> SampledHandGeometry:
    """Sample model-owned hand commands in their canonical contact frame.

    Args:
        defaults: Explicit calibrated model data, including the two pad links
            and native named open/grasp commands.
        sample_count: Number of uniformly spaced command samples, at least two.
        point_spacing: Collision-surface sampling spacing in metres.

    Returns:
        Owned CPU body/pad surfaces and measured pad gaps. These describe
        nominal kinematic closure, not runtime contact or physical attachment.
        The shared fixed-geometry loader retains URDF mimic and limit checks.
    """
    import numpy as np
    import torch
    from embodichain.toolkits.graspkit.pg_grasp.contact_fit import SampledHandGeometry
    from embodichain.toolkits.graspkit.pg_grasp.fixed_geometry import (
        FixedGripperCollisionGeometry,
    )

    if type(sample_count) is not int or sample_count < 2:
        raise ValueError("sample_count must be an integer of at least two.")
    opened = dict(defaults["joint_commands"]["open"])
    closed = dict(defaults["joint_commands"]["grasp"])
    if opened.keys() != closed.keys():
        raise ValueError("Closing geometry needs matching named hand commands.")
    pad_names = tuple(defaults["pad_links"])
    if len(pad_names) != 2 or len(set(pad_names)) != 2:
        raise ValueError("Closing geometry requires two distinct pad links.")
    initial = build_gripper_collision_geometry(defaults, point_spacing=point_spacing)
    source = initial.provenance
    bodies, pads = [], []
    for index, fraction in enumerate(np.linspace(0, 1, sample_count)):
        geometry = (
            initial
            if index == 0
            else FixedGripperCollisionGeometry.from_urdf(
                source["urdf_path"],
                root_link=source["root_link"],
                tcp_link=source["tcp_link"],
                collision_links=tuple(defaults["collision_links"]),
                joint_positions={
                    name: float(value + fraction * (closed[name] - value))
                    for name, value in opened.items()
                },
                joint_limits=source["joint_limits"],
                grasp_from_tcp=np.asarray(source["grasp_from_tcp"]),
                point_spacing=point_spacing,
                link_from_tcp=np.asarray(source["link_from_tcp"]),
                mesh_paths=source["mesh_paths"],
            )
        )
        per_link = {}
        offset = 0
        for part in geometry.provenance["parts"]:
            count = part["surface_point_count"]
            per_link.setdefault(part["link"], []).append(
                geometry.points[offset : offset + count]
            )
            offset += count
        per_link = {name: torch.cat(parts) for name, parts in per_link.items()}
        if index == 0:
            pad_names = tuple(
                sorted(pad_names, key=lambda name: float(per_link[name][:, 0].mean()))
            )
        pad_points = [per_link[name] for name in pad_names]
        maximum = max(map(len, pad_points))
        # Repeated vertices preserve minima and pad geometry for unequal meshes.
        pad_points = [
            torch.cat((points, points[-1:].expand(maximum - len(points), -1)))
            for points in pad_points
        ]
        pads.append(torch.stack(pad_points))
        bodies.append(
            torch.cat(
                [points for name, points in per_link.items() if name not in pad_names]
            )
        )
    pad_tensor = torch.stack(pads)
    gaps = pad_tensor[:, 1, :, 0].amin(-1) - pad_tensor[:, 0, :, 0].amax(-1)
    return SampledHandGeometry(torch.stack(bodies), pad_tensor, gaps)


def build_gripper_collision_geometry(
    defaults: Mapping[str, object],
    *,
    point_spacing: float = 0.012,
    root_from_tcp: object | None = None,
    root_from_contact_frame: object | None = None,
) -> FixedGripperCollisionGeometry:
    """Build owned open-hand surfaces using caller-selected native model data.

    Args:
        defaults: Complete primitive gripper description, including source asset,
            native commands and default_root_from_tcp. Optional qpos_limits use
            native joint names/regexes, or rows in the native eef joint order.
        point_spacing: Maximum triangle sampling edge in metres.
        root_from_tcp: Explicit runtime TCP relative to the native hand root.
        root_from_contact_frame: Explicit contact-frame calibration, if selected.

    Returns:
        Independent collision geometry converting canonical candidates to TCP.
    """
    from embodichain.data import get_data_path
    from embodichain.toolkits.graspkit.pg_grasp.fixed_geometry import (
        FixedGripperCollisionGeometry,
        _rigid_transform,
    )
    import numpy as np

    if not defaults["root_link"] or not tuple(defaults["collision_links"]):
        raise ValueError("Explicit root_link and collision_links are required.")
    canonical = _rigid_transform(
        (
            defaults["root_from_contact_frame"]
            if root_from_contact_frame is None
            else root_from_contact_frame
        ),
        error_message="root_from_contact_frame must be a proper rigid transform.",
    )
    tcp = _rigid_transform(
        defaults["default_root_from_tcp"] if root_from_tcp is None else root_from_tcp,
        error_message="root_from_tcp must be a proper rigid transform.",
    )
    from pathlib import Path
    import hashlib

    source_path = get_data_path(defaults["fpath"])
    if (
        defaults.get("source_sha256") is not None
        and hashlib.sha256(Path(source_path).read_bytes()).hexdigest()
        != defaults["source_sha256"]
    ):
        raise ValueError("Gripper source asset differs from its pinned description.")
    joint_limits = defaults.get("qpos_limits")
    if isinstance(joint_limits, Mapping):
        from embodichain.utils.string import resolve_matching_names_values

        _, names, bounds = resolve_matching_names_values(
            dict(joint_limits), list(defaults["joint_commands"]["open"])
        )
        joint_limits = dict(zip(names, bounds))
    elif joint_limits is not None:
        names = defaults["control_parts"]["eef"]
        bounds = np.asarray(joint_limits, dtype=float)
        if bounds.shape != (len(names), 2):
            raise ValueError("qpos_limits must have native hand shape (num_dofs, 2).")
        joint_limits = dict(zip(names, bounds.tolist()))
    geometry = FixedGripperCollisionGeometry.from_urdf(
        source_path,
        root_link=defaults["root_link"],
        tcp_link=defaults["root_link"],
        link_from_tcp=tcp,
        collision_links=tuple(defaults["collision_links"]),
        joint_positions=dict(defaults["joint_commands"]["open"]),
        joint_limits=joint_limits,
        grasp_from_tcp=np.linalg.inv(canonical) @ tcp,
        point_spacing=point_spacing,
        mesh_paths={
            k: get_data_path(v) for k, v in defaults.get("mesh_paths", {}).items()
        }
        or None,
    )
    return FixedGripperCollisionGeometry(
        geometry.points,
        {
            **geometry.provenance,
            "model_id": defaults["model_id"],
            "root_from_contact_frame": canonical.tolist(),
            "default_root_from_tcp": defaults["default_root_from_tcp"],
            "runtime_tcp_overridden": root_from_tcp is not None,
        },
    )


def grasp_generator_parameters(data: Mapping[str, object]) -> dict[str, object]:
    """Project an owned numeric algorithm input from a resolved physical hand.

    Args:
        data: Caller-selected primitive hand data with calibrated opening bounds.

    Returns:
        Scalar generator inputs; assets, topology and runtime objects are excluded.
    """
    return {
        key: data[key] for key in ("model_id", "min_opening_width", "max_opening_width")
    }
