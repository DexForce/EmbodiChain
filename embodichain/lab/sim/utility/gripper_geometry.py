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
    from embodichain.toolkits.graspkit.pg_grasp.fixed_geometry import (
        FixedGripperCollisionGeometry,
    )

__all__ = [
    "attach_closure_samples",
    "build_gripper_collision_geometry",
    "grasp_generator_parameters",
]


def _closure_samples(
    geometry: "FixedGripperCollisionGeometry",
    defaults: Mapping[str, object],
    *,
    samples: int,
) -> tuple[object, object] | tuple[None, None]:
    """Sample pad surfaces at closing apertures between open and grasp commands."""
    import numpy as np
    import torch

    pad_links = tuple(defaults.get("pad_links") or ())
    if len(pad_links) != 2:
        return None, None
    open_state = dict(defaults["joint_commands"]["open"])
    grasp_state = dict(defaults["joint_commands"]["grasp"])
    if set(grasp_state) != set(open_state):
        raise ValueError("Open and grasp commands must bind the same hand joints.")
    offsets: dict[str, tuple[int, int]] = {}
    start = 0
    for part in geometry.provenance["parts"]:
        offsets[part["link"]] = (start, int(part["surface_point_count"]))
        start += int(part["surface_point_count"])
    missing = [link for link in pad_links if link not in offsets]
    if missing:
        raise ValueError(f"Pad links are not sampled collision links: {missing}.")
    left, right = pad_links
    ladder: list[tuple[float, torch.Tensor]] = []
    for fraction in np.linspace(0.0, 1.0, samples):
        state = {
            name: (1.0 - fraction) * float(open_state[name])
            + fraction * float(grasp_state[name])
            for name in open_state
        }
        points = geometry.surface_points_at(state)
        pads = torch.cat(
            [
                points[offsets[left][0] : offsets[left][0] + offsets[left][1]],
                points[offsets[right][0] : offsets[right][0] + offsets[right][1]],
            ]
        )
        aperture = float(
            points[offsets[right][0] : offsets[right][0] + offsets[right][1], 0].min()
            - points[offsets[left][0] : offsets[left][0] + offsets[left][1], 0].max()
        )
        if aperture > 0.0:
            ladder.append((aperture, pads))
    if len(ladder) < 2:
        raise ValueError("Closure sweep requires at least two positive apertures.")
    ladder.sort(key=lambda item: item[0])
    apertures = torch.tensor([item[0] for item in ladder], dtype=geometry.points.dtype)
    surfaces = torch.stack([item[1] for item in ladder])
    return apertures, surfaces


def build_gripper_collision_geometry(
    defaults: Mapping[str, object],
    *,
    point_spacing: float = 0.012,
    root_from_tcp: object | None = None,
    root_from_contact_frame: object | None = None,
    closure_sweep: bool = False,
    closure_samples: int = 65,
) -> FixedGripperCollisionGeometry:
    """Build owned open-hand surfaces using caller-selected native model data.

    Args:
        defaults: Complete primitive gripper description, including source asset,
            native commands and default_root_from_tcp. Optional qpos_limits use
            native joint names/regexes, or rows in the native eef joint order.
        point_spacing: Maximum triangle sampling edge in metres.
        root_from_tcp: Explicit runtime TCP relative to the native hand root.
        root_from_contact_frame: Explicit contact-frame calibration, if selected.
        closure_sweep: Also sample pad surfaces at closing apertures so callers
            can qualify support clearance during closure.
        closure_samples: Number of closing states between the open and grasp
            commands. Apertures are sorted before they are stored.

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
    result = FixedGripperCollisionGeometry(
        geometry.points,
        {
            **geometry.provenance,
            "model_id": defaults["model_id"],
            "root_from_contact_frame": canonical.tolist(),
            "default_root_from_tcp": defaults["default_root_from_tcp"],
            "runtime_tcp_overridden": root_from_tcp is not None,
        },
    )
    if closure_sweep:
        result = attach_closure_samples(result, defaults, samples=closure_samples)
    return result


def attach_closure_samples(
    geometry: FixedGripperCollisionGeometry,
    defaults: Mapping[str, object],
    *,
    samples: int = 65,
) -> FixedGripperCollisionGeometry:
    """Attach pad surfaces sampled at closing apertures to a built geometry.

    Args:
        geometry: Geometry whose canonical frame and sampled parts are retained.
        defaults: Native gripper description owning the commands and pad links.
        samples: Number of closing states between the open and grasp commands.

    Returns:
        A geometry with closure apertures and pad surfaces attached, or the
        unchanged geometry when the description declares no pad pair.

    Raises:
        ValueError: If ``samples`` is less than two.
    """
    from embodichain.toolkits.graspkit.pg_grasp.fixed_geometry import (
        FixedGripperCollisionGeometry,
    )

    if samples < 2:
        raise ValueError("samples must be at least two.")
    apertures, points = _closure_samples(geometry, defaults, samples=samples)
    if apertures is None or points is None:
        return geometry
    return FixedGripperCollisionGeometry(
        geometry.points,
        {**geometry.provenance, "closure_sweep_samples": len(apertures)},
        closure_apertures=apertures,
        closure_points=points,
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
