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
    from .fixed_geometry import FixedGripperCollisionGeometry

__all__ = ["build_gripper_collision_geometry"]


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
            grasp_cfg, native commands and default_root_from_tcp.
        point_spacing: Maximum triangle sampling edge in metres.
        root_from_tcp: Explicit runtime TCP relative to the native hand root.
        root_from_contact_frame: Explicit contact-frame calibration, if selected.

    Returns:
        Independent collision geometry converting canonical candidates to TCP.
    """
    from embodichain.data import get_data_path
    from embodichain.toolkits.graspkit.gripper_models import ParallelJawGripperModelCfg
    from .fixed_geometry import FixedGripperCollisionGeometry, _rigid_transform
    import numpy as np

    model = ParallelJawGripperModelCfg(**defaults["grasp_cfg"])
    if not model.root_link or not model.collision_links:
        raise ValueError("Explicit root_link and collision_links are required.")
    canonical = _rigid_transform(
        (
            model.root_from_contact_frame
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
    geometry = FixedGripperCollisionGeometry.from_urdf(
        source_path,
        root_link=model.root_link,
        tcp_link=model.root_link,
        link_from_tcp=tcp,
        collision_links=model.collision_links,
        joint_positions=dict(defaults["joint_commands"]["open"]),
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
            "model_id": model.model_id,
            "root_from_contact_frame": canonical.tolist(),
            "default_root_from_tcp": defaults["default_root_from_tcp"],
            "runtime_tcp_overridden": root_from_tcp is not None,
        },
    )
