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

"""Native Marvin source placement and explicit contact overlays, without simulation."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
from pathlib import Path
from typing import Any

__all__ = [
    "source_tianji_mount_declaration",
    "source_tianji_contact_pad_declaration",
    "source_tianji_lateral_positions",
    "require_tianji_task_assets",
]


def source_tianji_mount_declaration(
    embodiment: Mapping[str, Any], *, table_top_z: float, table_bounds: Any
) -> dict[str, Any]:
    """Place the native floor-standing body outside explicit source table bounds.

    This source deployment owns a half-turn toward the scene, a 20 mm bench
    reserve and a 2 mm floor reserve. Native hand/TCP/solver/drive declarations
    are unchanged; the shared component remains untouched. No Robot or IK is made.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation
    from embodichain.lab.sim.scene.source_robot_geometry import source_fixed_frame_pose
    from embodichain.lab.sim.utility.urdf_collision import load_urdf_collision_inputs

    if embodiment.get("embodiment_id") != "tianji_marvin" or not math.isfinite(
        float(table_top_z)
    ):
        raise ValueError(
            "Native Marvin source placement requires its own component and measured table."
        )
    bounds = np.asarray(table_bounds, dtype=float)
    if (
        bounds.shape != (2, 3)
        or not np.isfinite(bounds).all()
        or (bounds[0, :2] >= bounds[1, :2]).any()
        or bounds[0, 2] > bounds[1, 2]
    ):
        raise ValueError("Marvin floor deployment requires measured table bounds.")
    owned = deepcopy(dict(embodiment))
    simulation = owned["simulation"]
    if simulation.get("init_local_pose") is not None:
        raise ValueError(
            "Source floor placement cannot override an explicitly owned local pose matrix."
        )
    path = Path(simulation["fpath"])
    body = load_urdf_collision_inputs(
        path,
        ("base_link",),
        body_scale=simulation.get("body_scale", (1.0, 1.0, 1.0)),
        vertices_only=True,
    )["base_link"]
    vertices = np.concatenate([part.vertices for part in body])
    body_pose = source_fixed_frame_pose(path, "base_link")
    vertices = vertices @ body_pose[:3, :3].T + body_pose[:3, 3]
    simulation["init_rot"] = [0.0, 0.0, 180.0]
    rotation = Rotation.from_euler(
        "XYZ", simulation["init_rot"], degrees=True
    ).as_matrix()
    vertices = vertices @ rotation.T
    simulation["init_pos"] = [
        float(bounds[1, 0] - vertices[:, 0].min() + 0.02),
        float(simulation.get("init_pos", [0.0, 0.0, 0.0])[1]),
        float(-vertices[:, 2].min() + 0.002),
    ]
    return owned


def source_tianji_contact_pad_declaration(
    embodiment: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply only an explicitly selected contact experiment to the native pads."""
    from .task_program import tianji_marvin_task_hand_calibrations

    owned = deepcopy(dict(embodiment))
    hands = tianji_marvin_task_hand_calibrations(owned)
    overlays = owned["simulation"].setdefault("link_attrs", {})
    if "gen_sim_contact_pads" in overlays:
        raise ValueError("The contact pad material overlay is already declared.")
    overlays["gen_sim_contact_pads"] = dict(
        link_names_expr=[name for hand in hands for name in hand.pad_link_names],
        attrs=dict(material_props=dict(static_friction=2.85, dynamic_friction=2.70)),
    )
    owned["simulation"]["asset_physics_mode"] = "overlay"
    return owned


def source_tianji_lateral_positions(
    embodiment: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Read actual native shoulder frames for semantic world-Y resource hints."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    from embodichain.lab.sim.scene.source_robot_geometry import source_arm_base_position
    from .cfg import TianjiMarvinCfg

    if embodiment is None:
        cfg = TianjiMarvinCfg.from_dict({})
        simulation = cfg.to_dict()
        simulation["init_rot"] = [0.0, 0.0, 180.0]
        parts = {side: f"{side}_arm" for side in ("left", "right")}
    else:
        if embodiment.get("embodiment_id") != "tianji_marvin":
            raise ValueError(
                "Native lateral hints require the standard Marvin component."
            )
        simulation = embodiment["simulation"]
        parts = {
            resource["resource_id"]: next(
                endpoint["control_part"]
                for endpoint in resource["endpoints"]
                if endpoint["endpoint_id"] == "motion"
            )
            for resource in embodiment["skill_profile"]["resources"]
        }
    if simulation.get("init_local_pose") is not None:
        pose = np.asarray(simulation["init_local_pose"], dtype=float)
    else:
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_euler(
            "XYZ", simulation.get("init_rot", [0.0, 0.0, 0.0]), degrees=True
        ).as_matrix()
        pose[:3, 3] = simulation.get("init_pos", [0.0, 0.0, 0.0])
    return {
        side: float(
            (
                pose[:3, :3] @ source_arm_base_position(simulation, parts[side])
                + pose[:3, 3]
            )[1]
        )
        for side in ("left", "right")
    }


def require_tianji_task_assets() -> None:
    """Preflight only the existing native full robot asset and configuration."""
    from embodichain.data import get_data_path
    from .cfg import TianjiMarvinCfg

    path = Path(get_data_path("TianjiMarvin/robot_with_ee_acd.urdf"))
    if not path.is_file():
        raise ValueError(
            "Standard Marvin requires its registered native gripper asset."
        )
    TianjiMarvinCfg.from_dict({"fpath": str(path), "with_gripper": True}).validate()
