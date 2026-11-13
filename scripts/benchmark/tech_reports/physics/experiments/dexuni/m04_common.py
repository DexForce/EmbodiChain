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
"""M04: one physical W1/Pika T-shirt fold using the requested demo garment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "CONFIG",
    "OUTPUT",
    "REFERENCE",
    "ASSETS",
    "digest",
    "write_json",
    "supports",
    "shirt_mesh",
    "motion",
    "phase",
    "orientation",
]
TECH = Path(__file__).resolve().parents[3]
OUTPUT = (
    TECH.parents[2]
    / "outputs/benchmarks/tech_reports/dexuni/m04_bimanual_tshirt_single_fold"
)
REFERENCE = Path(
    "/home/oem/code/repos/dexsimforembodychain/dexsim/modules/dexsim/examples/python/physics/multiphys/demo_dexuni_w1_bimanual_fold_tshirt_waic_house.py"
)
ASSETS = Path("/tmp/dexuni_w1_bimanual_fold_tshirt_waic_house_assets/shirt")
CONFIG: dict[str, Any] = {
    "task": "M04",
    "description": "W1/Pika folds the demo T-shirt once, hem toward neckline; one-way kinematic surface-fast",
    "robot": {
        "urdf": "/home/oem/.cache/dexsim/resources/mjvbd_v2/w1_bag_packing/w1_v030/robot.urdf",
        "base_position": [0.08, 0.0, 0.0],
        "ik_seed": {
            "LEFT_J1": -0.07856255769729614,
            "LEFT_J2": -0.7473036050796509,
            "LEFT_J3": -0.07098092883825302,
            "LEFT_J4": -1.3992054462432861,
            "LEFT_J5": 0.9050648808479309,
            "LEFT_J6": -1.6580628156661987,
            "LEFT_J7": 0.14426538348197937,
            "RIGHT_J1": -0.07856231182813644,
            "RIGHT_J2": 0.7473055124282837,
            "RIGHT_J3": -0.07097947597503662,
            "RIGHT_J4": 1.3991999626159668,
            "RIGHT_J5": 0.9050433039665222,
            "RIGHT_J6": 1.6580628156661987,
            "RIGHT_J7": -0.14424188435077667,
        },
        "tcp_offset": [0.0, 0.0, 0.140],
        "open_finger": 0.025,
        "closed_finger": 0.006,
    },
    "support": {
        "table_z": 1.04,
        "table_center_x": 0.62,
        "table_size_xy": [0.64, 1.24],
    },
    "cloth": {
        "source_mesh": str(ASSETS / "shirt_mesh.txt"),
        "material_dir": "/home/oem/.cache/dexsim/resources/Material/FabricGray",
        "scale": 0.0064,
        "position": [0.54, 0.0, 1.054],
        "yaw_deg": -90.0,
        "surface_density": 0.02,
        "particle_radius": 0.008,
        "tri_ke": 1500.0,
        "tri_ka": 1500.0,
        "tri_kd": 1e-5,
        "edge_ke": 1.2,
        "edge_kd": 0.1,
        "edge_rest_angle": 0.0,
        "air_drag_rate": 8.0,
    },
    "contact": {
        "robot_ke": 9e5,
        "support_ke": 5e5,
        "particle_ke": 3e5,
        "particle_kd": 0.05,
        "particle_mu": 0.1,
        "table_mu": 0.1,
        "open_mu": 0.25,
        "grasp_mu": 2.3,
        "release_mu": 0.0,
        "release_ramp_s": 1.6,
        "gap": 0.0,
        "margin": 0.0,
        "soft_margin": 0.008,
    },
    "solver": {
        "dt": 1 / 60,
        "num_substeps": 10,
        "coupling": "one_way",
        "joint_mode": "kinematic",
        "contact_mode": "soft",
        "vbd_preset": "surface-fast",
        "use_cuda_graph": True,
        "self_contact_radius": 0.002,
        "self_contact_margin": 0.002,
        "vbd_options": {"iterations": 16},
    },
    "motion": {
        "pre_roll_s": 4.0,
        "pitch_deg": 90.0,
        "side_yaw_deg": 30.0,
        "roll_deg": 0.0,
        "grasp_x": 0.342,
        "grasp_y": 0.151,
        "right_grasp_y": 0.145,
        "grasp_above_table": 0.010,
        "left_grasp_above_table": 0.010,
        "place_x": 0.67,
        "place_y": 0.15,
        "place_above_table": 0.055,
        "fold_line_x": 0.525,
        "lift_above_table": 0.16,
        "apex_above_table": 0.18,
        "turn_above_table": 0.12,
        "phases": [
            [4.0, "pre-roll"],
            [6.0, "approach hem"],
            [8.0, "descend"],
            [9.0, "close"],
            [10.0, "grasp settle"],
            [14.0, "lift hem"],
            [18.0, "fold apex"],
            [21.0, "turn hem over"],
            [23.0, "lay folded hem"],
            [26.0, "hold folded hem"],
            [28.0, "release"],
            [29.5, "vertical retreat"],
            [30.0, "park"],
            [32.0, "observe"],
        ],
    },
    "candidate_criteria": {
        "max_grip_relative_drift_m": 0.015,
        "max_support_node_penetration_m": 0.0005,
        "max_final_hem_xy_error_m": 0.06,
        "min_lift_far_end_slide_m": 0.02,
        "max_lift_edge_length_ratio_p95": 1.08,
    },
}


def digest(path: Path) -> str:
    """Hash a reproducible input."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Save detached experiment data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            default=lambda v: np.asarray(v).tolist(),
        )
        + "\n"
    )


def supports(config: dict[str, Any]) -> list[tuple[str, tuple, tuple]]:
    """One fixed table; no moving stand or frustum."""
    c = config["support"]
    tz = c["table_z"]
    sx, sy = c["table_size_xy"]
    result = [("table", (c["table_center_x"], 0, tz - 0.025), (sx, sy, 0.05))]
    for i, x in enumerate(
        (c["table_center_x"] - sx / 2 + 0.03, c["table_center_x"] + sx / 2 - 0.03)
    ):
        for j, y in enumerate((-sy / 2 + 0.04, sy / 2 - 0.04)):
            result.append(
                (f"leg_{i}_{j}", (x, y, (tz - 0.05) / 2), (0.05, 0.05, tz - 0.05))
            )
    return result


def shirt_mesh(config: dict[str, Any]) -> dict[str, np.ndarray]:
    """Preserve all 6436 original vertices and 12736 original faces."""
    c = config["cloth"]
    vertices = []
    faces = []
    section = ""
    for line in Path(c["source_mesh"]).read_text().splitlines():
        line = line.strip()
        if line in ("shirt_vertices", "shirt_indices"):
            section = line
        elif ":" in line:
            values = line.split(":", 1)[1].split()[:3]
            if section == "shirt_vertices":
                vertices.append([float(v) for v in values])
            elif section == "shirt_indices":
                faces.append([int(v) for v in values])
    local = np.array(vertices, np.float32)
    faces = np.array(faces, np.int32)
    assert local.shape == (6436, 3) and faces.shape == (12736, 3)
    local[:, :2] -= (local[:, :2].min(0) + local[:, :2].max(0)) / 2
    local[:, 2] -= local[:, 2].min()
    local *= c["scale"]
    from scipy.spatial.transform import Rotation

    rot = Rotation.from_euler("z", c["yaw_deg"], degrees=True).as_matrix()
    world = local @ rot.T + np.array(c["position"])
    hem = np.flatnonzero(world[:, 0] < world[:, 0].min() + 0.015)
    neckline = np.flatnonzero(world[:, 0] > world[:, 0].max() - 0.05)
    return dict(
        vertices=local,
        triangles=faces,
        initial_positions=world.astype(np.float32),
        hem_ids=hem.astype(np.int32),
        neckline_ids=neckline.astype(np.int32),
    )


def phase(time: float, config: dict[str, Any]) -> str:
    """Read the one-fold schedule."""
    return next(
        (name for end, name in config["motion"]["phases"] if time <= end + 1e-9),
        "observe",
    )


def motion(time: float, config: dict[str, Any]) -> tuple[np.ndarray, float]:
    """One grasp and one hem-to-neckline fold, with quintic interpolation."""
    m = config["motion"]
    r = config["robot"]
    tz = config["support"]["table_z"]
    at = lambda x, y, z: np.array(
        ((x, y, z), (x, -m["right_grasp_y"] if abs(y - m["grasp_y"]) < 1e-8 else -y, z))
    )
    park = at(0.38, 0.22, 1.30)
    grasp = at(m["grasp_x"], m["grasp_y"], tz + m["grasp_above_table"])
    grasp[0, 2] = tz + m["left_grasp_above_table"]
    place = at(m["place_x"], m["place_y"], tz + m["place_above_table"])
    ends = {name: end for end, name in m["phases"]}
    o = r["open_finger"]
    closed = r["closed_finger"]
    keys = [
        (0, park, o),
        (ends["pre-roll"], park, o),
        (ends["approach hem"], grasp + [0, 0, 0.10], o),
        (ends["descend"], grasp, o),
        (ends["close"], grasp, closed),
        (ends["grasp settle"], grasp, closed),
        (
            ends["lift hem"],
            at(m["grasp_x"], m["grasp_y"], tz + m["lift_above_table"]),
            closed,
        ),
        (
            ends["fold apex"],
            at(m["fold_line_x"], m["grasp_y"], tz + m.get("apex_above_table", 0.21)),
            closed,
        ),
        (
            ends["turn hem over"],
            at(m["place_x"], m["place_y"], tz + m.get("turn_above_table", 0.22)),
            closed,
        ),
        (ends["lay folded hem"], place, closed),
        (ends["hold folded hem"], place, closed),
        (ends["release"], place, o),
        (ends["vertical retreat"], place + [0, 0, 0.16], o),
        (ends["park"], place + [0, 0, 0.16], o),
        (ends["observe"], place + [0, 0, 0.16], o),
    ]
    for a, b in zip(keys, keys[1:]):
        if time <= b[0] + 1e-9:
            u = float(np.clip((time - a[0]) / (b[0] - a[0]), 0, 1))
            u = u**3 * (10 - 15 * u + 6 * u * u)
            return (1 - u) * a[1] + u * b[1], float((1 - u) * a[2] + u * b[2])
    return keys[-1][1], keys[-1][2]


def orientation(time: float, config: dict[str, Any]) -> tuple[float, float]:
    """Keep the cloth near vertical until release; rotate empty wrists on retreat."""
    ends = {name: end for end, name in config["motion"]["phases"]}
    keys = [
        (0.0, 90.0),
        (ends["grasp settle"], 90.0),
        (ends["lift hem"], 60.0),
        (ends["release"], 60.0),
        (ends["vertical retreat"], 40.0),
        (ends["observe"], 40.0),
    ]
    yaw = config["motion"]["side_yaw_deg"]
    for a, b in zip(keys, keys[1:]):
        if time <= b[0] + 1e-9:
            u = float(np.clip((time - a[0]) / (b[0] - a[0]), 0, 1))
            u = u**3 * (10 - 15 * u + 6 * u * u)
            return (1 - u) * a[1] + u * b[1], yaw
    return keys[-1][1], yaw
