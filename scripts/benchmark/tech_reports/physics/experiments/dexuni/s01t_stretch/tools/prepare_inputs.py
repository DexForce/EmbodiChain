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
"""Generate a NEW local S01T input snapshot; never write source assets.

Run: <dexsim-python> scripts/benchmark/tech_reports/physics/experiments/dexuni/s01t_stretch/tools/prepare_inputs.py --out-dir <new-cache-directory>
The authored JSON is a pilot starting point, not a qualified frozen config.
Use --config FILE for an alternative authored config; --skip-ik is geometry-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import sys

REPO_ROOT = Path(__file__).resolve().parents[8]
sys.path.insert(0, str(REPO_ROOT))
from scripts.benchmark.tech_reports.physics.experiments.dexuni.paths import (
    ASSETS,
    output_dir,
)

SUBTASK_DIR = Path(__file__).resolve().parents[1]
INPUTS = None
CONFIG = ASSETS / "s01t" / "s01t_config.json"


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Mesh and material lines
# --------------------------------------------------------------------------


def box_tet_mesh(size, cells):
    """Structured tet box centred at the origin (Kuhn 6-tet split).

    ``cells`` gives the cell count per axis. Every tet has positive volume.
    """
    size = np.asarray(size, dtype=np.float64)
    n = [int(c) for c in cells]
    axes = [np.linspace(-0.5 * s, 0.5 * s, k + 1) for s, k in zip(size, n)]
    vertices = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

    def vid(i, j, k):
        return (i * (n[1] + 1) + j) * (n[2] + 1) + k

    perms = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))
    tets = []
    for i in range(n[0]):
        for j in range(n[1]):
            for k in range(n[2]):
                for perm in perms:
                    corner = [i, j, k]
                    chain = [vid(*corner)]
                    for axis in perm:
                        corner[axis] += 1
                        chain.append(vid(*corner))
                    tets.append(chain)
    tets = np.asarray(tets, dtype=np.int32)
    x = vertices[tets]
    volume = np.einsum(
        "ij,ij->i", x[:, 1] - x[:, 0], np.cross(x[:, 2] - x[:, 0], x[:, 3] - x[:, 0])
    )
    flip = volume < 0
    tets[flip] = tets[flip][:, (0, 2, 1, 3)]
    return vertices, tets


def tet_surface_triangles(tets: np.ndarray) -> np.ndarray:
    faces = np.concatenate(
        [tets[:, (0, 2, 1)], tets[:, (0, 1, 3)], tets[:, (0, 3, 2)], tets[:, (1, 2, 3)]]
    )
    key = np.sort(faces, axis=1)
    _, inverse, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    return faces[counts[inverse.reshape(-1)] == 1].astype(np.int32)


def material_lines(config, local: np.ndarray) -> np.ndarray:
    """Ordered node ids of each material line, shape (lines, nodes)."""
    spec = config["material_lines"]
    centre_x = config["stretch_block"]["center"][0]
    x0, x1 = (v - centre_x for v in spec["x_range"])
    lines = []
    for y, z in spec["lines_yz"]:
        pick = np.flatnonzero(
            (np.abs(local[:, 1] - y) < 1e-6)
            & (np.abs(local[:, 2] - z) < 1e-6)
            & (local[:, 0] >= x0 - 1e-6)
            & (local[:, 0] <= x1 + 1e-6)
        )
        if len(pick) < 2:
            raise RuntimeError(f"material line ({y}, {z}) has no mesh nodes")
        lines.append(pick[np.argsort(local[pick, 0])])
    if len({len(line) for line in lines}) != 1:
        raise RuntimeError("material lines have different node counts")
    return np.asarray(lines, dtype=np.int32)


def write_mesh(config) -> dict:
    block = config["stretch_block"]
    local, tets = box_tet_mesh(block["size"], block["cells"])
    surface = tet_surface_triangles(tets)
    lines = material_lines(config, local)
    x = local[tets]
    volumes = (
        np.einsum(
            "ij,ij->i",
            x[:, 1] - x[:, 0],
            np.cross(x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]),
        )
        / 6.0
    )
    line_l0 = [
        float(np.linalg.norm(np.diff(local[l], axis=0), axis=1).sum()) for l in lines
    ]
    np.savez(
        INPUTS / "s01t_block_mesh.npz",
        local_vertices=local.astype(np.float32),
        tets=tets,
        surface_triangles=surface,
        material_lines=lines,
    )
    return {
        "nodes": len(local),
        "tets": len(tets),
        "surface_triangles": len(surface),
        "mass_kg": float(volumes.sum() * block["density"]),
        "material_line_nodes": int(lines.shape[1]),
        "material_line_rest_length_m": line_l0,
    }


# --------------------------------------------------------------------------
# Clamp
# --------------------------------------------------------------------------


def write_clamp(config) -> dict:
    clamp = config["clamp"]
    block = config["stretch_block"]
    table_z = config["scene"]["table_z"]
    x0, x1 = clamp["x_range"]
    cy = block["center"][1]
    hw = clamp["half_width"]
    jaw_top = clamp["jaw_bottom_z"] + clamp["jaw_thickness"]
    boxes = [
        {
            "name": "clamp_base",
            "center": [0.5 * (x0 + x1), cy, 0.5 * (table_z + clamp["base_top_z"])],
            "size": [x1 - x0, 2 * hw, clamp["base_top_z"] - table_z],
        },
        {
            "name": "clamp_jaw",
            "center": [0.5 * (x0 + x1), cy, 0.5 * (clamp["jaw_bottom_z"] + jaw_top)],
            "size": [x1 - x0, 2 * hw, clamp["jaw_thickness"]],
        },
    ]
    post_w = hw - clamp["post_inner_y"]
    for side, sign in (("left", 1.0), ("right", -1.0)):
        boxes.append(
            {
                "name": f"clamp_post_{side}",
                "center": [
                    0.5 * (x0 + x1),
                    cy + sign * 0.5 * (clamp["post_inner_y"] + hw),
                    0.5 * (clamp["base_top_z"] + clamp["jaw_bottom_z"]),
                ],
                "size": [x1 - x0, post_w, clamp["jaw_bottom_z"] - clamp["base_top_z"]],
            }
        )
    data = {"frame": "world; static boxes", "boxes": boxes}
    with open(INPUTS / "s01t_clamp.json", "w") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")
    return {"boxes": len(boxes)}


# --------------------------------------------------------------------------
# Initial states
# --------------------------------------------------------------------------


def pre_squeeze(config, rest: np.ndarray) -> np.ndarray:
    """Squeeze the near-end nodes in Z towards the bottom node level."""
    clamp = config["clamp"]
    block = config["stretch_block"]
    bottom = block["center"][2] - 0.5 * block["size"][2]
    edge = clamp["x_range"][1]
    ramp = clamp["pre_squeeze_ramp"]
    weight = np.clip((edge + ramp - rest[:, 0]) / ramp, 0.0, 1.0)
    out = rest.copy()
    out[:, 2] = bottom + (rest[:, 2] - bottom) * (1.0 - clamp["pre_squeeze"] * weight)
    return out


def write_initial_states(config) -> dict:
    spec = config["initial_states"]
    block = config["stretch_block"]
    clamp = config["clamp"]
    rng = np.random.default_rng(spec["seed"])
    count = int(spec["count"])
    lo, hi = spec["dx_dy_range"]
    ylo, yhi = spec["yaw_range_deg"]
    offsets = np.zeros((count, 2))
    yaws = np.zeros(count)
    offsets[1:] = rng.uniform(lo, hi, size=(count - 1, 2))
    yaws[1:] = rng.uniform(ylo, yhi, size=count - 1)
    local = np.load(INPUTS / "s01t_block_mesh.npz")["local_vertices"].astype(np.float64)
    radius = block["particle_radius"]
    rest_all, init_all, centers = [], [], np.zeros((count, 3))
    worst_post = 0.0
    for i in range(count):
        c, s = np.cos(np.deg2rad(yaws[i])), np.sin(np.deg2rad(yaws[i]))
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        centers[i] = np.asarray(block["center"]) + np.r_[offsets[i], 0.0]
        rest = local @ rot.T + centers[i]
        init = pre_squeeze(config, rest)
        inside = rest[:, 0] <= clamp["x_range"][1]
        worst_post = max(
            worst_post,
            float(np.abs(rest[inside, 1] - block["center"][1]).max()) + radius,
        )
        if init[inside, 2].max() + radius > clamp["jaw_bottom_z"] + 1e-6:
            raise RuntimeError("pre-squeezed nodes overlap the clamp jaw")
        rest_all.append(rest)
        init_all.append(init)
    if worst_post >= clamp["post_inner_y"]:
        raise RuntimeError(f"block reaches the clamp posts ({worst_post:.4f})")
    np.savez(
        INPUTS / "s01t_initial_states.npz",
        state_ids=np.asarray([f"s{i:02d}" for i in range(count)]),
        block_offset_xy=offsets,
        block_yaw_deg=yaws,
        block_center=centers,
        rest_positions=np.asarray(rest_all, dtype=np.float32),
        initial_positions=np.asarray(init_all, dtype=np.float32),
        seed=spec["seed"],
    )
    return {
        "count": count,
        "post_clearance_m": clamp["post_inner_y"] - worst_post,
        "offsets_mm": np.round(1e3 * offsets, 3).tolist(),
        "yaw_deg": np.round(yaws, 3).tolist(),
    }


# --------------------------------------------------------------------------
# IK table (DexSim IK), TCP moving along X
# --------------------------------------------------------------------------


def write_ik_table(config, device) -> dict:
    import warp as wp
    from scipy.spatial.transform import Rotation
    from dexsim.kit.ik import (
        IKSolver,
        JointLimitTarget,
        KinematicModel,
        PositionTarget,
        RotationTarget,
    )

    robot = config["robot"]
    traj = config["trajectory"]
    grip = config["grip"]
    model = KinematicModel.from_urdf(
        robot["urdf"], floating=False, hide_visuals=True, collapse_fixed_joints=False
    )
    starts = model.joint_q_start.numpy()
    dof_starts = model.joint_qd_start.numpy()
    coords = {
        label.rsplit("/", 1)[-1]: int(starts[j])
        for j, label in enumerate(model.joint_label)
        if starts[j + 1] > starts[j]
    }
    body = next(
        i
        for i, label in enumerate(model.body_label)
        if label.endswith("/" + robot["tcp_body"])
    )
    base = np.asarray(robot["base_position"], dtype=np.float64)
    position = PositionTarget(
        body,
        wp.vec3(*robot["tcp_offset"]),
        wp.array([wp.vec3(0.0, 0.0, 0.5)], dtype=wp.vec3, device=device),
    )
    rotation = RotationTarget(
        body,
        wp.quat_identity(),
        wp.array([wp.vec4(*robot["tcp_rotation_xyzw"])], dtype=wp.vec4, device=device),
    )
    limits = JointLimitTarget(
        model.joint_limit_lower, model.joint_limit_upper, weight=10.0
    )
    mask = np.zeros(model.joint_dof_count, dtype=bool)
    for j, label in enumerate(model.joint_label):
        if label.rsplit("/", 1)[-1] in robot["arm_joints"]:
            mask[int(dof_starts[j]) : int(dof_starts[j + 1])] = True
    solver = IKSolver(
        model, 1, [position, rotation, limits], lambda_initial=0.03, joint_dof_mask=mask
    )
    q = model.joint_q.numpy().astype(np.float32)
    for name, value in zip(robot["arm_joints"], robot["seed_q"]):
        q[coords[name]] = value
    q_wp = wp.array(q.reshape(1, -1), device=device)
    origin = np.array([grip["tcp_x"], grip["tcp_y"], grip["tcp_z"]])

    def solve(dx, iterations):
        target = origin + np.array([dx, 0.0, 0.0]) - base
        position.set_target_position(0, wp.vec3(*target))
        solver.step(q_wp, q_wp, iterations=iterations)

    lo, hi = traj["ik_table_range"]
    step = traj["ik_table_step"]
    offsets = np.round(np.arange(lo, hi + 0.5 * step, step), 7)
    solve(offsets[0], 200)
    tcp_offset = np.asarray(robot["tcp_offset"])
    lower = model.joint_limit_lower.numpy()
    upper = model.joint_limit_upper.numpy()
    target_rot = Rotation.from_quat(robot["tcp_rotation_xyzw"])
    rows, errors, rot_errors = [], [], []
    for dx in offsets:
        solve(dx, 40)
        row = q_wp.numpy()[0]
        if np.any(row < lower - 1e-5) or np.any(row > upper + 1e-5):
            raise RuntimeError(f"IK left the joint limits at dx={dx:+.4f}")
        state = model.state()
        model.forward_kinematics(wp.array(row.astype(np.float32), device=device), state)
        pose = state.body_q.numpy()[body]
        rot = Rotation.from_quat(pose[3:7])
        achieved = pose[:3] + rot.apply(tcp_offset) + base
        errors.append(np.linalg.norm(achieved - origin - [dx, 0.0, 0.0]))
        rot_errors.append((rot * target_rot.inv()).magnitude())
        rows.append([row[coords[name]] for name in robot["arm_joints"]])
    errors, rot_errors = np.asarray(errors), np.asarray(rot_errors)
    if errors.max() > 2e-4 or rot_errors.max() > 1e-3:
        raise RuntimeError(
            f"IK error too large: {errors.max():.2e} m, " f"{rot_errors.max():.2e} rad"
        )
    np.savez(
        INPUTS / "s01t_ik_table.npz",
        offset=offsets,
        axis="x",
        arm_q=np.asarray(rows, dtype=np.float64),
        joint_names=np.asarray(robot["arm_joints"]),
        tcp_origin=origin,
        ik_tcp_error=errors,
        ik_rot_error=rot_errors,
        urdf_sha256=sha256_file(robot["urdf"]),
    )
    return {
        "rows": len(offsets),
        "max_tcp_error_m": float(errors.max()),
        "max_rot_error_rad": float(rot_errors.max()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-ik", action="store_true")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as stream:
        config = json.load(stream)
    global INPUTS
    INPUTS = output_dir("s01t", args.out_dir)
    INPUTS.mkdir(parents=True, exist_ok=False)
    with open(INPUTS / "s01t_config.json", "w") as stream:
        json.dump(config, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    report = {
        "mesh": write_mesh(config),
        "clamp": write_clamp(config),
        "initial_states": write_initial_states(config),
    }
    if not args.skip_ik:
        import warp as wp

        with wp.ScopedDevice(args.device):
            report["ik_table"] = write_ik_table(config, args.device)
    files = sorted(
        p for p in INPUTS.iterdir() if p.is_file() and p.name != "inputs_manifest.json"
    )
    manifest = {
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": sha256_file(__file__),
        "robot_urdf": config["robot"]["urdf"],
        "robot_urdf_sha256": sha256_file(config["robot"]["urdf"]),
        "files_sha256": {p.name: sha256_file(p) for p in files},
        "report": report,
    }
    with open(INPUTS / "inputs_manifest.json", "w") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
