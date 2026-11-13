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
"""Generate a NEW local S02 input snapshot; never write source assets.

Run: <dexsim-python> scripts/benchmark/tech_reports/physics/experiments/dexuni/s02_soft_pick_place/tools/prepare_inputs.py --out-dir <new-cache-directory>
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

TASK_DIR = Path(__file__).resolve().parents[1]
INPUTS = None
CONFIG = ASSETS / "s02" / "s02_config.json"


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Mesh and width patches
# --------------------------------------------------------------------------


def box_tet_mesh(size, cells):
    """Structured tet box centred at the origin (Kuhn 6-tet split)."""
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


def width_patches(config, local: np.ndarray) -> np.ndarray:
    """Node ids of the -Y and +Y face patches, shape (2, nodes)."""
    hx, hz = config["width_patches"]["half_extent"]
    half_y = 0.5 * config["soft_block"]["size"][1]
    patches = []
    for sign in (-1.0, 1.0):
        pick = np.flatnonzero(
            (np.abs(local[:, 1] - sign * half_y) < 1e-6)
            & (np.abs(local[:, 0]) <= hx + 1e-6)
            & (np.abs(local[:, 2]) <= hz + 1e-6)
        )
        patches.append(pick)
    if len(patches[0]) != len(patches[1]) or len(patches[0]) < 1:
        raise RuntimeError("width patches have different node counts")
    return np.asarray(patches, dtype=np.int32)


def write_mesh(config) -> dict:
    block = config["soft_block"]
    local, tets = box_tet_mesh(block["size"], block["cells"])
    surface = tet_surface_triangles(tets)
    patches = width_patches(config, local)
    x = local[tets]
    volumes = (
        np.einsum(
            "ij,ij->i",
            x[:, 1] - x[:, 0],
            np.cross(x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]),
        )
        / 6.0
    )
    np.savez(
        INPUTS / "s02_block_mesh.npz",
        local_vertices=local.astype(np.float32),
        tets=tets,
        surface_triangles=surface,
        width_patches=patches,
    )
    return {
        "nodes": len(local),
        "tets": len(tets),
        "surface_triangles": len(surface),
        "mass_kg": float(volumes.sum() * block["density"]),
        "width_patch_nodes": int(patches.shape[1]),
        "rest_width_m": float(
            local[patches[1], 1].mean() - local[patches[0], 1].mean()
        ),
    }


# --------------------------------------------------------------------------
# Initial states
# --------------------------------------------------------------------------


def write_initial_states(config) -> dict:
    spec = config["initial_states"]
    block = config["soft_block"]
    rng = np.random.default_rng(spec["seed"])
    count = int(spec["count"])
    lo, hi = spec["dx_dy_range"]
    ylo, yhi = spec["yaw_range_deg"]
    offsets = np.zeros((count, 2))
    yaws = np.zeros(count)
    offsets[1:] = rng.uniform(lo, hi, size=(count - 1, 2))
    yaws[1:] = rng.uniform(ylo, yhi, size=count - 1)
    local = np.load(INPUTS / "s02_block_mesh.npz")["local_vertices"].astype(np.float64)
    radius = block["particle_radius"]
    q_open = config["robot"]["finger_initial_q"]
    table_z = config["scene"]["table_z"]
    positions, centers = [], np.zeros((count, 3))
    worst_clearance = np.inf
    for i in range(count):
        c, s = np.cos(np.deg2rad(yaws[i])), np.sin(np.deg2rad(yaws[i]))
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        centers[i] = np.asarray(block["center"]) + np.r_[offsets[i], 0.0]
        world = local @ rot.T + centers[i]
        if world[:, 2].min() - radius < table_z - 1e-6:
            raise RuntimeError("block overlaps the table")
        # Finger inner faces sit at tcp_y +- q_open.
        tcp_y = config["grip"]["tcp"][1]
        reach = np.abs(world[:, 1] - tcp_y).max() + radius
        worst_clearance = min(worst_clearance, q_open - reach)
        positions.append(world)
    if worst_clearance <= 0.001:
        raise RuntimeError(
            f"open fingers too close to the block " f"({worst_clearance:.4f} m)"
        )
    np.savez(
        INPUTS / "s02_initial_states.npz",
        state_ids=np.asarray([f"s{i:02d}" for i in range(count)]),
        block_offset_xy=offsets,
        block_yaw_deg=yaws,
        block_center=centers,
        initial_positions=np.asarray(positions, dtype=np.float32),
        seed=spec["seed"],
    )
    return {
        "count": count,
        "finger_clearance_m": float(worst_clearance),
        "offsets_mm": np.round(1e3 * offsets, 3).tolist(),
        "yaw_deg": np.round(yaws, 3).tolist(),
    }


# --------------------------------------------------------------------------
# TCP path and IK table (DexSim IK)
# --------------------------------------------------------------------------


def path_waypoints(config) -> np.ndarray:
    path, grip = config["path"], config["grip"]
    start = np.asarray(grip["tcp"], dtype=np.float64)
    lifted = start + [0.0, 0.0, path["lift_height"]]
    above = np.array([*path["place_xy"], lifted[2]])
    place = np.array([*path["place_xy"], start[2] + path["place_clearance"]])
    retreat = place + [0.0, 0.0, path["retreat_height"]]
    return np.stack([start, lifted, above, place, retreat])


def write_ik_path(config, device) -> dict:
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

    def solve(point, iterations):
        position.set_target_position(0, wp.vec3(*(point - base)))
        solver.step(q_wp, q_wp, iterations=iterations)

    waypoints = path_waypoints(config)
    seg = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    waypoint_s = np.r_[0.0, np.cumsum(seg)]
    step = config["path"]["ik_step"]
    s_values = np.unique(np.r_[np.arange(0.0, waypoint_s[-1], step), waypoint_s])

    def point_at(s):
        k = min(int(np.searchsorted(waypoint_s, s, side="right")) - 1, len(seg) - 1)
        u = (s - waypoint_s[k]) / seg[k]
        return waypoints[k] + u * (waypoints[k + 1] - waypoints[k])

    solve(point_at(0.0), 200)
    tcp_offset = np.asarray(robot["tcp_offset"])
    lower = model.joint_limit_lower.numpy()
    upper = model.joint_limit_upper.numpy()
    target_rot = Rotation.from_quat(robot["tcp_rotation_xyzw"])
    rows, errors, rot_errors, points = [], [], [], []
    for s in s_values:
        point = point_at(s)
        solve(point, 40)
        row = q_wp.numpy()[0]
        if np.any(row < lower - 1e-5) or np.any(row > upper + 1e-5):
            raise RuntimeError(f"IK left the joint limits at s={s:.4f}")
        state = model.state()
        model.forward_kinematics(wp.array(row.astype(np.float32), device=device), state)
        pose = state.body_q.numpy()[body]
        rot = Rotation.from_quat(pose[3:7])
        achieved = pose[:3] + rot.apply(tcp_offset) + base
        errors.append(np.linalg.norm(achieved - point))
        rot_errors.append((rot * target_rot.inv()).magnitude())
        rows.append([row[coords[name]] for name in robot["arm_joints"]])
        points.append(point)
    errors, rot_errors = np.asarray(errors), np.asarray(rot_errors)
    rows = np.asarray(rows, dtype=np.float64)
    if errors.max() > 2e-4 or rot_errors.max() > 1e-3:
        raise RuntimeError(
            f"IK error too large: {errors.max():.2e} m, " f"{rot_errors.max():.2e} rad"
        )
    jump = np.abs(np.diff(rows, axis=0)).max()
    if jump > 0.02:
        raise RuntimeError(f"IK path jumps {jump:.3f} rad between rows")
    np.savez(
        INPUTS / "s02_ik_path.npz",
        s=s_values,
        arm_q=rows,
        tcp=np.asarray(points),
        waypoints=waypoints,
        waypoint_s=waypoint_s,
        waypoint_names=np.asarray(
            ["grip", "lifted", "above_target", "place", "retreat"]
        ),
        joint_names=np.asarray(robot["arm_joints"]),
        ik_tcp_error=errors,
        ik_rot_error=rot_errors,
        urdf_sha256=sha256_file(robot["urdf"]),
    )
    return {
        "rows": len(s_values),
        "length_m": float(waypoint_s[-1]),
        "waypoint_s": waypoint_s.tolist(),
        "max_tcp_error_m": float(errors.max()),
        "max_rot_error_rad": float(rot_errors.max()),
        "max_row_jump_rad": float(jump),
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
    INPUTS = output_dir("s02", args.out_dir)
    INPUTS.mkdir(parents=True, exist_ok=False)
    with open(INPUTS / "s02_config.json", "w") as stream:
        json.dump(config, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    report = {
        "mesh": write_mesh(config),
        "initial_states": write_initial_states(config),
    }
    if not args.skip_ik:
        import warp as wp

        with wp.ScopedDevice(args.device):
            report["ik_path"] = write_ik_path(config, args.device)
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
