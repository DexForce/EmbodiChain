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
"""Generate a NEW local S01C input snapshot; never write source assets.

Run: <dexsim-python> scripts/benchmark/tech_reports/physics/experiments/dexuni/s01c_compress/tools/prepare_inputs.py --out-dir <new-cache-directory>
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
CONFIG = ASSETS / "s01c" / "s01c_config.json"


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Soft block mesh
# --------------------------------------------------------------------------


def box_tet_mesh(size, cells: int):
    """Structured tet box centred at the origin (Kuhn 6-tet split).

    Neighbouring cells share faces conformingly; every tet has positive
    volume ``dot(x1-x0, cross(x2-x0, x3-x0)) > 0``.
    """
    size = np.asarray(size, dtype=np.float64)
    n = int(cells)
    axes = [np.linspace(-0.5 * s, 0.5 * s, n + 1) for s in size]
    vertices = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

    def vid(i, j, k):
        return (i * (n + 1) + j) * (n + 1) + k

    perms = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))
    tets = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
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
    """Outward boundary faces of a positively oriented tet mesh."""
    faces = np.concatenate(
        [tets[:, (0, 2, 1)], tets[:, (0, 1, 3)], tets[:, (0, 3, 2)], tets[:, (1, 2, 3)]]
    )
    key = np.sort(faces, axis=1)
    _, inverse, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    return faces[counts[inverse.reshape(-1)] == 1].astype(np.int32)


def write_mesh(config) -> dict:
    block = config["soft_block"]
    local, tets = box_tet_mesh(block["size"], block["cells_per_side"])
    surface = tet_surface_triangles(tets)
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
        INPUTS / "s01c_soft_block_mesh.npz",
        local_vertices=local.astype(np.float32),
        tets=tets,
        surface_triangles=surface,
    )
    return {
        "nodes": len(local),
        "tets": len(tets),
        "surface_triangles": len(surface),
        "volume_m3": float(volumes.sum()),
        "min_tet_volume_m3": float(volumes.min()),
        "mass_kg": float(volumes.sum() * block["density"]),
    }


# --------------------------------------------------------------------------
# Press plate
# --------------------------------------------------------------------------


def plate_parts(plate) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Plate boxes in the plate body frame (origin = base bottom centre)."""
    base = np.asarray(plate["base_size"], dtype=np.float64)
    handle = np.asarray(plate["handle_size"], dtype=np.float64)
    parts = [
        ("base", np.array([0.0, 0.0, 0.5 * base[2]]), base),
        ("handle", np.array([0.0, 0.0, base[2] + 0.5 * handle[2]]), handle),
    ]
    for name in ("collar", "cap"):
        spec = plate[name]
        z0, z1 = spec["z_range"]
        y0, y1 = spec["y_extent"]
        size = np.array([spec["x_size"], y1 - y0, z1 - z0])
        for side, sign in (("left", 1.0), ("right", -1.0)):
            centre = np.array([0.0, sign * 0.5 * (y0 + y1), 0.5 * (z0 + z1)])
            parts.append((f"{name}_{side}", centre, size))
    spec = plate["guides"]
    x0, x1 = spec["x_range"]
    y0, y1 = spec["y_extent"]
    z0, z1 = spec["z_range"]
    size = np.array([x1 - x0, y1 - y0, z1 - z0])
    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            centre = np.array(
                [sx * 0.5 * (x0 + x1), sy * 0.5 * (y0 + y1), 0.5 * (z0 + z1)]
            )
            parts.append(
                (
                    f"guide_{'p' if sx > 0 else 'n'}x_" f"{'p' if sy > 0 else 'n'}y",
                    centre,
                    size,
                )
            )
    return parts


def write_plate(config) -> dict:
    plate = config["press_plate"]
    parts = plate_parts(plate)
    volume = float(sum(np.prod(size) for _, _, size in parts))
    density = float(plate["mass"]) / volume
    masses = [density * np.prod(size) for _, _, size in parts]
    total = float(sum(masses))
    com = sum(m * c for m, (_, c, _) in zip(masses, parts)) / total
    inertia = np.zeros((3, 3))
    for m, (_, c, s) in zip(masses, parts):
        box = (
            m
            / 12.0
            * np.diag(
                [s[1] ** 2 + s[2] ** 2, s[0] ** 2 + s[2] ** 2, s[0] ** 2 + s[1] ** 2]
            )
        )
        d = c - com
        inertia += box + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    data = {
        "frame": "plate body frame; origin at the base bottom centre, axes = world axes at rest",
        "parts": [
            {"name": n, "center": c.tolist(), "size": s.tolist()} for n, c, s in parts
        ],
        "density": density,
        "mass": total,
        "com": com.tolist(),
        "inertia": inertia.tolist(),
        "tcp_above_base_bottom": plate["tcp_above_base_bottom"],
    }
    with open(INPUTS / "s01c_plate.json", "w") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")
    return {"parts": len(parts), "mass": total, "density": density}


# --------------------------------------------------------------------------
# Initial states
# --------------------------------------------------------------------------


def write_initial_states(config) -> dict:
    spec = config["initial_states"]
    block = config["soft_block"]
    plate = config["press_plate"]
    rng = np.random.default_rng(spec["seed"])
    count = int(spec["count"])
    lo, hi = spec["dx_dy_range"]
    ylo, yhi = spec["yaw_range_deg"]
    offsets = np.zeros((count, 2))
    yaws = np.zeros(count)
    offsets[1:] = rng.uniform(lo, hi, size=(count - 1, 2))
    yaws[1:] = rng.uniform(ylo, yhi, size=count - 1)
    mesh = np.load(INPUTS / "s01c_soft_block_mesh.npz")
    local = mesh["local_vertices"].astype(np.float64)
    centers = np.zeros((count, 3))
    vertices = np.zeros((count, len(local), 3), dtype=np.float32)
    half_plate = 0.5 * plate["base_size"][0]
    worst = 0.0
    for i in range(count):
        c, s = np.cos(np.deg2rad(yaws[i])), np.sin(np.deg2rad(yaws[i]))
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        centers[i] = np.asarray(block["center"]) + np.r_[offsets[i], 0.0]
        world = local @ rot.T + centers[i]
        vertices[i] = world
        rel = world[:, :2] - np.asarray(config["trajectory"]["block_xy"])
        worst = max(worst, float(np.abs(rel).max()))
    if worst >= half_plate:
        raise RuntimeError(f"block footprint {worst:.4f} m leaves the plate")
    np.savez(
        INPUTS / "s01c_initial_states.npz",
        state_ids=np.asarray([f"s{i:02d}" for i in range(count)]),
        block_offset_xy=offsets,
        block_yaw_deg=yaws,
        block_center=centers,
        particle_positions=vertices,
        seed=spec["seed"],
    )
    return {
        "count": count,
        "max_footprint_from_plate_centre_m": worst,
        "offsets_mm": np.round(1e3 * offsets, 3).tolist(),
        "yaw_deg": np.round(yaws, 3).tolist(),
    }


# --------------------------------------------------------------------------
# IK table (DexSim IK)
# --------------------------------------------------------------------------


def contact_tcp_height(config) -> float:
    """Nominal TCP z where the held plate's base touches the unloaded block."""
    block = config["soft_block"]
    top = block["center"][2] + 0.5 * block["size"][2]
    return float(
        top + block["particle_radius"] + config["press_plate"]["tcp_above_base_bottom"]
    )


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
    contact_z = contact_tcp_height(config)
    x, y = traj["block_xy"]

    def solve(z, iterations):
        target = np.array([x, y, contact_z + z]) - base
        position.set_target_position(0, wp.vec3(*target))
        solver.step(q_wp, q_wp, iterations=iterations)

    lo, hi = traj["ik_table_z_range"]
    step = traj["ik_table_step"]
    offsets = np.round(np.arange(hi, lo - 0.5 * step, -step), 7)
    solve(offsets[0], 200)
    tcp_offset = np.asarray(robot["tcp_offset"])
    lower = model.joint_limit_lower.numpy()
    upper = model.joint_limit_upper.numpy()
    target_rot = Rotation.from_quat(robot["tcp_rotation_xyzw"])
    rows, errors, rot_errors = [], [], []
    for z in offsets:
        solve(z, 40)
        row = q_wp.numpy()[0]
        if np.any(row < lower - 1e-5) or np.any(row > upper + 1e-5):
            raise RuntimeError(f"IK left the joint limits at z={z:+.4f}")
        state = model.state()
        model.forward_kinematics(wp.array(row.astype(np.float32), device=device), state)
        pose = state.body_q.numpy()[body]
        rot = Rotation.from_quat(pose[3:7])
        achieved = pose[:3] + rot.apply(tcp_offset) + base
        errors.append(np.linalg.norm(achieved - [x, y, contact_z + z]))
        rot_errors.append((rot * target_rot.inv()).magnitude())
        rows.append([row[coords[name]] for name in robot["arm_joints"]])
    order = np.argsort(offsets)
    errors, rot_errors = np.asarray(errors), np.asarray(rot_errors)
    if errors.max() > 2e-4 or rot_errors.max() > 1e-3:
        raise RuntimeError(
            f"IK error too large: {errors.max():.2e} m, " f"{rot_errors.max():.2e} rad"
        )
    np.savez(
        INPUTS / "s01c_ik_table.npz",
        z_offset=offsets[order],
        arm_q=np.asarray(rows, dtype=np.float64)[order],
        joint_names=np.asarray(robot["arm_joints"]),
        contact_tcp_z=contact_z,
        block_xy=np.asarray(traj["block_xy"]),
        ik_tcp_error=errors[order],
        ik_rot_error=rot_errors[order],
        urdf_sha256=sha256_file(robot["urdf"]),
    )
    return {
        "rows": len(offsets),
        "max_tcp_error_m": float(errors.max()),
        "max_rot_error_rad": float(rot_errors.max()),
        "contact_tcp_z": contact_z,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-ik", action="store_true")
    parser.add_argument(
        "--out-dir",
        required=True,
        help="write an alternative inputs set here (the "
        "config is copied with --cells applied)",
    )
    parser.add_argument(
        "--cells", type=int, help="soft_block.cells_per_side for --out-dir"
    )
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    global INPUTS
    with open(args.config, encoding="utf-8") as stream:
        config = json.load(stream)
    INPUTS = output_dir("s01c", args.out_dir)
    INPUTS.mkdir(parents=True, exist_ok=False)
    if args.cells:
        config["soft_block"]["cells_per_side"] = args.cells
    with open(INPUTS / "s01c_config.json", "w") as stream:
        json.dump(config, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    report = {
        "mesh": write_mesh(config),
        "plate": write_plate(config),
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
