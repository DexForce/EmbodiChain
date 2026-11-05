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
"""Check recorded reference geometry without running simulation.

Run with repository Python and --run <recording>. Original pure NumPy paper
checks are extracted from the unchanged frozen reference by AST. This does
not recreate original runtime velocity checks or claim continuous clearance.
"""

from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

__all__ = ["analyze", "main"]


def analyze(run: Path, output: Path | None = None) -> dict:
    """Apply original paper and posture criteria to actual outer-step poses."""
    reference = run / "source/reference/demo_dexuni_w1_bag_packing.py"
    tree = ast.parse(reference.read_text())
    names = {"lip_deviation", "count_handle_crossings"}
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {"np": np}
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(reference), "exec"),
        namespace,
    )
    with np.load(run / "preview/states.npz") as f:
        s = dict(f)
    with np.load(run / "inputs/bag.npz") as f:
        bag = dict(f)
    meta = json.loads((run / "manifest.json").read_text())
    t = s["time"]
    rest = bag["vertices"]
    rim = bag["rim"]
    n = int(bag["paper_count"])
    nf = int(bag["paper_faces"])
    chains = []
    for side in (-1, 1):
        chain = rim[side * rest[rim, 0] > 0.13 / 2 - 0.005]
        chains.append(chain[np.argsort(rest[chain, 1])])
    faces = bag["faces"][nf:]
    edges = np.unique(
        np.sort(
            np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]),
            axis=1,
        ),
        axis=0,
    )
    edges = edges[np.all(edges >= n, axis=1)]
    bow = np.array([namespace["lip_deviation"](p, chains) for p in s["bag"]])
    crossing = {}
    for i in range(0, len(t), 10):
        count = namespace["count_handle_crossings"](
            s["bag"][i], bag["faces"][:nf], edges
        )
        if count:
            crossing[str(i)] = int(count)
    links = meta["link_state_order"]
    joints = meta["joint_state_order"]
    q = s["qpos"]
    matrices = (
        Rotation.from_quat(s["links"][..., :4].reshape(-1, 4))
        .as_matrix()
        .reshape(len(t), -1, 3, 3)
    )
    pre = t <= 50 + 1e-8
    after = t >= 11 - 1e-8
    lower = {}
    upper = {}
    for joint in ET.parse(run / "inputs/w1_v030/robot.urdf").getroot().findall("joint"):
        limit = joint.find("limit")
        if limit is not None and joint.attrib["name"] in joints:
            lower[joint.attrib["name"]] = float(limit.attrib.get("lower", "-inf"))
            upper[joint.attrib["name"]] = float(limit.attrib.get("upper", "inf"))
    lo = np.array([lower.get(n, -np.inf) for n in joints])
    hi = np.array([upper.get(n, np.inf) for n in joints])
    arm = np.array(
        [
            joints.index(f"{side}_J{i}")
            for side in ("LEFT", "RIGHT")
            for i in range(1, 8)
        ],
        dtype=int,
    )
    clearance = np.minimum(q - lo, hi - q)
    bottom = s["bag"][:, bag["bottom"], 2].min(1) - 0.92
    tip = (t >= 5.5 - 1e-8) & (t <= 11 + 1e-8)
    shape_limit = np.where(t < 11, 0.04, 0.025)
    head = links.index("head_pitch_j2_link")
    left = links.index("left_gripper_base")
    shoulder = links.index("shoulder_pitch_l_j1_link")
    elbow = links.index("elbow_yaw_l_j4_link")
    active = (t > 3) & pre
    blend = np.clip((t - 23.0) / 2, 0, 1)
    blend = blend**3 * (10 + blend * (-15 + 6 * blend))
    target = (1 - blend[:, None]) * s["tcp"][:, 0] + blend[:, None] * s["tcp"][:, 1]
    direction = target - s["links"][:, head, 4:]
    direction /= np.maximum(np.linalg.norm(direction, axis=-1, keepdims=True), 1e-12)
    head_cos = (matrices[:, head, :, 0] * direction).sum(-1)
    checks = dict(
        finite=all(np.isfinite(v).all() for v in s.values()),
        lip_bow=bool(bow.max() <= 0.03),
        handle_paper_crossings=not crossing,
        original_shape_limits=bool((s["shape_error"] <= shape_limit).all()),
        upright=bool(
            not after.any()
            or (s["bag_rotation"][after, 2, 2] >= np.cos(np.radians(30))).all()
        ),
        tip_bottom_on_table=bool(not tip.any() or bottom[tip].max() <= 0.01),
        tcp_tracking=bool(
            np.linalg.norm(s["tcp"] - s["tcp_targets"], axis=-1).max() <= 0.04
        ),
        right_wait_clear=bool(
            np.linalg.norm(s["tcp"][t < 24, 1] - s["tcp_targets"][0, 1], axis=-1).max()
            <= 0.01
        ),
        right_relaxed_before_packing=bool(
            np.abs(
                q[t < 24][:, [joints.index(f"RIGHT_FINGER{i}_JOINT") for i in (1, 2)]]
                - 0.015
            ).max()
            <= 1e-4
        ),
        joint_limits=bool((clearance >= -1e-5).all()),
        arm_stop_clearance=bool(
            len(arm) > 0 and clearance[:, arm].min() >= np.radians(10) - 1e-5
        ),
        joint_velocity=bool(
            len(t) < 2
            or np.abs(np.diff(q, axis=0) / np.diff(t)[:, None]).max() <= 4.145
        ),
        left_elbow=bool(
            not active.any()
            or (
                s["links"][active, shoulder, 6] - s["links"][active, elbow, 6] >= 0.05
            ).all()
        ),
        head_follows_hands=bool(
            not active.any() or (head_cos[active] >= np.cos(np.radians(35))).all()
        ),
        left_rim_support_orientation=bool(
            not (t >= 24.0).any()
            or (np.abs(matrices[t >= 24.0, left, 2, 2]) <= 0.35).all()
        ),
    )
    if t[-1] >= 50 - 1e-8:
        checks["both_goods_physically_lifted"] = bool(
            (s["snack_pose"][pre, :, 6].max(0) >= 1.18).all()
        )
        checks["both_goods_retained_at_50s"] = bool(
            s["loaded"][np.flatnonzero(pre)[-1]].all()
        )
    collider_summary = None
    provisioning = run / "analysis/authored_robot_colliders.json"
    if provisioning.exists():
        import trimesh

        records = json.loads(provisioning.read_text())["shapes"]
        actual = np.load(run / "analysis/initial_model.npz")
        meshes = np.load(run / "analysis/authored_robot_collision_meshes.npz")
        authored = ET.parse(run / "inputs/w1_v030/robot.urdf").getroot()
        roots = {link.attrib["name"]: link for link in authored.findall("link")}
        world = np.repeat(np.eye(4)[None, None], len(t), axis=0)
        world = np.repeat(world, len(links), axis=1)
        world[:, :, :3, :3] = matrices
        world[:, :, :3, 3] = s["links"][:, :, 4:]
        max_pose_error = 0.0
        for r in records:
            index = r["shape"]
            name = r["source_link"]
            source = run / r["source_file"]
            assert hashlib.sha256(source.read_bytes()).hexdigest() == r["source_sha256"]
            collision = roots[name].findall("collision")[
                int(str(actual["shape_labels"][index]).rsplit("/", 1)[-1])
            ]
            geometry = collision.find("geometry/mesh")
            original = trimesh.load(source, force="mesh")
            original.apply_scale(np.fromstring(geometry.get("scale", "1 1 1"), sep=" "))
            assert np.array_equal(
                np.asarray(original.vertices, dtype=np.float32),
                meshes[f"vertices_{index}"],
            )
            assert np.array_equal(
                np.asarray(original.faces, dtype=np.int32), meshes[f"faces_{index}"]
            )
            origin = collision.find("origin")
            local = np.eye(4)
            if origin is not None:
                local[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                local[:3, :3] = Rotation.from_euler(
                    "xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
                ).as_matrix()
            transform = actual["shape_transform"][index]
            physical = np.eye(4)
            physical[:3, :3] = Rotation.from_quat(transform[3:]).as_matrix()
            physical[:3, 3] = transform[:3]
            expected = world[:, links.index(name)] @ local
            observed = world[:, links.index(r["body_link"])] @ physical
            max_pose_error = max(
                max_pose_error, float(np.abs(expected - observed).max())
            )
            assert actual["shape_type"][index] == 8 and (
                int(actual["shape_flags"][index]) & 4
            )
            assert np.array_equal(
                actual["shape_scale"][index], np.ones(3, dtype=np.float32)
            )
        checks["authored_collision_geometry"] = (
            len(records) == 29 and max_pose_error < 1e-6
        )
        collider_summary = dict(
            shapes=len(records),
            finger_sdfs=sum(r["finger_sdf"] for r in records),
            all_vertices_faces_byte_equal=True,
            max_world_transform_error=max_pose_error,
            samples=len(t),
        )
        (run / "analysis/authored_colliders_check.json").write_text(
            json.dumps(collider_summary, indent=2) + "\n"
        )
    result = dict(
        authored_robot_collision=collider_summary,
        status="passed" if all(checks.values()) else "failed",
        checks=checks,
        samples=len(t),
        actual_seconds=float(t[-1]),
        reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
        max_lip_bow_m=float(bow.max()),
        handle_crossings=crossing,
        max_shape_error_m=float(s["shape_error"].max()),
        min_upright_cosine=(
            float(s["bag_rotation"][after, 2, 2].min()) if after.any() else None
        ),
        arm_joint_names=[joints[i] for i in arm],
        min_arm_stop_margin_rad=float(clearance[:, arm].min()) if len(arm) else None,
        max_joint_speed_rad_s=(
            float(np.abs(np.diff(q, axis=0) / np.diff(t)[:, None]).max())
            if len(t) > 1
            else 0.0
        ),
        min_head_direction_cosine=(
            float(head_cos[active].min()) if active.any() else None
        ),
        scope="Original paper shape/lip/upright/handle crossings and robot posture on recorded 60Hz poses; handle crossings every10 frames. Does not replace original runtime velocity/final settling checks, FCL or continuous collision checks.",
    )
    (output or run / "analysis/scene_checks.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    return result


def main() -> None:
    """Analyze an existing recording."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    print(json.dumps(analyze(a.run, a.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
