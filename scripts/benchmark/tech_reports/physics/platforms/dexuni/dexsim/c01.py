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
"""Record the native W1 supermarket packing scene (C01).

Run with the dexsimforembodychain staged package and conda dexsim1 Python;
the case README gives the complete environment and command.
Pilot: add ``--pilot --out <scratch dir>`` to save still frames instead of the video.

Reference: demo_dexuni_supermarket_packing.py (shopping bag, rail rack, film
drag, bread and its contact/solver settings). W1 V030 with Pika grippers, one-way
coupling with kinematic joints; the IK trajectory is baked offline. The left
arm packs bread and returns home before the right arm packs the milk pouch at
a separate drop site. Extra rigid cargo is omitted.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[7]))

from scripts.benchmark.tech_reports.physics.experiments.dexuni.c01_common import (
    BAG_MOUTH_LOCAL_Z,
    BAG_POSITION,
    BAG_YAW_DEG,
    GOODS,
    GOOD_ARMS,
    BLOCK_DROP,
    BLOCK_HALF,
    BLOCK_MASS,
    BLOCK_START,
    BREAD_DROP,
    BREAD_START,
    BREAD_YAW_DEG,
    DURATION,
    OUTPUT,
    PHASES,
    POUCH_DROP,
    POUCH_START,
    TABLE_Z,
    bag_groups,
    bag_world,
    end_of,
    packing_metrics,
    pouch_mesh,
    phase as _phase,
    rack_foot,
    rack_parts,
    worktop_pieces,
)
from scripts.benchmark.tech_reports.physics.platforms.dexuni.dexsim.m02 import (
    TrajectoryBaker,
)

REFERENCE = Path("/home/oem/code/engine/dexsim/examples/python/physics/multiphys")
REFERENCE_DEMO = "demo_dexuni_supermarket_packing.py"
REFERENCE_NAMES = (
    REFERENCE_DEMO,
    "dexuni_w1_two_way_utils.py",
    "demo_dexuni_w1_bag_packing.py",
    "dexuni_asset_utils.py",
)
SUPERMARKET = Path("/home/oem/.cache/dexsim/resources/mjvbd_v2/supermarket")
SUPERMARKET_FILES = (
    "shopping_bag.json",
    "bread.npz",
    "bag_print.png",
    "bread_crust.png",
    "milk_pouch.png",
)
ROBOT = Path("/home/oem/.cache/dexsim/resources/mjvbd_v2/w1_bag_packing/w1_v030")
FPS = 60
STEPS = round(DURATION * FPS)
# ---- Supermarket example: solver, contact, bag and bread ----
# Resolve the milk impact without the transient bread-tet inversion observed
# in the sequential two-arm eight-substep recording; material values are fixed.
SUBSTEPS = 16
ITERATIONS = 24
SELF_CONTACT = {
    "particle_enable_self_contact": True,
    "particle_self_contact_radius": 0.002,
    "particle_self_contact_margin": 0.004,
    "particle_topological_contact_filter_threshold": 2,
    "particle_rest_shape_contact_exclusion_radius": 0.004,
    "particle_vertex_contact_buffer_size": 64,
    "particle_edge_contact_buffer_size": 64,
    "particle_enable_truncation_cache": True,
}
SOFT_CONTACT_MARGIN = 0.004
# Consistent compliant interfaces for bag, bread and milk. Particle-particle
# parameters are averaged, so leaving bread at the reference's 2e5/10 would
# retain a stiff bag/bread interface. Rigid finger materials remain unchanged.
PARTICLE_CONTACT = {"stiffness": 2e3, "damping": 0.25, "dynamic_friction": 0.4}
BAG_CONTACT = dict(PARTICLE_CONTACT)
FILM_DRAG = {"air_density": 1.2, "normal_drag": 1.2, "tangential_drag": 0.02}
RACK_MU = 0.35
BREAD = {"volume_density": 250.0, "k_mu": 15000.0, "k_lambda": 30000.0, "k_damp": 1.0}
BREAD_PARTICLE_RADIUS = 0.002
# ---- Original supermarket milk pouch (geometry and all material/gas values) ----
POUCH = {
    "surface_density": 0.18,
    "tri_ke": 20000.0,
    "tri_ka": 20000.0,
    "tri_kd": 1.0,
    "edge_ke": 0.015,
    "edge_kd": 0.002,
}
POUCH_PARTICLE_RADIUS = 0.0015
POUCH_GAS = {
    "reference_absolute_pressure": 101800.0,
    "ambient_pressure": 101_325.0,
    "bulk_damping": 10.0,
    "max_absolute_pressure": 180000.0,
}
POUCH_CONTACT = dict(BAG_CONTACT)
# ---- Shapes (M05 values unless noted) ----
CONTACT_MARGIN = 0.0005
CONTACT_GAP = 0.0
# Use the supermarket demo's smoothing, rather than inheriting M05's 1e-4.
# This setting alone did not resolve the compressed pouch's release rebound.
FRICTION_EPSILON = 1e-3
CONTACT_KD = 100.0
ROBOT_KE = 2e4
LEFT_FINGER_KE = 2e5  # retain the existing left-finger contact coefficient
RIGHT_FINGER_KE = 6e5  # example hand stiffness on the pouch-gripping fingers (M05)
FINGER_MU = 2.0
BLOCK_KE = 2e5
BLOCK_MU = 2.0
TABLE_KE = 2e5
TABLE_MU = 0.9
# ---- Task motion ----
FINGER_LINKS = ("left_finger1", "left_finger2", "right_finger1", "right_finger2")
DOWNWARD_X_FINGERS = np.array((-np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0))
YAW_DEG = 90.0  # fingers close along world Y
# Per-arm gripper lean (IK scan: the right arm needs ~30 deg to reach the
# pouch input and the bag centre near worktop height; 15 deg suffices left).
PITCH_DEG = {"left": 15.0, "right": 30.0}
OPEN = {"left": 0.045, "right": 0.05}  # jaw gap ~ 2 q - 1.4 mm
# Release opening: the block and bread open by ~20 mm of jaw only, so the outer
# finger stays clear of the bag handles and walls. The pouch also opens only
# enough to release its thin sealed edge; its inflated body hangs below it.
RELEASE_EXTRA = {"block": 0.010, "bread": 0.010, "pouch": 0.010}
TIP_CLEARANCE = {"block": 0.010, "bread": 0.006, "pouch": 0.003}
# Bottom of each good above the worktop at rest (for TCP-to-bottom offsets).
REST_BOTTOM = {
    "block": 2 * CONTACT_MARGIN,
    "bread": BREAD_PARTICLE_RADIUS,
    "pouch": POUCH_PARTICLE_RADIUS,
}
APPROACH_RISE = 0.08
CARRY_BOTTOM = (
    0.20  # goods' bottom above the worktop over the rails (handles top +0.15)
)
# Goods' bottom at release, as the example: above the bag mouth, so the fingers
# never enter the bag (the hanging mouth sits ~0.02 m above the worktop and is
# open ~183 x 300 mm). Releasing inside the mouth let the fingers touch the film:
# the bag shook while the right gripper was in it and was kicked into a lasting
# swing when the open gripper rose after the pouch release.
RELEASE_BOTTOM = 0.04
# An unloaded two-item bag sits higher than the previous cube-loaded bag.
# Keep the gripper base above its free handles during bread release; this also
# brings the bread TCP release height back to the source demo's +0.32 m.
# The mirrored milk drop is nearer the right handle; its base needs extra
# height while the long pouch continues to hang down from the sealed edge.
RELEASE_RISE = {"bread": 0.025, "pouch": 0.040}
RETREAT_BOTTOM = 0.20
__all__ = ["main"]


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, default=lambda item: np.asarray(item).tolist())
        + "\n"
    )


def grip_rotation(side: str):
    """Downward gripper turned so the fingers close along world Y, leaning back."""
    from scipy.spatial.transform import Rotation

    return (
        Rotation.from_euler("y", -PITCH_DEG[side], degrees=True)
        * Rotation.from_euler("z", YAW_DEG, degrees=True)
        * Rotation.from_quat(DOWNWARD_X_FINGERS)
    )


class SideBaker(TrajectoryBaker):
    """M02's offline IK with per-arm orientations and finger openings."""

    def solve_sides(self, targets, fingers, iterations: int) -> np.ndarray:
        wp = self.wp
        for side, objective in zip(("left", "right"), self.rotations, strict=True):
            objective.set_target_rotation(
                0, wp.vec4(*grip_rotation(side).as_quat().astype(np.float32))
            )
        for objective, point in zip(self.positions, targets, strict=True):
            objective.set_target_position(0, wp.vec3(*point))
        self.ik.step(self.q, self.q, iterations=iterations)
        q = np.clip(self.q.numpy()[0], self.lower, self.upper)
        for side, value in zip(("LEFT", "RIGHT"), fingers, strict=True):
            for k in (1, 2):
                q[self.coords[f"{side}_FINGER{k}_JOINT"]] = value
        self.q.assign(q.reshape(1, -1))
        return q

    def lowest_side_finger_z(self, side: str) -> float:
        """Lowest collision-mesh point of one side's fingers."""
        from scipy.spatial.transform import Rotation

        poses = self.body_poses()
        return min(
            float(
                (
                    Rotation.from_quat(poses[b][3:]).apply(self.finger_meshes[n])
                    + poses[b][:3]
                )[:, 2].min()
            )
            for n, b in self.fingers.items()
            if n.startswith(side)
        )


def keyframes(
    grasp: dict,
    close: dict,
    release_extra: dict = RELEASE_EXTRA,
    hold_test: str | None = None,
    pouch_grasp_y_offset: float = 0.0,
) -> list:
    """(time, [left, right] TCP targets, [left, right] finger openings)."""
    up = np.array((0.0, 0.0, 1.0))
    drops = {"block": BLOCK_DROP, "bread": BREAD_DROP, "pouch": POUCH_DROP}
    keys: list = []
    held = {GOOD_ARMS[item]: grasp[item] + 0.15 * up for item in GOODS}
    home = dict(held)
    fingers = dict(OPEN)

    def release_open(item, side):
        return (
            min(OPEN[side], close[item] + release_extra[item])
            if item in release_extra
            else OPEN[side]
        )

    def key(name):
        t = 0.0 if name is None else end_of(name)
        keys.append(
            (
                t,
                np.array((held["left"], held["right"])),
                np.array((fingers["left"], fingers["right"])),
            )
        )

    key(None)
    key("home")
    for item in GOODS:
        side = GOOD_ARMS[item]
        g = grasp[item]
        offset = g[2] - TABLE_Z - REST_BOTTOM[item]  # TCP above the good's bottom
        carry_z = TABLE_Z + CARRY_BOTTOM + offset
        drop_xy = drops[item].copy()
        if item == "pouch":
            if pouch_grasp_y_offset:
                carry_z += 0.06  # clear the handles with a pouch hanging by its seam
        pick_open = 0.007 if item == "pouch" and pouch_grasp_y_offset else OPEN[side]
        release_bottom = RELEASE_BOTTOM + RELEASE_RISE.get(item, 0.0)
        drop = np.array((*drop_xy, TABLE_Z + release_bottom + offset))
        for name, point, opening in (
            (
                "approach",
                np.array((*g[:2], max(g[2] + APPROACH_RISE, held[side][2]))),
                pick_open,
            ),
            ("descend", g, pick_open),
            ("close", g, close[item]),
            ("settle", g, close[item]),
            ("lift", np.array((*g[:2], carry_z)), close[item]),
            (
                "carry",
                np.array((*(g[:2] if item == hold_test else drop[:2]), carry_z)),
                close[item],
            ),
            ("lower", drop, close[item]),
            ("release", drop, release_open(item, side)),
            (
                "retreat",
                np.array((*drop[:2], TABLE_Z + RETREAT_BOTTOM + offset)),
                release_open(item, side),
            ),
        ):
            held[side], fingers[side] = point, opening
            key(f"{item} {name}")
        if side == "left":
            held["left"], fingers["left"] = home["left"], OPEN["left"]
            key("left home")
    held["right"], fingers["right"] = home["right"], OPEN["right"]
    key("right home")
    key("settle")
    return keys


def bake(baker: SideBaker, utils, args) -> dict:
    """Pick grasp heights from finger geometry, then bake every frame."""
    close = {
        "block": args.block_close,
        "bread": args.bread_close,
        "pouch": args.pouch_close,
    }
    starts = {
        "block": BLOCK_START,
        "bread": np.asarray(args.bread_xy),
        "pouch": POUCH_START + np.array((0.0, args.pouch_grasp_y_offset)),
    }
    park = {
        GOOD_ARMS[item]: np.array((*starts[item], TABLE_Z + 0.15)) for item in GOODS
    }
    grasp = {}
    for item in GOODS:
        side = GOOD_ARMS[item]
        # Lower the TCP until that arm's lowest finger point is the clearance up.
        z = TABLE_Z + 0.03
        for _ in range(3):
            point = np.array((*starts[item], z))
            targets = (
                np.array((point, park["right"]))
                if side == "left"
                else np.array((park["left"], point))
            )
            baker.solve_sides(targets, (OPEN["left"], OPEN["right"]), 300)
            z += TABLE_Z + TIP_CLEARANCE[item] - baker.lowest_side_finger_z(side)
        grasp[item] = np.array((*starts[item], z))
    extra = dict(RELEASE_EXTRA)
    if args.pouch_release_extra is not None:
        extra["pouch"] = args.pouch_release_extra
    frames = keyframes(grasp, close, extra, args.hold_test, args.pouch_grasp_y_offset)
    baker.solve_sides(frames[0][1], frames[0][2], 300)
    rows, tcp_targets, tcp_actual, gaps, lowest = [], [], [], [], []
    for frame in range(STEPS + 1):
        targets, fingers = utils.smoothstep(frames, frame / FPS)
        rows.append(
            baker.solve_sides(targets, fingers, 150 if frame == 0 else 30).copy()
        )
        tcp_targets.append(targets)
        tcp_actual.append(baker.tcp())
        gaps.append(baker.camera_gap())
        lowest.append(baker.lowest_finger_z())
    q = np.asarray(rows)
    arm = baker.arm
    margins = np.minimum(q[:, arm] - baker.lower[arm], baker.upper[arm] - q[:, arm])
    errors = np.linalg.norm(np.asarray(tcp_actual) - np.asarray(tcp_targets), axis=2)
    return {
        "names": baker.names,
        "targets": np.asarray(
            [[row[baker.coords[n]] for n in baker.names] for row in q], dtype=np.float32
        ),
        "tcp_target": np.asarray(tcp_targets),
        "tcp_baked": np.asarray(tcp_actual),
        "grasp_tcp": grasp,
        "checks": {
            "max_joint_step_rad": float(np.abs(np.diff(q[:, arm], axis=0)).max()),
            "min_limit_margin_rad": float(margins.min()),
            "frames_at_joint_limit": int((margins.min(axis=1) < 1e-4).sum()),
            "max_tcp_error_m": float(errors.max()),
            "worst_tcp_error_time_s": float(np.argmax(errors.max(axis=1)) / FPS),
            "min_hand_camera_gap_m": float(min(gaps)),
            "min_finger_height_above_table_m": float(min(lowest) - TABLE_Z),
            **{
                f"{k}_grasp_tcp_z_above_table_m": float(v[2] - TABLE_Z)
                for k, v in grasp.items()
            },
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument(
        "--until", type=float, default=None, help="pilot: stop early (s)"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-cuda-graph", action="store_true")
    parser.add_argument("--bake-only", action="store_true")
    parser.add_argument("--commands-from", type=Path, default=None)
    parser.add_argument(
        "--pouch-bulk-damping", type=float, default=POUCH_GAS["bulk_damping"]
    )
    parser.add_argument("--pouch-tri-damping", type=float, default=POUCH["tri_kd"])
    parser.add_argument("--pouch-edge-damping", type=float, default=POUCH["edge_kd"])
    parser.add_argument(
        "--bag-contact-stiffness", type=float, default=BAG_CONTACT["stiffness"]
    )
    parser.add_argument(
        "--bag-contact-damping", type=float, default=BAG_CONTACT["damping"]
    )
    parser.add_argument(
        "--bag-contact-friction",
        type=float,
        default=PARTICLE_CONTACT["dynamic_friction"],
    )
    parser.add_argument(
        "--pouch-contact-stiffness", type=float, default=POUCH_CONTACT["stiffness"]
    )
    parser.add_argument(
        "--pouch-contact-damping", type=float, default=POUCH_CONTACT["damping"]
    )
    parser.add_argument(
        "--pouch-contact-friction",
        type=float,
        default=POUCH_CONTACT["dynamic_friction"],
    )
    # Jaw gaps: ~49.6 mm on the 50 mm block (no preset deep overlap, as M05);
    # ~50 mm across the 55 mm bread. The milk pouch is pinched at its sealed
    # edge, with a 2.6 mm jaw gap across the two particle contact skins, rather
    # than squeezing the gas-filled body and storing a large release impulse.
    parser.add_argument("--block-close", type=float, default=0.0255)
    parser.add_argument("--bread-close", type=float, default=0.0257)
    parser.add_argument(
        "--pouch-release-extra",
        type=float,
        default=None,
        help="pilot: partial pouch release opening",
    )
    parser.add_argument("--eye", type=float, nargs=3, default=None, help="camera eye")
    parser.add_argument(
        "--look", type=float, nargs=3, default=None, help="camera target"
    )
    parser.add_argument(
        "--pouch-radius", type=float, default=POUCH_PARTICLE_RADIUS, help="pilot"
    )
    parser.add_argument(
        "--bread-mu",
        type=float,
        default=PARTICLE_CONTACT["dynamic_friction"],
        help="pilot",
    )
    parser.add_argument(
        "--friction-epsilon", type=float, default=FRICTION_EPSILON, help="pilot"
    )
    parser.add_argument(
        "--hold-test",
        default=None,
        help="pilot: hold this good still in the air during its carry window",
    )
    parser.add_argument(
        "--bread-xy", type=float, nargs=2, default=tuple(BREAD_START), help="pilot"
    )
    parser.add_argument("--bread-yaw", type=float, default=BREAD_YAW_DEG, help="pilot")
    parser.add_argument("--iterations", type=int, default=ITERATIONS, help="pilot")
    parser.add_argument("--substeps", type=int, default=SUBSTEPS, help="pilot")
    parser.add_argument(
        "--bread-stiffness-scale", type=float, default=1.0, help="pilot"
    )
    parser.add_argument(
        "--full-surface",
        action="store_true",
        help="pilot: full-surface contact on every shape (example)",
    )
    parser.add_argument("--block-mass", type=float, default=BLOCK_MASS)
    parser.add_argument("--block-friction", type=float, default=BLOCK_MU)
    parser.add_argument(
        "--no-block",
        action="store_true",
        default=True,
        help="omit the retired extra rigid cargo",
    )
    parser.add_argument("--block-node-contact", action="store_true", help="pilot")
    parser.add_argument(
        "--no-pouch-gas",
        action="store_true",
        help="pilot: closed pouch without pneumatics",
    )
    parser.add_argument(
        "--extra-stills",
        type=float,
        nargs="*",
        default=(),
        help="pilot: extra still times (s)",
    )
    parser.add_argument("--pouch-close", type=float, default=0.002)
    parser.add_argument(
        "--pouch-grasp-y-offset", type=float, default=-0.045, help="pilot"
    )
    return parser


def main() -> None:  # noqa: C901 - one linear recording script
    """Bake the trajectory, build the scene and record one preview."""
    args = _parser().parse_args()
    output = args.out
    source = output / "source/reference"
    source.mkdir(parents=True, exist_ok=True)
    for name in REFERENCE_NAMES:
        shutil.copy2(REFERENCE / name, source / name)
    inputs = output / "inputs"
    inputs.mkdir(exist_ok=True)
    robot_root = inputs / "w1_v030"
    if not robot_root.exists():
        shutil.copytree(ROBOT, robot_root)
    for name in SUPERMARKET_FILES:
        shutil.copy2(SUPERMARKET / name, inputs / name)
    shutil.copy2(__file__, output / "source/c01.py")
    shutil.copy2(
        Path(__file__).resolve().parents[3] / "experiments/dexuni/c01_common.py",
        output / "source/c01_common.py",
    )
    urdf = robot_root / "robot.urdf"
    sys.path.insert(0, str(source))
    utils = importlib.import_module("dexuni_w1_two_way_utils")

    import dexsim
    import dexsim.spawn as spawn
    import imageio.v2 as imageio
    import newton
    import warp as wp
    from PIL import Image, ImageDraw, ImageFont
    from scipy.spatial.transform import Rotation
    from dexsim.engine.newton_physics import (
        ContactMaterialSchedule,
        DexUniSolverCfg,
        FilmDragControl,
        KinematicJointTrajectoryControl,
        NewtonCfg,
    )
    from dexsim.engine.newton_physics.backend_registry import get_newton_backend

    # ---- Frozen inputs ----
    bag_asset = json.loads((inputs / "shopping_bag.json").read_text())
    bag_material = bag_asset["simulation_material"]
    bag_local = np.asarray(bag_asset["vertices"], dtype=np.float64)
    bag_triangles = np.asarray(bag_asset["triangles"], dtype=np.int32)
    bag_vertices = bag_world(bag_local)
    groups = bag_groups(bag_local)
    with np.load(inputs / "bread.npz") as data:
        bread_local, bread_tets = (
            data["vertices"].astype(np.float64),
            data["tetrahedra"],
        )
    yaw = Rotation.from_euler("z", args.bread_yaw, degrees=True).as_matrix()
    bread_vertices = bread_local @ yaw.T
    bread_vertices += np.array(
        (
            *args.bread_xy,
            TABLE_Z
            + BREAD_PARTICLE_RADIUS
            + CONTACT_MARGIN
            - bread_vertices[:, 2].min(),
        )
    )
    pouch_local, pouch_triangles, pouch_rest_volume = pouch_mesh()
    pouch_vertices = pouch_local + np.array(
        (*POUCH_START, TABLE_Z + args.pouch_radius - pouch_local[:, 2].min())
    )
    if args.commands_from is not None:
        recorded = np.load(args.commands_from / "inputs/joint_commands.npz")
        recorded_manifest = json.loads(
            (args.commands_from / "manifest.json").read_text()
        )
        assert np.array_equal(recorded["time"], np.arange(STEPS + 1) / FPS)
        baked = {
            "names": recorded["joint_names"].tolist(),
            **{
                key: recorded[key].copy()
                for key in ("targets", "tcp_target", "tcp_baked")
            },
            "checks": recorded_manifest["trajectory_checks"],
        }
        shutil.copy2(
            args.commands_from / "inputs/joint_commands.npz",
            inputs / "joint_commands.npz",
        )
    else:
        with wp.ScopedDevice(args.device):
            baked = bake(SideBaker(urdf, utils, args.device), utils, args)
        np.savez_compressed(
            inputs / "joint_commands.npz",
            joint_names=np.asarray(baked["names"]),
            targets=baked["targets"],
            time=np.arange(STEPS + 1) / FPS,
            tcp_target=baked["tcp_target"],
            tcp_baked=baked["tcp_baked"],
        )
    pouch_gas = {**POUCH_GAS, "bulk_damping": args.pouch_bulk_damping}
    pouch_material = {
        **POUCH,
        "tri_kd": args.pouch_tri_damping,
        "edge_kd": args.pouch_edge_damping,
    }
    bag_contact = {
        **PARTICLE_CONTACT,
        "stiffness": args.bag_contact_stiffness,
        "damping": args.bag_contact_damping,
        "dynamic_friction": args.bag_contact_friction,
    }
    pouch_contact = {
        **POUCH_CONTACT,
        "stiffness": args.pouch_contact_stiffness,
        "damping": args.pouch_contact_damping,
        "dynamic_friction": args.pouch_contact_friction,
    }
    np.savez_compressed(
        inputs / "goods_meshes.npz",
        bag_vertices=bag_vertices,
        bag_triangles=bag_triangles,
        bag_mouth_group=groups["mouth"],
        bag_bottom_group=groups["bottom"],
        bread_vertices=bread_vertices,
        bread_tetrahedra=bread_tets,
        pouch_vertices=pouch_vertices,
        pouch_triangles=pouch_triangles,
        pouch_rest_volume=pouch_rest_volume,
    )

    full_surface_targets = {
        name: None for name, *_ in rack_parts(bag_asset["rack_rail_centers_local"])
    }
    full_surface_targets.update({name: None for name, *_ in worktop_pieces()})
    full_surface_targets.update(rack_foot=None)
    if not args.no_block and not args.block_node_contact:
        full_surface_targets.update(rigid_block=None)
    width, height = 1280, 720
    eye = tuple(args.eye) if args.eye else (3.30, 0.0, 2.15)
    target = tuple(args.look) if args.look else (0.36, 0.0, 0.84)
    hfov = 60.0
    config = dexsim.WorldConfig()
    config.use_default_physics = False
    config.open_windows = False
    config.renderer = dexsim.types.Renderer.OFFLINERT
    config.newton_cfg = NewtonCfg(
        device=args.device,
        dt=1 / FPS,
        num_substeps=args.substeps,
        solver_cfg=DexUniSolverCfg(
            mujoco_articulation_paths=("w1",),
            joint_mode="kinematic",
            contact_mode="full",
            coupling="one_way",
            # Full-surface rigid-soft contact on the support shapes only (the
            # example uses it on every shape). The bag needs it on the rails:
            # with per-node contact the 10 mm rails pass between the slot nodes
            # and the bag falls. The fingers use per-node contact: in hold tests
            # the full-surface path let the gripped bread slide 4-8 mm/s whatever
            # its friction (0.4-2.0), smoothing (1e-3/1e-4), solver iterations
            # (24/48) or squeeze, while per-node contact held it (0 mm).
            full_surface_contact_targets=(
                None if args.full_surface else full_surface_targets
            ),
            vbd_options={
                "iterations": args.iterations,
                "friction_epsilon": args.friction_epsilon,
                "pneumatic_enable_incremental_volume": False,
                "enable_cuda_fast_path": True,
                "rigid_body_contact_buffer_size": 4096,
                "rigid_body_particle_contact_buffer_size": 8192,
                "rigid_contact_hard": False,
                "rigid_contact_history": False,
                **SELF_CONTACT,
            },
            collision_options={
                "soft_contact_max": 65536,
                "rigid_contact_max": 4096,
                "soft_contact_margin": SOFT_CONTACT_MARGIN,
                "include_static_kinematic_pairs": False,
                "enable_rigid_soft_full_surface_contact": True,
                "enable_cuda_fast_path": True,
                "contact_matching": "disabled",
            },
        ),
    )
    config.newton_cfg.use_cuda_graph = not args.no_cuda_graph
    preview = output / "preview"
    preview.mkdir(exist_ok=True)
    manifest = {
        "task": "C01",
        "mode": "native_scene_preview_only",
        "status": "recording",
        "user_scene_approved": False,
        "performance_measurements_started": False,
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "executable": sys.executable,
        "dexsim_version": dexsim.__version__,
        "dexsim_module": dexsim.__file__,
        "newton_version": newton.__version__,
        "newton_module": newton.__file__,
        "warp_version": wp.__version__,
        "arguments": {k: str(v) for k, v in vars(args).items()},
        "coupling": "one_way",
        "joint_mode": "kinematic",
        "robot": "W1 V030 + Pika parallel grippers; baked IK trajectory replayed kinematically; preset finger openings",
        "scene": {
            "table_z": TABLE_Z,
            "worktop_slabs": [
                {"name": name, "centre": centre, "size": size}
                for name, centre, size in worktop_pieces()
                if name.startswith("table_")
            ],
            "bag_position": BAG_POSITION,
            "bag_yaw_deg": BAG_YAW_DEG,
            "bag_material": bag_material,
            "bag_particle_contact": bag_contact,
            "film_drag": FILM_DRAG,
            "bag_mouth_local_z": BAG_MOUTH_LOCAL_Z,
            "rack_mu": RACK_MU,
            "goods": (
                ["bread", "pouch"] if args.no_block else ["block", "bread", "pouch"]
            ),
            "good_arms": GOOD_ARMS,
            "block": (
                None
                if args.no_block
                else {
                    "start_xy": BLOCK_START,
                    "size_m": 2 * BLOCK_HALF,
                    "mass_kg": args.block_mass,
                    "ke": BLOCK_KE,
                    "mu": args.block_friction,
                }
            ),
            "bread": {
                "start_xy": args.bread_xy,
                "yaw_deg": args.bread_yaw,
                **BREAD,
                "particle_radius": BREAD_PARTICLE_RADIUS,
                "contact": {**PARTICLE_CONTACT, "dynamic_friction": args.bread_mu},
            },
            "pouch": {
                "start_xy": POUCH_START,
                "source": REFERENCE_DEMO + "::pouch_mesh",
                "yaw_deg": 0.0,
                **pouch_material,
                "particle_radius": args.pouch_radius,
                "gas": {"mode": "ISOTHERMAL", **pouch_gas},
                "contact": pouch_contact,
                "rest_volume_m3": pouch_rest_volume,
            },
            "drops_xy": {"block": BLOCK_DROP, "bread": BREAD_DROP, "pouch": POUCH_DROP},
            "contact": {
                "margin": CONTACT_MARGIN,
                "gap": CONTACT_GAP,
                "soft_contact_margin": SOFT_CONTACT_MARGIN,
                "friction_epsilon": args.friction_epsilon,
                "robot_ke": ROBOT_KE,
                "left_finger_ke": LEFT_FINGER_KE,
                "right_finger_ke": RIGHT_FINGER_KE,
                "finger_mu": FINGER_MU,
                "table_ke": TABLE_KE,
                "table_mu": TABLE_MU,
                "self_contact": SELF_CONTACT,
            },
            "full_surface_contact_targets": (
                None if args.full_surface else sorted(full_surface_targets)
            ),
            "solver": {
                "substeps": args.substeps,
                "iterations": args.iterations,
                "enable_cuda_fast_path": True,
                "pneumatic_enable_incremental_volume": False,
                "collision_enable_cuda_fast_path": True,
            },
            "gripper_pitch_deg": PITCH_DEG,
            "gripper_yaw_deg": YAW_DEG,
            "pouch_grasp_y_offset_m": args.pouch_grasp_y_offset,
            "pouch_seam_carry_extra_m": 0.06 if args.pouch_grasp_y_offset else 0.0,
            "open": OPEN,
            "close": {
                "block": args.block_close,
                "bread": args.bread_close,
                "pouch": args.pouch_close,
            },
            "tip_clearance": TIP_CLEARANCE,
            "release_extra": {
                **RELEASE_EXTRA,
                **(
                    {"pouch": args.pouch_release_extra}
                    if args.pouch_release_extra is not None
                    else {}
                ),
            },
            "carry_bottom_m": CARRY_BOTTOM,
            "release_bottom_m": {
                item: RELEASE_BOTTOM + RELEASE_RISE.get(item, 0.0) for item in GOODS
            },
            "phases": PHASES,
        },
        "trajectory_checks": baked["checks"],
        "camera": {
            "eye": eye,
            "target": target,
            "resolution": [width, height],
            "hfov": hfov,
        },
        "reference_sha256": {
            n: hashlib.sha256((source / n).read_bytes()).hexdigest()
            for n in REFERENCE_NAMES
        },
        "input_sha256": {
            n: hashlib.sha256((inputs / n).read_bytes()).hexdigest()
            for n in SUPERMARKET_FILES
        },
        "robot_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
    }
    _write_json(output / "manifest.json", manifest)
    print(json.dumps(baked["checks"], indent=2), flush=True)
    if args.bake_only:
        manifest["status"] = "baked_inputs_only"
        _write_json(output / "manifest.json", manifest)
        return
    if baked["checks"]["max_tcp_error_m"] > 0.002:
        raise RuntimeError(
            "Baked TCP error exceeds 2 mm; fix the trajectory before simulation"
        )

    world = dexsim.World(config)
    world.set_manual_update(True)
    scene = camera = writer = None
    states: dict[str, list] = {}
    metrics = []
    try:
        builder = spawn.SceneBuilder(world)
        builder.add_urdf(
            "w1",
            str(urdf),
            per_env=False,
            newton_collision=spawn.NewtonCollisionDesc(
                ke=ROBOT_KE,
                kd=CONTACT_KD,
                mu=FINGER_MU,
                margin=CONTACT_MARGIN,
                gap=CONTACT_GAP,
                force_sdf=True,
            ),
        )
        builder.add_runtime_control(
            KinematicJointTrajectoryControl(
                "w1",
                baked["targets"],
                fps=FPS,
                joint_names=baked["names"],
                initialize_pose=True,
            )
        )

        def material(name, color, **options):
            builder.materials[name] = spawn.MaterialDesc(
                name=name, base_color=(*color, 1.0), **options
            )
            return name

        def box(name, center, size, color, *, density=None, mu=TABLE_MU, ke=TABLE_KE):
            geometry = spawn.GeometryDesc.cube(tuple(size))
            collision = spawn.CollisionDesc.from_geometry(geometry)
            collision.newton = spawn.NewtonCollisionDesc(
                ke=ke, kd=CONTACT_KD, mu=mu, margin=CONTACT_MARGIN, gap=CONTACT_GAP
            )
            obj = builder.add_cube(
                name,
                tuple(size),
                pose=utils.pose_at(center),
                physics=(
                    spawn.RigidBodyPhysicsDesc.static()
                    if density is None
                    else spawn.RigidBodyPhysicsDesc.dynamic(density=density)
                ),
                collisions=[collision],
                per_env=False,
            )
            obj.renders[0].material_ref = material(name, color)

        def tube(name, a, b, radius):
            direction = b - a
            length = float(np.linalg.norm(direction))
            rotation = Rotation.align_vectors([direction / length], [[0, 0, 1]])[
                0
            ].as_matrix()
            pose = np.eye(4)
            pose[:3, :3], pose[:3, 3] = rotation, 0.5 * (a + b)
            geometry = spawn.GeometryDesc.capsule(radius, length / 2)
            collision = spawn.CollisionDesc.from_geometry(geometry)
            # Example rack contact: ke 2e5, kd 30, margin 1 mm, mu 0.35.
            collision.newton = spawn.NewtonCollisionDesc(
                ke=2e5,
                kd=30.0,
                mu=RACK_MU,
                margin=0.001,
                gap=None,
                has_particle_collision=True,
            )
            obj = builder.add_capsule(
                name,
                radius,
                length / 2,
                pose=pose,
                physics=spawn.RigidBodyPhysicsDesc.static(),
                collisions=[collision],
                per_env=False,
            )
            obj.renders[0].material_ref = material(
                name, (0.62, 0.65, 0.67), metallic=0.6, roughness=0.35
            )

        ground = spawn.CollisionDesc.from_geometry(spawn.GeometryDesc.plane(8.0))
        ground.newton = spawn.NewtonCollisionDesc(ke=2e4, kd=100.0, mu=0.8, gap=0.002)
        floor = builder.add_plane(
            "ground",
            8.0,
            physics=spawn.RigidBodyPhysicsDesc.static(),
            collisions=[ground],
            per_env=False,
        )
        floor.renders[0].material_ref = material("floor", (0.55, 0.56, 0.54))
        for name, center, size in worktop_pieces():
            box(name, center, size, (0.5, 0.5, 0.5))
        for name, a, b, radius in rack_parts(bag_asset["rack_rail_centers_local"]):
            tube(name, a, b, radius)
        foot_centre, foot_size = rack_foot()
        box("rack_foot", foot_centre, foot_size, (0.18, 0.2, 0.21))
        if not args.no_block:
            box(
                "rigid_block",
                (*BLOCK_START, TABLE_Z + BLOCK_HALF + 2 * CONTACT_MARGIN),
                (2 * BLOCK_HALF,) * 3,
                (0.18, 0.55, 0.78),
                density=args.block_mass / (2 * BLOCK_HALF) ** 3,
                mu=args.block_friction,
                ke=BLOCK_KE,
            )
        # Shopping bag (example asset, material and film drag; textured as the example).
        bag_uv = np.column_stack(
            ((bag_local[:, 0] + 0.16) / 0.32, bag_local[:, 2] / 0.40)
        ).astype(np.float32)
        builder.materials["film"] = spawn.MaterialDesc(
            name="film",
            base_color=(1.0, 1.0, 1.0, 0.90),
            base_color_map=str(inputs / "bag_print.png"),
            roughness=0.27,
            double_sided=True,
        )
        bag_mesh = spawn.GeometryDesc.mesh(
            vertices=bag_vertices, triangles=bag_triangles
        )
        bag_mesh.uv_coords = bag_uv
        builder.add_clothbody(
            "bag",
            spawn.RenderDesc.from_geometry(bag_mesh, material_ref="film"),
            particle_radius=bag_material["particle_radius"],
            physics=spawn.ClothPhysicsDesc(
                surface_density=bag_material["density"],
                **{
                    k: bag_material[k]
                    for k in ("tri_ke", "tri_ka", "tri_kd", "edge_ke", "edge_kd")
                },
                contact_material=spawn.NewtonParticleContactMaterialDesc(**bag_contact),
            ),
            per_env=False,
        )
        builder.add_runtime_control(FilmDragControl("bag", **FILM_DRAG))
        # Bread (example tetrahedral asset and material).
        bread_desc = builder.create_soft_mesh(
            "bread",
            bread_vertices.astype(np.float32),
            bread_tets,
            particle_radius=BREAD_PARTICLE_RADIUS,
            physics=spawn.SoftBodyPhysicsDesc(
                **{
                    **BREAD,
                    "k_mu": BREAD["k_mu"] * args.bread_stiffness_scale,
                    "k_lambda": BREAD["k_lambda"] * args.bread_stiffness_scale,
                },
                contact_material=spawn.NewtonParticleContactMaterialDesc(
                    **{**PARTICLE_CONTACT, "dynamic_friction": args.bread_mu}
                ),
            ),
            per_env=False,
        )
        surface = np.unique(bread_desc.surface_triangles)
        bread_uv = (bread_local[surface, :2] / np.array((0.055, 0.070)) + 0.5).astype(
            np.float32
        )
        builder.materials["bread"] = spawn.MaterialDesc(
            name="bread",
            base_color=(1.0, 1.0, 1.0, 1.0),
            base_color_map=str(inputs / "bread_crust.png"),
            roughness=0.82,
            double_sided=True,
        )
        bread_desc.mesh.uv_coords = bread_uv
        bread_desc.mesh = spawn.RenderDesc.from_geometry(
            bread_desc.mesh, material_ref="bread"
        )
        # Original supermarket milk pouch, rotated only through scene layout.
        builder.materials["pouch"] = spawn.MaterialDesc(
            name="pouch",
            base_color=(1.0, 1.0, 1.0, 1.0),
            base_color_map=str(inputs / "milk_pouch.png"),
            roughness=0.38,
            double_sided=True,
        )
        pouch_geometry = spawn.GeometryDesc.mesh(
            vertices=pouch_vertices, triangles=pouch_triangles
        )
        pouch_geometry.uv_coords = (
            pouch_local[:, :2] / np.array((0.066, 0.092)) + 0.5
        ).astype(np.float32)
        builder.add_clothbody(
            "pouch",
            spawn.RenderDesc.from_geometry(pouch_geometry, material_ref="pouch"),
            particle_radius=args.pouch_radius,
            physics=spawn.ClothPhysicsDesc(
                **pouch_material,
                contact_material=spawn.NewtonParticleContactMaterialDesc(
                    **pouch_contact
                ),
                pneumatic=(
                    None
                    if args.no_pouch_gas
                    else spawn.PneumaticPhysicsDesc(
                        mode=spawn.PneumaticMode.ISOTHERMAL, **pouch_gas
                    )
                ),
            ),
            per_env=False,
            validate_mesh=True,
        )
        builder.add_runtime_control(
            ContactMaterialSchedule(
                "w1",
                {"stiffness": ((0.0, LEFT_FINGER_KE),)},
                link_names=("left_finger1", "left_finger2"),
            )
        )
        builder.add_runtime_control(
            ContactMaterialSchedule(
                "w1",
                {"stiffness": ((0.0, RIGHT_FINGER_KE),)},
                link_names=("right_finger1", "right_finger2"),
            )
        )
        builder.add_light(
            spawn.LightDesc(
                name="key_light",
                light_type=dexsim.render.LightType.POINT,
                pose=utils.pose_at((1.5, -1.0, 3.0)),
                intensity=50.0,
                shadow=False,
            )
        )
        builder.add_camera(
            spawn.CameraDesc(
                "preview",
                resolution=(width, height),
                near=0.01,
                far=10.0,
                headless=True,
            )
        )
        scene = builder.finalize()
        world.enable_physics(True)
        robot = scene.get_articulation("w1")
        layout = {
            item.name.rsplit("/", 1)[-1]: item.dof_start
            for item in robot.joint_dof_layout
        }
        joint_ids = [layout[name] for name in baked["names"]]
        manifest["joint_state_order"] = baked["names"]
        handles = {
            name: scene.get_particle_set(name) for name in ("bag", "bread", "pouch")
        }
        for name, handle in handles.items():
            if not np.isfinite(handle.get_particle_positions().numpy()).all():
                raise RuntimeError(f"Nonfinite initial {name}")
        if not np.allclose(
            handles["bag"].get_particle_positions().numpy(), bag_vertices, atol=1e-5
        ):
            raise RuntimeError("Bag node order differs from the authored mesh")
        block_obj = None if args.no_block else scene.get_object("rigid_block")
        camera = scene.get_camera("preview")
        vfov = 2 * np.arctan(np.tan(np.deg2rad(hfov) / 2) * height / width)
        camera.set_perspective(float(np.rad2deg(vfov)), width / height, 0.01, 10.0)
        camera.look_at(eye, target, (0, 0, 1))
        camera.open_camera()
        backend = get_newton_backend(world)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 20)
        except OSError:
            font = ImageFont.load_default()
        if not args.pilot:
            writer = imageio.get_writer(
                str(preview / "dexsim.mp4"),
                fps=30,
                codec="libx264",
                quality=8,
                macro_block_size=2,
            )
        still_times = (
            [0.0, 2.0]
            + [
                end_of(f"{i} {p}")
                for i in GOODS
                for p in ("settle", "carry", "release")
            ]
            + [DURATION]
        )
        stills = {round(t * FPS) for t in (*still_times, *args.extra_stills)}
        tcp_local = np.array((*utils.TCP, 1.0))
        last = STEPS if args.until is None else min(STEPS, round(args.until * FPS))
        for frame in range(last + 1):
            if frame:
                world.update(1 / FPS)
            time = float(scene.get_simulation_time())
            if abs(time - frame / FPS) > 1e-7:
                raise RuntimeError(
                    f"Actual simulation clock mismatch at frame{frame}: {time}"
                )
            actual_qpos = robot.get_current_qpos()[joint_ids]
            if not np.allclose(actual_qpos, baked["targets"][frame], rtol=0, atol=2e-5):
                raise RuntimeError(f"Named joint replay mismatch at frame{frame}")
            particles = {
                n: h.get_particle_positions().numpy() for n, h in handles.items()
            }
            block_pose = (
                None if block_obj is None else np.asarray(block_obj.get_world_pose())
            )
            tcp = np.array(
                [
                    (robot.get_link_pose(f"{s}_gripper_base") @ tcp_local)[:3]
                    for s in ("left", "right")
                ]
            )
            if not (
                all(np.isfinite(v).all() for v in particles.values())
                and (block_pose is None or np.isfinite(block_pose).all())
            ):
                raise AssertionError(f"Nonfinite state at frame {frame}")
            centres = {
                "bread": particles["bread"].mean(axis=0),
                "pouch": particles["pouch"].mean(axis=0),
            }
            if block_pose is not None:
                centres["block"] = block_pose[:3, 3]
            row = packing_metrics(particles["bag"], centres, groups)
            row.update(
                frame=frame,
                time=time,
                phase=_phase(time),
                tcp=tcp,
                tcp_error_m=np.linalg.norm(tcp - baked["tcp_baked"][frame], axis=1),
            )
            if max(row["tcp_error_m"]) > 2e-5:
                raise RuntimeError(f"Actual TCP replay mismatch at frame{frame}")
            if frame % 120 == 0:
                print(f"frame={frame}/{last} t={time:.2f}", flush=True)
            metrics.append(row)
            for key, value in (
                ("time", time),
                ("qpos", actual_qpos),
                ("target_q", baked["targets"][frame]),
                ("tcp", tcp),
                *particles.items(),
            ):
                states.setdefault(key, []).append(np.array(value, copy=True))
            if block_pose is not None:
                states.setdefault("block_pose", []).append(block_pose.copy())
            if (writer is not None and frame % 2 == 0) or frame in stills:
                backend.sync_to_dexsim(world)
                world._flush_render_updates()
                camera.render()
                image = Image.fromarray(np.asarray(camera.get_rgb_map())[..., :3])
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, width, 42), fill="black")
                packed = sum(row[f"{g}_in_bag"] for g in centres)
                arm = GOOD_ARMS.get(row["phase"].split()[0])
                phase_label = row["phase"] + (f" ({arm})" if arm else "")
                draw.text(
                    (14, 10),
                    f"DexSim | C01 supermarket packing (one-way) | t={time:5.2f}s | {phase_label} | goods in bag: {packed}/{len(centres)}",
                    font=font,
                    fill="white",
                )
                if writer is not None and frame % 2 == 0:
                    writer.append_data(np.asarray(image))
                if frame in stills:
                    image.save(preview / f"still_{time:05.2f}s.jpg")
        effective = dict(scene.get_solver_diagnostics())
        effective.update(
            sim_time=float(scene.get_simulation_time()),
            step_index=int(backend.step_index),
        )
        manifest.update(status="completed", frames=last, effective=effective)
    except Exception as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        if writer is not None:
            writer.close()
        if states:
            np.savez_compressed(
                preview / "states.npz", **{k: np.stack(v) for k, v in states.items()}
            )
        if metrics:
            _write_json(preview / "metrics.json", metrics)
        manifest["ended_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        _write_json(output / "manifest.json", manifest)
        if camera is not None:
            camera.close_camera()
        if scene is not None:
            scene.close()
    print(f"Scene preview ready: {preview}", flush=True)


if __name__ == "__main__":
    main()
