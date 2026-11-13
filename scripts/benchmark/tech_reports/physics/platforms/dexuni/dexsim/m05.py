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
"""Record the native W1 inflatable-bag place and rigid load scene (M05).

Run: /home/oem/code/engine/dexsim/.env-3.12/bin/python scripts/benchmark/tech_reports/physics/platforms/dexuni/dexsim/m05.py
Pilot: add ``--pilot --out <scratch dir>`` to save still frames instead of the video.

One-way coupling with kinematic robot joints (latest user choice). The IK
trajectory is baked offline. The bag asset comes from the specified
inflatable-bag demo. This task uses elastic membranes and sealed isothermal
gas; bending plasticity is disabled.
The robot stays W1 V030 with Pika parallel grippers. The bag lies flat on the
worktop; the right gripper, pointing down, squeezes it across its width,
carries it and lowers it onto the target. After an unloaded recovery window, the
left gripper sets a 30 mm, 4 kg rigid block on the bag centre. Compression and grip
stability must be verified from the completed episode.
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

from scripts.benchmark.tech_reports.physics.experiments.dexuni.m05_common import (
    BAG_SCALE,
    BAG_REST_BULGE,
    BAG_START,
    BAG_TARGET,
    BLOCK_HALF,
    BLOCK_MASS,
    BLOCK_START,
    OUTPUT,
    PHASES,
    TIMING,
    DURATION,
    TABLE_Z,
    chip_bag_mesh,
    load_metrics,
    phase as _phase,
    top_centre_group,
    supports,
)
from scripts.benchmark.tech_reports.physics.platforms.dexuni.dexsim.m02 import (
    TrajectoryBaker,
)

REFERENCE = Path("/home/oem/code/engine/dexsim/examples/python/physics/multiphys")
REFERENCE_DEMO = "demo_dexuni_w1_plastic_inflatable_bag_pick_release.py"
BAG_OBJ = Path("/tmp/mjvbd_v2_inflatable_bag/newton_chip_bag_sealed_cylinder.obj")
ROBOT = Path("/home/oem/.cache/dexsim/resources/mjvbd_v2/w1_bag_packing/w1_v030")
REFERENCE_NAMES = (
    "dexuni_w1_two_way_utils.py",
    "demo_dexuni_w1_bag_packing.py",
    "dexuni_asset_utils.py",
    REFERENCE_DEMO,
)
FPS = 60
STEPS = round(DURATION * FPS)
# ---- Inflatable-bag example parameters ----
# Keep the established ordinary budget while validating the scene changes.
SUBSTEPS = 8
ITERATIONS = 24
BAG_DENSITY = 0.12
BAG_AMBIENT_PRESSURE = 101_325.0
BAG_MAX_ABSOLUTE_PRESSURE = 500_000.0
BAG_BULK_DAMPING = 3_000_000.0
BAG_PARTICLE_RADIUS = 0.00075
BAG_TRI_KD = 60.0
BAG_EDGE_KE = 5.0  # same initial effective stiffness as old 20 x 0.25, now elastic
BAG_EDGE_KD = 0.4
# Compliant elastic membrane and low initial gauge pressure allow local
# indentation while the sealed gas supports the rest of the pouch.
BAG_GAUGE_PRESSURE = 16_000.0
BAG_TRI_KE = 2.5e3
BAG_TRI_KA = 2.5e3
# Small physical offsets remove visible gaps. The soft margin is only the
# collision detection envelope; support uses particle radius and shape margin.
CONTACT_MARGIN = 0.0005
SOFT_CONTACT_MARGIN = 0.002  # detection envelope, not the support offset
FRICTION_EPSILON = 1e-4
CONTACT_GAP = 0.0
CONTACT_KE = 2.0e5
HAND_CONTACT_KE = 6.0e5  # example hand-bag stiffness: applied to the right fingers only
ROBOT_KE = 2.0e4  # bodies other than the separately configured fingers
CONTACT_KD = 100.0
CONTACT_MU = 2.0
BLOCK_CONTACT_KE = 2.0e5
RELEASE_FRICTION = 1.0
TABLE_FRICTION = 0.9
RIGID_BODY_CONTACT_BUFFER_SIZE = 4096
RIGID_BODY_PARTICLE_CONTACT_BUFFER_SIZE = 8192
# ---- Task motion (this benchmark) ----
FINGER_LINKS = ("left_finger1", "left_finger2", "right_finger1", "right_finger2")
RIGHT_FINGER_LINKS = ("right_finger1", "right_finger2")
LEFT_HAND_LINKS = ("left_gripper_base", "left_finger1", "left_finger2")
# Both grippers point down, lean back 15 deg and are turned 90 deg about the
# vertical so the fingers close along world Y (bag width, block faces).
DOWNWARD_X_FINGERS = np.array((-np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0))
PITCH_DEG = 15.0
BLOCK_PITCH_DEG = 0.0
BLOCK_PLACE_PITCH_DEG = 5.0
YAW_DEG = 90.0
# Jaw gap ~ 2 q - 1.4 mm. Open: joint limit, ~98.6 mm around the bag.
BAG_OPEN = 0.05
BLOCK_OPEN = 0.045
TIP_CLEARANCE = 0.003  # lowest right finger point above the worktop at the bag grasp
# Vertical block grip covers its centre of mass. A tilted grip with higher
# fingertips slid out during the first diagnostic carry.
BLOCK_TIP_CLEARANCE = 0.005
PLACE_RAISE = 0.004  # bag place TCP above its grasp height
LIFT = 0.12
BLOCK_RELEASE_CLEARANCE = 0.001
BLOCK_REST_GAP = 2 * CONTACT_MARGIN
BLOCK_SPAWN_GAP = BLOCK_REST_GAP
RELEASE_TIME = 9.0  # right fingers start opening (bag release)
__all__ = ["main"]


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, default=lambda item: np.asarray(item).tolist())
        + "\n"
    )


def grip_rotation(side: int = 1, block_pitch: float = BLOCK_PITCH_DEG):
    """Orientation of both grippers as a SciPy rotation."""
    from scipy.spatial.transform import Rotation

    return (
        Rotation.from_euler(
            "y", -(block_pitch if side == 0 else PITCH_DEG), degrees=True
        )
        * Rotation.from_euler("z", YAW_DEG, degrees=True)
        * Rotation.from_quat(DOWNWARD_X_FINGERS)
    )


class SideBaker(TrajectoryBaker):
    """M02's offline IK with per-side finger openings."""

    def solve_sides(self, targets, fingers, iterations: int) -> np.ndarray:
        wp = self.wp
        for side, objective in enumerate(self.rotations):
            quat = wp.vec4(
                *grip_rotation(side, getattr(self, "block_pitch", BLOCK_PITCH_DEG))
                .as_quat()
                .astype(np.float32)
            )
            objective.set_target_rotation(0, quat)
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


def keyframes(block, bag, place, release, args) -> list:
    """(time, [left, right] TCP targets, [left, right] finger openings)."""
    up = np.array((0.0, 0.0, 1.0))
    left_home = block + 0.12 * up
    right_home = bag + 0.12 * up
    bag_lift = bag + LIFT * up
    bag_carry = np.array((*place[:2], bag_lift[2]))
    block_lift = block + LIFT * up
    block_carry = np.array((*release[:2], max(block_lift[2], release[2] + 0.06)))
    ao, ac, bo, bc = BAG_OPEN, args.bag_close, BLOCK_OPEN, args.block_close

    def key(t, left, right, fl, fr):
        return (t, np.array((left, right)), np.array((fl, fr)))

    t = TIMING
    return [
        key(0.0, left_home, right_home, bo, ao),
        key(0.5, left_home, right_home, bo, ao),
        key(2.0, left_home, bag + 0.06 * up, bo, ao),
        key(3.0, left_home, bag, bo, ao),
        key(4.0, left_home, bag, bo, ac),
        key(4.4, left_home, bag, bo, ac),
        key(6.0, left_home, bag_lift, bo, ac),
        key(8.0, left_home, bag_carry, bo, ac),
        key(RELEASE_TIME, left_home, place, bo, ac),
        key(9.6, left_home, place, bo, ao),
        key(10.1, left_home, place + 0.08 * up, bo, ao),
        key(10.6, left_home, right_home, bo, ao),
        key(t["unloaded reference"], left_home, right_home, bo, ao),
        key(t["block approach"], block + 0.06 * up, right_home, bo, ao),
        key(t["block descend"], block, right_home, bo, ao),
        key(t["block close"], block, right_home, bc, ao),
        key(t["block settle"], block, right_home, bc, ao),
        key(t["block lift"], block_lift, right_home, bc, ao),
        key(t["block carry"], block_carry, right_home, bc, ao),
        key(t["block lower"], release, right_home, bc, ao),
        key(t["block load hold"], release, right_home, bc, ao),
        key(t["block release"], release, right_home, bo, ao),
        key(t["left retreat"], release + 0.08 * up, right_home, bo, ao),
        key(DURATION, release + 0.08 * up, right_home, bo, ao),
    ]


def bake(baker: SideBaker, utils, args) -> dict:
    """Pick grasp heights from finger geometry, then bake every frame."""
    park_right = np.array((*BAG_START, TABLE_Z + 0.15))

    def grasp_z(side: str, xy, opening, other, clearance: float) -> float:
        # Lower the TCP until that side's lowest finger point is `clearance` up.
        z = TABLE_Z + 0.03
        for _ in range(3):
            point = np.array((*xy, z))
            targets = (
                np.array((point, other)) if side == "left" else np.array((other, point))
            )
            fingers = (opening, BAG_OPEN) if side == "left" else (BLOCK_OPEN, opening)
            baker.solve_sides(targets, fingers, 300)
            z += TABLE_Z + clearance - baker.lowest_side_finger_z(side)
        return z

    z = grasp_z("left", BLOCK_START, BLOCK_OPEN, park_right, BLOCK_TIP_CLEARANCE)
    block = np.array((*BLOCK_START, z))
    left_park = block + (0.0, 0.0, 0.12)
    bz = grasp_z("right", BAG_START, BAG_OPEN, left_park, TIP_CLEARANCE)
    bag = np.array((*BAG_START, bz))
    place = np.array((*BAG_TARGET, bz + PLACE_RAISE))
    tcp_to_bottom = z - (TABLE_Z + BLOCK_REST_GAP)
    release_xy = BAG_TARGET + np.array((args.release_dx, 0.0))
    release = np.array(
        (
            *release_xy,
            TABLE_Z + args.placed_top + BLOCK_RELEASE_CLEARANCE + tcp_to_bottom,
        )
    )
    frames = keyframes(block, bag, place, release, args)
    baker.solve_sides(frames[0][1], frames[0][2], 300)
    rows, tcp_targets, tcp_actual, gaps, lowest = [], [], [], [], []
    for frame in range(STEPS + 1):
        # Vertical contact during the lift/carry avoids a gravity-driven torque.
        # Tilt only when lowering at the central target, where vertical IK hits
        # a wrist limit; preserve a reachable slightly tilted placement posture.
        u = float(
            np.clip(
                (frame / FPS - TIMING["block carry"])
                / (TIMING["block lower"] - TIMING["block carry"]),
                0,
                1,
            )
        )
        baker.block_pitch = BLOCK_PLACE_PITCH_DEG * u**3 * (10 - 15 * u + 6 * u**2)
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
    names = baker.names
    margins = np.minimum(q[:, arm] - baker.lower[arm], baker.upper[arm] - q[:, arm])
    return {
        "names": names,
        "targets": np.asarray(
            [[row[baker.coords[n]] for n in names] for row in q], dtype=np.float32
        ),
        "tcp_target": np.asarray(tcp_targets),
        "tcp_baked": np.asarray(tcp_actual),
        "bag_grasp_tcp": bag,
        "bag_place_tcp": place,
        "block_grasp_tcp": block,
        "block_release_tcp": release,
        "checks": {
            "max_joint_step_rad": float(np.abs(np.diff(q[:, arm], axis=0)).max()),
            "min_limit_margin_rad": float(margins.min()),
            "frames_at_joint_limit": int((margins.min(axis=1) < 1e-4).sum()),
            "max_tcp_error_m": float(
                np.linalg.norm(
                    np.asarray(tcp_actual) - np.asarray(tcp_targets), axis=2
                ).max()
            ),
            "min_hand_camera_gap_m": float(min(gaps)),
            "min_finger_height_above_table_m": float(min(lowest) - TABLE_Z),
            "bag_grasp_tcp_z_above_table_m": float(bz - TABLE_Z),
            "block_grasp_tcp_z_above_table_m": float(z - TABLE_Z),
            "release_tcp_z_above_table_m": float(release[2] - TABLE_Z),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument(
        "--bake-only",
        action="store_true",
        help="save inputs without creating a simulation",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-cuda-graph", action="store_true")
    parser.add_argument(
        "--until", type=float, default=None, help="pilot: stop early (s)"
    )
    parser.add_argument("--bag-scale", type=float, default=BAG_SCALE)
    # Jaw gap ~30.6 mm across the ~80 mm wide bag; 0.024 (47 mm) and 0.034
    # (67 mm) only bulged the bag up, which slipped out at the lift (pilots).
    parser.add_argument("--bag-close", type=float, default=0.016)
    # Jaw gap about 30.1 mm around a 30 mm block, without preset deep overlap.
    parser.add_argument("--block-close", type=float, default=0.01575)
    # Release-height planning input, corrected for the observed carried block
    # rising along the fingers. It is not a measured bag surface height.
    parser.add_argument("--placed-top", type=float, default=0.040)
    # Block release TCP X offset from the bag target (carried-block offset, pilot).
    parser.add_argument("--release-dx", type=float, default=0.0)

    return parser


def main() -> None:
    """Bake the trajectory, build the scene and record one preview."""
    args = _parser().parse_args()
    output = args.out
    source = output / "source/reference"
    source.mkdir(parents=True, exist_ok=True)
    for name in REFERENCE_NAMES:
        shutil.copy2(REFERENCE / name, source / name)
    robot_root = output / "inputs/w1_v030"
    if not robot_root.exists():
        shutil.copytree(ROBOT, robot_root)
    (output / "inputs").mkdir(exist_ok=True)
    shutil.copy2(BAG_OBJ, output / "inputs" / BAG_OBJ.name)
    shutil.copy2(__file__, output / "source/m05.py")
    urdf = robot_root / "robot.urdf"
    sys.path.insert(0, str(source))
    utils = importlib.import_module("dexuni_w1_two_way_utils")

    import dexsim
    import imageio.v2 as imageio
    import newton
    import warp as wp
    from PIL import Image, ImageDraw, ImageFont
    from dexsim.engine.newton_physics import (
        ContactMaterialSchedule,
        DexUniParticleContactMaterialSchedule,
        DexUniSolverCfg,
        KinematicJointTrajectoryControl,
        NewtonCfg,
    )
    from dexsim.engine.newton_physics.backend_registry import get_newton_backend

    local, triangles, rest_volume = chip_bag_mesh(
        output / "inputs" / BAG_OBJ.name, args.bag_scale
    )
    # Example placement: lowest node one particle radius above the worktop.
    bag_vertices = local + np.array(
        (*BAG_START, TABLE_Z + BAG_PARTICLE_RADIUS - local[:, 2].min())
    )
    group = top_centre_group(bag_vertices)
    with wp.ScopedDevice(args.device):
        baked = bake(SideBaker(urdf, utils, args.device), utils, args)
    np.savez_compressed(
        output / "inputs/joint_commands.npz",
        joint_names=np.asarray(baked["names"]),
        targets=baked["targets"],
        time=np.arange(STEPS + 1) / FPS,
        tcp_target=baked["tcp_target"],
        tcp_baked=baked["tcp_baked"],
    )
    np.savez_compressed(
        output / "inputs/bag_mesh.npz",
        vertices=bag_vertices,
        triangles=triangles,
        top_centre_group=group,
        rest_volume=rest_volume,
    )
    gas = {
        "reference_absolute_pressure": BAG_AMBIENT_PRESSURE + BAG_GAUGE_PRESSURE,
        "ambient_pressure": BAG_AMBIENT_PRESSURE,
        "bulk_damping": BAG_BULK_DAMPING,
        "max_absolute_pressure": BAG_MAX_ABSOLUTE_PRESSURE,
    }

    width, height = 1280, 720
    eye = (1.75, -1.30, 1.62)  # M02 framing
    target = (0.40, 0.0, 1.02)
    hfov = 60.0
    config = dexsim.WorldConfig()
    config.use_default_physics = False
    config.open_windows = False
    config.renderer = dexsim.types.Renderer.OFFLINERT
    config.newton_cfg = NewtonCfg(
        device=args.device,
        dt=1 / FPS,
        num_substeps=SUBSTEPS,
        solver_cfg=DexUniSolverCfg(
            mujoco_articulation_paths=("w1",),
            joint_mode="kinematic",
            contact_mode="full",
            coupling="one_way",
            full_surface_contact_targets={
                "w1": FINGER_LINKS,
                "rigid_block": None,
                "table": None,
            },
            vbd_options={
                "pneumatic_enable_incremental_volume": True,
                "iterations": ITERATIONS,
                "friction_epsilon": FRICTION_EPSILON,
                "rigid_body_contact_buffer_size": RIGID_BODY_CONTACT_BUFFER_SIZE,
                "rigid_body_particle_contact_buffer_size": RIGID_BODY_PARTICLE_CONTACT_BUFFER_SIZE,
                "particle_enable_self_contact": False,
                "rigid_contact_hard": False,
                "rigid_contact_history": False,
                "rigid_contact_stick_freeze_translation_eps": 0.0,
                "rigid_contact_stick_freeze_angular_eps": 0.0,
            },
            collision_options={
                "broad_phase": "nxn",
                "soft_contact_margin": SOFT_CONTACT_MARGIN,
                "include_static_kinematic_pairs": False,
                "enable_rigid_soft_full_surface_contact": True,
                "contact_matching": "disabled",
            },
        ),
    )
    config.newton_cfg.use_cuda_graph = not args.no_cuda_graph
    preview = output / "preview"
    preview.mkdir(exist_ok=True)
    manifest = {
        "task": "M05",
        "mode": "native_scene_preview_only",
        "status": "recording",
        "user_scene_approved": False,
        "performance_measurements_started": False,
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "executable": sys.executable,
        "dexsim_version": dexsim.__version__,
        "newton_version": newton.__version__,
        "warp_version": wp.__version__,
        "arguments": {k: str(v) for k, v in vars(args).items()},
        "coupling": "one_way",
        "joint_mode": "kinematic",
        "user_choices": [
            "temporarily kinematic control as requested",
            "same bag asset as specified demo; elastic materials and sealed isothermal gas",
            "robot stays W1 V030 + Pika; bag gripped lying flat, no stand",
            "slower rigid motion, smaller 30 mm / 4 kg block, small contact offsets, no plasticity",
            "one main camera without an inset; more sealed gas and less damping",
            "faster block grasp and transfer; tabletop included in full-surface contact",
        ],
        "deviations_from_example": {
            "bag_scale": f"{args.bag_scale} (example 0.36: the 96 mm wide bag left no clearance in the 98.6 mm jaw)",
            "bag_rest_bulge_source_units": f"{BAG_REST_BULGE} (example 0.02)",
            "bag_gauge_pressure_pa": f"{BAG_GAUGE_PRESSURE} (example 23675)",
            "bag_tri_ke_ka": f"{BAG_TRI_KE} (example 1e5)",
            "contact_margins_m": f"{CONTACT_MARGIN} (example 0), gap={CONTACT_GAP}, particle radius={BAG_PARTICLE_RADIUS}",
            "plasticity": "disabled; elastic bending stiffness 5, no changing rest angles",
            "gas_law": "ISOTHERMAL instead of TARGET_VOLUME; constant sealed gas amount",
            "friction_epsilon": "1e-4 (example default 1e-2)",
            "solver": "8 substeps x 24 iterations (example 5 x 12)",
        },
        "robot": "W1 V030 parallel grippers; baked IK trajectory replayed kinematically; preset finger openings",
        "scene": {
            "supports": supports(),
            "robot_base_position": [0, 0, 0],
            "table_z": TABLE_Z,
            "bag_obj": BAG_OBJ.name,
            "bag_extent_m": bag_vertices.max(0) - bag_vertices.min(0),
            "bag_rest_volume_m3": rest_volume,
            "bag_start_xy": BAG_START,
            "bag_scale": args.bag_scale,
            "bag_rest_bulge_source_units": BAG_REST_BULGE,
            "bag_target_xy": BAG_TARGET,
            "bag_membrane": {
                "surface_density": BAG_DENSITY,
                "tri_ke": BAG_TRI_KE,
                "tri_ka": BAG_TRI_KA,
                "tri_kd": BAG_TRI_KD,
                "edge_ke": BAG_EDGE_KE,
                "edge_kd": BAG_EDGE_KD,
                "particle_radius": BAG_PARTICLE_RADIUS,
            },
            "bag_gas": {"mode": "ISOTHERMAL", **gas},
            "bag_plasticity": None,
            "duration_s": DURATION,
            "contact": {
                "particle_ke": CONTACT_KE,
                "particle_kd": CONTACT_KD,
                "particle_mu": CONTACT_MU,
                "hand_ke_right_fingers": HAND_CONTACT_KE,
                "hand_ke_left_fingers": BLOCK_CONTACT_KE,
                "robot_ke_other_shapes": ROBOT_KE,
                "hand_mu": CONTACT_MU,
                "release_friction": RELEASE_FRICTION,
                "release_time_s": RELEASE_TIME,
                "table_mu": TABLE_FRICTION,
                "margin": CONTACT_MARGIN,
                "soft_contact_margin": SOFT_CONTACT_MARGIN,
                "gap": CONTACT_GAP,
                "friction_epsilon": FRICTION_EPSILON,
                "full_surface_contact_links": FINGER_LINKS,
                "full_surface_contact_objects": ["rigid_block", "table"],
                "robot_force_sdf": True,
            },
            "solver": {"substeps": SUBSTEPS, "iterations": ITERATIONS},
            "rigid_contact_strategy": {
                "hard_contact": False,
                "history": False,
                "matching": "disabled",
                "body_deadzone_snap": False,
            },
            "block_start_xy": BLOCK_START,
            "block_size_m": 2 * BLOCK_HALF,
            "block_mass_kg": BLOCK_MASS,
            "gripper_pitch_deg": PITCH_DEG,
            "block_gripper_pitch_deg": BLOCK_PITCH_DEG,
            "block_place_pitch_deg": BLOCK_PLACE_PITCH_DEG,
            "gripper_yaw_deg": YAW_DEG,
            "bag_open": BAG_OPEN,
            "bag_close": args.bag_close,
            "block_mu": CONTACT_MU,
            "block_ke": BLOCK_CONTACT_KE,
            "block_open": BLOCK_OPEN,
            "block_close": args.block_close,
            "lift": LIFT,
            "block_release_height_planning_m": args.placed_top,
            "block_release_clearance": BLOCK_RELEASE_CLEARANCE,
            "block_release_dx": args.release_dx,
            "tip_clearance": TIP_CLEARANCE,
            "block_tip_clearance": BLOCK_TIP_CLEARANCE,
            "place_raise": PLACE_RAISE,
            "top_centre_group": group,
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
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in REFERENCE_NAMES
        },
        "bag_obj_sha256": hashlib.sha256(BAG_OBJ.read_bytes()).hexdigest(),
        "robot_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
    }
    _write_json(output / "manifest.json", manifest)
    if args.bake_only:
        manifest["status"] = "baked_inputs_only"
        _write_json(output / "manifest.json", manifest)
        print(json.dumps(baked["checks"], indent=2), flush=True)
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
        builder = dexsim.spawn.SceneBuilder(world)
        builder.add_urdf(
            "w1",
            str(urdf),
            per_env=False,
            newton_collision=dexsim.spawn.NewtonCollisionDesc(
                ke=ROBOT_KE,
                kd=CONTACT_KD,
                mu=CONTACT_MU,
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

        def box(
            name, center, size, color, *, density=None, mu=TABLE_FRICTION, ke=CONTACT_KE
        ):
            geometry = dexsim.spawn.GeometryDesc.cube(tuple(size))
            collision = dexsim.spawn.CollisionDesc.from_geometry(geometry)
            collision.newton = dexsim.spawn.NewtonCollisionDesc(
                ke=ke, kd=CONTACT_KD, mu=mu, margin=CONTACT_MARGIN, gap=CONTACT_GAP
            )
            builder.materials[name] = dexsim.spawn.MaterialDesc(
                name=name, base_color=(*color, 1.0)
            )
            obj = builder.add_cube(
                name,
                tuple(size),
                pose=utils.pose_at(center),
                physics=(
                    dexsim.spawn.RigidBodyPhysicsDesc.static()
                    if density is None
                    else dexsim.spawn.RigidBodyPhysicsDesc.dynamic(density=density)
                ),
                collisions=[collision],
                per_env=False,
            )
            obj.renders[0].material_ref = name

        for name, center, size in supports():
            box(name, center, size, (0.5, 0.5, 0.5))
        box(
            "rigid_block",
            (*BLOCK_START, TABLE_Z + BLOCK_HALF + BLOCK_SPAWN_GAP),
            (2 * BLOCK_HALF,) * 3,
            (0.18, 0.55, 0.78),
            density=BLOCK_MASS / (2 * BLOCK_HALF) ** 3,
            mu=CONTACT_MU,
            ke=BLOCK_CONTACT_KE,
        )
        builder.materials["bag"] = dexsim.spawn.MaterialDesc(
            name="bag",
            base_color=(0.86, 0.68, 0.34, 1.0),
            roughness=0.62,
            double_sided=True,
        )
        builder.add_clothbody(
            "bag",
            dexsim.spawn.RenderDesc.from_geometry(
                dexsim.spawn.GeometryDesc.mesh(
                    vertices=bag_vertices, triangles=triangles
                ),
                material_ref="bag",
            ),
            particle_radius=BAG_PARTICLE_RADIUS,
            physics=dexsim.spawn.ClothPhysicsDesc(
                surface_density=BAG_DENSITY,
                tri_ke=BAG_TRI_KE,
                tri_ka=BAG_TRI_KA,
                tri_kd=BAG_TRI_KD,
                edge_ke=BAG_EDGE_KE,
                edge_kd=BAG_EDGE_KD,
                contact_material=dexsim.spawn.NewtonParticleContactMaterialDesc(
                    stiffness=CONTACT_KE,
                    damping=CONTACT_KD,
                    dynamic_friction=CONTACT_MU,
                ),
                pneumatic=dexsim.spawn.PneumaticPhysicsDesc(
                    mode=dexsim.spawn.PneumaticMode.ISOTHERMAL, **gas
                ),
            ),
            per_env=False,
            validate_mesh=True,
        )
        builder.add_runtime_control(
            ContactMaterialSchedule(
                "w1",
                {"stiffness": ((0.0, BLOCK_CONTACT_KE),)},
                link_names=("left_finger1", "left_finger2"),
            )
        )
        # As in the example: finger and bag friction drop when the bag is released.
        release_track = ((0.0, CONTACT_MU), (RELEASE_TIME, RELEASE_FRICTION))
        builder.add_runtime_control(
            ContactMaterialSchedule(
                "w1",
                {
                    "dynamic_friction": release_track,
                    "stiffness": ((0.0, HAND_CONTACT_KE),),
                },
                link_names=RIGHT_FINGER_LINKS,
            )
        )
        builder.add_runtime_control(
            DexUniParticleContactMaterialSchedule(
                {
                    "stiffness": ((0.0, CONTACT_KE), (RELEASE_TIME, CONTACT_KE)),
                    "damping": ((0.0, CONTACT_KD), (RELEASE_TIME, CONTACT_KD)),
                    "dynamic_friction": release_track,
                },
                target="bag",
            )
        )
        builder.add_light(
            dexsim.spawn.LightDesc(
                name="key_light",
                light_type=dexsim.render.LightType.POINT,
                pose=utils.pose_at((1.5, -1.0, 3.0)),
                intensity=50.0,
                shadow=False,
            )
        )
        builder.add_camera(
            dexsim.spawn.CameraDesc(
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
        bag_set = scene.get_particle_set("bag")
        inverse_masses = bag_set.get_inverse_masses().numpy()
        if np.any(inverse_masses <= 0) or not np.isfinite(inverse_masses).all():
            raise RuntimeError("All bag nodes must remain dynamic and have finite mass")
        manifest["initial_model"] = {
            "bag_nodes": int(len(inverse_masses)),
            "bag_mass_kg": float((1 / inverse_masses).sum()),
            "pinned_nodes": 0,
        }
        block_obj = scene.get_object("rigid_block")
        camera = scene.get_camera("preview")
        vfov = 2 * np.arctan(np.tan(np.deg2rad(hfov) / 2) * height / width)
        camera.set_perspective(float(np.rad2deg(vfov)), width / height, 0.01, 10.0)
        camera.look_at(eye, target, (0, 0, 1))
        camera.open_camera()
        backend = get_newton_backend(world)
        vbd = backend.solver.vbd_solver
        manifest["effective_rigid_contact"] = {
            "hard": bool(vbd.rigid_contact_hard),
            "history": bool(vbd.rigid_contact_history),
            "alpha": float(vbd.rigid_contact_alpha),
        }
        pipeline = backend.solver.backend.pipeline
        face_pairs = pipeline.soft_face_rigid_pairs.numpy()
        edge_pairs = pipeline.soft_edge_rigid_pairs.numpy()
        surface_shape_ids = set(face_pairs[:, 1].tolist()) | set(
            edge_pairs[:, 1].tolist()
        )
        block_shape_ids = {
            i
            for i, label in enumerate(pipeline.model.shape_label)
            if "rigid_block" in label
        }
        table_shape_ids = {
            i
            for i, label in enumerate(pipeline.model.shape_label)
            if label.startswith("table_collision_")
        }
        if not block_shape_ids or not block_shape_ids <= surface_shape_ids:
            raise RuntimeError(
                "Dynamic block is missing full-surface contact candidates"
            )
        if not table_shape_ids or not table_shape_ids <= surface_shape_ids:
            raise RuntimeError("Table is missing full-surface contact candidates")
        manifest["effective_full_surface_contact"] = {
            "enabled": bool(pipeline.enable_rigid_soft_full_surface_contact),
            "shape_labels": [
                pipeline.model.shape_label[i] for i in sorted(surface_shape_ids)
            ],
            "face_candidate_pairs": len(face_pairs),
            "edge_candidate_pairs": len(edge_pairs),
            "dynamic_block_verified": True,
            "table_verified": True,
        }
        if not np.allclose(
            bag_set.get_particle_positions().numpy(), bag_vertices, atol=1e-5
        ):
            raise RuntimeError("Bag node order differs from the authored mesh")
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
        stills = {0} | {round(t * FPS) for t in TIMING.values()}
        tcp_local = np.array((*utils.TCP, 1.0))
        last = STEPS if args.until is None else min(STEPS, round(args.until * FPS))
        for frame in range(last + 1):
            if frame:
                world.update(1 / FPS)
            time = float(scene.get_simulation_time())
            if abs(time - frame / FPS) > 1e-7:
                raise RuntimeError(f"Actual clock mismatch at frame {frame}: {time}")
            bag = bag_set.get_particle_positions().numpy()
            block_pose = np.asarray(block_obj.get_world_pose())
            tcp = np.array(
                [
                    (robot.get_link_pose(f"{s}_gripper_base") @ tcp_local)[:3]
                    for s in ("left", "right")
                ]
            )
            if not (np.isfinite(bag).all() and np.isfinite(block_pose).all()):
                raise AssertionError(f"Nonfinite state at frame {frame}")
            row = load_metrics(bag, block_pose, group)
            qpos = robot.get_current_qpos()[joint_ids]
            if not np.allclose(qpos, baked["targets"][frame], atol=2e-5, rtol=0):
                raise RuntimeError(f"Named joint replay mismatch at frame {frame}")
            a, b, c = (bag[triangles[:, i]] for i in range(3))
            volume = abs(
                float(
                    np.einsum("ij,ij->i", a - a[0], np.cross(b - a[0], c - a[0])).sum()
                    / 6
                )
            )
            row.update(
                frame=frame,
                time=time,
                phase=_phase(time),
                tcp=tcp,
                tcp_error_m=np.linalg.norm(tcp - baked["tcp_baked"][frame], axis=1),
                volume_ratio=volume / rest_volume,
                block_tilt_deg=float(
                    np.rad2deg(
                        np.arccos(np.clip(np.abs(block_pose[2, :3]).max(), 0, 1))
                    )
                ),
                block_initial_axis_tilt_deg=float(
                    np.rad2deg(np.arccos(np.clip(block_pose[2, 2], -1, 1)))
                ),
            )
            metrics.append(row)
            for key, value in (
                ("time", time),
                ("qpos", qpos),
                ("target_q", baked["targets"][frame]),
                ("bag", bag),
                ("block_pose", block_pose),
                ("tcp", tcp),
                (
                    "left_hand_poses",
                    np.stack([robot.get_link_pose(name) for name in LEFT_HAND_LINKS]),
                ),
            ):
                states.setdefault(key, []).append(np.array(value, copy=True))
            if (writer is not None and frame % 2 == 0) or frame in stills:
                backend.sync_to_dexsim(world)
                world._flush_render_updates()
                camera.render()
                image = Image.fromarray(np.asarray(camera.get_rgb_map())[..., :3])
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, width, 42), fill="black")
                draw.text(
                    (14, 10),
                    f"DexSim | M05 inflatable bag + rigid load (one-way) | t={time:.2f}s | {row['phase']} | "
                    f"bag h {row['bag_height_m'] * 1e3:.1f} mm | block bottom {(row['block_bottom_z_m'] - TABLE_Z) * 1e3:.1f} mm",
                    font=font,
                    fill="white",
                )
                if writer is not None and frame % 2 == 0:
                    writer.append_data(np.asarray(image))
                if frame in stills:
                    image.save(preview / f"still_{time:05.2f}s.jpg")
            if frame % 120 == 0:
                print(
                    f"frame={frame}/{last} t={time:.2f} bag={bag.mean(0).round(3)} block={block_pose[:3,3].round(3)}",
                    flush=True,
                )
        manifest.update(
            status="completed", frames=last, effective=scene.get_solver_diagnostics()
        )
        manifest["actual_simulation_time_s"] = time
        manifest["named_joint_replay_verified"] = True
        if (
            not args.no_cuda_graph
            and manifest["effective"]["cuda_graph_status"] != "captured"
        ):
            raise RuntimeError("Requested CUDA Graph was not captured")
        if manifest["effective"]["features"]["pneumatic_cavity_count"] != 1:
            raise RuntimeError("Expected one active closed pneumatic cavity")
        manifest["input_sha256"] = {
            name: hashlib.sha256((output / "inputs" / name).read_bytes()).hexdigest()
            for name in (
                "joint_commands.npz",
                "bag_mesh.npz",
                BAG_OBJ.name,
                "w1_v030/robot.urdf",
            )
        }
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
