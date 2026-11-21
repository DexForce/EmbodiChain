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
"""Record the native W1 one-way single cloth fold (M02) for user review.

Run: /home/oem/code/engine/dexsim/.env-3.12/bin/python scripts/benchmark/tech_reports/physics/platforms/dexuni/dexsim/m02.py
Pilot: add ``--pilot --out <scratch dir>`` to save still frames instead of the video.

The W1 picks the sheet up along its centre line, lifts it so both halves hang
down, then lays it forward onto the worktop as two layers (a single fold).
One-way coupling with kinematic joints: an offline IK pass bakes the whole
joint trajectory (inputs/joint_commands.npz), which DexSim replays through
KinematicJointTrajectoryControl. The cloth does not push back on the robot;
the fingers close to a preset opening.
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

from scripts.benchmark.tech_reports.physics.experiments.dexuni.m02_common import (
    CLOTH_CENTER,
    CLOTH_SIZE,
    OUTPUT,
    PHASES,
    TABLE_Z,
    cloth_mesh,
    fold_metrics,
    node_groups,
    phase,
)

REFERENCE = Path("/home/oem/code/engine/dexsim/examples/python/physics/multiphys")
ROBOT = Path("/home/oem/.cache/dexsim/resources/mjvbd_v2/w1_bag_packing/w1_v030")
REFERENCE_NAMES = (
    "dexuni_w1_two_way_utils.py",
    "demo_dexuni_w1_bag_packing.py",
    "dexuni_asset_utils.py",
)
FPS = 60
STEPS = 990  # 16.5 s
# Downward gripper with fingers closing along world X (pads lie along Y, the
# fold line), turned per side. The gripper stays vertical the whole time.
DOWNWARD_X_FINGERS = np.array((-np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0))
SIDE_YAW_DEG = (180.0, 0.0)  # left, right
GRIP_Y = 0.17  # grasp points on the centre line, 70 mm inside each side edge
TIP_CLEARANCE = 0.003  # lowest finger point above the worktop at grasp
LIFT_HEIGHT = 0.30  # TCP above the worktop while the two halves hang
# Laying: start with the hanging free edges just above the worktop, then move
# forward and down together so the sheet lies down as two layers, ending with
# the fold line (held ridge) at LAY_END_X.
LAY_START_HEIGHT = 0.25  # TCP above the worktop at the start of laying
LAY_END_X = 0.48
LAY_TRAVEL = 0.16
PLACE_CLEARANCE = 0.025  # lowest finger point above the worktop at release
FINGER_OPEN = 0.045
RELEASE_OPEN = 0.048
LIFT_OFF = 0.08

__all__ = ["main"]


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, default=lambda item: np.asarray(item).tolist())
        + "\n"
    )


def side_rotation(side: int, pitch_deg: float):
    """Grip orientation of one side (0 left, 1 right) as a SciPy rotation."""
    from scipy.spatial.transform import Rotation

    return (
        Rotation.from_euler("y", -pitch_deg, degrees=True)
        * Rotation.from_euler("z", SIDE_YAW_DEG[side], degrees=True)
        * Rotation.from_quat(DOWNWARD_X_FINGERS)
    )


class TrajectoryBaker:
    """Offline IK for both W1 arms; produces named joint rows per frame."""

    def __init__(self, urdf: Path, utils, device: str) -> None:
        import warp as wp
        from dexsim.kit.ik import (
            JointLimitTarget,
            KinematicModel,
            MultiTargetIK,
            PositionTarget,
            RotationTarget,
        )

        self.wp, self.utils, self.urdf = wp, utils, urdf
        model = KinematicModel.from_urdf(
            urdf, floating=False, hide_visuals=True, collapse_fixed_joints=False
        )
        self.model = model
        starts = model.joint_q_start.numpy()
        dofs = model.joint_qd_start.numpy()
        self.coords = {
            label.rsplit("/", 1)[-1]: int(starts[j])
            for j, label in enumerate(model.joint_label)
            if starts[j + 1] > starts[j]
        }
        self.names = list(self.coords)
        initial = model.joint_q.numpy()
        for side, sign in (("LEFT", -1), ("RIGHT", 1)):
            initial[self.coords[f"{side}_J2"]] = sign * 0.9
            initial[self.coords[f"{side}_J4"]] = sign * 1.0
        self.q = wp.array(initial.reshape(1, -1), device=device)
        body = lambda name: next(  # noqa: E731
            i for i, label in enumerate(model.body_label) if label.endswith("/" + name)
        )
        self.ee = [body(f"{s}_gripper_base") for s in ("left", "right")]
        self.fingers = {
            f"{s}_finger{k}": body(f"{s}_finger{k}")
            for s in ("left", "right")
            for k in (1, 2)
        }
        self.cameras = [body(f"{s}_camera") for s in ("left", "right")]
        self.lower = model.joint_limit_lower.numpy()
        self.upper = model.joint_limit_upper.numpy()
        self.arm = [
            self.coords[f"{s}_J{j}"] for s in ("LEFT", "RIGHT") for j in range(1, 8)
        ]
        mask = np.zeros(model.joint_dof_count, dtype=bool)
        arm_names = {f"{s}_J{j}" for s in ("LEFT", "RIGHT") for j in range(1, 8)}
        for j, label in enumerate(model.joint_label):
            if label.rsplit("/", 1)[-1] in arm_names:
                mask[int(dofs[j]) : int(dofs[j + 1])] = True
        self.positions = [
            PositionTarget(
                b,
                wp.vec3(*utils.TCP),
                wp.array([wp.vec3(0, 0, 1)], dtype=wp.vec3, device=device),
            )
            for b in self.ee
        ]
        self.rotations = [
            RotationTarget(
                b,
                wp.quat_identity(),
                wp.array(
                    [wp.vec4(*side_rotation(i, 0.0).as_quat())],
                    dtype=wp.vec4,
                    device=device,
                ),
            )
            for i, b in enumerate(self.ee)
        ]
        self.ik = MultiTargetIK(
            model,
            1,
            self.positions
            + self.rotations
            + [
                JointLimitTarget(
                    model.joint_limit_lower, model.joint_limit_upper, weight=10.0
                )
            ],
            lambda_initial=0.03,
            joint_dof_mask=mask,
        )
        self.state = model.state()
        self.finger_meshes = self._finger_meshes()

    def _finger_meshes(self) -> dict[str, np.ndarray]:
        import trimesh

        root = self.urdf.parent
        return {
            name: np.asarray(
                trimesh.load(
                    root
                    / f"collision/{name.split('_')[0]}_hand/link{7 if name.endswith('1') else 8}_acd.obj",
                    force="mesh",
                ).vertices
            )
            for name in self.fingers
        }

    def solve(
        self, targets: np.ndarray, finger: float, pitch: float, iterations: int
    ) -> np.ndarray:
        wp = self.wp
        for objective, point in zip(self.positions, targets, strict=True):
            objective.set_target_position(0, wp.vec3(*point))
        for side, objective in enumerate(self.rotations):
            objective.set_target_rotation(
                0, wp.vec4(*side_rotation(side, pitch).as_quat().astype(np.float32))
            )
        self.ik.step(self.q, self.q, iterations=iterations)
        q = np.clip(self.q.numpy()[0], self.lower, self.upper)
        for side in ("LEFT", "RIGHT"):
            for k in (1, 2):
                q[self.coords[f"{side}_FINGER{k}_JOINT"]] = finger
        self.q.assign(q.reshape(1, -1))
        return q

    def body_poses(self) -> np.ndarray:
        self.model.forward_kinematics(self.q.flatten(), self.state)
        return self.state.body_q.numpy()

    def tcp(self) -> np.ndarray:
        from scipy.spatial.transform import Rotation

        poses = self.body_poses()
        return np.array(
            [
                Rotation.from_quat(poses[b][3:]).apply(self.utils.TCP) + poses[b][:3]
                for b in self.ee
            ]
        )

    def camera_gap(self) -> float:
        poses = self.body_poses()
        return float(
            np.linalg.norm(poses[self.cameras[0]][:3] - poses[self.cameras[1]][:3])
        )

    def lowest_finger_z(self) -> float:
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
        )


def grasp_points(z: float) -> np.ndarray:
    """TCP grasp points on the fold line (cloth centre line along Y)."""
    x = CLOTH_CENTER[0]
    return np.array(((x, GRIP_Y, z), (x, -GRIP_Y, z)))


def _quintic(u: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    return u**3 * (10 + u * (-15 + 6 * u))


class FoldPlan:
    """TCP targets, finger opening and gripper pitch (always 0) over time."""

    def __init__(self, grasp: np.ndarray, place_z: float, close: float) -> None:
        z0 = grasp[0, 2]
        at = lambda x, z: np.array(((x, GRIP_Y, z), (x, -GRIP_Y, z)))  # noqa: E731
        lifted = at(CLOTH_CENTER[0], TABLE_Z + LIFT_HEIGHT)
        lay_start = at(LAY_END_X - LAY_TRAVEL, TABLE_Z + LAY_START_HEIGHT)
        place = at(LAY_END_X, place_z)
        o, c, r = FINGER_OPEN, close, RELEASE_OPEN
        self.frames = [
            (0.0, grasp + (0, 0, 0.12), o, 0.0),
            (0.5, grasp + (0, 0, 0.12), o, 0.0),
            (2.0, grasp + (0, 0, 0.06), o, 0.0),
            (3.0, grasp, o, 0.0),
            (4.0, grasp, c, 0.0),
            (4.4, grasp, c, 0.0),
            (7.0, lifted, c, 0.0),
            (8.0, lifted, c, 0.0),
            (9.5, lay_start, c, 0.0),
            (12.5, place, c, 0.0),
            (13.5, place, r, 0.0),
            (14.5, place + (0, 0, LIFT_OFF), r, 0.0),
            (16.5, place + (0, 0, LIFT_OFF), r, 0.0),
        ]
        self.place = place
        del z0

    def __call__(self, time: float, utils) -> tuple[np.ndarray, float, float]:
        targets, finger, pitch = utils.smoothstep(self.frames, time)
        return np.asarray(targets), float(finger), float(pitch)


def bake(baker: TrajectoryBaker, utils, close: float) -> dict:
    """Pick grasp/place heights from finger geometry, then bake every frame."""
    z = TABLE_Z + 0.03
    for _ in range(3):
        baker.solve(grasp_points(z), FINGER_OPEN, 0.0, 300)
        z += TABLE_Z + TIP_CLEARANCE - baker.lowest_finger_z()
    grasp = grasp_points(z)
    place_z = z
    for _ in range(3):
        place = grasp_points(place_z)
        place[:, 0] = LAY_END_X
        baker.solve(place, close, 0.0, 300)
        place_z += TABLE_Z + PLACE_CLEARANCE - baker.lowest_finger_z()
    plan = FoldPlan(grasp, place_z, close)
    baker.solve(plan(0.0, utils)[0], FINGER_OPEN, 0.0, 300)
    rows, tcp_targets, tcp_actual, camera_gap = [], [], [], []
    for frame in range(STEPS + 1):
        targets, finger, pitch = plan(frame / FPS, utils)
        rows.append(
            baker.solve(targets, float(finger), float(pitch), 150 if frame == 0 else 30)
        )
        tcp_targets.append(targets)
        tcp_actual.append(baker.tcp())
        camera_gap.append(baker.camera_gap())
    q = np.asarray(rows)
    arm = baker.arm
    names = baker.names
    return {
        "names": names,
        "targets": np.asarray(
            [[row[baker.coords[n]] for n in names] for row in q], dtype=np.float32
        ),
        "tcp_target": np.asarray(tcp_targets),
        "tcp_baked": np.asarray(tcp_actual),
        "grasp_tcp": grasp,
        "checks": {
            "max_joint_step_rad": float(np.abs(np.diff(q[:, arm], axis=0)).max()),
            "min_limit_margin_rad": float(
                np.min(
                    np.minimum(
                        q[:, arm] - baker.lower[arm], baker.upper[arm] - q[:, arm]
                    )
                )
            ),
            "max_tcp_error_m": float(
                np.linalg.norm(
                    np.asarray(tcp_actual) - np.asarray(tcp_targets), axis=2
                ).max()
            ),
            "min_hand_camera_gap_m": float(min(camera_gap)),
            "grasp_tcp_z_above_table_m": float(z - TABLE_Z),
            "place_tcp_z_above_table_m": float(place_z - TABLE_Z),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--substeps", type=int, default=10)
    # 0 leaves the VBD iteration count to the surface-fast preset (8 plus its
    # Chebyshev/multilevel acceleration). With plain VBD and 10 iterations the
    # hanging halves fluttered (frame-to-frame 2nd difference p95 3 mm while
    # the grippers were still); 20 iterations cut it to 1 mm, the preset to
    # 0.6 mm. Self-contact is required: without it the halves pass through
    # each other.
    parser.add_argument("--iterations", type=int, default=0)
    # Preset jaw opening per finger (gap ~2q - 1.4 mm).
    parser.add_argument("--finger-close", type=float, default=0.002)
    parser.add_argument("--robot-mu", type=float, default=1.0)
    parser.add_argument("--yaw", type=float, nargs=2, default=list(SIDE_YAW_DEG))
    parser.add_argument("--no-cuda-graph", action="store_true")
    parser.add_argument("--no-self-contact", action="store_true")
    parser.add_argument("--lift-height", type=float, default=LIFT_HEIGHT)
    parser.add_argument("--lay-start-height", type=float, default=LAY_START_HEIGHT)
    parser.add_argument(
        "--vbd-preset", choices=("none", "surface-fast"), default="surface-fast"
    )
    # surface-fast only engages with contact_mode auto/soft (no dynamic VBD bodies).
    parser.add_argument(
        "--contact-mode", choices=("full", "soft", "auto"), default="soft"
    )
    return parser


def main() -> None:
    """Bake the trajectory, build the scene and record one preview."""
    args = _parser().parse_args()
    global SIDE_YAW_DEG, LIFT_HEIGHT, LAY_START_HEIGHT
    SIDE_YAW_DEG = tuple(args.yaw)
    LIFT_HEIGHT = args.lift_height
    LAY_START_HEIGHT = args.lay_start_height
    output = args.out
    source = output / "source/reference"
    source.mkdir(parents=True, exist_ok=True)
    for name in REFERENCE_NAMES:
        shutil.copy2(REFERENCE / name, source / name)
    robot_root = output / "inputs/w1_v030"
    if not robot_root.exists():
        shutil.copytree(ROBOT, robot_root)
    shutil.copy2(__file__, output / "source/m02.py")
    urdf = robot_root / "robot.urdf"
    sys.path.insert(0, str(source))
    utils = importlib.import_module("dexuni_w1_two_way_utils")

    import dexsim
    import imageio.v2 as imageio
    import newton
    import warp as wp
    from PIL import Image, ImageDraw, ImageFont
    from dexsim.engine.newton_physics import (
        DexUniSolverCfg,
        KinematicJointTrajectoryControl,
        NewtonCfg,
    )
    from dexsim.engine.newton_physics.backend_registry import get_newton_backend

    with wp.ScopedDevice(args.device):
        baked = bake(
            TrajectoryBaker(urdf, utils, args.device), utils, args.finger_close
        )
    (output / "inputs").mkdir(exist_ok=True)
    np.savez_compressed(
        output / "inputs/joint_commands.npz",
        joint_names=np.asarray(baked["names"]),
        targets=baked["targets"],
        time=np.arange(STEPS + 1) / FPS,
        tcp_target=baked["tcp_target"],
        tcp_baked=baked["tcp_baked"],
    )
    mesh = cloth_mesh()
    groups = node_groups(mesh["vertices"])
    np.savez_compressed(
        output / "inputs/cloth_mesh.npz",
        **mesh,
        **{f"group_{k}": v for k, v in groups.items()},
    )

    width, height = 1280, 720
    eye = (1.75, -1.30, 1.62)
    target = (0.40, 0.0, 1.02)
    hfov = 60.0
    self_contact = {
        "particle_enable_self_contact": not args.no_self_contact,
        "particle_self_contact_radius": 0.002,
        "particle_self_contact_margin": 0.002,
        "particle_rest_shape_contact_exclusion_radius": 0.005,
    }
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
            contact_mode=args.contact_mode,
            coupling="one_way",
            vbd_preset=None if args.vbd_preset == "none" else args.vbd_preset,
            vbd_options={
                # iterations <= 0: leave the count to the preset/solver default.
                **({"iterations": args.iterations} if args.iterations > 0 else {}),
                "friction_epsilon": 1e-4,
                "rigid_contact_hard": False,
                "rigid_contact_history": False,
                "rigid_body_contact_buffer_size": 4096,
                **self_contact,
            },
            # The kinematic soft-only backend accepts only the contact margin.
            collision_options=(
                {
                    "broad_phase": "nxn",
                    "include_static_kinematic_pairs": False,
                    "soft_contact_margin": 0.003,
                }
                if args.contact_mode == "full"
                else {"soft_contact_margin": 0.003}
            ),
        ),
    )
    config.newton_cfg.use_cuda_graph = not args.no_cuda_graph
    preview = output / "preview"
    preview.mkdir(exist_ok=True)
    manifest = {
        "task": "M02",
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
        "robot": "W1 V030 parallel grippers; baked IK trajectory replayed kinematically",
        "scene": {
            "table_z": TABLE_Z,
            "cloth_center": CLOTH_CENTER,
            "cloth_size": CLOTH_SIZE,
            "cloth_source": "S03 node spacing (10 mm) and material; mirror-symmetric triangulation",
            "motion": "pick up along the centre line, lift so both halves hang, lay down forwards as two layers",
            "grasp_tcp": baked["grasp_tcp"],
            "grip_y": GRIP_Y,
            "tip_clearance": TIP_CLEARANCE,
            "lift_height": LIFT_HEIGHT,
            "lay_start_height": LAY_START_HEIGHT,
            "lay_end_x": LAY_END_X,
            "lay_travel": LAY_TRAVEL,
            "place_clearance": PLACE_CLEARANCE,
            "gripper_pitch_deg": 0.0,
            "side_yaw_deg": SIDE_YAW_DEG,
            "finger_open": FINGER_OPEN,
            "release_open": RELEASE_OPEN,
            "release_lift_off": LIFT_OFF,
            "robot_mu": args.robot_mu,
            "contact_mode": args.contact_mode,
            "vbd_preset": args.vbd_preset,
            "finger_close": args.finger_close,
            "phases": PHASES,
        },
        "self_contact": self_contact,
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
        "robot_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
        "group_sizes": {k: int(v.size) for k, v in groups.items()},
    }
    _write_json(output / "manifest.json", manifest)
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
                ke=2e4, kd=100.0, mu=args.robot_mu, gap=0.002
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

        def static_box(name, center, size, color, mu=0.8):
            geometry = dexsim.spawn.GeometryDesc.cube(tuple(size))
            collision = dexsim.spawn.CollisionDesc.from_geometry(geometry)
            collision.newton = dexsim.spawn.NewtonCollisionDesc(
                ke=2e4, kd=100.0, mu=mu, gap=0.002
            )
            builder.materials[name] = dexsim.spawn.MaterialDesc(
                name=name, base_color=(*color, 1.0)
            )
            obj = builder.add_cube(
                name,
                tuple(size),
                pose=utils.pose_at(center),
                physics=dexsim.spawn.RigidBodyPhysicsDesc.static(),
                collisions=[collision],
                per_env=False,
            )
            obj.renders[0].material_ref = name

        static_box(
            "table", (0.46, 0.0, TABLE_Z - 0.025), (0.6, 1.64, 0.05), (0.5, 0.5, 0.5)
        )
        for i, x in enumerate((0.21, 0.71)):
            for j, y in enumerate((-0.74, 0.74)):
                static_box(
                    f"leg_{i}_{j}",
                    (x, y, (TABLE_Z - 0.05) / 2),
                    (0.05, 0.05, TABLE_Z - 0.05),
                    (0.5, 0.5, 0.5),
                )
        builder.materials["cloth"] = dexsim.spawn.MaterialDesc(
            name="cloth",
            base_color=(0.25, 0.65, 0.9, 1.0),
            roughness=0.7,
            double_sided=True,
        )
        builder.add_cloth_object(
            dexsim.spawn.ClothDesc(
                name="cloth",
                mesh=dexsim.spawn.RenderDesc.from_geometry(
                    dexsim.spawn.GeometryDesc.mesh(
                        vertices=mesh["vertices"], triangles=mesh["triangles"]
                    ),
                    material_ref="cloth",
                ),
                physics=dexsim.spawn.ClothPhysicsDesc(
                    surface_density=0.002,
                    tri_ke=1000.0,
                    tri_ka=1000.0,
                    tri_kd=1e-5,
                    tri_drag=0.0,
                    tri_lift=0.0,
                    edge_ke=0.1,
                    edge_kd=0.05,
                    contact_material=dexsim.spawn.NewtonParticleContactMaterialDesc(
                        stiffness=1e4, damping=0.01, dynamic_friction=0.3
                    ),
                ),
                particle_radius=0.003,
                per_env=False,
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
        cloth = scene.get_particle_set("cloth")
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
        stills = {
            round(t * FPS)
            for t in (0.0, 4.4, 7.0, 8.0, 9.5, 11.0, 12.5, 13.5, 14.5, 16.5)
        }
        tcp_local = np.array((*utils.TCP, 1.0))
        for frame in range(STEPS + 1):
            if frame:
                world.update(1 / FPS)
            time = frame / FPS
            particles = cloth.get_particle_positions().numpy()
            if not np.isfinite(particles).all():
                raise AssertionError(f"Nonfinite cloth at frame {frame}")
            tcp = np.array(
                [
                    (robot.get_link_pose(f"{s}_gripper_base") @ tcp_local)[:3]
                    for s in ("left", "right")
                ]
            )
            row = fold_metrics(particles, groups)
            row.update(
                frame=frame,
                time=time,
                phase=phase(time),
                tcp=tcp,
                tcp_error_m=np.linalg.norm(tcp - baked["tcp_target"][frame], axis=1),
            )
            metrics.append(row)
            for key, value in (
                ("time", time),
                ("qpos", robot.get_current_qpos()),
                ("target_q", baked["targets"][frame]),
                ("particles", particles),
                ("tcp", tcp),
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
                    f"DexSim | M02 cloth single fold (one-way) | t={time:.2f}s | {row['phase']} | "
                    f"corner XY error L {row['left_xy_error_m'] * 1e3:.0f} / R {row['right_xy_error_m'] * 1e3:.0f} mm",
                    font=font,
                    fill="white",
                )
                if writer is not None and frame % 2 == 0:
                    writer.append_data(np.asarray(image))
                if frame in stills:
                    image.save(preview / f"still_{time:05.2f}s.jpg")
        manifest.update(
            status="completed",
            frames=STEPS,
            effective=scene.get_solver_diagnostics(),
        )
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
