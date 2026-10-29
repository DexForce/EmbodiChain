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
"""Unqualified S01C dexsim pilot adapter.

Run from the repository root with DEXUNI_ALLOW_PILOT=1:
    <platform-python> scripts/benchmark/tech_reports/physics/platforms/dexuni/dexsim/s01c.py --state s00 --images keyframes

Inputs: DEXUNI_INPUT_ROOT/s01c, or --inputs DIR (read-only snapshot).
Outputs: outputs/benchmarks/tech_reports/dexuni/development/s01c, or --out-root DIR.
Read physics/experiments/dexuni/PLAN.md and SCRIPT_CONTRACT.md before use.
Migration changes paths only; the S01 review gates are NOT implemented here.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import warp as wp

REPO_ROOT = Path(__file__).resolve().parents[7]
sys.path.insert(0, str(REPO_ROOT))
from scripts.benchmark.tech_reports.physics.experiments.dexuni.paths import (
    EXPERIMENTS,
    PLATFORMS,
    input_dir,
    output_dir,
    franka_usd,
    require_pilot,
)

SUBTASK_DIR = EXPERIMENTS / "s01c_compress"
INPUTS = input_dir("s01c")
BENCH_ROOT = PLATFORMS
if __name__ == "__main__":
    require_pilot()
PLATFORM = "dexsim"
sys.path.insert(0, str(BENCH_ROOT / "_lib"))  # benchlib copy

import dexsim  # noqa: E402
from dexsim.engine.newton_physics import (  # noqa: E402
    DexUniSolverCfg,
    NewtonCfg,
    RealtimeJointTargetControl,
)

# Common segmentation label ids (config measurement.images.labels).
LABELS = {
    "ground": 1,
    "table": 2,
    "robot_stand": 3,
    "franka": 4,
    "press_plate": 5,
    "soft_block": 6,
}


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_path(config: dict, key: str, value) -> None:
    node = config
    parts = key.split(".")
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value


def load_inputs(args) -> dict:
    global INPUTS
    if args.inputs:
        INPUTS = Path(args.inputs).resolve()
    with open(INPUTS / "s01c_config.json", encoding="utf-8") as stream:
        config = json.load(stream)
    from scripts.benchmark.tech_reports.physics.experiments.dexuni.paths import (
        frozen_robot_urdf,
    )

    config["robot"]["urdf"] = frozen_robot_urdf(INPUTS, config["robot"]["urdf"])
    for key, value in config["variants"][args.variant].items():
        set_path(config, key, value)
    for item in args.set or []:
        key, value = item.split("=", 1)
        set_path(config, key, json.loads(value))
        config.setdefault("_overrides", []).append(item)
    with open(INPUTS / "s01c_plate.json", encoding="utf-8") as stream:
        plate = json.load(stream)
    mesh = np.load(INPUTS / "s01c_soft_block_mesh.npz")
    states = np.load(INPUTS / "s01c_initial_states.npz")
    index = [str(s) for s in states["state_ids"]].index(args.state)
    ik = np.load(INPUTS / "s01c_ik_table.npz")
    return {
        "config": config,
        "plate": plate,
        "tets": mesh["tets"],
        "surface": mesh["surface_triangles"],
        "vertices": states["particle_positions"][index].astype(np.float32),
        "state": {
            "id": args.state,
            "block_center": states["block_center"][index].tolist(),
            "block_yaw_deg": float(states["block_yaw_deg"][index]),
        },
        "ik_z": np.asarray(ik["z_offset"], dtype=np.float64),
        "ik_q": np.asarray(ik["arm_q"], dtype=np.float64),
        "contact_tcp_z": float(ik["contact_tcp_z"]),
    }


def arm_target(inputs, z_offset: float) -> np.ndarray:
    """Interpolate the shared IK table (TCP z offset -> 7 arm joints)."""
    z = inputs["ik_z"]
    if not z[0] <= z_offset <= z[-1]:
        raise ValueError(f"TCP offset {z_offset:+.4f} m outside IK table")
    return np.array([np.interp(z_offset, z, col) for col in inputs["ik_q"].T])


def lame_parameters(youngs_modulus: float, poisson_ratio: float):
    e, nu = float(youngs_modulus), float(poisson_ratio)
    return e / (2.0 * (1.0 + nu)), e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))


# --------------------------------------------------------------------------
# Closed-loop compression rule (identical class in s01c_isaaclab.py)
# --------------------------------------------------------------------------


class CompressionController:
    """S01-C closed-loop press rule; kept verbatim in both platform scripts.

    Input: frame time and measured block height ``h = max(z) - min(z)``.
    Output: commanded TCP z offset from the nominal contact height, finger
    target and phase name.
    """

    PHASES = ("rest", "approach", "press", "hold", "unload", "recover", "done")

    def __init__(self, config: dict):
        traj = config["trajectory"]
        self.traj = traj
        self.dt = 1.0 / traj["frame_rate"]
        self.compression = float(config["target_compression"])
        self.reference_window = config["measurement"]["reference_window"]
        self.filter_frames = int(traj["height_filter_frames"])
        self.z = float(traj["rest_clearance"])
        self.phase = "rest"
        self.phase_start = 0.0
        self.h0 = None
        self.h_target = None
        self.target_reached = False
        self.events = {"rest": 0.0}
        self._ref = []
        self._window = []
        self._unload_from = None

    @staticmethod
    def _quintic(s: float) -> float:
        s = min(max(float(s), 0.0), 1.0)
        return s**3 * (10.0 + s * (-15.0 + 6.0 * s))

    def _enter(self, phase: str, time: float) -> None:
        self.phase, self.phase_start = phase, time
        self.events[phase] = time

    def step(self, time: float, height: float):
        """Advance to ``time`` given the height measured at ``time - dt``."""
        traj = self.traj
        self._window = (self._window + [float(height)])[-self.filter_frames :]
        height = sum(self._window) / len(self._window)
        if self.phase == "rest":
            lo, hi = self.reference_window
            if lo <= time - self.dt <= hi + 1e-9:
                self._ref.append(height)
            if time >= traj["rest_duration"] - 1e-9:
                self.h0 = sum(self._ref) / len(self._ref)
                self.h_target = self.h0 * (1.0 - self.compression)
                self._enter("approach", time)
        elif self.phase == "approach":
            s = self._quintic((time - self.phase_start) / traj["approach_duration"])
            self.z = (1 - s) * traj["rest_clearance"] + s * traj["approach_clearance"]
            if s >= 1.0:
                self._enter("press", time)
        elif self.phase == "press":
            if height <= self.h_target:
                self.target_reached = True
                self._enter("hold", time)
            elif time - self.phase_start >= traj["press_timeout"]:
                self._enter("hold", time)
            else:
                self.z -= traj["press_speed"] * self.dt
        elif self.phase == "hold":
            rate = -traj["hold_integral_gain"] * (height - self.h_target)
            limit = traj["hold_max_speed"]
            self.z += min(max(rate, -limit), limit) * self.dt
            if time - self.phase_start >= traj["hold_duration"] - 1e-9:
                self._unload_from = self.z
                self._enter("unload", time)
        elif self.phase == "unload":
            s = self._quintic((time - self.phase_start) / traj["unload_duration"])
            self.z = (1 - s) * self._unload_from + s * traj["unload_clearance"]
            if s >= 1.0:
                self._enter("recover", time)
        elif self.phase == "recover":
            if time - self.phase_start >= traj["recover_duration"] - 1e-9:
                self._enter("done", time)
        return self.z, traj["finger_closed_target"], self.phase

    @property
    def finished(self) -> bool:
        return self.phase == "done"

    def max_frames(self) -> int:
        traj = self.traj
        total = (
            traj["rest_duration"]
            + traj["approach_duration"]
            + traj["press_timeout"]
            + traj["hold_duration"]
            + traj["unload_duration"]
            + traj["recover_duration"]
        )
        return int(total * traj["frame_rate"] + 0.999999) + 2


# --------------------------------------------------------------------------
# Scene
# --------------------------------------------------------------------------


def pose_at(position, rotation=None):
    pose = np.eye(4, dtype=np.float32)
    if rotation is not None:
        pose[:3, :3] = rotation
    pose[:3, 3] = position
    return pose


def plate_start_position(inputs, bench: bool) -> np.ndarray:
    """Plate base-bottom centre at t=0."""
    config = inputs["config"]
    x, y = config["trajectory"]["block_xy"]
    if bench:
        top = float(inputs["vertices"][:, 2].max())
        radius = config["soft_block"]["particle_radius"]
        return np.array([x, y, top + radius + config["bench"]["bench_gap"]])
    tcp_z = inputs["contact_tcp_z"] + config["trajectory"]["rest_clearance"]
    return np.array([x, y, tcp_z - inputs["plate"]["tcp_above_base_bottom"]])


def newton_collision(config):
    contact = config["contact"]
    return dexsim.spawn.NewtonCollisionDesc(
        ke=contact["shape_ke"],
        kd=contact["shape_kd"],
        mu=contact["friction"],
        gap=contact["shape_gap"],
        margin=contact["shape_margin"],
    )


def bench_park_row(inputs) -> np.ndarray:
    """Bench robot pose: top IK row, joint 1 turned +90 deg, fingers open."""
    config = inputs["config"]
    arm = arm_target(inputs, float(inputs["ik_z"][-1]))
    arm[0] += 0.5 * np.pi
    finger = config["trajectory"]["finger_open"]
    return np.r_[arm, finger, finger]


def solver_config(config, device):
    solver = config["solver"]
    # DexUni does not expose the proxy mode; its two-way path is staggered.
    if solver["coupling_mode"] != "staggered":
        raise ValueError("DexUni two-way coupling is fixed to 'staggered'")
    cfg = NewtonCfg(
        device=device,
        dt=solver["dt"],
        num_substeps=solver["num_substeps"],
        collision_pipeline_cfg=None,
        solver_cfg=DexUniSolverCfg(
            mujoco_articulation_paths=("franka",),
            joint_mode=config["joint_mode"],
            contact_mode="full",
            coupling=config["coupling"],
            coupling_options=(
                {
                    "mass_scale": solver["proxy_mass_scale"],
                    "proxy_relaxation": solver["proxy_relaxation"],
                    "iterations": solver["coupling_iterations"],
                }
                if config["coupling"] == "two_way"
                else {}
            ),
            # DexUni proxies every MuJoCo link (solver.proxied_robot_links).
            mujoco_options={"cone": solver["mujoco_cone"]},
            vbd_options={
                "iterations": solver["vbd_iterations"],
                "friction_epsilon": solver["vbd_friction_epsilon"],
                "rigid_contact_hard": solver["vbd_rigid_contact_hard"],
                "rigid_contact_history": False,
                "rigid_body_contact_buffer_size": 4096,
                **solver.get("dexsim_vbd_extra", {}),
            },
            collision_options={
                "broad_phase": "nxn",
                "include_static_kinematic_pairs": False,
                "soft_contact_margin": config["contact"]["soft_contact_margin"],
            },
        ),
    )
    cfg.use_cuda_graph = solver["use_cuda_graph"]
    return cfg


def add_static_box(builder, config, name, center, size, color):
    geometry = dexsim.spawn.GeometryDesc.cube(tuple(size))
    collision = dexsim.spawn.CollisionDesc.from_geometry(geometry)
    collision.newton = newton_collision(config)
    builder.materials[name] = dexsim.spawn.MaterialDesc(
        name=name, base_color=(*color, 1.0)
    )
    desc = builder.add_cube(
        name,
        tuple(size),
        pose=pose_at(center),
        physics=dexsim.spawn.RigidBodyPhysicsDesc.static(),
        collisions=[collision],
        per_env=False,
    )
    desc.renders[0].material_ref = name


def add_robot(builder, config, initial_row):
    robot_cfg = config["robot"]
    robot = builder.add_urdf(
        "franka",
        robot_cfg["urdf"],
        pose=pose_at(robot_cfg["base_position"]),
        fix_root_link=True,
        per_env=False,
        newton_gravity_compensation=1.0,
        newton_collision=newton_collision(config),
    )
    arm, fingers = robot_cfg["arm_joints"], robot_cfg["finger_joints"]
    for i, name in enumerate(arm):
        robot.newton_joint_drives[name] = dexsim.spawn.NewtonJointDesc(
            target_ke=robot_cfg["arm_kp"][i],
            target_kd=robot_cfg["arm_kd"][i],
            effort_limit=robot_cfg["arm_effort_limit"][i],
            armature=robot_cfg["arm_armature"],
            target_mode=3,
        )
    for name in fingers:
        robot.newton_joint_drives[name] = dexsim.spawn.NewtonJointDesc(
            target_ke=robot_cfg["finger_kp"],
            target_kd=robot_cfg["finger_kd"],
            effort_limit=robot_cfg["finger_effort_limit"],
            armature=robot_cfg["finger_armature"],
            target_mode=3,
        )
    names = list(arm) + list(fingers)
    control = RealtimeJointTargetControl(
        "franka",
        initial_row.astype(np.float32),
        joint_names=names,
        initialize_pose=True,
    )
    builder.add_runtime_control(control)
    return control, names


def bench_weight(inputs) -> dict:
    """Flat bench weight in the same format as inputs/s01c_plate.json."""
    bench = inputs["config"]["bench"]
    sx, sy, sz = bench["weight_size"]
    m = float(bench["bench_mass"])
    return {
        "parts": [
            {"name": "weight", "center": [0.0, 0.0, 0.5 * sz], "size": [sx, sy, sz]}
        ],
        "density": m / (sx * sy * sz),
        "mass": m,
        "com": [0.0, 0.0, 0.5 * sz],
        "inertia": (
            m / 12.0 * np.diag([sy**2 + sz**2, sx**2 + sz**2, sx**2 + sy**2])
        ).tolist(),
    }


def add_press_plate(builder, inputs, bench: bool):
    config = inputs["config"]
    plate = bench_weight(inputs) if bench else inputs["plate"]
    mass, density = plate["mass"], plate["density"]
    builder.materials["plate"] = dexsim.spawn.MaterialDesc(
        name="plate", base_color=(0.18, 0.55, 0.78, 1.0)
    )
    collisions, renders = [], []
    for part in plate["parts"]:
        geometry = dexsim.spawn.GeometryDesc.cube(
            tuple(part["size"]), local_transform=pose_at(part["center"])
        )
        collision = dexsim.spawn.CollisionDesc.from_geometry(geometry)
        collision.newton = newton_collision(config)
        collisions.append(collision)
        renders.append(
            dexsim.spawn.RenderDesc.from_geometry(geometry, material_ref="plate")
        )
    builder.add_object(
        dexsim.spawn.ObjectDesc(
            name="press_plate",
            pose=pose_at(plate_start_position(inputs, bench)),
            renders=renders,
            collisions=collisions,
            physics=dexsim.spawn.RigidBodyPhysicsDesc.dynamic(density=density),
            per_env=False,
        )
    )
    return mass


def add_soft_block(builder, inputs):
    config = inputs["config"]
    block, contact = config["soft_block"], config["contact"]
    mu, lam = lame_parameters(block["youngs_modulus"], block["poisson_ratio"])
    builder.materials["soft_block"] = dexsim.spawn.MaterialDesc(
        name="soft_block", base_color=(0.90, 0.40, 0.20, 1.0)
    )
    soft = builder.create_soft_mesh(
        "soft_block",
        inputs["vertices"],
        inputs["tets"],
        surface_triangles=inputs["surface"],
        physics=dexsim.spawn.SoftBodyPhysicsDesc(
            volume_density=block["density"],
            k_mu=mu,
            k_lambda=lam,
            k_damp=block["k_damp"],
            contact_material=dexsim.spawn.NewtonParticleContactMaterialDesc(
                stiffness=contact["particle_contact_ke"],
                damping=contact["particle_contact_kd"],
                dynamic_friction=contact["friction"],
            ),
        ),
        particle_radius=block["particle_radius"],
        per_env=False,
    )
    # create_soft_mesh stores a bare GeometryDesc; wrap it so the render
    # mesh uses the registered material.
    soft.mesh = dexsim.spawn.RenderDesc.from_geometry(
        soft.mesh, material_ref="soft_block"
    )


def build_scene(world, inputs, initial_row, cameras: bool, bench: bool):
    config = inputs["config"]
    scene_cfg = config["scene"]
    builder = dexsim.spawn.SceneBuilder(world)
    ground = dexsim.spawn.CollisionDesc.from_geometry(
        dexsim.spawn.GeometryDesc.plane(8.0)
    )
    ground.newton = newton_collision(config)
    builder.materials["ground"] = dexsim.spawn.MaterialDesc(
        name="ground", base_color=(0.35, 0.36, 0.38, 1.0)
    )
    ground_desc = builder.add_plane(
        "ground",
        8.0,
        physics=dexsim.spawn.RigidBodyPhysicsDesc.static(),
        collisions=[ground],
        per_env=False,
    )
    ground_desc.renders[0].material_ref = "ground"
    add_static_box(
        builder,
        config,
        "table",
        scene_cfg["table"]["center"],
        scene_cfg["table"]["size"],
        (0.57, 0.48, 0.36),
    )
    add_static_box(
        builder,
        config,
        "robot_stand",
        scene_cfg["robot_stand"]["center"],
        scene_cfg["robot_stand"]["size"],
        (0.30, 0.32, 0.35),
    )
    control, names = add_robot(builder, config, initial_row)
    plate_mass = add_press_plate(builder, inputs, bench)
    add_soft_block(builder, inputs)
    builder.add_light(
        dexsim.spawn.LightDesc(
            name="key_light",
            light_type=dexsim.render.LightType.POINT,
            pose=pose_at((1.2, -1.0, 2.4)),
            intensity=60.0,
            shadow=False,
        )
    )
    if cameras:
        width, height = config["measurement"]["camera_resolution"]
        for name in config["measurement"]["cameras"]:
            builder.add_camera(
                dexsim.spawn.CameraDesc(
                    name, resolution=(width, height), near=0.01, far=10.0, headless=True
                )
            )
    scene = builder.finalize()
    return scene, control, names, plate_mass


def shape_table(world) -> dict:
    """Read-only per-shape contact settings of the finalized Newton model."""
    from dexsim.engine.newton_physics.backend_registry import get_newton_backend

    model = get_newton_backend(world).model
    labels = list(model.body_label)
    bodies = model.shape_body.numpy()
    table = {"body": [labels[b] if b >= 0 else "static" for b in bodies]}
    for name in (
        "shape_material_ke",
        "shape_material_kd",
        "shape_material_mu",
        "shape_margin",
        "shape_gap",
        "shape_flags",
    ):
        if hasattr(model, name):
            table[name] = getattr(model, name).numpy().tolist()
    for name in ("soft_contact_ke", "soft_contact_kd", "soft_contact_mu"):
        table[name] = float(getattr(model, name))
    table["particle_radius_max"] = float(model.particle_radius.numpy().max())
    return table


def dump_newton_model(model, path: Path) -> None:
    """Every array and scalar of the finalized Newton model (parity check)."""
    out = {}
    for name in dir(model):
        if name.startswith("_"):
            continue
        try:
            value = getattr(model, name)
        except Exception:
            continue
        if isinstance(value, wp.array):
            try:
                out[name] = value.numpy()
            except Exception:
                pass
        elif isinstance(value, (bool, int, float, np.integer, np.floating)):
            out[name] = np.asarray(value)
        elif (
            isinstance(value, (list, tuple))
            and value
            and all(isinstance(item, str) for item in value)
        ):
            out[name] = np.asarray(value)
        elif hasattr(value, "__len__") and not isinstance(value, (str, dict)):
            try:
                array = np.asarray(value, dtype=np.float64)
            except Exception:
                continue
            if array.ndim == 1 and array.size <= 16:
                out[name] = array
    np.savez_compressed(path, **out)


# --------------------------------------------------------------------------
# Readings and cameras
# --------------------------------------------------------------------------


class Readings:
    """Public Scene readers (no backend internals)."""

    def __init__(self, scene, config, names, bench: bool):
        self.plate = scene.get_object("press_plate")
        self.soft = scene.get_particle_set("soft_block")
        self.robot = scene.get_articulation("franka")
        layout = {
            item.name.rsplit("/", 1)[-1]: item.dof_start
            for item in self.robot.joint_dof_layout
        }
        self.dof_ids = np.array([layout[name] for name in names])
        self.tcp_body = config["robot"]["tcp_body"]
        self.tcp_offset = np.array([*config["robot"]["tcp_offset"], 1.0])
        self.finger_links = ("fr3_leftfinger", "fr3_rightfinger")

    def qpos(self):
        return np.asarray(self.robot.get_current_qpos())[self.dof_ids]

    def tcp(self):
        return (self.robot.get_link_pose(self.tcp_body) @ self.tcp_offset)[:3]

    def finger_poses(self):
        return np.stack([self.robot.get_link_pose(n) for n in self.finger_links])

    def plate_pose(self):
        return np.asarray(self.plate.get_world_pose())

    def particles(self):
        return self.soft.get_particle_positions().numpy().copy()


def intrinsics(config) -> np.ndarray:
    width, height = config["measurement"]["camera_resolution"]
    fx = (
        0.5 * width / np.tan(0.5 * np.deg2rad(config["measurement"]["camera_hfov_deg"]))
    )
    return np.array([[fx, 0, 0.5 * width], [0, fx, 0.5 * height], [0, 0, 1]])


class Cameras:
    """RGB + depth + visible-mask capture on the public camera handles."""

    def __init__(self, world, scene, config):
        self.world = world
        self.config = config
        width, height = config["measurement"]["camera_resolution"]
        k = intrinsics(config)
        vfov = float(np.rad2deg(2 * np.arctan(0.5 * height / k[1, 1])))
        self.cameras = {}
        for name, spec in config["measurement"]["cameras"].items():
            camera = scene.get_camera(name)
            camera.set_perspective(vfov, width / height, 0.01, 10.0)
            camera.set_intrinsic(k.astype(np.float32))
            camera.look_at(spec["eye"], spec["target"], spec["up"])
            camera.enable_depth()
            camera.enable_visible_mask()
            camera.open_camera()
            self.cameras[name] = camera
        self.label_of_id = self._label_table(scene)

    @staticmethod
    def _label_table(scene) -> dict[int, int]:
        """Native render user id -> common label id."""
        table = {}
        for name in ("ground", "table", "robot_stand", "press_plate"):
            table[int(scene.get_object(name).get_user_id())] = LABELS[name]
        for uid in scene.get_articulation("franka").get_user_ids():
            table[int(uid)] = LABELS["franka"]
        # The particle-set facade exposes no render user id. The soft block
        # is the only other render entity, so the one remaining non-zero id
        # is labelled soft_block (see Cameras.capture); tools/analyze.py
        # checks those pixels back-project onto the block's particles.
        return table

    def intrinsics(self):
        return {
            n: np.asarray(c.get_intrinsic()).tolist() for n, c in self.cameras.items()
        }

    def extrinsics(self):
        return {
            n: np.asarray(c.get_world_pose()).tolist() for n, c in self.cameras.items()
        }

    def capture(self, with_depth: bool):
        # Headless worlds skip the per-step Newton -> renderer sync; push the
        # current state for captured frames only, outside the step timing.
        from dexsim.engine.newton_physics.backend_registry import get_newton_backend

        get_newton_backend(self.world).sync_to_dexsim(self.world)
        self.world._flush_render_updates()
        frames = {}
        for name, camera in self.cameras.items():
            camera.render()
            frame = {"rgb": np.array(camera.get_rgb_map(), copy=True)[..., :3]}
            if with_depth:
                frame["depth"] = np.array(
                    camera.get_depth_map(), dtype=np.float32
                ).reshape(frame["rgb"].shape[:2])
                native = (
                    np.array(camera.get_visible_mask())
                    .reshape(frame["rgb"].shape[:2])
                    .astype(np.int64)
                )
                seg = np.zeros(native.shape, dtype=np.uint8)
                for uid, label in self.label_of_id.items():
                    seg[native == uid] = label
                others = [
                    v
                    for v in np.unique(native)
                    if v != 0 and int(v) not in self.label_of_id
                ]
                if len(others) == 1:
                    seg[native == others[0]] = LABELS["soft_block"]
                elif others:
                    seg[np.isin(native, others)] = 255
                frame["seg"] = seg
                frame["native_ids"] = np.unique(native)
            frames[name] = frame
        return frames


# --------------------------------------------------------------------------
# Bookkeeping
# --------------------------------------------------------------------------


def utc_stamp() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def git_state() -> dict:
    def run(*cmd):
        try:
            return subprocess.run(
                ["git", "-C", str(REPO_ROOT), *cmd],
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout.strip()
        except Exception as exc:  # pragma: no cover - diagnostic only
            return f"unavailable: {exc}"

    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty_files": run("status", "--porcelain").splitlines()[:200],
        "diff_sha256": hashlib.sha256(run("diff", "HEAD").encode()).hexdigest(),
    }


def gpu_info() -> list:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except Exception:
        return []
    return [
        dict(zip(("name", "memory", "driver"), [p.strip() for p in line.split(",")]))
        for line in out.strip().splitlines()
    ]


def to_json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value)}")


def write_json(path: Path, data) -> None:
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, default=to_json)
        stream.write("\n")


def downsample(image, factor: int):
    """Every factor-th pixel (rows/columns 0, f, 2f, ...)."""
    return np.ascontiguousarray(image[::factor, ::factor])


def stored_intrinsics(k, factor: int) -> np.ndarray:
    """K of an image built by ``downsample``.

    Stored pixel u is render pixel f*u, whose centre f*u + 0.5 maps to
    stored coordinate u + 0.5/f; hence fx/f and cx/f - 0.5/f + 0.5.
    """
    k = np.asarray(k, dtype=np.float64)
    out = k.copy()
    out[0, 0] = k[0, 0] / factor
    out[1, 1] = k[1, 1] / factor
    out[0, 2] = k[0, 2] / factor - 0.5 / factor + 0.5
    out[1, 2] = k[1, 2] / factor - 0.5 / factor + 0.5
    return out


class VideoWriter:

    def __init__(self, path: Path, fps: int):
        import imageio.v2 as imageio

        path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = imageio.get_writer(
            str(path),
            fps=fps,
            codec="libx264",
            quality=9,
            macro_block_size=16,
            pixelformat="yuv420p",
        )

    def append(self, rgb):
        self.writer.append_data(np.ascontiguousarray(rgb[..., :3]))

    def close(self):
        self.writer.close()


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------


def run(args):
    inputs = load_inputs(args)
    config = inputs["config"]
    bench = args.scenario == "bench"
    if bench:
        config["solver"]["vbd_iterations"] = args.bench_iterations
        kind_tag = f"bench_it{args.bench_iterations}"
    else:
        kind_tag = args.kind
    controller = CompressionController(config)
    finger0 = config["robot"]["finger_initial_q"]
    initial_row = (
        bench_park_row(inputs)
        if bench
        else np.r_[arm_target(inputs, controller.z), finger0, finger0]
    )
    run_id = (
        f"s01c_{PLATFORM}_{args.state}_{args.variant}_r{args.repeat:02d}"
        f"_{kind_tag}_{utc_stamp()}_{os.urandom(2).hex()}"
    )
    out_root = output_dir("s01c", args.out_root)
    out = out_root / "raw_data" / PLATFORM / run_id
    out.mkdir(parents=True, exist_ok=True)
    with_cameras = args.kind == "full" and args.images != "none"
    files = {
        "script": Path(__file__).resolve(),
        **{p.name: p for p in sorted(INPUTS.iterdir()) if p.is_file()},
    }
    manifest = {
        "run_id": run_id,
        "qualification": "unqualified_pilot",
        "protocol_revision": "dexuni-replan-v2",
        "task": config["task"],
        "subtask": config["subtask"],
        "platform": PLATFORM,
        "scenario": args.scenario,
        "state": inputs["state"],
        "variant": args.variant,
        "variant_settings": config["variants"][args.variant],
        "kind": args.kind,
        "images": args.images if with_cameras else "none",
        "repeat": args.repeat,
        "target_compression": config["target_compression"],
        "coupling": config["coupling"],
        "joint_mode": config["joint_mode"],
        "requested": {
            "num_substeps": config["solver"]["num_substeps"],
            "vbd_iterations": config["solver"]["vbd_iterations"],
            "use_cuda_graph": config["solver"]["use_cuda_graph"],
        },
        "started_utc": utc_stamp(),
        "files_sha256": {k: sha256_file(v) for k, v in files.items()},
        "robot_urdf_sha256": sha256_file(config["robot"]["urdf"]),
        "python": sys.version,
        "executable": sys.executable,
        "host": platform.node(),
        "gpus": gpu_info(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "repo": git_state(),
        "dexsim_version": getattr(dexsim, "__version__", None),
        "dexsim_package": dexsim.__file__,
        "newton_version": __import__("newton").__version__,
        "warp_version": wp.__version__,
        "solver_path": config["platform_solvers"][PLATFORM],
        "contact_force": "unavailable (no public per-contact force readout)",
        "joint_torque": "unavailable (not read; drive targets and positions are recorded)",
        "status": "running",
    }
    log_lines = []

    def log(message):
        print(message, flush=True)
        log_lines.append(message)

    from benchlib.gpu_memory import GpuProcessMemoryTracker

    memory = GpuProcessMemoryTracker(gpu_id=0)
    memory_ok = memory.start()
    columns: dict[str, list] = {}
    image_columns: dict[str, dict[str, list]] = {}
    timing_rows, videos = [], {}
    t_start = time.perf_counter()
    try:
        world_cfg = dexsim.WorldConfig()
        world_cfg.use_default_physics = False
        world_cfg.open_windows = False
        world_cfg.newton_cfg = solver_config(config, args.device)
        world = dexsim.World(world_cfg)
        world.set_manual_update(True)
        scene, control, names, plate_mass = build_scene(
            world, inputs, initial_row, with_cameras, bench
        )
        world.enable_physics(True)
        manifest["plate_mass"] = plate_mass
        if args.kind == "full":
            from dexsim.engine.newton_physics.backend_registry import get_newton_backend

            dump_newton_model(get_newton_backend(world).model, out / "newton_model.npz")
        readings = Readings(scene, config, names, bench)
        cameras = None
        if with_cameras:
            cameras = Cameras(world, scene, config)
            manifest["camera_intrinsics"] = cameras.intrinsics()
            manifest["camera_extrinsics_world_pose"] = cameras.extrinsics()
            manifest["camera_extrinsics_convention"] = (
                "SpawnedCamera.get_world_pose(): camera-to-world 4x4, "
                "camera looks along its local -Z, +Y up (OpenGL)"
            )
            manifest["segmentation_native_to_label"] = cameras.label_of_id
            fps = config["measurement"]["camera_rate_hz"]
            for name in cameras.cameras:
                if args.images == "full":
                    # Raw source for the single comparison video.
                    videos[name] = VideoWriter(out / f"rgb_{name}.mp4", fps)
                image_columns[name] = {
                    "frame": [],
                    "time": [],
                    "rgb": [],
                    "depth": [],
                    "seg": [],
                }
        init_seconds = time.perf_counter() - t_start
        memory.sample("init")
        log(
            f"[{run_id}] init {init_seconds:.2f}s, particles="
            f"{readings.soft.particle_count}"
        )
        rate = config["trajectory"]["frame_rate"]
        stride = rate // config["measurement"]["camera_rate_hz"]
        frame_dt = 1.0 / rate
        rest_particles = readings.particles()
        record_states = args.kind == "full"
        factor = int(config["measurement"]["image_downsample"])

        def record(now, target, z_cmd, phase):
            particles = readings.particles()
            row = {
                "time": now,
                "phase": controller.PHASES.index(phase),
                "plate_pose": readings.plate_pose(),
                "particles": particles,
                "target_q": target,
                "tcp_cmd_z_offset": z_cmd,
                "qpos": readings.qpos(),
                "tcp": readings.tcp(),
                "finger_poses": readings.finger_poses(),
            }
            for key, value in row.items():
                columns.setdefault(key, []).append(np.array(value, copy=True))
            return particles

        def capture(frame, now, keyframe):
            depth = args.images == "full" or keyframe
            for name, data in cameras.capture(depth).items():
                if name in videos:
                    videos[name].append(data["rgb"])
                if depth:
                    col = image_columns[name]
                    col["frame"].append(frame)
                    col["time"].append(now)
                    if name not in videos:  # keyframe runs keep RGB here
                        col["rgb"].append(downsample(data["rgb"], factor))
                    col["depth"].append(downsample(data["depth"], factor))
                    col["seg"].append(downsample(data["seg"], factor))
                    manifest.setdefault("segmentation_native_ids_seen", set()).update(
                        int(v) for v in data["native_ids"]
                    )

        if record_states:
            record(0.0, initial_row, controller.z, controller.phase)
            if cameras:
                capture(0, 0.0, True)
        sim_seconds = 0.0
        height = float(np.ptp(rest_particles[:, 2]))
        frame = 0
        bench_frames = int(round(config["bench"]["duration"] * rate))
        last_frames = bench_frames if bench else controller.max_frames()
        phase = controller.phase
        while frame < last_frames and not (not bench and controller.finished):
            frame += 1
            now = frame * frame_dt
            previous = phase
            target = initial_row
            z_cmd = float("nan")
            if not bench:
                z_cmd, finger, phase = controller.step(now, height)
                target = np.r_[arm_target(inputs, z_cmd), finger, finger]
                control.set_target(target.astype(np.float32))
            wp.synchronize_device(args.device)
            started = time.perf_counter()
            world.update(config["solver"]["dt"])
            wp.synchronize_device(args.device)
            step = time.perf_counter() - started
            sim_seconds += step
            capture_seconds = 0.0
            if record_states:
                started = time.perf_counter()
                particles = record(now, target, z_cmd, phase)
                last = frame == last_frames or (not bench and controller.finished)
                if cameras and frame % stride == 0:
                    capture(frame, now, phase != previous or last)
                elif cameras and (phase != previous or last):
                    capture(frame, now, True)
                capture_seconds = time.perf_counter() - started
            else:
                particles = readings.particles()
            # The closed-loop rule needs the measured height every frame.
            height = float(np.ptp(particles[:, 2]))
            timing_rows.append((frame, now, step, capture_seconds, phase))
            if frame == 2:
                manifest["cuda_graph_status_after_first_steps"] = (
                    scene.get_solver_diagnostics().get("cuda_graph_status")
                )
            if frame % 30 == 0:
                memory.sample(f"frame_{frame}")
                extra = (
                    f" z_cmd={z_cmd * 1e3:+6.2f}mm "
                    f"tcp_z={readings.tcp()[2]:.4f}"
                    f" finger={readings.qpos()[7] * 1e3:.2f}mm"
                )
                log(
                    f"t={now:5.2f} {phase:8s} h={height * 1e3:6.2f}mm{extra}"
                    f" step={step * 1e3:.2f}ms"
                )
                if not np.isfinite(particles).all():
                    raise FloatingPointError("Non-finite particle state")
        manifest["newton_shapes"] = shape_table(world)
        diagnostics = scene.get_solver_diagnostics()
        manifest["solver_diagnostics"] = diagnostics
        manifest["effective"] = {
            "num_substeps": diagnostics.get("num_substeps"),
            "cuda_graph_status": diagnostics.get("cuda_graph_status"),
            "dt": diagnostics.get("dt"),
        }
        if not bench:
            manifest["controller"] = {
                "h0": controller.h0,
                "h_target": controller.h_target,
                "target_reached": controller.target_reached,
                "events": controller.events,
                "finished": controller.finished,
            }
        manifest.update(
            status="completed",
            sim_seconds=sim_seconds,
            init_seconds=init_seconds,
            frames=frame,
        )
        if record_states:
            arrays = {k: np.stack(v) for k, v in columns.items()}
            np.savez_compressed(
                out / "states.npz",
                **arrays,
                joint_names=np.asarray(names),
                rest_particles=rest_particles,
                tets=inputs["tets"],
            )
        for name, col in image_columns.items():
            if col["frame"]:
                np.savez_compressed(
                    out / f"images_{name}.npz",
                    frame=np.asarray(col["frame"]),
                    time=np.asarray(col["time"]),
                    rgb=(
                        np.stack(col["rgb"])
                        if col["rgb"]
                        else np.zeros((0, 0, 0, 3), np.uint8)
                    ),
                    depth=np.stack(col["depth"]),
                    seg=np.stack(col["seg"]),
                    K=stored_intrinsics(manifest["camera_intrinsics"][name], factor),
                    K_render=np.asarray(manifest["camera_intrinsics"][name]),
                    camera_to_world=np.asarray(
                        manifest["camera_extrinsics_world_pose"][name]
                    ),
                )
        del readings
        scene.close()
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = repr(exc)
        log(traceback.format_exc())
        raise
    finally:
        for writer in videos.values():
            writer.close()
        if "segmentation_native_ids_seen" in manifest:
            manifest["segmentation_native_ids_seen"] = sorted(
                manifest["segmentation_native_ids_seen"]
            )
        manifest["gpu_memory"] = (
            memory.metrics() if memory_ok else {"status": "unavailable"}
        )
        memory.close()
        manifest["finished_utc"] = utc_stamp()
        with open(out / "timing.csv", "w") as stream:
            stream.write("frame,sim_time,step_seconds,capture_seconds,phase\n")
            for row in timing_rows:
                stream.write(",".join(str(v) for v in row) + "\n")
        write_json(out / "config.json", config)
        write_json(out / "manifest.json", manifest)
        (out / "run.log").write_text("\n".join(log_lines) + "\n")
        print(f"run directory: {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--scenario", choices=("task", "bench"), default="task")
    parser.add_argument("--state", default="s00")
    parser.add_argument("--variant", default="nominal")
    parser.add_argument("--kind", choices=("full", "timing"), default="full")
    parser.add_argument(
        "--images", choices=("full", "keyframes", "none"), default="full"
    )
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--bench-iterations", type=int, default=40)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--inputs", help="alternative inputs directory (mesh-level tests)"
    )
    parser.add_argument(
        "--set", action="append", help="debug override a.b=<json>, recorded in config"
    )
    parser.add_argument(
        "--out-root", help="write raw_data/ and videos/ under this root"
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
