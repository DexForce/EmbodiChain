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
"""Unqualified S02 embodichain pilot adapter.

Run from the repository root with DEXUNI_ALLOW_PILOT=1:
    <platform-python> scripts/benchmark/tech_reports/physics/platforms/dexuni/embodichain/s02.py --state s00 --images keyframes

Inputs: DEXUNI_INPUT_ROOT/s02, or --inputs DIR (read-only snapshot).
Outputs: outputs/benchmarks/tech_reports/dexuni/development/s02, or --out-root DIR.
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
import torch
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

TASK_DIR = EXPERIMENTS / "s02_soft_pick_place"
INPUTS = input_dir("s02")
BENCH_ROOT = PLATFORMS
if __name__ == "__main__":
    require_pilot()
PLATFORM = "embodichain"
sys.path.insert(0, str(BENCH_ROOT / "_lib"))  # benchlib copy

import dexsim  # noqa: E402
import embodichain  # noqa: E402
from dexsim.engine.newton_physics import RealtimeJointTargetControl  # noqa: E402
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg  # noqa: E402
from embodichain.lab.sim.cfg import (  # noqa: E402
    ArticulationCfg,
    ArticulationRootPropertiesCfg,
    LightCfg,
    NewtonCollisionPropertiesCfg,
    NewtonJointDrivePropertiesCfg,
    NewtonPhysicsCfg,
    NewtonRigidBodyMaterialCfg,
    RenderCfg,
    RigidBodyPhysicsCfg,
    RigidObjectCfg,
    VolumeDeformableObjectCfg,
    VolumeDeformablePhysicsCfg,
)
from embodichain.lab.sim.material import VisualMaterialCfg  # noqa: E402
from embodichain.lab.sim.sensors import CameraCfg  # noqa: E402
from embodichain.lab.sim.shapes import CubeCfg, MeshCfg  # noqa: E402

# Common segmentation label ids (config measurement.images.labels).
LABELS = {"ground": 1, "table": 2, "robot_stand": 3, "franka": 4, "soft_block": 5}


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
    with open(INPUTS / "s02_config.json", encoding="utf-8") as stream:
        config = json.load(stream)
    for key, value in config["variants"][args.variant].items():
        set_path(config, key, value)
    for item in args.set or []:
        key, value = item.split("=", 1)
        set_path(config, key, json.loads(value))
        config.setdefault("_overrides", []).append(item)
    mesh = np.load(INPUTS / "s02_block_mesh.npz")
    states = np.load(INPUTS / "s02_initial_states.npz")
    index = [str(s) for s in states["state_ids"]].index(args.state)
    ik = np.load(INPUTS / "s02_ik_path.npz")
    return {
        "config": config,
        "tets": mesh["tets"],
        "surface": mesh["surface_triangles"],
        "patches": mesh["width_patches"],
        # World-space initial node positions; the block starts at rest.
        "initial": states["initial_positions"][index].astype(np.float32),
        "state": {
            "id": args.state,
            "block_center": states["block_center"][index].tolist(),
            "block_yaw_deg": float(states["block_yaw_deg"][index]),
        },
        "ik_s": np.asarray(ik["s"], dtype=np.float64),
        "ik_q": np.asarray(ik["arm_q"], dtype=np.float64),
        "waypoint_s": np.asarray(ik["waypoint_s"], dtype=np.float64),
    }


def arm_target(inputs, s: float) -> np.ndarray:
    """Interpolate the shared IK path table (arc length s -> 7 arm joints)."""
    x = inputs["ik_s"]
    if not x[0] - 1e-9 <= s <= x[-1] + 1e-9:
        raise ValueError(f"path parameter {s:.4f} m outside the IK table")
    return np.array([np.interp(s, x, col) for col in inputs["ik_q"].T])


def block_width(particles: np.ndarray, patches: np.ndarray) -> float:
    """Distance between the -Y and +Y face patch centroids [m]."""
    return float(
        np.linalg.norm(
            particles[patches[1]].mean(axis=0) - particles[patches[0]].mean(axis=0)
        )
    )


def lame_parameters(youngs_modulus: float, poisson_ratio: float):
    e, nu = float(youngs_modulus), float(poisson_ratio)
    return e / (2.0 * (1.0 + nu)), e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))


# --------------------------------------------------------------------------
# Closed-loop pick-and-place rule (identical class in every platform script)
# --------------------------------------------------------------------------


class PickPlaceController:
    """S02 closed-loop pick-and-place rule; kept verbatim in all scripts.

    Input: frame time and the measured block width ``w``. Output: commanded
    path arc length ``s`` (IK table parameter), finger target and phase
    name. ``w0`` is the mean filtered width over the reference window.
    """

    PHASES = (
        "settle",
        "squeeze",
        "squeeze_settle",
        "lift",
        "hold",
        "transport",
        "lower",
        "release",
        "retreat",
        "observe",
        "done",
    )
    # Path moves: phase -> (from waypoint index, to waypoint index, duration key)
    MOVES = {
        "lift": (0, 1, "lift_duration"),
        "transport": (1, 2, "transport_duration"),
        "lower": (2, 3, "lower_duration"),
        "retreat": (3, 4, "retreat_duration"),
    }
    NEXT = {
        "settle": "squeeze",
        "squeeze": "squeeze_settle",
        "squeeze_settle": "lift",
        "lift": "hold",
        "hold": "transport",
        "transport": "lower",
        "lower": "release",
        "release": "retreat",
        "retreat": "observe",
        "observe": "done",
    }

    def __init__(self, config: dict, waypoint_s):
        traj = config["trajectory"]
        self.traj = traj
        self.grip = config["grip"]
        self.waypoint_s = [float(v) for v in waypoint_s]
        self.dt = 1.0 / traj["frame_rate"]
        self.target = float(self.grip["target_compression"])
        self.reference_window = config["measurement"]["reference_window"]
        self.filter_frames = int(traj["width_filter_frames"])
        self.q_open0 = float(config["robot"]["finger_initial_q"])
        self.s = 0.0
        self.finger = self.q_open0
        self.finger_closed = None
        self.phase = "settle"
        self.phase_start = 0.0
        self.w0 = None
        self.squeeze_stop = None
        self.events = {"settle": 0.0}
        self._ref = []
        self._window = []

    @staticmethod
    def _quintic(s: float) -> float:
        s = min(max(float(s), 0.0), 1.0)
        return s**3 * (10.0 + s * (-15.0 + 6.0 * s))

    def _enter(self, phase: str, time: float) -> None:
        self.phase, self.phase_start = phase, time
        self.events[phase] = time

    def compression(self) -> float | None:
        if self.w0 is None or not self._window:
            return None
        return 1.0 - sum(self._window) / len(self._window) / self.w0

    def _duration(self, phase: str) -> float:
        if phase in self.MOVES:
            return self.traj[self.MOVES[phase][2]]
        return {
            "settle": self.traj["settle_duration"],
            "squeeze_settle": self.grip["settle_duration"],
            "hold": self.traj["hold_duration"],
            "release": self.grip["release_duration"],
            "observe": self.traj["observe_duration"],
        }[phase]

    def step(self, time: float, width: float):
        """Advance to ``time`` given the width measured at ``time - dt``."""
        grip = self.grip
        self._window = (self._window + [float(width)])[-self.filter_frames :]
        filtered = sum(self._window) / len(self._window)
        elapsed = time - self.phase_start
        if self.phase == "settle":
            lo, hi = self.reference_window
            if lo <= time - self.dt <= hi + 1e-9:
                self._ref.append(filtered)
            if elapsed >= self._duration("settle") - 1e-9:
                self.w0 = sum(self._ref) / len(self._ref)
                self._enter("squeeze", time)
        elif self.phase == "squeeze":
            compression = 1.0 - filtered / self.w0
            reason = None
            if compression >= self.target:
                reason = "target"
            elif self.finger <= grip["finger_min"] + 1e-12:
                reason = "finger_min"
            elif elapsed >= grip["squeeze_timeout"] - 1e-9:
                reason = "timeout"
            if reason is None:
                self.finger = max(
                    self.finger - grip["close_speed"] * self.dt, grip["finger_min"]
                )
            else:
                self.finger_closed = self.finger
                self.squeeze_stop = {
                    "reason": reason,
                    "time": time,
                    "compression": compression,
                    "finger_target": self.finger,
                }
                self._enter("squeeze_settle", time)
        elif self.phase in self.MOVES:
            a, b, _ = self.MOVES[self.phase]
            u = self._quintic(elapsed / self._duration(self.phase))
            self.s = (1 - u) * self.waypoint_s[a] + u * self.waypoint_s[b]
            if u >= 1.0:
                self._enter(self.NEXT[self.phase], time)
        elif self.phase == "release":
            u = self._quintic(elapsed / self._duration("release"))
            self.finger = (1 - u) * self.finger_closed + u * grip["finger_open"]
            if u >= 1.0:
                self._enter("retreat", time)
        elif self.phase in ("squeeze_settle", "hold", "observe"):
            if elapsed >= self._duration(self.phase) - 1e-9:
                self._enter(self.NEXT[self.phase], time)
        return self.s, self.finger, self.phase

    @property
    def finished(self) -> bool:
        return self.phase == "done"

    def max_frames(self) -> int:
        traj, grip = self.traj, self.grip
        total = (
            traj["settle_duration"]
            + grip["squeeze_timeout"]
            + grip["settle_duration"]
            + traj["lift_duration"]
            + traj["hold_duration"]
            + traj["transport_duration"]
            + traj["lower_duration"]
            + grip["release_duration"]
            + traj["retreat_duration"]
            + traj["observe_duration"]
        )
        return int(total * traj["frame_rate"] + 0.999999) + 2


# --------------------------------------------------------------------------
# Scene
# --------------------------------------------------------------------------


def contact_physics(config) -> RigidBodyPhysicsCfg:
    """Shape contact settings shared by every rigid shape (config.contact)."""
    contact = config["contact"]
    return RigidBodyPhysicsCfg(
        collision_props=NewtonCollisionPropertiesCfg(
            margin=contact["shape_margin"], gap=contact["shape_gap"]
        ),
        material_props=NewtonRigidBodyMaterialCfg(
            ke=contact["shape_ke"],
            kd=contact["shape_kd"],
            dynamic_friction=contact["friction"],
        ),
    )


def physics_config(config, device, render: bool) -> SimulationManagerCfg:
    solver = config["solver"]
    # DexUni does not expose the proxy mode; its two-way path is staggered.
    if solver["coupling_mode"] != "staggered":
        raise ValueError("DexUni two-way coupling is fixed to 'staggered'")
    solver_cfg = {
        "solver_type": "dexuni",
        # mujoco_articulation_paths is left unset: the Franka is the only
        # articulation, so DexUni's default selection is the same set.
        "joint_mode": config["joint_mode"],
        "contact_mode": "full",
        "coupling": config["coupling"],
        "coupling_options": (
            {
                "mass_scale": solver["proxy_mass_scale"],
                "proxy_relaxation": solver["proxy_relaxation"],
                "iterations": solver["coupling_iterations"],
            }
            if config["coupling"] == "two_way"
            else {}
        ),
        "mujoco_options": {"cone": solver["mujoco_cone"]},
        "vbd_options": {
            "iterations": solver["vbd_iterations"],
            "friction_epsilon": solver["vbd_friction_epsilon"],
            "rigid_contact_hard": solver["vbd_rigid_contact_hard"],
            "rigid_contact_history": False,
            "rigid_body_contact_buffer_size": 4096,
            **solver.get("dexsim_vbd_extra", {}),
        },
        "collision_options": {
            "broad_phase": "nxn",
            "include_static_kinematic_pairs": False,
            "soft_contact_margin": config["contact"]["soft_contact_margin"],
        },
    }
    width, height = config["measurement"]["camera_resolution"]
    return SimulationManagerCfg(
        width=width,
        height=height,
        headless=True,
        num_envs=1,
        device=device,
        enable_entity_gizmo=False,
        robot_ik_gizmo=None,
        render_cfg=RenderCfg(renderer="hybrid" if render else "no-render"),
        physics_cfg=NewtonPhysicsCfg(
            device=device,
            physics_dt=solver["dt"],
            num_substeps=solver["num_substeps"],
            use_cuda_graph=solver["use_cuda_graph"],
            collision_cfg=None,
            solver_cfg=solver_cfg,
        ),
    )


def visual(uid, rgb) -> VisualMaterialCfg:
    return VisualMaterialCfg(uid=uid, base_color=[*rgb, 1.0])


def add_static_box(sim, config, name, center, size, color):
    return sim.add_rigid_object(
        RigidObjectCfg(
            uid=name,
            shape=CubeCfg(size=list(size), visual_material=visual(name, color)),
            body_type="static",
            init_pos=tuple(center),
            attrs=contact_physics(config),
        )
    )


def add_robot(sim, config, initial_row):
    robot_cfg = config["robot"]
    arm, fingers = robot_cfg["arm_joints"], robot_cfg["finger_joints"]
    names = list(arm) + list(fingers)
    kp = {
        **dict(zip(arm, robot_cfg["arm_kp"])),
        **{n: robot_cfg["finger_kp"] for n in fingers},
    }
    kd = {
        **dict(zip(arm, robot_cfg["arm_kd"])),
        **{n: robot_cfg["finger_kd"] for n in fingers},
    }
    effort = {
        **dict(zip(arm, robot_cfg["arm_effort_limit"])),
        **{n: robot_cfg["finger_effort_limit"] for n in fingers},
    }
    armature = {
        **{n: robot_cfg["arm_armature"] for n in arm},
        **{n: robot_cfg["finger_armature"] for n in fingers},
    }
    robot = sim.add_articulation(
        ArticulationCfg(
            uid="franka",
            fpath=robot_cfg["urdf"],
            init_pos=tuple(robot_cfg["base_position"]),
            init_qpos=[float(v) for v in initial_row],
            asset_physics_mode="overlay",
            build_pk_chain=False,
            attrs=contact_physics(config),
            # Keep the configured finger_joint2 drive and the native MuJoCo
            # mimic equality, as s02_dexsim.py does (also EmbodiChain's
            # default; written out because results depend on it).
            root_props=ArticulationRootPropertiesCfg(
                fixed_base=True,
                newton_gravity_compensation=1.0,
                newton_mimic_stabilizer=False,
            ),
            joint_drive_props=NewtonJointDrivePropertiesCfg(
                target_mode="position_velocity",
                stiffness=kp,
                damping=kd,
                max_effort=effort,
                armature=armature,
            ),
        )
    )
    # Same real-time target control as s02_dexsim.py, registered through
    # EmbodiChain's public runtime-control entry point.
    (control,) = sim.add_runtime_control(
        lambda arena, _env: RealtimeJointTargetControl(
            f"{arena}/franka",
            initial_row.astype(np.float32),
            joint_names=names,
            initialize_pose=True,
        )
    )
    return robot, control, names, control.target


def add_soft_block(sim, inputs):
    """Soft block A at its (perturbed) initial pose; rest shape = initial."""
    config = inputs["config"]
    block, contact = config["soft_block"], config["contact"]
    return sim.add_deformable_object(
        VolumeDeformableObjectCfg(
            uid="soft_block",
            shape=MeshCfg(
                vertices=inputs["initial"],
                triangles=inputs["surface"],
                visual_material=visual("soft_block", (0.90, 0.40, 0.20)),
            ),
            tetrahedra=inputs["tets"],
            particle_radius=block["particle_radius"],
            attrs=VolumeDeformablePhysicsCfg(
                youngs=block["youngs_modulus"],
                poissons=block["poisson_ratio"],
                density=block["density"],
                elasticity_damping=block["k_damp"],
                contact_material={
                    "stiffness": contact["particle_contact_ke"],
                    "damping": contact["particle_contact_kd"],
                    "dynamic_friction": contact["friction"],
                },
            ),
        )
    )


def intrinsics(config) -> np.ndarray:
    width, height = config["measurement"]["camera_resolution"]
    fx = (
        0.5 * width / np.tan(0.5 * np.deg2rad(config["measurement"]["camera_hfov_deg"]))
    )
    return np.array([[fx, 0, 0.5 * width], [0, fx, 0.5 * height], [0, 0, 1]])


def build_scene(sim, inputs, initial_row, cameras: bool):
    config = inputs["config"]
    scene_cfg = config["scene"]
    # EmbodiChain always owns a global ground plane ("default_plane",
    # mu 0.5, Newton default stiffness); nothing in S02 touches it.
    add_static_box(
        sim,
        config,
        "table",
        scene_cfg["table"]["center"],
        scene_cfg["table"]["size"],
        (0.57, 0.48, 0.36),
    )
    add_static_box(
        sim,
        config,
        "robot_stand",
        scene_cfg["robot_stand"]["center"],
        scene_cfg["robot_stand"]["size"],
        (0.30, 0.32, 0.35),
    )
    robot, control, names, robot_path = add_robot(sim, config, initial_row)
    block = add_soft_block(sim, inputs)
    camera_sensors = {}
    if cameras:
        sim.add_light(
            LightCfg(
                uid="key_light",
                light_type="point",
                init_pos=(1.2, -1.0, 2.4),
                intensity=60.0,
                enable_shadow=False,
            )
        )
        width, height = config["measurement"]["camera_resolution"]
        k = intrinsics(config)
        for name, spec in config["measurement"]["cameras"].items():
            camera_sensors[name] = sim.add_sensor(
                CameraCfg(
                    uid=name,
                    width=width,
                    height=height,
                    near=0.01,
                    far=10.0,
                    intrinsics=(
                        float(k[0, 0]),
                        float(k[1, 1]),
                        float(k[0, 2]),
                        float(k[1, 2]),
                    ),
                    enable_depth=True,
                    enable_mask=True,
                    extrinsics=CameraCfg.ExtrinsicsCfg(
                        eye=tuple(spec["eye"]),
                        target=tuple(spec["target"]),
                        up=tuple(spec["up"]),
                    ),
                )
            )
    sim.prepare()
    return robot, block, control, names, camera_sensors, robot_path


def newton_model(sim):
    from dexsim.engine.newton_physics.backend_registry import get_newton_backend

    return get_newton_backend(sim.get_world()).model


def shape_table(model) -> dict:
    """Read-only per-shape contact settings of the finalized Newton model."""
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
    """EmbodiChain object readers (single environment)."""

    def __init__(self, robot, block, config, names):
        self.robot, self.block = robot, block
        self.joint_ids = [robot.joint_names.index(name) for name in names]
        self.tcp_body = config["robot"]["tcp_body"]
        self.tcp_offset = torch.tensor(
            [*config["robot"]["tcp_offset"], 1.0], device=robot.device
        )
        self.finger_links = ("fr3_leftfinger", "fr3_rightfinger")

    def qpos(self):
        return self.robot.get_qpos()[0, self.joint_ids].cpu().numpy()

    def tcp(self):
        pose = self.robot.get_link_pose(self.tcp_body, to_matrix=True)[0]
        return (pose @ self.tcp_offset)[:3].cpu().numpy()

    def finger_poses(self):
        return np.stack(
            [
                self.robot.get_link_pose(n, to_matrix=True)[0].cpu().numpy()
                for n in self.finger_links
            ]
        )

    def particles(self):
        return self.block.data.nodal_pos_w[0].cpu().numpy().copy()


class Cameras:
    """RGB + depth + visible-mask capture on EmbodiChain camera sensors."""

    def __init__(self, sim, sensors, robot_path):
        self.sim = sim
        self.cameras = sensors
        self.label_of_id = self._label_table(sim, robot_path)

    @staticmethod
    def _label_table(sim, robot_path) -> dict[int, int]:
        """Native render user id -> common label id."""
        result = sim.spawn_result
        table = {
            int(result.get_object("default_plane").get_user_id()): LABELS["ground"]
        }
        for name in ("table", "robot_stand"):
            for uid in sim.get_rigid_object(name).user_ids.reshape(-1):
                table[int(uid)] = LABELS[name]
        # Articulation.user_ids keeps one id per link; the Spawn handle lists
        # every render id of the robot.
        for uid in result.get_articulation(robot_path).get_user_ids():
            table[int(uid)] = LABELS["franka"]
        # As in s02_dexsim.py, the particle set exposes no render user id;
        # the one remaining non-zero id is labelled soft_block.
        return table

    def intrinsics(self):
        return {
            n: c.get_intrinsics()[0].cpu().numpy().tolist()
            for n, c in self.cameras.items()
        }

    def extrinsics(self):
        return {
            n: c.get_arena_pose(to_matrix=True)[0].cpu().numpy().tolist()
            for n, c in self.cameras.items()
        }

    def capture(self, with_depth: bool):
        self.sim.sync_render_state()
        frames = {}
        for name, camera in self.cameras.items():
            camera.update()
            data = camera.get_data()
            frame = {"rgb": data["color"][0, ..., :3].cpu().numpy()}
            if with_depth:
                frame["depth"] = data["depth"][0].cpu().numpy().astype(np.float32)
                native = data["mask"][0].cpu().numpy().astype(np.int64)
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


def embodichain_git_state() -> dict:
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

    return {"branch": run("rev-parse", "--abbrev-ref", "HEAD")}


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------


def run(args):
    inputs = load_inputs(args)
    config = inputs["config"]
    controller = PickPlaceController(config, inputs["waypoint_s"])
    finger0 = config["robot"]["finger_initial_q"]
    initial_row = np.r_[arm_target(inputs, 0.0), finger0, finger0]
    run_id = (
        f"s02_{PLATFORM}_{args.state}_{args.variant}_r{args.repeat:02d}"
        f"_{args.kind}_{utc_stamp()}_{os.urandom(2).hex()}"
    )
    out_root = output_dir("s02", args.out_root)
    out = out_root / "raw_data" / PLATFORM / run_id
    out.mkdir(parents=True, exist_ok=True)
    with_cameras = args.kind == "full" and args.images != "none"
    files = {
        "script": Path(__file__).resolve(),
        **{p.name: p for p in sorted(INPUTS.iterdir()) if p.is_file()},
    }
    manifest = {
        "run_id": run_id,
        "qualification": "native_platform_comparison",
        "control_contract": "native platform control; not matched replay",
        "timing_protocol": {
            "warmup_frames": 30,
            "scope": "synchronized completed outer step; excludes control, readback and capture",
            "window": "episode frames after 30 initial settling frames; no reset",
        },
        "protocol_revision": "dexuni-fixed-five-v1",
        "task": config["task"],
        "subtask": config["subtask"],
        "platform": PLATFORM,
        "scenario": "task",
        "state": inputs["state"],
        "variant": args.variant,
        "variant_settings": config["variants"][args.variant],
        "kind": args.kind,
        "images": args.images if with_cameras else "none",
        "repeat": args.repeat,
        "target_compression": config["grip"]["target_compression"],
        "coupling": config["coupling"],
        "joint_mode": config["joint_mode"],
        "requested": {
            "num_substeps": config["solver"]["num_substeps"],
            "vbd_iterations": config["solver"]["vbd_iterations"],
            "use_cuda_graph": config["solver"]["use_cuda_graph"],
        },
        "renderer": "hybrid" if with_cameras else "no-render",
        "started_utc": utc_stamp(),
        "files_sha256": {k: sha256_file(v) for k, v in files.items()},
        "robot_urdf_sha256": sha256_file(config["robot"]["urdf"]),
        "python": sys.version,
        "executable": sys.executable,
        "host": platform.node(),
        "gpus": gpu_info(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "repo": {**git_state(), **embodichain_git_state()},
        "embodichain_package": embodichain.__file__,
        "dexsim_version": getattr(dexsim, "__version__", None),
        "dexsim_package": dexsim.__file__,
        "newton_version": __import__("newton").__version__,
        "warp_version": wp.__version__,
        "solver_path": config["platform_solvers"]["dexsim"],
        "differences_from_dexsim_script": [
            "EmbodiChain adds its own global ground plane (mu 0.5, Newton "
            "default stiffness, 1000 m); the DexSim script's 8 m plane uses "
            "the config contact values. Nothing in S02 touches the ground.",
            "mujoco_articulation_paths unset (DexSim script: ('franka',)); "
            "the Franka is the only articulation",
        ],
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
    sim = None
    t_start = time.perf_counter()
    try:
        sim = SimulationManager(physics_config(config, args.device, with_cameras))
        robot, block, control, names, sensors, robot_path = build_scene(
            sim, inputs, initial_row, with_cameras
        )
        model = newton_model(sim)
        if args.kind == "full":
            dump_newton_model(model, out / "newton_model.npz")
        readings = Readings(robot, block, config, names)
        # Width-patch node ids assume the simulated node order equals the
        # mesh order.
        order_error = float(np.abs(readings.particles() - inputs["initial"]).max())
        if order_error > 1e-5:
            raise RuntimeError(
                f"node order/positions differ from the "
                f"initial mesh by {order_error:.2e} m"
            )
        cameras = None
        if with_cameras:
            cameras = Cameras(sim, sensors, robot_path)
            manifest["camera_intrinsics"] = cameras.intrinsics()
            manifest["camera_extrinsics_world_pose"] = cameras.extrinsics()
            manifest["camera_extrinsics_convention"] = (
                "Camera.get_arena_pose(): camera-to-world 4x4 (single arena "
                "at the origin), camera looks along its local -Z, +Y up "
                "(OpenGL)"
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
            f"{len(inputs['initial'])}"
        )
        rate = config["trajectory"]["frame_rate"]
        stride = rate // config["measurement"]["camera_rate_hz"]
        frame_dt = 1.0 / rate
        patches = inputs["patches"]
        start_particles = readings.particles()
        record_states = args.kind == "full"
        factor = int(config["measurement"]["image_downsample"])

        def record(now, target, s_cmd, finger, phase, particles):
            row = {
                "time": now,
                "phase": controller.PHASES.index(phase),
                "particles": particles,
                "target_q": target,
                "path_s_cmd": s_cmd,
                "finger_target": finger,
                "qpos": readings.qpos(),
                "tcp": readings.tcp(),
                "finger_poses": readings.finger_poses(),
                "width": block_width(particles, patches),
            }
            for key, value in row.items():
                columns.setdefault(key, []).append(np.array(value, copy=True))

        def capture(frame, now, keyframe):
            depth = args.images == "full" or keyframe
            for name, data in cameras.capture(depth).items():
                if name in videos and frame % stride == 0:
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
            record(
                0.0,
                initial_row,
                controller.s,
                controller.finger,
                controller.phase,
                start_particles,
            )
            if cameras:
                capture(0, 0.0, True)
        sim_seconds = 0.0
        width = block_width(start_particles, patches)
        frame = 0
        last_frames = controller.max_frames()
        phase = controller.phase
        while frame < last_frames and not controller.finished:
            frame += 1
            now = frame * frame_dt
            previous = phase
            s_cmd, finger, phase = controller.step(now, width)
            target = np.r_[arm_target(inputs, s_cmd), finger, finger]
            control.set_target(target.astype(np.float32))
            wp.synchronize_device(args.device)
            started = time.perf_counter()
            sim.update(step=1, render_final_step=False)
            wp.synchronize_device(args.device)
            step = time.perf_counter() - started
            sim_seconds += step
            readback_started = time.perf_counter()
            capture_seconds = 0.0
            particles = readings.particles()
            if record_states:
                started = time.perf_counter()
                record(now, target, s_cmd, finger, phase, particles)
                last = frame == last_frames or controller.finished
                if cameras and frame % stride == 0:
                    capture(frame, now, phase != previous or last)
                elif cameras and (phase != previous or last):
                    capture(frame, now, True)
                capture_seconds = time.perf_counter() - started
            # The closed-loop rule needs the measured width every frame.
            width = block_width(particles, patches)
            timing_rows.append(
                (
                    frame,
                    now,
                    step,
                    capture_seconds,
                    phase,
                    time.perf_counter() - readback_started,
                )
            )
            if frame == 2:
                manifest["cuda_graph_status_after_first_steps"] = (
                    sim.spawn_result.get_solver_diagnostics().get("cuda_graph_status")
                )
            if frame % 30 == 0:
                memory.sample(f"frame_{frame}")
                comp = controller.compression()
                comp_text = "   n/a" if comp is None else f"{100 * comp:5.1f}%"
                tcp = readings.tcp()
                log(
                    f"t={now:5.2f} {phase:14s} w={width * 1e3:6.2f}mm "
                    f"comp={comp_text} s={s_cmd * 1e3:6.1f}mm "
                    f"tcp=({tcp[0]:.3f},{tcp[1]:.3f},{tcp[2]:.3f}) "
                    f"zmin={particles[:, 2].min():.4f} "
                    f"finger={readings.qpos()[7] * 1e3:.2f}mm "
                    f"step={step * 1e3:.2f}ms"
                )
                if not np.isfinite(particles).all():
                    raise FloatingPointError("Non-finite particle state")
        manifest["newton_shapes"] = shape_table(model)
        diagnostics = sim.spawn_result.get_solver_diagnostics()
        manifest["solver_diagnostics"] = diagnostics
        manifest["effective"] = {
            "num_substeps": diagnostics.get("num_substeps"),
            "cuda_graph_status": diagnostics.get("cuda_graph_status"),
            "dt": diagnostics.get("dt"),
        }
        manifest["controller"] = {
            "w0": controller.w0,
            "squeeze_stop": controller.squeeze_stop,
            "finger_closed": controller.finger_closed,
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
                initial_particles=inputs["initial"],
                width_patches=patches,
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
            stream.write(
                "frame,sim_time,step_seconds,capture_seconds,phase,readback_record_capture_seconds\n"
            )
            for row in timing_rows:
                stream.write(",".join(str(v) for v in row) + "\n")
        write_json(out / "config.json", config)
        write_json(out / "manifest.json", manifest)
        (out / "run.log").write_text("\n".join(log_lines) + "\n")
        print(f"run directory: {out}", flush=True)
        if sim is not None:
            sim.destroy(exit_process=False)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--state", default="s00")
    parser.add_argument("--variant", default="nominal")
    parser.add_argument("--kind", choices=("full", "timing"), default="full")
    parser.add_argument(
        "--images", choices=("full", "keyframes", "none"), default="full"
    )
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--inputs", help="alternative inputs directory")
    parser.add_argument(
        "--set", action="append", help="debug override a.b=<json>, recorded in config"
    )
    parser.add_argument("--out-root", help="write raw_data/ under this root")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
