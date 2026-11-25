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

"""Independent parallel-jaw grasp trials on batched physics arenas."""

from __future__ import annotations

from dataclasses import replace
from numbers import Real
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import xml.etree.ElementTree as ET

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation
import torch

from embodichain.robot_registry import (
    _validate_gripper_command,
    _validate_gripper_stiffness,
    _validate_gripper_transform,
    resolve_robot_cfg,
)
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import (
    DefaultPhysicsCfg,
    MimicComplianceCfg,
    NewtonPhysicsCfg,
    RenderCfg,
    RigidObjectCfg,
)
from embodichain.lab.sim.sensors import ContactSensorCfg, ArticulationContactFilterCfg
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.lab.sim.utility.gripper_geometry import _native_joint_limits

__all__ = ["GraspValidator"]

DT = 0.01
_APPROACH_STEPS = 60
_CLOSE_STEPS = 40
_LIFT_STEPS = 80
_HOLD_STEPS = 300
_LEG_STEPS = 30

# Device trace layout, one sample per completed physics step.
_COM = slice(0, 3)
_ROTATION = slice(3, 12)
_CLEARANCE = 12
_PADS = slice(13, 15)
_SUPPORT = 15
_ACCEL = 16


def _stage(hand, directory, physics, close_qpos=None):
    """Mount the native hand on three axes and hard-couple its PhysX mimics."""
    source = Path(hand.fpath or hand.urdf_cfg.assemble_urdf())
    xml = ET.parse(source)
    root = xml.getroot()
    if close_qpos is not None:
        names = hand.control_parts["eef"]
        command = _validate_gripper_command(close_qpos, names)
        limits = _native_joint_limits(hand.qpos_limits, names)
        joints = {joint.attrib["name"]: joint for joint in root.findall("joint")}
        for name, value in command.items():
            joint = joints.get(name)
            if joint is None or joint.get("type") == "fixed":
                raise ValueError(
                    f"close_qpos joint {name!r} must be movable in the hand URDF"
                )
            bounds = limits.get(name)
            if bounds is None and joint.get("type") != "continuous":
                limit = joint.find("limit")
                if limit is None:
                    raise ValueError(
                        f"Missing joint limits for close_qpos joint {name!r}"
                    )
                bounds = (float(limit.attrib["lower"]), float(limit.attrib["upper"]))
            if bounds is not None and not bounds[0] <= value <= bounds[1]:
                raise ValueError(
                    f"close_qpos for {name!r} exceeds joint limits {bounds}"
                )
            mimic = joint.find("mimic")
            if mimic is not None:
                master = mimic.attrib["joint"]
                if master not in command or not np.isclose(
                    value,
                    command[master] * float(mimic.get("multiplier", "1"))
                    + float(mimic.get("offset", "0")),
                    rtol=0,
                    atol=1e-8,
                ):
                    raise ValueError(
                        f"close_qpos for {name!r} contradicts its mimic of {master!r}"
                    )
        hand.joint_commands["grasp"] = command
    if physics == "default":
        masters = {mimic.attrib["joint"] for mimic in root.findall("joint/mimic")}
        hand.mimic_compliance = {
            name: MimicComplianceCfg(natural_frequency=0.0, damping_ratio=0.0)
            for name in sorted(masters)
        }
    children = {joint.find("child").get("link") for joint in root.findall("joint")}
    (mount,) = [
        link.get("name")
        for link in root.findall("link")
        if link.get("name") not in children
    ]
    (Path(directory) / "hand_assets").symlink_to(
        source.parent, target_is_directory=True
    )
    for mesh in root.iter("mesh"):
        mesh.set("filename", str(Path("hand_assets") / mesh.get("filename")))
    parent = "stage_base"
    ET.SubElement(root, "link", name=parent)
    for i, axis in enumerate(("1 0 0", "0 1 0", "0 0 1")):
        child = mount if i == 2 else f"stage_link{i}"
        if child != mount:
            link = ET.SubElement(root, "link", name=child)
        else:
            link = root.find(f"link[@name='{mount}']")
        if link.find("inertial") is None:
            inertia = ET.SubElement(link, "inertial")
            ET.SubElement(inertia, "mass", value="0.1")
            ET.SubElement(
                inertia,
                "inertia",
                ixx=".001",
                iyy=".001",
                izz=".001",
                ixy="0",
                ixz="0",
                iyz="0",
            )
        joint = ET.SubElement(root, "joint", name=f"stage_{i}", type="prismatic")
        for tag, attrs in (
            ("parent", {"link": parent}),
            ("child", {"link": child}),
            ("axis", {"xyz": axis}),
            ("limit", {"lower": "-1", "upper": "1", "velocity": "2", "effort": "2000"}),
        ):
            ET.SubElement(joint, tag, attrs)
        parent = child
    path = Path(directory) / "stage.urdf"
    xml.write(path)
    hand.urdf_cfg = None
    hand.fpath = str(path)
    hand.init_qpos = None
    for name, value in (
        ("stiffness", 30000.0),
        ("damping", 350.0),
        ("max_effort", 2000.0),
    ):
        old = getattr(hand.joint_drive_props, name)
        setattr(
            hand.joint_drive_props,
            name,
            {**{j: old for j in hand.joint_commands["open"]}, "stage_.*": value},
        )


def _rotation_shift_matrix(rotation_shift_deg: ArrayLike | None) -> np.ndarray | None:
    """Convert an optional arena-frame XYZ offset in degrees to a rotation."""
    if rotation_shift_deg is None:
        return None
    if isinstance(rotation_shift_deg, torch.Tensor):
        rotation_shift_deg = rotation_shift_deg.tolist()
    angles = np.asarray(rotation_shift_deg, dtype=object)
    if angles.shape != (3,) or any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
        for value in angles
    ):
        raise ValueError("rotation_shift_deg requires three finite angles in degrees")
    angles = angles.astype(float)
    if not np.isfinite(angles).all():
        raise ValueError("rotation_shift_deg requires three finite angles in degrees")
    if not angles.any():
        return None
    return Rotation.from_euler("xyz", angles, degrees=True).as_matrix()


def _smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x**3 * (10 + x * (6 * x - 15))


class GraspValidator:
    """Own a prepared hand, support and dynamic target for independent trials.

    Args:
        object_cfg: Dynamic target shared by the replicated arenas.
        gripper: Catalog ID of a calibrated parallel-jaw hand.
        num_envs: Number of independent trial arenas.
        variant: Catalog variant for hands with multiple models, such as the
            left or right Tianji Marvin hand.
        gripper_urdf: Optional standalone URDF to mount using this hand's
            catalog calibration and joint controls.
        physics: GPU backend, either ``default`` or ``newton``. Default (PhysX)
            hard-couples every URDF mimic joint. Newton runs DexUni with
            two-way coupling between the hand and target.
        headless: Set to False to open a native window when the first grasp
            starts and watch arena 0 in real time.
        root_from_tcp: Optional 4×4 TCP pose in the native hand root, in metres.
            Candidate transforms refer to this TCP.
        stiffness: Optional position gain for the native gripper drives. The
            translation stage has its own gains and mimic joints follow the
            backend's coupling rules.
        close_qpos: Optional closed target by native joint name, including mimic
            followers. Positions use metres for prismatic joints and radians for
            revolute joints.
        rotation_shift_deg: Optional object rotation about the fixed arena X, Y
            and Z axes, in that order, in degrees. A rotated target starts with
            its lowest collision point on the support, then settles.
        close_wait_steps: Extra physics steps to hold the fully closed command
            at the grasp pose after the 40-step closing ramp and before lifting.
    """

    def __init__(
        self,
        object_cfg: RigidObjectCfg,
        gripper: str,
        num_envs: int = 1,
        *,
        variant: str | None = None,
        gripper_urdf: str | Path | None = None,
        physics: str = "default",
        headless: bool = True,
        root_from_tcp: ArrayLike | None = None,
        stiffness: float | None = None,
        close_qpos: dict[str, float] | None = None,
        close_wait_steps: int = 0,
        rotation_shift_deg: ArrayLike | None = None,
    ) -> None:
        if type(close_wait_steps) is not int or close_wait_steps < 0:
            raise ValueError("close_wait_steps must be a nonnegative integer")
        self.close_wait_steps = close_wait_steps
        self._show_trials = not headless
        rotation = _rotation_shift_matrix(rotation_shift_deg)
        if object_cfg.env_variants:
            raise ValueError("Each validator batch requires a single target geometry")
        if physics == "default":
            physics_cfg = DefaultPhysicsCfg(
                device="cuda:0", physics_dt=DT, num_substeps=4
            )
        elif physics == "newton":
            physics_cfg = NewtonPhysicsCfg(
                physics_dt=DT,
                num_substeps=4,
                solver_cfg={"solver_type": "dexuni", "coupling": "two_way"},
            )
        else:
            raise ValueError("physics must be default or newton")
        overrides = {}
        if gripper_urdf is not None:
            overrides["fpath"] = str(Path(gripper_urdf).resolve(strict=True))
        if root_from_tcp is not None:
            overrides["default_root_from_tcp"] = [
                list(row)
                for row in _validate_gripper_transform(
                    np.asarray(root_from_tcp).tolist()
                )
            ]
        if stiffness is not None:
            stiffness = _validate_gripper_stiffness(stiffness)
        self.temp = TemporaryDirectory()
        self.sim = SimulationManager(
            SimulationManagerCfg(
                num_envs=num_envs,
                headless=True,
                width=1280,
                height=720,
                enable_ground_plane=False,
                enable_entity_gizmo=False,
                robot_ik_gizmo=None,
                render_cfg=RenderCfg(renderer="no-render" if headless else "fast-rt"),
                physics_cfg=physics_cfg,
            )
        )
        try:
            cfg = resolve_robot_cfg(gripper, overrides, variant=variant)
            if stiffness is not None:
                cfg.joint_drive_props.stiffness = stiffness
            _stage(cfg, self.temp.name, physics, close_qpos)
            cfg.init_pos = [0, 0, 1.5]
            self.hand = self.sim.add_robot(cfg)
            self.sim.add_rigid_object(
                RigidObjectCfg(
                    uid="support",
                    shape=CubeCfg(size=(4.0, 4.0, 0.1)),
                    init_pos=(0.0, 0.0, -0.05),
                    # Direct GPU exposes an actor ID for a stationary kinematic support.
                    body_type="kinematic" if physics == "default" else "static",
                )
            )
            object_cfg = object_cfg.copy()
            object_cfg.uid = "target"
            self.obj = self.sim.add_rigid_object(object_cfg)
            self.sensor = self.sim.add_sensor(
                ContactSensorCfg(
                    uid="contacts",
                    rigid_uid_list=["support", "target"],
                    articulation_cfg_list=[
                        ArticulationContactFilterCfg(
                            articulation_uid=cfg.uid, link_name_list=list(cfg.pad_links)
                        )
                    ],
                    max_contacts_per_env=256,
                )
            )
            self.sim.prepare()
            if self.obj.is_non_dynamic:
                raise ValueError("Grasp target must be dynamic")
            self.shapes = [
                replace(
                    shape,
                    local_pose=self._tensor(shape.local_pose),
                    half_extents=(
                        None
                        if shape.half_extents is None
                        else self._tensor(shape.half_extents)
                    ),
                    vertices=(
                        None if shape.vertices is None else self._tensor(shape.vertices)
                    ),
                )
                for shape in self.obj.get_collision_shapes(env_id=0)
            ]
            if not self.shapes:
                raise ValueError("Target has no collision geometry")
            if rotation is not None:
                self._place_rotated_target(rotation)
            self.initial = self._settled_pose()
            self.com = self.obj.body_data.com_pose[0, :3].clone()
            self.cfg = cfg
            self.root_tcp = self._tensor(cfg.default_root_from_tcp)
            self.root_contact = self._tensor(cfg.root_from_contact_frame)
            self.pad_ids = self.sensor.get_actor_ids(cfg.uid, list(cfg.pad_links))
            self.target_id = self.sensor.get_actor_ids("target")
            self.support_id = self.sensor.get_actor_ids("support")
            self.opened = self._tensor(
                [cfg.joint_commands["open"].get(j, 0.0) for j in self.hand.joint_names]
            )
            self.closed = self._tensor(
                [cfg.joint_commands["grasp"].get(j, 0.0) for j in self.hand.joint_names]
            )
            self.axes = [self.hand.joint_names.index(f"stage_{i}") for i in range(3)]
            self._geometry_contacts = physics == "newton"
            if (
                not self._geometry_contacts
                and not self.sensor.contact_capabilities.impulse
            ):
                raise ValueError("Grasp validation requires measured contact impulses")
            fingers, support, lost = self._contacts()
            if not bool((support & ~lost & ~fingers.any(1)).all()):
                raise ValueError(
                    "Each target must settle on the support without hand contact"
                )
        except BaseException:
            self.close()
            raise

    def _tensor(self, value: object) -> torch.Tensor:
        return torch.as_tensor(value, device=self.sim.device, dtype=torch.float32)

    def _place_rotated_target(self, rotation: np.ndarray) -> None:
        """Rotate each native root in arena axes and rest it on the support."""
        pose = self.obj.get_local_pose(to_matrix=True).clone()
        pose[:, :3, :3] = pose.new_tensor(rotation) @ pose[:, :3, :3]
        pose[:, 2, 3] -= self._clearance(pose)
        # Discard contacts from the source orientation before placing it on edge.
        separated = pose.clone()
        separated[:, 2, 3] += 0.5
        self.obj.set_local_pose(separated)
        self.sim.update(step=2)
        self.obj.set_local_pose(pose)
        zero = torch.zeros_like(pose[:, :3, 3])
        self.obj.set_velocity(zero, zero)

    def _settled_pose(self) -> torch.Tensor:
        """Wait up to five simulated seconds for every arena to remain still for 0.5 s."""
        self.sim.update(step=100)
        current = self.obj.get_local_pose(to_matrix=True)
        for _ in range(8):
            reference = current.clone()
            settled = torch.ones(
                self.sim.num_envs, dtype=torch.bool, device=self.sim.device
            )
            for _ in range(5):
                self.sim.update(step=10)
                current = self.obj.get_local_pose(to_matrix=True)
                displacement = torch.linalg.vector_norm(
                    current[:, :3, 3] - reference[:, :3, 3], dim=1
                )
                cosine = (
                    (current[:, :3, :3] * reference[:, :3, :3]).sum((1, 2)) - 1
                ) * 0.5
                settled &= (displacement <= 0.001) & (cosine >= np.cos(0.005))
            if settled.all().item():
                return current[0].clone()
        raise ValueError("Target did not settle within 5 simulated seconds")

    def close(self) -> None:
        """Release physics resources and the temporary mounting URDF."""
        self.sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()
        self.temp.cleanup()

    def _clearance(self, pose):
        heights = []
        for shape in self.shapes:
            transform = pose @ shape.local_pose
            z = transform[:, 2, :3]
            kind = shape.shape_type.name.upper()
            if kind == "BOX":
                depth = (z.abs() * shape.half_extents).sum(-1)
            elif kind == "SPHERE":
                depth = shape.radius
            elif kind == "CAPSULE":
                depth = shape.radius + shape.half_height * z[:, 2].abs()
            elif shape.vertices is not None and len(shape.vertices):
                depth = -(z @ shape.vertices.T).amin(-1)
            else:
                raise ValueError(f"Unsupported physical clearance geometry {kind}")
            heights.append(transform[:, 2, 3] - depth)
        return torch.stack(heights).amin(0)

    def _contacts(self):
        self.sensor.update()
        data = self.sensor.get_data()
        a, b = data["user_ids"].unbind(-1)
        paired = torch.where(
            a == self.target_id, b, torch.where(b == self.target_id, a, -1)
        )
        impulse = data["impulse"]
        measured = data["impulse_valid"]
        evidence = measured & torch.isfinite(impulse)
        touching = evidence & (impulse > 0)
        if self._geometry_contacts:
            # DexUni can publish geometric VBD rows alongside measured impulses.
            distance = data["distance"]
            geometry = ~measured & torch.isfinite(distance)
            evidence |= geometry
            touching |= geometry & (distance <= 0)
        valid = data["is_valid"] & touching
        fingers = (
            (paired[:, :, None] == self.pad_ids[:, None, :]) & valid[:, :, None]
        ).any(1)
        supported = ((paired == self.support_id) & valid).any(1)
        lost = self.sensor.get_latest_contact_loss_mask()
        lost = lost | (data["is_valid"] & ~evidence).any(1)
        return fingers, supported, lost

    def validate(self, poses: np.ndarray, directions: list[str]) -> list[dict]:
        """Run one full arena batch of object-local TCP poses and named conditions."""
        poses = np.asarray(poses)
        n = self.sim.num_envs
        if (
            poses.shape != (n, 4, 4)
            or len(directions) != n
            or not set(directions) <= {"base", "x", "y"}
        ):
            raise ValueError(
                "One object_from_tcp pose and valid condition per arena required"
            )
        rotations = poses[:, :3, :3]
        if not (
            np.isfinite(poses).all()
            and np.allclose(poses[:, 3], [0, 0, 0, 1])
            and np.allclose(
                rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-4
            )
            and np.allclose(np.linalg.det(rotations), 1, atol=1e-4)
        ):
            raise ValueError("Candidates must be proper rigid transforms")
        goal = self.initial @ self._tensor(poses) @ torch.linalg.inv(self.root_tcp)
        axis = (goal @ self.root_contact)[:, :3, 2]
        base = goal.clone()
        base[:, :3, 3] -= 0.08 * axis
        self.hand.set_local_pose(base)
        opened = self.opened.expand(n, -1).contiguous()
        self.hand.set_qpos(opened, target=False)
        self.hand.set_qvel(torch.zeros_like(opened), target=False)
        self.hand.set_qpos(opened)
        self.obj.set_local_pose(self.initial.expand(n, -1, -1).contiguous())
        zero = torch.zeros_like(axis)
        self.obj.set_velocity(zero, zero)
        self.sensor.reset()
        if self._show_trials and self.sim.viewer is None:
            if not self.sim.open_viewer():
                raise RuntimeError("Could not open the native grasp benchmark window")
            target = (self.initial[:3, 3] + self.sim.arena_offsets[0]).cpu().numpy()
            target = target + [0, 0, 0.05]  # Frame both the pickup and the lift.
            eye = target + [0.36, -0.42, 0.25]
            self.sim.viewer.set_look_at(eye.tolist(), target.tolist(), [0, 0, 1])
        shift = self._tensor(
            [
                {"base": [0, 0, 0], "x": [0.03, 0, 0], "y": [0, 0.03, 0]}[d]
                for d in directions
            ]
        )
        up = self._tensor([0, 0, 0.1])
        history = []
        unavailable = torch.zeros(n, device=axis.device, dtype=torch.bool)
        lift_start = _APPROACH_STEPS + _CLOSE_STEPS + self.close_wait_steps
        lift_end = lift_start + _LIFT_STEPS
        hold_end = lift_end + _HOLD_STEPS
        end = hold_end + (2 * _LEG_STEPS if any(d != "base" for d in directions) else 0)
        deadline = time.monotonic() if self._show_trials else None
        previous = velocity = None
        for step in range(1, end + 1):
            if deadline is not None:
                if not self.sim.is_viewer_open:
                    raise KeyboardInterrupt("Grasp benchmark window closed")
                delay = deadline - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                deadline = time.monotonic() + DT
            position = 0.08 * axis * _smooth(step / _APPROACH_STEPS) + up * _smooth(
                (step - lift_start) / _LIFT_STEPS
            )
            position += shift * (
                _smooth((step - hold_end) / _LEG_STEPS)
                - _smooth((step - hold_end - _LEG_STEPS) / _LEG_STEPS)
            )
            q = (
                opened
                + (self.closed - self.opened)
                * _smooth((step - _APPROACH_STEPS) / _CLOSE_STEPS)
            ).clone()
            q[:, self.axes] = (base[:, :3, :3].mT @ position.unsqueeze(-1)).squeeze(-1)
            self.hand.set_qpos(q)
            self.sim.update(step=1)
            if step < lift_end - 1:
                continue
            tcp = (
                self.hand.get_link_pose(self.cfg.root_link, to_matrix=True)
                @ self.root_tcp
            )
            pose = self.obj.get_local_pose(to_matrix=True)
            rel = tcp[:, :3, :3].mT @ pose[:, :3, :3]
            center = (pose[:, :3, :3] @ self.com) + pose[:, :3, 3]
            p = (tcp[:, :3, :3].mT @ (center - tcp[:, :3, 3]).unsqueeze(-1)).squeeze(-1)
            v = zero if previous is None else (tcp[:, :3, 3] - previous) / DT
            acc = zero if velocity is None else (v - velocity) / DT
            previous, velocity = tcp[:, :3, 3].clone(), v
            if step < lift_end:
                continue
            fingers, support, lost = self._contacts()
            unavailable |= (
                lost if step <= hold_end else lost & (shift.abs().sum(-1) > 0)
            )
            history.append(
                torch.cat(
                    [
                        p,
                        rel.flatten(1),
                        self._clearance(pose)[:, None],
                        fingers,
                        support[:, None],
                        acc,
                    ],
                    dim=1,
                )
            )
        samples = torch.stack(history).transpose(0, 1).cpu().numpy()
        return [
            _result(row, d, bool(bad))
            for row, d, bad in zip(samples, directions, unavailable.cpu().tolist())
        ]


def _window(row):
    p, rot = row[:, _COM], row[:, _ROTATION].reshape(-1, 3, 3)
    cosine = ((rot * rot[0]).sum(axis=(1, 2)) - 1) / 2
    pads = row[:, _PADS].astype(bool)
    held = pads.any(1) & (row[:, _SUPPORT] == 0)
    entered = bool(held[0] and row[0, _CLEARANCE] >= 0.02)
    retained = (row[:, _CLEARANCE] >= 0.02) & held
    failed = np.flatnonzero(~held)
    end = int(failed[0]) + 1 if len(failed) else len(row)
    p, cosine = p[:end], cosine[:end]
    return dict(
        success=bool(retained.all() and pads[-1].all()),
        entered=entered,
        drop_s=float(failed[0] * DT) if entered and len(failed) else None,
        censored=bool(entered and not len(failed)),
        duration_s=(len(row) - 1) * DT,
        slip_mm=float(np.linalg.norm(p - p[0], axis=1).max() * 1000),
        rotation_deg=float(np.rad2deg(np.arccos(np.clip(cosine, -1, 1))).max()),
        bilateral_fraction=float(pads.all(1).mean()),
    )


def _result(row, direction, bad):
    if direction == "base":
        row = row[: _HOLD_STEPS + 1]
    if bad or not np.isfinite(row).all():
        return {"status": "missing", "reason": "incomplete physics observations"}
    hold = _window(row[: _HOLD_STEPS + 1])
    cutoffs = {
        str(s): _window(row[: round(s / DT) + 1])["success"] for s in (0.5, 1.0, 3.0)
    }
    shake = (
        _window(row[_HOLD_STEPS:]) if direction != "base" and hold["success"] else None
    )
    acceleration = row[_HOLD_STEPS + 1 :, _ACCEL:] if direction != "base" else None
    return dict(
        status="ok",
        success=hold["success"] and (shake is None or shake["success"]),
        failure_stage=(
            ("lift" if not hold["entered"] else "hold")
            if not hold["success"]
            else ("disturbance" if shake and not shake["success"] else None)
        ),
        hold=hold,
        retention=cutoffs,
        disturbance=shake,
        peak_acceleration_m_s2=(
            float(np.linalg.norm(acceleration, axis=1).max())
            if acceleration is not None
            else None
        ),
        peak_axis_acceleration_m_s2=(
            float(
                np.abs(
                    row[_HOLD_STEPS + 1 :, _ACCEL + ("x", "y").index(direction)]
                ).max()
            )
            if direction != "base"
            else None
        ),
    )
