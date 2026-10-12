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

"""Measure bilateral PGI contact and sampled pick/orient/assemble clearances."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.transform import Rotation, Slerp

from scripts.tools.assemble._collision import ExactCollision
from scripts.tools.assemble._geometry import load_mesh
from scripts.tools.assemble._json_io import pose_matrix
from scripts.tools.assemble._validation import PlacementValidator

__all__ = ["GraspGeometry", "interpolate_poses", "object_waypoints"]

TCP_Z = 0.16


def interpolate_poses(
    start: np.ndarray,
    end: np.ndarray,
    step: float = 0.01,
    rotation_step_degrees: float = 5.0,
    rotation_direction: str = "shortest",
) -> list[np.ndarray]:
    """Sample SE(3) with bounded translation and either rotation sense.

    Args:
        start: Start pose.
        end: End pose.
        step: Maximum translation increment.
        rotation_step_degrees: Maximum angular increment.
        rotation_direction: Shortest rotation or the equivalent opposite sense.

    Returns:
        Samples including both endpoints; these are discrete collision checks.
    """
    if rotation_direction not in ("shortest", "opposite"):
        raise ValueError("rotation_direction must be shortest or opposite")
    vector = Rotation.from_matrix(start[:3, :3].T @ end[:3, :3]).as_rotvec()
    angle = np.linalg.norm(vector)
    if rotation_direction == "opposite" and angle > 1e-5:
        vector *= (angle - 2 * np.pi) / angle
        angle = np.linalg.norm(vector)
    count = max(
        1,
        int(np.ceil(np.linalg.norm(end[:3, 3] - start[:3, 3]) / step)),
        int(np.ceil(angle / np.deg2rad(rotation_step_degrees))),
    )
    fractions = np.linspace(0, 1, count + 1)
    if rotation_direction == "shortest":
        rotations = Slerp(
            [0, 1], Rotation.from_matrix(np.stack([start[:3, :3], end[:3, :3]]))
        )(fractions).as_matrix()
    else:
        rotations = (
            start[:3, :3]
            @ Rotation.from_rotvec(fractions[:, None] * vector).as_matrix()
        )
    poses = np.repeat(np.eye(4)[None], len(fractions), axis=0)
    poses[:, :3, :3] = rotations
    poses[:, :3, 3] = start[:3, 3] + fractions[:, None] * (end[:3, 3] - start[:3, 3])
    poses[0], poses[-1] = start, end
    return list(poses)


def object_waypoints(
    initial: np.ndarray,
    target: np.ndarray,
    base_world: trimesh.Trimesh,
    assemble_mesh: trimesh.Trimesh,
    clearance: float,
    insertion_direction: np.ndarray | None = None,
    insertion_distance: float | None = None,
) -> list[np.ndarray]:
    """Lift, orient, transfer, then approach the assembly pose along a direction.

    Args:
        initial: Initial object world pose, with any valid orientation.
        target: Assembled object world pose.
        base_world: Base mesh already transformed into world coordinates.
        assemble_mesh: Assemble mesh in its local frame.
        clearance: Additional free height above the base.
        insertion_direction: World-frame direction of travel toward the target.
            Omit to reproduce the legacy four-waypoint vertical route.
        insertion_distance: Distance from the pre-insertion pose to the target.

    Returns:
        World-frame object keyframes with the exact assembly pose last.
    """
    preinsert = target.copy()
    if insertion_direction is not None:
        direction = np.asarray(insertion_direction, dtype=float)
        if (
            direction.shape != (3,)
            or not np.isfinite(direction).all()
            or np.linalg.norm(direction) < 1e-8
            or insertion_distance is None
            or not np.isfinite(insertion_distance)
            or insertion_distance <= 0
        ):
            raise ValueError(
                "Insertion requires a finite direction and positive distance"
            )
        preinsert[:3, 3] -= direction / np.linalg.norm(direction) * insertion_distance
    height = (
        max(base_world.bounds[1, 2], initial[2, 3])
        + np.linalg.norm(assemble_mesh.vertices, axis=1).max()
        + clearance
    )
    if insertion_direction is not None:
        height = max(height, preinsert[2, 3])
    lifted, oriented, transit = initial.copy(), target.copy(), preinsert.copy()
    lifted[2, 3] = height
    oriented[:3, 3] = lifted[:3, 3]
    transit[2, 3] = height
    if insertion_direction is None:
        return [lifted, oriented, transit, target.copy()]
    return [lifted, oriented, transit, preinsert, target.copy()]


def _translation(vector: np.ndarray) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, 3] = vector
    return pose


def _pgi_meshes() -> list[trimesh.Trimesh]:
    from embodichain.data import get_data_path

    path = Path(get_data_path("DH_PGI_140_80/DH_PGI_140_80.urdf"))
    root = ET.parse(path).getroot()
    meshes = []
    for name in (
        "gripper_base_link_1",
        "gripper_finger1_link_1",
        "gripper_finger2_link_1",
    ):
        link = root.find(f"./link[@name='{name}']/collision")
        filename = link.find("geometry/mesh").attrib["filename"]
        mesh = trimesh.load(path.parent / filename, force="mesh")
        # The supplied collision OBJ is a collection of convex components.
        # Union their closed hulls to make containment queries well defined.
        mesh = trimesh.boolean.union(
            [part.convex_hull for part in mesh.split(only_watertight=False)],
            engine="manifold",
        )
        offset = np.zeros(3)
        for joint in root.findall("joint"):
            if joint.find("child").attrib["link"] == name:
                offset = np.fromstring(joint.find("origin").attrib["xyz"], sep=" ")
        offset[2] -= TCP_Z
        mesh.apply_translation(offset)
        meshes.append(mesh)
    return meshes


class GraspGeometry:
    """Validate one calibrated UR5/PGI grasp against both actual object meshes."""

    def __init__(self, config: dict, assembly: dict, cache: Path) -> None:
        self.config, self.assembly = config, assembly
        self.assemble_mesh = load_mesh(Path(assembly["assets"]["assemble"]["path"]))
        self.base_mesh = load_mesh(Path(assembly["assets"]["base"]["path"]))
        settings = assembly["resolved_validation"] | {"collision": "exact"}
        check = PlacementValidator(
            {"assets": assembly["assets"]}, settings, cache
        ).validate(assembly["T_base_assemble"])
        if not check["accepted"]:
            raise ValueError(
                f"Assembly no longer passes original-mesh validation: {check['reasons']}"
            )
        self.parts = _pgi_meshes()
        self.assemble_checks = [
            ExactCollision(self.assemble_mesh, part) for part in self.parts
        ]
        self.set_layout(config)

    def set_layout(self, config: dict) -> None:
        """Rebuild world collision geometry for a proposed initial layout.

        Args:
            config: Resolved world poses. Object-local meshes and grasps stay fixed.
        """
        self.config = config
        self.initial = pose_matrix(config["T_world_assemble_initial"])
        self.target = pose_matrix(config["T_world_base"]) @ pose_matrix(
            self.assembly["T_base_assemble"]
        )
        self.base_world = self.base_mesh.copy()
        self.base_world.apply_transform(pose_matrix(config["T_world_base"]))
        self.base_checks = [
            ExactCollision(self.base_world, part) for part in self.parts
        ]
        self.object_check = ExactCollision(self.base_world, self.assemble_mesh)
        plan = config.get("task_plan")
        base_rotation = pose_matrix(config["T_world_base"])[:3, :3]
        self.waypoints = object_waypoints(
            self.initial,
            self.target,
            self.base_world,
            self.assemble_mesh,
            config["motion"]["clearance"],
            insertion_direction=(
                base_rotation @ np.asarray(plan["insertion_direction_base"])
                if plan
                else None
            ),
            insertion_distance=plan["insertion_distance"] if plan else None,
        )
        self.waypoint_names = (
            ("grasp", "lift", "rotate", "transit", "hover", "place")
            if plan
            else ("grasp", "lift", "rotate", "hover", "place")
        )

    def _object_clearance(self, pose: np.ndarray) -> str | None:
        """Check the complete carried mesh against the ground and base."""
        if (self.assemble_mesh.vertices @ pose[2, :3] + pose[2, 3]).min() < -1e-5:
            return "Assemble object intersects the ground"
        if self._overlap(self.object_check, pose):
            return "Assemble object intersects the base"
        return None

    def _initial_support(self) -> dict:
        """Estimate resting stability from a uniform solid's projected COM.

        Vertices within 0.5 mm of the lowest world point approximate contacts
        on a horizontal plane. The projected center of mass must lie inside
        their convex hull with at least five degrees of tipping reserve.
        This geometry heuristic does not establish dynamic stability or model
        an unknown density distribution, compliant contacts, or friction.
        """
        vertices = (
            self.assemble_mesh.vertices @ self.initial[:3, :3].T + self.initial[:3, 3]
        )
        center = (
            self.initial[:3, :3] @ self.assemble_mesh.center_mass + self.initial[:3, 3]
        )
        minimum_z = float(vertices[:, 2].min())
        contact_band = 0.0005
        contacts = np.unique(
            vertices[vertices[:, 2] <= minimum_z + contact_band, :2], axis=0
        )
        height = float(center[2] - minimum_z)
        report = {
            "accepted": False,
            "contact_band_m": contact_band,
            "minimum_world_z_m": minimum_z,
            "center_of_mass_world": center.tolist(),
            "center_of_mass_height_m": height,
            "contact_points": len(contacts),
            "support_area_m2": 0.0,
            "minimum_edge_margin_m": None,
            "minimum_tipping_angle_degrees": None,
            "required_tipping_angle_degrees": 5.0,
            "model": "Uniform solid density; planar contact-band support hull; geometric stability heuristic only",
        }
        if not np.isfinite(center).all() or height <= 0:
            report["reason"] = "Initial support has an invalid center of mass"
        elif len(contacts) < 3:
            report["reason"] = "Initial support consists of only a point or line"
        else:
            try:
                hull = ConvexHull(contacts)
            except QhullError:
                report["reason"] = "Initial support points form no planar area"
            else:
                equations = hull.equations
                margins = -(equations[:, :2] @ center[:2] + equations[:, 2])
                margins /= np.linalg.norm(equations[:, :2], axis=1)
                margin = float(margins.min())
                angle = float(np.rad2deg(np.arctan2(margin, height)))
                report.update(
                    support_area_m2=float(hull.volume),
                    minimum_edge_margin_m=margin,
                    minimum_tipping_angle_degrees=angle,
                )
                report["accepted"] = margin > 1e-8 and angle >= 5.0
                if not report["accepted"]:
                    report["reason"] = (
                        "Initial support is too easy to tip: "
                        f"COM edge margin {margin:.6f} m, "
                        f"tipping reserve {angle:.2f} degrees (requires >= 5 degrees)"
                    )
        if not report["accepted"]:
            report[
                "reason"
            ] += "; place the assemble object on a broad stable face before grasping"
        return report

    def _retract_pose(self, place: np.ndarray) -> np.ndarray:
        """Resolve the generated withdrawal direction in the current base frame."""
        plan = self.config.get("task_plan")
        retract = place.copy()
        if plan:
            direction = np.asarray(plan["retract_direction_base"], dtype=float)
            direction = pose_matrix(self.config["T_world_base"])[:3, :3] @ direction
            retract[:3, 3] += direction * plan["retract_distance"]
        else:
            retract[2, 3] += self.config["motion"]["retract_distance"]
        return retract

    @staticmethod
    def _offset(index: int, q: float) -> np.ndarray:
        return _translation(np.array([(-q if index == 1 else q) if index else 0, 0, 0]))

    @staticmethod
    def _overlap(check: ExactCollision, pose: np.ndarray) -> bool:
        return check.surface_collision(pose) or check.intersection_volume(pose) > 1e-12

    def _assemble_collision(self, index: int, pose: np.ndarray, q: float) -> bool:
        return self._overlap(self.assemble_checks[index], pose @ self._offset(index, q))

    def _world_clearance(self, tcp: np.ndarray, q: float) -> str | None:
        for i, part in enumerate(self.parts):
            pose = tcp @ self._offset(i, q)
            if (part.vertices @ pose[:3, :3].T + pose[:3, 3])[:, 2].min() < -1e-5:
                return f"gripper link {i} intersects the ground"
            if self._overlap(self.base_checks[i], pose):
                return f"gripper link {i} intersects the base object"
        return None

    def evaluate(self, value: object) -> dict:
        """Measure contacts, opening/release and sampled carried-object paths.

        Args:
            value: Proposed T_assemble_tcp, in the assemble mesh's original local frame.

        Returns:
            Candidate poses, measured hand closure and acceptance observations.
        """
        pose = pose_matrix(value)
        result = {
            "T_assemble_tcp": pose.tolist(),
            "accepted": False,
            "reasons": [],
            "initial_support": self._initial_support(),
        }
        initial_reason = self._object_clearance(self.initial)
        if initial_reason:
            result["reasons"].append(f"Initial placement: {initial_reason}")
            return result
        if not result["initial_support"]["accepted"]:
            result["reasons"].append(result["initial_support"]["reason"])
            return result
        for i in range(3):
            if self._assemble_collision(i, pose, 0):
                result["reasons"].append(
                    f"Open gripper link {i} intersects the assemble object"
                )
        if result["reasons"]:
            return result
        contacts = []
        for i in (1, 2):
            if not self._assemble_collision(i, pose, 0.04):
                result["reasons"].append(
                    f"Finger {i} cannot reach the assemble object while closing"
                )
                continue
            low, high = 0.0, 0.04
            for _ in range(18):
                mid = (low + high) / 2
                if self._assemble_collision(i, pose, mid):
                    high = mid
                else:
                    low = mid
            contacts.append(low)
        if len(contacts) != 2:
            return result
        q = max(0.0, min(contacts) - 0.00005)
        distances = [
            self.assemble_checks[i].distance(pose @ self._offset(i, q)) for i in (1, 2)
        ]
        result.update(finger_contact_qpos=contacts, finger_gap_m=distances)
        if max(distances) > 0.001:
            result["reasons"].append(
                "Grasp is off-center: both fingers must be within 1 mm of contact"
            )
            return result
        pick, place = self.initial @ pose, self.target @ pose
        pre = pick.copy()
        pre[:3, 3] -= pick[:3, 2] * self.config["motion"]["pre_grasp_distance"]
        for tcp in interpolate_poses(pre, pick):
            local = np.linalg.inv(self.initial) @ tcp
            if any(self._assemble_collision(i, local, 0) for i in range(3)):
                result["reasons"].append(
                    "Open-gripper approach intersects the assemble object"
                )
                return result
            reason = self._world_clearance(tcp, 0)
            if reason:
                result["reasons"].append(reason)
                return result
        for width in np.linspace(q, 0, 10):
            reason = self._world_clearance(place, float(width))
            if reason:
                result["reasons"].append(f"Release: {reason}")
                return result
        route = [self.initial, *self.waypoints]
        checked = 0
        for phase, (start, end) in enumerate(zip(route, route[1:])):
            # A shortest-sense obstruction must not prevent the robot planner
            # from trying the other rotation. Both directions share exact endpoints.
            directions = ("shortest", "opposite") if phase == 1 else ("shortest",)
            failures = []
            for direction in directions:
                failure = None
                for object_pose in interpolate_poses(
                    start, end, rotation_direction=direction
                ):
                    checked += 1
                    reason = self._world_clearance(object_pose @ pose, q)
                    reason = reason or self._object_clearance(object_pose)
                    if reason:
                        failure = reason
                        break
                if failure is None:
                    break
                failures.append(failure)
            else:
                result["reasons"].append(
                    "Rotation blocked in both directions: " + "; ".join(failures)
                    if phase == 1
                    else failures[0]
                )
                return result
        retract = self._retract_pose(place)
        for tcp in interpolate_poses(place, retract):
            local = np.linalg.inv(self.target) @ tcp
            reason = self._world_clearance(tcp, 0)
            if reason or any(self._assemble_collision(i, local, 0) for i in range(3)):
                result["reasons"].append(
                    reason
                    or "Open-gripper withdrawal intersects the placed assemble object"
                )
                return result
        result.update(
            accepted=True,
            gripper_clear_qpos=q,
            gripper_close_qpos=min(0.04, min(contacts) + 0.003),
            T_world_tcp_pick=pick.tolist(),
            T_world_tcp_place=place.tolist(),
            T_world_tcp_retract=retract.tolist(),
            T_world_assemble_target=self.target.tolist(),
            waypoints={
                name: {
                    "T_world_assemble": item.tolist(),
                    "T_world_tcp": (item @ pose).tolist(),
                }
                for name, item in zip(
                    self.waypoint_names,
                    [self.initial, *self.waypoints],
                )
            },
            lift_height=float(self.waypoints[0][2, 3] - self.initial[2, 3]),
            geometry_samples=checked,
            checks="Bilateral finger proximity, calibrated gripper meshes, sampled approach/transfer/release; not a force-closure or dynamics proof",
        )
        return result
