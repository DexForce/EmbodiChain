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
"""CPU kinematics for the explicitly supported source-near refinement layout."""

from __future__ import annotations
import numpy as np
import pytorch_kinematics as pk
import torch
import xml.etree.ElementTree as ET
from .inputs import RobotModel

__all__: list[str] = []


class Robot:
    def __init__(self, model: RobotModel) -> None:
        self.path = model.urdf
        self.base_pose = np.array(model.base_pose_world, copy=True)
        self.chain = pk.build_chain_from_urdf(self.path.read_bytes()).to(
            dtype=torch.float64
        )
        self.names = list(model.joint_names)
        chain_names = self.chain.get_joint_parameter_names()
        document = ET.parse(self.path).getroot()
        joints = {j.get("name"): j for j in document.findall("joint")}
        self.calibrations = model.gripper_calibrations
        if model.arm_joint_groups:
            if set(self.names) != set(chain_names):
                raise ValueError("Source and URDF joints must match")
            if len(model.arm_joint_groups) != len(model.tcp_links) or len(
                self.calibrations
            ) != len(model.tcp_links):
                raise ValueError(
                    "Explicit arms require one gripper calibration per TCP"
                )
            self.arm_columns = [
                np.array([self.names.index(n) for n in group])
                for group in model.arm_joint_groups
            ]
            self.jaw_columns = [
                np.array([self.names.index(n) for n in item["joint_names"]])
                for item in self.calibrations
            ]
            for arm in self.arm_columns:
                if any(
                    joints[self.names[i]].get("type") not in {"revolute", "continuous"}
                    for i in arm
                ):
                    raise ValueError("Arm calibration contains a non-rotational joint")
        else:
            if self.names != chain_names:
                raise ValueError(
                    "The current method requires the source's URDF joint ordering"
                )
            if (
                len(self.names) not in (8, 16)
                or len(model.tcp_links) * 8 != len(self.names)
                or len(model.side_names) != len(model.tcp_links)
            ):
                raise ValueError(
                    "Expected one/two six-axis arms with paired prismatic jaws"
                )
            for i, name in enumerate(self.names):
                expected = {"revolute", "continuous"} if i % 8 < 6 else {"prismatic"}
                if joints[name].get("type") not in expected:
                    raise ValueError("Unsupported arm/jaw ordering: " + name)
            self.arm_columns = [
                np.arange(i * 8, i * 8 + 6) for i in range(len(model.tcp_links))
            ]
            self.jaw_columns = [
                np.array([i * 8 + 6, i * 8 + 7]) for i in range(len(model.tcp_links))
            ]
        self.joint_units = tuple(
            "m" if joints[name].get("type") == "prismatic" else "rad"
            for name in self.names
        )
        self.arm_limits: list[tuple[np.ndarray, np.ndarray]] = []
        for columns in self.arm_columns:
            bounds = []
            for column in columns:
                joint = joints[self.names[column]]
                if joint.get("type") == "continuous":
                    bounds.append((-np.inf, np.inf))
                    continue
                limit = joint.find("limit")
                if limit is None:
                    raise ValueError("Missing URDF arm limits: " + self.names[column])
                lower, upper = float(limit.get("lower")), float(limit.get("upper"))
                if not np.isfinite([lower, upper]).all() or lower >= upper:
                    raise ValueError("Invalid URDF arm limits: " + self.names[column])
                bounds.append((lower, upper))
            lower, upper = np.asarray(bounds, dtype=float).T
            self.arm_limits.append((lower.copy(), upper.copy()))
        self.offset = np.array(model.flange_to_tcp, dtype=float, copy=True)
        if (
            self.offset.shape != (4, 4)
            or not np.isfinite(self.offset).all()
            or not np.allclose(self.offset[3], [0, 0, 0, 1])
            or not np.allclose(
                self.offset[:3, :3].T @ self.offset[:3, :3], np.eye(3), atol=1e-5
            )
            or not np.isclose(np.linalg.det(self.offset[:3, :3]), 1)
        ):
            raise ValueError("Invalid flange_to_tcp calibration")
        self.prefixes = list(model.side_names)
        self.opening_limits: list[tuple[float, float]] = []
        for side, columns in enumerate(self.jaw_columns):
            limits = [joints[self.names[i]].find("limit") for i in columns]
            lower = np.array([float(item.get("lower")) for item in limits])
            upper = np.array([float(item.get("upper")) for item in limits])
            if self.calibrations:
                calibration = self.calibrations[side]
                grid = np.asarray(calibration["qpos"])
                if np.any(grid < lower - 1e-6) or np.any(grid > upper + 1e-6):
                    raise ValueError("Gripper calibration exceeds URDF joint limits")
                widths = np.asarray(calibration["width_m"]) / 2
                lo, hi = float(widths.min()), float(widths.max())
            else:
                lo, hi = float(lower.max()), float(upper.min())
            hi = min(hi, model.max_half_open_m)
            if not 0 <= lo < hi:
                raise ValueError("Gripper has no valid opening range")
            self.opening_limits.append((lo, hi))
        self.config = dict(
            tcp_links=list(model.tcp_links),
            side_names=list(model.side_names),
            pad_center_tcp_z_m=model.pad_center_tcp_z_m,
            robot_type="explicit",
        )

    def links(self, q: np.ndarray) -> dict[str, np.ndarray]:
        if not np.asarray(q).flags.writeable:
            q = np.array(q, copy=True)
        return {
            k: self.base_pose @ v.get_matrix().numpy()
            for k, v in self.chain.forward_kinematics(
                (
                    {
                        name: torch.as_tensor(
                            np.atleast_2d(q)[:, i], dtype=torch.float64
                        )
                        for i, name in enumerate(self.names)
                    }
                    if self.calibrations
                    else torch.as_tensor(q, dtype=torch.float64)
                )
            ).items()
        }

    def tcp(self, q: np.ndarray, side: int) -> np.ndarray:
        return self.links(q)[self.config["tcp_links"][side]] @ self.offset

    def ik(
        self, source: np.ndarray, side: int, target: np.ndarray
    ) -> tuple[np.ndarray, dict]:
        """Use the isolated Real2Sim approximate IK, not a native motion solver."""
        from ._real2sim_ik import solve_real2sim_ik

        return solve_real2sim_ik(self, source, side, target)

    def half_open(self, q: np.ndarray, side: int) -> np.ndarray:
        values = np.asarray(q)
        if not self.calibrations:
            return values[..., self.jaw_columns[side]].mean(axis=-1)
        item = self.calibrations[side]
        grid = np.asarray(item["qpos"])
        driver = int(item.get("driver_index", 0))
        order = np.argsort(grid[:, driver])
        return (
            np.interp(
                values[..., self.jaw_columns[side][driver]],
                grid[order, driver],
                np.asarray(item["width_m"])[order],
            )
            / 2
        )

    def set_half_open(
        self, q: np.ndarray, side: int, half: np.ndarray | float
    ) -> np.ndarray:
        result = np.array(q, copy=True)
        half = np.clip(half, *self.opening_limits[side])
        if not self.calibrations:
            result[..., self.jaw_columns[side]] = np.asarray(half)[..., None]
            return result
        item = self.calibrations[side]
        width = np.asarray(half) * 2
        grid = np.asarray(item["width_m"])
        order = np.argsort(grid)
        qpos_grid = np.asarray(item["qpos"])[order]
        for i, column in enumerate(self.jaw_columns[side]):
            result[..., column] = np.interp(width, grid[order], qpos_grid[:, i])
        return result

    def finger_surfaces(self, side: int) -> tuple[list[str], list[np.ndarray]]:
        if self.calibrations:
            item = self.calibrations[side]
            return list(item["pad_links"]), [
                np.asarray(a, dtype=float) for a in item["outward_axes"]
            ]
        urdf = ET.fromstring(self.path.read_bytes())
        joints = [
            urdf.find(f"joint[@name='{self.names[j]}']") for j in self.jaw_columns[side]
        ]
        return (
            [j.find("child").get("link") for j in joints],
            [np.fromstring(j.find("axis").get("xyz"), sep=" ") for j in joints],
        )
