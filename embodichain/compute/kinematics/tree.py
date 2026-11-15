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

"""Floating-base tree kinematics with differentiable link-origin velocities."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from xml.etree import ElementTree as ET

import torch

from embodichain.utils.math import matrix_from_euler
from .matrix import matrix_product, matrix_vector_product

__all__ = ["TreeKinematics", "LinkKinematics"]


@dataclass(frozen=True)
class LinkKinematics:
    """World-frame link origins, rotations and velocities with leading batch axes."""

    position: torch.Tensor
    rotation: torch.Tensor
    linear_velocity: torch.Tensor
    angular_velocity: torch.Tensor


class TreeKinematics:
    """Evaluate a URDF tree independently of a simulator.

    Fixed, revolute, continuous and prismatic joints are supported. Joint input
    order is explicit; output links follow a parent-before-child traversal.
    Construction parses geometry once. Evaluation batches links by tree depth
    and preserves derivatives through root state, joint position and velocity.

    Args:
        urdf: URDF XML text.
        joint_names: Ordered names of all independent movable joints.
        device: Device for immutable geometry tensors.
        dtype: Floating point precision shared with evaluation inputs.
    """

    def __init__(
        self,
        urdf: str,
        joint_names: Sequence[str],
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("Tree kinematics supports float32 or float64")
        model = ET.fromstring(urdf)
        self.urdf = urdf
        self.joint_names = tuple(joint_names)
        if not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("Joint names must be unique")
        self.device = torch.empty(0, device=device).device
        self.dtype = dtype
        joints = model.findall("joint")
        links = [link.attrib["name"] for link in model.findall("link")]
        children = {joint.find("child").attrib["link"] for joint in joints}
        roots = set(links) - children
        if len(roots) != 1:
            raise ValueError("URDF must contain one rooted tree")
        movable = [j.attrib["name"] for j in joints if j.attrib["type"] != "fixed"]
        if set(movable) != set(self.joint_names):
            raise ValueError(
                "Joint order must name every movable URDF joint exactly once"
            )
        by_parent: dict[str, list[ET.Element]] = {}
        for joint in joints:
            if (
                joint.attrib["type"]
                not in ("fixed", "revolute", "continuous", "prismatic")
                or joint.find("mimic") is not None
            ):
                raise ValueError("Tree kinematics requires independent scalar joints")
            by_parent.setdefault(joint.find("parent").attrib["link"], []).append(joint)
        names = [next(iter(roots))]
        parent, depth, joint_ids = [0], [0], [len(self.joint_names)]
        positions, angles, axes, slides = [[0.0] * 3], [[0.0] * 3], [[0.0] * 3], [0.0]
        limits: dict[str, tuple[float, float]] = {}

        def vector(node: ET.Element | None, key: str) -> list[float]:
            return [
                float(x)
                for x in (
                    node.get(key, "0 0 0") if node is not None else "0 0 0"
                ).split()
            ]

        def visit(name: str, index: int) -> None:
            for joint in by_parent.get(name, ()):
                child = joint.find("child").attrib["link"]
                if child in names:
                    raise ValueError("URDF has a cycle or repeated child link")
                names.append(child)
                child_id = len(names) - 1
                kind = joint.attrib["type"]
                parent.append(index)
                depth.append(depth[index] + 1)
                joint_ids.append(
                    len(self.joint_names)
                    if kind == "fixed"
                    else self.joint_names.index(joint.attrib["name"])
                )
                origin = joint.find("origin")
                positions.append(vector(origin, "xyz"))
                angles.append(vector(origin, "rpy"))
                axis = (
                    vector(joint.find("axis"), "xyz")
                    if joint.find("axis") is not None
                    else [1.0, 0.0, 0.0]
                )
                if kind == "fixed":
                    axis = [0.0] * 3
                axes.append(axis)
                slides.append(float(kind == "prismatic"))
                if kind != "fixed":
                    limit = joint.find("limit")
                    limits[joint.attrib["name"]] = (
                        (float("-inf"), float("inf"))
                        if kind == "continuous"
                        else (
                            float(limit.attrib["lower"]),
                            float(limit.attrib["upper"]),
                        )
                    )
                visit(child, child_id)

        visit(names[0], 0)
        if set(names) != set(links):
            raise ValueError("URDF contains disconnected links")
        self.link_names = tuple(names)
        self.moving_link_names = tuple(
            [names[0]]
            + [
                names[i]
                for i, j in enumerate(joint_ids[1:], 1)
                if j < len(self.joint_names)
            ]
        )
        self.position_limits = torch.tensor(
            [limits[n] for n in self.joint_names], device=self.device, dtype=dtype
        )
        self._positions = torch.tensor(positions, device=self.device, dtype=dtype)
        # URDF fixed-axis roll/pitch/yaw equals intrinsic ZYX with reversed angles.
        self._rotations = matrix_from_euler(
            torch.tensor(angles, device=self.device, dtype=dtype).flip(-1), "ZYX"
        )
        axes_tensor = torch.tensor(axes, device=self.device, dtype=dtype)
        norms = axes_tensor.norm(dim=-1, keepdim=True)
        if any(
            j < len(self.joint_names) and float(norms[i]) == 0
            for i, j in enumerate(joint_ids)
        ):
            raise ValueError("Movable joint axes must be nonzero")
        self._axes = axes_tensor / norms.clamp_min(1e-12)
        self._slides = torch.tensor(slides, device=self.device, dtype=dtype)
        self._joint_ids = torch.tensor(joint_ids, device=self.device, dtype=torch.long)
        self._parent_axes = (self._rotations @ self._axes.unsqueeze(-1)).squeeze(-1)
        x, y, z = self._axes.unbind(-1)
        zero = torch.zeros_like(x)
        self._axis_skew = torch.stack(
            (zero, -z, y, z, zero, -x, -y, x, zero), -1
        ).reshape(-1, 3, 3)
        self._axis_skew_squared = self._axis_skew @ self._axis_skew
        self._identity = torch.eye(3, device=self.device, dtype=dtype)
        self._levels = []
        previous_level = [0]
        level_order = [0]
        for level in range(1, max(depth) + 1):
            selected = [i for i, d in enumerate(depth) if d == level]
            self._levels.append(
                tuple(
                    torch.tensor(a, device=self.device, dtype=torch.long)
                    for a in (
                        selected,
                        [previous_level.index(parent[i]) for i in selected],
                        [joint_ids[i] for i in selected],
                    )
                )
            )
            level_order.extend(selected)
            previous_level = selected
        self._link_order = torch.tensor(
            [level_order.index(i) for i in range(len(names))], device=self.device
        )

    def link_indices(self, names: Sequence[str]) -> torch.Tensor:
        """Return device indices in the requested link order."""
        return torch.tensor(
            [self.link_names.index(n) for n in names],
            device=self.device,
            dtype=torch.long,
        )

    def evaluate(
        self,
        position: torch.Tensor,
        rotation: torch.Tensor,
        qpos: torch.Tensor,
        linear_velocity: torch.Tensor,
        angular_velocity: torch.Tensor,
        qvel: torch.Tensor,
    ) -> LinkKinematics:
        """Return world link states for ``(..., dof)`` joint and root inputs.

        Linear velocities refer to link origins, including the floating root.
        Returned tensors own their storage and retain the input computation graph.
        """
        batch = qpos.shape[:-1]
        if (
            qpos.shape[-1] != len(self.joint_names)
            or qvel.shape != qpos.shape
            or position.shape != (*batch, 3)
            or rotation.shape != (*batch, 3, 3)
            or linear_velocity.shape != position.shape
            or angular_velocity.shape != position.shape
        ):
            raise ValueError("Inconsistent root/joint batch shapes")
        values = (position, rotation, qpos, linear_velocity, angular_velocity, qvel)
        if any(x.device != self.device or x.dtype != self.dtype for x in values):
            raise ValueError("Kinematics inputs must share model device and dtype")
        # The final column supplies zero motion to fixed joints.
        q = torch.cat((qpos, torch.zeros_like(qpos[..., :1])), -1)
        dq = torch.cat((qvel, torch.zeros_like(qvel[..., :1])), -1)
        # Axes are constant and already normalized. Evaluate joint rotations
        # together; only the parent-to-child propagation depends on tree depth.
        angles = q.index_select(-1, self._joint_ids) * (1 - self._slides)
        local_rotation = (
            self._identity
            + angles.sin()[..., None, None] * self._axis_skew
            + (2 * (angles * 0.5).sin().square())[..., None, None]
            * self._axis_skew_squared
        )
        relative_rotation = matrix_product(self._rotations, local_rotation)
        p, r = [position.unsqueeze(-2)], [rotation.unsqueeze(-3)]
        v, w = [linear_velocity.unsqueeze(-2)], [angular_velocity.unsqueeze(-2)]
        for children, parents, joints in self._levels:
            parent_r = r[-1].index_select(-3, parents)
            slide = self._slides[children]
            coordinate = q.index_select(-1, joints)
            velocity = dq.index_select(-1, joints)
            axis_world = matrix_vector_product(parent_r, self._parent_axes[children])
            translation = matrix_vector_product(
                parent_r, self._positions[children]
            ) + axis_world * (coordinate * slide).unsqueeze(-1)
            child_p = p[-1].index_select(-2, parents) + translation
            child_r = matrix_product(
                parent_r, relative_rotation.index_select(-3, children)
            )
            parent_w = w[-1].index_select(-2, parents)
            child_v = (
                v[-1].index_select(-2, parents)
                + torch.cross(parent_w, translation, dim=-1)
                + axis_world * (velocity * slide).unsqueeze(-1)
            )
            child_w = parent_w + axis_world * (velocity * (1 - slide)).unsqueeze(-1)
            p.append(child_p)
            r.append(child_r)
            v.append(child_v)
            w.append(child_w)
        return LinkKinematics(
            torch.cat(p, -2).index_select(-2, self._link_order),
            torch.cat(r, -3).index_select(-3, self._link_order),
            torch.cat(v, -2).index_select(-2, self._link_order),
            torch.cat(w, -2).index_select(-2, self._link_order),
        )
