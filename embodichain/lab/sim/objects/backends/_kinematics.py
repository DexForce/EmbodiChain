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

"""Adapt resolved Spawn joint frames to a pytorch-kinematics tree."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from embodichain.lab.sim.utility.import_utils import lazy_import_pytorch_kinematics

if TYPE_CHECKING:
    import pytorch_kinematics as pk
    from dexsim.scene import SpawnedArticulation

__all__ = []


def _build_pk_chain(entity: SpawnedArticulation, device: torch.device) -> pk.Chain:
    """Build root-relative FK from resolved topology without reading the asset.

    Each movable joint is represented by a joint frame followed by a fixed
    physical-link frame. This preserves ``origin @ motion(q) @ inv(target)``:
    PK's link offsets precede motion and cannot represent a child-side joint
    frame directly. The resolved descriptors already own unit conversions.
    """
    pk = lazy_import_pytorch_kinematics()
    link_names = list(entity.get_link_names())
    root_name = entity.get_root_link_name()
    if root_name not in link_names or len(set(link_names)) != len(link_names):
        raise ValueError("Kinematic topology requires unique links and a known root.")

    joints = [entity.get_joint_desc(name) for name in entity.get_joint_names()]
    joint_names = [joint.name for joint in joints]
    if len(set(joint_names)) != len(joint_names):
        raise ValueError("Kinematic topology contains duplicate joint names.")
    children = {name: [] for name in link_names}
    parent_by_child: dict[str, str] = {}
    for joint in joints:
        parent, child = joint.parent_link_name, joint.child_link_name
        joint_type = str(getattr(joint.joint_type, "name", joint.joint_type)).lower()
        # A fixed world attachment does not participate in root-relative FK.
        if child == root_name and not parent and joint_type == "fixed":
            continue
        if parent not in children or child not in children:
            raise ValueError(f"Joint {joint.name!r} references an unknown link.")
        if child == root_name or child in parent_by_child:
            raise ValueError(
                f"Joint {joint.name!r} introduces multiple parents or a cycle."
            )
        parent_by_child[child] = parent
        children[parent].append(joint)

    visited: set[str] = set()
    pending = [root_name]
    while pending:
        name = pending.pop()
        if name in visited:
            raise ValueError("Kinematic topology contains a cycle.")
        visited.add(name)
        pending.extend(joint.child_link_name for joint in children[name])
    if visited != set(link_names):
        raise ValueError("Kinematic topology contains disconnected links or a cycle.")

    frames = {name: pk.Frame(name, link=pk.Link(name=name)) for name in link_names}
    used_frame_names = set(link_names)
    for parent, child_joints in children.items():
        for joint in child_joints:
            joint_type = str(
                getattr(joint.joint_type, "name", joint.joint_type)
            ).lower()
            if joint_type == "continuous":
                joint_type = "revolute"
            if joint_type not in ("fixed", "revolute", "prismatic"):
                raise NotImplementedError(
                    f"Joint {joint.name!r} has unsupported kinematic type {joint_type!r}; "
                    "set build_pk_chain=False for simulation without FK."
                )
            origin = _joint_pose(joint.origin_pose, joint.name, "origin_pose")
            target = _joint_pose(joint.target_pose, joint.name, "target_pose")
            child = frames[joint.child_link_name]
            if joint_type == "fixed":
                child.joint = pk.Joint(
                    name=joint.name,
                    offset=pk.Transform3d(matrix=origin @ torch.linalg.inv(target)),
                )
                frames[parent].add_child(child)
                continue

            axis = torch.as_tensor(joint.axis, dtype=torch.float32).clone()
            if axis.shape != (3,) or not torch.isfinite(axis).all() or axis.norm() == 0:
                raise ValueError(
                    f"Joint {joint.name!r} requires a finite nonzero axis."
                )
            frame_name = f"__embodichain_joint__{joint.name}"
            while frame_name in used_frame_names:
                frame_name += "_"
            used_frame_names.add(frame_name)
            limits = None
            if joint.lower_limit is not None and joint.upper_limit is not None:
                limits = (float(joint.lower_limit), float(joint.upper_limit))
            joint_frame = pk.Frame(
                frame_name,
                link=pk.Link(name=frame_name),
                joint=pk.Joint(
                    name=joint.name,
                    joint_type=joint_type,
                    axis=axis,
                    offset=pk.Transform3d(matrix=origin),
                    limits=limits,
                ),
            )
            child.joint = pk.Joint(
                name=f"{frame_name}__link",
                offset=pk.Transform3d(matrix=torch.linalg.inv(target)),
            )
            frames[parent].add_child(joint_frame)
            joint_frame.add_child(child)
    return pk.Chain(frames[root_name]).to(device=device)


def _joint_pose(value: object, joint_name: str, field: str) -> torch.Tensor:
    """Copy a resolved homogeneous joint frame to the CPU chain builder."""
    pose = torch.as_tensor(value, dtype=torch.float32, device="cpu").clone()
    if pose.shape != (4, 4) or not torch.isfinite(pose).all():
        raise ValueError(f"Joint {joint_name!r} requires a finite 4x4 {field}.")
    return pose
