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

"""Shared scene and dual-arm helpers for atomic-action tutorials."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from embodichain.data import get_data_path
from embodichain.lab.sim import SimulationManager
from embodichain.lab.sim.atomic_actions import Affordance, ObjectSemantics
from embodichain.lab.sim.cfg import (
    JointDrivePropertiesCfg,
    RigidBodyAttributesCfg,
    RigidObjectCfg,
    RobotCfg,
    URDFCfg,
)
from embodichain.lab.sim.objects import RigidObject, Robot
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.lab.sim.solvers import PytorchSolverCfg, SolverCfg, URSolverCfg
from embodichain.utils import logger

ARM_URDF_PATH = "UniversalRobots/UR5/UR5.urdf"
GRIPPER_URDF_PATH = "DH_PGI_140_80/DH_PGI_140_80.urdf"
LEFT_ARM_HOME = (0.0, 0.0, -1.57, -1.57, 1.57, 1.57)
RIGHT_ARM_HOME = (-1.57, -1.57, -1.57, -1.57, 0.0, 0.0)
DUAL_UR5_INIT_POS = (1.95, 0.0, 0.1)
DUAL_UR5_INIT_ROT = (0.0, 0.0, -90.0)


def resolve_cached_data_path(data_path: str) -> str:
    """Resolve an asset from the local cache, falling back to project data."""
    if os.path.isabs(data_path):
        return data_path

    data_root = Path(
        os.environ.get(
            "EMBODICHAIN_DATA_ROOT",
            str(Path.home() / ".cache" / "embodichain_data"),
        )
    )
    for candidate in (data_root / data_path, data_root / "extract" / data_path):
        if candidate.exists():
            return str(candidate)
    return get_data_path(data_path)


def make_yaw_transform(xyz: tuple[float, float, float], yaw: float) -> np.ndarray:
    """Build a homogeneous transform from translation and world yaw."""
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = np.array(
        (
            (cos_yaw, -sin_yaw, 0.0),
            (sin_yaw, cos_yaw, 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float32,
    )
    transform[:3, 3] = np.asarray(xyz, dtype=np.float32)
    return transform


def make_dual_ur5_solver_cfg(
    tcp_z: float,
    *,
    solver: Literal["pytorch", "ur"] = "ur",
    ur_ik_nearest_weight: Sequence[float] | None = None,
    clear_urdf_path: bool = False,
    pytorch_num_samples: int = 30,
) -> dict[str, SolverCfg]:
    """Build matching left/right solver configs for dual-UR5 tutorials."""
    tcp = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, tcp_z],
        [0.0, 0.0, 0.0, 1.0],
    ]
    configs: dict[str, SolverCfg] = {}
    for prefix in ("left", "right"):
        if solver == "pytorch":
            config = PytorchSolverCfg(
                end_link_name=f"{prefix}_ee_link",
                root_link_name=f"{prefix}_base_link",
                tcp=tcp,
                num_samples=pytorch_num_samples,
            )
        else:
            config = URSolverCfg(
                ur_type="ur5",
                end_link_name=f"{prefix}_ee_link",
                root_link_name=f"{prefix}_base_link",
                tcp=tcp,
                ik_nearest_weight=ur_ik_nearest_weight,
            )
            if clear_urdf_path:
                config.urdf_path = None
        configs[f"{prefix}_arm"] = config
    return configs


def add_dual_ur5_robot(
    sim: SimulationManager,
    *,
    uid: str,
    urdf_name: str,
    solver_cfg: Mapping[str, SolverCfg],
    init_pos: Sequence[float] = DUAL_UR5_INIT_POS,
    init_rot: Sequence[float] = DUAL_UR5_INIT_ROT,
    left_arm_home: Sequence[float] = LEFT_ARM_HOME,
    right_arm_home: Sequence[float] = RIGHT_ARM_HOME,
    arm_urdf_path: str = ARM_URDF_PATH,
    gripper_urdf_path: str = GRIPPER_URDF_PATH,
    joint_name_case: str = "upper",
    set_urdf_name_case: bool = True,
    hand_stiffness: float = 1e3,
    hand_damping: float = 1e2,
    hand_max_effort: float = 1e4,
) -> Robot:
    """Add the common dual-UR5 and dual-gripper tutorial embodiment."""
    if joint_name_case not in {"lower", "upper"}:
        raise ValueError("joint_name_case must be 'lower' or 'upper'.")
    prefix = str.upper if joint_name_case == "upper" else str.lower
    left_joint = prefix("left_joint[0-9]")
    right_joint = prefix("right_joint[0-9]")
    left_hand = prefix("left_gripper_finger[1-2]_joint_1")
    right_hand = prefix("right_gripper_finger[1-2]_joint_1")
    left_hand_control = prefix("left_gripper_finger1_joint_1")
    right_hand_control = prefix("right_gripper_finger1_joint_1")

    urdf_cfg = URDFCfg(
        components=[
            {
                "component_type": "left_arm",
                "urdf_path": arm_urdf_path,
                "transform": make_yaw_transform((-0.3, -1.45, 0.4), np.pi / 2),
            },
            {
                "component_type": "right_arm",
                "urdf_path": arm_urdf_path,
                "transform": make_yaw_transform((0.3, -1.45, 0.4), np.pi / 2),
            },
            {"component_type": "left_hand", "urdf_path": gripper_urdf_path},
            {"component_type": "right_hand", "urdf_path": gripper_urdf_path},
        ],
        fname=urdf_name,
    )
    if set_urdf_name_case:
        urdf_cfg.name_case = {"joint": joint_name_case, "link": "lower"}

    cfg = RobotCfg(
        uid=uid,
        urdf_cfg=urdf_cfg,
        drive_pros=JointDrivePropertiesCfg(
            stiffness={
                left_joint: 1e4,
                right_joint: 1e4,
                left_hand: hand_stiffness,
                right_hand: hand_stiffness,
            },
            damping={
                left_joint: 1e3,
                right_joint: 1e3,
                left_hand: hand_damping,
                right_hand: hand_damping,
            },
            max_effort={
                left_joint: 1e5,
                right_joint: 1e5,
                left_hand: hand_max_effort,
                right_hand: hand_max_effort,
            },
            drive_type="force",
        ),
        control_parts={
            "left_arm": [left_joint],
            "right_arm": [right_joint],
            "dual_arm": [left_joint, right_joint],
            "left_hand": [left_hand_control],
            "right_hand": [right_hand_control],
        },
        solver_cfg=dict(solver_cfg),
        init_pos=list(init_pos),
        init_rot=list(init_rot),
        init_qpos=list(left_arm_home) + list(right_arm_home) + [0.0, 0.0, 0.0, 0.0],
    )
    return sim.add_robot(cfg=cfg)


def add_support_surface(
    sim: SimulationManager,
    *,
    size: Sequence[float],
    center: Sequence[float],
) -> RigidObject:
    """Add the standard static support slab used by dual-arm tutorials."""
    return sim.add_rigid_object(
        cfg=RigidObjectCfg(
            uid="support_surface",
            shape=CubeCfg(size=list(size)),
            attrs=RigidBodyAttributesCfg(
                mass=10.0,
                dynamic_friction=0.9,
                static_friction=0.95,
                restitution=0.01,
            ),
            body_type="static",
            init_pos=list(center),
            init_rot=[0.0, 0.0, 0.0],
        )
    )


def settle_object(sim: SimulationManager, obj: RigidObject, step: int = 5) -> None:
    """Reset, settle, and freeze an object before tutorial planning."""
    if sim.device.type == "cuda":
        sim.init_gpu_physics()
    obj.reset()
    if step > 0:
        sim.update(step=step)
    obj.clear_dynamics()


def create_manual_object_semantics(obj: RigidObject, label: str) -> ObjectSemantics:
    """Create minimal semantics for a caller-provided grasp pose."""
    return ObjectSemantics(
        label=label,
        geometry={},
        affordance=Affordance(object_label=label),
        entity=obj,
    )


def get_local_vertices(obj: RigidObject) -> torch.Tensor:
    """Return scaled local vertices from the first environment."""
    return obj.get_vertices(env_ids=[0], scale=True)[0]


def compute_local_bounds(
    vertices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a local mesh axis-aligned bounding box."""
    return vertices.min(dim=0).values, vertices.max(dim=0).values


def invert_pose(pose: torch.Tensor) -> torch.Tensor:
    """Invert a batch of homogeneous transforms."""
    inv_pose = pose.clone()
    rot_t = pose[:, :3, :3].transpose(1, 2)
    inv_pose[:, :3, :3] = rot_t
    inv_pose[:, :3, 3] = -torch.bmm(rot_t, pose[:, :3, 3:4]).squeeze(-1)
    return inv_pose


def transform_points(pose: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Transform local points by a homogeneous pose."""
    return points @ pose[:3, :3].transpose(0, 1) + pose[:3, 3]


def compute_world_bounds(
    object_pose: torch.Tensor,
    local_vertices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a world-space AABB for local mesh vertices."""
    world_vertices = transform_points(object_pose, local_vertices)
    return world_vertices.min(dim=0).values, world_vertices.max(dim=0).values


def normalize_vector(vector: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
    """Normalize a vector with a deterministic degenerate fallback."""
    norm = torch.linalg.norm(vector)
    if norm < 1e-6:
        return fallback.to(device=vector.device, dtype=vector.dtype)
    return vector / norm


def rotate_pose_about_world_z(pose: torch.Tensor, yaw_deg: float) -> torch.Tensor:
    """Rotate pose orientation about world Z while preserving translation."""
    yaw = math.radians(yaw_deg)
    rot = torch.eye(3, dtype=pose.dtype, device=pose.device)
    rot[0, 0] = math.cos(yaw)
    rot[0, 1] = -math.sin(yaw)
    rot[1, 0] = math.sin(yaw)
    rot[1, 1] = math.cos(yaw)
    rotated_pose = pose.clone()
    rotated_pose[:3, :3] = rot @ pose[:3, :3]
    return rotated_pose


def log_action_plan(
    robot: Robot,
    action_name: str,
    trajectory: torch.Tensor,
    joint_ids: list[int],
    segments: Mapping[str, int] | None = None,
) -> None:
    """Log joint and segment details for a planned tutorial action."""
    joint_names = [robot.joint_names[joint_id] for joint_id in joint_ids]
    logger.log_info(f"{action_name} joint ids: {joint_ids}")
    logger.log_info(f"{action_name} joint names: {joint_names}")
    logger.log_info(f"{action_name} trajectory shape: {tuple(trajectory.shape)}")
    if segments is not None:
        logger.log_info(f"{action_name} trajectory segments: {dict(segments)}")


__all__ = [
    "ARM_URDF_PATH",
    "DUAL_UR5_INIT_POS",
    "DUAL_UR5_INIT_ROT",
    "GRIPPER_URDF_PATH",
    "LEFT_ARM_HOME",
    "RIGHT_ARM_HOME",
    "add_dual_ur5_robot",
    "add_support_surface",
    "compute_local_bounds",
    "compute_world_bounds",
    "create_manual_object_semantics",
    "get_local_vertices",
    "invert_pose",
    "log_action_plan",
    "make_dual_ur5_solver_cfg",
    "make_yaw_transform",
    "normalize_vector",
    "resolve_cached_data_path",
    "rotate_pose_about_world_z",
    "settle_object",
    "transform_points",
]
