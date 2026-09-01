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
    RigidObjectCfg,
    RobotCfg,
)
from embodichain.lab.sim.objects import RigidObject, Robot
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.lab.sim.robots import build_dual_arm_cfg
from embodichain.lab.sim.solvers import PytorchSolverCfg
from embodichain.utils import logger
from scripts.tutorials.atomic_action.tutorial_utils import (
    ROBOTIQ_2F_140_TCP,
    TutorialRobot,
    configure_newton_gripper_contacts,
    create_tutorial_rigid_body_physics,
    create_tutorial_robot_cfg,
)

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


def create_dual_tutorial_robot_cfg(
    *,
    robot_type: TutorialRobot,
    uid: str,
    urdf_name: str,
    tcp_z: float,
    solver: Literal["ur", "pytorch"] = "ur",
    ur_ik_nearest_weight: Sequence[float] | None = None,
    pytorch_num_samples: int = 30,
    init_pos: Sequence[float] = DUAL_UR5_INIT_POS,
    init_rot: Sequence[float] = DUAL_UR5_INIT_ROT,
    left_arm_home: Sequence[float] | None = None,
    right_arm_home: Sequence[float] | None = None,
    hand_stiffness: float = 1e3,
    hand_damping: float = 1e2,
    hand_max_effort: float = 1e4,
) -> RobotCfg:
    """Build a dual tutorial robot from the selected arm and hand.

    Franka always uses its PyTorch kinematics solver; ``solver="ur"`` selects
    the analytical solver for UR5 and UR10. UR5 and Franka use the shared PGI
    hand, while ``ur10`` uses the six-DOF Robotiq 2F-140 and its
    rotated 0.23 m TCP. The mounting layout, control-part names, and downstream
    action bindings stay identical across robot choices.

    Args:
        robot_type: Arm family to mount on both sides.
        uid: Simulation robot identifier.
        urdf_name: Cache name for the assembled dual-arm URDF.
        tcp_z: PGI tool-center-point offset along local Z. The UR10/Robotiq
            variant uses its fixed mounting TCP instead.
        solver: Preferred UR solver implementation.
        ur_ik_nearest_weight: Optional nearest-solution weights for UR IK.
        pytorch_num_samples: Number of PyTorch IK seed samples.
        init_pos: Root position of the assembled robot.
        init_rot: Root xyz Euler rotation in degrees.
        left_arm_home: Optional left-arm initial configuration.
        right_arm_home: Optional right-arm initial configuration.
        hand_stiffness: Hand joint drive stiffness.
        hand_damping: Hand joint drive damping.
        hand_max_effort: Hand joint maximum effort.

    Returns:
        A dual-arm robot configuration with two matching grippers.
    """
    base_cfg = create_tutorial_robot_cfg(robot_type)
    tcp = (
        ROBOTIQ_2F_140_TCP
        if robot_type == "ur10"
        else (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, tcp_z),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    base_solver = base_cfg.solver_cfg["arm"]
    if robot_type in {"ur5", "ur10"} and solver == "ur":
        base_solver.tcp = tcp
        base_solver.ik_nearest_weight = ur_ik_nearest_weight
    else:
        base_solver = PytorchSolverCfg(
            end_link_name=base_solver.end_link_name,
            root_link_name=base_solver.root_link_name,
            tcp=tcp,
            num_samples=pytorch_num_samples,
        )
        base_cfg.solver_cfg["arm"] = base_solver

    hand_joint_pattern = base_cfg.control_parts["hand"][0]
    for property_name, value in (
        ("stiffness", hand_stiffness),
        ("damping", hand_damping),
        ("max_effort", hand_max_effort),
    ):
        getattr(base_cfg.joint_drive_props, property_name)[hand_joint_pattern] = value

    arm_facing_rotation = make_yaw_transform(
        (0.0, 0.0, 0.0),
        math.radians(float(base_cfg.init_rot[2])),
    )
    mounts = {
        "left": make_yaw_transform((-0.3, -1.45, 0.4), np.pi / 2) @ arm_facing_rotation,
        "right": make_yaw_transform((0.3, -1.45, 0.4), np.pi / 2) @ arm_facing_rotation,
    }
    cfg = build_dual_arm_cfg(base_cfg, mounts)

    # ``build_dual_arm_cfg`` duplicates the arm component and all control
    # parts. Tutorial robots intentionally keep their gripper as a separate
    # URDF component, so mount one copy on each assembled arm as well.
    cfg.urdf_cfg.fname = urdf_name
    hand_component = base_cfg.urdf_cfg.components["hand"]
    for side in ("left", "right"):
        cfg.urdf_cfg.add_component(
            f"{side}_hand",
            hand_component["urdf_path"],
            hand_component["transform"],
            **hand_component.get("params", {}),
        )

    arm_dof = len(base_cfg.control_parts["arm"])
    base_arm_home = list(base_cfg.init_qpos[:arm_dof])
    base_hand_home = list(base_cfg.init_qpos[arm_dof:])
    if left_arm_home is None:
        left_arm_home = base_arm_home
    if right_arm_home is None:
        right_arm_home = base_arm_home
    if len(left_arm_home) != arm_dof or len(right_arm_home) != arm_dof:
        raise ValueError(
            f"Dual {robot_type} arm homes must each contain {arm_dof} joints."
        )

    cfg.uid = uid
    cfg.init_pos = list(init_pos)
    cfg.init_rot = list(init_rot)
    # DexSim traverses the two arm branches breadth-first, so their active
    # joints appear left/right interleaved even though the URDF components are
    # emitted one after the other. Match that runtime order before appending
    # the two gripper components.
    cfg.init_qpos = (
        [
            qpos
            for joint_pair in zip(left_arm_home, right_arm_home, strict=True)
            for qpos in joint_pair
        ]
        + base_hand_home
        + base_hand_home
    )
    return cfg


def add_dual_tutorial_robot(
    sim: SimulationManager,
    *,
    robot_type: TutorialRobot,
    uid: str,
    urdf_name: str,
    tcp_z: float,
    solver: Literal["ur", "pytorch"] = "ur",
    ur_ik_nearest_weight: Sequence[float] | None = None,
    pytorch_num_samples: int = 30,
    init_pos: Sequence[float] = DUAL_UR5_INIT_POS,
    init_rot: Sequence[float] = DUAL_UR5_INIT_ROT,
    left_arm_home: Sequence[float] | None = None,
    right_arm_home: Sequence[float] | None = None,
    hand_stiffness: float = 1e3,
    hand_damping: float = 1e2,
    hand_max_effort: float = 1e4,
) -> Robot:
    """Add a supported dual-arm tutorial robot to a simulation.

    Args:
        sim: Simulation manager that owns the robot.
        robot_type: Arm family to mount on both sides.
        uid: Simulation robot identifier.
        urdf_name: Cache name for the assembled dual-arm URDF.
        tcp_z: PGI tool-center-point offset along local Z. The UR10/Robotiq
            variant uses its fixed mounting TCP instead.
        solver: Preferred UR solver implementation.
        ur_ik_nearest_weight: Optional nearest-solution weights for UR IK.
        pytorch_num_samples: Number of PyTorch IK seed samples.
        init_pos: Root position of the assembled robot.
        init_rot: Root xyz Euler rotation in degrees.
        left_arm_home: Optional left-arm initial configuration.
        right_arm_home: Optional right-arm initial configuration.
        hand_stiffness: Hand joint drive stiffness.
        hand_damping: Hand joint drive damping.
        hand_max_effort: Hand joint maximum effort.

    Returns:
        The added dual-arm robot instance.
    """
    robot_cfg = create_dual_tutorial_robot_cfg(
        robot_type=robot_type,
        uid=uid,
        urdf_name=urdf_name,
        tcp_z=tcp_z,
        solver=solver,
        ur_ik_nearest_weight=ur_ik_nearest_weight,
        pytorch_num_samples=pytorch_num_samples,
        init_pos=init_pos,
        init_rot=init_rot,
        left_arm_home=left_arm_home,
        right_arm_home=right_arm_home,
        hand_stiffness=hand_stiffness,
        hand_damping=hand_damping,
        hand_max_effort=hand_max_effort,
    )
    configure_newton_gripper_contacts(sim, robot_cfg)
    return sim.add_robot(cfg=robot_cfg)


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
            attrs=create_tutorial_rigid_body_physics(
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
        entity_id=obj.uid,
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
    "DUAL_UR5_INIT_POS",
    "DUAL_UR5_INIT_ROT",
    "add_dual_tutorial_robot",
    "add_support_surface",
    "compute_local_bounds",
    "compute_world_bounds",
    "create_manual_object_semantics",
    "create_dual_tutorial_robot_cfg",
    "get_local_vertices",
    "invert_pose",
    "log_action_plan",
    "make_yaw_transform",
    "normalize_vector",
    "resolve_cached_data_path",
    "rotate_pose_about_world_z",
    "settle_object",
    "transform_points",
]
