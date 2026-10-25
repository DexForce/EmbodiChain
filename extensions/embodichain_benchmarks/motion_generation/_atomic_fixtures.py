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

"""Robot, grasp and mesh fixtures owned by installed Atomic benchmarks.

Keep fixture construction independent of the optional examples data tree.
The calibrations and defaults preserve the benchmark's original sample scenes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import numpy as np
import torch

from embodichain.data import get_data_path
from embodichain.lab.sim.atomic_skills import AntipodalAffordance, ObjectSemantics
from embodichain.lab.sim.cfg import RobotCfg
from embodichain.lab.sim.objects import RigidObject
from embodichain.lab.sim.robots import build_dual_arm_cfg
from embodichain.lab.sim.motion.solvers import PytorchSolverCfg
from embodichain.toolkits.graspkit import ParallelJawGripperModelCfg
from embodichain.toolkits.graspkit.pg_grasp import (
    AntipodalGraspPoseGenerator,
    AntipodalGraspPoseGeneratorCfg,
    GraspAnnotationCfg,
    ParallelJawGraspCollisionCfg,
)
from embodichain_robots import FrankaPandaPgiCfg, URRobotCfg

from ._tcp_presets import ToolPoint, sample_tool_tcp

__all__ = [
    "DUAL_UR5_INIT_POS",
    "DUAL_UR5_INIT_ROT",
    "create_antipodal_semantics",
    "create_parallel_jaw_grasp_pose_generator",
    "create_ur5_gripper_robot_cfg",
    "create_dual_tutorial_robot_cfg",
    "get_local_vertices",
    "compute_local_bounds",
    "compute_world_bounds",
    "transform_points",
    "normalize_vector",
    "rotate_pose_about_world_z",
]

GRIPPER_URDF_PATH = "DH_PGI_140_80/DH_PGI_140_80.urdf"


GRIPPER_HAND_JOINT_PATTERN = "gripper_finger1_joint_1"


ROBOTIQ_2F_140_URDF_PATH = "Robotiq/robotiq_arg2f_140/robotiq_arg2f_140.urdf"


ROBOTIQ_HAND_JOINT_PATTERN = r".*(?:finger|knuckle)_joint"


_UR_TUTORIAL_QPOS_LIMITS: tuple[tuple[float, float], ...] = tuple(
    (-2.0 * math.pi, 2.0 * math.pi) for _ in range(6)
)


TUTORIAL_PARALLEL_JAW_MODEL = ParallelJawGripperModelCfg(
    model_id="dh_pgi_140_80",
    min_opening_width=0.003,
    max_opening_width=0.100,
    finger_length=0.10,
    finger_width=0.040,
    finger_thickness=0.01,
    palm_depth=0.096,
)


_FRANKA_TUTORIAL_BASE_ROTATION = (0.0, 0.0, 180.0)


TutorialRobot = Literal["ur5", "franka", "ur10"]


TUTORIAL_ROBOTS: tuple[TutorialRobot, ...] = (
    "ur5",
    "franka",
    "ur10",
)


def create_antipodal_semantics(
    obj: RigidObject,
    *,
    label: str,
) -> ObjectSemantics:
    """Describe a rigid object using target-local antipodal geometry.

    Args:
        obj: Rigid object that will be grasped.
        label: Human-readable object category.
    Returns:
        Object semantics with mesh data stored only on its affordance.
    """
    vertices = obj.get_vertices(env_ids=[0], scale=True)[0]
    triangles = obj.get_triangles(env_ids=[0])[0]
    return ObjectSemantics(
        label=label,
        geometry={},
        affordance=AntipodalAffordance(
            mesh_vertices=vertices,
            mesh_triangles=triangles,
        ),
        entity_id=obj.uid,
    )


def create_parallel_jaw_grasp_pose_generator(
    *,
    n_sample: int,
    force_refresh: bool,
    opening_margin: float = 0.03,
    gripper_model: ParallelJawGripperModelCfg | None = None,
) -> AntipodalGraspPoseGenerator:
    """Create a standalone generator for the sample's mounted gripper.

    Args:
        n_sample: Number of antipodal contact-pair samples.
        force_refresh: Whether to regenerate cached object annotations.
        opening_margin: Additional clearance used for grasp collision checks.
        gripper_model: Explicit fixture geometry; omission uses the PGI sample.

    Returns:
        A generator with an independently owned gripper model.
    """
    return AntipodalGraspPoseGenerator(
        TUTORIAL_PARALLEL_JAW_MODEL if gripper_model is None else gripper_model,
        algorithm_cfg=AntipodalGraspPoseGeneratorCfg(sample_count=n_sample),
        collision_cfg=ParallelJawGraspCollisionCfg(
            opening_margin=opening_margin,
            point_sample_density=0.012,
            filter_ground_collision=False,
        ),
        annotation_cfg=GraspAnnotationCfg(
            selection_mode="whole_mesh",
            viser_port=11801,
            force_refresh=force_refresh,
        ),
    )


def create_ur5_gripper_robot_cfg(
    init_pos: Sequence[float] = (0.0, 0.0, 0.0),
    init_qpos: Sequence[float] | None = None,
    tcp_z: float | None = None,
    *,
    tcp_point: ToolPoint = "default",
    **kwargs: object,
) -> RobotCfg:
    """Build a UR5 arm + DH_PGI_140_80 gripper robot configuration.

    The arm is taken from :class:`~embodichain_robots.ur_robot.URRobotCfg`
    so the URDF, joint names and drive defaults match the canonical UR family
    config. The gripper and tool-center-point offset are added on top to match
    the existing atomic-skill tutorial setups.

    .. attention::
        :class:`~embodichain.lab.sim.cfg.URDFCfg` defaults to upper-casing joint
        names during multi-component assembly. The override dict passed to
        :meth:`URRobotCfg.from_dict` sets
        ``urdf_cfg.name_case = {"joint": "lower", "link": "lower"}`` so the
        assembled robot keeps the source URDF's lowercase joint names
        (``joint1``..``joint6`` and ``gripper_finger1_joint_1``), matching the
        control parts produced by
        :class:`~embodichain_robots.ur_robot.URRobotCfg`.

    Args:
        init_pos: Initial root position of the robot in the arena.
        tcp_z: Explicit local-Z override; omission uses the UR5/PGI preset.
        tcp_point: Named sample tool point in the fixture preset.

    Returns:
        A fully populated :class:`~embodichain.lab.sim.cfg.RobotCfg`.
    """
    qpos = (
        [0.0, -1.57, 1.57, -1.57, -1.57, 0.0, 0.0, 0.0]
        if init_qpos is None
        else list(init_qpos)
    )
    return URRobotCfg.from_dict(
        {
            "robot_type": "ur5",
            "uid": "UR5",
            "urdf_cfg": {
                "components": [
                    {
                        "component_type": "hand",
                        "urdf_path": GRIPPER_URDF_PATH,
                    }
                ],
            },
            "control_parts": {
                "eef": [GRIPPER_HAND_JOINT_PATTERN],
            },
            "joint_drive_props": {
                "stiffness": {
                    "arm": 5e4,
                    GRIPPER_HAND_JOINT_PATTERN: 1e3,
                },
                "damping": {
                    "arm": 5e3,
                    GRIPPER_HAND_JOINT_PATTERN: 1e2,
                },
                "max_effort": {
                    "arm": 1e6,
                    GRIPPER_HAND_JOINT_PATTERN: 1e4,
                },
            },
            "solver_cfg": {
                "arm": {
                    "tcp": sample_tool_tcp("ur5", point=tcp_point, tcp_z=tcp_z),
                    "user_qpos_limits": _UR_TUTORIAL_QPOS_LIMITS,
                }
            },
            "init_qpos": qpos,
            "init_pos": init_pos,
        }
    )


def create_franka_panda_robot_cfg(
    init_pos: Sequence[float] = (0.0, 0.0, 0.0),
    init_qpos: Sequence[float] | None = None,
    *,
    tcp_z: float | None = None,
    tcp_point: ToolPoint = "default",
    **kwargs: object,
) -> RobotCfg:
    """Build the Franka arm + PGI gripper configuration for the tutorials.

    :class:`~embodichain_robots.FrankaPandaPgiCfg` owns the assembly,
    control parts, drives, gripper contacts, and flange kinematic chain. The
    sample owns its placement, initial pose, and TCP calibration in
    ``tcp_presets.py``. The shared scenes place manipulation targets on the
    robot's negative-X side, so the base is rotated 180 degrees.

    Args:
        init_pos: Initial root position of the robot in the arena.
        init_qpos: Optional full robot joint configuration.
        tcp_z: Explicit local-Z override; omission uses the Franka/PGI preset.
        tcp_point: Named sample tool point in the fixture preset.

    Returns:
        A fully populated :class:`~embodichain.lab.sim.cfg.RobotCfg`.
    """
    overrides = {
        "uid": "FrankaPanda",
        "init_pos": init_pos,
        "init_rot": _FRANKA_TUTORIAL_BASE_ROTATION,
        "solver_cfg": {
            "arm": {
                "tcp": sample_tool_tcp("franka", point=tcp_point, tcp_z=tcp_z),
            }
        },
    }
    if init_qpos is not None:
        overrides["init_qpos"] = list(init_qpos)
    return FrankaPandaPgiCfg.from_dict(overrides)


def create_ur10_robotiq_robot_cfg(
    init_pos: Sequence[float] = (0.0, 0.0, 0.0),
    init_qpos: Sequence[float] | None = None,
    *,
    tcp_point: ToolPoint = "default",
    **kwargs: object,
) -> RobotCfg:
    """Build a UR10 arm with a six-DOF Robotiq 2F-140 gripper.

    The hand control part is intentionally expressed as one regular expression.
    Robot materialization expands it over the six active finger/knuckle joints
    while excluding the gripper's fixed joints. The solver TCP includes the
    Robotiq mounting rotation and its 0.21 m tool offset. Legacy PGI ``tcp_z``
    arguments do not replace this fixture's full mounting transform.

    Args:
        init_pos: Initial root position of the robot in the arena.
        init_qpos: Optional full 12-DOF arm-plus-hand configuration.
        tcp_point: Named sample tool point; Robotiq uses one calibrated pose.

    Returns:
        A UR10/Robotiq configuration exposing ``arm`` and six-joint ``eef``
        control parts.
    """
    qpos = (
        [0.0, -1.57, 1.57, -1.57, -1.57, 0.0] + [0.0] * 6
        if init_qpos is None
        else list(init_qpos)
    )
    if len(qpos) != 12:
        raise ValueError(
            "UR10/Robotiq init_qpos must contain 12 values: six arm and six "
            f"gripper joints, got {len(qpos)}."
        )
    return URRobotCfg.from_dict(
        {
            "robot_type": "ur10",
            "uid": "UR10Robotiq2F140",
            "urdf_cfg": {
                "components": [
                    {
                        "component_type": "hand",
                        "urdf_path": get_data_path(ROBOTIQ_2F_140_URDF_PATH),
                    }
                ],
            },
            "control_parts": {
                "eef": [ROBOTIQ_HAND_JOINT_PATTERN],
            },
            "joint_drive_props": {
                "stiffness": {ROBOTIQ_HAND_JOINT_PATTERN: 1e3},
                "damping": {ROBOTIQ_HAND_JOINT_PATTERN: 1e2},
                "max_effort": {ROBOTIQ_HAND_JOINT_PATTERN: 1e3},
            },
            "solver_cfg": {
                "arm": {
                    "tcp": sample_tool_tcp("ur10", point=tcp_point),
                    "user_qpos_limits": _UR_TUTORIAL_QPOS_LIMITS,
                }
            },
            "init_qpos": qpos,
            "init_pos": init_pos,
        }
    )


def create_tutorial_robot_cfg(
    robot_type: TutorialRobot,
    init_pos: Sequence[float] = (0.0, 0.0, 0.0),
    init_qpos: Sequence[float] | None = None,
    **kwargs: object,
) -> RobotCfg:
    """Build a selected tutorial arm with the common PGI gripper contract.

    Args:
        robot_type: Tutorial robot family to construct.
        init_pos: Initial root position of the robot in its arena.
        init_qpos: Optional full robot joint configuration.

    Returns:
        A UR5, Franka, or UR10/Robotiq robot configuration exposing ``arm``
        and ``eef``.

    Raises:
        ValueError: If ``robot_type`` is not supported.
    """
    if robot_type == "ur5":
        return create_ur5_gripper_robot_cfg(
            init_pos=init_pos,
            init_qpos=init_qpos,
            **kwargs,
        )
    if robot_type == "franka":
        return create_franka_panda_robot_cfg(
            init_pos=init_pos,
            init_qpos=init_qpos,
            **kwargs,
        )
    if robot_type == "ur10":
        return create_ur10_robotiq_robot_cfg(
            init_pos=init_pos,
            init_qpos=init_qpos,
            **kwargs,
        )
    raise ValueError(
        f"Unsupported tutorial robot {robot_type!r}; expected one of {TUTORIAL_ROBOTS}."
    )


DUAL_UR5_INIT_POS = (1.95, 0.0, 0.1)


DUAL_UR5_INIT_ROT = (0.0, 0.0, -90.0)


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
    tcp_z: float | None = None,
    gripper: Literal["robotiq_2f_140"] | None = None,
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

    Franka retains its preset FEP analytical solver and redundancy search.
    ``solver="ur"`` selects the analytical solver for UR5 and UR10, while
    ``solver="pytorch"`` explicitly selects numerical IK for any preset.
    UR5 and Franka default to the PGI hand; Handover explicitly selects the
    six-joint Robotiq 2F-140, which UR10 already uses. Its TCP includes the
    mounting rotation and an independently calibrated tool offset. Control-part
    names and downstream action bindings stay identical across robot choices.

    Args:
        robot_type: Arm family to mount on both sides.
        uid: Simulation robot identifier.
        urdf_name: Cache name for the assembled dual-arm URDF.
        tcp_z: Optional explicit PGI local-Z override for Python callers.
            Omission selects the fixture's handover TCP; UR10/Robotiq uses
            its separate calibrated mounting transform.
        gripper: Explicit Handover gripper selection. Robotiq uses its full
            fixture TCP and does not consume the PGI ``tcp_z`` override.
        solver: UR solver selection; Franka retains its preset unless
            ``"pytorch"`` is explicitly selected.
        ur_ik_nearest_weight: Optional nearest-solution weights for UR IK.
        pytorch_num_samples: Seed count when explicitly selecting PyTorch IK.
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
    if gripper not in (None, "robotiq_2f_140"):
        raise ValueError(f"Unknown dual sample gripper {gripper!r}")
    base_cfg = create_tutorial_robot_cfg(robot_type)
    if gripper == "robotiq_2f_140":
        old_hand_patterns = tuple(base_cfg.control_parts["eef"])
        hand_component = base_cfg.urdf_cfg.components["hand"]
        base_cfg.urdf_cfg.add_component(
            "hand",
            get_data_path(ROBOTIQ_2F_140_URDF_PATH),
            hand_component["transform"],
            **hand_component.get("params", {}),
        )
        base_cfg.control_parts["eef"] = [ROBOTIQ_HAND_JOINT_PATTERN]
        base_cfg.init_qpos = (
            base_cfg.init_qpos[: len(base_cfg.control_parts["arm"])] + [0.0] * 6
        )
        # Drop retired PGI drive selectors before adding Robotiq's six joints.
        for field in base_cfg.joint_drive_props.__dataclass_fields__:
            rules = getattr(base_cfg.joint_drive_props, field)
            if isinstance(rules, dict):
                for pattern in old_hand_patterns:
                    rules.pop(pattern, None)
    tcp = sample_tool_tcp(
        robot_type,
        point="handover",
        tcp_z=tcp_z if robot_type != "ur10" and gripper is None else None,
        gripper=gripper,
    )
    base_solver = base_cfg.solver_cfg["arm"]
    if solver == "ur":
        base_solver.tcp = tcp
        if robot_type in {"ur5", "ur10"}:
            base_solver.ik_nearest_weight = ur_ik_nearest_weight
    else:
        base_solver = PytorchSolverCfg(
            end_link_name=base_solver.end_link_name,
            root_link_name=base_solver.root_link_name,
            tcp=tcp,
            num_samples=pytorch_num_samples,
            user_qpos_limits=base_solver.user_qpos_limits,
        )
        base_cfg.solver_cfg["arm"] = base_solver

    hand_joint_pattern = base_cfg.control_parts["eef"][0]
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


def get_local_vertices(obj: RigidObject) -> torch.Tensor:
    """Return scaled local vertices from the first environment."""
    return obj.get_vertices(env_ids=[0], scale=True)[0]


def compute_local_bounds(
    vertices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a local mesh axis-aligned bounding box."""
    return vertices.min(dim=0).values, vertices.max(dim=0).values


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
