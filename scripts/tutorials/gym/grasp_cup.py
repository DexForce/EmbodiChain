#!/usr/bin/env python3
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

"""Agent-skill pick-place demo in the atomic_actions.py scene.

This script keeps the robot, cup, light, and camera style aligned with
``scripts/tutorials/sim/atomic_actions.py``. The difference is execution:
instead of directly calling ``AtomicActionEngine.execute_static()``, it calls
Agent atomic skill functions such as ``grasp`` and ``place_on_table``.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from dexsim.utility import images_to_video

ROOT_DIR = Path(__file__).resolve().parents[3]
SIM_TUTORIAL_DIR = ROOT_DIR / "scripts/tutorials/sim"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SIM_TUTORIAL_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_TUTORIAL_DIR))

from atomic_actions import create_mug, initialize_simulation
from embodichain.data import get_data_path
from embodichain.lab.sim import SimulationManager
from embodichain.lab.sim.agent.atom_actions import (
    back_to_initial_pose,
    grasp,
    move_by_relative_offset,
    move_to_absolute_position,
    place_on_table,
)
from embodichain.lab.sim.agent.atomic_action_adapter import (
    validate_pending_public_grasp_after_action,
    validate_pending_public_place_after_action,
)
from embodichain.lab.sim.cfg import (
    JointDrivePropertiesCfg,
    RobotCfg,
    URDFCfg,
)
from embodichain.lab.sim.objects import Robot
from embodichain.lab.sim.sensors import Camera, CameraCfg
from embodichain.lab.sim.solvers import PytorchSolverCfg
from embodichain.utils import logger

FPS = 20
HAND_OPEN = 0.0
HAND_CLOSE = 0.025


@dataclass(frozen=True)
class SkillStep:
    name: str
    action: Callable


def timestamped_output_root() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    return ROOT_DIR / "outputs" / f"{timestamp}_grasp_cup"


def _parse_xyz(value: str) -> list[float]:
    parts = [float(part) for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected three comma-separated floats.")
    return parts


def _parse_xyz_list(value: str) -> list[list[float]]:
    directions = []
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        directions.append(_parse_xyz(item))
    if not directions:
        raise argparse.ArgumentTypeError(
            "Expected at least one direction, e.g. 0,0,-1;1,0,0."
        )
    return directions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a UR10/CoffeeCup pick-place demo through Agent atomic skills "
            "in the same scene as scripts/tutorials/sim/atomic_actions.py."
        )
    )
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--enable_rt", action="store_true")
    parser.add_argument(
        "--open_window",
        action="store_true",
        default=False,
        help="Open the interactive render window while recording.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Output directory. Defaults to outputs/YYYYMMDD_HHMM_grasp_cup.",
    )
    parser.add_argument(
        "--place_x",
        type=float,
        default=0.2489,
        help="Target x coordinate copied from atomic_actions.py place_xpos.",
    )
    parser.add_argument(
        "--place_y",
        type=float,
        default=0.3970,
        help="Target y coordinate copied from atomic_actions.py place_xpos.",
    )
    parser.add_argument(
        "--place_z",
        type=float,
        default=0.2400,
        help="Target EEF z coordinate copied from atomic_actions.py place_xpos.",
    )
    parser.add_argument(
        "--lift_dz",
        type=float,
        default=0.15,
        help="Vertical lift after grasp before moving to the place pose.",
    )
    parser.add_argument(
        "--pre_grasp_dis",
        type=float,
        default=0.15,
        help="Pre-grasp distance passed to the Agent grasp skill.",
    )
    parser.add_argument(
        "--disable_grasp_physical_validation",
        action="store_true",
        default=False,
        help="Skip the post-grasp object-motion validation after the normal lift step.",
    )
    parser.add_argument(
        "--grasp_validation_min_object_lift",
        type=float,
        default=0.05,
        help="Minimum object z displacement required after the normal lift step.",
    )
    parser.add_argument(
        "--pre_place_dis",
        type=float,
        default=0.15,
        help="Retract/lift distance passed to the Agent place_on_table skill.",
    )
    parser.add_argument(
        "--min_video_seconds",
        type=float,
        default=30.0,
        help="Pad the recorded video to at least this duration.",
    )
    parser.add_argument(
        "--disable_public_atomic_actions",
        action="store_true",
        default=False,
        help="Disable the public atomic action adapter inside Agent skills.",
    )
    parser.add_argument(
        "--disable_public_grasp_action",
        action="store_true",
        default=False,
        help="Disable public PickUpActionCfg for Agent grasp.",
    )
    parser.add_argument(
        "--use_public_grasp_semantics",
        action="store_true",
        default=False,
        help="Use ObjectSemantics + AntipodalAffordance for Agent grasp.",
    )
    parser.add_argument(
        "--public_grasp_strategy",
        choices=[
            "top_down",
            "bottle_lateral",
            "lateral_down",
            "legacy_guided",
            "auto_try_all",
            "auto_general",
        ],
        default="auto_general",
        help="Named semantic grasp strategy passed to the Agent adapter.",
    )
    parser.add_argument(
        "--require_public_grasp_action",
        action="store_true",
        default=False,
        help="Fail instead of falling back when public grasp cannot run.",
    )
    parser.add_argument(
        "--allow_public_grasp_annotation",
        action="store_true",
        default=False,
        help="Allow viser annotation if semantic grasp cache is missing.",
    )
    parser.add_argument(
        "--force_public_grasp_reannotate",
        action="store_true",
        default=False,
        help="Force viser re-annotation for semantic grasp.",
    )
    parser.add_argument(
        "--public_grasp_candidate_num",
        type=int,
        default=32,
        help="Number of semantic grasp candidates to retry.",
    )
    parser.add_argument(
        "--public_grasp_pre_grasp_distance",
        type=float,
        default=None,
        help="Optional public PickUpActionCfg pre_grasp_distance override.",
    )
    parser.add_argument(
        "--generate_public_grasp_candidates",
        action="store_true",
        default=False,
        help="Generate whole-mesh antipodal candidates without opening viser.",
    )
    parser.add_argument(
        "--public_grasp_auto_approach_direction",
        action="store_true",
        default=False,
        help="Use the horizontal arm-base-to-object direction for semantic grasp.",
    )
    parser.add_argument(
        "--public_grasp_try_approach_directions",
        action="store_true",
        default=False,
        help=(
            "Try multiple semantic grasp approach directions before falling back "
            "or failing in strict mode, including arm-relative and object-local sides."
        ),
    )
    parser.add_argument(
        "--public_grasp_approach_direction",
        type=_parse_xyz,
        default=None,
        help="Explicit semantic grasp approach direction, e.g. 0,0,-1.",
    )
    parser.add_argument(
        "--public_grasp_approach_directions",
        type=_parse_xyz_list,
        default=None,
        help=(
            "Semicolon-separated semantic grasp approach directions, "
            "e.g. 0,0,-1;1,0,0. Overrides auto/multi-direction defaults."
        ),
    )
    parser.add_argument(
        "--public_grasp_lift_height",
        type=float,
        default=None,
        help=(
            "Optional lift_height passed directly to PickUpActionCfg. "
            "When omitted, semantic grasp uses a short lift and the demo's "
            "separate Agent lift step performs the main raise."
        ),
    )
    parser.add_argument(
        "--public_grasp_pose_offset_world",
        type=_parse_xyz,
        default=[0.0, 0.0, 0.0],
        help="Optional world-frame xyz offset applied to public semantic grasp poses.",
    )
    parser.add_argument(
        "--public_grasp_pose_offset_along_approach",
        type=float,
        default=0.0,
        help="Optional offset along public semantic grasp approach direction.",
    )
    parser.add_argument(
        "--validate_public_grasp_after_action",
        action="store_true",
        default=False,
        help=(
            "Validate object lift immediately after a public PickUpActionCfg grasp. "
            "Use with --public_grasp_lift_height for same-action validation."
        ),
    )
    parser.add_argument(
        "--public_grasp_validation_min_object_lift",
        type=float,
        default=0.0,
        help="Minimum object z displacement for adapter-level public grasp validation.",
    )
    parser.add_argument(
        "--public_grasp_validation_max_object_lift",
        type=float,
        default=None,
        help="Optional maximum object z displacement for adapter-level validation.",
    )
    parser.add_argument(
        "--public_grasp_validation_max_object_xy_displacement",
        type=float,
        default=None,
        help="Optional maximum object xy displacement for adapter-level validation.",
    )
    parser.add_argument(
        "--public_grasp_rank_by_legacy_pose",
        action="store_true",
        default=False,
        help="Rank planned semantic grasp candidates by legacy grasp_pose_obj similarity.",
    )
    parser.add_argument(
        "--public_grasp_use_legacy_orientation",
        action="store_true",
        default=False,
        help="Use semantic grasp centers with legacy grasp_pose_obj orientation.",
    )
    parser.add_argument(
        "--public_grasp_legacy_pose_position_weight",
        type=float,
        default=1.0,
        help="Position weight for legacy-pose semantic grasp candidate scoring.",
    )
    parser.add_argument(
        "--public_grasp_legacy_pose_rotation_weight",
        type=float,
        default=0.05,
        help="Rotation weight for legacy-pose semantic grasp candidate scoring.",
    )
    parser.add_argument(
        "--public_grasp_legacy_pose_max_position_error",
        type=float,
        default=None,
        help="Optional maximum candidate position error against legacy grasp pose.",
    )
    parser.add_argument(
        "--public_grasp_legacy_pose_max_rotation_error",
        type=float,
        default=None,
        help="Optional maximum candidate rotation error against legacy grasp pose.",
    )
    parser.add_argument(
        "--public_grasp_validate_relative_to_legacy_pose",
        action="store_true",
        default=False,
        help="Validate object-in-EEF pose after grasp against legacy grasp reference.",
    )
    parser.add_argument(
        "--public_grasp_max_legacy_relative_pos_error",
        type=float,
        default=0.08,
        help="Maximum post-grasp object-in-EEF position error against legacy reference.",
    )
    parser.add_argument(
        "--public_grasp_max_legacy_relative_rot_error",
        type=float,
        default=0.7,
        help="Maximum post-grasp object-in-EEF rotation error against legacy reference.",
    )
    parser.add_argument("--grasp_max_open_length", type=float, default=0.088)
    parser.add_argument("--grasp_min_open_length", type=float, default=0.003)
    parser.add_argument("--grasp_finger_length", type=float, default=0.078)
    parser.add_argument("--grasp_x_thickness", type=float, default=0.01)
    parser.add_argument("--grasp_y_thickness", type=float, default=0.03)
    parser.add_argument("--grasp_root_z_width", type=float, default=0.08)
    parser.add_argument("--grasp_open_check_margin", type=float, default=0.01)
    parser.add_argument("--grasp_point_sample_dense", type=float, default=0.012)
    parser.add_argument("--grasp_antipodal_n_sample", type=int, default=20000)
    parser.add_argument(
        "--grasp_max_deviation_angle", type=float, default=float(np.pi / 6)
    )
    parser.add_argument(
        "--disable_public_place_action",
        action="store_true",
        default=False,
        help="Disable public PlaceActionCfg for Agent place_on_table.",
    )
    parser.add_argument(
        "--disable_public_place_upright",
        action="store_true",
        default=False,
        help=(
            "Disable object-upright target pose construction for public "
            "place_on_table."
        ),
    )
    parser.add_argument(
        "--public_place_upright_eps",
        type=float,
        default=0.0,
        help="Extra object-center height offset for public upright placement.",
    )
    parser.add_argument(
        "--public_place_post_open_wait_steps",
        type=int,
        default=20,
        help="Hold waypoints inserted after gripper opening before retreating.",
    )
    parser.add_argument(
        "--validate_public_place_after_action",
        action="store_true",
        default=False,
        help=(
            "Validate the released object's final upright state after "
            "place_on_table. Disabled by default because this demo mirrors "
            "atomic_actions.py's EEF-place target rather than a task-level "
            "upright-object success condition."
        ),
    )
    parser.add_argument(
        "--disable_final_place_physical_validation",
        action="store_true",
        default=False,
        help="Skip the final object-upright validation at the end of the demo.",
    )
    parser.add_argument(
        "--final_place_validation_min_upright_dot",
        type=float,
        default=0.65,
        help="Minimum world-z alignment required for the placed cup.",
    )
    parser.add_argument(
        "--final_place_validation_max_height_drop",
        type=float,
        default=0.08,
        help="Maximum final cup center-height drop from the initial upright pose.",
    )
    parser.add_argument(
        "--disable_public_gripper_action",
        action="store_true",
        default=False,
        help="Disable public GripperActionCfg for Agent open/close gripper.",
    )
    parser.add_argument(
        "--require_public_non_grasp_actions",
        action="store_true",
        default=False,
        help=(
            "Fail if non-grasp Agent skills cannot use public atomic actions "
            "instead of falling back to legacy planning."
        ),
    )
    return parser.parse_args()


def _arm_solver_cfg() -> PytorchSolverCfg:
    return PytorchSolverCfg(
        end_link_name="ee_link",
        root_link_name="base_link",
        tcp=[
            [0.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.12],
            [0.0, 0.0, 0.0, 1.0],
        ],
    )


def create_agent_skill_robot(
    sim: SimulationManager,
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Robot:
    """Create the same UR10/DH gripper as atomic_actions.py, plus Agent aliases."""
    ur10_urdf_path = get_data_path("UniversalRobots/UR10/UR10.urdf")
    gripper_urdf_path = get_data_path("DH_PGC_140_50_M/DH_PGC_140_50_M.urdf")
    cfg = RobotCfg(
        uid="UR10",
        urdf_cfg=URDFCfg(
            components=[
                {"component_type": "arm", "urdf_path": ur10_urdf_path},
                {"component_type": "hand", "urdf_path": gripper_urdf_path},
            ]
        ),
        drive_pros=JointDrivePropertiesCfg(
            stiffness={"JOINT[0-9]": 1e4, "FINGER[1-2]": 1e2},
            damping={"JOINT[0-9]": 1e3, "FINGER[1-2]": 1e1},
            max_effort={"JOINT[0-9]": 1e5, "FINGER[1-2]": 1e3},
            drive_type="force",
        ),
        control_parts={
            "arm": ["JOINT[0-9]"],
            "hand": ["FINGER[1-2]"],
            "left_arm": ["JOINT[0-9]"],
            "left_eef": ["FINGER[1-2]"],
        },
        solver_cfg={
            "arm": _arm_solver_cfg(),
            "left_arm": _arm_solver_cfg(),
        },
        init_qpos=[
            0.0,
            -np.pi / 2,
            -np.pi / 2,
            np.pi / 2,
            -np.pi / 2,
            0.0,
            HAND_OPEN,
            HAND_OPEN,
        ],
        init_pos=position,
    )
    return sim.add_robot(cfg=cfg)


def create_camera(sim: SimulationManager) -> Camera:
    return sim.add_sensor(
        sensor_cfg=CameraCfg(
            uid="cam1",
            width=640,
            height=480,
            intrinsics=(600, 600, 320.0, 240.0),
            extrinsics=CameraCfg.ExtrinsicsCfg(
                eye=(1.15, -0.75, 0.85),
                target=(0.45, 0.0, 0.12),
                up=(0.0, 0.0, 1.0),
            ),
            near=0.01,
            far=10.0,
            enable_color=True,
        )
    )


def capture_frame(sim: SimulationManager, camera: Camera) -> np.ndarray:
    if camera.is_rt_enabled:
        sim.render_camera_group([camera.group_id])
    camera.update(fetch_only=camera.is_rt_enabled)
    rgb = camera.get_data()["color"][0, :, :, :3]
    return rgb.detach().cpu().numpy()


def _frames_have_visible_content(frames: list[np.ndarray]) -> bool:
    if not frames:
        return False
    sample_count = min(10, len(frames))
    sample_indices = np.linspace(0, len(frames) - 1, sample_count, dtype=int)
    for idx in sample_indices:
        frame = np.asarray(frames[idx])
        if frame.size > 0 and float(frame.mean()) > 1.0 and int(frame.max()) > 8:
            return True
    return False


def _cup_side_grasp_pose(device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [
            [0.32039, -0.03227, 0.94673, 0.0],
            [0.00675, -0.99932, -0.03635, 0.0],
            [0.94726, 0.01803, -0.31996, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )


class AgentSkillSimEnvAdapter:
    """Minimum AgentEnv-like surface needed by Agent atomic skills."""

    def __init__(
        self,
        sim: SimulationManager,
        robot: Robot,
        camera: Camera,
        *,
        obj_name: str,
        place_z: float,
    ) -> None:
        self.sim = sim
        self.robot = robot
        self.camera = camera
        self.obj_name = obj_name
        self.place_z = place_z
        self.frames: list = []

        self.init_qpos = robot.get_qpos().squeeze(0).detach().clone()
        self.left_arm_joints = robot.get_joint_ids("left_arm")
        self.left_eef_joints = robot.get_joint_ids("left_eef")
        self.right_arm_joints: list[int] = []
        self.right_eef_joints: list[int] = []

        self.open_state = torch.tensor([HAND_OPEN], dtype=torch.float32)
        self.close_state = torch.tensor([HAND_CLOSE], dtype=torch.float32)

        self.left_arm_init_qpos = robot.get_qpos("left_arm").squeeze(0).detach().clone()
        self.left_arm_current_qpos = self.left_arm_init_qpos.clone()
        self.left_arm_init_xpos = self.get_arm_fk(self.left_arm_init_qpos, is_left=True)
        self.left_arm_current_xpos = self.left_arm_init_xpos.clone()
        self.left_arm_current_gripper_state = self.open_state.clone()
        self.left_arm_base_pose = robot.get_control_part_base_pose(
            "left_arm", to_matrix=True
        ).squeeze(0)

        self.right_arm_init_qpos = torch.empty(
            0, dtype=torch.float32, device=robot.device
        )
        self.right_arm_current_qpos = self.right_arm_init_qpos.clone()
        self.right_arm_init_xpos = torch.eye(
            4, dtype=torch.float32, device=robot.device
        )
        self.right_arm_current_xpos = self.right_arm_init_xpos.clone()
        self.right_arm_current_gripper_state = self.open_state.clone()
        self.right_arm_base_pose = self.left_arm_base_pose.clone()

        self.obj_info: dict[str, dict] = {}
        self._initial_object_pose: torch.Tensor | None = None
        self.update_obj_info()

    def update_obj_info(self) -> None:
        obj_pose = self.get_object_pose()
        if self._initial_object_pose is None:
            self._initial_object_pose = obj_pose.detach().clone()
        self.obj_info[self.obj_name] = {
            "height": float(self.place_z),
            "place_object_height": float(obj_pose[2, 3].item()),
            "initial_pose": self._initial_object_pose.detach().clone(),
            "grasp_pose_obj": _cup_side_grasp_pose(self.robot.device),
        }

    def get_arm_fk(self, qpos, is_left: bool = True) -> torch.Tensor:
        if not is_left:
            raise ValueError("grasp_cup.py uses a single UR10 arm mapped to left_arm.")
        qpos_tensor = torch.as_tensor(
            qpos, dtype=torch.float32, device=self.robot.device
        )
        pose = self.robot.compute_fk(qpos=qpos_tensor, name="left_arm", to_matrix=True)
        return pose.squeeze(0).detach().clone()

    def get_arm_ik(self, target_xpos, is_left: bool = True, qpos_seed=None):
        if not is_left:
            raise ValueError("grasp_cup.py uses a single UR10 arm mapped to left_arm.")
        target_pose = torch.as_tensor(
            target_xpos, dtype=torch.float32, device=self.robot.device
        )
        seed = None
        if qpos_seed is not None:
            seed = torch.as_tensor(
                qpos_seed, dtype=torch.float32, device=self.robot.device
            )
        ret, qpos = self.robot.compute_ik(
            pose=target_pose,
            joint_seed=seed,
            name="left_arm",
        )
        return bool(torch.all(ret).item()), qpos.squeeze(0).detach().clone()

    def set_current_qpos_agent(self, qpos, is_left: bool = True) -> None:
        if not is_left:
            return
        self.left_arm_current_qpos = torch.as_tensor(
            qpos, dtype=torch.float32, device=self.robot.device
        ).flatten()

    def set_current_xpos_agent(self, xpos, is_left: bool = True) -> None:
        if not is_left:
            return
        self.left_arm_current_xpos = torch.as_tensor(
            xpos, dtype=torch.float32, device=self.robot.device
        ).clone()

    def set_current_gripper_state_agent(
        self, gripper_state, is_left: bool = True
    ) -> None:
        if not is_left:
            return
        self.left_arm_current_gripper_state = torch.as_tensor(
            gripper_state, dtype=torch.float32, device=self.robot.device
        ).flatten()

    def get_current_qpos_agent(self, is_left: bool | None = None):
        if is_left is None:
            return self.left_arm_current_qpos, self.right_arm_current_qpos
        return self.left_arm_current_qpos if is_left else self.right_arm_current_qpos

    def get_current_xpos_agent(self, is_left: bool | None = None):
        if is_left is None:
            return self.left_arm_current_xpos, self.right_arm_current_xpos
        return self.left_arm_current_xpos if is_left else self.right_arm_current_xpos

    def get_current_gripper_state_agent(self, is_left: bool | None = None):
        if is_left is None:
            return (
                self.left_arm_current_gripper_state,
                self.right_arm_current_gripper_state,
            )
        return (
            self.left_arm_current_gripper_state
            if is_left
            else self.right_arm_current_gripper_state
        )

    def get_object_pose(self) -> torch.Tensor:
        obj = self.sim.get_rigid_object(self.obj_name)
        return obj.get_local_pose(to_matrix=True).squeeze(0).detach().clone()

    def apply_agent_action(self, action: np.ndarray) -> int:
        action_np = np.asarray(action, dtype=np.float32)
        joint_ids = self.left_arm_joints + self.left_eef_joints
        if action_np.ndim != 2:
            raise ValueError(f"Agent skill action must be 2-D, got {action_np.shape}.")
        if action_np.shape[1] != len(joint_ids):
            raise ValueError(
                "Agent skill action width does not match UR10 arm+hand joints: "
                f"{action_np.shape[1]} vs {len(joint_ids)}."
            )

        for row in action_np:
            full_qpos = self.robot.get_qpos().clone()
            full_qpos[:, joint_ids] = torch.as_tensor(
                row, dtype=torch.float32, device=self.robot.device
            ).reshape(1, -1)
            self.robot.set_qpos(full_qpos)
            self.sim.update(step=4)
            self.frames.append(capture_frame(self.sim, self.camera))
            time.sleep(1e-3)
        return len(action_np)

    def hold(self, seconds: float) -> None:
        for _ in range(max(1, int(round(seconds * FPS)))):
            self.sim.update(step=1)
            self.frames.append(capture_frame(self.sim, self.camera))


def _use_public_grasp_semantics_from_args(args: argparse.Namespace) -> bool:
    return bool(
        args.use_public_grasp_semantics
        or args.public_grasp_strategy is not None
        or args.public_grasp_rank_by_legacy_pose
        or args.public_grasp_use_legacy_orientation
        or args.public_grasp_legacy_pose_max_position_error is not None
        or args.public_grasp_legacy_pose_max_rotation_error is not None
    )


def _skill_kwargs(args: argparse.Namespace) -> dict:
    use_public_grasp_semantics = _use_public_grasp_semantics_from_args(args)
    public_grasp_pre_grasp_distance = args.public_grasp_pre_grasp_distance
    if public_grasp_pre_grasp_distance is None and use_public_grasp_semantics:
        public_grasp_pre_grasp_distance = 0.05
    public_grasp_lift_height = args.public_grasp_lift_height
    if public_grasp_lift_height is None and use_public_grasp_semantics:
        public_grasp_lift_height = min(args.lift_dz, 0.03)

    return {
        "use_public_atomic_actions": not args.disable_public_atomic_actions,
        "use_public_grasp_action": (
            not args.disable_public_grasp_action
            or use_public_grasp_semantics
            or args.require_public_grasp_action
        ),
        "use_public_grasp_semantics": use_public_grasp_semantics,
        "require_public_grasp_action": args.require_public_grasp_action,
        "allow_public_grasp_annotation": args.allow_public_grasp_annotation,
        "force_public_grasp_reannotate": args.force_public_grasp_reannotate,
        "public_grasp_strategy": args.public_grasp_strategy,
        "public_grasp_candidate_num": args.public_grasp_candidate_num,
        "public_grasp_pre_grasp_distance": public_grasp_pre_grasp_distance,
        "generate_public_grasp_candidates": args.generate_public_grasp_candidates,
        "public_grasp_auto_approach_direction": (
            args.public_grasp_auto_approach_direction
        ),
        "public_grasp_try_approach_directions": (
            args.public_grasp_try_approach_directions
        ),
        "public_grasp_approach_direction": args.public_grasp_approach_direction,
        "public_grasp_approach_directions": args.public_grasp_approach_directions,
        "public_grasp_lift_height": public_grasp_lift_height,
        "public_grasp_pose_offset_world": args.public_grasp_pose_offset_world,
        "public_grasp_pose_offset_along_approach": (
            args.public_grasp_pose_offset_along_approach
        ),
        "validate_public_grasp_after_action": args.validate_public_grasp_after_action,
        "public_grasp_validation_min_object_lift": (
            args.public_grasp_validation_min_object_lift
        ),
        "public_grasp_validation_max_object_lift": (
            args.public_grasp_validation_max_object_lift
        ),
        "public_grasp_validation_max_object_xy_displacement": (
            args.public_grasp_validation_max_object_xy_displacement
        ),
        "public_grasp_rank_by_legacy_pose": args.public_grasp_rank_by_legacy_pose,
        "public_grasp_use_legacy_orientation": (
            args.public_grasp_use_legacy_orientation
        ),
        "public_grasp_legacy_pose_position_weight": (
            args.public_grasp_legacy_pose_position_weight
        ),
        "public_grasp_legacy_pose_rotation_weight": (
            args.public_grasp_legacy_pose_rotation_weight
        ),
        "public_grasp_legacy_pose_max_position_error": (
            args.public_grasp_legacy_pose_max_position_error
        ),
        "public_grasp_legacy_pose_max_rotation_error": (
            args.public_grasp_legacy_pose_max_rotation_error
        ),
        "public_grasp_validate_relative_to_legacy_pose": (
            args.public_grasp_validate_relative_to_legacy_pose
        ),
        "public_grasp_max_legacy_relative_pos_error": (
            args.public_grasp_max_legacy_relative_pos_error
        ),
        "public_grasp_max_legacy_relative_rot_error": (
            args.public_grasp_max_legacy_relative_rot_error
        ),
        "grasp_max_open_length": args.grasp_max_open_length,
        "grasp_min_open_length": args.grasp_min_open_length,
        "grasp_finger_length": args.grasp_finger_length,
        "grasp_x_thickness": args.grasp_x_thickness,
        "grasp_y_thickness": args.grasp_y_thickness,
        "grasp_root_z_width": args.grasp_root_z_width,
        "grasp_open_check_margin": args.grasp_open_check_margin,
        "grasp_point_sample_dense": args.grasp_point_sample_dense,
        "grasp_antipodal_n_sample": args.grasp_antipodal_n_sample,
        "grasp_max_deviation_angle": args.grasp_max_deviation_angle,
        "use_public_place_action": not args.disable_public_place_action,
        "public_place_upright": not args.disable_public_place_upright,
        "public_place_upright_eps": args.public_place_upright_eps,
        "public_place_post_open_wait_steps": args.public_place_post_open_wait_steps,
        "validate_place_preconditions": True,
        "validate_public_place_after_action": args.validate_public_place_after_action,
        "use_public_gripper_action": not args.disable_public_gripper_action,
        "require_public_non_grasp_actions": args.require_public_non_grasp_actions,
        "log_dir": getattr(args, "public_grasp_log_dir", None),
        "sample_num": 45,
    }


def _build_skill_steps(
    env: AgentSkillSimEnvAdapter, args: argparse.Namespace
) -> list[SkillStep]:
    return [
        SkillStep(
            "grasp",
            partial(
                grasp,
                robot_name="left_arm",
                obj_name=env.obj_name,
                pre_grasp_dis=args.pre_grasp_dis,
            ),
        ),
        SkillStep(
            "lift",
            partial(
                move_by_relative_offset,
                robot_name="left_arm",
                dx=0.0,
                dy=0.0,
                dz=args.lift_dz,
                mode="extrinsic",
                sample_num=25,
            ),
        ),
        SkillStep(
            "move_above_place",
            partial(
                move_to_absolute_position,
                robot_name="left_arm",
                x=args.place_x,
                y=args.place_y,
                z=args.place_z + args.lift_dz,
                sample_num=35,
            ),
        ),
        SkillStep(
            "place_on_table",
            partial(
                place_on_table,
                robot_name="left_arm",
                obj_name=env.obj_name,
                x=args.place_x,
                y=args.place_y,
                pre_place_dis=args.pre_place_dis,
                eps=0.0,
            ),
        ),
        SkillStep(
            "back_to_initial_pose",
            partial(back_to_initial_pose, robot_name="left_arm", sample_num=45),
        ),
    ]


def _execute_skill_step(
    env: AgentSkillSimEnvAdapter,
    args: argparse.Namespace,
    step: SkillStep,
) -> dict[str, str]:
    logger.log_info(f"Running Agent atomic skill: {step.name}.", color="cyan")
    skill_kwargs = _skill_kwargs(args)
    action = step.action(env=env, **skill_kwargs)
    step_count = env.apply_agent_action(action)
    validate_pending_public_grasp_after_action(env, skill_kwargs)
    validate_pending_public_place_after_action(env, skill_kwargs)
    return {"skill": step.name, "steps": str(step_count), "status": "ok"}


def _validate_grasp_lift_result(
    env: AgentSkillSimEnvAdapter,
    args: argparse.Namespace,
    post_grasp_pose: torch.Tensor,
) -> dict[str, str]:
    if args.disable_grasp_physical_validation:
        return {"skill": "grasp_physical_validation", "steps": "0", "status": "skipped"}

    after_pose = env.get_object_pose()
    object_lift = float((after_pose[2, 3] - post_grasp_pose[2, 3]).item())
    status = (
        f"ok object_lift={object_lift:.4f}m"
        if object_lift >= args.grasp_validation_min_object_lift
        else f"failed object_lift={object_lift:.4f}m"
    )
    if object_lift < args.grasp_validation_min_object_lift:
        raise RuntimeError(
            "Post-grasp physical validation failed: "
            f"object_lift={object_lift:.4f}m, "
            f"required>={args.grasp_validation_min_object_lift:.4f}m."
        )
    return {"skill": "grasp_physical_validation", "steps": "0", "status": status}


def _validate_final_place_result(
    env: AgentSkillSimEnvAdapter,
    args: argparse.Namespace,
) -> dict[str, str]:
    if args.disable_final_place_physical_validation:
        return {
            "skill": "final_place_physical_validation",
            "steps": "0",
            "status": "skipped",
        }

    pose = env.get_object_pose()
    initial_pose = env._initial_object_pose
    vertical_alignment = float(torch.abs(pose[:3, 2][2]).item())
    height_drop = 0.0
    if initial_pose is not None:
        height_drop = float(initial_pose[2, 3].item() - pose[2, 3].item())

    status = (
        "ok "
        if (
            vertical_alignment >= args.final_place_validation_min_upright_dot
            and height_drop <= args.final_place_validation_max_height_drop
        )
        else "failed "
    )
    status += (
        f"vertical_alignment={vertical_alignment:.4f} "
        f"height_drop={height_drop:.4f}m"
    )
    if vertical_alignment < args.final_place_validation_min_upright_dot:
        raise RuntimeError(
            "Final place physical validation failed: "
            f"vertical_alignment={vertical_alignment:.4f}, "
            f"required>={args.final_place_validation_min_upright_dot:.4f}."
        )
    if height_drop > args.final_place_validation_max_height_drop:
        raise RuntimeError(
            "Final place physical validation failed: "
            f"height_drop={height_drop:.4f}m, "
            f"allowed<={args.final_place_validation_max_height_drop:.4f}m."
        )
    return {
        "skill": "final_place_physical_validation",
        "steps": "0",
        "status": status,
    }


def _pad_video(env: AgentSkillSimEnvAdapter, min_seconds: float) -> None:
    target_frames = max(1, int(round(min_seconds * FPS)))
    missing_frames = target_frames - len(env.frames)
    if missing_frames <= 0:
        return
    env.hold(missing_frames / FPS)


def run_demo(args: argparse.Namespace, output_root: Path) -> Path:
    if args.num_envs != 1:
        raise ValueError(
            "Agent skill grasp_cup.py currently supports --num_envs 1 only."
        )

    output_root.mkdir(parents=True, exist_ok=True)
    args.public_grasp_log_dir = str(output_root)
    video_dir = output_root / "outputs/videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "summary.tsv"

    sim = initialize_simulation(args)
    robot = create_agent_skill_robot(sim)
    mug = create_mug(sim)
    camera = create_camera(sim)
    sim.init_gpu_physics()
    if args.open_window:
        sim.open_window()

    env = AgentSkillSimEnvAdapter(
        sim=sim,
        robot=robot,
        camera=camera,
        obj_name=mug.uid,
        place_z=args.place_z,
    )
    env.hold(0.5)

    rows: list[dict[str, str]] = []
    video_path = video_dir / "episode_0_cam1.mp4"
    video_status = "not_recorded"
    run_error: Exception | None = None
    try:
        try:
            post_grasp_pose: torch.Tensor | None = None
            for step in _build_skill_steps(env, args):
                rows.append(_execute_skill_step(env, args, step))
                if step.name == "grasp":
                    post_grasp_pose = env.get_object_pose()
                elif step.name == "lift":
                    if post_grasp_pose is None:
                        raise RuntimeError(
                            "Cannot validate grasp physics before the grasp step runs."
                        )
                    rows.append(_validate_grasp_lift_result(env, args, post_grasp_pose))
            rows.append(_validate_final_place_result(env, args))
        except Exception as exc:
            run_error = exc
            rows.append(
                {
                    "skill": "error",
                    "steps": type(exc).__name__,
                    "status": str(exc),
                }
            )
        finally:
            _pad_video(env, args.min_video_seconds)
            if env.frames:
                images_to_video(env.frames, str(video_dir), "episode_0_cam1", fps=FPS)
                video_status = (
                    "ok"
                    if _frames_have_visible_content(env.frames)
                    else "black_frames_detected"
                )
    finally:
        sim.destroy()

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["skill", "steps", "status"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(
            {
                "skill": "scene",
                "steps": "UR10 + DH_PGC_140_50_M + CoffeeCup/cup.ply",
                "status": "matches scripts/tutorials/sim/atomic_actions.py",
            }
        )
        writer.writerow(
            {
                "skill": "video",
                "steps": str(video_path),
                "status": video_status,
            }
        )

    logger.log_info(f"Agent skill grasp-cup demo finished. Video: {video_path}")
    logger.log_info(f"Summary: {summary_path}")
    if run_error is not None:
        raise run_error
    return summary_path


def main() -> None:
    args = parse_args()
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else timestamped_output_root()
    )
    run_demo(args, output_root)


if __name__ == "__main__":
    main()
