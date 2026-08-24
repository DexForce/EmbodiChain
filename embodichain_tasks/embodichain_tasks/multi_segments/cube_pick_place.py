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

"""Repeated cube pick-and-place task using lazy demonstration segments.

Each segment plans one complete ``PickUp -> Place -> settle`` cycle. The outer
segment generator resumes only after the previous segment has executed and its
free-falling cube has settled. Consequently, the next pickup always plans from
the cube pose currently measured in simulation instead of a pose predicted
before the episode started.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from functools import partial
from typing import TYPE_CHECKING, Any

import torch

from embodichain.lab.gym.envs import DemoSegment, EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.utils.registration import register_env
from embodichain.lab.sim.cfg import (
    LightCfg,
    RigidBodyAttributesCfg,
    RigidObjectCfg,
)
from embodichain.lab.sim.robots import URRobotCfg
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.utils import logger

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import RigidObject
    from embodichain.lab.sim.planners import MotionGenerator
    from embodichain.toolkits.graspkit import GraspPoseGenerator

__all__ = ["MultiSegmentsCubePickPlaceEnv"]

CUBE_UID = "cube"
CUBE_SIZE = 0.05
DEFAULT_NUM_CYCLES = 3
DEFAULT_GRASP_HOLD_STEPS = 45
DEFAULT_PLACE_POSITIONS = (
    (-0.40, 0.48, 0.10),
    (-0.42, -0.08, 0.10),
)

GRIPPER_URDF_PATH = "DH_PGI_140_80/DH_PGI_140_80.urdf"
GRIPPER_HAND_JOINT_PATTERN = "gripper_finger1_joint_1"
GRIPPER_TCP_Z = 0.15
GRIPPER_MAX_OPEN_WIDTH = 0.100
GRIPPER_FINGER_LENGTH = 0.12
GRIPPER_ROOT_Z_WIDTH = 0.096
GRIPPER_Y_THICKNESS = 0.040
DEFAULT_GRIPPER_CLOSE_QPOS = 0.024


def _create_default_grasp_pose_generator() -> GraspPoseGenerator:
    """Create the task's typed, standalone parallel-jaw grasp service."""
    from embodichain.toolkits.graspkit import ParallelJawGripperModelCfg
    from embodichain.toolkits.graspkit.pg_grasp import (
        AntipodalGraspPoseGenerator,
        AntipodalGraspPoseGeneratorCfg,
        GraspAnnotationCfg,
        ParallelJawGraspCollisionCfg,
    )

    return AntipodalGraspPoseGenerator(
        ParallelJawGripperModelCfg(
            model_id="dh_pgi_140_80",
            min_opening_width=0.005,
            max_opening_width=GRIPPER_MAX_OPEN_WIDTH,
            finger_length=GRIPPER_FINGER_LENGTH,
            finger_width=GRIPPER_Y_THICKNESS,
            finger_thickness=0.01,
            palm_depth=GRIPPER_ROOT_Z_WIDTH,
        ),
        algorithm_cfg=AntipodalGraspPoseGeneratorCfg(sample_count=10_000),
        collision_cfg=ParallelJawGraspCollisionCfg(
            opening_margin=0.002,
            point_sample_density=0.012,
            filter_ground_collision=False,
        ),
        annotation_cfg=GraspAnnotationCfg(
            selection_mode="whole_mesh",
            viser_port=11801,
        ),
    )


def _create_default_robot_cfg() -> URRobotCfg:
    """Create the UR5 and parallel gripper used by the direct-planning demo."""
    return URRobotCfg.from_dict(
        {
            "robot_type": "ur5",
            "uid": "UR5",
            "urdf_cfg": {
                "components": [
                    {
                        "component_type": "hand",
                        "urdf_path": GRIPPER_URDF_PATH,
                    },
                ],
            },
            "control_parts": {
                "hand": [GRIPPER_HAND_JOINT_PATTERN],
            },
            "drive_pros": {
                "stiffness": {
                    GRIPPER_HAND_JOINT_PATTERN: 1e3,
                },
                "damping": {
                    GRIPPER_HAND_JOINT_PATTERN: 1e2,
                },
                "max_effort": {
                    GRIPPER_HAND_JOINT_PATTERN: 1e4,
                },
            },
            "solver_cfg": {
                "arm": {
                    "tcp": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, GRIPPER_TCP_Z],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                }
            },
            "init_qpos": [0.0, -1.57, 1.57, -1.57, -1.57, 0.0, 0.0, 0.0],
        }
    )


def _create_default_env_cfg() -> EmbodiedEnvCfg:
    """Create a directly-instantiable default task configuration."""
    cfg = EmbodiedEnvCfg()
    cfg.max_episode_steps = 1200
    cfg.robot = _create_default_robot_cfg()
    cfg.light = EmbodiedEnvCfg.EnvLightCfg(
        direct=[
            LightCfg(
                uid="main_light",
                color=(0.6, 0.6, 0.6),
                intensity=30.0,
                init_pos=(1.0, 0.0, 3.0),
            )
        ]
    )
    cfg.rigid_object = [
        RigidObjectCfg(
            uid=CUBE_UID,
            shape=CubeCfg(size=[CUBE_SIZE, CUBE_SIZE, CUBE_SIZE]),
            attrs=RigidBodyAttributesCfg(
                mass=0.05,
                dynamic_friction=0.97,
                static_friction=0.99,
                enable_ccd=True,
                linear_damping=0.2,
                angular_damping=0.2,
            ),
            max_convex_hull_num=16,
            init_pos=(-0.42, -0.08, 0.5 * CUBE_SIZE),
        )
    ]
    cfg.extensions = {
        "num_cycles": DEFAULT_NUM_CYCLES,
        "place_positions": [list(position) for position in DEFAULT_PLACE_POSITIONS],
        "grasp_hold_steps": DEFAULT_GRASP_HOLD_STEPS,
        "settle_min_steps": 15,
        "settle_max_steps": 80,
        "settle_stable_steps": 5,
        "linear_velocity_threshold": 0.03,
        "angular_velocity_threshold": 0.20,
        "place_position_tolerance": 0.12,
    }
    return cfg


@register_env("MultiSegmentsCubePickPlace-v1", max_episode_steps=1200)
class MultiSegmentsCubePickPlaceEnv(EmbodiedEnv):
    """Repeatedly pick up and freely place one cube.

    The demonstration planner is intentionally lazy. It yields one complete
    pick/place cycle at a time, waits for that cycle to execute and settle, and
    only then reads the cube pose and plans the following cycle.
    """

    PICK_SAMPLE_INTERVAL = 120
    PLACE_SAMPLE_INTERVAL = 120
    HAND_INTERP_STEPS = 12
    PRE_GRASP_DISTANCE = 0.15
    PICK_LIFT_HEIGHT = 0.16
    PLACE_RETRACT_HEIGHT = 0.14

    def __init__(
        self,
        cfg: EmbodiedEnvCfg | None = None,
        *,
        grasp_pose_generator: GraspPoseGenerator | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the task with an optional standalone grasp service.

        Args:
            cfg: Gym environment and scene configuration.
            grasp_pose_generator: Optional service override for direct Python
                assembly. When omitted, the task creates its typed antipodal
                parallel-jaw default outside the Gym ``extensions`` payload.
            **kwargs: Additional base-environment arguments.
        """
        if grasp_pose_generator is not None:
            from embodichain.toolkits.graspkit import GraspPoseGenerator

            if not isinstance(grasp_pose_generator, GraspPoseGenerator):
                raise TypeError(
                    "grasp_pose_generator must be a GraspPoseGenerator or None."
                )
        self._grasp_pose_generator = grasp_pose_generator
        if cfg is None:
            cfg = _create_default_env_cfg()

        extensions = getattr(cfg, "extensions", {}) or {}
        self.num_cycles = int(extensions.get("num_cycles", DEFAULT_NUM_CYCLES))
        self.place_positions = self._validate_place_positions(
            extensions.get("place_positions", DEFAULT_PLACE_POSITIONS)
        )
        self.grasp_hold_steps = int(
            extensions.get("grasp_hold_steps", DEFAULT_GRASP_HOLD_STEPS)
        )
        self.settle_min_steps = int(extensions.get("settle_min_steps", 15))
        self.settle_max_steps = int(extensions.get("settle_max_steps", 80))
        self.settle_stable_steps = int(extensions.get("settle_stable_steps", 5))
        self.linear_velocity_threshold = float(
            extensions.get("linear_velocity_threshold", 0.03)
        )
        self.angular_velocity_threshold = float(
            extensions.get("angular_velocity_threshold", 0.20)
        )
        self.place_position_tolerance = float(
            extensions.get("place_position_tolerance", 0.12)
        )
        self._validate_settings()

        super().__init__(cfg, **kwargs)

        # ``EmbodiedEnv`` exposes extension values as instance attributes.
        # Re-normalize them because that binding intentionally preserves the
        # JSON-native list/scalar types supplied by the launcher.
        self.num_cycles = int(self.num_cycles)
        self.place_positions = self._validate_place_positions(self.place_positions)
        self.grasp_hold_steps = int(self.grasp_hold_steps)
        self.settle_min_steps = int(self.settle_min_steps)
        self.settle_max_steps = int(self.settle_max_steps)
        self.settle_stable_steps = int(self.settle_stable_steps)
        self.linear_velocity_threshold = float(self.linear_velocity_threshold)
        self.angular_velocity_threshold = float(self.angular_velocity_threshold)
        self.place_position_tolerance = float(self.place_position_tolerance)
        self._validate_settings()

        cube = self.sim.get_rigid_object(CUBE_UID)
        if cube is None:
            raise RuntimeError(f"Task requires a rigid object with uid {CUBE_UID!r}.")
        self._cube: RigidObject = cube
        self._completed_cycles = 0
        self._planned_cycle_count = 0
        self._last_target_position: torch.Tensor | None = None
        self._initialize_planning_services()

    @staticmethod
    def _validate_place_positions(
        positions: Sequence[Sequence[float]],
    ) -> tuple[tuple[float, float, float], ...]:
        """Validate and normalize release positions from task configuration."""
        normalized = tuple(
            tuple(float(value) for value in position) for position in positions
        )
        if not normalized or any(len(position) != 3 for position in normalized):
            raise ValueError("place_positions must contain at least one XYZ position.")
        return normalized

    def _validate_settings(self) -> None:
        """Validate task settings before allocating a simulation."""
        if self.num_cycles < 1:
            raise ValueError("num_cycles must be at least 1.")
        if self.grasp_hold_steps < 0:
            raise ValueError("grasp_hold_steps must be non-negative.")
        if not 0 <= self.settle_min_steps <= self.settle_max_steps:
            raise ValueError(
                "settle_min_steps must be non-negative and no larger than "
                "settle_max_steps."
            )
        if self.settle_stable_steps < 1:
            raise ValueError("settle_stable_steps must be at least 1.")
        if self.linear_velocity_threshold < 0 or self.angular_velocity_threshold < 0:
            raise ValueError("Velocity thresholds must be non-negative.")
        if self.place_position_tolerance <= 0:
            raise ValueError("place_position_tolerance must be positive.")

    def _initialize_planning_services(self) -> None:
        """Create the standalone grasp service and direct motion generator."""
        from embodichain.lab.sim.planners import (
            MotionGenCfg,
            MotionGenerator,
            ToppraPlannerCfg,
        )

        hand_limits = self.robot.get_qpos_limits(name="hand")[0].to(
            device=self.device, dtype=torch.float32
        )
        hand_open_qpos = hand_limits[:, 0]
        hand_close_qpos = torch.clamp(
            torch.full_like(hand_limits[:, 1], DEFAULT_GRIPPER_CLOSE_QPOS),
            min=hand_limits[:, 0],
            max=hand_limits[:, 1],
        )
        self._motion_generator: MotionGenerator = MotionGenerator(
            cfg=MotionGenCfg(planner_cfg=ToppraPlannerCfg(robot_uid=self.robot.uid))
        )
        if self._grasp_pose_generator is None:
            self._grasp_pose_generator = _create_default_grasp_pose_generator()
        self._arm_joint_ids = tuple(self.robot.get_joint_ids("arm"))
        self._hand_joint_ids = tuple(self.robot.get_joint_ids("hand"))
        self._hand_open_qpos = hand_open_qpos
        self._hand_close_qpos = hand_close_qpos
        self._cube_mesh_vertices = self._cube.get_vertices(env_ids=[0], scale=True)[0]
        self._cube_mesh_triangles = self._cube.get_triangles(env_ids=[0])[0]

    def create_demo_segments(
        self, *, num_cycles: int | None = None, **kwargs: Any
    ) -> Iterable[DemoSegment]:
        """Lazily plan repeated cube pick-and-place segments.

        Args:
            num_cycles: Optional per-rollout override for the configured cycle count.
            **kwargs: Reserved for future expert-planning options.

        Yields:
            One :class:`DemoSegment` for every pickup/place cycle.
        """
        del kwargs
        cycle_count = self.num_cycles if num_cycles is None else int(num_cycles)
        if cycle_count < 1:
            raise ValueError("num_cycles must be at least 1.")

        self._completed_cycles = 0
        self._planned_cycle_count = cycle_count
        self._last_target_position = None
        for cycle_index in range(cycle_count):
            target_position = torch.tensor(
                self.place_positions[cycle_index % len(self.place_positions)],
                dtype=torch.float32,
                device=self.device,
            )
            plan_success, actions, source_pose = self._plan_pick_place_cycle(
                target_position
            )
            self._last_target_position = target_position
            source_position = source_pose[:, :3, 3].detach().cpu().tolist()
            logger.log_info(
                f"Planned cube pick/place segment {cycle_index + 1}/{cycle_count} "
                f"from {source_position} to {target_position.detach().cpu().tolist()}."
            )
            yield DemoSegment(
                actions=actions,
                name=f"cube_pick_place_{cycle_index + 1}",
                target_uid=CUBE_UID,
                instruction=(
                    "Pick up the cube from its current settled pose and freely "
                    f"place it at target {cycle_index + 1}."
                ),
                metadata={
                    "cycle_index": cycle_index,
                    "cycle_count": cycle_count,
                    "planning_success": plan_success.detach().cpu().tolist(),
                    "planned_source_poses": source_pose.detach().cpu().tolist(),
                    "target_position": target_position.detach().cpu().tolist(),
                    "free_fall_settle": True,
                },
                validator=partial(
                    self._validate_cycle,
                    plan_success.detach().clone(),
                    target_position.detach().clone(),
                ),
            )
            # Execution and validation happen while the generator is suspended at
            # ``yield``. Advancing to the next iteration therefore means that the
            # cube has already reached its new, measured scene pose.
            self._completed_cycles = cycle_index + 1

    def _plan_pick_place_cycle(
        self, target_position: torch.Tensor
    ) -> tuple[torch.Tensor, Iterable[torch.Tensor], torch.Tensor]:
        """Plan one cycle by composing the grasp and motion services directly."""
        from embodichain.lab.sim.planners import normalize_success_mask
        from embodichain.utils.math import pose_inv

        source_pose = self._cube.get_local_pose(to_matrix=True).to(
            device=self.device, dtype=torch.float32
        )
        if self._grasp_pose_generator is None:
            raise RuntimeError("The grasp-pose generator was not initialized.")
        approach_direction = torch.tensor(
            [0.0, 0.0, -1.0],
            dtype=torch.float32,
            device=self.device,
        )
        grasp_success, grasp_pose, _ = self._grasp_pose_generator.get_best_grasp_poses(
            mesh_vertices=self._cube_mesh_vertices,
            mesh_triangles=self._cube_mesh_triangles,
            obj_poses=source_pose,
            approach_direction=approach_direction,
        )
        grasp_success = normalize_success_mask(
            grasp_success,
            num_envs=self.num_envs,
            device=self.device,
            name="Direct grasp-pose success",
        )
        grasp_pose = grasp_pose.to(device=self.device, dtype=torch.float32)
        if not bool(grasp_success.any().item()):
            hold = self.robot.get_qpos().clone().unsqueeze(1)
            return (
                grasp_success,
                self._iter_cycle_actions(hold, clear_dynamics_step=None),
                source_pose,
            )

        start_qpos = self.robot.get_qpos().to(device=self.device, dtype=torch.float32)
        start_arm_qpos = start_qpos[:, self._arm_joint_ids]
        n_approach, n_close, n_lift = self._split_motion_samples(
            self.PICK_SAMPLE_INTERVAL
        )
        pre_grasp_pose = self._translated_pose(
            grasp_pose,
            -approach_direction * self.PRE_GRASP_DISTANCE,
        )
        approach_success, approach_arm = self._generate_arm_trajectory(
            torch.stack((pre_grasp_pose, grasp_pose), dim=1),
            start_qpos=start_arm_qpos,
            sample_count=n_approach,
            phase="pick approach",
        )
        lift_pose = self._translated_pose(
            grasp_pose,
            torch.tensor(
                [0.0, 0.0, self.PICK_LIFT_HEIGHT],
                dtype=torch.float32,
                device=self.device,
            ),
        )
        lift_success, lift_arm = self._generate_arm_trajectory(
            lift_pose,
            start_qpos=approach_arm[:, -1],
            sample_count=n_lift,
            phase="pick lift",
        )
        pick_success = grasp_success & approach_success & lift_success
        pick_trajectory = self._compose_arm_hand_trajectory(
            base_qpos=start_qpos,
            first_arm=approach_arm,
            second_arm=lift_arm,
            first_hand=self._hand_open_qpos,
            second_hand=self._hand_close_qpos,
            transition_steps=n_close,
        )
        if not bool(pick_success.all().item()):
            trajectory = self._ensure_nonempty_trajectory(pick_trajectory)
            return (
                torch.zeros_like(pick_success),
                self._iter_cycle_actions(trajectory, clear_dynamics_step=None),
                source_pose,
            )

        close_end_step = approach_arm.shape[1] + n_close
        pick_trajectory, clear_dynamics_step = self._insert_grasp_hold(
            pick_trajectory,
            close_end_step=close_end_step,
        )
        object_to_eef = torch.bmm(pose_inv(source_pose), grasp_pose)
        desired_cube_pose = source_pose.clone()
        desired_cube_pose[:, :3, 3] = target_position.unsqueeze(0).expand(
            self.num_envs, -1
        )
        place_eef_pose = torch.bmm(desired_cube_pose, object_to_eef)
        place_retract_pose = self._translated_pose(
            place_eef_pose,
            torch.tensor(
                [0.0, 0.0, self.PLACE_RETRACT_HEIGHT],
                dtype=torch.float32,
                device=self.device,
            ),
        )
        n_place, n_open, n_retract = self._split_motion_samples(
            self.PLACE_SAMPLE_INTERVAL
        )
        place_success, place_arm = self._generate_arm_trajectory(
            torch.stack((place_retract_pose, place_eef_pose), dim=1),
            start_qpos=lift_arm[:, -1],
            sample_count=n_place,
            phase="place approach",
        )
        retract_success, retract_arm = self._generate_arm_trajectory(
            place_retract_pose,
            start_qpos=place_arm[:, -1],
            sample_count=n_retract,
            phase="place retract",
        )
        place_trajectory = self._compose_arm_hand_trajectory(
            base_qpos=pick_trajectory[:, -1],
            first_arm=place_arm,
            second_arm=retract_arm,
            first_hand=self._hand_close_qpos,
            second_hand=self._hand_open_qpos,
            transition_steps=n_open,
        )
        trajectory = self._ensure_nonempty_trajectory(
            torch.cat((pick_trajectory, place_trajectory), dim=1)
        )
        return (
            pick_success & place_success & retract_success,
            self._iter_cycle_actions(trajectory, clear_dynamics_step),
            source_pose,
        )

    @classmethod
    def _split_motion_samples(cls, sample_count: int) -> tuple[int, int, int]:
        """Split one sample budget into motion, hand, and motion phases."""
        motion_samples = sample_count - cls.HAND_INTERP_STEPS
        first = int(round(motion_samples * 0.6))
        third = motion_samples - first
        if first < 2 or third < 2:
            raise ValueError("Motion sample budgets require two motion waypoints.")
        return first, cls.HAND_INTERP_STEPS, third

    def _generate_arm_trajectory(
        self,
        target_poses: torch.Tensor,
        *,
        start_qpos: torch.Tensor,
        sample_count: int,
        phase: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate one arm trajectory through the direct MotionGenerator API."""
        from embodichain.lab.sim.planners import (
            MotionGenOptions,
            MoveType,
            PlanState,
            normalize_success_mask,
        )

        if target_poses.dim() == 3:
            target_poses = target_poses.unsqueeze(1)
        result = self._motion_generator.generate(
            target_states=[
                PlanState(
                    move_type=MoveType.EEF_MOVE,
                    xpos=target_poses[:, waypoint_index],
                )
                for waypoint_index in range(target_poses.shape[1])
            ],
            options=MotionGenOptions(
                sample_count=sample_count,
                start_qpos=start_qpos,
                control_part="arm",
                is_interpolate=True,
            ),
        )
        success = normalize_success_mask(
            result.success,
            num_envs=self.num_envs,
            device=self.device,
            name=f"{phase} motion-generation success",
        )
        if result.positions is None:
            logger.log_warning(f"MotionGenerator returned no positions for {phase}.")
            return torch.zeros_like(success), start_qpos.unsqueeze(1)
        return success, result.positions.to(device=self.device, dtype=torch.float32)

    def _compose_arm_hand_trajectory(
        self,
        *,
        base_qpos: torch.Tensor,
        first_arm: torch.Tensor,
        second_arm: torch.Tensor,
        first_hand: torch.Tensor,
        second_hand: torch.Tensor,
        transition_steps: int,
    ) -> torch.Tensor:
        """Embed two arm motions and one hand transition in full robot qpos."""
        transition_hand = self._interpolate_hand_qpos(
            first_hand,
            second_hand,
            transition_steps,
        )
        first_count = first_arm.shape[1]
        second_count = second_arm.shape[1]
        full = (
            base_qpos.unsqueeze(1)
            .expand(
                -1,
                first_count + transition_steps + second_count,
                -1,
            )
            .clone()
        )
        full[:, :first_count, self._arm_joint_ids] = first_arm
        full[:, :first_count, self._hand_joint_ids] = first_hand.unsqueeze(1)
        full[:, first_count : first_count + transition_steps, self._arm_joint_ids] = (
            first_arm[:, -1].unsqueeze(1)
        )
        full[:, first_count : first_count + transition_steps, self._hand_joint_ids] = (
            transition_hand
        )
        full[:, first_count + transition_steps :, self._arm_joint_ids] = second_arm
        full[:, first_count + transition_steps :, self._hand_joint_ids] = (
            second_hand.unsqueeze(1)
        )
        return full

    @staticmethod
    def _interpolate_hand_qpos(
        start: torch.Tensor,
        end: torch.Tensor,
        sample_count: int,
    ) -> torch.Tensor:
        """Linearly interpolate batched hand joint positions."""
        weights = torch.linspace(
            0.0,
            1.0,
            sample_count,
            dtype=start.dtype,
            device=start.device,
        )
        return torch.lerp(
            start.unsqueeze(1),
            end.to(device=start.device, dtype=start.dtype).unsqueeze(1),
            weights[None, :, None],
        )

    @staticmethod
    def _translated_pose(pose: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        """Return a batched pose translated by a world-frame offset."""
        translated = pose.clone()
        translated[:, :3, 3] += offset.to(device=pose.device, dtype=pose.dtype)
        return translated

    def _insert_grasp_hold(
        self,
        pick_trajectory: torch.Tensor,
        *,
        close_end_step: int | None = None,
    ) -> tuple[torch.Tensor, int]:
        """Hold the closed command at the grasp pose before beginning the lift."""
        if close_end_step is None:
            close_end_step = (
                int(round(self.PICK_SAMPLE_INTERVAL - self.HAND_INTERP_STEPS) * 0.6)
                + self.HAND_INTERP_STEPS
            )
        close_end_step = min(close_end_step, pick_trajectory.shape[1])
        if self.grasp_hold_steps == 0:
            return pick_trajectory, close_end_step

        grasp_hold = pick_trajectory[:, close_end_step - 1 : close_end_step, :].repeat(
            1, self.grasp_hold_steps, 1
        )
        augmented = torch.cat(
            (
                pick_trajectory[:, :close_end_step, :],
                grasp_hold,
                pick_trajectory[:, close_end_step:, :],
            ),
            dim=1,
        )
        return augmented, close_end_step + self.grasp_hold_steps

    def _ensure_nonempty_trajectory(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Return at least one hold command so planning failure is recordable."""
        if trajectory.shape[1] > 0:
            return trajectory
        return self.robot.get_qpos().clone().unsqueeze(1)

    def _iter_cycle_actions(
        self,
        trajectory: torch.Tensor,
        clear_dynamics_step: int | None,
    ) -> Iterable[torch.Tensor]:
        """Replay a planned trajectory, then hold until the cube is stable."""
        for step_index, action in enumerate(trajectory.unbind(dim=1), start=1):
            yield action
            if clear_dynamics_step is not None and step_index == clear_dynamics_step:
                # Match the pickup tutorial: clear residual object velocity just
                # after gripper closure and before the lift phase.
                self._cube.clear_dynamics()

        hold_action = trajectory[:, -1].clone()
        stable_steps = 0
        for settle_step in range(self.settle_max_steps):
            yield hold_action
            if settle_step + 1 < self.settle_min_steps:
                continue
            if bool(self._cube_is_stable().all().item()):
                stable_steps += 1
                if stable_steps >= self.settle_stable_steps:
                    break
            else:
                stable_steps = 0

    def _cube_is_stable(self) -> torch.Tensor:
        """Return whether cube linear and angular speeds are below thresholds."""
        linear_speed = torch.linalg.vector_norm(self._cube.body_data.lin_vel, dim=-1)
        angular_speed = torch.linalg.vector_norm(self._cube.body_data.ang_vel, dim=-1)
        return (linear_speed <= self.linear_velocity_threshold) & (
            angular_speed <= self.angular_velocity_threshold
        )

    def _cube_settled_near(self, target_position: torch.Tensor) -> torch.Tensor:
        """Validate that the cube settled near a release target after free fall."""
        cube_position = self._cube.get_local_pose(to_matrix=True)[:, :3, 3]
        target_position = target_position.to(
            device=cube_position.device, dtype=cube_position.dtype
        )
        xy_error = torch.linalg.vector_norm(
            cube_position[:, :2] - target_position[None, :2], dim=-1
        )
        valid_height = (cube_position[:, 2] >= -0.01) & (
            cube_position[:, 2] <= target_position[2] + CUBE_SIZE
        )
        return (
            (xy_error <= self.place_position_tolerance)
            & valid_height
            & self._cube_is_stable()
        )

    def _validate_cycle(
        self, plan_success: torch.Tensor, target_position: torch.Tensor
    ) -> torch.Tensor:
        """Combine motion-planning and post-free-fall validation."""
        return plan_success.to(device=self.device, dtype=torch.bool) & (
            self._cube_settled_near(target_position)
        )

    def is_task_success(self, **kwargs: Any) -> torch.Tensor:
        """Return success after all lazy segments have executed and validated.

        Args:
            **kwargs: Reserved for task-evaluation options.

        Returns:
            One success flag per parallel environment.
        """
        del kwargs
        if (
            self._planned_cycle_count < 1
            or self._completed_cycles < self._planned_cycle_count
            or self._last_target_position is None
        ):
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return self._cube_settled_near(self._last_target_position)
