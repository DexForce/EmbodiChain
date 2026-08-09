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
    from embodichain.lab.sim.atomic_actions import AtomicActionEngine, ObjectSemantics
    from embodichain.lab.sim.objects import RigidObject

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


def _create_default_robot_cfg() -> URRobotCfg:
    """Create the UR5 and parallel-gripper setup used by atomic-action demos."""
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
        "grasp_samples": 10000,
        "force_reannotate": False,
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

    def __init__(self, cfg: EmbodiedEnvCfg | None = None, **kwargs: Any) -> None:
        if cfg is None:
            cfg = _create_default_env_cfg()

        extensions = getattr(cfg, "extensions", {}) or {}
        self.num_cycles = int(extensions.get("num_cycles", DEFAULT_NUM_CYCLES))
        self.place_positions = self._validate_place_positions(
            extensions.get("place_positions", DEFAULT_PLACE_POSITIONS)
        )
        self.grasp_samples = int(extensions.get("grasp_samples", 10000))
        self.force_reannotate = bool(extensions.get("force_reannotate", False))
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
        self.grasp_samples = int(self.grasp_samples)
        self.force_reannotate = bool(self.force_reannotate)
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
        self._initialize_atomic_actions()

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
        if self.grasp_samples < 1:
            raise ValueError("grasp_samples must be at least 1.")
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

    def _initialize_atomic_actions(self) -> None:
        """Create the motion generator, action engine and cube semantics."""
        from embodichain.lab.sim.atomic_actions import (
            AtomicActionEngine,
            ControlPartCommandProfile,
        )
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
        motion_generator = MotionGenerator(
            cfg=MotionGenCfg(planner_cfg=ToppraPlannerCfg(robot_uid=self.robot.uid))
        )
        self._action_engine: AtomicActionEngine = AtomicActionEngine(
            motion_generator,
            control_profiles={
                "hand": ControlPartCommandProfile.joint_positions(
                    open=hand_open_qpos,
                    grasp=hand_close_qpos,
                )
            },
        )
        self._cube_semantics: ObjectSemantics = self._create_cube_semantics()

    def _create_cube_semantics(self) -> ObjectSemantics:
        """Create reusable antipodal semantics for the task cube."""
        from embodichain.lab.sim.atomic_actions import (
            AntipodalAffordance,
            ObjectSemantics,
        )
        from embodichain.toolkits.graspkit.pg_grasp.antipodal_generator import (
            AntipodalSamplerCfg,
            GraspGeneratorCfg,
        )
        from embodichain.toolkits.graspkit.pg_grasp.gripper_collision_checker import (
            GripperCollisionCfg,
        )

        vertices = self._cube.get_vertices(env_ids=[0], scale=True)[0]
        triangles = self._cube.get_triangles(env_ids=[0])[0]
        return ObjectSemantics(
            label=CUBE_UID,
            geometry={},
            affordance=AntipodalAffordance(
                mesh_vertices=vertices,
                mesh_triangles=triangles,
                gripper_collision_cfg=GripperCollisionCfg(
                    max_open_length=GRIPPER_MAX_OPEN_WIDTH,
                    finger_length=GRIPPER_FINGER_LENGTH,
                    y_thickness=GRIPPER_Y_THICKNESS,
                    root_z_width=GRIPPER_ROOT_Z_WIDTH,
                    open_check_margin=0.002,
                    point_sample_dense=0.012,
                ),
                generator_cfg=GraspGeneratorCfg(
                    viser_port=11801,
                    antipodal_sampler_cfg=AntipodalSamplerCfg(
                        n_sample=self.grasp_samples,
                        max_length=GRIPPER_MAX_OPEN_WIDTH,
                        min_length=0.005,
                    ),
                    is_partial_annotate=False,
                    is_filter_ground_collision=False,
                ),
                force_reannotate=self.force_reannotate,
            ),
            entity=self._cube,
        )

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
        """Plan one pickup/place cycle from the cube's current measured pose."""
        from embodichain.lab.sim.atomic_actions import (
            ActionBinding,
            ActionInvocation,
            GraspGoal,
            MotionPolicy,
            PickUpOptions,
            PlaceGoal,
            PlaceOptions,
        )

        source_pose = self._cube.get_local_pose(to_matrix=True).to(
            device=self.device, dtype=torch.float32
        )
        binding = ActionBinding(
            manipulators={"primary": "arm"},
            end_effectors={"primary": "hand"},
        )
        pick_compiled = self._action_engine.compile(
            (
                ActionInvocation(
                    skill_id="pick_up",
                    goal=GraspGoal(self._cube_semantics),
                    binding=binding,
                    motion_policy=MotionPolicy(sample_count=self.PICK_SAMPLE_INTERVAL),
                    skill_options=PickUpOptions(
                        pre_grasp_distance=0.15,
                        lift_height=0.16,
                        hand_interp_steps=self.HAND_INTERP_STEPS,
                    ),
                ),
            )
        )
        pick_success = pick_compiled.plan_success
        pick_trajectory = pick_compiled.trajectory.positions
        picked_context = pick_compiled.projected_context
        held = picked_context.get_held_object("arm")
        if held is None or not bool(pick_success.all().item()):
            trajectory = self._ensure_nonempty_trajectory(pick_trajectory)
            return (
                torch.zeros_like(pick_success, dtype=torch.bool),
                self._iter_cycle_actions(trajectory, clear_dynamics_step=None),
                source_pose,
            )

        pick_trajectory, clear_dynamics_step = self._insert_grasp_hold(pick_trajectory)
        desired_cube_pose = source_pose.clone()
        desired_cube_pose[:, :3, 3] = target_position.unsqueeze(0).expand(
            self.num_envs, -1
        )
        place_eef_pose = torch.bmm(desired_cube_pose, held.object_to_eef)
        place_compiled = self._action_engine.compile(
            (
                ActionInvocation(
                    skill_id="place",
                    goal=PlaceGoal(place_eef_pose),
                    binding=binding,
                    motion_policy=MotionPolicy(sample_count=self.PLACE_SAMPLE_INTERVAL),
                    skill_options=PlaceOptions(
                        lift_height=0.14,
                        hand_interp_steps=self.HAND_INTERP_STEPS,
                    ),
                ),
            ),
            picked_context,
        )
        place_success = place_compiled.plan_success
        place_trajectory = place_compiled.trajectory.positions
        trajectory = self._ensure_nonempty_trajectory(
            torch.cat((pick_trajectory, place_trajectory), dim=1)
        )
        return (
            pick_success & place_success,
            self._iter_cycle_actions(trajectory, clear_dynamics_step),
            source_pose,
        )

    def _insert_grasp_hold(
        self, pick_trajectory: torch.Tensor
    ) -> tuple[torch.Tensor, int]:
        """Hold the closed command at the grasp pose before beginning the lift."""
        close_end_step = min(
            int(round(self.PICK_SAMPLE_INTERVAL - self.HAND_INTERP_STEPS) * 0.6)
            + self.HAND_INTERP_STEPS,
            pick_trajectory.shape[1],
        )
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
