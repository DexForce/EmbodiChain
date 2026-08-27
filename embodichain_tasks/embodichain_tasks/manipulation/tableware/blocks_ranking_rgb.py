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

from __future__ import annotations

from collections.abc import Iterable
from functools import partial
from typing import TYPE_CHECKING, Any

import torch

from embodichain.lab.gym.envs import DemoSegment, EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.utils.registration import register_env
from embodichain.utils import logger

if TYPE_CHECKING:
    from embodichain.lab.sim.atomic_actions import (
        AtomicActionEngine,
        ObjectSemantics,
    )
    from embodichain.lab.sim.objects import RigidObject

__all__ = ["BlocksRankingRGBEnv"]

REFERENCE_BLOCK_UID = "block_2"
PICK_SAMPLE_INTERVAL = 90
PLACE_SAMPLE_INTERVAL = 90
HAND_INTERP_STEPS = 10
GRASP_HOLD_STEPS = 45
FREE_FALL_RELEASE_HEIGHT = 0.08
SETTLE_MIN_STEPS = 15
SETTLE_MAX_STEPS = 60
SETTLE_STABLE_STEPS = 5
LINEAR_VELOCITY_THRESHOLD = 0.03
ANGULAR_VELOCITY_THRESHOLD = 0.20
PLACEMENT_XY_TOLERANCE = (0.035, 0.03)

BLOCK_PLANS = (
    {
        "uid": "block_1",
        "color": "red",
        "arm": "right_arm",
        "hand": "right_eef",
        "x_offset": -0.08,
    },
    {
        "uid": "block_3",
        "color": "blue",
        "arm": "left_arm",
        "hand": "left_eef",
        "x_offset": 0.08,
    },
)


@register_env("BlocksRankingRGB-v1", max_episode_steps=600)
class BlocksRankingRGBEnv(EmbodiedEnv):
    """Arrange the red and blue blocks around the stationary green block."""

    def __init__(self, cfg: EmbodiedEnvCfg | None = None, **kwargs: Any) -> None:
        super().__init__(cfg, **kwargs)
        self._blocks = self._get_required_blocks()
        self._initialize_atomic_actions()

    def _get_required_blocks(self) -> dict[str, RigidObject]:
        """Resolve the three blocks required by the ranking task."""
        blocks = {
            uid: self.sim.get_rigid_object(uid)
            for uid in ("block_1", REFERENCE_BLOCK_UID, "block_3")
        }
        missing = [uid for uid, block in blocks.items() if block is None]
        if missing:
            raise RuntimeError(f"BlocksRankingRGB requires objects {missing}.")
        return blocks

    def _initialize_atomic_actions(self) -> None:
        """Create the dual-arm atomic-action engine and object semantics."""
        from embodichain.lab.sim.atomic_actions import (
            Affordance,
            AtomicActionEngine,
            ControlPartCommandProfile,
            ObjectSemantics,
        )
        from embodichain.lab.sim.planners import (
            MotionGenCfg,
            MotionGenerator,
            ToppraPlannerCfg,
        )

        motion_generator = MotionGenerator(
            cfg=MotionGenCfg(planner_cfg=ToppraPlannerCfg(robot_uid=self.robot.uid))
        )
        self._object_semantics: dict[str, ObjectSemantics] = {}
        control_profiles: dict[str, ControlPartCommandProfile] = {}

        for plan in BLOCK_PLANS:
            uid = str(plan["uid"])
            hand = str(plan["hand"])
            hand_limits = self.robot.get_qpos_limits(name=hand)[0].to(
                device=self.device, dtype=torch.float32
            )
            hand_close_qpos = hand_limits[:, 0]
            hand_open_qpos = hand_limits[:, 1]
            control_profiles[hand] = ControlPartCommandProfile.joint_positions(
                open=hand_open_qpos,
                grasp=hand_close_qpos,
            )
            self._object_semantics[uid] = ObjectSemantics(
                label=uid,
                geometry={},
                affordance=Affordance(),
                entity_id=uid,
            )
        self._action_engine: AtomicActionEngine = AtomicActionEngine(
            motion_generator,
            control_profiles=control_profiles,
        )

    def create_demo_segments(self, **kwargs: Any) -> Iterable[DemoSegment]:
        """Lazily plan one atomic pick/place segment per manipulated block.

        The green block is deliberately kept stationary as the ranking reference.
        Planning is lazy so the blue-block target uses the reference pose measured
        after the red-block segment has completed.

        Args:
            **kwargs: Reserved for future expert-planning options.

        Yields:
            A red-block segment followed by a blue-block segment.
        """
        del kwargs
        for segment_index, plan in enumerate(BLOCK_PLANS):
            uid = str(plan["uid"])
            color = str(plan["color"])
            target_position = self._target_position(float(plan["x_offset"]))
            plan_success, actions, source_pose = self._plan_block_segment(
                uid=uid,
                arm=str(plan["arm"]),
                hand=str(plan["hand"]),
                target_position=target_position,
            )
            logger.log_info(
                f"Planned RGB ranking segment {segment_index + 1}/"
                f"{len(BLOCK_PLANS)} for {uid}."
            )
            yield DemoSegment(
                actions=actions,
                name=f"place_{color}_block",
                target_uid=uid,
                instruction=(
                    f"Pick up the {color} block and place it "
                    f"{'left' if float(plan['x_offset']) < 0 else 'right'} "
                    "of the green block."
                ),
                metadata={
                    "segment_index": segment_index,
                    "segment_count": len(BLOCK_PLANS),
                    "color": color,
                    "arm": str(plan["arm"]),
                    "hand": str(plan["hand"]),
                    "reference_uid": REFERENCE_BLOCK_UID,
                    "atomic_actions": ["pick_up", "place"],
                    "free_fall_release_height": FREE_FALL_RELEASE_HEIGHT,
                    "free_fall_settle": True,
                    "planning_success": plan_success.detach().cpu().tolist(),
                    "planned_source_poses": source_pose.detach().cpu().tolist(),
                    "target_position": target_position.detach().cpu().tolist(),
                },
                validator=partial(
                    self._validate_block_placement,
                    uid,
                    plan_success.detach().clone(),
                    target_position.detach().clone(),
                ),
            )

    def _target_position(self, x_offset: float) -> torch.Tensor:
        """Return one placement position per environment around the green block."""
        reference_pose = self._blocks[REFERENCE_BLOCK_UID].get_local_pose(
            to_matrix=True
        )
        target_position = reference_pose[:, :3, 3].clone()
        target_position[:, 0] += x_offset
        return target_position

    def _plan_block_segment(
        self,
        *,
        uid: str,
        arm: str,
        hand: str,
        target_position: torch.Tensor,
    ) -> tuple[torch.Tensor, Iterable[torch.Tensor], torch.Tensor]:
        """Plan an atomic PickUp followed by Place for one block."""
        from embodichain.lab.sim.atomic_actions import (
            ActionInvocation,
            EntityState,
            GraspGoal,
            MotionPolicy,
            PickUpOptions,
            PlaceGoal,
            PlaceOptions,
            SceneSnapshot,
        )

        block = self._blocks[uid]
        source_pose = block.get_local_pose(to_matrix=True).to(
            device=self.device, dtype=torch.float32
        )
        arm_ids = self.robot.get_joint_ids(name=arm)
        grasp_pose = self.robot.compute_fk(
            qpos=self.robot.get_qpos()[:, arm_ids], name=arm, to_matrix=True
        )
        local_grasp_offset = self._block_grasp_offset(block)
        world_grasp_offset = torch.bmm(
            source_pose[:, :3, :3], local_grasp_offset.unsqueeze(-1)
        ).squeeze(-1)
        grasp_pose[:, :3, 3] = source_pose[:, :3, 3] + world_grasp_offset
        endpoints = {
            "primary": {
                "motion": arm,
                "grasp": hand,
            }
        }
        pick_binding = self._action_engine.bind_control_parts(
            "pick_up",
            endpoints,
        )
        place_binding = self._action_engine.bind_control_parts(
            "place",
            endpoints,
        )
        pick_compiled = self._action_engine.compile(
            (
                ActionInvocation(
                    skill_id="pick_up",
                    goal=GraspGoal(
                        self._object_semantics[uid],
                        grasp_xpos=grasp_pose,
                    ),
                    binding=pick_binding,
                    motion_policy=MotionPolicy(sample_count=PICK_SAMPLE_INTERVAL),
                    skill_options=PickUpOptions(
                        pre_grasp_distance=0.12,
                        lift_height=0.15,
                        hand_interp_steps=HAND_INTERP_STEPS,
                    ),
                ),
            ),
            self._action_engine.initial_context(
                scene=SceneSnapshot(
                    timestamp=0.0,
                    version=0,
                    entities={uid: EntityState(source_pose)},
                ),
                control_dt=self.step_dt,
            ),
        )
        pick_success = pick_compiled.plan_success
        pick_trajectory = pick_compiled.trajectory.positions
        picked_context = pick_compiled.projected_context
        pick_trajectory = self._insert_grasp_hold(pick_trajectory)
        held = picked_context.get_held_object(arm)
        if held is None or not bool(pick_success.all().item()):
            trajectory = self._ensure_nonempty_trajectory(pick_trajectory)
            return (
                torch.zeros_like(pick_success, dtype=torch.bool),
                self._iter_segment_actions(block, trajectory),
                source_pose,
            )

        desired_object_pose = source_pose.clone()
        desired_object_pose[:, :3, 3] = target_position
        desired_object_pose[:, 2, 3] += FREE_FALL_RELEASE_HEIGHT
        place_eef_pose = torch.bmm(desired_object_pose, held.object_to_eef)
        place_compiled = self._action_engine.compile(
            (
                ActionInvocation(
                    skill_id="place",
                    goal=PlaceGoal(place_eef_pose),
                    binding=place_binding,
                    motion_policy=MotionPolicy(sample_count=PLACE_SAMPLE_INTERVAL),
                    skill_options=PlaceOptions(
                        lift_height=0.15,
                        hand_interp_steps=HAND_INTERP_STEPS,
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
            self._iter_segment_actions(block, trajectory),
            source_pose,
        )

    def _block_grasp_offset(self, block: RigidObject) -> torch.Tensor:
        """Return a size-aware local TCP offset at the block's lower edge.

        The CobotMagic gripper calibration places the TCP at the lower edge of
        the grasped cube.  Deriving the offset from the scaled mesh keeps that
        point on the object instead of below small or randomized cubes.
        """
        vertices = block.get_vertices(scale=True).to(
            device=self.device, dtype=torch.float32
        )
        extents = vertices.amax(dim=1) - vertices.amin(dim=1)
        offset = torch.zeros_like(extents)
        offset[:, 0] = extents[:, 0] * 0.5
        offset[:, 2] = extents[:, 2] * -0.5
        return offset

    def _insert_grasp_hold(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Dwell at the closed grasp pose before beginning the lift phase."""
        close_end_step = (
            round((PICK_SAMPLE_INTERVAL - HAND_INTERP_STEPS) * 0.6) + HAND_INTERP_STEPS
        )
        if trajectory.shape[1] < close_end_step:
            return trajectory
        grasp_action = trajectory[:, close_end_step - 1 : close_end_step]
        grasp_hold = grasp_action.repeat(1, GRASP_HOLD_STEPS, 1)
        return torch.cat(
            (
                trajectory[:, :close_end_step],
                grasp_hold,
                trajectory[:, close_end_step:],
            ),
            dim=1,
        )

    def _ensure_nonempty_trajectory(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Return at least one hold command so a failed plan is recordable."""
        if trajectory.shape[1] > 0:
            return trajectory
        return self.robot.get_qpos().clone().unsqueeze(1)

    def _iter_segment_actions(
        self, block: RigidObject, trajectory: torch.Tensor
    ) -> Iterable[torch.Tensor]:
        """Replay one pick/place trajectory and allow the released block to settle."""
        clear_dynamics_step = (
            round((PICK_SAMPLE_INTERVAL - HAND_INTERP_STEPS) * 0.6)
            + HAND_INTERP_STEPS
            + GRASP_HOLD_STEPS
        )
        for step_index, action in enumerate(trajectory.unbind(dim=1), start=1):
            yield action
            if step_index == clear_dynamics_step:
                block.clear_dynamics()

        hold_action = trajectory[:, -1].clone()
        stable_steps = 0
        for settle_step in range(SETTLE_MAX_STEPS):
            yield hold_action
            if settle_step + 1 < SETTLE_MIN_STEPS:
                continue
            if bool(self._block_is_stable(block).all().item()):
                stable_steps += 1
                if stable_steps >= SETTLE_STABLE_STEPS:
                    break
            else:
                stable_steps = 0

    @staticmethod
    def _block_is_stable(block: RigidObject) -> torch.Tensor:
        """Return whether block linear and angular speeds are below thresholds."""
        linear_speed = torch.linalg.vector_norm(block.body_data.lin_vel, dim=-1)
        angular_speed = torch.linalg.vector_norm(block.body_data.ang_vel, dim=-1)
        return (linear_speed <= LINEAR_VELOCITY_THRESHOLD) & (
            angular_speed <= ANGULAR_VELOCITY_THRESHOLD
        )

    def _validate_block_placement(
        self,
        uid: str,
        plan_success: torch.Tensor,
        target_position: torch.Tensor,
    ) -> torch.Tensor:
        """Validate planning and the final XY placement for one segment."""
        actual_position = self._blocks[uid].get_local_pose(to_matrix=True)[:, :3, 3]
        position_error = torch.abs(actual_position[:, :2] - target_position[:, :2])
        is_stable = self._block_is_stable(self._blocks[uid])
        tolerance = torch.tensor(
            PLACEMENT_XY_TOLERANCE,
            dtype=torch.float32,
            device=self.device,
        )
        success = (
            plan_success & torch.all(position_error < tolerance, dim=1) & is_stable
        )
        if not bool(success.all().item()):
            logger.log_warning(
                f"Segment placement validation failed for {uid}: "
                f"planning_success={plan_success.detach().cpu().tolist()}, "
                f"stable={is_stable.detach().cpu().tolist()}, "
                f"actual_xy={actual_position[:, :2].detach().cpu().tolist()}, "
                f"target_xy={target_position[:, :2].detach().cpu().tolist()}, "
                f"error_xy={position_error.detach().cpu().tolist()}."
            )
        return success

    def is_task_success(self, **kwargs) -> torch.Tensor:
        """Determine if the task is successfully completed.

        The task is successful if:
        1. Three blocks are arranged in RGB order from front to back:
           - Red block (block_1) x < Green block (block_2) x < Blue block (block_3) x
        2. All blocks are close together (within tolerance)

        Args:
            **kwargs: Additional arguments for task-specific success criteria.

        Returns:
            torch.Tensor: A boolean tensor indicating success for each environment in the batch.
        """
        try:
            block1 = self.sim.get_rigid_object("block_1")  # Red
            block2 = self.sim.get_rigid_object("block_2")  # Green
            block3 = self.sim.get_rigid_object("block_3")  # Blue
        except Exception as e:
            logger.log_warning(f"Blocks not found: {e}, returning False.")
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Get block poses
        block1_pose = block1.get_local_pose(to_matrix=True)
        block2_pose = block2.get_local_pose(to_matrix=True)
        block3_pose = block3.get_local_pose(to_matrix=True)

        # Extract positions (x, y, z)
        block1_pos = block1_pose[:, :3, 3]  # (num_envs, 3)
        block2_pos = block2_pose[:, :3, 3]
        block3_pos = block3_pose[:, :3, 3]

        # Tolerance for checking if blocks are close together
        eps = torch.tensor([0.13, 0.03], dtype=torch.float32, device=self.device)

        # Check if blocks are close together in x-y plane
        # block1 and block2 should be close
        block1_block2_diff = torch.abs(block1_pos[:, :2] - block2_pos[:, :2])
        blocks_close_12 = torch.all(block1_block2_diff < eps.unsqueeze(0), dim=1)

        # block2 and block3 should be close
        block2_block3_diff = torch.abs(block2_pos[:, :2] - block3_pos[:, :2])
        blocks_close_23 = torch.all(block2_block3_diff < eps.unsqueeze(0), dim=1)

        # Check RGB order: block1 (red) x < block2 (green) x < block3 (blue) x
        rgb_order = (block1_pos[:, 0] < block2_pos[:, 0]) & (
            block2_pos[:, 0] < block3_pos[:, 0]
        )

        # Task succeeds if blocks are close together and in RGB order
        success = blocks_close_12 & blocks_close_23 & rgb_order

        return success
