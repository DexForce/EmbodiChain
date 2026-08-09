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

"""Stack two blocks with one atomic-action demonstration segment."""

from __future__ import annotations

from collections.abc import Iterable
from functools import partial
from typing import TYPE_CHECKING, Any

import torch

from embodichain.lab.gym.envs import DemoSegment, EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.utils.registration import register_env
from embodichain.utils import logger

if TYPE_CHECKING:
    from embodichain.lab.sim.atomic_actions import AtomicActionEngine, ObjectSemantics
    from embodichain.lab.sim.objects import RigidObject

__all__ = ["StackBlocksTwoEnv"]

BASE_BLOCK_UID = "block_1"
STACK_BLOCK_UID = "block_2"
CONTROL_PART = "right_arm"
HAND_CONTROL_PART = "right_eef"
BLOCK_HEIGHT = 0.05
GRASP_OFFSET = (0.02, 0.0, -0.025)
HAND_OPEN_QPOS = 0.05
HAND_CLOSE_QPOS = 0.0
PICK_SAMPLE_INTERVAL = 90
PLACE_SAMPLE_INTERVAL = 90
HAND_INTERP_STEPS = 10
GRASP_HOLD_STEPS = 45
SETTLE_STEPS = 30


@register_env("StackBlocksTwo-v1", max_episode_steps=600)
class StackBlocksTwoEnv(EmbodiedEnv):
    """Pick up ``block_2`` and place it on ``block_1`` as one segment."""

    def __init__(self, cfg: EmbodiedEnvCfg | None = None, **kwargs: Any) -> None:
        super().__init__(cfg, **kwargs)

        base_block = self.sim.get_rigid_object(BASE_BLOCK_UID)
        stack_block = self.sim.get_rigid_object(STACK_BLOCK_UID)
        if base_block is None or stack_block is None:
            raise RuntimeError(
                "StackBlocksTwo-v1 requires rigid objects 'block_1' and 'block_2'."
            )
        self._base_block: RigidObject = base_block
        self._stack_block: RigidObject = stack_block
        self._initialize_atomic_actions()

    def _initialize_atomic_actions(self) -> None:
        """Create the right-arm atomic-action engine and object semantics."""
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

        hand_dof = len(self.robot.get_joint_ids(name=HAND_CONTROL_PART))
        hand_open_qpos = torch.full(
            (hand_dof,), HAND_OPEN_QPOS, dtype=torch.float32, device=self.device
        )
        hand_close_qpos = torch.full(
            (hand_dof,), HAND_CLOSE_QPOS, dtype=torch.float32, device=self.device
        )
        motion_generator = MotionGenerator(
            cfg=MotionGenCfg(planner_cfg=ToppraPlannerCfg(robot_uid=self.robot.uid))
        )
        self._action_engine: AtomicActionEngine = AtomicActionEngine(
            motion_generator,
            control_profiles={
                HAND_CONTROL_PART: ControlPartCommandProfile.joint_positions(
                    open=hand_open_qpos,
                    grasp=hand_close_qpos,
                )
            },
        )
        self._stack_block_semantics: ObjectSemantics = ObjectSemantics(
            affordance=Affordance(),
            geometry={},
            label=STACK_BLOCK_UID,
            entity=self._stack_block,
        )

    def create_demo_segments(self, **kwargs: Any) -> tuple[DemoSegment]:
        """Plan the complete stacking task as exactly one semantic segment."""
        del kwargs
        plan_success, trajectory, source_pose, target_pose = self._plan_stack()
        return (
            DemoSegment(
                actions=self._iter_segment_actions(trajectory),
                name="stack_block_2_on_block_1",
                target_uid=STACK_BLOCK_UID,
                instruction="Pick up block 2 and place it on top of block 1.",
                metadata={
                    "segment_index": 0,
                    "segment_count": 1,
                    "planning_success": plan_success.detach().cpu().tolist(),
                    "source_pose": source_pose.detach().cpu().tolist(),
                    "target_pose": target_pose.detach().cpu().tolist(),
                    "atomic_actions": ["pick_up", "place"],
                },
                validator=partial(self._validate_stack, plan_success.detach().clone()),
            ),
        )

    def _plan_stack(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Plan PickUp then Place while threading the held-object state."""
        from embodichain.lab.sim.atomic_actions import (
            ActionBinding,
            ActionInvocation,
            GraspGoal,
            MotionPolicy,
            PickUpOptions,
            PlaceGoal,
            PlaceOptions,
        )

        source_pose = self._stack_block.get_local_pose(to_matrix=True).to(
            device=self.device, dtype=torch.float32
        )
        base_pose = self._base_block.get_local_pose(to_matrix=True).to(
            device=self.device, dtype=torch.float32
        )
        grasp_pose = self.robot.compute_fk(
            qpos=self.robot.get_qpos()[:, self.robot.get_joint_ids(name=CONTROL_PART)],
            name=CONTROL_PART,
            to_matrix=True,
        )
        grasp_pose[:, :3, 3] = source_pose[:, :3, 3] + torch.tensor(
            GRASP_OFFSET, dtype=torch.float32, device=self.device
        )
        binding = ActionBinding(
            manipulators={"primary": CONTROL_PART},
            end_effectors={"primary": HAND_CONTROL_PART},
        )
        pick_compiled = self._action_engine.compile(
            (
                ActionInvocation(
                    skill_id="pick_up",
                    goal=GraspGoal(
                        self._stack_block_semantics,
                        grasp_xpos=grasp_pose,
                    ),
                    binding=binding,
                    motion_policy=MotionPolicy(sample_count=PICK_SAMPLE_INTERVAL),
                    skill_options=PickUpOptions(
                        pre_grasp_distance=0.12,
                        lift_height=0.15,
                        hand_interp_steps=HAND_INTERP_STEPS,
                    ),
                ),
            )
        )
        pick_success = pick_compiled.plan_success
        pick_trajectory = pick_compiled.trajectory.positions
        picked_context = pick_compiled.projected_context
        pick_trajectory = self._insert_grasp_hold(pick_trajectory)

        target_pose = source_pose.clone()
        target_pose[:, :3, 3] = base_pose[:, :3, 3]
        target_pose[:, 2, 3] += BLOCK_HEIGHT
        held = picked_context.get_held_object(CONTROL_PART)
        if held is None or not bool(pick_success.all().item()):
            return (
                torch.zeros_like(pick_success, dtype=torch.bool),
                self._ensure_nonempty_trajectory(pick_trajectory),
                source_pose,
                target_pose,
            )

        place_eef_pose = torch.bmm(target_pose, held.object_to_eef)
        place_compiled = self._action_engine.compile(
            (
                ActionInvocation(
                    skill_id="place",
                    goal=PlaceGoal(place_eef_pose),
                    binding=binding,
                    motion_policy=MotionPolicy(sample_count=PLACE_SAMPLE_INTERVAL),
                    skill_options=PlaceOptions(
                        lift_height=0.10,
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
        return pick_success & place_success, trajectory, source_pose, target_pose

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
        """Return at least one hold command so planning failure is observable."""
        if trajectory.shape[1] > 0:
            return trajectory
        return self.robot.get_qpos().clone().unsqueeze(1)

    def _iter_segment_actions(self, trajectory: torch.Tensor) -> Iterable[torch.Tensor]:
        """Replay the atomic trajectory and wait for the released block to settle."""
        close_end_step = min(
            round((PICK_SAMPLE_INTERVAL - HAND_INTERP_STEPS) * 0.6)
            + HAND_INTERP_STEPS
            + GRASP_HOLD_STEPS,
            trajectory.shape[1],
        )
        for step_index, action in enumerate(trajectory.unbind(dim=1), start=1):
            yield action
            if step_index == close_end_step:
                self._stack_block.clear_dynamics()

        hold_action = trajectory[:, -1].clone()
        for _ in range(SETTLE_STEPS):
            yield hold_action

    def _validate_stack(self, plan_success: torch.Tensor) -> torch.Tensor:
        """Require both a successful atomic plan and a physically valid stack."""
        task_success = self.is_task_success()
        success = plan_success.to(device=self.device) & task_success
        if not bool(success.all().item()):
            base_pose = self._base_block.get_local_pose(to_matrix=True)
            stack_pose = self._stack_block.get_local_pose(to_matrix=True)
            base_pos = base_pose[:, :3, 3]
            stack_pos = stack_pose[:, :3, 3]
            logger.log_warning(
                "Stack validation failed: "
                f"planning_success={plan_success.detach().cpu().tolist()}, "
                f"base_position={base_pos.detach().cpu().tolist()}, "
                f"stack_position={stack_pos.detach().cpu().tolist()}, "
                f"base_fallen={self._is_fall(base_pose).detach().cpu().tolist()}, "
                f"stack_fallen={self._is_fall(stack_pose).detach().cpu().tolist()}, "
                f"stack_z_axis={stack_pose[:, :3, 2].detach().cpu().tolist()}."
            )
        return success

    def is_task_success(self, **kwargs: Any) -> torch.Tensor:
        """Return whether block 2 is upright and centered on block 1."""
        del kwargs
        block1_pose = self._base_block.get_local_pose(to_matrix=True)
        block2_pose = self._stack_block.get_local_pose(to_matrix=True)
        block1_pos = block1_pose[:, :3, 3]
        block2_pos = block2_pose[:, :3, 3]

        expected_block2_pos = block1_pos.clone()
        expected_block2_pos[:, 2] += BLOCK_HEIGHT
        tolerance = torch.tensor(
            [0.025, 0.025, 0.012], dtype=torch.float32, device=self.device
        )
        within_tolerance = torch.all(
            torch.abs(block2_pos - expected_block2_pos) < tolerance, dim=1
        )
        return (
            within_tolerance & ~self._is_fall(block1_pose) & ~self._is_fall(block2_pose)
        )

    @staticmethod
    def _is_fall(pose: torch.Tensor) -> torch.Tensor:
        """Return whether an object's local z-axis tilts by at least 45 degrees."""
        pose_rz = pose[:, :3, 2]
        world_z_axis = torch.tensor([0, 0, 1], dtype=pose.dtype, device=pose.device)
        dot_product = torch.sum(pose_rz * world_z_axis, dim=-1).clamp(-1.0, 1.0)
        return torch.arccos(dot_product) >= torch.pi / 4
