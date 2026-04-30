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

# conda activate embodichain
# python robot_challenge_tasks/scripts/run_env.py \
#   --gym_config robot_challenge_tasks/configs/stack_bowls/aloha_stack_bowls_atomic.json \
#   --device cuda \
#   --num_envs 1

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import os
import numpy as np
import torch

from embodichain.lab.gym.envs import EmbodiedEnvCfg
from embodichain.lab.gym.utils.registration import register_env
from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    MoveAction,
    MoveActionCfg,
    ObjectSemantics,
    PickUpAction,
    PickUpActionCfg,
    PlaceAction,
    PlaceActionCfg,
)
from embodichain.lab.sim.atomic_actions.core import AntipodalAffordance
from embodichain.lab.sim.planners import MotionGenerator, MotionGenCfg, ToppraPlannerCfg
from embodichain.toolkits.graspkit.pg_grasp.antipodal_generator import (
    AntipodalSamplerCfg,
    GraspGeneratorCfg,
)
from embodichain.toolkits.graspkit.pg_grasp.gripper_collision_checker import (
    GripperCollisionCfg,
)
from embodichain.utils import logger
from robot_challenge_tasks.tasks.stack_bowls import StackBowlsEnv

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import RigidObject

__all__ = ["StackBowlsAtomicEnv"]


@register_env("StackBowlsAtomic-v1")
class StackBowlsAtomicEnv(StackBowlsEnv):
    """Atomic-action version of the bowl stacking task."""

    def __init__(self, cfg: EmbodiedEnvCfg | None = None, **kwargs) -> None:
        super().__init__(cfg, **kwargs)

        self.force_reannotate = bool(
            self._get_extension_value("force_reannotate", True)
        )
        self.atomic_annotator_port = int(
            self._get_extension_value("atomic_annotator_port", 11801)
        )

        # Keep the waypoint count below the configured episode horizon.
        self.pick_sample_interval = 45
        self.place_sample_interval = 45
        self.move_sample_interval = 25
        self.hand_interp_steps = 5
        self.pre_grasp_distance = 0.1
        self.lift_height = 0.1
        self.place_lift_height = 0.1

        self.gripper_collision_cfg = GripperCollisionCfg(
            max_open_length=0.088,
            finger_length=0.078,
            point_sample_dense=0.012,
        )
        self.approach_direction = torch.tensor(
            [0.0, -0.5, -1.0], dtype=torch.float32, device=self.device
        )
        self.approach_direction /= self.approach_direction.norm()

        self.atomic_engines = {
            "left_arm": self._build_atomic_engine("left_arm"),
            "right_arm": self._build_atomic_engine("right_arm"),
        }

    def create_demo_action_list(self, *args, **kwargs) -> torch.Tensor | None:
        """Create the bowl stacking rollout using atomic pick/place/move skills."""
        if self.num_envs > 1:
            logger.log_error(
                "Atomic bowl stacking only supports a single environment, "
                f"but got num_envs={self.num_envs}."
            )

        bowl_max = self.sim.get_rigid_object("bowl_max")
        bowl_mid = self.sim.get_rigid_object("bowl_mid")
        bowl_min = self.sim.get_rigid_object("bowl_min")

        bowl_max_pose = bowl_max.get_local_pose(to_matrix=True)
        bowl_mid_pose = bowl_mid.get_local_pose(to_matrix=True)
        bowl_min_pose = bowl_min.get_local_pose(to_matrix=True)

        arm_to_use_mid = self.choice_arm_by_distance(bowl_mid_pose[:, :3, 3])
        arm_to_use_min = self.choice_arm_by_distance(bowl_min_pose[:, :3, 3])

        bowl_mid_place_pose = self._build_place_pose(
            bowl_max_pose=bowl_max_pose,
            arm_name=arm_to_use_mid,
            place_offset=self._get_mid_place_offset(arm_to_use_mid),
        )
        bowl_min_place_pose = self._build_place_pose(
            bowl_max_pose=bowl_max_pose,
            arm_name=arm_to_use_min,
            place_offset=self._get_min_place_offset(arm_to_use_min),
        )

        full_qpos = self.robot.get_qpos().clone()
        segment_list: list[torch.Tensor] = []

        mid_segments, full_qpos = self._plan_stack_stage(
            bowl_uid="bowl_mid",
            arm_name=arm_to_use_mid,
            place_pose=bowl_mid_place_pose,
            start_full_qpos=full_qpos,
            annotator_port=self.atomic_annotator_port,
        )
        if mid_segments is None:
            return None
        segment_list.extend(mid_segments)

        if (arm_to_use_mid != arm_to_use_min) or (arm_to_use_mid == arm_to_use_min):
            return_home_segment, full_qpos = self._plan_move_segment(
                arm_name=arm_to_use_mid,
                target_pose=self._get_home_pose(arm_to_use_mid),
                start_full_qpos=full_qpos,
            )
            if return_home_segment is None:
                return None
            segment_list.append(return_home_segment)

        min_segments, full_qpos = self._plan_stack_stage(
            bowl_uid="bowl_min",
            arm_name=arm_to_use_min,
            place_pose=bowl_min_place_pose,
            start_full_qpos=full_qpos,
            annotator_port=self.atomic_annotator_port + 1,
        )
        if min_segments is None:
            return None
        segment_list.extend(min_segments)

        final_return_segment, full_qpos = self._plan_move_segment(
            arm_name=arm_to_use_min,
            target_pose=self._get_home_pose(arm_to_use_min),
            start_full_qpos=full_qpos,
        )
        if final_return_segment is None:
            return None
        segment_list.append(final_return_segment)

        full_trajectory = torch.cat(segment_list, dim=1)
        return full_trajectory.permute(1, 0, 2)[:, :, self.active_joint_ids]

    def _plan_stack_stage(
        self,
        bowl_uid: str,
        arm_name: str,
        place_pose: torch.Tensor,
        start_full_qpos: torch.Tensor,
        annotator_port: int,
    ) -> tuple[list[torch.Tensor], torch.Tensor] | tuple[None, None]:
        """Plan one pick-and-place stage for a single bowl."""
        bowl_semantics = self._build_bowl_semantics(
            bowl_uid=bowl_uid,
            annotator_port=annotator_port,
        )

        pick_segment, full_qpos = self._execute_atomic_segment(
            arm_name=arm_name,
            action_name="pick_up",
            target=bowl_semantics,
            start_full_qpos=start_full_qpos,
        )
        if pick_segment is None:
            return None, None

        place_segment, full_qpos = self._execute_atomic_segment(
            arm_name=arm_name,
            action_name="place",
            target=place_pose,
            start_full_qpos=full_qpos,
        )
        if place_segment is None:
            return None, None

        return [pick_segment, place_segment], full_qpos

    def _plan_move_segment(
        self,
        arm_name: str,
        target_pose: torch.Tensor,
        start_full_qpos: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Plan a single-arm return-to-home move."""
        return self._execute_atomic_segment(
            arm_name=arm_name,
            action_name="move",
            target=target_pose,
            start_full_qpos=start_full_qpos,
        )

    def _execute_atomic_segment(
        self,
        arm_name: str,
        action_name: str,
        target: ObjectSemantics | torch.Tensor,
        start_full_qpos: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Execute one atomic action and expand it to a full-robot trajectory."""
        arm_joint_ids = self._get_arm_joint_ids(arm_name)
        start_arm_qpos = start_full_qpos[:, arm_joint_ids]
        engine = self.atomic_engines[arm_name]

        is_success, partial_trajectory, joint_ids = engine.execute(
            action_name=action_name,
            target=target,
            start_qpos=start_arm_qpos,
        )
        if not self._is_success(is_success):
            logger.log_warning(
                f"Atomic action '{action_name}' failed for arm '{arm_name}'."
            )
            return None, None
        if partial_trajectory.numel() == 0:
            logger.log_warning(
                f"Atomic action '{action_name}' returned an empty trajectory."
            )
            return None, None

        packed_trajectory = self._pack_partial_trajectory(
            partial_trajectory=partial_trajectory,
            joint_ids=joint_ids,
            start_full_qpos=start_full_qpos,
        )
        return packed_trajectory, packed_trajectory[:, -1, :].clone()

    def _pack_partial_trajectory(
        self,
        partial_trajectory: torch.Tensor,
        joint_ids: list[int],
        start_full_qpos: torch.Tensor,
    ) -> torch.Tensor:
        """Expand a single-arm atomic trajectory into a full-robot trajectory."""
        num_waypoints = partial_trajectory.shape[1]
        packed_trajectory = start_full_qpos.unsqueeze(1).repeat(1, num_waypoints, 1)
        packed_trajectory[:, :, joint_ids] = partial_trajectory
        return packed_trajectory

    def _build_atomic_engine(self, arm_name: str) -> AtomicActionEngine:
        """Build a dedicated atomic action engine for one arm."""
        eef_name = self._get_eef_name(arm_name)
        hand_open_qpos = self.robot.get_qpos_limits(name=eef_name)[0, :, 1].clone()
        hand_close_qpos = self.robot.get_qpos_limits(name=eef_name)[0, :, 0].clone()
        print("***********self.approach_direction\t", self.approach_direction)
        pickup_cfg = PickUpActionCfg(
            control_part=arm_name,
            hand_control_part=eef_name,
            hand_open_qpos=hand_open_qpos,
            hand_close_qpos=hand_close_qpos,
            approach_direction=self.approach_direction,
            pre_grasp_distance=self.pre_grasp_distance,
            lift_height=self.lift_height,
            sample_interval=self.pick_sample_interval,
            hand_interp_steps=self.hand_interp_steps,
        )
        place_cfg = PlaceActionCfg(
            control_part=arm_name,
            hand_control_part=eef_name,
            hand_open_qpos=hand_open_qpos,
            hand_close_qpos=hand_close_qpos,
            lift_height=self.place_lift_height,
            sample_interval=self.place_sample_interval,
            hand_interp_steps=self.hand_interp_steps,
        )
        move_cfg = MoveActionCfg(
            control_part=arm_name,
            sample_interval=self.move_sample_interval,
        )

        motion_generator = MotionGenerator(
            cfg=MotionGenCfg(planner_cfg=ToppraPlannerCfg(robot_uid=self.robot.uid))
        )
        engine = AtomicActionEngine(
            robot=self.robot,
            motion_generator=motion_generator,
            device=self.device,
            actions_cfg_dict={
                "pick_up": pickup_cfg,
                "place": place_cfg,
                "move": move_cfg,
            },
        )

        engine.register_action(
            "pick_up",
            PickUpAction(
                motion_generator=motion_generator,
                cfg=pickup_cfg,
            ),
        )
        engine.register_action(
            "place",
            PlaceAction(
                motion_generator=motion_generator,
                cfg=place_cfg,
            ),
        )
        engine.register_action(
            "move",
            MoveAction(
                motion_generator=motion_generator,
                cfg=move_cfg,
            ),
        )
        return engine

    def _build_bowl_semantics(
        self,
        bowl_uid: str,
        annotator_port: int,
    ) -> ObjectSemantics:
        """Create a fresh semantic object for a bowl grasp stage."""
        bowl = self._get_bowl(bowl_uid)
        affordance = AntipodalAffordance(
            object_label=bowl_uid,
            force_reannotate=self.force_reannotate,
            custom_config={
                "gripper_collision_cfg": self.gripper_collision_cfg,
                "generator_cfg": GraspGeneratorCfg(
                    viser_port=annotator_port,
                    antipodal_sampler_cfg=AntipodalSamplerCfg(
                        n_sample=20000,
                        max_length=0.088,
                        min_length=0.003,
                    ),
                    max_deviation_angle=np.pi / 3,
                ),
            },
        )
        return ObjectSemantics(
            label=bowl_uid,
            geometry={
                "mesh_vertices": bowl.get_vertices(env_ids=[0], scale=True)[0],
                "mesh_triangles": bowl.get_triangles(env_ids=[0])[0],
            },
            affordance=affordance,
            entity=bowl,
        )

    def _build_place_pose(
        self,
        bowl_max_pose: torch.Tensor,
        arm_name: str,
        place_offset: torch.Tensor,
    ) -> torch.Tensor:
        """Build the placement pose above the largest bowl."""
        place_pose = bowl_max_pose.clone()
        place_pose[:, :3, :3] = self._get_down_rotation(arm_name)
        place_pose[:, :3, 3] = place_pose[:, :3, 3] + place_offset

        debug_verbose = 1
        if debug_verbose:
            # This task only supports a single env in demo generation,
            # so we log env0 for readability.
            bowl_max_xyz = bowl_max_pose[0, :3, 3].detach().cpu().tolist()
            place_xyz = place_pose[0, :3, 3].detach().cpu().tolist()
            place_rot = place_pose[0, :3, :3].detach().cpu().tolist()
            left_mid_offset = (
                self.left_bowl_mid_place_offset.detach().cpu().tolist()
            )
            left_min_offset = (
                self.left_bowl_min_place_offset.detach().cpu().tolist()
            )
            logger.log_warning(
                f"[StackBowlsAtomic Pose Debug] arm_name={arm_name}, "
                f"bowl_max_pose[:3,3]={bowl_max_xyz}, "
                f"left_bowl_mid_place_offset={left_mid_offset}, "
                f"left_bowl_min_place_offset={left_min_offset}, "
                f"place_pose[:3,3]={place_xyz}"
            )
            logger.log_warning(
                f"[StackBowlsAtomic Pose Debug] place_pose[:3,:3]={place_rot}"
            )
        return place_pose

    def _get_extension_value(self, key: str, default: Any) -> Any:
        """Read one environment extension value from dict-like or object config."""
        extensions = self.cfg.extensions
        if hasattr(extensions, "get"):
            return extensions.get(key, default)
        return getattr(extensions, key, default)

    def _get_bowl(self, bowl_uid: str) -> RigidObject:
        """Return the bowl rigid object by uid."""
        return self.sim.get_rigid_object(bowl_uid)

    def _get_eef_name(self, arm_name: str) -> str:
        """Map an arm control part to its corresponding gripper control part."""
        return "left_eef" if arm_name == "left_arm" else "right_eef"

    def _get_arm_joint_ids(self, arm_name: str) -> list[int]:
        """Return joint ids for the selected arm."""
        return self.left_arm_ids if arm_name == "left_arm" else self.right_arm_ids

    def _get_down_rotation(self, arm_name: str) -> torch.Tensor:
        """Return the original arm-specific downward placement rotation."""
        return self.left_down_rota if arm_name == "left_arm" else self.right_down_rota

    def _get_mid_place_offset(self, arm_name: str) -> torch.Tensor:
        """Return the original bowl_mid placement offset for one arm."""
        if arm_name == "left_arm":
            return self.left_bowl_mid_place_offset
        return self.right_bowl_mid_place_offset

    def _get_min_place_offset(self, arm_name: str) -> torch.Tensor:
        """Return the original bowl_min placement offset for one arm."""
        if arm_name == "left_arm":
            return self.left_bowl_min_place_offset
        return self.right_bowl_min_place_offset

    def _get_home_pose(self, arm_name: str) -> torch.Tensor:
        """Return the original end-effector home pose for the selected arm."""
        if arm_name == "left_arm":
            return self.init_left_arm_xpos.clone()
        return self.init_right_arm_xpos.clone()

    def _is_success(self, is_success: bool | torch.Tensor) -> bool:
        """Convert scalar or tensor planning status to a Python bool."""
        if isinstance(is_success, torch.Tensor):
            return bool(torch.all(is_success).item())
        return bool(is_success)
