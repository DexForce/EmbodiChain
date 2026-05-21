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

import torch
from embodichain.utils import logger


class BaseAgentEnv:

    def _init_agents(self, agent_config, task_name, agent_config_path=None):
        from embodichain.agents.hierarchy.task_agent import TaskAgent
        from embodichain.agents.hierarchy.compile_agent import CompileAgent
        from embodichain.agents.hierarchy.recovery_agent import RecoveryAgent
        from embodichain.agents.hierarchy.llm import (
            task_llm,
            compile_llm,
            recovery_llm,
        )

        self.task_agent = TaskAgent(
            task_llm,
            **agent_config["Agent"],
            **agent_config["TaskAgent"],
            task_name=task_name,
            config_dir=agent_config_path,
        )
        self.recovery_agent = RecoveryAgent(
            recovery_llm,
            **agent_config["Agent"],
            **agent_config["RecoveryAgent"],
            task_name=task_name,
            config_dir=agent_config_path,
        )
        self.compile_agent = CompileAgent(
            compile_llm,
            **agent_config["Agent"],
            **agent_config["CompileAgent"],
            task_name=task_name,
            config_dir=agent_config_path,
        )

    def get_states(self):
        # TODO: only support num_env = 1 for now
        # store robot states in each env.reset
        self.init_qpos = self.robot.get_qpos().squeeze(0)

        self.left_arm_joints = self.robot.get_joint_ids(name="left_arm")
        self.right_arm_joints = self.robot.get_joint_ids(name="right_arm")
        self.left_eef_joints = self.robot.get_joint_ids(name="left_eef")
        self.right_eef_joints = self.robot.get_joint_ids(name="right_eef")

        self.left_arm_init_qpos = self.init_qpos[self.left_arm_joints]
        self.right_arm_init_qpos = self.init_qpos[self.right_arm_joints]

        self.left_arm_init_xpos = self.robot.compute_fk(
            self.left_arm_init_qpos, name="left_arm", to_matrix=True
        ).squeeze(0)
        self.right_arm_init_xpos = self.robot.compute_fk(
            self.right_arm_init_qpos, name="right_arm", to_matrix=True
        ).squeeze(0)

        self.left_arm_current_qpos = self.left_arm_init_qpos
        self.right_arm_current_qpos = self.right_arm_init_qpos

        self.left_arm_current_xpos = self.left_arm_init_xpos
        self.right_arm_current_xpos = self.right_arm_init_xpos

        self.left_arm_base_pose = self.robot.get_control_part_base_pose(
            "left_arm", to_matrix=True
        ).squeeze(0)
        self.right_arm_base_pose = self.robot.get_control_part_base_pose(
            "right_arm", to_matrix=True
        ).squeeze(0)

        self.open_state = torch.tensor([0.05])
        self.close_state = torch.tensor([0.0])
        self.left_arm_current_gripper_state = self.open_state
        self.right_arm_current_gripper_state = self.open_state

        self.update_obj_info()

    def update_obj_info(self):
        # store some useful obj information
        obj_info = getattr(self, "obj_info", {})
        obj_uids = self.sim.get_rigid_object_uid_list()
        for obj_name in obj_uids:
            obj = self.sim.get_rigid_object(obj_name)
            obj_pose = obj.get_local_pose(to_matrix=True).squeeze(0)

            if obj_name not in obj_info:
                obj_height = obj_pose[2, 3]  # Extract the height (z-coordinate)
                obj_grasp_pose = self.affordance_datas.get(
                    f"{obj_name}_grasp_pose_object", None
                )
                obj_info[obj_name] = {
                    "initial_pose": obj_pose.clone(),
                    "pose": obj_pose,  # Store the full pose (4x4 matrix)
                    "height": obj_height,  # Store the initial height (z-coordinate)
                    "grasp_pose_obj": (
                        obj_grasp_pose.squeeze(0)
                        if obj_grasp_pose is not None
                        else None
                    ),  # Store the grasp pose if available
                }
            else:
                obj_info[obj_name]["pose"] = obj_pose

        self.obj_info = obj_info

    # -------------------- Common getters / setters --------------------

    def get_obs_for_agent(self):
        obs = self.get_obs()
        rgb = obs["sensor"]["cam_high"]["color"].squeeze(0)

        result = {"rgb": rgb}
        validation_event = (
            getattr(self.event_manager.cfg, "validation_cameras", None)
            if self.event_manager is not None
            else None
        )
        if validation_event is not None:
            validation_functor = self.event_manager.get_functor("validation_cameras")
            if validation_functor is not None:
                camera_data = validation_functor(self, None)
                result.update({k: v.squeeze(0) for k, v in camera_data.items()})
        return result

    def get_current_qpos_agent(self):
        return self.left_arm_current_qpos, self.right_arm_current_qpos

    def set_current_qpos_agent(self, arm_qpos, is_left):
        if is_left:
            self.left_arm_current_qpos = arm_qpos
        else:
            self.right_arm_current_qpos = arm_qpos

    def get_current_xpos_agent(self):
        return self.left_arm_current_xpos, self.right_arm_current_xpos

    def set_current_xpos_agent(self, arm_xpos, is_left):
        if is_left:
            self.left_arm_current_xpos = arm_xpos
        else:
            self.right_arm_current_xpos = arm_xpos

    def get_current_gripper_state_agent(self):
        return self.left_arm_current_gripper_state, self.right_arm_current_gripper_state

    def set_current_gripper_state_agent(self, arm_gripper_state, is_left):
        if is_left:
            self.left_arm_current_gripper_state = arm_gripper_state
        else:
            self.right_arm_current_gripper_state = arm_gripper_state

    # -------------------- IK / FK --------------------
    def get_arm_ik(self, target_xpos, is_left, qpos_seed=None):
        control_part = "left_arm" if is_left else "right_arm"
        ret, qpos = self.robot.compute_ik(
            name=control_part, pose=target_xpos, joint_seed=qpos_seed
        )
        return ret.all().item(), qpos.squeeze(0)

    def get_arm_fk(self, qpos, is_left):
        control_part = "left_arm" if is_left else "right_arm"
        xpos = self.robot.compute_fk(
            name=control_part, qpos=torch.as_tensor(qpos), to_matrix=True
        )
        return xpos.squeeze(0)

    # -------------------- get compiled graph for action list --------------------
    def generate_graph_for_actions(self, regenerate=False, recovery=False, **kwargs):
        logger.log_info(
            f"Generate graph for creating {'recovery' if recovery else ''} action list for {self.compile_agent.task_name}.",
            color="yellow" if recovery else "green",
        )

        print(f"\033[92m\nStart task graph generation.\n\033[0m")
        task_agent_input = self.task_agent.get_composed_observations(
            env=self,
            regenerate=regenerate,
            observations=self.get_obs_for_agent(),
            **kwargs,
        )
        task_graph = self.task_agent.generate(**task_agent_input)

        recovery_spec = None
        if recovery:
            print(f"\033[91m\nStart recovery spec generation.\n\033[0m")
            recovery_agent_input = self.recovery_agent.get_composed_observations(
                env=self,
                regenerate=regenerate,
                task_graph=task_graph,
                **kwargs,
            )
            recovery_spec = self.recovery_agent.generate(**recovery_agent_input)

        print(f"\033[94m\nStart graph compilation.\n\033[0m")
        compile_agent_input = self.compile_agent.get_composed_observations(
            env=self,
            regenerate=regenerate,
            task_graph=task_graph,
            recovery_spec=recovery_spec,
            recovery_enabled=recovery,
            **kwargs,
        )
        graph_file_path, kwargs, graph_content = self.compile_agent.generate(
            **compile_agent_input
        )

        return graph_file_path, kwargs, graph_content

    # -------------------- get action list --------------------
    def create_demo_action_list(
        self, regenerate=False, recovery=False, *args, **kwargs
    ):
        graph_file_path, compile_kwargs, _ = self.generate_graph_for_actions(
            regenerate=regenerate, recovery=recovery, **kwargs
        )
        compile_kwargs["interactive_error_injection"] = kwargs.get(
            "interactive_error_injection", False
        )
        if "forced_recovery_injection" in kwargs:
            compile_kwargs["forced_recovery_injection"] = kwargs[
                "forced_recovery_injection"
            ]
        if "disable_recovery_branches" in kwargs:
            compile_kwargs["disable_recovery_branches"] = kwargs[
                "disable_recovery_branches"
            ]
        for key in (
            "use_public_atomic_actions",
            "use_atomic_action_graph",
            "require_atomic_action_graph",
            "use_public_grasp_action",
            "require_public_grasp_action",
            "use_public_grasp_semantics",
            "allow_public_grasp_annotation",
            "force_public_grasp_reannotate",
            "recovery_public_grasp_strategy",
            "public_grasp_candidate_num",
            "recovery_public_grasp_candidate_num",
            "public_grasp_pre_grasp_distance",
            "recovery_public_grasp_pre_grasp_distance",
            "generate_public_grasp_candidates",
            "public_grasp_auto_approach_direction",
            "recovery_public_grasp_auto_approach_direction",
            "public_grasp_try_approach_directions",
            "recovery_public_grasp_try_approach_directions",
            "public_grasp_approach_direction",
            "recovery_public_grasp_approach_direction",
            "public_grasp_approach_directions",
            "recovery_public_grasp_approach_directions",
            "public_grasp_lift_height",
            "recovery_public_grasp_lift_height",
            "public_grasp_pose_offset_world",
            "recovery_public_grasp_pose_offset_world",
            "public_grasp_pose_offset_along_approach",
            "recovery_public_grasp_pose_offset_along_approach",
            "validate_public_grasp_after_action",
            "recovery_validate_public_grasp_after_action",
            "public_grasp_validation_min_object_lift",
            "public_grasp_validation_max_object_lift",
            "public_grasp_validation_max_object_xy_displacement",
            "recovery_public_grasp_validation_min_object_lift",
            "recovery_public_grasp_validation_max_object_lift",
            "recovery_public_grasp_validation_max_object_xy_displacement",
            "recovery_public_grasp_rank_by_legacy_pose",
            "recovery_public_grasp_use_legacy_orientation",
            "grasp_max_open_length",
            "grasp_min_open_length",
            "grasp_finger_length",
            "grasp_x_thickness",
            "grasp_y_thickness",
            "grasp_root_z_width",
            "grasp_open_check_margin",
            "grasp_point_sample_dense",
            "grasp_antipodal_n_sample",
            "grasp_max_deviation_angle",
            "use_public_place_action",
            "public_place_upright",
            "public_place_upright_eps",
            "use_public_gripper_action",
            "require_public_non_grasp_actions",
            "allow_move_relative_orientation_fallback",
            "force_valid",
            "log_dir",
            "runtime_llm_recovery",
            "prefer_runtime_llm_recovery",
            "runtime_recovery_use_llm",
            "runtime_recovery_max_total_attempts",
            "runtime_recovery_max_monitor_attempts",
            "runtime_recovery_max_exception_attempts",
        ):
            if key in kwargs:
                compile_kwargs[key] = kwargs[key]
        if kwargs.get("runtime_llm_recovery", False):
            compile_kwargs["runtime_recovery_agent"] = self.recovery_agent
        action_list = self.compile_agent.act(graph_file_path, **compile_kwargs)
        return action_list
