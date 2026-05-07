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

import torch
from embodichain.utils import logger


class BaseAgentEnv:

    def _init_agents(self, agent_config, task_name, agent_config_path=None):
        from embodichain.agents.hierarchy.task_agent import TaskAgent
        from embodichain.agents.hierarchy.code_agent import CodeAgent
        from embodichain.agents.hierarchy.failure_anticipation_agent import FailureAnticipationAgent
        from embodichain.agents.hierarchy.llm import task_llm, code_llm, failure_anticipation_llm

        self.task_agent = TaskAgent(
            task_llm,
            **agent_config["Agent"],
            **agent_config["TaskAgent"],
            task_name=task_name,
            config_dir=agent_config_path,
        )
        self.failure_anticipation_agent = FailureAnticipationAgent(
            failure_anticipation_llm,
            **agent_config["Agent"],
            **agent_config["FailureAnticipationAgent"],
            task_name=task_name,
            config_dir=agent_config_path,
        )
        self.code_agent = CodeAgent(
            code_llm,
            **agent_config["Agent"],
            **agent_config["CodeAgent"],
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
        obj_info = {}
        obj_uids = self.sim.get_rigid_object_uid_list()
        for obj_name in obj_uids:
            obj = self.sim.get_rigid_object(obj_name)
            obj_pose = obj.get_local_pose(to_matrix=True).squeeze(0)
            obj_height = obj_pose[2, 3]  # Extract the height (z-coordinate)
            obj_grasp_pose = self.affordance_datas.get(
                f"{obj_name}_grasp_pose_object", None
            )
            obj_info[obj_name] = {
                "pose": obj_pose,  # Store the full pose (4x4 matrix)
                "height": obj_height,  # Store the height (z-coordinate)
                "grasp_pose_obj": (
                    obj_grasp_pose.squeeze(0) if obj_grasp_pose is not None else None
                ),  # Store the grasp pose if available
            }
        self.obj_info = obj_info

    # -------------------- Common getters / setters --------------------

    def get_obs_for_agent(self):
        obs = self.get_obs()
        rgb = obs["sensor"]["cam_high"]["color"].squeeze(0)

        # Get validation camera data
        camera_data = self.event_manager.get_functor("validation_cameras")(self, None)
        result = {"rgb": rgb}
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

    # -------------------- get only code for action list --------------------
    def generate_code_for_actions(self, regenerate=False, recovery=False, **kwargs):
        logger.log_info(
            f"Generate code for creating {'recovery' if recovery else ''} action list for {self.code_agent.task_name}.",
            color="yellow" if recovery else "green",
        )

        # Task planning
        print(f"\033[92m\nStart task planning.\n\033[0m")

        task_agent_input = self.task_agent.get_composed_observations(
            env=self, regenerate=regenerate, observations=self.get_obs_for_agent(), **kwargs
        )
        task_plan = self.task_agent.generate(**task_agent_input)

        # Failure anticipation
        anticipated_failures = ''
        if recovery:
            print(f"\033[91m\nStart failure anticipation for recovery.\n\033[0m")
            failure_anticipation_input = self.failure_anticipation_agent.get_composed_observations(
                env=self, regenerate=regenerate, task_plan=task_plan, **kwargs
            )
            anticipated_failures = self.failure_anticipation_agent.generate(
                **failure_anticipation_input
            )

        # Code generation
        print(f"\033[94m\nStart code generation.\n\033[0m")
        code_agent_input = self.code_agent.get_composed_observations(
            env=self, regenerate=regenerate, task_plan=task_plan, anticipated_failures=anticipated_failures, **kwargs
        )
        code_file_path, kwargs, code = self.code_agent.generate(**code_agent_input)

        return code_file_path, kwargs, code

    # -------------------- get action list --------------------
    def create_demo_action_list(self, regenerate=False, recovery=False, *args, **kwargs):
        code_file_path, kwargs, _ = self.generate_code_for_actions(
            regenerate=regenerate, recovery=recovery
        )
        action_list = self.code_agent.act(code_file_path, **kwargs)
        return action_list
