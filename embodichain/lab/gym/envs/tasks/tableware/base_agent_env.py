import torch
from embodichain.utils import logger


class BaseAgentEnv:

    def _init_agents(self, agent_config, task_name, agent_config_path=None):
        from embodichain.agents.hierarchy.task_agent import TaskAgent
        from embodichain.agents.hierarchy.code_agent import CodeAgent
        from embodichain.agents.hierarchy.validation_agent import ValidationAgent
        from embodichain.agents.hierarchy.llm import (
            task_llm,
            code_llm,
            validation_llm,
        )

        if agent_config.get("TaskAgent") is not None:
            self.task_agent = TaskAgent(
                task_llm,
                **agent_config["Agent"],
                **agent_config["TaskAgent"],
                task_name=task_name,
                config_dir=agent_config_path,
            )
        self.code_agent = CodeAgent(
            code_llm,
            **agent_config["Agent"],
            **agent_config.get("CodeAgent"),
            task_name=task_name,
            config_dir=agent_config_path,
        )
        self.validation_agent = ValidationAgent(
            validation_llm,
            task_name=task_name,
            task_description=self.code_agent.prompt_kwargs.get("task_prompt")[
                "content"
            ],
            basic_background=self.code_agent.prompt_kwargs.get("basic_background")[
                "content"
            ],
            atom_actions=self.code_agent.prompt_kwargs.get("atom_actions")["content"],
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

        # store some useful obj information
        init_obj_info = {}
        obj_uids = self.sim.get_rigid_object_uid_list()
        for obj_name in obj_uids:
            obj = self.sim.get_rigid_object(obj_name)
            obj_pose = obj.get_local_pose(to_matrix=True).squeeze(0)
            obj_height = obj_pose[2, 3]  # Extract the height (z-coordinate)
            obj_grasp_pose = self.affordance_datas.get(
                f"{obj_name}_grasp_pose_object", None
            )
            init_obj_info[obj_name] = {
                "pose": obj_pose,  # Store the full pose (4x4 matrix)
                "height": obj_height,  # Store the height (z-coordinate)
                "grasp_pose_obj": (
                    obj_grasp_pose.squeeze(0) if obj_grasp_pose is not None else None
                ),  # Store the grasp pose if available
            }
        self.init_obj_info = init_obj_info

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
    def generate_code_for_actions(self, regenerate=False, **kwargs):
        logger.log_info(
            f"Generate code for creating action list for {self.code_agent.task_name}.",
            color="green",
        )

        # Task planning
        print(f"\033[92m\nStart task planning.\n\033[0m")

        task_agent_input = self.task_agent.get_composed_observations(
            env=self, regenerate=regenerate, **kwargs
        )
        task_plan = self.task_agent.generate(**task_agent_input)

        # Code generation
        print(f"\033[94m\nStart code generation.\n\033[0m")
        code_agent_input = self.code_agent.get_composed_observations(
            env=self, regenerate=regenerate, **kwargs
        )
        code_agent_input["task_plan"] = task_plan

        code_file_path, kwargs, code = self.code_agent.generate(**code_agent_input)
        return code_file_path, kwargs, code

    # -------------------- get action list --------------------
    def create_demo_action_list(self, regenerate=False, *args, **kwargs):
        code_file_path, kwargs, _ = self.generate_code_for_actions(
            regenerate=regenerate
        )
        action_list = self.code_agent.act(code_file_path, **kwargs)
        self.action_length = len(action_list)
        return action_list
