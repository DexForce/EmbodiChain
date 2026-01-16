import torch
from embodichain.utils import logger
import traceback
from embodichain.data import database_agent_prompt_dir
from pathlib import Path
import os
from embodichain.toolkits.interfaces import extract_drive_calls, draw_axis
from embodichain.agents.hierarchy.code_agent import format_execution_history
from embodichain.agents.hierarchy.validation_agent import (
    save_obs_image,
    get_obj_position_info,
)


class BaseAgentEnv:

    def _init_agents(self, agent_config, task_name):
        from embodichain.agents.hierarchy.task_agent import TaskAgent
        from embodichain.agents.hierarchy.code_agent import CodeAgent
        from embodichain.agents.hierarchy.validation_agent import ValidationAgent
        from embodichain.agents.hierarchy.llm import (
            create_llm,
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
            )
        self.code_agent = CodeAgent(
            code_llm,
            **agent_config["Agent"],
            **agent_config.get("CodeAgent"),
            task_name=task_name,
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
        obs = self.get_obs(get_valid_sensor_data=True)
        rgb = obs["sensor"]["cam_high"]["color"].squeeze(0)
        valid_rgb_1 = obs["sensor"]["valid_cam_1"]["color"].squeeze(0)
        valid_rgb_2 = obs["sensor"]["valid_cam_2"]["color"].squeeze(0)
        valid_rgb_3 = obs["sensor"]["valid_cam_3"]["color"].squeeze(0)

        # obs_image_path = save_obs_image(obs_image=self.get_obs_for_agent()["rgb_1"], save_dir='./', step_id=0)

        return {
            "rgb": rgb,
            "valid_rgb_1": valid_rgb_1,
            "valid_rgb_2": valid_rgb_2,
            "valid_rgb_3": valid_rgb_3,
        }

        # depth = obs["sensor"]["cam_high"]["depth"].squeeze(0)
        # mask = obs["sensor"]["cam_high"]["mask"].squeeze(0)
        # semantic_mask = obs["sensor"]["cam_high"]["semantic_mask_l"].squeeze(0)
        # return {"rgb": rgb, "depth": depth, "mask": mask, "semantic_mask": semantic_mask}

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

        # # Task planning (not used currently)
        # print(f"\033[92m\nStart task planning.\n\033[0m")
        # task_agent_input = self.task_agent.get_composed_observations(env=self)
        # query = self.task_agent.generate(**task_agent_input, regenerate=regenerate, **kwargs)

        # Code generation
        print(f"\033[94m\nStart code generation.\n\033[0m")
        code_agent_input = self.code_agent.get_composed_observations(
            env=self, regenerate=regenerate, **kwargs
        )
        code_file_path, kwargs, code = self.code_agent.generate(**code_agent_input)
        return code_file_path, kwargs, code

    # -------------------- get action list --------------------
    def create_demo_action_list(self, regenerate=False):
        code_file_path, kwargs, _ = self.generate_code_for_actions(
            regenerate=regenerate
        )
        action_list = self.code_agent.act(code_file_path, **kwargs)
        return action_list

    def create_demo_action_list_with_self_correction(self, **kwargs):
        logger.log_info(
            f"Generate code for creating action list for {self.code_agent.task_name} with self correction.",
            color="green",
        )

        # Create log file name with timestamp
        import datetime

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = (
            Path(database_agent_prompt_dir)
            / self.code_agent.task_name
            / "self_correction_logs"
            / timestamp
        )
        os.makedirs(log_dir, exist_ok=True)
        img_dir = log_dir / "observation_images"

        kwargs.setdefault("env", self)
        kwargs.setdefault("log_dir", log_dir)
        kwargs.setdefault("file_path", log_dir / "agent_generated_code.py")
        kwargs.setdefault("md_path", log_dir / "agent_llm_responses.md")
        kwargs.setdefault("last_task_plan", "None.")
        kwargs.setdefault("last_executed_failure", "None.")
        kwargs.setdefault("last_executed_history", "None.")

        # TODO: rethink which part should be divided to task / code agents. Important!
        # TODO: use the task agent to select which needs the validation (mainly interaction with the objects), not all steps.
        # TODO: add logs
        # TODO： maybe use a sequence of images for task planning

        step_id = 0
        save_obs_image(
            obs_image=self.get_obs_for_agent()["valid_rgb_1"],
            save_dir=img_dir / "cam_1",
            step_id=step_id,
        )
        save_obs_image(
            obs_image=self.get_obs_for_agent()["valid_rgb_2"],
            save_dir=img_dir / "cam_2",
            step_id=step_id,
        )
        save_obs_image(
            obs_image=self.get_obs_for_agent()["valid_rgb_3"],
            save_dir=img_dir / "cam_3",
            step_id=step_id,
        )

        task_agent_input = self.task_agent.get_composed_observations(**kwargs)
        code_agent_input = self.code_agent.get_composed_observations(**kwargs)
        while True:
            exec_code = []
            print(f"\033[94m\nStart task planning.\n\033[0m")
            task_plan, plan_list, validation_list = (
                self.task_agent.generate_for_correction(
                    img_dir=img_dir / "cam_1", **task_agent_input
                )
            )

            # TODO: maybe here I need to insert an error-occurred agent, calling some error-occurred apis, maybe with correction action too.
            # TODO：maybe the validation agent can provide correction action, and no need to generate the subsequent full task by the task agent.

            print(f"\033[92m\nStart code generation.\n\033[0m")
            code_agent_input, code = self.code_agent.generate_according_to_task_plan(
                task_plan=task_plan, **code_agent_input
            )
            drive_list = extract_drive_calls(code)
            for action_id, single_action in enumerate(drive_list):
                try:
                    # ---------- execute ----------
                    self.code_agent.act_single_action(single_action, **code_agent_input)
                    exec_success = True
                    exec_trace = None

                    # # # # TODO: manually adjust the bottle pose for testing
                    # if step_id == 2:
                    #
                    #     # pose = torch.tensor(
                    #     #     [[[0.99989, -0.00457, -0.01415, 0.72850],
                    #     #       [0.00457, 0.99999, -0.00041, -0.20441],
                    #     #       [0.01415, 0.00034, 0.99990, 0.92571],
                    #     #       [0.00000, 0.00000, 0.00000, 1.00000]]],
                    #     #     dtype=torch.float32
                    #     # )
                    #     # self.sim.get_rigid_object('bottle').set_local_pose(pose)
                    #
                    #     pose = torch.tensor(
                    #         [[[0.99989, -0.00457, -0.01415, 0.722850],
                    #           [0.00457, 0.99999, -0.00041, 0.20441],
                    #           [0.01415, 0.00034, 0.99990, 0.92571],
                    #           [0.00000, 0.00000, 0.00000, 1.00000]]],
                    #         dtype=torch.float32
                    #     )
                    #     self.sim.get_rigid_object('cup').set_local_pose(pose)
                    #
                    #     # pose = self.sim.get_rigid_object('spoon').get_local_pose(to_matrix=True).squeeze(0)
                    #     # pose[0, 3] = 0.6
                    #     # pose[1, 3] = -0.35
                    #     # pose[2, 3] = 0.8
                    #     # self.sim.get_rigid_object('spoon').set_local_pose(pose.unsqueeze(0))
                    #
                    #     for i in range(5):
                    #         _ = self.step(action=self.robot.get_qpos())

                except Exception:
                    exec_success = False
                    exec_trace = traceback.format_exc()
                    print(f"Execution failed:\n{exec_trace}")

                # ---------- step transition ----------
                step_id += 1

                save_obs_image(
                    obs_image=self.get_obs_for_agent()["valid_rgb_1"],
                    save_dir=img_dir / "cam_1",
                    step_id=step_id,
                )
                save_obs_image(
                    obs_image=self.get_obs_for_agent()["valid_rgb_2"],
                    save_dir=img_dir / "cam_2",
                    step_id=step_id,
                )
                save_obs_image(
                    obs_image=self.get_obs_for_agent()["valid_rgb_3"],
                    save_dir=img_dir / "cam_3",
                    step_id=step_id,
                )

                # ---------- post-execution handling ----------
                if exec_success:
                    if code_agent_input.get("validation_agent"):
                        print(
                            f"\033[33mStarting validation with condition '{validation_list[action_id]}'!\033[0m"
                        )
                        validation_info = self.validation_agent.validate_single_action(
                            single_action,
                            plan_list[action_id],
                            validation_list[action_id],
                            img_dir,
                            get_obj_position_info(self),
                        )

                        if "SUCCESS" in validation_info:
                            print(f"\033[33mValid info:\n{validation_info}\033[0m")
                            is_success = True
                            exec_code.append(plan_list[action_id])
                            continue
                        else:
                            print(f"\033[31mValid info:\n{validation_info}\033[0m")
                            info = (
                                "Validation Result: FAILED\n\n"
                                "Failed Step (currently executing step):\n"
                                f"{plan_list[action_id]}\n\n"
                                "Failure Analysis (why this step failed):\n"
                                f"{validation_info}"
                            )
                            history = (
                                "Executed History (previous steps):\n"
                                f"{format_execution_history(exec_code)}\n\n"
                            )
                            is_success = False
                    else:
                        is_success = True
                        exec_code.append(plan_list[action_id])
                        continue
                else:
                    info = (
                        "Action Execution: FAILED\n\n"
                        "Failed Step (currently executing step):\n"
                        f"{plan_list[action_id]}\n\n"
                        "Execution Error Trace:\n"
                        f"{exec_trace}\n\n"
                        "Note: You may try `force_valid=True` for the current action to find the nearest valid pose."
                    )
                    history = (
                        "Executed History (previous steps):\n"
                        f"{format_execution_history(exec_code)}\n\n"
                    )

                    is_success = False

                task_agent_input["last_task_plan"] = task_plan
                task_agent_input["last_executed_failure"] = info
                task_agent_input["last_executed_history"] = history
                break

            if single_action == drive_list[-1] and is_success:
                # ---------- termination ----------
                print(
                    "\033[91mExecuted all the plans. The task is considered complete.\033[0m"
                )
                break
