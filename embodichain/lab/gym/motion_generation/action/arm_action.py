# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------

import torch
import numpy as np
from copy import deepcopy
from typing import Dict, List, Tuple, Union
from embodichain.utils import logger
from embodichain.lab.gym.motion_generation.action.action import Action
from embodichain.lab.gym.motion_generation.planner.utils import (
    TrajectorySampleMethod,
)
from embodichain.lab.gym.motion_generation.planner.toppra_planner import (
    ToppraPlanner,
)


class ArmAction(Action):
    r"""Initialize the ArmAction class."""

    def __init__(self, env, robot_uid, **kwargs) -> None:
        super().__init__(env, **kwargs)
        self.agent_uid = robot_uid
        if "LeftManipulator" == robot_uid or "RightManipulator" == robot_uid:
            self.agent = self.scene.get_robot("DualManipulator")
        else:
            self.agent = self.scene.get_robot(self.agent_uid)

        init_qpos = self.agent.get_init_qpos(self.agent_uid)
        self.init_ee_xpos = self.agent.get_fk(qpos=init_qpos, uid=self.agent_uid)
        self.init_base_xpos = self.agent.get_base_xpos(self.agent_uid)

        self.drive_controller = self.agent.drive_controllers[self.agent_uid]

    def move(
        self,
        xpos_list: np.ndarray,
        is_linear: bool = False,
        is_wait: bool = True,
        is_world_coordinates=False,
        **kwargs,
    ):
        r"""Move the robot to a specified position.

        Args:
            xpos_list (np.ndarray): List of target positions.
            is_linear (bool): If True, move in a linear path.
            is_wait (bool): If True, wait until the movement is completed.
            is_world_coordinates (bool): If True, interpret positions in world coordinates.
            kwargs (dict): Additional arguments.

        Returns:
            bool: True if movement is successful, else False.
        """
        if hasattr(self.agent, "move"):
            res = self.agent.move(
                xpos_list,
                is_linear=is_linear,
                is_wait=is_wait,
                is_world_coordinates=is_world_coordinates,
            )

            return res
        else:
            return False

    def move_in_joints(
        self,
        qpos_list: np.ndarray,
        is_linear: bool = False,
        is_wait: bool = True,
        **kwargs,
    ):
        r"""Move the robot joints to specified positions.

        Args:
            qpos_list (np.ndarray): List of target joint positions.
            is_linear (bool): If True, move joints in a linear path.
            is_wait (bool): If True, wait until the movement is completed.
            kwargs (dict): Additional arguments.

        Returns:
            bool: True if movement is successful, else False.
        """
        if hasattr(self.agent, "move_in_joints"):
            res = self.agent.move_in_joints(
                qpos_list, is_linear=is_linear, is_wait=is_wait
            )
            return res
        else:
            return False

    def apply_transform(
        self, xpos: np.ndarray, lift_vector: np.ndarray, is_local: bool = False
    ):
        """Apply a lift to the given pose in either local or world coordinates.

        Args:
            pick_xpos (np.ndarray): The original 4x4 transformation matrix.
            lift_vector (np.ndarray): A 3-element vector representing the lift in [x, y, z] directions.
            is_local (bool): If True, apply the lift in local coordinates;
                        if False, apply in world coordinates.

        Returns:
            np.ndarray: The new 4x4 transformation matrix after applying the lift.
        """
        xpos = np.array(xpos)
        lift_vector = np.array(lift_vector)
        # Assert to ensure xpos is a 4x4 matrix and lift_vector has three components
        assert xpos.shape == (4, 4), "Target pose must be a 4x4 matrix."
        assert lift_vector.shape == (
            3,
        ), "Lift vector must have three components [x, y, z]."

        # Create a copy of the xpos
        new_xpos = deepcopy(xpos)

        # Create a translation matrix for lifting in world coordinates
        translation_matrix = np.array(
            [
                [1, 0, 0, lift_vector[0]],
                [0, 1, 0, lift_vector[1]],
                [0, 0, 1, lift_vector[2]],
                [0, 0, 0, 1],
            ]
        )

        if is_local:
            # Apply lift in local coordinates
            new_xpos = new_xpos @ translation_matrix
        else:
            # Apply the translation in the world coordinate system
            new_xpos = translation_matrix @ new_xpos

        return new_xpos

    @staticmethod
    def create_discrete_trajectory(
        agent,
        uid,
        xpos_list: List[np.ndarray] = None,
        is_use_current_qpos: bool = True,
        is_linear: bool = False,
        sample_method: TrajectorySampleMethod = TrajectorySampleMethod.QUANTITY,
        sample_num: Union[float, int] = 20,
        qpos_seed: np.ndarray = None,
        **kwargs,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        r"""Generate a discrete trajectory between waypoints using cartesian or joint space interpolation.

        This method supports two trajectory planning approaches:
        1. Linear interpolation: Fast, uniform spacing, no dynamics constraints
        2. ToppraPlanner: Smooth, considers velocity/acceleration limits, realistic motion

        Args:
            agent: The robot agent instance
            uid: Unique identifier for the robot agent
            xpos_list: List of waypoints as 4x4 transformation matrices
            is_use_current_qpos: Whether to use current joint angles as IK seed
            is_linear: If True, use cartesian linear interpolation, else joint space
            sample_method: Sampling method for ToppraPlanner (QUANTITY or TIME)
            sample_num: Number of interpolated points for final trajectory
            qpos_seed: Initial joint configuration for IK solving
            **kwargs: Additional arguments:
                - qpos_list: Optional list of joint configurations

        Returns:
            A tuple containing:
            - List[np.ndarray]: Joint space trajectory as a list of joint configurations
            - List[np.ndarray]: Cartesian space trajectory as a list of 4x4 matrices
        """
        from scipy.spatial.transform import Rotation, Slerp
        import numpy as np

        def interpolate_xpos(
            current_xpos: np.ndarray, target_xpos: np.ndarray, num_samples: int
        ) -> list[np.ndarray]:
            if num_samples < 2:
                num_samples = 2

            slerp = Slerp(
                [0, 1],
                Rotation.from_matrix([current_xpos[:3, :3], target_xpos[:3, :3]]),
            )
            interpolated_poses = []
            for s in np.linspace(0, 1, num_samples):
                interp_rot = slerp(s).as_matrix()
                interp_trans = (1 - s) * current_xpos[:3, 3] + s * target_xpos[:3, 3]
                interp_pose = np.eye(4)
                interp_pose[:3, :3] = interp_rot
                interp_pose[:3, 3] = interp_trans
                interpolated_poses.append(interp_pose)
            return interpolated_poses

        def calculate_point_allocations(
            xpos_list, step_size=0.002, angle_step=np.pi / 90
        ):
            point_allocations = []

            for i in range(len(xpos_list) - 1):
                start_pose = xpos_list[i]
                end_pose = xpos_list[i + 1]

                if isinstance(start_pose, torch.Tensor):
                    start_pose = start_pose.squeeze().cpu().numpy()
                if isinstance(end_pose, torch.Tensor):
                    end_pose = end_pose.squeeze().cpu().numpy()

                pos_dist = np.linalg.norm(end_pose[:3, 3] - start_pose[:3, 3])
                pos_points = max(1, int(pos_dist / step_size))

                angle_diff = Rotation.from_matrix(
                    start_pose[:3, :3].T @ end_pose[:3, :3]
                )
                angle = abs(angle_diff.as_rotvec()).max()
                rot_points = max(1, int(angle / angle_step))

                num_points = max(pos_points, rot_points)
                point_allocations.append(num_points)

            return point_allocations

        def create_qpos_dict(position: np.ndarray, dof: int) -> Dict:
            """Create qpos dictionary with zero velocity and acceleration"""
            return {
                "position": (
                    position.tolist() if isinstance(position, np.ndarray) else position
                ),
                "velocity": [0.0] * dof,
                "acceleration": [0.0] * dof,
            }

        if hasattr(agent, "get_dof"):
            agent_dof = agent.get_dof(uid)
        elif hasattr(agent, "control_parts"):
            agent_dof = len(agent.control_parts[uid])

        # TODO(@Jietao Chen): max_constraints should be read from URDF file
        max_constraints = kwargs.get("max_constraints", None)
        if max_constraints is None:
            max_constraints = {
                "velocity": [0.2] * agent_dof,
                "acceleration": [0.5] * agent_dof,
            }
        planner = ToppraPlanner(agent_dof, max_constraints)

        out_qpos_list = []
        out_xpos_list = []

        # Handle input arguments
        qpos_list = kwargs.get("qpos_list", None)
        if qpos_list is not None:
            qpos_list = np.asarray(qpos_list)
            # TODO: It will use computed fk in the future
            if hasattr(agent, "get_fk"):
                xpos_list = [agent.get_fk(uid=uid, qpos=q) for q in qpos_list]
            elif hasattr(agent, "compute_fk"):
                qpos_list = (
                    torch.tensor(qpos_list)
                    if not isinstance(qpos_list, torch.Tensor)
                    else qpos_list
                )
                xpos_list = [
                    agent.compute_fk(qpos=q, name=uid, to_matrix=True)
                    for q in qpos_list
                ]
            else:
                logger.log_warning("Agent does not support FK computation")

        if is_use_current_qpos:
            current_qpos = agent.get_current_qpos(uid)
            # TODO: It will use computed fk in the future
            if hasattr(agent, "get_fk"):
                current_xpos = agent.get_fk(uid=uid, qpos=current_qpos)
            elif hasattr(agent, "compute_fk"):
                current_xpos = agent.compute_fk(
                    qpos=current_qpos, name=uid, to_matrix=True
                )
            else:
                logger.log_warning("Agent does not support FK computation")
                return [], []

            pos_diff = np.linalg.norm(current_xpos[:3, 3] - xpos_list[0][:3, 3])
            rot_diff = np.linalg.norm(current_xpos[:3, :3] - xpos_list[0][:3, :3])

            if pos_diff > 0.001 or rot_diff > 0.01:
                xpos_list = np.concatenate(
                    [current_xpos[None, :, :], xpos_list], axis=0
                )
                if qpos_list is not None:
                    qpos_list = np.concatenate(
                        [current_qpos[None, :], qpos_list], axis=0
                    )

        if qpos_seed is None and qpos_list is not None:
            qpos_seed = qpos_list[0]

        # Input validation
        if xpos_list is None or len(xpos_list) < 2:
            logger.log_warning("xpos_list must contain at least 2 points")
            return [], []

        # Calculate point allocations for interpolation
        interpolated_point_allocations = calculate_point_allocations(
            xpos_list, step_size=0.002, angle_step=np.pi / 90
        )

        # Generate trajectory
        interpolate_qpos_list = []
        if is_linear or qpos_list is None:
            # Linear cartesian interpolation
            for i in range(len(xpos_list) - 1):
                interpolated_poses = interpolate_xpos(
                    xpos_list[i], xpos_list[i + 1], interpolated_point_allocations[i]
                )

                for xpos in interpolated_poses:
                    # TODO: It will use computed ik in the future
                    if hasattr(agent, "get_ik"):
                        success, qpos = agent.get_ik(xpos, qpos_seed=qpos_seed, uid=uid)
                    elif hasattr(agent, "compute_ik"):
                        success, qpos = agent.compute_ik(
                            pose=xpos, joint_seed=qpos_seed, name=uid
                        )
                    else:
                        logger.log_warning("Agent does not support IK computation")

                    if not success:
                        logger.log_warning(f"IK solving failed for pose {xpos}")
                        return [], []
                    interpolate_qpos_list.append(qpos)
                    qpos_seed = qpos
        else:
            # Joint space interpolation
            interpolate_qpos_list = qpos_list

        # Create trajectory dictionary
        current_qpos_dict = create_qpos_dict(interpolate_qpos_list[0], agent_dof)
        target_qpos_dict_list = [
            create_qpos_dict(pos, agent_dof) for pos in interpolate_qpos_list[1:]
        ]

        # Plan trajectory
        res, out_qpos_list, *_ = planner.plan(
            current_qpos_dict,
            target_qpos_dict_list,
            sample_method=sample_method,
            sample_interval=sample_num,
        )
        if not res:
            logger.log_warning("Failed to plan trajectory with ToppraPlanner")
            return [], []

        # TODO: It will use computed fk in the future
        if hasattr(agent, "get_fk"):
            out_xpos_list = [agent.get_fk(uid=uid, qpos=q) for q in out_qpos_list]
        elif hasattr(agent, "compute_fk"):
            out_qpos_list = (
                torch.tensor(out_qpos_list)
                if not isinstance(out_qpos_list, torch.Tensor)
                else out_qpos_list
            )
            out_xpos_list = [
                agent.compute_fk(qpos=q, name=uid, to_matrix=True)
                for q in out_qpos_list
            ]
        else:
            logger.log_warning("Agent does not support FK computation")

        return out_qpos_list, out_xpos_list

    @staticmethod
    def estimate_trajectory_sample_count(
        agent,
        uid,
        xpos_list: List[np.ndarray] = None,
        qpos_list: List[np.ndarray] = None,
        step_size: float = 0.01,
        angle_step: float = np.pi / 90,
        **kwargs,
    ) -> int:
        """Estimate the number of trajectory sampling points required.

        This function estimates the total number of sampling points needed to generate
        a trajectory based on the given waypoints and sampling parameters. It can be
        used to predict computational load and memory requirements before actual
        trajectory generation.

        Args:
            agent: Robot agent instance
            uid: Unique identifier for the robot agent
            xpos_list: List of 4x4 transformation matrices representing waypoints
            qpos_list: List of joint positions (optional)
            is_linear: Whether to use linear interpolation
            step_size: Maximum allowed distance between consecutive points (in meters)
            angle_step: Maximum allowed angular difference between consecutive points (in radians)
            **kwargs: Additional parameters for further customization

        Returns:
            int: Estimated number of trajectory sampling points
        """

        def rotation_matrix_to_angle(self, rot_matrix: np.ndarray) -> float:
            cos_angle = (np.trace(rot_matrix) - 1) / 2
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            return np.arccos(cos_angle)

        # Input validation
        if xpos_list is None and qpos_list is None:
            return 0

        # If joint position list is provided but end effector position list is not,
        # convert through forward kinematics
        if qpos_list is not None and xpos_list is None:
            if len(qpos_list) < 2:
                return 1 if len(qpos_list) == 1 else 1
            try:
                if hasattr(agent, "get_fk_batch"):
                    xpos_list = agent.get_fk_batch(uid=uid, qpos_list=qpos_list)
                else:
                    xpos_list = [agent.get_fk(uid=uid, qpos=q) for q in qpos_list]
            except Exception as e:
                logger.log_warning(f"Forward kinematics failed: {e}")
                return 0

        if xpos_list is None or len(xpos_list) == 0:
            return 1

        if len(xpos_list) == 1:
            return 1

        total_samples = 1  # Starting point
        angle_step_inv = 1.0 / angle_step

        total_pos_dist = 0.0
        total_angle = 0.0

        for i in range(len(xpos_list) - 1):
            start_pose = xpos_list[i]
            end_pose = xpos_list[i + 1]

            pos_diff = end_pose[:3, 3] - start_pose[:3, 3]
            total_pos_dist += np.linalg.norm(pos_diff)

            try:
                rot_matrix = start_pose[:3, :3].T @ end_pose[:3, :3]
                angle = rotation_matrix_to_angle(rot_matrix)
                total_angle += angle
            except Exception:
                pass

        pos_samples = max(1, int(total_pos_dist / step_size))
        rot_samples = max(1, int(total_angle / angle_step))

        total_samples = max(pos_samples, rot_samples)

        return max(2, total_samples)

    def create_action_dict_list(
        self,
        xpos_list: List[np.ndarray],
        qpos_list: List[np.ndarray],
        ee_state: float = 0.0,
    ) -> List[Dict]:
        """Constructs a list of actions based on the given end effector poses on agent base coordinates and joint positions.

        Args:
            xpos_list (List[np.ndarray]): A list of end effector poses.
            qpos_list (List[np.ndarray]): A list of joint positions.
            ee_state (float, optional): The state of the end effector (e.g., open or closed). Defaults to 0.0.

        Returns:
            List[Dict]: A list of actions, where each action contains:
                        - "ef_pose": The end effector pose at the step.
                        - "qpos": The joint positions corresponding to the step.
                        - "ee_state": The state of the end effector (e.g., open or closed).
        """
        # Check if xpos_list or qpos_list is None
        if xpos_list is None or qpos_list is None:
            return []

        # Check if xpos_list and qpos_list have the same length
        if len(xpos_list) != len(qpos_list):
            logger.log_warning("The xpos_list and qpos_list must have the same length.")
            return []

        action_list = [
            {
                "ef_pose": xpos_list[i],
                "qpos": qpos_list[i],
                "ee_state": ee_state,
            }
            for i in range(len(xpos_list))
        ]

        return action_list

    def create_back_action_list(
        self,
        start_xpos: np.ndarray = None,
        is_move_linear: bool = False,
        qpos_seed: np.ndarray = None,
        lift_height: float = 0.25,
        reference_xpos: np.ndarray = None,
        back_distance_z: float = 0.02,
        traj_num: int = 20,
        **kwargs,
    ) -> List[Dict]:
        r"""Generate a list of actions for the robot to move back to its initial joint position after completing a task.

        Args:
            start_xpos (np.ndarray, optional): The starting position of the end effector (EE) in agent base coordinates,
                                                represented as a 4x4 transformation matrix. If None, the agent's current EE
                                                position is used. Defaults to None.
            is_move_linear (bool, optional): True for linear movement and False for joint space interpolation. Defaults to False.
            qpos_seed (np.ndarray, optional): Qpos seed for solving Inverse Kinematics (IK). Defaults to None, which uses the current Qpos.
            lift_height (float, optional): The vertical distance the EE should be lifted. Defaults to 0.25 meters.
            reference_xpos (np.ndarray, optional): An optional reference position used to compute the back path. If None, the path will be a simple lift and return. Defaults to None.
            back_distance_z (float, optional): Distance to offset reference_xpos in the -z direction. Defaults to 0.02.
            traj_num (int, optional): The number of discrete steps (trajectory points) to generate for the move back action.
                                    More steps result in a smoother trajectory. Defaults to 20.
            **kwargs: Additional parameters for further customization.

        Returns:
            List[Dict]: A list of actions, where each action is represented as a dictionary containing:
                        - "ef_pose": The end effector pose at the step.
                        - "qpos": The joint positions corresponding to the step.
                        - "ee_state": The state of the end effector (e.g., open or closed).

        Note:
            - The initial joint position ('init_qpos') is the home configuration of the robot's joints, representing the agent's Qpos
            at the start or in a safe/rest position. It serves as a fallback in case no valid IK solutions are found for the lifting position,
            ensuring that the robot can still return to its last known configuration before the task.
        """
        if start_xpos is None:
            start_xpos = self.agent.get_current_xpos(
                name=self.agent_uid, is_world_coordinates=False
            )

        if reference_xpos is None:
            lift_xpos = self.apply_transform(
                xpos=start_xpos, lift_vector=[0.0, 0.0, lift_height], is_local=False
            )
            back_path = [start_xpos, lift_xpos]
        else:
            z_back_xpos = self._compute_offset_xpos(
                start_xpos, reference_xpos, back_distance_z
            )
            lift_xpos = self.apply_transform(
                xpos=z_back_xpos, lift_vector=[0.0, 0.0, lift_height], is_local=False
            )
            back_path = [start_xpos, z_back_xpos, lift_xpos]

        back_qpos_path = []

        for p in back_path:
            res, qpos = self.drive_controller.get_ik(p, qpos_seed)
            if res:
                back_qpos_path.append(qpos)

        init_qpos = self.agent.get_current_qpos(self.agent_uid)
        if back_qpos_path:
            back_qpos_path.append(init_qpos)
        else:
            back_qpos_path = [init_qpos, init_qpos]

        qpos_list, xpos_list = self.drive_controller.create_discrete_trajectory(
            qpos_list=back_qpos_path,
            is_use_current_qpos=False,
            is_linear=is_move_linear,
            sample_num=traj_num,
            qpos_seed=qpos_seed,
        )

        action_list = self.create_action_dict_list(
            xpos_list=xpos_list,
            qpos_list=qpos_list,
            ee_state=self.end_effector.open_state,
        )
        if not action_list:
            logger.log_warning(
                "Create approach action list failed. Please check the approach path!"
            )

        return action_list

    def supplyment_action_data(
        self, action_list: Dict, connected_qpos: np.ndarray = None
    ) -> Dict:
        r"""Supplement the action data for a DualManipulator agent.

            This function checks if the agent is a DualManipulator and determines the
            appropriate end effector index based on the end effector's unique identifier.
            It retrieves the current open states of the end effectors and updates the
            provided action list with new joint positions and end effector states.

            Args:
                action_list (Dict): A list of actions to be modified, where each action
                                    contains 'qpos' for joint positions and 'ee_state' for end effector states.
                connected_qpos (connected_qpos):

            Returns:
                Dict: The modified action list with updated joint positions and end
        `             effector states. If the agent is not a DualManipulator or if the end
                      effector UID is invalid, the original action list is returned.
        """
        if self.agent.__class__.__name__ != "DualManipulator":
            return action_list

        # TODO: Does the number here really correspond to the ee obtained by step_action?
        if "Left" in self.end_effector_uid:
            ee_idx = 0
        elif "Right" in self.end_effector_uid:
            ee_idx = 1
        else:
            logger.log_warning("There is no left or right gripper, no processing.")
            return action_list

        all_ee_state = np.array([])
        # TODO: Here we assume that the results are obtained in the order of left and then right.
        ee_list = self.env.get_end_effector()
        for ee in ee_list:
            ee_open_state = ee.get_open_state()
            all_ee_state = np.append(all_ee_state, ee_open_state)

        if connected_qpos is None:
            current_qpos = self.agent.get_current_qpos("DualManipulator")
        else:
            current_qpos = connected_qpos

        target_joint_ids = self.agent.get_joint_ids(self.agent_uid)

        left_current_xpos = self.agent.get_current_xpos("LeftManipulator")
        right_current_xpos = self.agent.get_current_xpos("RightManipulator")

        all_xpos = np.array([left_current_xpos, right_current_xpos])

        for action in action_list:
            new_qpos = np.copy(current_qpos)
            new_qpos[target_joint_ids] = action["qpos"]
            action["qpos"] = new_qpos

            all_xpos[ee_idx] = action["ef_pose"]
            action["ef_pose"] = all_xpos

            new_ee_state = np.copy(all_ee_state)
            new_ee_state[ee_idx] = action["ee_state"]
            action["ee_state"] = new_ee_state

        return action_list

    def merge_action_data(
        self, left_action_list: Dict, right_action_list: Dict
    ) -> Dict:
        r"""Merge action data from left and right action lists.

        This function is designed to combine action data from two separate action
        lists (left and right) into a single unified action list. The implementation
        details for merging the action lists will depend on the specific requirements
        of the application.

        Args:
            left_action_list (Dict): A dictionary containing actions for the left end effector.
            right_action_list (Dict): A dictionary containing actions for the right end effector.

        Returns:
            Dict: A merged dictionary containing combined action data from both
            left and right action lists. The exact structure of the returned
            dictionary will depend on the merging logic implemented in this method.
        """
        merged_action_list = []
        if self.agent.__class__.__name__ != "DualManipulator":
            return merged_action_list

        current_qpos = self.agent.get_current_qpos("DualManipulator")
        # Assuming both action lists have the same length
        for left_action, right_action in zip(left_action_list, right_action_list):
            merged_action = {}

            # Get joint IDs for left and right actions
            left_joint_ids = self.agent.get_joint_ids("LeftManipulator")
            right_joint_ids = self.agent.get_joint_ids("RightManipulator")

            # Initialize new qpos and ee_state for the merged action
            new_qpos = np.zeros(
                len(current_qpos)
            )  # Assuming total count includes both left and right

            # Set joint positions based on left action
            new_qpos[left_joint_ids] = left_action["qpos"][left_joint_ids]

            # Set joint positions based on right action
            new_qpos[right_joint_ids] = right_action["qpos"][right_joint_ids]

            # Set end effector states
            new_ee_state = []  # Assuming two end effectors: left and right
            new_ee_state.extend(left_action["ee_state"])
            new_ee_state.extend(right_action["ee_state"])

            # Construct the merged action
            merged_action["qpos"] = new_qpos
            merged_action["ee_state"] = new_ee_state

            # Append the merged action to the list
            merged_action_list.append(merged_action)

        return merged_action_list
