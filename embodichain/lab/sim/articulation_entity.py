# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------

import numpy as np
import typing
from typing import List, Tuple, Union, Dict, Any
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field

import dexsim
from dexsim.models import Entity
from dexsim.engine import Articulation
from dexsim.types import DriveType, PhysicalAttr, ArticulationFlag
from embodichain.utils import logger

# Try to import DriveController, but make it optional
try:
    from rlia.kit.drive_controllers import DriveController
except ImportError:
    # If rlia is not available, use Any as a fallback type
    DriveController = Any

from dexsim.utility import inv_transform
from dexsim.utility.env_utils import load_first_environment

__all__ = ["ArticulationEntity"]


@dataclass
class ArticulationPosition:
    r"""Represents the position of an articulation in a robotic system.

    Attributes:
        init_qpos (Union[np.ndarray, Dict[str, np.ndarray]]):
            The initial joint positions of the articulation, which can be a
            NumPy array or a dictionary mapping joint names to their initial
            positions.

        init_base_xpos (Union[np.ndarray, Dict[str, np.ndarray]], optional):
            The initial base position of the articulation, which can also be a
            NumPy array or a dictionary mapping base names to their initial
            positions. Defaults to None.
    """

    init_qpos: Union[np.ndarray, Dict[str, np.ndarray]] = field(default_factory=dict)
    init_base_xpos: Union[np.ndarray, Dict[str, np.ndarray]] = None


@dataclass
class ArticulationControl:
    r"""Controls the behavior of an articulation in a robotic system.

    Attributes:
        speed_ratio (float):
            The ratio of speed for the articulation control. Default is 0.5.

        time_step (float):
            The time step for control updates in seconds. Default is 0.02.

        drive_type (DriveType):
            The type of drive used for the articulation control. Default is 'TARGET'.
    """

    speed_ratio: float = 0.5
    time_step: float = 0.02
    drive_type: "DriveType" = "TARGET"


@dataclass
class ArticulationJointConfiguration:
    link_names: List[str] = field(default_factory=list)
    joint_names: List[str] = field(default_factory=list)

    root_link_name: str = field(default_factory=dict)
    end_link_name: str = field(default_factory=dict)


class ArticulationEntity(metaclass=ABCMeta):
    r"""
    Abstract class for articulation entity in simulation.
    """

    def __init__(
        self,
        urdf_path: Union[str, List[str]] = dict(),
        init_qpos: Union[np.ndarray, Dict[str, np.ndarray]] = dict(),
        init_base_xpos: Union[np.ndarray, Dict[str, np.ndarray]] = None,
        speed_ratio: float = 0.5,
        time_step: float = 0.02,
        drive_type: DriveType = DriveType.FORCE,
        env: dexsim.environment.Arena = None,
        **kwargs,
    ):
        r"""Initialize the articulation entity.

        Args:
            urdf_path (str): urdf file path of robot
            init_qpos (np.ndarray, optional): [dof] of double. Init robot joint state(home joint state).
            init_base_xpos (np.ndarray, optional): [4, 4] of double. Robot base pose in arena coordinate system.
            speed_ratio (float, optional): 0 ~ 1. Robot speed ratio.
            time_step (float, optional): wait time between two update. Defaults to 1/50.
            drive_type (DriveType, optional): DriveType.FORCE or DriveType.FORCE. Defaults to DriveType.FORCE.
            env (Arena, optional): dexsim.environment.Arena. Load the first world(None defaults).
            kwargs(optional): Accepts additional keyword arguments.
        """
        # placeholder for articulations to be created to the robot.
        # a robot can have multiple articulations, for example,
        # 1. a arm with a gripper (manipulator)
        # 2. two arms
        # 3. mobile manipulator
        self.articulation = None

        ## Additional variable for DualManipulator, Humanoids and DexterousHands:
        # Dictionary to map child to its parent articulation "self.articulation"
        self.child_articulations: Dict[str, Articulation] = dict()

        # URDF file path(s) for the robot
        self.urdf_path = urdf_path

        # initial joint positions of the robot.
        self.init_qpos = init_qpos

        # initial base pose of the robot in arena coordinate system.
        self.init_base_xpos = init_base_xpos

        # Dictionary to store degrees of freedom for each articulation
        self._dof: Dict[str, int] = dict()

        # Dictionary for actual control joint indices of articulations
        self._joint_ids: Dict[str, np.ndarray] = dict()

        # self._actived_joint_names = dict()

        # TODO: Maybe turn to dict stored joint pos, vel, acc limits.
        # List to store the limits for each joint's motion.
        self._joint_limit = []

        # placeholder for actors to attach to the robot.
        self.attached_actors: Dict[str, Entity] = dict()

        # Dictionary to map control group names to their corresponding root link names,
        # used for accessing the base position of each control group.
        self.root_link_names: Dict[str] = kwargs.get("root_link_names", {})

        # Dictionary to map control group names to their corresponding end link names,
        # used for accessing the terminal position of each control group.
        self.end_link_names: Dict[str] = kwargs.get("end_link_names", {})

        # Speed ratio for the robot's movement
        self.speed_ratio = speed_ratio

        # Time step for control updates
        self.time_step = time_step

        # Validate and set the drive type
        if drive_type not in [DriveType.FORCE, DriveType.FORCE]:
            logger.log_error(f"Invalid drive type: {drive_type}.")
        self.drive_type = drive_type

        # Dictionary to map child to its parent init_base_xpos "self.init_base_xpos"
        self.child_init_base_xpos = dict()

        # Dictionaries for drive and task controllers
        self.drive_controllers: Dict[str, DriveController] = dict()

        # Load the first environment if not provided
        self._env, self._world = load_first_environment(env)

    def get_articulation(self, uid: str = None) -> dexsim.engine.Articulation:
        r"""Get articulation based on its unique identifier (uid).

        This method returns the articulation associated with the provided uid.
        If uid is not specified (None), it returns all articulations. If the
        uid is invalid, a warning is logged, and None is returned.

        Args:
            uid (str, optional): The unique identifier for the articulation. If None, all articulations will be returned.

        Returns:
            dexsim.engine.Articulation or Dict: The articulation corresponding to the provided uid, or a dictionary of all articulations if uid is None. Returns None if the uid is invalid.
        """

        if uid is None or uid == self.uid:
            return self.articulation

        if uid in self.child_articulations:
            return self.child_articulations[uid]
        else:
            logger.log_warning(
                f"Current uid {self.uid} cannot find the corresponding Articulation."
            )
            return None

    def _setup_child_articulations(self, uid: str, control_parts: Dict):
        r"""Initialize child articulations and establish a mapping between parent and child articulations.

        This method sets up child articulations associated with a parent articulation identified by its UID.
        It verifies the existence of the parent articulation before proceeding to initialize the child articulations.

        Args:
            uid (str): The unique identifier (UID) of the parent articulation.
            control_parts (Dict): A dictionary of control parts to initialize as child articulations.

        Returns:
            bool: True if the child articulations were successfully set up; False otherwise.
        """
        # Use a list comprehension to filter valid control parts and log warnings for the invalid ones
        control_parts_dict = {}

        # Check if the articulation is valid and if the provided UID matches the instance's UID
        if self.articulation is None or uid != self.uid:
            logger.log_warning(f"Articulation with UID '{uid}' not found.")
            return False

        # Iterate over control parts to set up child articulations
        for control_part in control_parts:
            # Add to child articulations
            control_parts_dict[control_part] = self.articulation

        # Establish the relationship between the child articulations and their parent
        self.child_articulations = control_parts_dict

        return True

    @property
    def default_physical_attrs(self) -> PhysicalAttr:
        physical_attr = PhysicalAttr()
        if self.drive_type == DriveType.FORCE:
            physical_attr.static_friction = 1.0
            physical_attr.dynamic_friction = 0.9
            physical_attr.linear_damping = 0.7
            physical_attr.angular_damping = 0.7
            physical_attr.contact_offset = 0.005
            physical_attr.rest_offset = 0.001
            physical_attr.restitution = 0.05
            physical_attr.has_gravity = True
            physical_attr.max_linear_velocity = 4000
            physical_attr.max_angular_velocity = 25
            physical_attr.max_depenetration_velocity = 1e1
        else:  # DriveType.FORCE and so on
            physical_attr.static_friction = 1.0
            physical_attr.dynamic_friction = 0.9
            physical_attr.linear_damping = 0.7
            physical_attr.angular_damping = 0.7
            physical_attr.contact_offset = 0.005
            physical_attr.rest_offset = 0.001
            physical_attr.restitution = 0.05
            physical_attr.has_gravity = False
            physical_attr.max_linear_velocity = 1e6
            physical_attr.max_angular_velocity = 1e6
            physical_attr.max_depenetration_velocity = 1e1
        return physical_attr

    @property
    def default_drive_param(self) -> Dict:
        # Stiffness:
        #   Recommended range: 2000 N/m to 10000 N/m
        #   Note: Higher stiffness is suitable for tasks that require precise position control,
        #         such as gripping and assembly. You can start with 5000 N/m and fine-tune based on feedback from the actual application.
        # Damping:
        #   Recommended range: 200 Ns/m to 1000 Ns/m
        #   Note: Damping values ​​should be high enough to dampen oscillations,
        #         but not too high to excessively hinder motion. You can start with 500 Ns/m and adjust based on dynamic performance.
        # Max force:
        #   Recommended range: 10000 N to 100000 N
        #   Note: The maximum force should be set according to the load capacity of the robot arm
        #         to ensure that it does not exceed its load capacity when working. You can start with 50000 N, depending on the specific task load.
        if self.drive_type == DriveType.FORCE:
            param = {"stiffness": 2e3, "damping": 2e2, "max_force": 2e4}
        elif self.drive_type == DriveType.FORCE:
            param = {"stiffness": 1e8, "damping": 1e6, "max_force": 1e10}
        return param

    def set_uid(self, uid: str) -> None:
        r"""Set unique id of the robot.

        Args:
            uid (str): Unique id of the robot.
        """
        if uid == self.uid:
            logger.log_warning(
                f"The uid: {uid} is the same as the current: {self.uid}."
            )
        else:
            self.uid = uid

    def get_urdf_path(self) -> str:
        r"""Provides the file path to the Unified Robot Description Format (URDF) file.

        Returns:
            str: A string representing the file path to the robot's URDF file.
        """
        return self.urdf_path

    def get_dof(self, name: str = None) -> Union[int, Dict[str, int]]:
        r"""Get degree of freedom (DoF) of the robot.

        Args:
            name (str, optional): Name of the articulation. Defaults to None.

        Returns:
            Union[int, Dict[str, int]]:
                - If `name` is None, returns the total DoF of the robot as an integer.
                - If `name` is provided and found, returns the DoF of the specified articulation as an integer.
                - If `name` is provided but not found, logs a warning and returns 0.
        """
        # TODO: Need to clarify behavior.
        if name is None:
            if isinstance(self._dof, dict):
                return sum(self._dof.values())
            else:
                return (
                    self._dof
                )  # Assuming _dof is an integer representing the total DoF
        elif name in self._dof:
            return self._dof[
                name
            ]  # Assuming _dof[name] is an integer representing the DoF of the specified articulation

        logger.log_warning(f"Articulation '{name}' not found.")
        return 0

    def _convert_pose(self, pose: np.ndarray, is_to_arena: bool) -> np.ndarray:
        r"""Convert a given pose to the specified coordinate system.

        Args:
            pose (np.ndarray): A [4, 4] transformation matrix representing the pose to be converted.
            is_to_arena (bool): If True, convert to arena coordinate system; otherwise, convert to world coordinate system.

        Returns:
            np.ndarray: A [4, 4] transformation matrix representing the pose in the specified coordinate system.
        """
        if pose is None:
            return np.eye(4)

        pose_array = np.array(pose)

        if pose_array.shape == (4, 4):
            poses_to_convert = [pose_array]
        elif pose_array.ndim == 3 and pose_array.shape[1:] == (4, 4):
            poses_to_convert = pose_array
        else:
            logger.log_warning(f"Invalid shape for pose: {pose.shape}")
            return np.eye(4)

        # Retrieve the world pose of the arena's root node
        arena_root_pose = self._env.get_root_node().get_world_pose()

        # Determine the transformation logic based on the value of is_to_arena
        if is_to_arena:
            # Apply the inverse transformation to convert to the arena coordinate system
            inv_arena_root_pose = np.linalg.inv(arena_root_pose)
            converted_poses = [inv_arena_root_pose @ p for p in poses_to_convert]
        else:
            # Directly apply the transformation to convert to the world coordinate system
            converted_poses = [arena_root_pose @ p for p in poses_to_convert]

        # Return the result in the same format as the input
        if pose_array.shape == (4, 4):
            return converted_poses[0]  # Return single pose
        else:
            return np.array(converted_poses)  # Return list/array of poses

    def set_joint_ids(self, joint_ids: np.ndarray, uid: str = None):
        r"""Set joint IDs for the given UID.

        Args:
            joint_ids (np.ndarray): Joint IDs to set.
            uid (str, optional): The unique identifier for the joint. Defaults to None.
        """
        uid = uid or self.uid
        self._joint_ids[uid] = joint_ids

    def get_joint_ids(self, name: str = None) -> List:
        r"""Gets joint IDs from the internal storage.

        Args:
            name (str, optional): The name of the joint to look up.
                                If None, all joint IDs are returned.

        Returns:
            List: A list of joint IDs associated with the specified name,
                or a dictionary of all joint IDs if no name is given.
                Returns an empty list if the name is not found.
        """
        if name is None:
            return {key: value for key, value in self._joint_ids.items()}
        if name in self._joint_ids:
            return self._joint_ids[name]
        else:
            logger.log_warning(
                f"Joint ids with name '{name}' not found in self._joint_ids."
            )
            return []

    def get_joint_limits(
        self, name: str = None
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        r"""Get joint limits for the specified articulation.

        Args:
            name (str): Name of the articulation. Defaults to None.

        Returns:
            np.ndarray: [dof, 2] of float. Lower and upper joint limits.
            Dict[str, np.ndarray]: [dof, 2] of float. Lower and upper joint limits for all articulations.
        """
        limits = self.articulation.get_joint_limits()

        if name is None:
            return limits
        else:
            if self.uid == name:
                return limits[self._joint_ids[name]]

            if name not in self.child_articulations:
                logger.log_warning(f"Articulation '{name}' not found.")
                return None
            return limits[self._joint_ids[name]]

    def get_link_names(
        self, name: str = None
    ) -> Union[List[str], Dict[str, List[str]]]:
        r"""Gets the list of link names for a given articulation.

        Args:
            name (str, optional): The name of the articulation. If None, returns link names for all articulations.

        Returns:
            List[str]: A list of link names for the specified articulation if `name` is provided.
            Dict[str, List[str]]: A dictionary mapping articulation names to their respective link name lists if `name` is None.
            None: Returns None if the specified articulation name is not found.
        """
        # todo: Articulation needs to distinguish between some parents and children.
        link_names = self.articulation.get_link_names()

        if name is None or name == self.uid:
            # Return a dictionary of link names for all articulations
            return link_names
        else:
            if name in self.child_articulations:
                return link_names[self._joint_ids[name]]

    def _get_link_velocity(
        self, name: str = None, is_linear: bool = True, is_root: bool = False
    ) -> Union[np.ndarray, None]:
        r"""Get the link velocity of the specified articulation.

        Args:
            name (str, optional): Name of the articulation. If None, retrieves velocities for all articulations.
            is_linear (bool, optional): If True, retrieves linear velocity; otherwise, retrieves angular velocity.
            is_root (bool, optional): If True, returns the root link velocity as a flattened array.

        Returns:
            Union[np.ndarray, None]: Returns the velocity of the specified joint as a numpy array, or None if not found.
        """

        def _get_link_velocity_helper(
            name: str, is_linear: bool = True, is_root: bool = False
        ) -> typing.Optional[np.ndarray]:
            """Helper function to get the link velocity for a specific articulation."""
            if name == self.uid:
                link_general_vel = self.articulation.get_link_general_velocities()
                link_velocity = (
                    link_general_vel[:, :3] if is_linear else link_general_vel[:, 3:]
                )
                return link_velocity[0].reshape(-1) if is_root else link_velocity
            elif name in self.child_articulations:
                link_general_vel = self.child_articulations[
                    name
                ].get_link_general_velocities()
                link_velocity = (
                    link_general_vel[:, :3] if is_linear else link_general_vel[:, 3:]
                )
                return link_velocity[0].reshape(-1) if is_root else link_velocity
            else:
                return None

        if name is None:
            link_velocity = _get_link_velocity_helper(
                name=self.uid, is_linear=is_linear, is_root=is_root
            )
        else:
            link_velocity = _get_link_velocity_helper(
                name=name, is_linear=is_linear, is_root=is_root
            )

        return link_velocity

    def get_body_link_linear_velocity(
        self,
        name: str = None,
    ) -> Union[np.ndarray, None]:
        r"""Get body link linear velocity in coordinate frame.

        Args:
            name (str, optional): The name of the articulation.
                                If None, retrieves the velocity of all articulations.

        Returns:
            Union[np.ndarray, None]:
                If a name is provided, returns an array of shape [link_num, 3]
                representing the linear velocity of the specified articulation.
                If name is None, returns a dictionary mapping articulation names
                to their corresponding linear velocities.
        """
        return self._get_link_velocity(name=name, is_linear=True, is_root=False)

    def get_body_link_angular_velocity(
        self,
        name: str = None,
    ) -> Union[np.ndarray, None]:
        r"""Get body link angular velocity in coordinate frame.

        Args:
            name (str, optional): The name of the articulation.
                                If None, retrieves the velocity of all articulations.

        Returns:
            Union[np.ndarray, None]:
                If a name is provided, returns an array of shape [link_num, 3]
                representing the angular velocity of the specified articulation.
                If name is None, returns a dictionary mapping articulation names
                to their corresponding angular velocities.
        """
        return self._get_link_velocity(name=name, is_linear=False, is_root=False)

    def get_root_link_linear_velocity(
        self,
        name: str = None,
    ) -> Union[np.ndarray, None]:
        r"""Get root link linear velocity in coordinate frame.

        Args:
            name (str, optional): The name of the articulation.
                                If None, retrieves the velocity of all articulations.

        Returns:
            Union[np.ndarray, None]:
                If a name is provided, returns an array of shape [3]
                representing the linear velocity of the root link.
                If name is None, returns a dictionary mapping articulation names
                to their corresponding linear velocities.
        """
        return self._get_link_velocity(name=name, is_linear=True, is_root=True)

    def get_root_link_angular_velocity(
        self,
        name: str = None,
    ) -> Union[np.ndarray, None]:
        r"""Get root link angular velocity in coordinate frame.

        Args:
            name (str, optional): The name of the articulation.
                                If None, retrieves the velocity of all articulations.

        Returns:
            Union[np.ndarray, None]:
                If a name is provided, returns an array of shape [3]
                representing the angular velocity of the root link.
                If name is None, returns a dictionary mapping articulation names
                to their corresponding angular velocities.
        """
        return self._get_link_velocity(name=name, is_linear=False, is_root=True)

    def _set_articulation_property(
        self,
        name: str,
        property_name: str,
        value: Union[np.ndarray, Dict[str, np.ndarray]],
        use_params: bool = True,
        **params,
    ) -> bool:
        r"""Helper function to set a property for a specific articulation.

        This function attempts to set a specified property (e.g., position, velocity)
        for the articulation identified by 'name'. It first checks if the articulation
        is a child articulation and then checks the main articulations. If the
        articulation is found and the property exists, the function sets the property
        with the provided value.

        Args:
            name (str): The name of the articulation to set the property for.
            property_name (str): The name of the property to set.
            value (Union[np.ndarray, Dict[str, np.ndarray]]): The value to set the property to.
            use_params (bool): Whether to use params when calling the property method.

        Returns:
            bool: True if the property was successfully set, False otherwise.
        """
        # Use self._joint_ids[name] if params is empty
        if use_params and not params:
            params = {"joint_ids": self._joint_ids[name]}

        # Check in child articulations first
        if name in self.child_articulations:
            child_articulation = self.child_articulations[name]
            if hasattr(child_articulation, property_name):
                # Call the property method with or without params
                if use_params:
                    getattr(child_articulation, property_name)(value, **params)
                else:
                    getattr(child_articulation, property_name)(value)
                return True

        # Check the main articulation
        if name == self.uid:
            if hasattr(self.articulation, property_name):
                # Call the property method with or without params
                if use_params:
                    getattr(self.articulation, property_name)(value, **params)
                else:
                    getattr(self.articulation, property_name)(value)
                return True

        logger.log_warning(f"Articulation '{name}' not found.")
        return False

    def get_current_xpos(
        self, name: str = None, is_world_coordinates: bool = True
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        r"""Get the current pose of the articulations.

        This method retrieves the current pose of specified articulation(s) in either world
        or base coordinates. It handles both single articulations and hierarchical structures
        with parent-child relationships.

        Args:
            name (str, optional): Name of the articulation. Defaults to None.
            is_world_coordinates (bool, optional):
                Whether to use the arena(world) coordinate system(WCS) or the Base
                coordinate system(BCS). Defaults to True.

        Returns:
            Union[np.ndarray, Dict[str, np.ndarray]]:
                Returns the xpos for the specified articulation if `name` is provided and found.
                If `name` is None, returns xpos for all articulations.
                Returns None if `name` is provided but not found.
        """

        # Function to calculate the current position based on qpos
        def calculate_xpos(
            key: str, qpos: np.ndarray, parent_key: str = None
        ) -> np.ndarray:
            if key == self.uid:
                articulation = self.articulation
            else:
                if key in self.child_articulations:
                    articulation = self.child_articulations.get(key, None)
                    if articulation is None:
                        return None  # Articulation not found

            # Case 1: Use parent's drive controller for forward kinematics
            if (
                parent_key
                and (parent_key in self.drive_controllers)
                and hasattr(self.drive_controllers[parent_key], "get_fk")
                and (self.drive_controllers.get(key, None) is None)
            ):
                end_link_name = self.end_link_names.get(key, None)
                if end_link_name is None:
                    end_link_index = -1
                else:
                    end_link_index = self.drive_controllers[parent_key].get_link_orders(
                        end_link_name
                    )

                _, xpos = self.drive_controllers[parent_key].get_fk(
                    qpos, index=end_link_index
                )
            # Case 2: Use articulation's own drive controller
            elif (key in self.drive_controllers) and hasattr(
                self.drive_controllers[key], "get_fk"
            ):
                if len(qpos) != self.drive_controllers[key]:
                    qpos = qpos[self._joint_ids[key]]
                end_link_name = self.end_link_names.get(key)
                if end_link_name is None:
                    end_link_index = -1
                else:
                    end_link_index = self.drive_controllers[key].get_link_orders(
                        end_link_name
                    )

                _, xpos = self.drive_controllers[key].get_fk(qpos, index=end_link_index)
            # Case 3: Fallback to direct world pose
            else:
                xpos = self._convert_pose(
                    articulation.get_world_pose(), is_to_arena=True
                )
                return xpos

            # Get the base xpos for the articulation
            # If parent_key exists, use it; otherwise use the current key
            base_xpos = self.get_base_xpos(parent_key if parent_key else key)

            # Get initial transformation matrix, default to identity if not found
            initial_xpos = self.init_base_xpos.get(key, np.eye(4))

            if is_world_coordinates:
                # Special handling for root links which require different transformation logic
                if self.root_link_names.get(key, None) is not None:
                    if key not in self.drive_controllers:
                        # For articulations without drive controllers,
                        # transform using base transformation matrix
                        return base_xpos @ xpos
                    else:
                        # For articulations with drive controllers,
                        # get an up-to-date base transformation and apply it
                        root_base_xpos = self.get_base_xpos(key)
                        return root_base_xpos @ xpos
                else:
                    # Handle non-root links
                    # TODO: judge by num of drive_controllers
                    return (
                        (initial_xpos @ xpos)
                        if parent_key is not None
                        else (base_xpos @ xpos)
                    )

            return xpos

        # If name is None, calculate for all articulations
        if name is None:
            current_xpos = {}
            qpos = self.get_current_qpos(self.uid)  # Get qpos once for all

            # Calculate for all main articulations
            xpos = calculate_xpos(self.uid, qpos)
            if xpos is not None:
                current_xpos[self.uid] = xpos

            # Calculate for child articulations using parent drive controller
            for child_key in self.child_articulations:
                xpos = calculate_xpos(child_key, qpos, self.uid)
                if xpos is not None:
                    current_xpos[child_key] = xpos

            return current_xpos

        # Check for articulation in child articulations
        if name in self.child_articulations:
            if self.uid in self._actived_joint_names:
                xpos = calculate_xpos(name, self.get_current_qpos()[self.uid], self.uid)
            else:
                xpos = calculate_xpos(name, self.get_current_qpos(self.uid), self.uid)
            if xpos is not None:
                return xpos

        # Check for articulation in main articulation
        xpos = calculate_xpos(name, self.get_current_qpos(name))
        if xpos is not None:
            return xpos

        logger.log_warning(f"Articulation '{name}' not found.")
        return None

    def get_base_xpos(
        self, name: str = None, is_init: bool = False
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        r"""Get current robot base pose in arena coordinate system.

        Args:
            name (str, optional): Name of the articulation. Defaults to None.
            is_init (bool, optional): Init base xpos or current base xpos. Current base xpos defaults.

        Returns:
            np.ndarray: Joint positions for the specified articulation if `name` is provided and found.
            Dict[str, np.ndarray]: Joint positions for all articulations if `name` is None.
            None: If `name` is provided but not found.
        """

        if is_init:
            # Return initial base positions
            return self.init_base_xpos.get(name) if name else self.init_base_xpos

        # Initialize a dictionary for current base positions
        current_base_xpos_dict = {}

        # Get the current base xpos for the main articulation
        base_xpos = self.articulation.get_link_pose(self.root_link_names[self.uid])
        current_base_xpos_dict[self.uid] = self._convert_pose(
            base_xpos, is_to_arena=True
        )

        # Populate the dictionary with joint positions for all child articulations
        for key in self.child_articulations:
            if (
                self.root_link_names.get(key, None)
                in self.child_articulations[key].get_link_names()
            ):
                child_base_xpos = self._get_articulation_property(
                    key, "get_link_pose", link_name=self.root_link_names[key]
                )
                current_base_xpos_dict[key] = self._convert_pose(
                    child_base_xpos, is_to_arena=True
                )

        if name is None:
            return current_base_xpos_dict

        # If a specific articulation name is provided
        if name == self.uid:
            return self._convert_pose(base_xpos, is_to_arena=True)

        # Get the base xpos for the specified articulation
        current_base_xpos = self._get_articulation_property(
            name, "get_link_pose", link_name=self.root_link_names[name]
        )
        return self._convert_pose(current_base_xpos, is_to_arena=True)

    def set_base_xpos(
        self, name: str = None, base_xpos: np.ndarray = np.eye(4)
    ) -> None:
        r"""Set the robot's base pose.

        Args:
            name (str, optional):
                Name of the articulation. If specified, the function will
                apply the base pose to the articulation with this name.
                Defaults to None, which means the base pose will be set for
                the entire robot.

            base_xpos (np.ndarray, optional):
                A [4, 4] matrix representing the transformation matrix that
                defines the base pose of the robot. The matrix should
                contain rotation and translation information. Defaults to
                the identity matrix (np.eye(4)), indicating no change in pose.
        """
        if base_xpos is None:
            logger.log_warning("base_xpos is None, no action taken.")
            return False

        if name is None or name == self.uid:
            if isinstance(base_xpos, dict):
                failed_cases = []
                for articulation_name, pos in base_xpos.items():
                    if not self._set_articulation_property(
                        articulation_name,
                        "set_world_pose",
                        self._convert_pose(pos, is_to_arena=False),
                        False,
                    ):
                        failed_cases.append(articulation_name)
                if failed_cases:
                    logger.log_warning(
                        f"Failed to set base xpos for articulations: {failed_cases}"
                    )
                    return False
                return True
            elif isinstance(base_xpos, (list, np.ndarray)):
                self._set_articulation_property(
                    name,
                    "set_world_pose",
                    self._convert_pose(base_xpos, is_to_arena=False),
                    False,
                )
                return True
            else:
                logger.log_warning(
                    f"Expected base xpos to be dict for articulations, got {type(base_xpos)}."
                )
                return False
        else:
            if isinstance(base_xpos, (list, np.ndarray)):
                return self._set_articulation_property(
                    name,
                    "set_world_pose",
                    self._convert_pose(base_xpos, is_to_arena=False),
                    False,
                )
            else:
                logger.log_warning(
                    f"Expected base xpos to be np.ndarray for articulation '{name}', got {type(base_xpos)}."
                )
                return False

    def get_current_joint_poses(
        self, name: str = None
    ) -> Union[List[np.ndarray], Dict[str, List[np.ndarray]]]:
        r"""Get current robot joint poses.

        Args:
            name (str, optional): Name of the articulation. Defaults to None.

        Returns:
            List[np.ndarray]: List of [4, 4]. Joint poses for the specified articulation if `name` is provided and found.
            Dict[str, List[np.ndarray]]: Joint poses for all articulations if `name` is None.
            None: If `name` is provided but not found.
        """
        name = name or self.uid

        if name == self.uid:
            current_joint_poses = dict()
            if hasattr(self.articulation, "get_joint_poses"):
                current_joint_poses = self._convert_pose(
                    self.articulation.get_joint_poses(self._joint_ids[self.uid]),
                    is_to_arena=True,
                )

            return current_joint_poses
        else:
            if name in self.child_articulations:
                logger.log_warning(f"Articulation {name} not found.")
                return None

            return self._convert_pose(
                self.child_articulations[name].get_joint_poses(self._joint_ids[name]),
                is_to_arena=True,
            )

    def get_init_qpos(self, name: str = None) -> None:
        r"""Get robot initial joint positions.

        Args:
            name (str, optional): Name of the articulation. Defaults to None.

        Returns:
            np.ndarray: initial joint positions for the specified articulation if `name` is provided and found.
            Dict[str, np.ndarray]: Joint positions for all articulations if `name` is None.
            None: If `name` is provided but not found.
        """
        if name is None:
            return self.init_qpos

        if name in self.child_articulations or name == self.uid:
            return self.init_qpos[name]

        logger.log_warning(f"Articulation {name} not found.")
        return None

    def set_init_qpos(
        self, name: str = None, qpos: Union[np.ndarray, Dict[str, np.ndarray]] = []
    ) -> None:
        r"""Set initial joint positions.

        Args:
            name (str, optional): Name of the articulation. Defaults to None.
            qpos (Union[np.ndarray, Dict[str, np.ndarray]]): [dof] of float. Robot initial joint positions.
        """
        if qpos is None:
            logger.log_warning("qpos is None, no action taken.")
            return

        if name is None or name == self.uid:
            if isinstance(qpos, dict):
                for articulation_name, pos in qpos.items():
                    if articulation_name in self.init_qpos:
                        self.init_qpos[articulation_name] = pos
                    else:
                        logger.log_warning(
                            f"Articulation '{articulation_name}' not found in init_qpos."
                        )
            elif isinstance(qpos, (list, np.ndarray)):
                self.init_qpos[self.uid] = qpos
            else:
                logger.log_warning(
                    f"Unsupported qpos type: {type(qpos)}, expected np.ndarray or dict."
                )
        else:
            if not isinstance(qpos, (list, np.ndarray)):
                logger.log_warning(
                    f"Expected qpos to be np.ndarray for articulation '{name}', got {type(qpos)}."
                )
                return

            if name in self.init_qpos:
                self.init_qpos[name] = qpos
            else:
                logger.log_warning(f"Articulation '{name}' not found in init_qpos.")

    def _get_articulation_property(
        self, name: str, property_name: str, **params
    ) -> Union[np.ndarray, None]:
        r"""Helper function to get a property for a specific articulation.

        This function retrieves the value of a specified property (e.g., position,
        velocity) for the articulation identified by 'name'. It first checks if the
        articulation is a main articulation and then checks child articulations. If
        the articulation is found and the property exists, the function returns the
        property's value.

        Args:
            name (str): The name of the articulation to get the property from.
            property_name (str): The name of the property to retrieve.

        Returns:
            Union[np.ndarray, None]: The value of the property if found, None otherwise.
        """
        # Use self._joint_ids[name] if params is empty
        if not params:
            if name in self._joint_ids:
                params = {"joint_ids": self._joint_ids[name]}
            else:
                logger.log_warning(f"Joint_id '{name}' not found.")
                has_similar_name = False
                for key, val in self._joint_ids.items():
                    if name in key:
                        params = {"joint_ids": val}
                        logger.log_warning(f"Joint_id '{key}' is used for {name}.")
                        name = key
                        has_similar_name = True
                        break
                if not has_similar_name:
                    return None

        if name == self.uid:
            return getattr(self.articulation, property_name)(**params)

        if len(self._joint_ids[name]):
            if name in self.child_articulations:
                child_articulation = self.child_articulations[name]
                return getattr(child_articulation, property_name)(**params)
        else:
            return None

        logger.log_warning(f"Articulation '{name}' not found.")
        return None

    def set_current_qpos(
        self, name: str = None, qpos: Union[np.ndarray, Dict[str, np.ndarray]] = None
    ):
        r"""Set current robot joint positions.

        Args:
            name (str, optional): Name of the articulation. Defaults to None.
            qpos (Union[np.ndarray, Dict[str, np.ndarray]]): [dof] of float. Robot current joint positions.

        Returns:
            bool: True if the positions were successfully set, False otherwise.
        """
        if qpos is None:
            logger.log_warning("qpos is None, no action taken.")
            return False

        if name is None or name == self.uid:
            if isinstance(qpos, dict):
                failed_cases = []
                for articulation_name, pos in qpos.items():
                    if not self._set_articulation_property(
                        articulation_name, "set_current_qpos", pos
                    ):
                        failed_cases.append(articulation_name)
                if failed_cases:
                    logger.log_warning(
                        f"Failed to set qpos for articulations: {failed_cases}"
                    )
                    return False
                return True
            elif isinstance(qpos, (list, np.ndarray)):
                return self._set_articulation_property(name, "set_current_qpos", qpos)
            else:
                logger.log_warning(
                    f"Expected qpos to be dict for articulations, got {type(qpos)}."
                )
                return False
        else:
            if isinstance(qpos, (list, np.ndarray)):
                return self._set_articulation_property(name, "set_current_qpos", qpos)
            else:
                logger.log_warning(
                    f"Expected qpos to be np.ndarray for articulation '{name}', got {type(qpos)}."
                )
                return False

    def get_current_qpos(
        self, name: str = None
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        r"""Get current robot joint positions.

        Args:
            name (str, optional): Name of the articulation. Defaults to None.

        Returns:
            np.ndarray: Joint positions for the specified articulation if `name` is provided and found.
            Dict[str, np.ndarray]: Joint positions for all articulations if `name` is None.
            None: If `name` is provided but not found.
        """
        # Validate the name parameter
        if name is not None and not isinstance(name, str):
            logger.log_warning(
                f"The 'name' parameter must be a string or None, got {type(name)}."
            )
            return None

        if name is None:
            # Initialize a dictionary to hold joint positions for all articulations
            current_qpos_dict = {}

            # Get the current joint positions for the main articulation
            qpos = self.articulation.get_current_qpos()
            current_qpos_dict[self.uid] = qpos

            # Populate the dictionary with joint positions for all child articulations
            for key in self.child_articulations:
                current_qpos_dict[key] = self._get_articulation_property(
                    key, "get_current_qpos"
                )

            return current_qpos_dict
        else:

            return self._get_articulation_property(name, "get_current_qpos")

    def set_current_qvel(
        self, name: str = None, qvel: Union[np.ndarray, Dict[str, np.ndarray]] = None
    ):
        r"""Set the current joint velocities of the robot.

        Args:
            name (str, optional):
                Name of the articulation. If None, the velocities will be set
                for all articulations.

            qvel (Union[np.ndarray, Dict[str, np.ndarray]], optional):
                Joint velocities. This can be a NumPy array for a single
                articulation or a dictionary mapping articulation names to
                their respective velocities.

        Returns:
            bool: Returns True if the joint velocities were successfully set,
            otherwise returns False if no action was taken or if there
            were errors in the input.
        """
        if qvel is None:
            logger.log_warning("qvel is None, no action taken.")
            return False

        if name is None or name == self.uid:
            if isinstance(qvel, dict):
                failed_cases = []
                for articulation_name, vel in qvel.items():
                    if not self._set_articulation_property(
                        articulation_name, "set_current_qvel", vel
                    ):
                        failed_cases.append(articulation_name)
                if failed_cases:
                    logger.log_warning(
                        f"Failed to set qvel for articulations: {failed_cases}"
                    )
                    return False
                return True
            else:
                logger.log_warning(
                    f"Expected qvel to be dict for articulations, got {type(qvel)}."
                )
                return False
        else:
            if isinstance(qvel, (list, np.ndarray)):
                return self._set_articulation_property(name, "set_current_qvel", qvel)
            else:
                logger.log_warning(
                    f"Expected qvel to be np.ndarray for articulation '{name}', got {type(qvel)}."
                )
                return False

    def get_current_qvel(
        self, name: str = None
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        r"""Get current robot joint velocities.

        Args:
            name (str, optional): Name of the articulation. Defaults to None.

        Returns:
            np.ndarray: Joint velocities for the specified articulation if `name` is provided and found.
            Dict[str, np.ndarray]: Joint velocities for all articulations if `name` is None.
            None: If `name` is provided but not found.
        """
        if name is None:
            # Initialize a dictionary to hold joint velocities for all articulations
            current_qvel_dict = {}

            # Get the current joint velocities for the main articulation
            qvel = self.articulation.get_current_qvel()
            # Store the velocity of the main articulation in the dictionary using its unique ID
            current_qvel_dict[self.uid] = qvel

            # Iterate over child articulations to get their velocities
            for key in self.child_articulations:
                # Retrieve and store the joint velocity for the child articulation in the dictionary
                current_qvel_dict[key] = self._get_articulation_property(
                    key, "get_current_qvel"
                )

            # Return the dictionary containing velocities for all articulations
            return current_qvel_dict
        else:
            return self._get_articulation_property(name, "get_current_qvel")

    def set_current_qf(
        self,
        name: str = None,
        qf: Union[np.ndarray, Dict[str, np.ndarray]] = None,
    ):
        r"""Set current robot joint force.

        Args:
            name (str, optional): Name of the articulation. Defaults to None.
            qf (Union[np.ndarray, Dict[str, np.ndarray]]): [dof] of float. Robot current joint force.

        """
        if qf is None:
            logger.log_warning("joint_force is None, no action taken.")
            return False

        if name is None:
            if isinstance(qf, dict):
                failed_cases = []
                for articulation_name, force in qf.items():
                    if not self._set_articulation_property(
                        articulation_name, "set_current_qf", force
                    ):
                        failed_cases.append(articulation_name)
                if failed_cases:
                    logger.log_warning(
                        f"Failed to set joint force for articulations: {failed_cases}"
                    )
                    return False
                return True
            else:
                logger.log_warning(
                    f"Expected joint_force to be dict for articulations, got {type(qf)}."
                )
                return False
        else:
            if isinstance(qf, (list, np.ndarray)):
                return self._set_articulation_property(name, "set_current_qf", qf)
            else:
                logger.log_warning(
                    f"Expected joint_force to be np.ndarray for articulation '{name}', got {type(qf)}."
                )
                return False

    def get_current_qf(
        self, name: str = None
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        r"""Get current robot joint force.

        Args:
            name (str, optional): Name of the articulation. Defaults to None.

        Returns:
            np.ndarray: Joint force for the specified articulation if `name` is provided and found.
            Dict[str, np.ndarray]: Joint force for all articulations if `name` is None.
            None: If `name` is provided but not found.
        """
        if name is None:
            # Initialize a dictionary to hold joint forces for all articulations
            current_qf_dict = {}

            # Get the current joint forces for the main articulation
            qvel = self.articulation.get_current_qvel()
            # Store the velocity of the main articulation in the dictionary using its unique ID
            current_qf_dict[self.uid] = qvel

            # Iterate over child articulations to get their forces
            for key in self.child_articulations:
                # Retrieve and store the joint velocity for the child articulation in the dictionary
                current_qf_dict[key] = self._get_articulation_property(
                    key, "get_current_qvel"
                )

            # Return the dictionary containing forces for all articulations
            return current_qf_dict
        else:
            return self._get_articulation_property(name, "get_current_qf")

    @staticmethod
    def is_approx(qpos1: np.ndarray, qpos2: np.ndarray, eps: float = 1e-5):
        r"""Evaluate whether qpos1 and qpos2 are 'close'.

        Args:
            qpos1 (np.ndarray): a object of joint
            qpos2 (np.ndarray): a object of other joint

        Returns:
            bool: is close
        """
        qpos1 = np.array(qpos1)
        qpos2 = np.array(qpos2)
        if qpos1.shape != qpos2.shape:
            logger.log_warning(
                "qpos1 shape {} does not match qpos2 shape {}, qpos1: {}, qpos2: {}.".format(
                    qpos1.shape, qpos2.shape, qpos1, qpos2
                )
            )
            return False

        dis = np.linalg.norm(qpos1 - qpos2, ord=1)
        return dis < eps

    def create_physical_visible_node(
        self, name: str, rgba: np.array = None, link_name: str = None
    ) -> bool:
        r"""Create a physical visible node for the articulation.

        Args:
            name (str):
                The name/identifier of the articulation to create the visible node for.
                Must match either the main articulation's UID or a child articulation's name.

            rgba (np.ndarray, optional):
                An array of 4 float values representing the RGBA color values:
                - Red component (0.0 to 1.0)
                - Green component (0.0 to 1.0)
                - Blue component (0.0 to 1.0)
                - Alpha/transparency (0.0 to 1.0)
                Defaults to [0.0, 1.0, 0.0, 0.6] (semi-transparent green).

            link_name (str, optional):
                The specific link name of the articulation to create the visible node for.
                If None, visible nodes will be created for all links of the articulation.
                Defaults to None.

        Returns:
            bool:
                True if the visible node was successfully created.
                False if:
                - The articulation name was not found
                - The link name was invalid
                - The creation process failed
        """
        if rgba is None:
            rgba = np.array([0.0, 1.0, 0.0, 0.6])
        else:
            rgba = np.array(rgba)

        assert rgba.shape == (4,), "RGBA array must have 4 elements."

        # Prepare parameters for the node creation
        params = {"rgba": rgba}

        # Add link_name to parameters if provided
        if link_name is not None:
            params["link_name"] = link_name

        # Check if the name matches the uid and create the node
        if name == self.uid:
            return self.articulation.create_physical_visible_node(**params)
        elif name in self.child_articulations:
            # Otherwise, create the node for the specified child articulation
            return self.child_articulations[name].create_physical_visible_node(**params)

        logger.log_warning(f"Articulation '{name}' not found.")
        return False

    def set_physical_visible(
        self,
        name: str,
        is_physic_visible: bool,
        is_render_body_visible: bool = True,
        link_name: str = None,
    ) -> bool:
        r"""Set whether the current physical collision is visible.

        Args:
            name (str): The name of the articulation.
            is_physic_visible (bool): Whether the current physical node is visible.
            is_render_body_visible (bool, optional): Whether the render body is visible. Defaults to True.
            link_name (str, optional): The link name of the articulation. If None, set all articulation visible. Defaults to None.

        Returns:
            bool: Returns True if the setting is successful, False otherwise.
        """
        # Prepare parameters for setting visibility
        params = {
            "is_physic_visible": is_physic_visible,
            "is_render_body_visible": is_render_body_visible,
        }

        # Add link_name to parameters if provided
        if link_name is not None:
            params["link_name"] = link_name

        # Check if the name matches the uid and set visibility
        if name == self.uid:
            self.articulation.set_physical_visible(**params)
            return True

        # Check if the name is in child articulations and set visibility for it
        elif name in self.child_articulations:
            self.child_articulations[name].set_physical_visible(**params)
            return True

        # Log a warning if the articulation name is not found
        logger.log_warning(f"Articulation '{name}' not found.")
        return False
