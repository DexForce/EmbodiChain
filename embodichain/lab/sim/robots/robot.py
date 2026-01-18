# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------

import numpy as np
from typing import List, Tuple, Union, Dict, Any, Optional
from abc import ABC, abstractmethod
from copy import deepcopy
import open3d as o3d
import os
from pathlib import Path
import pytorch_kinematics as pk
from matplotlib import colormaps

import dexsim
from dexsim.models import Entity, MeshObject

# from dexsim.engine import Articulation
from dexsim.types import DriveType, PhysicalAttr, ArticulationFlag, PrimitiveType


from embodichain.utils import logger
from dexsim.utility.env_utils import create_point_cloud_from_o3d_pcd

# Try to import DriveController, but make it optional
try:
    from rlia.kit.drive_controllers import DriveController
except ImportError:
    # If rlia is not available, create a dummy type for type checking
    DriveController = None

from embodichain.lab.sim.end_effector import EndEffector

# from dexsim.utility import inv_transform
from dexsim.sensor import Sensor, MonocularCam, BinocularCam
from embodichain.lab.sim.articulation_entity import ArticulationEntity

__all__ = ["Robot"]


class Robot(ArticulationEntity, ABC):
    r"""
    Abstract class for robot in simulation.
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
        r"""Initialize the robot.

        Args:
            urdf_path (str): urdf file path of robot
            init_qpos (np.ndarray, optional): [dof] of double. Init robot joint state(home joint state).
            init_base_xpos (np.ndarray, optional): [4, 4] of double. Robot base pose in arena coordinate system.
            speed_ratio (float, optional): 0 ~ 1. Robot speed ratio.
            time_step (float, optional): wait time between two update. Defaults to 1/50.
            drive_type (DriveType, optional): DriveType.FORCE or DriveType.FORCE.
            env (Arena, optional): dexsim.environment.Arena. Load the first world(None defaults).
            kwargs(optional): Accepts additional keyword arguments.
        """
        # unique name of the robot.
        self.uid = kwargs.get("uid", "Robot")

        super().__init__(
            urdf_path=urdf_path,
            init_qpos=init_qpos,
            init_base_xpos=init_base_xpos,
            speed_ratio=speed_ratio,
            time_step=time_step,
            drive_type=drive_type,
            env=env,
            **kwargs,
        )

        # Initialize the robot
        self._init_robot(**kwargs)

        # Disable self-collision avoidance for the articulation
        self.set_enable_self_collision_flag(self.uid, False)

        # Additional parameters
        self.attach_end_effectors = {}

        # Build pk_serial_chain
        self.pk_serial_chain = self.build_pk_serial_chain()

    def set_enable_self_collision_flag(self, name: str = None, is_enable: bool = False):
        r"""Set the self-collision flag for the specified articulation
            or all articulations.

        Args:
            name (str, optional): Name of the articulation.
                If None, apply to all articulations. Defaults to None.
            is_enable (bool, optional): Flag to enable
                or disable self-collision. Defaults to False.
        """
        if name is None or name == self.uid:
            self.articulation.set_articulation_flag(
                ArticulationFlag.DISABLE_SELF_COLLISION, not is_enable
            )
        else:
            if name in self.child_articulations:
                self.child_articulations[name].set_articulation_flag(
                    ArticulationFlag.DISABLE_SELF_COLLISION, not is_enable
                )
            else:
                logger.log_warning(f"Articulation '{name}' not found.")

    @abstractmethod
    def _init_robot(self, **kwargs) -> None:
        r"""Initializes the robot using the URDF path with necessary parameters."""
        pass

    def get_end_effector(self, uid: str = None):
        r"""Get the end effector by its unique identifier.

        Args:
            uid (str): Unique identifier for the end effector to be attached.
                       If None, returns a dictionary of all end effectors.

        Returns:
            EndEffector: The end effector associated with the given uid, or None if not found.
        """
        if uid is None:
            return self.attach_end_effectors

        end_effector = self.attach_end_effectors.get(uid)
        return end_effector

    def attach_end_effector(
        self,
        uid: str,
        end_effector: EndEffector,
        robot_uid: str = None,
        attach_xpos: np.ndarray = np.eye(4),
        ee_link_name: str = "ee_link",
        **kwargs,
    ):
        r"""Attach an end effector to the robotic system.

        Args:
            uid (str): Unique identifier for the end effector to be attached.
            end_effector (EndEffector): An instance of the EndEffector class representing the end effector to be attached.
            robot_uid (str, optional): Unique identifier for the robot to which the end effector is to be attached. Defaults to None.
            attach_xpos (np.ndarray, optional): 4x4 transformation matrix (homogeneous transformation matrix) representing the pose
                                                at which the end effector should be attached. Defaults to identity matrix.
            ee_link_name (str, optional): The link string that represents the end effector link in the robot. Defaults to "ee_link".
            **kwargs: Additional keyword arguments for extended functionality (if applicable).
        Returns:
            tuple: A tuple containing a boolean and a value:
                - (bool) False if the end effector is already attached, True otherwise.
                - (None) Always returns None as the second element.
        """
        # If robot_uid is not provided, use the current object's uid
        robot_uid = robot_uid or self.uid

        # Check if the end effector is already attached to the robot
        if robot_uid == self.uid or robot_uid in self.child_articulations:
            target_articulation = (
                self.articulation
                if robot_uid == self.uid
                else self.child_articulations[robot_uid]
            )

            # Get degrees of freedom for the target articulation and the end effector
            arm_dof = target_articulation.get_dof()
            ef_dof = end_effector.get_dof()

            # Get the root link name of the end effector
            ef_root_link_name = end_effector.articulation.get_root_link_name()
            ef_link_names = end_effector.articulation.get_link_names()
            end_effector.drive_type = self.drive_type

            end_effector_joint_names = (
                end_effector.articulation.get_actived_joint_names()
            )

            # Load the end effector's URDF into the target articulation at the specified position
            target_articulation.load_urdf(
                end_effector.get_urdf_path(), ee_link_name, attach_xpos
            )

            # Remove the previous articulation of the end effector
            ef_articulation = end_effector.get_articulation(end_effector.uid)
            self._env.remove_articulation(ef_articulation)

            # Assign the target articulation to the end effector
            end_effector.articulation = target_articulation

            target_articulation_joint_names = (
                target_articulation.get_actived_joint_names()
            )

            # Update joint indices for the end effector
            ef_joint_ids = arm_dof + np.arange(ef_dof)
            end_effector.set_joint_ids(ef_joint_ids)

            # Combine initial positions
            ef_init_qpos = end_effector._init_qpos
            joint_name_to_idx = {
                name: idx for idx, name in enumerate(target_articulation_joint_names)
            }
            ef_ids = np.array(
                [joint_name_to_idx[name] for name in end_effector_joint_names]
            )

            robot_ids = np.arange(arm_dof)
            target_articulation.set_current_qpos(
                self.get_init_qpos(robot_uid), joint_ids=robot_ids
            )
            target_articulation.set_current_qpos(ef_init_qpos, joint_ids=ef_ids)

            # Set physical attributes for the target articulation
            target_articulation.set_physical_attr(self.default_physical_attrs)
            target_articulation.set_drive(
                drive_type=self.drive_type, **self.default_drive_param
            )

            # Store end effector details in the class attributes
            self.child_articulations[uid] = end_effector.articulation
            self._dof[uid] = ef_dof
            self._joint_ids[uid] = ef_ids
            self.init_qpos[uid] = ef_init_qpos
            self.root_link_names[uid] = ef_root_link_name
            end_effector.attach_robot_uid = robot_uid

            end_effector._joint_ids[end_effector.uid] = ef_ids

            # TODO: update robot, etc.
            # Update the joint ids for other end effector
            for ee_name, ee in self.attach_end_effectors.items():
                ee_idx_list = np.array(
                    [joint_name_to_idx[name] for name in ee.actived_joint_names]
                )

                self._joint_ids[ee_name] = ee_idx_list
                ee._joint_ids[ee.uid] = ee_idx_list

                # ee_init_qpos = self.init_qpos[ee_name]
                # Update the initial positions in the class attributes
                # self.init_qpos[ee.uid] = ee_init_qpos[ee_idx_list]

            # Keep a reference of the attached end effector
            self.attach_end_effectors[uid] = end_effector

            # set end-effector physical param and drive param
            for link_name in ef_link_names:
                target_articulation.set_physical_attr(
                    attrib=end_effector.default_physical_attrs,
                    link_name=link_name,
                    is_replace_inertial=True,
                )
            target_articulation.set_drive(
                drive_type=self.drive_type,
                joint_ids=ef_joint_ids,
                **end_effector.default_drive_param,
            )
            # end_effector.set_drive(end_effector.drive_type)
            return True, end_effector
        else:
            logger.log_warning(f"Articulation '{uid}' not found.")
            return False, None

    def attach_sensor(
        self,
        sensor: Sensor,
        robot_uid: str = None,
        attach_xpos: np.ndarray = np.eye(4),
        link_name: str = "ee_link",
    ):
        r"""Attach a sensor to a robot.

        Note:
            Currently, this function is only available for Monocular and Binocular sensors.

        Args:
            sensor (Sensor): The sensor object to be attached. It can be a MonocularCam or BinocularCam.
            robot_uid (str, optional): Unique identifier for the robot to which the sensor will be attached. Defaults to None, which refers to the current robot.
            attach_xpos (np.ndarray, optional): 4x4 transformation matrix (homogeneous transformation matrix) representing the pose
                                                at which the sensor should be attached. Defaults to the identity matrix.
            link_name (str, optional): The link string that represents the attachment point on the robot. Defaults to "ee_link".

        Returns:
            None: This function does not return a value but logs warnings for unsupported sensor types or invalid robot identifiers.
        """
        robot_uid = robot_uid or self.uid

        # Check if the robot_uid matches the current robot or a child articulation
        if robot_uid == self.uid or robot_uid in self.child_articulations:
            target_articulation = (
                self.articulation
                if robot_uid == self.uid
                else self.child_articulations[robot_uid]
            )

            # Attach the sensor based on its type
            if isinstance(sensor, MonocularCam):
                target_articulation.attach_node(
                    obj=sensor.get_node(),
                    link_name=link_name,
                    relative_pose=attach_xpos,
                )
            elif isinstance(sensor, BinocularCam):
                # Attach the left camera node
                if sensor._coordinate_system == "center":
                    relative_pose = sensor._relativate_T_l
                else:
                    relative_pose = sensor.get_relative_transform()
                    relative_pose[:3, 3] = relative_pose[:3, 3] * -0.5
                target_articulation.attach_node(
                    obj=sensor.get_node(is_left=True),
                    link_name=link_name,
                    relative_pose=attach_xpos @ relative_pose,
                )
                # Attach the right camera node
                target_articulation.attach_node(
                    obj=sensor.get_node(is_left=False),
                    link_name=link_name,
                    relative_pose=attach_xpos @ np.linalg.inv(relative_pose),
                )
            else:
                logger.log_warning("Unsupported sensor type: %s", type(sensor).__name__)
        else:
            logger.log_warning(f"Articulation '{robot_uid}' not found.")

    # @deprecated(reason="Currently unable to detach this component.")
    def detach_end_effector(
        self,
        uid: str,
        robot_uid: str = None,
    ):
        r"""Detach an end effector from the robotic system.

        Args:
            uid (str): Unique identifier for the end effector to be detached.
            robot_uid (str, optional): Unique identifier for the robot from which the end effector is to be detached.

        Returns:
            bool: True if the end effector was successfully detached, False otherwise.
        """
        if uid not in self.child_articulations:
            logger.log_warning(f"End effector {uid} already detached.")
            return False

        robot_uid = robot_uid or self.uid
        if robot_uid is not self.uid:
            logger.log_warning(f"Articulation with UID '{robot_uid}' not found.")
            return False

        if uid in self.init_qpos:
            del self.init_qpos[uid]
        if uid in self.init_base_xpos:
            del self.init_base_xpos[uid]
        self.child_articulations[uid].detach_parent()
        self.child_articulations.pop(uid)
        return True

    def close(self, uid: str = None, target: float = 1.0) -> bool:
        r"""Closes the attached end effector, if this manipulator has one. If no UID is provided,
        it will close all end effectors associated with the manipulator.

        Args:
            uid (str, optional):
                A unique identifier for the specific end effector to be closed.
                If None, the method will attempt to close all end effectors.
                Defaults to None.
            target (float, optional):
                The target position for the close operation, typically representing
                the closure position of the end effector.
                Defaults to 1.0 (fully closed).

        Returns:
            bool:
                Returns True if the end effector(s) were closed successfully,
                and False otherwise. If no end effector is found with the given UID,
                a warning is logged.
        """
        is_success = False
        if uid is None or uid == self.uid:
            for key, value in self.attach_end_effectors.items():
                if isinstance(value, EndEffector):
                    value.close(target=target)
                    is_success = True  # Mark success if any end effector is closed
        else:
            if uid in self.attach_end_effectors:
                self.attach_end_effectors[uid].close(target=target)
                is_success = True
            else:
                logger.log_warning(f"End effector with UID '{uid}' not found.")

        return is_success

    def open(self, uid: str = None, target: float = 0.0) -> bool:
        r"""
        Opens the attached end effector, if this manipulator has one. If no UID is provided,
        it will open all end effectors associated with the manipulator.

        Args:
            uid (str, optional):
                A unique identifier for the specific end effector to be opened.
                If None, the method will attempt to open all end effectors.
                Defaults to None.
            target (float, optional):
                The target position for the open operation, typically representing
                the opening position of the end effector.
                Defaults to 0.0 (fully opened).

        Returns:
            bool:
                Returns True if the end effector(s) were opened successfully,
                and False otherwise. If no end effector is found with the given UID,
                a warning is logged.
        """
        is_success = False
        if uid is None or uid == self.uid:
            for key, value in self.attach_end_effectors.items():
                if isinstance(value, EndEffector):
                    value.open(target=target)
                    is_success = True  # Mark success if any end effector is opened
        else:
            if uid in self.attach_end_effectors:
                self.attach_end_effectors[uid].open(target=target)
                is_success = True
            else:
                logger.log_warning(f"End effector with UID '{uid}' not found.")

        return is_success

    def set_controller(self, controller=None, uid: str = None, **kwargs):
        r"""Set a drive or task controller to the robot.

        Args:
            controller (DriveController, optional):
                The controller instance to be added to the robot. Can be either:
                - DriveController: For low-level joint control
            uid (str, optional):
                Unique identifier for the articulation to be controlled.
                If None, uses the robot's main articulation ID.

        Returns:
            bool: True if controller was successfully set, False otherwise.
        """
        uid = uid or self.uid

        # Check if the robot_uid matches the current robot or a child articulation
        if uid == self.uid or uid in self.child_articulations:
            target_articulation = (
                self.articulation if uid == self.uid else self.child_articulations[uid]
            )

            if DriveController is not None and isinstance(controller, DriveController) and any(
                isinstance(controller, ctl_type)
                for ctl_type in self.supported_drive_controller_types.values()
            ):
                if hasattr(controller, "set_init_qpos"):
                    controller.set_init_qpos(self.init_qpos[uid])
                controller.set_articulation(target_articulation)
                controller.set_control_q_ids(self._joint_ids[uid])
                self.drive_controllers[uid] = controller
            else:
                logger.log_warning(f"Controller type '{type(controller)}' not support.")
                return False
        else:
            logger.log_warning(f"Articulation '{uid}' not found.")
            return False

        return True

    def set_speed_ratio(self, speed_ratio: float, uid: str = None):
        r"""Set speed ratio of the robot.

        Args:
            speed_ratio (float): 0.0~1.0. robot speed ratio.
            uid (str): Uid of the articulation.
        """
        uid = uid or self.uid

        if uid == self.uid or uid in self.child_articulations:
            self.speed_ratio = speed_ratio
            return True
        else:
            logger.log_warning(
                f"Drive controller with UID '{uid}' not found. Please add the drive controller before set speed ratio."
            )
            return False

    def get_speed_ratio(self, uid: str = None):
        r"""Get speed ratio of the robot.

        Args:
            uid (str): Uid of the articulation.
        """
        uid = uid or self.uid

        if uid == self.uid or uid in self.child_articulations:
            return self.speed_ratio
        else:
            logger.log_warning(
                f"Drive controller with UID '{uid}' not found. Please add the drive controller before set speed ratio."
            )
            return None

    @abstractmethod
    def get_fk(self, qpos: np.ndarray, uid: str = None) -> np.ndarray:
        r"""Get forward kinematic of given joints

        Args:
            qpos (np.ndarray): [dof] of float.
            uid (str, optional): uid of the articulation. Defaults to None.

        Returns:
            np.ndarray: Pose of the end-effector.
        """
        pass

    @abstractmethod
    def get_ik(self, xpos: np.ndarray, uid: str = None, **kwargs) -> np.ndarray:
        r"""Get inverse kinematic of given end-effector pose.

        Args:
            xpos (np.ndarray): [4, 4] of matrix.
            uid (str, optional): uid of the articulation. Defaults to None.
            **kwargs: Other parameters. which can be used to specify the IK method.

        Returns:
            np.ndarray: [dof] of float.
        """
        pass

    @abstractmethod
    def move(
        self,
        path: Union[np.ndarray, List[np.ndarray]],
        is_joint: bool = False,
        is_wait: bool = True,
        **kwargs,
    ) -> bool:
        r"""Move the robot to the given path.

        Args:
            path (np.ndarray): [4, 4] | [waypoint_num, 4, 4] | [dof] of float or
                [waypoint_num, dof] of float. Path in cartesian space or joint space.
            is_joint (bool, optional): Whether the path is in joint space. Defaults to False.
            is_wait (bool, optional): Whether to synchronize the robot movement. Defaults to True.
            **kwargs: Other parameters.

        Returns:
            bool: is_move_success
        """
        pass

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

    def get_proprioception(self, remove_index: bool = True) -> Dict[str, Any]:
        r"""Gets robot proprioception information, primarily for agent state representation in robot learning scenarios.

        The default proprioception information includes:
            - xpos: End-effector pose in the robot base coordinate system.
            - qpos: Joint positions.
            - qvel: Joint velocities.
            - qf (effort): Joint forces.

        Args:
            remove_index (bool, optional):
                If True, the suffix index of the UID will be removed.
                Defaults to True.

        Returns:
            Dict[str, Any]:
                A dictionary containing the robot's proprioception information,
                where keys are the UID or modified UID and values are dictionaries
                containing the proprioception data.
        """
        obs = {}

        # Helper function to populate proprioception data for a given name
        def populate_proprioception(name: str):
            return {
                "xpos": self.get_current_xpos(name=name, is_world_coordinates=False),
                "qpos": self.get_current_qpos(name=name),
                "qvel": self.get_current_qvel(name=name),
                "qf": self.get_current_qf(name=name),
            }

        # Process the main UID
        base_name = self.uid.split("_")[0] if remove_index else self.uid
        obs[base_name] = populate_proprioception(self.uid)

        # Process child articulations
        for child_name in self.child_articulations:
            if remove_index:
                import re

                modified_name = re.sub(r"(_\d+)$", "", child_name)
            else:
                modified_name = child_name

            if modified_name in obs:
                if isinstance(obs[modified_name], list):
                    obs[modified_name].append(populate_proprioception(child_name))
                else:
                    obs[modified_name] = [
                        obs[modified_name],
                        populate_proprioception(child_name),
                    ]
            else:
                obs[modified_name] = populate_proprioception(child_name)

        return obs

    def attach_actor(
        self, actor: Entity, relative_xpos: np.ndarray, uid: str = None, **kwargs
    ) -> Entity:
        r"""Attach an actor to the robot.

        Args:
            actor (Entity):
                The actor to be attached to the robot.
            relative_xpos (np.ndarray):
                A [4, 4] matrix representing the relative pose of the actor to the robot.
            uid (str, optional):
                Unique identifier of the articulation. If None, defaults to the robot's UID.
            **kwargs:
                Additional parameters for future extension.

        Returns:
            Entity:
                The attached actor, or None if the attachment failed.
        """
        uid = uid or self.uid

        # Define a function to attach the actor to the specified articulation
        def attach_to_articulation(articulation):
            actor_name = actor.get_name()
            self.attached_actors[actor_name] = actor
            articulation.attach_node(actor.node, "ee_link", relative_xpos)
            return actor

        # Check if UID matches the robot's UID
        if uid == self.uid:
            return attach_to_articulation(self.articulation)

        # Check if UID matches any child articulation
        elif uid in self.child_articulations:
            return attach_to_articulation(self.child_articulations[uid])

        # Log a warning if the articulation is not found
        logger.log_warning(f"Articulation with UID '{uid}' not found.")
        return None

    def remove_actor(self, actor_name: str, delete: bool = False) -> None:
        r"""Remove the attached actor from the robot.

        Args:
            actor_name (str): Name of the actor to be removed.
            delete (bool, optional): Whether to delete the actor from the simulation. Defaults to False.
        """
        if actor_name in self.attached_actors:
            for key, value in self.child_articulations.items():
                if isinstance(value, EndEffector):
                    value.detach(actor_name)
            self.attached_actors.pop(actor_name)
            if delete:
                self._env.remove_actor(actor_name)

    def get_attached_actor_names(self) -> List[str]:
        r"""Get names of all attached actors.

        Returns:
            List[str]: Names of all attached actors.
        """
        return list(self.attached_actors.keys())

    def compute_qpos_reachability(
        self,
        name: str,
        resolution: float = np.radians(50),
        qpos_limits: np.ndarray = None,
        cache_mode: str = "memory",
        visualize: bool = False,
        batch_size: int = 100000,
        use_cached: bool = True,
        **kwargs,
    ) -> Tuple[Optional[list[np.ndarray]], Optional[dexsim.models.PointCloud]]:
        """Compute the robot's reachable workspace by joint space sampling.

            Samples points in joint space and optionally visualizes the resulting end-effector positions
            as a colored point cloud. If `visualize` is True, points closer to the robot base are colored green,
            transitioning to red for points further away. If `visualize` is False, only the sampling is performed
            without any visualization.


        Args:
            name (str): Identifier of the robot drive controller to analyze
            resolution (float, optional): Angular resolution for joint space sampling in radians.
                                        Lower values provide finer sampling but increase computation time.
                                        Defaults to 50 degrees (≈0.873 radians)
            qpos_limits (np.ndarray, optional): Custom joint limits array of shape (n_joints, 2).
                                        If None, uses limits from drive controller or articulation.
                                        Defaults to None
            cache_mode (str, optional): Cache mode for workspace analysis. Options include "memory" and "disk".
                                        Defaults to "memory".
            visualize (bool, optional): If set to True, returns an extra Dexsim PointCloud handle for visualization.
                                        Defaults to False.
            batch_size (int, optional): Number of samples to process in each batch.
                                        Defaults to 100000.
            use_cached (bool, optional): If True and `cache_mode` is "disk", attempts to load precomputed results.
                                        Ignored for "memory" mode. Defaults to True.


        Returns:
            Tuple[Optional[list[np.ndarray]], Optional[dexsim.models.PointCloud]]:
                The first element is a list of sampled end-effector poses (4×4 transformation matrices) if sampling succeeds, otherwise None.
                The second element is a point cloud handle if visualization is enabled and successful, otherwise None.
        """
        from embodichain.lab.sim.utility.workspace_analyzer import (
            WorkspaceAnalyzer,
        )
        from embodichain.lab.sim import REACHABLE_XPOS_DIR

        if name not in self.drive_controllers:
            logger.log_warning(f"Drive controller '{name}' not found")
            return None, None

        # try:
        # Get robot configuration
        base_xpos = self.get_base_xpos(name=name)
        drive_controller = self.drive_controllers[name]

        if qpos_limits is None:
            if hasattr(drive_controller, "get_joint_limits"):
                res, upper_limits, lower_limits = self.drive_controllers[
                    name
                ].get_joint_limits()
                if not res:
                    logger.log_warning("Failed to get joint limits")
                    return None, None
                joint_ranges = np.column_stack((lower_limits, upper_limits))
            else:
                joint_limits = self.articulation.get_joint_limits()
                joint_ranges = joint_limits[self._joint_ids[name]]
        else:
            joint_ranges = qpos_limits
        paths = self.get_urdf_path()
        urdf_path = paths if isinstance(paths, str) else paths[self.uid]
        robot_name = os.path.splitext(os.path.basename(urdf_path))[0]
        # Initialize workspace analyzer
        analyzer = WorkspaceAnalyzer(
            robot=self, name=name, resolution=resolution, joint_ranges=joint_ranges
        )
        # Format resolution to avoid issues with decimal points in paths
        resolution_str = f"{resolution:.2f}".replace(".", "_")
        # Join into one directory name
        save_dir = REACHABLE_XPOS_DIR / f"{robot_name}_{name}_{resolution_str}"
        # Sample workspace points
        sampled_xpos = analyzer.sample_qpos_workspace(
            cache_mode=cache_mode,
            save_dir=save_dir,
            batch_size=batch_size,
            use_cached=use_cached,
        )
        if visualize == True:
            # Create and configure point cloud visualization
            # all_positions = [xpos[:3, 3] for xpos in sampled_xpos]
            N = len(sampled_xpos)
            all_pos = np.empty((N, 3), dtype=np.float16)
            for i, mat in enumerate(sampled_xpos):
                all_pos[i] = mat[:3, 3].astype(np.float16)
            pcd = analyzer._process_point_cloud(positions=all_pos)
            # Transfer to World Coordinate
            pcd.transform(base_xpos)
            pcd_handle = create_point_cloud_from_o3d_pcd(pcd=pcd, env=self._env)
        else:
            return sampled_xpos, None

        return sampled_xpos, pcd_handle

        # except Exception as e:
        #     logger.log_warning(f"Failed to visualize qpos workspace: {str(e)}")
        #     return None, None

    def compute_xpos_reachability(
        self,
        name: str,
        ref_xpos: np.ndarray,
        xpos_resolution: float = 0.2,
        qpos_resolution: float = np.radians(60),
        pos_eps: float = 5e-4,
        rot_eps: float = 5e-4,
        max_iterations: int = 1500,
        num_samples: int = 5,
        batch_size: int = 100000,
        save_threshold: int = 10000000,
        qpos_limits: np.ndarray = None,
        cache_mode: str = "memory",
        visualize: bool = True,
        use_cached: bool = True,
        **kwargs,
    ) -> Tuple[
        Optional[list[np.ndarray]],  # First return: list of sampled 4x4 poses
        Optional[
            dexsim.models.PointCloud
        ],  # Second return: point cloud handle if visualization is enabled
    ]:
        """Compute the robot's reachable workspace by Cartesian space sampling.

            Samples points in Cartesian space and checks reachability using inverse kinematics.
            If `visualize` is True, visualizes reachable positions as a colored point cloud;
            Otherwise, only performs the sampling result as open3d PointCloud.


        Args:
            name (str): Identifier of the robot drive controller to analyze
            ref_xpos (np.ndarray): Reference end-effector pose matrix (4x4) defining the
                                orientation for IK solutions
            xpos_resolution (float, optional): Cartesian space sampling resolution in meters.
                                            Smaller values provide finer sampling but increase
                                            computation time. Defaults to 0.2 meters.
            qpos_resolution (float, optional): Angular resolution for initial joint space
                                            sampling in radians. Used to determine workspace
                                            bounds. Defaults to 60 degrees.
            pos_eps (float, optional): Position tolerance for IK solutions in meters.
                                            Defaults to 2e-4 meters.
            rot_eps (float, optional): Rotation tolerance for IK solutions in radians.
                                            Defaults to 2e-4 radians.
            max_iterations (int, optional): Maximum number of IK iterations per sample.
                                            Defaults to 2000.
            num_samples (int, optional): Number of samples to generate in Cartesian space.
                                            Defaults to 10.
            qpos_limits (np.ndarray, optional): Custom joint limits array of shape (n_joints, 2).
                                            If None, uses limits from drive controller or
                                            articulation. Defaults to None
            cache_mode (str, optional): Cache mode for workspace analysis. Options include "memory" and "disk".
                                        Defaults to "memory".
            visualize (bool, optional): If set to True, returns an extra Dexsim PointCloud handle for visualization.
                                        Defaults to True.
            use_cached (bool, optional): If True and `cache_mode` is "disk", attempts to load precomputed results.
                                        Ignored for "memory" mode. Defaults to True.

        Returns:
            Tuple[Optional[list[np.ndarray]], Optional[dexsim.models.PointCloud]]:
                The first element is a list of sampled end-effector poses (4×4 transformation matrices) if sampling succeeds, otherwise None.
                The second element is a point cloud handle if visualization is enabled and successful, otherwise None.
        """
        from embodichain.lab.sim.utility.workspace_analyzer import (
            WorkspaceAnalyzer,
        )
        from embodichain.lab.sim import REACHABLE_XPOS_DIR

        if name not in self.drive_controllers:
            logger.log_warning(f"Drive controller '{name}' not found")
            return None, None

        # try:
        # Get robot configuration
        base_xpos = self.get_base_xpos(name=name)
        ref_xpos_robot = dexsim.utility.inv_transform(base_xpos) @ ref_xpos
        drive_controller = self.drive_controllers[name]

        if qpos_limits is None:
            if hasattr(drive_controller, "get_joint_limits"):
                res, upper_limits, lower_limits = self.drive_controllers[
                    name
                ].get_joint_limits()
                if not res:
                    logger.log_warning("Failed to get joint limits")
                    return None, None
                joint_ranges = np.column_stack((lower_limits, upper_limits))
            else:
                joint_limits = self.articulation.get_joint_limits()
                joint_ranges = joint_limits[self._joint_ids[name]]
        else:
            joint_ranges = qpos_limits

        paths = self.get_urdf_path()
        urdf_path = paths if isinstance(paths, str) else paths[self.uid]
        robot_name = os.path.splitext(os.path.basename(urdf_path))[0]

        qpos_resolution_str = f"{qpos_resolution:.2f}".replace(".", "_")
        xpos_resolution_str = f"{xpos_resolution:.2f}".replace(".", "_")
        # Join into one directory name
        save_dir = (
            REACHABLE_XPOS_DIR
            / f"{robot_name}_{name}_{qpos_resolution_str}_{xpos_resolution_str}"
        )

        # Initialize workspace analyzer
        analyzer = WorkspaceAnalyzer(
            robot=self,
            name=name,
            resolution=qpos_resolution,
            joint_ranges=joint_ranges,
        )
        # Sample workspace points
        sampled_xpos = analyzer.sample_xpos_workspace(
            ref_xpos=ref_xpos_robot,
            xpos_resolution=xpos_resolution,
            qpos_resolution=qpos_resolution,
            cache_mode=cache_mode,
            batch_size=batch_size,
            save_dir=save_dir,
            save_threshold=save_threshold,
            pos_eps=pos_eps,
            rot_eps=rot_eps,
            max_iterations=max_iterations,
            num_samples=num_samples,
            use_cached=use_cached,
        )

        if visualize == visualize:
            if sampled_xpos is None:
                logger.log_warning("No reachable positions found.")
                return None, None
            all_positions = [xpos[:3, 3] for xpos in sampled_xpos]
            pcd = analyzer._process_point_cloud(
                positions=all_positions, is_voxel_down=False
            )
            # Transfer to World Coordinate
            pcd.transform(base_xpos)
            # Create and configure point cloud visualization
            pcd_handle = create_point_cloud_from_o3d_pcd(pcd=pcd, env=self._env)
        else:
            return sampled_xpos, None

        return sampled_xpos, pcd_handle

    def compute_voxel_reachability(
        self,
        name: str,
        voxel_size: float = 0.04,
        num_directions: int = 50,
        num_yaws=6,
        pos_eps: float = 5e-4,
        rot_eps: float = 5e-4,
        max_iterations: int = 1500,
        num_samples: int = 5,
        qpos_limits: np.ndarray = None,
        cache_mode: str = "memory",
        visualize: bool = False,
        use_cached: bool = True,
        **kwargs,
    ) -> Tuple[Optional[List[np.ndarray]], Optional[List[MeshObject]]]:
        """
        Compute the robot's reachable workspace by voxel-based sampling.

        Samples voxel centers within a sphere around the robot’s end-effector base
        and checks reachability via inverse kinematics.
        If `visualize` is True, spawns a colored sphere actor at each voxel center
        to indicate success rate; otherwise returns only the sampled poses.

        Args:
            name (str): Identifier of the drive controller to analyze.
            voxel_size (float, optional): Edge length of each cubic voxel (m).
                Smaller values give finer resolution but increase computation time.
                Defaults to 0.04.
            num_directions (int, optional): Number of sample directions per voxel.
                Defaults to 50.
            num_yaws (int, optional): Number of discrete yaw rotations **around the local Z-axis**
                to try for each sample direction when solving IK. A higher value can
                increase rotational coverage but incurs more IK calls. Defaults to 6.
            qpos_limits (np.ndarray, optional): Custom joint limits array of shape
                (n_joints, 2). If None, retrieves limits from the controller or
                articulation. Defaults to None.
            cache_mode (str, optional): “memory” or “disk” mode for caching IK
                results. Defaults to "memory".
            visualize (bool, optional): If True, returns a list of DexSim actor
                handles for visualization; otherwise returns None for actors.
                Defaults to False.
            use_cached (bool, optional): If True and `cache_mode` is "disk", attempts to load precomputed results.
                                        Ignored for "memory" mode. Defaults to True.

        Returns:
            Tuple[Optional[List[np.ndarray]], Optional[List[MeshObject]]]:
                - List of sampled end-effector poses (4×4 matrices), or None on failure.
                - List of sphere actor handles if visualize=True, else None.
        """
        from embodichain.lab.sim.utility.workspace_analyzer import (
            WorkspaceAnalyzer,
        )
        from embodichain.lab.sim import REACHABLE_XPOS_DIR

        # 1) Validate drive controller
        if name not in self.drive_controllers:
            logger.log_warning(f"Drive controller '{name}' not found")
            return None, None

        try:
            drive_controller = self.drive_controllers[name]

            # 2) Determine joint limits
            if qpos_limits is None:
                if hasattr(drive_controller, "get_joint_limits"):
                    res, upper, lower = drive_controller.get_joint_limits()
                    if not res:
                        logger.log_warning("Failed to get joint limits")
                        return None, None
                    joint_ranges = np.column_stack((lower, upper))
                else:
                    all_limits = self.articulation.get_joint_limits()
                    joint_ranges = all_limits[self._joint_ids[name]]
            else:
                joint_ranges = qpos_limits

            # 3) Prepare save directory
            urdf_paths = self.get_urdf_path()
            urdf_path = (
                urdf_paths if isinstance(urdf_paths, str) else urdf_paths[self.uid]
            )
            robot_name = os.path.splitext(os.path.basename(urdf_path))[0]

            vs_str = f"{voxel_size:.2f}".replace(".", "_")
            nd_str = str(num_directions)
            save_dir = (
                REACHABLE_XPOS_DIR / f"Voxel_{robot_name}_{name}_{vs_str}_{nd_str}"
            )

            # 4) Set up workspace analyzer
            analyzer = WorkspaceAnalyzer(
                robot=self, name=name, joint_ranges=joint_ranges
            )

            # 5) Sample voxels and IK
            (
                voxel_centers,
                voxel_success_counts,
                sampled_xpos,
            ) = analyzer.sample_voxel_workspace(
                voxel_size=voxel_size,
                num_directions=num_directions,
                num_yaws=num_yaws,
                pos_eps=pos_eps,
                rot_eps=rot_eps,
                max_iterations=max_iterations,
                num_samples=num_samples,
                cache_mode=cache_mode,
                batch_size=5000,
                save_dir=save_dir,
                save_threshold=10_000_000,
                use_cached=use_cached,
            )

            # 6) Visualization (optional)
            if visualize:
                colormap = colormaps.get_cmap("jet")
                actor_handles: List[MeshObject] = []

                for idx, (center, count) in enumerate(
                    zip(voxel_centers, voxel_success_counts), start=1
                ):
                    # map success rate to color
                    frac = count / num_directions
                    color = colormap(1.0 - frac)[:3]

                    # build and color sphere mesh
                    sphere = o3d.geometry.TriangleMesh.create_sphere(voxel_size / 2)
                    sphere.paint_uniform_color(color)

                    verts = np.asarray(sphere.vertices)
                    inds = np.asarray(sphere.triangles)
                    cols = np.asarray(sphere.vertex_colors)
                    cols4 = np.ones((cols.shape[0], 4), dtype=float)
                    cols4[:, :3] = cols

                    # create uniquely named actor e.g. "sphere1", "sphere2", …
                    actor_name = f"sphere{idx}"
                    actor = self._env.create_actor(actor_name, True, True)
                    actor.set_mesh(
                        vertices=verts,
                        indices=inds,
                        shape=PrimitiveType.TRIANGLES,
                        smooth_angle=-1,
                        colors=cols4,
                    )
                    actor.set_location(*center)

                    actor_handles.append(actor)

                return sampled_xpos, actor_handles

            # 7) Return only sampled poses
            return sampled_xpos, None

        except Exception as e:
            print(f"Failed to visualize voxel workspace: {e}")
            return None, None

    def destroy(self) -> None:
        r"""Release the resources of the robot."""
        # Safely handle drive_controllers
        if hasattr(self, "drive_controllers") and isinstance(
            self.drive_controllers, dict
        ):
            for key in self.drive_controllers.keys():
                self.drive_controllers[key] = None

        # Safely handle task_controllers
        if hasattr(self, "task_controllers") and isinstance(
            self.task_controllers, dict
        ):
            for key in self.task_controllers.keys():
                self.task_controllers[key] = None

        # Safely handle articulation
        if hasattr(self, "articulation"):
            self.articulation = None

        # Safely handle child_articulations
        if hasattr(self, "child_articulations") and isinstance(
            self.child_articulations, dict
        ):
            for key in self.child_articulations.keys():
                if self.child_articulations[key] is not None:
                    if hasattr(self.child_articulations[key], "get_articulation"):
                        self._env.remove_articulation(
                            self.child_articulations[key].get_articulation()
                        )
                    else:
                        self._env.remove_articulation(self.child_articulations[key])

                self.child_articulations[key] = None

    @staticmethod
    def build_pk_serial_chain(**kwargs) -> Dict[str, pk.SerialChain]:
        """Build the serial chain from the URDF file.

        Args:
            **kwargs: Additional arguments for building the serial chain.

        Returns:
            Dict[str, pk.SerialChain]: The serial chain of the robot.
        """
        # paths = self.get_urdf_path()
        # urdf_path = paths if isinstance(paths, str) else paths[self.uid]
        # chain = pk.build_chain_from_urdf(open(urdf_path, mode="rb").read())

        # articulation = robot.get_articulation(self.uid)
        # link_names = articulation.get_link_names()
        # serial_chain = pk.SerialChain(chain, link_names[-1], link_names[0])

        # return {self.uid: serial_chain}
        return {}
