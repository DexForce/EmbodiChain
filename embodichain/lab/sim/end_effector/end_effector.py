# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------
import typing
import dexsim.engine
import numpy as np
import dexsim.environment
from dexsim.types import DriveType, PhysicalAttr, ArticulationFlag, ActorType

from embodichain.lab.sim.end_effector.utility import (
    load_model_from_file,
    inv_transform,
)
from abc import ABC, abstractmethod
from embodichain.lab.sim.articulation_entity import ArticulationEntity
from embodichain.utils import logger
import dexsim
import time


class EndEffector(ArticulationEntity, ABC):
    r"""
    Abstract class for end effector in simulation.
    """

    def __init__(
        self,
        env: dexsim.environment.Arena,
        file: str,
        drive_type: DriveType = DriveType.FORCE,
        **kwargs,
    ) -> None:
        """init end effector

        Args:
            env (dexsim.environment.Arena): dexsim environment.
            file (str): input file (urdf or mesh file)
            drive_type (DriveType, optional): DriveType.FORCE or DriveType.FORCE. Defaults to DriveType.FORCE.
            kwargs(optional): Accepts additional keyword arguments.
        """
        urdf_path = load_model_from_file(file_path=file)

        super().__init__(
            urdf_path=urdf_path,
            init_qpos=None,
            init_base_xpos=np.eye(4),
            speed_ratio=0.5,
            time_step=0.02,
            drive_type=drive_type,
            env=env,
            **kwargs,
        )

        self._init_end_effector(**kwargs)

        self.articulation.set_physical_attr(self.default_physical_attrs)
        self.articulation.set_drive(
            drive_type=self.drive_type, **self.default_drive_param
        )

    @abstractmethod
    def _init_end_effector(self, **kwargs) -> None:
        r"""Initializes the robot using the URDF path with necessary parameters."""
        pass

    def _set_ee_control_data(self, **kwargs):
        self._dof = self.articulation.get_dof()
        self._actived_joint_names = self.articulation.get_actived_joint_names()
        self._root_link_name = self.articulation.get_root_link_name()
        self._attached_nodes = dict()  # {node_name: [dexsim.engine.Node, ActorType]}
        self._leaf_link_names = self.articulation.get_leaf_link_names()

        if self._dof > 0:
            # ignore mimic information for 0-dof articulation
            self._joint_ids[self.uid] = np.arange(self._dof)
            self._joint_limit = self.articulation.get_joint_limits()
            self._set_mimic()

        self.attach_robot_uid = None  # if end-effector is attach to robot.

        # KWARGS. If true, set object to be dynamic when release object, otherwise do nothing.
        self._is_release_dynamic = kwargs.get("is_release_dynamic", True)

        # open state sample num
        self._open_state_sample_num = kwargs.get("open_state_sample_num", 30)

        # open state and close state
        self.open_state = np.array(
            [
                1.0,
            ]
        )
        self.close_state = np.array(
            [
                0.0,
            ]
        )

    @property
    def actived_joint_names(self) -> typing.List[str]:
        return self._actived_joint_names

    def _set_to_init_qpos(self):
        self._init_qpos = np.array([])
        if self._dof > 0:
            self._init_qpos = self._joint_limit[:, 0]
            self.articulation.set_current_qpos(
                self._init_qpos, self._joint_ids[self.uid]
            )

    def get_init_qpos(self) -> np.ndarray:
        return self._init_qpos

    @property
    def release_dynamic(self) -> bool:
        """get is release dynamic

        Returns:
            bool: If true, set object to be dynamic when release object, otherwise do nothing.
        """
        return self._is_release_dynamic

    @release_dynamic.setter
    def release_dynamic(self, is_release_dynamic: bool):
        """set is release dynamic

        Args:
            is_release_dynamic (bool): If true, set object to be dynamic when release object, otherwise do nothing.
        """
        self._is_release_dynamic = is_release_dynamic

    def _set_mimic(self) -> None:
        r"""Sets up the mimic configuration for the articulation.

        Attributes Updated:
            - self._mimic_joint_ids: Array of joint IDs that are mimicked.
            - self._mimic_master_ids: Array of master joint IDs that control the mimicked joints.
            - self._mimic_multipliers: Array of multipliers for the mimicked joints.
            - self._mimic_offsets: Array of offsets for the mimicked joints.
            - self._control_joint_ids: Array of joint IDs that are not mimicked and can be controlled.
            - self._control_limit: Joint limits for the controllable joints.
            - self._control_num: Number of controllable joints.
        """
        mimic_info = self.articulation.get_mimic_info()

        self._mimic_joint_ids = mimic_info.mimic_id
        self._mimic_master_ids = mimic_info.mimic_parent
        self._mimic_multipliers = mimic_info.mimic_multiplier
        self._mimic_offsets = mimic_info.mimic_offset

        # Using set for faster membership testing
        mimic_joint_set = set(self._mimic_joint_ids)

        # List comprehension for better readability and performance
        self._control_joint_ids = np.array(
            [i for i in range(self._dof) if i not in mimic_joint_set]
        )

        self._control_limit = self._joint_limit[self._control_joint_ids]
        self._control_num = self._control_joint_ids.shape[0]

    def _qpos_to_control_state(self, qpos: np.ndarray) -> np.ndarray:
        """full joint state to control joint state

        Args:
            qpos (np.ndarray): [dof] of float. Full joint state.

        Returns:
            np.ndarray: [control_joint_num] of float. control joint state
        """
        return qpos[self._control_joint_ids]

    def _control_state_to_qpos(self, control_state: np.ndarray) -> np.ndarray:
        """control joint state to full joint state

        Args:
            control_state (np.ndarray): [control_joint_num] of float. control joint state

        Returns:
            np.ndarray: [dof] of float. Full joint state.
        """
        qpos = np.empty(shape=(self._dof,), dtype=float)
        qpos[self._control_joint_ids] = control_state
        qpos[self._mimic_joint_ids] = (
            qpos[self._mimic_master_ids] * self._mimic_multipliers + self._mimic_offsets
        )
        return qpos

    def _qpos_to_control_state_path(self, qpos_path: np.ndarray):
        return qpos_path[:, self._control_joint_ids]

    def _control_state_to_qpos_path(self, control_state_path: np.ndarray):
        waypoint_num = control_state_path.shape[0]
        qpos_path = np.empty(shape=(waypoint_num, self._dof), dtype=float)
        qpos_path[:, self._control_joint_ids] = control_state_path
        qpos_path[:, self._mimic_joint_ids] = (
            qpos_path[:, self._mimic_master_ids] * self._mimic_multipliers
            + self._mimic_offsets
        )
        return qpos_path

    def _to_arena_pose(self, pose: np.ndarray) -> np.ndarray:
        return inv_transform(self._env.get_root_node().get_world_pose()) @ pose

    def get_xpos(self) -> np.ndarray:
        """get gripper root link pose

        Returns:
            np.ndarray: [4, 4] of float. root link 6d pose
        """
        return self._to_arena_pose(
            self.articulation.get_link_pose(self._root_link_name)
        )

    def set_xpos(self, pose: np.ndarray) -> None:
        """directly set gripper world pose

        Args:
            pose (np.ndarray): [4, 4] of float. root link 6d pose
        """
        # TODO: When gripper attach to robot base, this function result can be wild.
        assert pose.shape == (4, 4)
        self.set_world_pose(self._to_arena_pose(pose))

    def set_world_pose(self, pose: np.ndarray) -> None:
        """Set the world pose of the end effector."""
        assert pose.shape == (4, 4), "Pose must be a 4x4 transformation matrix."
        self.articulation.set_world_pose(pose)

    def get_qpos(self) -> np.ndarray:
        """get robot joint state array

        Returns:
            np.ndarray: (joint_num, ) of float. joint state array
        """
        return np.array(self.articulation.get_current_qpos(self._joint_ids[self.uid]))

    def set_qpos(self, qpos: np.ndarray) -> None:
        """set gripper joint state array

        Args:
            qpos (np.ndarray): (joint_num, ) of float. joint state array
        """
        assert qpos.shape == (self._dof,)
        self.articulation.set_current_qpos(qpos, self._joint_ids[self.uid])

    def get_control_qpos(self) -> np.ndarray:
        """get control joint state

        Returns:
            np.ndarray: (control_joint_num, ) of float.
        """
        return self._qpos_to_control_state(self.get_qpos())

    def set_control_qpos(self, control_state: np.ndarray) -> None:
        """set control joint state

        Args:
            control_state (np.ndarray): (control_joint_num, ) of float
        """
        assert control_state.shape == self._control_joint_ids.shape
        qpos = self._control_state_to_qpos(control_state)
        self.articulation.set_current_qpos(qpos, self._joint_ids[self.uid])

    def move_qpos(self, qpos_path: np.ndarray, is_wait=True, move_time: float = 1):
        assert qpos_path.shape[1] == self._dof
        self.move_joints(
            qpos_path,
            is_wait=is_wait,
            joint_ids=self._joint_ids[self.uid],
            move_time=move_time,
        )

    def get_leaf_link_pose(self) -> dict:
        """get leaf link pose.

        Returns:
            dict: {"link_name", np.ndarray [4, 4]}     pose of each leaf link
        """
        leaf_link_poses = dict()
        for leaf_link_name in self._leaf_link_names:
            leaf_link_pose = self.articulation.get_link_pose(leaf_link_name)
            leaf_link_poses[leaf_link_name] = leaf_link_pose
        return leaf_link_poses

    def get_leaf_contact(self, is_flatten: bool = False) -> dict:
        """Get leaf link contacts.
        Leaf link: 1. has physical body; 2. no child link; 3. parent link is not fixed.

        Args:
            is_flatten (bool): get flatten

        Returns:
            is_flatten == False:
                dict: {
                    "link_name": {
                        "nodes": [dexsim.engine.Node, ...],
                        "contact_positions": [link_contact_num, 3] of float. np.ndarray,
                        "contact_normals": [link_contact_num, 3] of float. np.ndarray,
                        "contact_distances": [link_contact_num] of float. np.ndarray,
                    },
                    ...
                }

            is_flatten == True:
                ContactInfo

                ContactInfo.nodes(List[dexsim.engine.Node]): List of Contact object node ptr
                ContactInfo.link_name(List[str]): List of contact link name
                ContactInfo.contact_positions(np.ndarray): [contact_num, 3] of float, matrix of contact_positions.
                ContactInfo.contact_normals(np.ndarray): [contact_num, 3] of float, matrix of contact normal.
                ContactInfo.contact_distances(np.ndarray): [contact_num] of float.  Contact distance. Negetive for peneration and postive for surface distance.
        """
        contact_info = self.articulation.get_leaf_contacts()
        if is_flatten:
            return contact_info
        link_contact_all_id = np.arange(len(contact_info.nodes))

        contact_info_dict = dict()
        # Tricky implementation. save str ing np.ndarray, and select link name by mask
        contact_link_names = np.array(contact_info.link_name)
        contact_link_name_unique = np.unique(contact_link_names)
        # unpack contact info
        for link_name in contact_link_name_unique:
            contact_info_dict[link_name] = dict()
            link_contact_mask = contact_link_names == link_name
            link_contact_ids = link_contact_all_id[link_contact_mask]
            contact_info_dict[link_name]["nodes"] = []
            for link_contact_idx in link_contact_ids:
                contact_info_dict[link_name]["nodes"].append(
                    contact_info.nodes[link_contact_idx]
                )
            contact_info_dict[link_name][
                "contact_positions"
            ] = contact_info.contact_positions[link_contact_ids]
            contact_info_dict[link_name][
                "contact_normals"
            ] = contact_info.contact_normals[link_contact_ids]
            contact_info_dict[link_name][
                "contact_distances"
            ] = contact_info.contact_distances[link_contact_ids]
        return contact_info_dict

    def get_cpp_articulation(self):
        return self.articulation

    def attach(self, node: dexsim.engine.Node) -> str:
        """attach certain actor to current end-effector
            (will attach to root link)

        Args:
            node (dexsim.engine.Node): dexsim actor

        Returns:
            str: Name of the attached actor, return none str if will attach wrong actor.
        """
        node_name = node.get_name()
        original_actor_type = node.get_actor_type()

        if original_actor_type == ActorType.STATIC:
            logger.log_info(
                "Skipping attachment to static object, its name: {}.".format(node_name)
            )
            return ""
        if original_actor_type == ActorType.DYNAMIC:
            # TODO: tricky implemetation. Fix dynamic actor to kinematic
            node.set_actor_type(ActorType.KINEMATIC)
            # node.enable_collision(False)

        node_pose = node.get_local_pose()
        self_pose = self.get_xpos()
        relative_pose = inv_transform(self_pose) @ node_pose

        self.articulation.attach_node(
            obj=node, link_name=self._root_link_name, relative_pose=relative_pose
        )

        self._attached_nodes[node_name] = [node, original_actor_type]
        return node_name

    def detach(self, node_name: str) -> bool:
        """detach certain actor to current suctor

        Args:
            actor (dexsim.models.Entity): dexsim actor

        Returns:
            bool: is_success
        """
        if node_name in self._attached_nodes:
            node = self._attached_nodes[node_name][0]
            original_actor_type = self._attached_nodes[node_name][1]
            arena_root_node = self._env.get_root_node()
            node.attach_node(arena_root_node)
            if original_actor_type != ActorType.STATIC and self._is_release_dynamic:
                node.set_actor_type(ActorType.DYNAMIC)
                # node.enable_collision(True)
            self._attached_nodes.pop(node_name)
            return True
        else:
            logger.log_warning(f"Actor {node_name} to be detach is not attached yet.")
            return False

    @abstractmethod
    def get_control_state(self, **kwargs) -> np.ndarray:
        """get control state of end-effector

        Returns:
            np.ndarray: [state_dof] of float. Control state array
        """

    @abstractmethod
    def get_open_state(self, **kwargs) -> np.ndarray:
        """get control state of end-effector

        Returns:
            np.ndarray: [state_dof] of float. Open state array
        """

    @abstractmethod
    def set_open_state(self, open_state: np.ndarray, **kwargs):
        """set control state of end-effector

        Args:
            open_state (np.ndarray): [state_dof] of float. Open state
        """

    def to_target_open_state_path(
        self,
        target_open_state: np.ndarray,
        start_open_state: np.ndarray = None,
        step_num: int = None,
        step_size: float = None,
        **kwargs,
    ) -> np.ndarray:
        """Generate a path from the start open state to the target open state for a gripper or a robotic hand.

        An "open state" refers to the configuration of the gripper or robotic hand at a given moment,
        which can include the positions of fingers, joints, and any gripping mechanisms.
        The "target state" is the desired configuration that the gripper or hand should achieve after
        the motion, typically used for grasping or releasing an object.

        Args:
            target_open_state (np.ndarray): Target open state, shape [state_dof].
            start_open_state (np.ndarray, optional): Starting open state, shape [state_dof]. Default is None, which uses the current open state.
            step_num (int, optional): Number of interpolation points. Default is None.
            step_size (float, optional): Step size for interpolation. Default is None.

        Returns:
            np.ndarray: Path as an array of shape [waypoint_num, state_dof].
        """

        if start_open_state is None:
            start_open_state = self.get_open_state()

        if step_num is not None and step_size is not None:
            logger.log_warning(
                "Please provide either 'step_num' or 'step_size', not both."
            )
            return []

        if step_num is not None:
            step_num = max(step_num, 1)
        elif step_size is not None:
            distance = np.linalg.norm(target_open_state - start_open_state)
            step_num = int(np.ceil(distance / step_size))
        else:
            state_range = np.abs(start_open_state - target_open_state).max()
            step_num = int(np.round(self._open_state_sample_num * state_range))

        open_state_path = np.linspace(start_open_state, target_open_state, step_num)

        return open_state_path

    def open(self, **kwargs):
        """open end-effector. only for demo"""
        if self._world is not None:
            if self._world.is_physics_manually_update():
                logger.log_warning("Cannot call open in physics manually update mode.")
                return
        open_state_path = self.to_target_open_state_path(self.open_state)
        for i in range(open_state_path.shape[0]):
            self.set_open_state(open_state_path[i])
            time.sleep(0.02)

    def close(self, **kwargs):
        """close end-effector. only for demo"""
        if self._world is not None:
            if self._world.is_physics_manually_update():
                logger.log_warning("Cannot call close in physics manually update mode.")
                return
        open_state_path = self.to_target_open_state_path(self.close_state)
        for i in range(open_state_path.shape[0]):
            self.set_open_state(open_state_path[i])
            time.sleep(0.02)

    @property
    def default_physical_attrs(self) -> PhysicalAttr:
        physical_attr = PhysicalAttr()
        if self.drive_type == DriveType.FORCE:
            physical_attr.mass = 0.01  # TODO: mass setting is not activated currently
            physical_attr.static_friction = 2.0
            physical_attr.dynamic_friction = 1.5
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
            physical_attr.mass = 0.01  # TODO: mass setting is not activated currently
            physical_attr.static_friction = 2.0
            physical_attr.dynamic_friction = 1.5
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
    def default_drive_param(self) -> typing.Dict:
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
            if hasattr(self, "max_force"):
                max_force = self.max_force
            else:
                max_force = 1e3
            param = {"stiffness": 1e2, "damping": 1e1, "max_force": max_force}
        elif self.drive_type == DriveType.FORCE:
            param = {"stiffness": 1e8, "damping": 1e6, "max_force": 1e10}
        return param
