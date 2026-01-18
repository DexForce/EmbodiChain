# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------

from embodichain.data.global_indices import (
    GLOBAL_INDICES,
    STATE_VEC_LEN,
)
from embodichain.data.global_mapping import GlobalMapping
import numpy as np
from typing import List, Dict, Tuple, Union
from embodichain.data.enum import (
    ArmEnum,
    Modality,
    JointType,
    ActionMode,
    EefType,
    ControlParts,
    EndEffector,
    Modality,
)
from embodichain.utils.logger import log_info, log_warning

DEFAULT_EMPTY_STATE = -1

__all__ = ["StateUnifier", "ActionIndicesGenerator"]

"""Unified state utilities for EmbodiChain.

This module provides helpers to construct and query a unified state/action
vector representation used across EmbodiChain environments and agents.

Classes:
    StateUnifier: Fill sparse per-modality state/action dictionaries into a
        fixed-length unified state vector where unspecified entries are set
        to a sentinel value (DEFAULT_EMPTY_STATE).

    ActionIndicesGenerator: Query index ranges in the unified vector for
        common action/state groups (e.g. qpos, delta qpos, end-effector pose).

Constants:
    DEFAULT_EMPTY_STATE (int): Sentinel value used to mark unspecified
        entries in the unified vector.
"""


class StateUnifier:
    """Convert per-modality state/action arrays into a unified vector.

    The StateUnifier is constructed with ``robot_meta`` (the robot's
    metadata) which should contain an ``observation`` mapping with keys for
    modalities (e.g. ``Modality.STATES``) and an ``actions`` specification.

    Attributes:
        metadata (dict): Robot metadata passed at construction.
        arm_dofs (int): Degrees of freedom for the arm (default: 12).
        indices_generator (ActionIndicesGenerator): Helper for action indices.
        proprio_meta: Metadata list for proprioceptive modalities.
        global_mapping (GlobalMapping): Mapping from names to unified indices.
        output: Action output specification from metadata.
        state_dim (int): Fixed length of the unified state vector.
    """

    def __init__(self, robot_meta: Dict) -> None:
        assert "arm_dofs" in robot_meta
        assert "observation" in robot_meta
        assert Modality.ACTIONS.value in robot_meta

        self.arm_dofs = robot_meta["arm_dofs"]
        self.indices_generator = ActionIndicesGenerator(self.arm_dofs)
        self.proprio_meta = robot_meta["observation"][Modality.STATES.value]
        self.global_mapping = GlobalMapping(self.arm_dofs)
        self.output = robot_meta[Modality.ACTIONS.value]

        self.state_dim = STATE_VEC_LEN

    def fill_in_state(
        self, values: Union[np.ndarray, Dict[str, np.ndarray]]
    ) -> np.ndarray:
        """Fill a unified state vector from given values.

        Args:
            values (np.ndarray or dict): If ``values`` is a numpy array it is
                assumed to already be aligned to the unified layout and will
                be placed into the output container. If it is a ``dict``,
                keys should match entries from the robot metadata
                ``observation[Modality.STATES]`` and values are numpy arrays
                with a trailing dimension matching each state's width.

        Returns:
            np.ndarray: An array with shape ``(..., STATE_VEC_LEN)`` containing
                the unified state with unspecified entries set to
                ``DEFAULT_EMPTY_STATE``.
        """
        if isinstance(values, np.ndarray):
            UNI_STATE_INDICES = self.global_mapping.get_indices(self.proprio_meta)
            uni_vec = (
                np.ones(values.shape[:-1] + (self.state_dim,)) * DEFAULT_EMPTY_STATE
            )
            uni_vec[..., UNI_STATE_INDICES] = values
            return uni_vec
        else:
            shape_tuple_list = []
            for val in values.values():
                shape_tuple = val.shape[:-1]
                if val.size != 0:
                    shape_tuple_list.append(shape_tuple)

            shape_tuple = list(set(shape_tuple_list))
            assert len(shape_tuple) == 1, "shape tuple {} is not unique.".format(
                shape_tuple
            )
            uni_vec = np.ones(shape_tuple[0] + (self.state_dim,)) * DEFAULT_EMPTY_STATE
            for state_name in self.proprio_meta:
                state_indices = self.global_mapping.get_indices([state_name])
                if values[state_name].size != 0:
                    uni_vec[..., state_indices] = values[state_name]

            return uni_vec

    def fill_in_action(
        self, values: Union[np.ndarray, Dict[str, np.ndarray]]
    ) -> np.ndarray:
        """Fill a unified action vector from given action values.

        This mirrors :meth:`fill_in_state` but uses the metadata's action
        output specification to determine which named outputs map into the
        unified vector.

        Args:
            values (np.ndarray or dict): Action values aligned to the unified
                layout or a mapping from output names to numpy arrays.

        Returns:
            np.ndarray: Unified vector shaped ``(..., STATE_VEC_LEN)`` with
                unspecified entries filled with ``DEFAULT_EMPTY_STATE``.
        """
        if isinstance(values, np.ndarray):
            UNI_STATE_INDICES = self.indices_generator.get(self.output)
            uni_vec = (
                np.ones(values.shape[:-1] + (self.state_dim,)) * DEFAULT_EMPTY_STATE
            )
            uni_vec[..., UNI_STATE_INDICES] = values
            return uni_vec
        else:
            shape_tuple_list = []
            for key, val in values.items():

                shape_tuple = val.shape[:-1]
                if val.size != 0:
                    shape_tuple_list.append(shape_tuple)

            shape_tuple = list(set(shape_tuple_list))
            assert len(shape_tuple) == 1, "shape tuple {} is not unique.".format(
                shape_tuple
            )

            uni_vec = np.ones(shape_tuple[0] + (self.state_dim,)) * DEFAULT_EMPTY_STATE
            for out_name in self.output:
                state_indices = self.global_mapping.get_indices([out_name])
                if out_name in values and values[out_name].size != 0:
                    uni_vec[..., state_indices] = values[out_name]
            return uni_vec


class ActionIndicesGenerator:
    """Utility for generating index lists for action/state groups.

    The ActionIndicesGenerator wraps :class:`GlobalMapping` to provide
    common queries like retrieving indices for all joint positions (qpos),
    delta qpos (relative mode), end-effector transforms/poses, and
    hand-specific selections (left/right/both).

    Args:
        dof (int, optional): If provided, a :class:`GlobalMapping` is
            constructed and reused for queries.
    """

    def __init__(self, dof: int = None):
        self.global_mapping = None
        self.dof = dof
        if dof is not None:
            self.global_mapping = GlobalMapping(dof)

    def get_all_qpos(
        self, dof: int = None, handness: str = ArmEnum.DUAL_ARM.value
    ) -> List[int]:
        """Return indices covering all joint position entries.

        Args:
            dof (int, optional): Degrees of freedom to construct a temporary
                :class:`GlobalMapping` if the generator was not initialized
                with a ``dof``.
            handness (str): One of values from :class:`ArmEnum` specifying
                which arm(s) to include.

        Returns:
            List[int]: Ordered list of indices in the unified vector
                corresponding to qpos entries for the requested arm
                selection.
        """
        qpos_name = JointType.QPOS.value
        delta_qpos_name = ActionMode.RELATIVE.value + qpos_name
        global_mapping = self.get_mapping(dof)

        all_names = list(global_mapping.mapping_from_name_to_indices.keys())
        if handness == ArmEnum.DUAL_ARM.value:
            return self.get(all_names, dof, [qpos_name], [delta_qpos_name])
        elif handness == ArmEnum.LEFT_ARM_ONLY.value:
            handness = ControlParts.LEFT_ARM.value
            inv_handness = ControlParts.RIGHT_ARM.value
            return self.get(
                all_names, dof, [qpos_name], [delta_qpos_name, inv_handness + qpos_name]
            )
        elif handness == ArmEnum.RIGHT_ARM_ONLY.value:
            handness = ControlParts.RIGHT_ARM.value
            inv_handness = ControlParts.LEFT_ARM.value
            return self.get(
                all_names, dof, [qpos_name], [delta_qpos_name, inv_handness + qpos_name]
            )

    def get_all_delta_qpos(
        self, dof: int = None, handness: str = ArmEnum.DUAL_ARM.value
    ) -> List[int]:
        """Return indices for delta (relative) joint position entries.

        Args and return are the same as :meth:`get_all_qpos` but select the
        ``ActionMode.RELATIVE`` named entries.
        """
        qpos_name = JointType.QPOS.value
        delta_qpos_name = ActionMode.RELATIVE.value + qpos_name
        global_mapping = self.get_mapping(dof)

        all_names = list(global_mapping.mapping_from_name_to_indices.keys())
        if handness == ArmEnum.DUAL_ARM.value:
            return self.get(all_names, dof, [delta_qpos_name], [])
        elif handness == ArmEnum.LEFT_ARM_ONLY.value:
            inv_handness = ControlParts.RIGHT_ARM.value
            return self.get(
                all_names, dof, [delta_qpos_name], [inv_handness + delta_qpos_name]
            )
        elif handness == ArmEnum.RIGHT_ARM_ONLY.value:
            inv_handness = ControlParts.LEFT_ARM.value
            return self.get(
                all_names, dof, [delta_qpos_name], [inv_handness + delta_qpos_name]
            )

    def get_all_eef(
        self,
        dof: int = None,
        eef_effector: str = "",
        handness: str = ArmEnum.DUAL_ARM.value,
    ) -> List[int]:
        """Retrieves the indices of all end-effectors (EEF) based on the specified parameters.

        Args:
            dof (int, optional): Degree of freedom to use for mapping. If None, uses default.
            eef_effector (str, optional): Type of end-effector. Must be one of
                EndEffector.DEXTROUSHAND.value, EndEffector.GRIPPER.value, or "" (empty string).
            handness (str, optional): Specifies which arm(s) to consider. Must be one of
                ArmEnum.DUAL_ARM.value, ArmEnum.LEFT_ARM_ONLY.value, or ArmEnum.RIGHT_ARM_ONLY.value.

        Returns:
            List[int]: List of indices corresponding to the selected end-effectors.

        Raises:
            AssertionError: If an invalid end-effector type is provided.
        """
        assert eef_effector in [
            EndEffector.DEXTROUSHAND.value,
            EndEffector.GRIPPER.value,
            "",
        ], "Invalid end-effector effector type {}.".format(eef_effector)
        global_mapping = self.get_mapping(dof)
        all_names = list(global_mapping.mapping_from_name_to_indices.keys())
        if handness == ArmEnum.DUAL_ARM.value:
            return self.get(
                all_names,
                dof,
                [
                    ControlParts.LEFT_EEF.value + eef_effector,
                    ControlParts.RIGHT_EEF.value + eef_effector,
                ],
                [],
            )
        elif handness == ArmEnum.LEFT_ARM_ONLY.value:
            handness = ControlParts.LEFT_EEF.value
            return self.get(
                all_names,
                dof,
                [handness + eef_effector],
                [],
            )
        elif handness == ArmEnum.RIGHT_ARM_ONLY.value:
            handness = ControlParts.RIGHT_EEF.value
            return self.get(
                all_names,
                dof,
                [handness + eef_effector],
                [],
            )

    def get_all_eef_pose(
        self, dof: int = None, handness: str = ArmEnum.DUAL_ARM.value
    ) -> List[int]:
        """Return indices specifically for EEF pose entries.

        Args:
            dof (int, optional): Degrees of freedom for mapping lookup.
            handness (str): Which arm(s) to include (left/right/both).

        Returns:
            List[int]: Indices corresponding to EEF poses.
        """
        global_mapping = self.get_mapping(dof)
        all_names = list(global_mapping.mapping_from_name_to_indices.keys())

        if handness == ArmEnum.DUAL_ARM.value:
            return self.get(all_names, dof, [EefType.POSE.value], [])
        elif handness == ArmEnum.LEFT_ARM_ONLY.value:
            handness = ControlParts.LEFT_ARM.value
            return self.get(all_names, dof, [handness + EefType.POSE.value], [])
        elif handness == ArmEnum.RIGHT_ARM_ONLY.value:
            handness = ControlParts.RIGHT_ARM.value
            return self.get(all_names, dof, [handness + EefType.POSE.value], [])

    def get_mapping(self, dof: int = None):
        """Return the :class:`GlobalMapping` used by this generator.

        If a mapping was created during initialization (because ``dof`` was
        provided), ensure any provided ``dof`` argument matches it. Otherwise
        construct and return a temporary :class:`GlobalMapping` for the
        requested ``dof``.

        Args:
            dof (int, optional): Degrees of freedom to construct a mapping
                if one was not provided at initialization.

        Returns:
            GlobalMapping: Mapping instance for name->index lookups.
        """
        if self.global_mapping is not None:
            assert dof is None or dof == self.dof
            global_mapping = self.global_mapping
        else:
            assert (
                dof is not None
            ), "Dof must be set when dof is not provided in initialization."
            global_mapping = GlobalMapping(dof)
        return global_mapping

    def get(
        self,
        output: List[str],
        dof: int = None,
        white_list: List[str] = None,
        black_list: List[str] = None,
    ) -> List[int]:
        """Select and return indices from ``output`` names applying optional
        white/black list filters.

        Args:
            output (List[str]): Names (keys) in a :class:`GlobalMapping`
                whose indices should be collected.
            dof (int, optional): Degrees of freedom used to construct a
                temporary :class:`GlobalMapping` if needed.
            white_list (List[str], optional): If provided, only include names
                that contain any of these substrings.
            black_list (List[str], optional): If provided, exclude names
                that contain any of these substrings.

        Returns:
            List[int]: Ordered list of unified-vector indices for the
                selected names.
        """

        action_indices = []
        global_mapping = self.get_mapping(dof)

        for action_type in output:
            if isinstance(white_list, list) and isinstance(black_list, list):
                if any([temp in action_type for temp in white_list]) and all(
                    [temp not in action_type for temp in black_list]
                ):
                    action_indices += global_mapping.mapping_from_name_to_indices[
                        action_type
                    ]
            else:
                action_indices += global_mapping.mapping_from_name_to_indices[
                    action_type
                ]

        return action_indices  # keep order.
