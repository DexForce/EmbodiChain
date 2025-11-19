# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------

"""
Gizmo utility functions for EmbodiSim.

This module provides utility functions for creating gizmo transform callbacks.
"""

from typing import Callable
from dexsim.types import TransformMask


def create_gizmo_callback() -> Callable:
    """Create a standard gizmo transform callback function.

    This callback handles basic translation and rotation operations for gizmo controls.
    It applies transformations directly to the node when gizmo controls are manipulated.

    Returns:
        Callable: A callback function that can be used with gizmo.node.set_flush_transform_callback()
    """

    def gizmo_transform_callback(node, translation, rotation, flag):
        if node is not None:
            if flag == (TransformMask.TRANSFORM_LOCAL | TransformMask.TRANSFORM_T):
                # Handle translation changes
                node.set_translation(translation)
            elif flag == (TransformMask.TRANSFORM_LOCAL | TransformMask.TRANSFORM_R):
                # Handle rotation changes
                node.set_rotation_rpy(rotation)

    return gizmo_transform_callback
