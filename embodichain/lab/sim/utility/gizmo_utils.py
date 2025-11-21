# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
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
