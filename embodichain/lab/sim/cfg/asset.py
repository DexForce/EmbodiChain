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

"""Base asset configuration and file-backed physics policy."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Dict, Literal

import numpy as np

from embodichain.utils import configclass, is_configclass, logger

AssetPhysicsMode = Literal["preserve", "overlay"]
"""Policy for applying EmbodiChain physics to a file-backed asset."""


def _resolve_asset_physics_mode(
    mode: AssetPhysicsMode,
) -> AssetPhysicsMode:
    """Validate and return a source-agnostic asset-physics policy."""
    if mode not in ("preserve", "overlay"):
        raise ValueError(
            f"asset_physics_mode must be 'preserve' or 'overlay', got {mode!r}."
        )
    return mode


@configclass
class ObjectBaseCfg:
    """Base configuration for an asset in the simulation.

    This class defines the basic properties of an asset, such as its type, initial state, and collision group.
    It is used as a base class for specific asset configurations.
    """

    uid: str | None = None

    init_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Position of the root in simulation world frame. Defaults to (0.0, 0.0, 0.0)."""

    init_rot: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Euler angles (in degree) of the root in simulation world frame. Defaults to (0.0, 0.0, 0.0)."""

    init_local_pose: np.ndarray | None = None
    """4x4 transformation matrix of the root in local frame. If specified, it will override init_pos and init_rot."""

    @classmethod
    def from_dict(cls, init_dict: Dict[str, str | float | tuple]) -> ObjectBaseCfg:
        """Initialize the configuration from a dictionary."""
        cfg = cls()  # Create a new instance of the class (cls)
        for key, value in init_dict.items():
            if hasattr(cfg, key):
                attr = getattr(cfg, key)
                if key == "attrs" and isinstance(value, Mapping):
                    # Keep the base module independent of rigid schemas at
                    # import time; only rigid-derived configs expose this key.
                    from .rigid import _rigid_body_physics_from_dict

                    setattr(cfg, key, _rigid_body_physics_from_dict(value))
                elif is_configclass(attr):
                    setattr(
                        cfg, key, attr.from_dict(value)
                    )  # Call from_dict on the attribute
                else:
                    setattr(cfg, key, value)
            else:
                logger.log_warning(
                    f"Key '{key}' not found in {cfg.__class__.__name__}."
                )

        # Automatically infer init_local_pose if not provided
        if cfg.init_local_pose is None:
            # If only init_pos or init_rot are provided, generate the 4x4 pose matrix
            from scipy.spatial.transform import Rotation as R

            T = np.eye(4)
            T[:3, 3] = np.array(cfg.init_pos)
            T[:3, :3] = R.from_euler("xyz", np.deg2rad(cfg.init_rot)).as_matrix()
            cfg.init_local_pose = T
        else:
            # If only init_local_pose is provided, extract init_pos and init_rot
            from scipy.spatial.transform import Rotation as R

            T = np.array(cfg.init_local_pose)
            cfg.init_pos = tuple(T[:3, 3])
            cfg.init_rot = tuple(R.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True))

        return cfg
