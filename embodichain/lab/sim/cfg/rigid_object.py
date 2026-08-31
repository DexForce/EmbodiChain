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

"""Rigid object and rigid-object-group configuration."""

from __future__ import annotations

from dataclasses import MISSING
import os
from typing import Any, Dict, Literal

from dexsim.types import ActorType

from embodichain.utils import configclass, is_configclass, logger

from ..shapes import ShapeCfg
from .asset import AssetPhysicsMode, ObjectBaseCfg, _resolve_asset_physics_mode
from .rigid import RigidBodyPhysicsCfg


@configclass
class RigidObjectCfg(ObjectBaseCfg):
    """Configuration for a rigid body asset in the simulation.

    This class extends the base asset configuration to include specific properties for rigid bodies,
    such as physical attributes and collision group.
    """

    shape: ShapeCfg = ShapeCfg()
    """Shape configuration for the rigid body. """

    # TODO: supoort basic primitive shapes, such as box, sphere, etc cfg and spawn method.

    attrs: RigidBodyPhysicsCfg = RigidBodyPhysicsCfg()
    """Rigid-body physics.

    :class:`RigidBodyPhysicsCfg` groups portable and backend-native intent.
    """

    body_type: Literal["dynamic", "kinematic", "static"] = "dynamic"

    body_scale: tuple | list = (1.0, 1.0, 1.0)
    """Scale of the rigid body in the simulation world frame."""

    asset_physics_mode: AssetPhysicsMode = "preserve"
    """How a file-backed asset's physical properties are handled.

    ``"preserve"`` keeps the USD-authored physics. ``"overlay"`` applies
    configured properties on top of the parsed asset. Procedural shapes always
    use config.
    """

    def resolve_asset_physics_mode(self) -> AssetPhysicsMode:
        """Return the effective file-backed physics policy."""
        return _resolve_asset_physics_mode(self.asset_physics_mode)

    def to_dexsim_body_type(self) -> ActorType:
        """Convert the body type to dexsim ActorType."""
        if self.body_type == "dynamic":
            return ActorType.DYNAMIC
        elif self.body_type == "kinematic":
            return ActorType.KINEMATIC
        elif self.body_type == "static":
            return ActorType.STATIC
        else:
            logger.log_error(
                f"Invalid body type '{self.body_type}' specified. Must be one of 'dynamic', 'kinematic', or 'static'."
            )


@configclass
class RigidObjectGroupCfg:
    """Configuration for a rigid object group asset in the simulation.

    Rigid object groups can be initialized from multiple rigid object configurations specified in a folder.
    If `folder_path` is specified, user should provide a RigidObjectCfg in `rigid_objects` as a template configuration for
    all objects in the group.

    For example:
    ```python
    rigid_object_group: RigidObjectGroupCfg(
        folder_path="path/to/folder",
        max_num=5,
        rigid_objects={
            "template_obj": RigidObjectCfg(
                shape=MeshCfg(
                    fpath="",  # fpath will be ignored when folder_path is specified
                ),
                body_type="dynamic",
            )
        }
    )
    """

    uid: str | None = None

    rigid_objects: Dict[str, RigidObjectCfg] = MISSING
    """Configuration for the rigid objects in the group."""

    body_type: Literal["dynamic", "kinematic"] = "dynamic"
    """Body type for all rigid objects in the group. """

    folder_path: str | None = None
    """Path to the folder containing the rigid object assets.
    
    This is used to initialize multiple rigid object configurations from a folder.
    """

    max_num: int = 1
    """Maximum number of rigid objects to initialize from the folder.
    
    This is only used when `folder_path` is specified.
    """

    ext: str = ".obj"
    """File extension for the rigid object assets.
    
    This is only used when `folder_path` is specified.
    """

    @classmethod
    def from_dict(cls, init_dict: Dict[str, Any]) -> RigidObjectGroupCfg:
        """Initialize the configuration from a dictionary."""
        cfg = cls()
        for key, value in init_dict.items():
            if hasattr(cfg, key):
                attr = getattr(cfg, key)
                if is_configclass(attr):
                    setattr(
                        cfg, key, attr.from_dict(value)
                    )  # Call from_dict on the attribute
                elif key == "rigid_objects" and "folder_path" not in init_dict:
                    rigid_objects_cfg = {}
                    for obj_name, obj_cfg in value.items():
                        rigid_objects_cfg[obj_name] = RigidObjectCfg.from_dict(obj_cfg)
                    setattr(cfg, key, rigid_objects_cfg)
                elif key == "rigid_objects" and "folder_path" in init_dict:
                    folder_path = init_dict["folder_path"]
                    max_num = init_dict.get("max_num", 1)
                    rigid_objects_cfg = {}
                    if os.path.exists(folder_path) and os.path.isdir(folder_path):
                        files = os.listdir(folder_path)
                        files = [f for f in files if f.endswith(cfg.ext)]
                        # select files up to max_num
                        n_file = len(files)
                        select_files = []
                        for i in range(max_num):
                            select_files.append(files[i % n_file])

                        for i, file_name in enumerate(select_files):
                            file_path = os.path.join(folder_path, file_name)
                            rigid_obj_cfg: RigidObjectCfg = RigidObjectCfg.from_dict(
                                list(init_dict["rigid_objects"].values())[0]
                            )
                            rigid_obj_cfg.uid = f"{cfg.uid}_obj_{i}"
                            rigid_obj_cfg.shape.fpath = file_path
                            rigid_objects_cfg[rigid_obj_cfg.uid] = rigid_obj_cfg
                        setattr(cfg, "rigid_objects", rigid_objects_cfg)
                    else:
                        logger.log_error(
                            f"Folder '{folder_path}' does not exist or is not a directory."
                        )
                else:
                    setattr(cfg, key, value)
            else:
                logger.log_warning(
                    f"Key '{key}' not found in {cfg.__class__.__name__}."
                )
        return cfg
