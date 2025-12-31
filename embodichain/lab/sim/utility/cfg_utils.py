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

from embodichain.lab.sim.cfg import RobotCfg
from embodichain.lab.sim.solvers import SolverCfg
from embodichain.utils import logger


def merge_solver_cfg(
    default: dict[str, SolverCfg], provided: dict[str, any]
) -> dict[str, SolverCfg]:
    """Merge provided solver configuration into the default solver config.

    Rules:
    - For each arm key in provided, if the key exists in default, update fields provided.
    - If a provided value is a dict, update attributes on the SolverCfg-like object (or dict) by setting keys.
    - Primitive values or arrays/lists replace the target value.
    - Unknown keys in provided create new entries in the result.
    """

    result = {}
    # copy defaults shallowly
    for k, v in default.items():
        result[k] = v

    for k, v in provided.items():
        if k in result:
            target = result[k]
            # if target has __dict__ or is a dataclass-like, set attrs
            if hasattr(target, "__dict__") or isinstance(target, dict):
                # if provided is a dict, set/override attributes
                if isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        # try to set attribute if possible, otherwise assign into dict
                        if hasattr(target, sub_k):
                            try:
                                setattr(target, sub_k, sub_v)
                            except Exception:
                                # fallback to dict assignment if object doesn't accept
                                try:
                                    target[sub_k] = sub_v
                                except Exception:
                                    pass
                        else:
                            try:
                                target[sub_k] = sub_v
                            except Exception:
                                setattr(target, sub_k, sub_v)
                else:
                    # non-dict provided value replaces the target entirely
                    result[k] = v
            else:
                # target is a primitive, replace
                result[k] = v
        else:
            # new solver entry provided; include as-is
            result[k] = v

    return result


def merge_robot_cfg(base_cfg: RobotCfg, override_cfg_dict: dict[str, any]) -> RobotCfg:
    """Merge current robot configuration with overriding values from a dictionary.

    Args:
        base_cfg (RobotCfg): The base robot configuration.
        override_cfg_dict (dict[str, any]): Dictionary of overriding configuration values.

    Returns:
        RobotCfg: The merged robot configuration.
    """

    robot_cfg = RobotCfg.from_dict(override_cfg_dict)

    for key, value in override_cfg_dict.items():
        if key == "solver_cfg":
            # merge provided solver_cfg values into default solver config
            provided_solver_cfg = override_cfg_dict.get("solver_cfg")
            if provided_solver_cfg:
                for part, item in provided_solver_cfg.items():
                    if "class_type" in provided_solver_cfg[part]:
                        base_cfg.solver_cfg[part] = robot_cfg.solver_cfg[part]
                    else:
                        try:
                            merged = merge_solver_cfg(
                                base_cfg.solver_cfg, provided_solver_cfg
                            )
                            base_cfg.solver_cfg = merged
                        except Exception:
                            logger.log_error(
                                f"Failed to merge solver_cfg, using provided config outright."
                            )
        elif key == "drive_pros":
            # merge drive_pros
            user_drive_pros_dict = override_cfg_dict.get("drive_pros")
            if isinstance(user_drive_pros_dict, dict):
                for prop, val in user_drive_pros_dict.items():
                    # Get the current value in cfg (which has defaults)
                    default_val = getattr(base_cfg.drive_pros, prop, None)

                    if isinstance(val, dict) and isinstance(default_val, dict):
                        # Merge dictionaries
                        default_val.update(val)
                    else:
                        # Overwrite if not both dicts
                        setattr(base_cfg.drive_pros, prop, val)
            else:
                setattr(base_cfg, key, getattr(robot_cfg, key))
        else:
            setattr(base_cfg, key, getattr(robot_cfg, key))

    return base_cfg
