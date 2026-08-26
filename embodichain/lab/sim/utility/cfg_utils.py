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

from typing import TypeVar

from embodichain.lab.sim.cfg import (
    JointDrivePropertiesCfg,
    RigidBodyAttributesCfg,
    RigidBodyPhysicsCfg,
    RobotCfg,
)
from embodichain.lab.sim.solvers import SolverCfg
from embodichain.utils import logger

_ConfigT = TypeVar("_ConfigT")


def _merge_non_none_config(base: _ConfigT | None, override: _ConfigT) -> _ConfigT:
    """Merge non-None configclass fields without discarding base defaults."""
    if base is None:
        return override
    for field_name in override.__dataclass_fields__:
        value = getattr(override, field_name)
        if value is not None:
            setattr(base, field_name, value)
    return base


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

    # Only parse keys the base RobotCfg recognizes, so subclass-only variant
    # fields (version, ...) set by _build_defaults don't trigger
    # spurious "Key not found in RobotCfg" warnings from the base from_dict.
    # NOTE: check RobotCfg.__dataclass_fields__ (not hasattr(base_cfg, k))
    # because base_cfg is the subclass instance which has subclass-only fields,
    # and @configclass strips class-level defaults so hasattr(RobotCfg, k)
    # returns False for all keys.
    base_fields = RobotCfg.__dataclass_fields__
    base_safe = {k: v for k, v in override_cfg_dict.items() if k in base_fields}
    robot_cfg = RobotCfg.from_dict(base_safe)

    for key, value in override_cfg_dict.items():
        if key == "solver_cfg":
            # Per-part merge of provided solver_cfg into default solver config.
            # Two modes:
            #   1. Part dict includes "class_type" → new/replacement solver
            #      (already deserialized into a SolverCfg subclass by
            #      RobotCfg.from_dict above).
            #   2. Part dict lacks "class_type" → attribute overrides for an
            #      existing solver part (e.g. {"tcp": ..., "stiffness": ...}).
            provided_solver_cfg = override_cfg_dict.get("solver_cfg")
            if isinstance(provided_solver_cfg, dict):
                if base_cfg.solver_cfg is None:
                    base_cfg.solver_cfg = {}
                if not provided_solver_cfg:
                    base_cfg.solver_cfg = {}
                    continue
                for part, item in provided_solver_cfg.items():
                    if isinstance(item, dict) and "class_type" in item:
                        # New or replacement solver part — use the deserialized
                        # SolverCfg object produced by RobotCfg.from_dict.
                        parsed = (
                            robot_cfg.solver_cfg.get(part)
                            if isinstance(robot_cfg.solver_cfg, dict)
                            else None
                        )
                        if parsed is not None:
                            base_cfg.solver_cfg[part] = parsed
                        else:
                            logger.log_warning(
                                f"Failed to deserialize solver_cfg['{part}'] "
                                f"with class_type={item.get('class_type')!r}. "
                                f"Skipping."
                            )
                    elif part in base_cfg.solver_cfg:
                        # Existing part — merge individual attribute overrides
                        # in-place so other parts are preserved.
                        target = base_cfg.solver_cfg[part]
                        if isinstance(item, dict):
                            for attr_name, attr_val in item.items():
                                if hasattr(target, attr_name):
                                    setattr(target, attr_name, attr_val)
                    else:
                        logger.log_warning(
                            f"Cannot add solver part {part!r} without "
                            f"'class_type'. Provide 'class_type' to create a "
                            f"new solver entry, or ensure the part name "
                            f"matches an existing solver."
                        )
        elif key == "drive_pros":
            # merge joint drive properties
            user_drive_pros_dict = override_cfg_dict.get("drive_pros")
            if isinstance(user_drive_pros_dict, dict):
                if (
                    user_drive_pros_dict.get("backend") == "newton"
                    or "target_mode" in user_drive_pros_dict
                ):
                    base_cfg.drive_pros = JointDrivePropertiesCfg.from_dict(
                        user_drive_pros_dict,
                        defaults=base_cfg.drive_pros,
                    )
                    continue
                for prop, val in user_drive_pros_dict.items():
                    if prop == "backend":
                        continue
                    # Get the current value in cfg (which has defaults)
                    default_val = getattr(base_cfg.drive_pros, prop, None)

                    if isinstance(val, dict) and isinstance(default_val, dict):
                        # Merge dictionaries
                        default_val.update(val)
                    else:
                        # Overwrite if not both dicts
                        setattr(base_cfg.drive_pros, prop, val)
            else:
                logger.log_warning(
                    "drive_pros should be a dictionary. Skipping drive_pros merge."
                )
        elif key == "attrs":
            # merge physics attributes
            user_attrs_dict = override_cfg_dict.get("attrs")
            if isinstance(user_attrs_dict, dict):
                grouped_fields = set(RigidBodyPhysicsCfg.__dataclass_fields__)
                if grouped_fields.intersection(user_attrs_dict):
                    parsed = RigidBodyPhysicsCfg.from_dict(user_attrs_dict)
                    if isinstance(base_cfg.attrs, RigidBodyPhysicsCfg):
                        for field_name in grouped_fields:
                            override = getattr(parsed, field_name)
                            if override is None:
                                continue
                            base = getattr(base_cfg.attrs, field_name)
                            if base is not None and type(base) is type(override):
                                _merge_non_none_config(base, override)
                            else:
                                setattr(base_cfg.attrs, field_name, override)
                    else:
                        base_cfg.attrs = parsed
                    continue
                if "newton" in user_attrs_dict:
                    raise ValueError(
                        "Deprecated flat attrs are Default-backend-only and no "
                        "longer accept attrs.newton. Use grouped "
                        "RigidBodyPhysicsCfg properties for Newton."
                    )
                if user_attrs_dict and isinstance(base_cfg.attrs, RigidBodyPhysicsCfg):
                    base_cfg.attrs = RigidBodyAttributesCfg.from_grouped(base_cfg.attrs)
                for attr_key, attr_val in user_attrs_dict.items():
                    if hasattr(base_cfg.attrs, attr_key):
                        setattr(base_cfg.attrs, attr_key, attr_val)
                    else:
                        logger.log_warning(
                            f"Key '{attr_key}' not found in " "RigidBodyAttributesCfg."
                        )
            else:
                logger.log_warning(
                    "attrs should be a dictionary. Skipping attrs merge."
                )
        elif key == "control_parts":
            # merge control parts
            user_control_parts_dict = override_cfg_dict.get("control_parts")
            if isinstance(user_control_parts_dict, dict):
                # Initialize control_parts if it is None to avoid TypeError on item assignment
                if base_cfg.control_parts is None:
                    base_cfg.control_parts = {}
                for part_key, part_val in user_control_parts_dict.items():
                    base_cfg.control_parts[part_key] = part_val
            else:
                logger.log_warning(
                    "control_parts should be a dictionary. Skipping control_parts merge."
                )
        elif key == "urdf_cfg":
            if base_cfg.urdf_cfg is None:
                logger.log_warning(
                    f"There is no defined urdf_cfg in base robot cfg. Skipping urdf_cfg merge."
                )
                continue

            # merge urdf components
            user_urdf_cfg = override_cfg_dict.get("urdf_cfg")
            if isinstance(user_urdf_cfg, dict):
                # Merge name_case policy if the override specifies one.
                user_name_case = user_urdf_cfg.get("name_case")
                if isinstance(user_name_case, dict):
                    if base_cfg.urdf_cfg.name_case is None:
                        base_cfg.urdf_cfg.name_case = dict(user_name_case)
                    else:
                        base_cfg.urdf_cfg.name_case.update(user_name_case)

                components = user_urdf_cfg.get("components", [])
                # to_dict serializes components as a dict keyed by type;
                # normalize to a list of dicts for merge_robot_cfg.
                if isinstance(components, dict):
                    components = [
                        {"component_type": k, **v} for k, v in components.items()
                    ]
                for component in components:
                    base_cfg.urdf_cfg.add_component(
                        component_type=component.get("component_type"),
                        urdf_path=component.get("urdf_path"),
                        transform=component.get("transform"),
                    )
            else:
                logger.log_warning(
                    "urdf_cfg should be a dictionary. Skipping urdf_cfg merge."
                )
        else:
            # Only apply keys the base RobotCfg.from_dict recognized.
            # Subclass-only variant fields (e.g. version) are not
            # present on a plain RobotCfg and are already set by _build_defaults;
            # skip them instead of raising AttributeError.
            if hasattr(robot_cfg, key):
                setattr(base_cfg, key, getattr(robot_cfg, key))

    return base_cfg
