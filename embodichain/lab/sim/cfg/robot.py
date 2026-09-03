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

"""Robot configuration, serialization, and backend preset selection."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import MISSING, fields
import enum
import json
from typing import Dict, List

import numpy as np
import torch

from embodichain.utils import configclass, is_configclass, logger
from embodichain.utils.utility import key_in_nested_dict

from ..workspace.cfg import RobotWorkspaceCfg
from .articulation import (
    ArticulationCfg,
    JointDrivePropertiesCfg,
    _raise_removed_articulation_cfg_fields,
    link_attrs_from_dict,
)
from .asset import AssetPhysicsMode
from .rigid import _rigid_body_physics_from_dict
from .simulation import (
    PhysicsBackendCfg,
    _normalize_newton_solver_type,
    physics_backend_from_cfg,
)
from .urdf import URDFCfg


def _get_data_path(path: str) -> str:
    """Resolve data through the public facade for monkeypatch compatibility."""
    from . import get_data_path

    return get_data_path(path)


@configclass
class RobotCfg(ArticulationCfg):
    from embodichain.lab.sim.solvers import SolverCfg

    """Configuration for a robot asset in the simulation.
    """

    joint_drive_props: JointDrivePropertiesCfg = JointDrivePropertiesCfg(
        drive_type="force",
        stiffness=1e4,
        damping=1e3,
        max_effort=1e10,
        max_velocity=1e10,
        friction=0.0,
        armature=0.0,
    )
    """Joint drive, limit, friction, and armature properties."""

    asset_physics_mode: AssetPhysicsMode = "overlay"
    """Apply configured robot physics on top of source-authored values."""

    control_parts: Dict[str, List[str]] | None = None
    """Control parts is the mapping from part name to joint names.

    For example, {'left_arm': ['joint1', 'joint2'], 'right_arm': ['joint3', 'joint4']}
    If no control part is specified, the robot will use all joints as a single control part.

    Note: 
        - if `control_parts` is specified, `solver_cfg` must be a dict with part names as
            keys corresponding to the control parts name.
        - The joint names in the control parts support regular expressions, e.g., 'joint[1-6]'.
            After initialization of robot, the names will be expanded to a list of full joint names.
        - `Robot` is a derived class of `Articulation`, with control parts support. So the `joint_drive_props`
            in `ArticulationCfg` can use control part as key to specify the corresponding joint drive properties, 
            which will be overridden if these joint names are already specified.
    """

    urdf_cfg: URDFCfg | None = None
    """URDF assembly configuration which allows for assembling a robot from multiple URDF components.
    """

    # TODO: how to support one solver for multiple parts?
    solver_cfg: SolverCfg | Dict[str, SolverCfg] | None = None
    """Solver is used to compute forward and inverse kinematics for the robot.
    """

    workspace_cfg: Dict[str, RobotWorkspaceCfg] | None = None
    """Runtime workspace cache configuration keyed by control-part name."""

    @classmethod
    def from_dict(cls, init_dict: Dict[str, str | float | tuple]) -> RobotCfg:
        """Initialize the configuration from a dictionary."""
        if isinstance(init_dict, cls):
            return init_dict

        _raise_removed_articulation_cfg_fields(init_dict)

        import importlib

        solver_module = importlib.import_module("embodichain.lab.sim.solvers")

        cfg = cls()  # Create a new instance of the class (cls)
        for key, value in init_dict.items():
            if key == "link_attrs" and isinstance(value, dict):
                cfg.link_attrs = link_attrs_from_dict(value)
            elif key == "attrs" and isinstance(value, Mapping):
                cfg.attrs = _rigid_body_physics_from_dict(value)
            elif hasattr(cfg, key):
                attr = getattr(cfg, key)
                if key == "urdf_cfg":
                    from embodichain.lab.sim.cfg import URDFCfg

                    setattr(cfg, key, URDFCfg.from_dict(value))
                elif key == "workspace_cfg" and isinstance(value, dict):
                    setattr(
                        cfg,
                        key,
                        {
                            part: (
                                part_cfg
                                if isinstance(part_cfg, RobotWorkspaceCfg)
                                else RobotWorkspaceCfg(**part_cfg)
                            )
                            for part, part_cfg in value.items()
                        },
                    )
                elif key == "fpath":
                    setattr(cfg, key, _get_data_path(value))
                elif isinstance(attr, JointDrivePropertiesCfg) and isinstance(
                    value, dict
                ):
                    setattr(
                        cfg,
                        key,
                        JointDrivePropertiesCfg.from_dict(value, defaults=attr),
                    )
                elif is_configclass(attr):
                    setattr(
                        cfg, key, attr.from_dict(value)
                    )  # Call from_dict on the attribute
                elif isinstance(value, dict) and "class_type" in value:
                    setattr(
                        cfg,
                        key,
                        getattr(solver_module, f"{value['class_type']}Cfg").from_dict(
                            value
                        ),
                    )
                elif isinstance(value, dict) and key_in_nested_dict(
                    value, "class_type"
                ):
                    setattr(
                        cfg,
                        key,
                        {
                            k: getattr(
                                solver_module, f"{v['class_type']}Cfg"
                            ).from_dict(v)
                            for k, v in value.items()
                        },
                    )

                else:
                    setattr(cfg, key, value)
            else:
                logger.log_warning(
                    f"Key '{key}' not found in {cfg.__class__.__name__}."
                )
        return cfg

    def _build_defaults(self, init_dict: dict | None = None) -> None:
        """Populate default config fields from ``init_dict``.

        Subclasses override this to read variant/version fields from
        ``init_dict``, set them on ``self``, and populate ``urdf_cfg``,
        ``control_parts``, ``solver_cfg``, ``joint_drive_props`` and ``attrs``.
        The base implementation is a no-op.

        .. attention::
            Do NOT call :func:`merge_robot_cfg` from here -- the subclass
            ``from_dict`` calls this hook first, then ``merge_robot_cfg``.
            Calling ``merge_robot_cfg`` here would recurse, because
            ``merge_robot_cfg`` itself calls ``RobotCfg.from_dict``.

        Args:
            init_dict: The raw override dict passed to ``from_dict``.
        """
        return None

    def to_dict(self):
        """Serialize config to a plain dict (enums, numpy, nested configclass)."""

        def serialize(obj, _visited=None):
            if _visited is None:
                _visited = set()
            if isinstance(obj, enum.Enum):
                return obj.value
            tracked_id = None
            if not isinstance(obj, (str, int, float, bool, type(None))):
                tracked_id = id(obj)
                if tracked_id in _visited:
                    return None
                _visited.add(tracked_id)

            try:
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, dict):
                    return {
                        (k.value if isinstance(k, enum.Enum) else str(k)): serialize(
                            v, _visited
                        )
                        for k, v in obj.items()
                    }
                if isinstance(obj, (list, tuple)):
                    return [serialize(v, _visited) for v in obj]
                if hasattr(obj, "to_dict") and obj is not self:
                    return serialize(obj.to_dict(), _visited)
                if hasattr(obj, "__dict__"):
                    return {
                        k: serialize(v, _visited)
                        for k, v in obj.__dict__.items()
                        if v is not None
                    }
                return obj
            finally:
                if tracked_id is not None:
                    _visited.remove(tracked_id)

        return serialize(self)

    def to_string(self):
        """Return config as a JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def save_to_file(self, filepath):
        """Save config to a local file as JSON."""
        with open(filepath, "w") as f:
            f.write(self.to_string())

    def build_pk_serial_chain(
        self, device: torch.device = torch.device("cpu"), **kwargs
    ) -> Dict[str, "pk.SerialChain"]:
        """Build the serial chain from the URDF file.

        Note:
            This method is usually used in imitation dataset saving (compute eef pose from qpos using FK)
            and model training (provide a differentiable FK layer or loss computation).

        Args:
            device (torch.device): The device to which the chain will be moved. Defaults to CPU.
            **kwargs: Additional arguments for building the serial chain.

        Returns:
            Dict[str, pk.SerialChain]: The serial chain of the robot for specified control part.
        """
        return {}


@configclass
class RobotPresetCfg:
    """Base class for replace-only robot configurations across physics backends.

    Subclasses declare complete :class:`RobotCfg` alternatives as fields.  A
    ``default`` field is required; optional fields use Newton backend or solver
    profile names such as ``newton``, ``newton_mujoco_warp``, or
    ``newton_mjwarp``. The active :class:`PhysicsBackendCfg` selects one
    complete alternative at
    :meth:`SimulationManager.add_robot`; alternatives are never field-merged.

    Portable robot properties should remain on one ordinary :class:`RobotCfg`.
    Use this wrapper only when an asset, actuator model, or native physics value
    genuinely requires a different complete robot definition.

    Example::

        @configclass
        class MyRobotPresetCfg(RobotPresetCfg):
            default: RobotCfg = MyRobotCfg()
            newton_mujoco_warp: RobotCfg = MyNewtonRobotCfg()
    """

    def resolve(
        self,
        physics_cfg: PhysicsBackendCfg,
        *,
        newton_solver_type: str | None = None,
    ) -> RobotCfg:
        """Return an isolated complete robot config for the active backend.

        Args:
            physics_cfg: The scene's backend-selecting physics configuration.
            newton_solver_type: Resolved Newton solver name when it is already
                available from the runtime. If omitted, it is inferred from
                ``physics_cfg``.

        Returns:
            A deep copy of the highest-priority complete robot alternative.

        Raises:
            TypeError: If a preset name is unsupported, ``default`` is
                undeclared, or a selected alternative is not a
                :class:`RobotCfg`.
            ValueError: If no declared alternative can satisfy the backend.
        """
        options = {item.name: getattr(self, item.name) for item in fields(self)}
        invalid_names = {
            name
            for name in options
            if name != "default" and name != "newton" and not name.startswith("newton_")
        }
        if invalid_names:
            raise TypeError(
                f"{type(self).__name__} uses unsupported preset name(s) "
                f"{sorted(invalid_names)}; use 'default' or 'newton[_<solver>]'."
            )
        if "default" not in options:
            raise TypeError(
                f"{type(self).__name__} must declare a 'default' RobotCfg preset."
            )

        backend = physics_backend_from_cfg(physics_cfg)
        if backend == "default":
            candidates = ("default",)
        else:
            solver_type = newton_solver_type
            if solver_type is None:
                solver_cfg = physics_cfg.solver_cfg
                if solver_cfg is None:
                    solver_type = "auto"
                elif isinstance(solver_cfg, Mapping):
                    solver_type = str(
                        solver_cfg.get("solver_type")
                        or solver_cfg.get("class_type")
                        or "auto"
                    )
                else:
                    solver_type = str(getattr(solver_cfg, "solver_type"))
            solver_type = _normalize_newton_solver_type(solver_type)
            solver_candidates = []
            if solver_type != "auto":
                solver_candidates.append(f"newton_{solver_type}")
                if solver_type == "mujoco_warp":
                    solver_candidates.append("newton_mjwarp")
            candidates = (*solver_candidates, "newton", "default")

        for candidate in candidates:
            selected = options.get(candidate)
            if selected is None or selected is MISSING:
                continue
            if not isinstance(selected, RobotCfg):
                raise TypeError(
                    f"{type(self).__name__}.{candidate} must be a RobotCfg, "
                    f"got {type(selected).__name__}."
                )
            return deepcopy(selected)

        raise ValueError(
            f"{type(self).__name__} has no usable preset for {candidates!r}; "
            f"declared options are {sorted(options)}."
        )
