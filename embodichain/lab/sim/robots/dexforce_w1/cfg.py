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

from __future__ import annotations

import enum
import json
import numpy as np
import typing
import torch

from typing import Dict

from embodichain.lab.sim.robots.dexforce_w1.types import (
    DexforceW1HandBrand,
    DexforceW1ArmSide,
    DexforceW1ArmKind,
    DexforceW1Version,
)
from embodichain.lab.sim.robots.dexforce_w1.utils import (
    build_dexforce_w1_cfg,
)
from embodichain.lab.sim.solvers import SolverCfg
from embodichain.lab.sim.cfg import (
    RobotCfg,
    JointDrivePropertiesCfg,
    RigidBodyAttributesCfg,
)
from embodichain.data import get_data_path
from embodichain.utils import configclass, logger


@configclass
class DexforceW1Cfg(RobotCfg):
    """DexforceW1 specific configuration, inherits from RobotCfg and allows custom parameters."""

    version: DexforceW1Version = DexforceW1Version.V021
    arm_kind: DexforceW1ArmKind = DexforceW1ArmKind.INDUSTRIAL

    @classmethod
    def from_dict(
        cls, init_dict: typing.Dict[str, typing.Union[str, float, tuple]]
    ) -> DexforceW1Cfg:
        """Initialize DexforceW1Cfg from a dictionary.

        Args:
            init_dict (Dict[str, Union[str, float, tuple]]): Dictionary of configuration parameters.

        Returns:
            DexforceW1Cfg: An instance of DexforceW1Cfg with parameters set.
        """
        from embodichain.lab.sim.solvers import merge_solver_cfg

        init_dict_m = init_dict.copy()
        version = init_dict_m.get("version", "v021")
        arm_kind = init_dict_m.get("arm_kind", "anthropomorphic")
        init_dict_m.pop("version", None)
        init_dict_m.pop("arm_kind", None)
        cfg: DexforceW1Cfg = cls()._build_default_cfg(
            version=version, arm_kind=arm_kind
        )

        default_physics_cfgs = cls()._build_default_physics_cfgs(arm_kind=arm_kind)
        for key, value in default_physics_cfgs.items():
            setattr(cfg, key, value)

        default_solver_cfg = cls()._build_default_solver_cfg(
            is_industrial=(arm_kind == "industrial")
        )
        cfg.solver_cfg = default_solver_cfg

        # override default values with those provided in init_dict.
        robot_cfg = RobotCfg.from_dict(init_dict_m)

        # set attrs into cfg from the robot_cfg, but merge solver_cfg specially
        for key, value in init_dict_m.items():
            if key == "solver_cfg":
                # merge provided solver_cfg values into default solver config
                provided_solver_cfg = init_dict_m.get("solver_cfg")
                if provided_solver_cfg:
                    for part, item in provided_solver_cfg.items():
                        if "class_type" in provided_solver_cfg[part]:
                            cfg.solver_cfg[part] = robot_cfg.solver_cfg[part]
                        else:
                            try:
                                merged = merge_solver_cfg(
                                    cfg.solver_cfg, provided_solver_cfg
                                )
                                cfg.solver_cfg = merged
                            except Exception:
                                logger.log_error(
                                    f"Failed to merge solver_cfg, using provided config outright."
                                )
            else:
                setattr(cfg, key, getattr(robot_cfg, key))

        return cfg

    @staticmethod
    def _build_default_solver_cfg(is_industrial: bool) -> SolverCfg:
        # TODO: maybe change default solver for DexforceW1
        from embodichain.lab.sim.solvers import PytorchSolverCfg
        from embodichain.lab.sim.solvers import SRSSolverCfg
        from embodichain.lab.sim.robots.dexforce_w1.params import (
            W1ArmKineParams,
        )

        if is_industrial:
            w1_left_arm_params = W1ArmKineParams(
                arm_side=DexforceW1ArmSide.LEFT,
                arm_kind=DexforceW1ArmKind.INDUSTRIAL,
                version=DexforceW1Version.V021,
            )
            w1_right_arm_params = W1ArmKineParams(
                arm_side=DexforceW1ArmSide.RIGHT,
                arm_kind=DexforceW1ArmKind.INDUSTRIAL,
                version=DexforceW1Version.V021,
            )
            left_arm_tcp = np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.15],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            right_arm_tcp = np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.15],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        else:
            w1_left_arm_params = W1ArmKineParams(
                arm_side=DexforceW1ArmSide.LEFT,
                arm_kind=DexforceW1ArmKind.ANTHROPOMORPHIC,
                version=DexforceW1Version.V021,
            )
            w1_right_arm_params = W1ArmKineParams(
                arm_side=DexforceW1ArmSide.RIGHT,
                arm_kind=DexforceW1ArmKind.ANTHROPOMORPHIC,
                version=DexforceW1Version.V021,
            )
            left_arm_tcp = np.array(
                [
                    [-1.0, 0.0, 0.0, 0.012],
                    [0.0, 0.0, 1.0, 0.0675],
                    [0.0, 1.0, 0.0, 0.127],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            right_arm_tcp = np.array(
                [
                    [1.0, 0.0, 0.0, 0.012],
                    [0.0, 0.0, -1.0, -0.0675],
                    [0.0, 1.0, 0.0, 0.127],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )

        return {
            "right_arm": SRSSolverCfg(
                end_link_name="right_ee",
                root_link_name="right_arm_base",
                dh_params=w1_right_arm_params.dh_params,
                qpos_limits=w1_right_arm_params.qpos_limits,
                T_e_oe=w1_right_arm_params.T_e_oe,
                T_b_ob=w1_right_arm_params.T_b_ob,
                link_lengths=w1_right_arm_params.link_lengths,
                rotation_directions=w1_right_arm_params.rotation_directions,
                tcp=right_arm_tcp,
            ),
            "left_arm": SRSSolverCfg(
                end_link_name="left_ee",
                root_link_name="left_arm_base",
                dh_params=w1_left_arm_params.dh_params,
                qpos_limits=w1_left_arm_params.qpos_limits,
                T_e_oe=w1_left_arm_params.T_e_oe,
                T_b_ob=w1_left_arm_params.T_b_ob,
                link_lengths=w1_left_arm_params.link_lengths,
                rotation_directions=w1_left_arm_params.rotation_directions,
                tcp=left_arm_tcp,
            ),
        }

    @staticmethod
    def _build_default_physics_cfgs(arm_kind: str) -> typing.Dict[str, typing.Any]:
        """Build default physics configurations for DexforceW1.

        Args:
            arm_kind: Type of arm, either "industrial" or "anthropomorphic"

        Returns:
            Dictionary containing physics configuration parameters
        """
        # Define common joint patterns
        arm_joints = "(RIGHT|LEFT)_J[0-9]"
        body_joints = "(ANKLE|KNEE|BUTTOCK|WAIST)"

        # Hand/gripper pattern differs by arm type
        hand_pattern = (
            "(LEFT|RIGHT)_FINGER[1-2]"
            if arm_kind == "industrial"
            else "(RIGHT|LEFT)_[A-Z|_]+"
        )

        # Define physics parameters for different joint types
        joint_params = {
            "stiffness": {
                arm_joints: 1e4,
                hand_pattern: 1e2,
                body_joints: 1e7,
            },
            "damping": {
                arm_joints: 1e3,
                hand_pattern: 1e1,
                body_joints: 1e4,
            },
            "max_effort": {
                arm_joints: 1e5,
                hand_pattern: 1e3,
                body_joints: 1e10,
            },
        }

        drive_pros = JointDrivePropertiesCfg(**joint_params)

        return {
            "min_position_iters": 32,
            "min_velocity_iters": 8,
            "drive_pros": drive_pros,
            # TODO: we may use the some properties from URDF as default values
            # eg. mass, friction, damping, etc.
            "attrs": RigidBodyAttributesCfg(
                mass=1.0,
                static_friction=0.95,
                dynamic_friction=0.9,
                linear_damping=0.7,
                angular_damping=0.7,
                contact_offset=0.005,
                rest_offset=0.001,
                restitution=0.05,
                max_depenetration_velocity=10.0,
            ),
        }

    @staticmethod
    def _build_default_cfg(
        version: str = "v021", arm_kind: str = "anthropomorphic"
    ) -> DexforceW1Cfg:

        if arm_kind == "industrial":
            hand_types = {
                DexforceW1ArmSide.LEFT: DexforceW1HandBrand.DH_PGC_GRIPPER_M,
                DexforceW1ArmSide.RIGHT: DexforceW1HandBrand.DH_PGC_GRIPPER_M,
            }
        else:
            hand_types = {
                DexforceW1ArmSide.LEFT: DexforceW1HandBrand.BRAINCO_HAND,
                DexforceW1ArmSide.RIGHT: DexforceW1HandBrand.BRAINCO_HAND,
            }

        hand_versions = {
            DexforceW1ArmSide.LEFT: DexforceW1Version(version),
            DexforceW1ArmSide.RIGHT: DexforceW1Version(version),
        }

        cfg = build_dexforce_w1_cfg(
            arm_kind=DexforceW1ArmKind(arm_kind),
            hand_types=hand_types,
            hand_versions=hand_versions,
        )
        cfg.version = DexforceW1Version(version)
        cfg.arm_kind = DexforceW1ArmKind(arm_kind)

        return cfg

    def to_dict(self):
        """Convert config to a Python dict, handling enums and numpy arrays."""

        def serialize(obj, _visited=None):
            if _visited is None:
                _visited = set()
            # Only skip recursion for mutable objects (dict, custom class)
            if isinstance(obj, (dict, object)) and not isinstance(
                obj, (str, int, float, bool, type(None))
            ):
                obj_id = id(obj)
                if obj_id in _visited:
                    return None  # Prevent infinite recursion
                _visited.add(obj_id)

            if isinstance(obj, enum.Enum):
                return obj.value
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                # Only serialize values, keep keys as str/int/float/bool/None
                return {str(k): serialize(v, _visited) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [serialize(v, _visited) for v in obj]
            if hasattr(obj, "to_dict") and obj is not self:
                return serialize(obj.to_dict(), _visited)
            if hasattr(obj, "__dict__"):
                return {k: serialize(v, _visited) for k, v in obj.__dict__.items()}
            return obj

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
        from embodichain.lab.sim.utility.solver_utils import (
            create_pk_chain,
            create_pk_serial_chain,
        )

        if DexforceW1ArmKind.INDUSTRIAL == self.arm_kind:
            urdf_path = get_data_path("DexforceW1V021/DexforceW1_v02_2.urdf")
        elif DexforceW1ArmKind.ANTHROPOMORPHIC == self.arm_kind:
            urdf_path = get_data_path("DexforceW1V021/DexforceW1_v02_1.urdf")

        chain = create_pk_chain(urdf_path, device)

        left_arm_chain = create_pk_serial_chain(
            chain=chain, end_link_name="left_ee", root_link_name="left_arm_base"
        ).to(device=device)
        right_arm_chain = create_pk_serial_chain(
            chain=chain, end_link_name="right_ee", root_link_name="right_arm_base"
        ).to(device=device)

        return {
            "left_arm": left_arm_chain,
            "right_arm": right_arm_chain,
        }


if __name__ == "__main__":
    # Example usage
    import numpy as np

    np.set_printoptions(precision=5, suppress=True)
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.robots.dexforce_w1.types import (
        DexforceW1ArmKind,
    )

    config = SimulationManagerCfg(headless=True, sim_device="cpu")
    sim = SimulationManager(config)

    cfg = DexforceW1Cfg.from_dict(
        {
            "uid": "dexforce_w1",
            "version": "v021",
            "arm_kind": "anthropomorphic",
        }
    )

    robot = sim.add_robot(cfg=cfg)
    sim.update(step=1)
    print("DexforceW1 robot added to the simulation.")

    from IPython import embed

    embed()
