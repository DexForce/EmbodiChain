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

from __future__ import annotations

import typing

import numpy as np
import torch

if __name__ == "__main__" and not __package__:
    # Support running this example by file path from an uninstalled source tree.
    import sys
    from pathlib import Path

    # Replace the script directory so its ``types.py`` cannot shadow the
    # standard-library ``types`` module in compiler subprocesses.
    sys.path[0] = str(Path(__file__).resolve().parents[5])

from typing import TYPE_CHECKING, Dict

from embodichain.lab.sim.robots.dexforce_w1.types import (
    DexforceW1ArmSide,
    DexforceW1HandBrand,
    DexforceW1HandVersion,
    DexforceW1Version,
)
from embodichain.lab.sim.robots.dexforce_w1.utils import (
    build_dexforce_w1_assembly_urdf_cfg,
    build_dexforce_w1_control_parts,
)
from embodichain.lab.sim.robots.dexforce_w1.hand_specs import (
    get_default_w1_hand_version,
    normalize_w1_hand_mappings,
)
from embodichain.lab.sim.robots.dexforce_w1.specs import get_w1_version_spec
from embodichain.lab.sim.cfg import (
    ArticulationRootPropertiesCfg,
    CollisionPropertiesCfg,
    RobotCfg,
    JointDrivePropertiesCfg,
    RigidBodyMaterialCfg,
    RigidBodyPhysicsCfg,
)
from embodichain.lab.sim.utility.cfg_utils import merge_robot_cfg
from embodichain.utils import configclass

if TYPE_CHECKING:
    import pytorch_kinematics as pk


@configclass
class DexforceW1Cfg(RobotCfg):
    """DexforceW1 specific configuration, inherits from RobotCfg and allows custom parameters."""

    version: DexforceW1Version = DexforceW1Version.V021
    with_default_eef: bool = True
    hand_types: dict | None = None
    hand_versions: dict[DexforceW1ArmSide, DexforceW1HandVersion] | None = None
    hand_attach_xposes: dict | None = None

    @classmethod
    def from_dict(
        cls, init_dict: Dict[str, str | float | tuple | dict]
    ) -> DexforceW1Cfg:
        """Initialize DexforceW1Cfg from a dictionary.

        Args:
            init_dict: Dictionary of configuration parameters.

        Returns:
            A DexforceW1Cfg instance. Defaults are built via
            :meth:`_build_defaults`, then ``init_dict`` overrides are merged.
        """
        unknown_fields = set(init_dict) - set(cls.__dataclass_fields__)
        if unknown_fields:
            unknown = ", ".join(sorted(unknown_fields))
            raise ValueError(f"Unknown DexforceW1 configuration fields: {unknown}")

        cfg = cls()
        cfg._build_defaults(init_dict)
        cfg = merge_robot_cfg(cfg, init_dict)
        cfg._compose_user_eef_overrides(init_dict)
        return cfg

    def _compose_user_eef_overrides(self, init_dict: dict) -> None:
        """Apply the version offset to user-supplied EEF transforms and TCPs."""
        side_by_component = {
            "left_hand": DexforceW1ArmSide.LEFT,
            "right_hand": DexforceW1ArmSide.RIGHT,
        }
        urdf_cfg = init_dict.get("urdf_cfg")
        components = (
            urdf_cfg.get("components", []) if isinstance(urdf_cfg, dict) else []
        )
        if isinstance(components, dict):
            component_names = set(components)
        else:
            component_names = {
                component.get("component_type")
                for component in components
                if isinstance(component, dict)
            }

        for component_name, arm_side in side_by_component.items():
            if component_name not in component_names:
                continue
            component = self.urdf_cfg.components.get(component_name)
            if component is None:
                continue
            spec = get_w1_version_spec(self.version)
            component["transform"] = spec.compose_eef_attach_xpos(
                arm_side, component["transform"]
            )

        solver_cfg = init_dict.get("solver_cfg")
        if not isinstance(solver_cfg, dict):
            return
        for arm_side in DexforceW1ArmSide:
            part_name = f"{arm_side.value}_arm"
            part_override = solver_cfg.get(part_name)
            if not isinstance(part_override, dict) or "tcp" not in part_override:
                continue
            solver = self.solver_cfg.get(part_name)
            if solver is None:
                continue
            spec = get_w1_version_spec(self.version)
            solver.tcp = spec.compose_eef_attach_xpos(arm_side, solver.tcp)

    def to_dict(self):
        """Serialize EEF-specific transforms without the derived version offset."""
        data = super().to_dict()
        side_by_component = {
            "left_hand": DexforceW1ArmSide.LEFT,
            "right_hand": DexforceW1ArmSide.RIGHT,
        }
        components = data.get("urdf_cfg", {}).get("components", {})
        for component_name, arm_side in side_by_component.items():
            component = components.get(component_name)
            if not isinstance(component, dict) or component.get("transform") is None:
                continue
            spec = get_w1_version_spec(self.version)
            offset_inv = np.linalg.inv(spec.eef_attach_xpos(arm_side))
            component["transform"] = (
                offset_inv @ np.asarray(component["transform"], dtype=float)
            ).tolist()

        solver_cfg = data.get("solver_cfg", {})
        for arm_side in DexforceW1ArmSide:
            part_name = f"{arm_side.value}_arm"
            solver = solver_cfg.get(part_name)
            if not isinstance(solver, dict) or solver.get("tcp") is None:
                continue
            spec = get_w1_version_spec(self.version)
            offset_inv = np.linalg.inv(spec.eef_attach_xpos(arm_side))
            solver["tcp"] = (
                offset_inv @ np.asarray(solver["tcp"], dtype=float)
            ).tolist()
        return data

    def _build_defaults(self, init_dict: dict | None = None) -> None:
        """Build default urdf/control/solver/physics from variant fields.

        Reads ``version``/``with_default_eef`` from ``init_dict``,
        sets them on ``self``, then populates ``urdf_cfg``, ``control_parts``,
        ``solver_cfg``, ``drive_pros`` and ``attrs``.
        """
        init_dict = init_dict or {}
        self.version = DexforceW1Version.parse(
            init_dict.get("version", DexforceW1Version.V021)
        )
        self.with_default_eef = bool(init_dict.get("with_default_eef", True))

        (
            self.hand_types,
            configured_hand_versions,
            self.hand_attach_xposes,
        ) = normalize_w1_hand_mappings(
            hand_types=init_dict.get("hand_types"),
            hand_versions=init_dict.get("hand_versions"),
            hand_attach_xposes=init_dict.get("hand_attach_xposes"),
        )
        self.hand_versions = {
            side: configured_hand_versions.get(
                side,
                get_default_w1_hand_version(
                    self.hand_types.get(side, DexforceW1HandBrand.BRAINCO_HAND)
                ),
            )
            for side in DexforceW1ArmSide
        }

        self.urdf_cfg = build_dexforce_w1_assembly_urdf_cfg(
            version=self.version,
            hand_types=self.hand_types,
            hand_versions=self.hand_versions,
            hand_attach_xposes=self.hand_attach_xposes,
            include_hand=self.with_default_eef,
        )
        self.control_parts = build_dexforce_w1_control_parts(
            version=self.version,
            hand_types=self.hand_types,
            hand_versions=self.hand_versions,
            include_hand=self.with_default_eef,
        )

        # physics
        physics = self._build_default_physics_cfgs(
            with_default_eef=self.with_default_eef
        )
        for key, value in physics.items():
            setattr(self, key, value)

        # solver (set exactly once -- was previously double-set)
        self.solver_cfg = self._build_default_solver_cfg()

    def _build_default_solver_cfg(self):
        """Build the version-matched default SRS solver configuration."""
        from embodichain.lab.sim.solvers import SRSSolverCfg
        from embodichain.lab.sim.robots.dexforce_w1.params import (
            W1ArmKineParams,
        )

        solver_cfg = {}
        for arm_side in DexforceW1ArmSide:
            params = W1ArmKineParams(arm_side=arm_side, version=self.version)
            part_name = f"{arm_side.value}_arm"
            solver_cfg[part_name] = SRSSolverCfg(
                end_link_name=f"{arm_side.value}_ee",
                root_link_name=f"{arm_side.value}_arm_base",
                dh_params=params.dh_params,
                user_qpos_limits=params.qpos_limits,
                T_e_oe=params.T_e_oe,
                T_b_ob=params.T_b_ob,
                link_lengths=params.link_lengths,
                rotation_directions=params.rotation_directions,
                tcp=get_w1_version_spec(self.version).tcp(arm_side),
            )
        return solver_cfg

    def _build_default_physics_cfgs(
        self, with_default_eef: bool = True
    ) -> typing.Dict[str, typing.Any]:
        """Build default physics configurations for DexforceW1.

        Args:
            with_default_eef: Whether to include default end-effector configurations

        Returns:
            Dictionary containing physics configuration parameters
        """
        DEFAULT_EEF_JOINT_DRIVE_PARAMS = {
            "stiffness": 1e2,
            "damping": 1e1,
            "max_effort": 1e3,
        }

        DEFAULT_EEF_HAND_JOINT_NAMES = (
            "(LEFT|RIGHT)_HAND_(THUMB[12]|INDEX|MIDDLE|RING|PINKY)"
        )
        ARM_JOINTS = "(RIGHT|LEFT)_J[0-9]"
        BODY_JOINTS = "(ANKLE|KNEE|BUTTOCK|WAIST)"
        HEAD_JOINTS = "(NECK1|NECK2)"

        joint_params = {
            "stiffness": {
                ARM_JOINTS: 1e4,
                BODY_JOINTS: 1e7,
                HEAD_JOINTS: 1e4,
            },
            "damping": {ARM_JOINTS: 1e3, BODY_JOINTS: 1e4, HEAD_JOINTS: 1e3},
            "max_effort": {ARM_JOINTS: 1e5, BODY_JOINTS: 1e10, HEAD_JOINTS: 1e5},
        }
        drive_pros = JointDrivePropertiesCfg(
            drive_type="force",
            **joint_params,
        )

        if with_default_eef:
            eef_joint_names = DEFAULT_EEF_HAND_JOINT_NAMES
            drive_pros.stiffness.update(
                {eef_joint_names: DEFAULT_EEF_JOINT_DRIVE_PARAMS["stiffness"]}
            )
            drive_pros.damping.update(
                {eef_joint_names: DEFAULT_EEF_JOINT_DRIVE_PARAMS["damping"]}
            )
            drive_pros.max_effort.update(
                {eef_joint_names: DEFAULT_EEF_JOINT_DRIVE_PARAMS["max_effort"]}
            )

        return {
            "drive_pros": drive_pros,
            "articulation_props": ArticulationRootPropertiesCfg(
                min_position_iters=32,
                min_velocity_iters=8,
            ),
            "attrs": RigidBodyPhysicsCfg(
                collision_props=CollisionPropertiesCfg(
                    contact_offset=0.001,
                    rest_offset=0.0,
                ),
                material_props=RigidBodyMaterialCfg(
                    static_friction=0.95,
                    dynamic_friction=0.9,
                ),
            ),
        }

    # to_dict, to_string, save_to_file inherited from RobotCfg

    def _pk_urdf_path(self, arm_side: DexforceW1ArmSide) -> str:
        """Return the selected arm component URDF for FK/IK.

        .. attention::
            The root_link->end_link kinematics here must match the arms in the
            simulation (assembled) URDF. A DOF drift guard in the tests checks this.
        """
        from embodichain.lab.sim.robots.dexforce_w1.utils import arm_manager

        return arm_manager.get_urdf(side=arm_side, version=self.version)

    def build_pk_serial_chain(
        self, device: torch.device = torch.device("cpu"), **kwargs
    ) -> Dict[str, "pk.SerialChain"]:
        from embodichain.lab.sim.utility.solver_utils import (
            create_pk_serial_chain,
        )

        return {
            f"{arm_side.value}_arm": create_pk_serial_chain(
                urdf_path=self._pk_urdf_path(arm_side),
                device=device,
                end_link_name=f"{arm_side.value}_ee",
                root_link_name=f"{arm_side.value}_arm_base",
            )
            for arm_side in DexforceW1ArmSide
        }


if __name__ == "__main__":
    import argparse

    np.set_printoptions(precision=5, suppress=True)
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import physics_cfg_for_backend

    parser = argparse.ArgumentParser(description="Launch the Dexforce W1 robot")
    parser.add_argument(
        "--physics",
        choices=("default", "newton"),
        default="newton",
        help="Physics backend to launch (default: newton).",
    )
    args = parser.parse_args()

    config = SimulationManagerCfg(
        headless=True,
        device="cpu",
        num_envs=4,
        physics_cfg=physics_cfg_for_backend(args.physics),
    )
    sim = SimulationManager(config)

    cfg = DexforceW1Cfg.from_dict({"uid": "dexforce_w1", "version": "v021"})

    robot = sim.add_robot(cfg=cfg)
    sim.prepare()
    sim.update(step=1)
    print("DexforceW1 robot added to the simulation.", flush=True)
    sim.open_window()
    from IPython import embed

    embed()  # noqa: E702
    sim.destroy()
