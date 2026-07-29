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

from typing import TYPE_CHECKING, Dict

from embodichain.lab.sim.robots.dexforce_w1.types import (
    DexforceW1HandBrand,
    DexforceW1ArmSide,
    DexforceW1Version,
    DexforceW1Type,
)
from embodichain.lab.sim.robots.dexforce_w1.utils import (
    build_dexforce_w1_cfg,
)
from embodichain.lab.sim.robots.dexforce_w1.specs import (
    get_w1_version_spec,
    normalize_component_versions,
)
from embodichain.lab.sim.cfg import (
    RobotCfg,
    JointDrivePropertiesCfg,
    RigidBodyAttributesCfg,
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
    component_versions: dict | None = None

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

    def _arm_version(self, arm_side: DexforceW1ArmSide) -> DexforceW1Version:
        component_type = (
            DexforceW1Type.LEFT_ARM
            if arm_side == DexforceW1ArmSide.LEFT
            else DexforceW1Type.RIGHT_ARM
        )
        return (self.component_versions or {}).get(component_type, self.version)

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
            spec = get_w1_version_spec(self._arm_version(arm_side))
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
            spec = get_w1_version_spec(self._arm_version(arm_side))
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
            spec = get_w1_version_spec(self._arm_version(arm_side))
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
            spec = get_w1_version_spec(self._arm_version(arm_side))
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
        version = init_dict.get("version", DexforceW1Version.V021)
        with_default_eef = init_dict.get("with_default_eef", True)

        self.version = (
            DexforceW1Version(version) if isinstance(version, str) else version
        )
        self.with_default_eef = with_default_eef
        self.component_versions = normalize_component_versions(
            init_dict.get("component_versions")
        )

        # Build the version-matched URDF assembly and control-part definitions.
        hand_types = {
            DexforceW1ArmSide.LEFT: DexforceW1HandBrand.BRAINCO_HAND,
            DexforceW1ArmSide.RIGHT: DexforceW1HandBrand.BRAINCO_HAND,
        }
        hand_versions = {
            DexforceW1ArmSide.LEFT: self.version,
            DexforceW1ArmSide.RIGHT: self.version,
        }
        base_cfg = build_dexforce_w1_cfg(
            version=self.version,
            hand_types=hand_types,
            hand_versions=hand_versions,
            include_hand=with_default_eef,
            component_versions=self.component_versions,
            solver_cfg={},
        )
        self.urdf_cfg = base_cfg.urdf_cfg
        self.control_parts = base_cfg.control_parts

        # physics
        physics = self._build_default_physics_cfgs(with_default_eef=with_default_eef)
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

        left_version = self._arm_version(DexforceW1ArmSide.LEFT)
        right_version = self._arm_version(DexforceW1ArmSide.RIGHT)
        left_version_spec = get_w1_version_spec(left_version)
        right_version_spec = get_w1_version_spec(right_version)
        w1_left_arm_params = W1ArmKineParams(
            arm_side=DexforceW1ArmSide.LEFT,
            version=left_version,
        )
        w1_right_arm_params = W1ArmKineParams(
            arm_side=DexforceW1ArmSide.RIGHT,
            version=right_version,
        )

        left_arm_tcp = left_version_spec.tcp(DexforceW1ArmSide.LEFT)
        right_arm_tcp = right_version_spec.tcp(DexforceW1ArmSide.RIGHT)

        return {
            "right_arm": SRSSolverCfg(
                end_link_name="right_ee",
                root_link_name="right_arm_base",
                dh_params=w1_right_arm_params.dh_params,
                user_qpos_limits=w1_right_arm_params.qpos_limits,
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
                user_qpos_limits=w1_left_arm_params.qpos_limits,
                T_e_oe=w1_left_arm_params.T_e_oe,
                T_b_ob=w1_left_arm_params.T_b_ob,
                link_lengths=w1_left_arm_params.link_lengths,
                rotation_directions=w1_left_arm_params.rotation_directions,
                tcp=left_arm_tcp,
            ),
        }

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
        drive_pros = JointDrivePropertiesCfg(**joint_params)

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
            "min_position_iters": 32,
            "min_velocity_iters": 8,
            "drive_pros": drive_pros,
            "attrs": RigidBodyAttributesCfg(
                static_friction=0.95,
                dynamic_friction=0.9,
                contact_offset=0.001,
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

        component_type = (
            DexforceW1Type.LEFT_ARM
            if arm_side == DexforceW1ArmSide.LEFT
            else DexforceW1Type.RIGHT_ARM
        )
        version = (self.component_versions or {}).get(component_type, self.version)
        return arm_manager.get_urdf(side=arm_side, version=version)

    def build_pk_serial_chain(
        self, device: torch.device = torch.device("cpu"), **kwargs
    ) -> Dict[str, "pk.SerialChain"]:
        from embodichain.lab.sim.utility.solver_utils import (
            create_pk_serial_chain,
        )

        left_arm_chain = create_pk_serial_chain(
            urdf_path=self._pk_urdf_path(DexforceW1ArmSide.LEFT),
            device=device,
            end_link_name="left_ee",
            root_link_name="left_arm_base",
        )
        right_arm_chain = create_pk_serial_chain(
            urdf_path=self._pk_urdf_path(DexforceW1ArmSide.RIGHT),
            device=device,
            end_link_name="right_ee",
            root_link_name="right_arm_base",
        )

        return {
            "left_arm": left_arm_chain,
            "right_arm": right_arm_chain,
        }


if __name__ == "__main__":
    # Example usage
    import numpy as np

    np.set_printoptions(precision=5, suppress=True)
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg

    config = SimulationManagerCfg(headless=True, sim_device="cpu", num_envs=4)
    sim = SimulationManager(config)

    cfg = DexforceW1Cfg.from_dict({"uid": "dexforce_w1", "version": "v021"})

    robot = sim.add_robot(cfg=cfg)
    sim.update(step=1)
    print("DexforceW1 robot added to the simulation.")
