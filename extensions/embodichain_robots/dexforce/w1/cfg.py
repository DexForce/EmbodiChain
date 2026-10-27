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
    sys.path[0] = str(Path(__file__).resolve().parents[3])
    sys.path.insert(1, str(Path(__file__).resolve().parents[4]))

from typing import TYPE_CHECKING, Dict

from embodichain_robots.dexforce.w1.types import (
    DexforceW1ArmSide,
    DexforceW1HandBrand,
    DexforceW1HandVersion,
    DexforceW1Version,
)
from embodichain_robots.dexforce.w1.utils import (
    build_dexforce_w1_assembly_urdf_cfg,
    build_dexforce_w1_control_parts,
)
from embodichain_robots.dexforce.w1.hand_specs import (
    get_default_w1_hand_version,
    get_w1_hand_spec,
    normalize_w1_hand_mappings,
)
from embodichain_robots.dexforce.w1.specs import get_w1_version_spec
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
    from embodichain.lab.sim.sensors import CameraCfg


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
        ``solver_cfg``, ``joint_drive_props`` and ``attrs``.
        """
        init_dict = init_dict or {}
        self.version = DexforceW1Version.parse(
            init_dict.get("version", DexforceW1Version.V021)
        )
        spec = get_w1_version_spec(self.version)
        self.with_default_eef = bool(
            init_dict.get("with_default_eef", spec.default_with_eef)
        )

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
            fname=(
                f"{spec.assembly_name}_pika"
                if self.with_default_eef
                and all(
                    self.hand_types.get(side) == DexforceW1HandBrand.PIKA_GRIPPER
                    for side in DexforceW1ArmSide
                )
                else None
            ),
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
        self.sensors = self._build_default_sensors()
        if (
            self.version == DexforceW1Version.V030
            and self.with_default_eef
            and all(
                self.hand_types.get(side) == DexforceW1HandBrand.PIKA_GRIPPER
                for side in DexforceW1ArmSide
            )
        ):
            # Source articulation order is breadth-first: torso, head/arm
            # branches interleaved, then fingers (not XML joint-element order).
            # A shallow S bend keeps the upper body upright. Relaxed upper
            # arms put the elbows below the shoulders and forearms forward.
            # Gripper opening axes are horizontal; approach axes point about
            # ten degrees down. Head pitch supplies the tabletop view; the
            # head camera itself faces forward in its parent link frame.
            # Scene layouts choose their own work height.
            left = [
                0.0,
                -1.25,
                0.0,
                -1.70,
                -0.32,
                0.30,
                0.0,
            ]
            right = [
                0.0,
                1.25,
                0.0,
                1.70,
                -0.32,
                -0.30,
                0.0,
            ]
            self.init_qpos = (
                [
                    -0.15,
                    0.30,
                    -0.15,
                    0.0,
                    0.0,
                    left[0],
                    right[0],
                    float(np.deg2rad(40.0)),
                ]
                + [value for pair in zip(left[1:], right[1:]) for value in pair]
                + [0.05] * 4
            )

    def _build_default_sensors(self) -> list[CameraCfg]:
        """Declare ideal RGB-D cameras for the optional V030 Pika suite."""
        from embodichain.lab.sim.sensors import CameraCfg
        from scipy.spatial.transform import Rotation

        pika_sides = [
            side
            for side in DexforceW1ArmSide
            if self.with_default_eef
            and self.version == DexforceW1Version.V030
            and self.hand_types.get(side) == DexforceW1HandBrand.PIKA_GRIPPER
        ]
        if not pika_sides:
            return []

        def mount(
            parent: str,
            eye: tuple[float, float, float],
            target: tuple[float, float, float],
            up: tuple[float, float, float],
        ) -> CameraCfg.ExtrinsicsCfg:
            # Native mounted views use OpenGL: -Z forward, +Y up.
            forward = np.subtract(target, eye, dtype=float)
            forward /= np.linalg.norm(forward)
            right = np.cross(forward, up)
            right /= np.linalg.norm(right)
            rotation = np.column_stack((right, np.cross(right, forward), -forward))
            return CameraCfg.ExtrinsicsCfg(
                parent=parent,
                pos=eye,
                quat=tuple(Rotation.from_matrix(rotation).as_quat()),
            )

        # Simulation-designed mounts, not hardware calibration. The raised
        # wrist views clear the housing and retain the near grasp below image
        # centre, with room ahead of the fingertips for approach context.
        sensors = [
            CameraCfg(
                uid="cam_high",
                width=960,
                height=540,
                intrinsics=(505.81446, 487.09289, 480.0, 270.0),
                near=0.05,
                far=3.0,
                enable_color=True,
                enable_depth=True,
                extrinsics=mount(
                    "head_pitch_j2_link",
                    (0.105, 0.0, 0.06),
                    (1.105, 0.0, 0.06),
                    (0.0, 0.0, 1.0),
                ),
            )
        ]
        for side in pika_sides:
            sensors.append(
                CameraCfg(
                    uid=f"cam_{side.value}_wrist",
                    width=640,
                    height=360,
                    intrinsics=(337.20964, 324.72860, 320.0, 180.0),
                    near=0.01,
                    far=2.0,
                    enable_color=True,
                    enable_depth=True,
                    extrinsics=mount(
                        f"{side.value}_gripper_base",
                        (0.055, 0.0, 0.135),
                        (0.195, 0.0, 0.00064),
                        (0.0, 0.0, 1.0),
                    ),
                )
            )
        return sensors

    def _build_default_solver_cfg(self):
        """Build version-matched solvers with URDF joint coordinates and limits."""
        from embodichain.lab.sim.motion.solvers import SRSSolverCfg, SolverCfg
        from embodichain_robots.dexforce.w1.params import (
            W1ArmKineParams,
        )

        solver_cfg = {}
        spec = get_w1_version_spec(self.version)
        for arm_side in DexforceW1ArmSide:
            part_name = f"{arm_side.value}_arm"
            tcp = spec.tcp(arm_side)
            if self.with_default_eef:
                hand = get_w1_hand_spec(
                    self.hand_types.get(arm_side, DexforceW1HandBrand.BRAINCO_HAND),
                    self.hand_versions[arm_side],
                ).for_side(arm_side)
                if hand.tcp is not None:
                    attach = self.hand_attach_xposes.get(arm_side, hand.attach_xpos)
                    tcp = spec.compose_eef_attach_xpos(
                        arm_side, np.asarray(attach) @ np.asarray(hand.tcp)
                    )
            if spec.arm_solver_configs is not None:
                solver_cfg[part_name] = SolverCfg.from_dict(
                    {
                        **spec.arm_solver_configs[arm_side],
                        "root_link_name": f"{arm_side.value}_arm_base",
                        "end_link_name": f"{arm_side.value}_ee",
                        "tcp": tcp,
                    }
                )
                continue
            params = W1ArmKineParams(arm_side=arm_side, version=self.version)
            solver_cfg[part_name] = SRSSolverCfg(
                end_link_name=f"{arm_side.value}_ee",
                root_link_name=f"{arm_side.value}_arm_base",
                dh_params=params.dh_params,
                user_qpos_limits=params.qpos_limits,
                T_e_oe=params.T_e_oe,
                T_b_ob=params.T_b_ob,
                link_lengths=params.link_lengths,
                rotation_directions=params.rotation_directions,
                tcp=tcp,
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
        joint_drive_props = JointDrivePropertiesCfg(
            drive_type="force",
            **joint_params,
        )

        if with_default_eef:
            eef_joint_names = DEFAULT_EEF_HAND_JOINT_NAMES
            if any(
                self.hand_types.get(side, DexforceW1HandBrand.BRAINCO_HAND)
                == DexforceW1HandBrand.BRAINCO_HAND
                for side in DexforceW1ArmSide
            ):
                joint_drive_props.stiffness.update(
                    {eef_joint_names: DEFAULT_EEF_JOINT_DRIVE_PARAMS["stiffness"]}
                )
                joint_drive_props.damping.update(
                    {eef_joint_names: DEFAULT_EEF_JOINT_DRIVE_PARAMS["damping"]}
                )
                joint_drive_props.max_effort.update(
                    {eef_joint_names: DEFAULT_EEF_JOINT_DRIVE_PARAMS["max_effort"]}
                )
            for side, brand in self.hand_types.items():
                if brand == DexforceW1HandBrand.PIKA_GRIPPER:
                    part = f"{side.value}_eef"
                    joint_drive_props.stiffness[part] = 1000.0
                    joint_drive_props.damping[part] = 100.0
                    joint_drive_props.max_effort[part] = 40.0
                    if joint_drive_props.max_velocity is None:
                        joint_drive_props.max_velocity = {}
                    joint_drive_props.max_velocity[part] = 0.25

        return {
            "joint_drive_props": joint_drive_props,
            "root_props": ArticulationRootPropertiesCfg(
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
        from embodichain_robots.dexforce.w1.utils import arm_manager

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
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Runtime device override; otherwise the selected backend default is used.",
    )
    args = parser.parse_args()

    config = SimulationManagerCfg(
        headless=True,
        device=args.device,
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


__all__ = ["DexforceW1Cfg"]
