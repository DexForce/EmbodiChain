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

"""Tianji Marvin configuration for the two complete ACD URDF assets."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from embodichain.data import get_data_path
from embodichain.lab.sim.cfg import (
    ArticulationRootPropertiesCfg,
    CollisionPropertiesCfg,
    JointDrivePropertiesCfg,
    RigidBodyMaterialCfg,
    RigidBodyPhysicsCfg,
    RobotCfg,
    RenderCfg,
    LightCfg,
)
from embodichain.lab.sim.motion.solvers import PytorchSolverCfg
from embodichain.lab.sim.utility.cfg_utils import merge_robot_cfg
from embodichain.utils import configclass

if TYPE_CHECKING:
    import pytorch_kinematics as pk

__all__ = ["TianjiMarvinCfg"]


_PART_FRAMES = {
    True: {
        "left_arm": ("left_arm_base", "left_hand_tool_link"),
        "right_arm": ("right_arm_base", "right_hand_tool_link"),
    },
    False: {
        "left_arm": ("left_arm_base", "left_ee"),
        "right_arm": ("right_arm_base", "right_ee"),
    },
}


# Source articulation order: L1, R1, ..., L7, R7. Keep the embodiment's
# explicit ready-pose declaration aligned with this standalone default.
_READY_ARM_QPOS = (
    -1.5875022065801416,
    1.5875023792379233,
    1.2248796197752276,
    1.2248821108425374,
    1.0822212035745409,
    -1.0822204356660736,
    -2.3307274153917774,
    -2.3307274153917774,
    0.5068066018093935,
    -0.5068065733236865,
    0.8471975511965976,
    0.8471975511965976,
    -1.3707963267948964,
    1.3707963267948964,
)


@configclass
class TianjiMarvinCfg(RobotCfg):
    """Configure Tianji Marvin with or without its parallel-jaw grippers.

    Both variants expose seven-joint ``left_arm`` and ``right_arm`` parts.
    Source mass, inertia and COM values are preserved without automatic unit
    conversion, including for custom assets. Their authored units have not been
    confirmed by the manufacturer. The bundled Task Program embodiment opts
    into explicit simulation-calibrated inertial overlays.
    With grippers, ``left_eef`` and ``right_eef`` each include the two
    finger joints; finger 2 mimics finger 1. Hands use joint-space control.
    Both variants start with lowered elbows and forward-folded forearms;
    the gripper-equipped variant starts with open fingers. Explicit
    ``init_qpos`` overrides replace this ready pose in source joint order.
    This variant also declares two mounted RGB-D cameras with D405-equivalent
    nominal fields of view. Their extrinsics are estimates from the bundled
    hand mesh, not hardware calibration; depth is ideal rendered geometry.
    The bare-arm variant has no default cameras. The existing ``enable_sensor``
    switch controls creation, and an explicit ``sensors`` list replaces defaults.
    Both variants apply an additional TCP transform to their source end links:
    a translation of 0.13 m along the end link's +X axis and a +90 degree
    rotation about its Y axis. With grippers the source end links are
    ``left_hand_tool_link`` / ``right_hand_tool_link``; without grippers they
    are ``left_ee`` / ``right_ee``.

    Solver FK/IK uses the resulting TCP pose relative to each arm's base link.
    Named ``Robot.compute_fk`` / ``Robot.compute_ik`` uses that same TCP in the
    local arena frame. To use source end-link poses directly, override each
    arm's ``solver_cfg[part].tcp`` with the 4x4 identity matrix.

    Example:
        >>> cfg = TianjiMarvinCfg.from_dict({"with_gripper": True})
        >>> bare_cfg = TianjiMarvinCfg.from_dict({"with_gripper": False})
    """

    with_gripper: bool = True
    """Select ``robot_with_ee_acd.urdf``; False selects ``robot_acd.urdf``."""

    @classmethod
    def from_dict(cls, init_dict: dict[str, Any]) -> TianjiMarvinCfg:
        """Build variant defaults and apply robot configuration overrides.

        Args:
            init_dict: Robot fields to override, including ``with_gripper``.

        Returns:
            The configured Tianji Marvin preset.

        Raises:
            TypeError: If ``with_gripper`` is not a boolean.
        """
        cfg = cls()
        cfg._build_defaults(init_dict)
        return merge_robot_cfg(cfg, init_dict)

    def _build_defaults(self, init_dict: dict[str, Any] | None = None) -> None:
        """Populate the selected asset, joint groups, solvers, and physics."""
        from embodichain.lab.sim.sensors import CameraCfg

        init_dict = init_dict or {}
        self.with_gripper = init_dict.get("with_gripper", self.with_gripper)
        if not isinstance(self.with_gripper, bool):
            raise TypeError("with_gripper must be a boolean.")

        self.uid = "TianjiMarvin"
        self.fpath = self._pk_urdf_path
        self.init_qpos = list(_READY_ARM_QPOS) + (
            [0.0, 0.0, 0.0, 0.0] if self.with_gripper else []
        )
        # Nominal D405 FOV: 87 x 58 degrees at half-HD resolution.
        # https://www.realsenseai.com/product-family/d405-series/
        # The ACD hand mesh has a single-lens housing. Estimate the optical
        # origin 1 mm ahead of its lens face, tilted 12 degrees toward the jaws;
        # rotate the image so the gripper occupies its lower side. Mounted
        # CameraCfg poses use OpenGL (-Z forward, +Y up), so convert the optical
        # (+Z forward, +Y down) estimate by a local half-turn about X. This is an
        # aligned ideal RGB-D approximation, not a model of stereo matching.
        self.sensors = (
            [
                CameraCfg(
                    uid=f"cam_{side}_wrist",
                    width=640,
                    height=360,
                    intrinsics=(337.20964, 324.72860, 320.0, 180.0),
                    near=0.005,
                    far=3.0,
                    enable_color=True,
                    enable_depth=True,
                    extrinsics=CameraCfg.ExtrinsicsCfg(
                        parent=f"{side}_hand_base_link",
                        pos=(0.000669, 0.057374, 0.061580),
                        quat=(0.0, 0.994522, 0.104528, 0.0),
                    ),
                )
                for side in ("left", "right")
            ]
            if self.with_gripper
            else []
        )
        self.control_parts = {
            f"{side}_arm": [
                f"SHOULDER_PITCH_{label}_J1",
                f"SHOULDER_ROLL_{label}_J2",
                f"ELBOW_PITCH_{label}_J3",
                f"ELBOW_YAW_{label}_J4",
                f"WRIST_PITCH_{label}_J5",
                f"WRIST_YAW_{label}_J6",
                f"WRIST_ROLL_{label}_J7",
            ]
            for side, label in (("left", "L"), ("right", "R"))
        }
        if self.with_gripper:
            self.control_parts.update(
                left_eef=["LEFT_HAND_FINGER_1", "LEFT_HAND_FINGER_2"],
                right_eef=["RIGHT_HAND_FINGER_1", "RIGHT_HAND_FINGER_2"],
            )
        self.solver_cfg = {
            part: PytorchSolverCfg(
                root_link_name=root,
                end_link_name=end,
                # End-link-to-TCP transform, shared by both URDF variants.
                tcp=[
                    [0.0, 0.0, 1.0, 0.13],
                    [0.0, 1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            )
            for part, (root, end) in _PART_FRAMES[self.with_gripper].items()
        }
        # Preserve source inertials; opt into task-specific simulation overlays
        # through the bundled embodiment or explicit caller link_attrs.
        self.link_attrs = {}
        # Simulation gains follow Aloha Mini, not factory motor specifications.
        self.joint_drive_props = JointDrivePropertiesCfg(
            drive_type="force",
            stiffness={
                part: 3e2 if part.endswith("_eef") else 7e4
                for part in self.control_parts
            },
            damping={
                part: 3e1 if part.endswith("_eef") else 1e3
                for part in self.control_parts
            },
            max_effort={
                part: 3e3 if part.endswith("_eef") else 3e6
                for part in self.control_parts
            },
        )
        self.root_props = ArticulationRootPropertiesCfg(
            fixed_base=True,
            min_position_iters=8,
            min_velocity_iters=2,
        )
        self.attrs = RigidBodyPhysicsCfg(
            collision_props=CollisionPropertiesCfg(
                contact_offset=0.001,
                rest_offset=0.0,
            ),
            material_props=RigidBodyMaterialCfg(
                static_friction=0.95,
                dynamic_friction=0.9,
            ),
        )

    @property
    def _pk_urdf_path(self) -> str:
        """Use the variant's simulation URDF, honoring an ``fpath`` override."""
        filename = "robot_with_ee_acd.urdf" if self.with_gripper else "robot_acd.urdf"
        return self.fpath or get_data_path(f"TianjiMarvin/{filename}")

    def build_pk_serial_chain(
        self, device: torch.device = torch.device("cpu"), **kwargs: Any
    ) -> dict[str, pk.SerialChain]:
        """Build the two seven-joint arm chains in their local base frames.

        These raw URDF chains end at the selected source end links and omit
        ``solver_cfg[part].tcp``. Right-multiply their FK matrices by that
        transform to obtain the corresponding solver TCP poses.

        The parallel fingers branch from the hand base and are controlled in
        joint space, so they are not represented by a single serial chain.

        Args:
            device: Device for the kinematics tensors. Defaults to CPU.
            **kwargs: Reserved for compatibility with ``RobotCfg``.

        Returns:
            Serial chains keyed by ``left_arm`` and ``right_arm``.
        """
        from embodichain.lab.sim.utility.solver_utils import create_pk_serial_chain

        urdf_path = self._pk_urdf_path
        return {
            part: create_pk_serial_chain(
                urdf_path=urdf_path,
                device=device,
                root_link_name=root,
                end_link_name=end,
            )
            for part, (root, end) in _PART_FRAMES[self.with_gripper].items()
        }


def _main() -> None:
    """Visualize both TCPs and verify their FK/IK round trips."""
    import argparse

    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import MarkerCfg, physics_cfg_for_backend
    from embodichain.utils.math import quat_error_magnitude, quat_from_matrix

    parser = argparse.ArgumentParser(description="Launch the Tianji Marvin robot")
    parser.add_argument(
        "--with-gripper", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--physics", choices=("default", "newton"), default="default")
    parser.add_argument("--device", default=None)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            device=args.device,
            num_envs=1,
            render_cfg=RenderCfg(renderer="rt"),
            physics_cfg=physics_cfg_for_backend(args.physics),
        )
    )

    sim.add_light(
        LightCfg(
            uid=f"light_",
            intensity=60.0,
            radius=600,
            init_pos=[0, 0, 6.0],
        )
    )
    try:
        robot = sim.add_robot(
            cfg=TianjiMarvinCfg.from_dict(
                {
                    "with_gripper": args.with_gripper,
                    "init_pos": [0.0, 0.0, 0.1],
                }
            )
        )
        sim.prepare()
        sim.update(step=10)

        # Keep these results available in the interactive session below.
        fk_poses = {}
        ik_qpos = {}
        ik_fk_poses = {}
        position_tolerance = 1e-3  # metres
        rotation_tolerance = 1e-3  # radians
        for part in ("left_arm", "right_arm"):
            qpos = robot.get_qpos(name=part).clone()
            fk_pose = robot.compute_fk(qpos=qpos, name=part, to_matrix=True)
            fk_poses[part] = fk_pose
            sim.draw_marker(
                MarkerCfg(
                    name=f"{part}_tcp_fk",
                    axis_xpos=fk_pose,
                    axis_len=0.12,
                    axis_size=0.003,
                    arena_index=0,
                )
            )

            # Perturb the seed so IK must solve back to the FK target.
            limits = robot.get_qpos_limits(name=part)
            joint_seed = (qpos + 0.05).clamp(min=limits[..., 0], max=limits[..., 1])
            success, solved_qpos = robot.compute_ik(
                pose=fk_pose, joint_seed=joint_seed, name=part
            )
            if not bool(success.all()) or not bool(torch.isfinite(solved_qpos).all()):
                raise RuntimeError(f"{part}: IK failed for the current FK TCP pose.")

            reconstructed_pose = robot.compute_fk(
                qpos=solved_qpos, name=part, to_matrix=True
            )
            ik_qpos[part] = solved_qpos
            ik_fk_poses[part] = reconstructed_pose
            position_error = torch.linalg.vector_norm(
                reconstructed_pose[:, :3, 3] - fk_pose[:, :3, 3], dim=-1
            )
            rotation_error = quat_error_magnitude(
                quat_from_matrix(reconstructed_pose[:, :3, :3]),
                quat_from_matrix(fk_pose[:, :3, :3]),
            )
            # IK can return another joint solution; compare the TCP poses.
            matches = (position_error <= position_tolerance) & (
                rotation_error <= rotation_tolerance
            )
            print(
                f"\n{part}:\n"
                f"  original qpos (rad): {qpos[0].tolist()}\n"
                f"  IK qpos (rad):       {solved_qpos[0].tolist()}\n"
                f"  FK TCP xyz (m):      {fk_pose[0, :3, 3].tolist()}\n"
                f"  FK(IK) TCP xyz (m):  {reconstructed_pose[0, :3, 3].tolist()}\n"
                f"  position error: {position_error.max().item():.3e} m\n"
                f"  rotation error: {rotation_error.max().item():.3e} rad\n"
                f"  FK -> IK -> FK: {'PASS' if bool(matches.all()) else 'FAIL'}",
                flush=True,
            )
            if not bool(matches.all()):
                raise RuntimeError(f"{part}: FK/IK TCP poses do not match.")
            sim.draw_marker(
                MarkerCfg(
                    name=f"{part}_tcp_ik_fk",
                    axis_xpos=reconstructed_pose,
                    axis_len=0.08,
                    axis_size=0.005,
                    arena_index=0,
                )
            )

        if not args.headless:
            sim.open_window()
            from IPython import embed

            embed()
    finally:
        sim.destroy(exit_process=False)


if __name__ == "__main__":
    from embodichain.lab.sim import SimulationManager

    try:
        _main()
    finally:
        SimulationManager.flush_cleanup_queue()
