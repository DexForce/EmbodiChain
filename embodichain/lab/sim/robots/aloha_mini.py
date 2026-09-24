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

"""Aloha Mini 2 Pro dual-arm robot configuration."""

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
)
from embodichain.lab.sim.motion.solvers import PytorchSolverCfg
from embodichain.lab.sim.utility.cfg_utils import merge_robot_cfg
from embodichain.utils import configclass

if TYPE_CHECKING:
    import pytorch_kinematics as pk

__all__ = ["AlohaMiniCfg"]


_PART_FRAMES = {
    "left_arm": ("left_Base", "left_tcp"),
    "right_arm": ("right_Base", "right_tcp"),
    "left_hand": ("left_Fixed_Jaw", "left_Moving_Jaw"),
    "right_hand": ("right_Fixed_Jaw", "right_Moving_Jaw"),
    "torso": ("base_cad_link", "vertical_link"),
}


@configclass
class AlohaMiniCfg(RobotCfg):
    """Configure Aloha Mini 2 Pro from its complete, source-named URDF.

    The arms have six joints each, the hands one gripper joint each, and the
    torso one vertical prismatic joint. Arm solvers target the URDF TCP links;
    the torso solver constrains position only. Hands use joint-space control.

    The source's planar base joints and wheels remain outside these five
    control parts and have position drives to hold their initial positions.

    Example:
        >>> cfg = AlohaMiniCfg.from_dict({"uid": "aloha_mini"})
        >>> robot = sim.add_robot(cfg=cfg)
    """

    @classmethod
    def from_dict(cls, init_dict: dict[str, Any]) -> AlohaMiniCfg:
        """Build the robot defaults and apply configuration overrides.

        Args:
            init_dict: Robot configuration fields to override.

        Returns:
            The configured Aloha Mini robot preset.
        """
        cfg = cls()
        cfg._build_defaults(init_dict)
        return merge_robot_cfg(cfg, init_dict)

    def _build_defaults(self, init_dict: dict | None = None) -> None:
        """Populate asset, joint groups, solvers, and physics defaults."""
        self.uid = "AlohaMini"
        self.fpath = self._pk_urdf_path
        self.control_parts = {
            f"{side}_arm": [
                f"{side}_shoulder_pan",
                f"{side}_shoulder_lift",
                f"{side}_elbow_flex",
                f"{side}_wrist_flex",
                f"{side}_wrist_yaw_joint",
                f"{side}_wrist_roll",
            ]
            for side in ("left", "right")
        }
        self.control_parts.update(
            left_hand=["left_gripper"],
            right_hand=["right_gripper"],
            torso=["vertical_move"],
        )
        self.solver_cfg = {
            part: PytorchSolverCfg(
                root_link_name=_PART_FRAMES[part][0],
                end_link_name=_PART_FRAMES[part][1],
                is_only_position_constraint=part == "torso",
            )
            for part in ("left_arm", "right_arm", "torso")
        }
        # Match the CobotMagic simulation gains, with separate hand drives.
        self.joint_drive_props = JointDrivePropertiesCfg(
            drive_type="force",
            stiffness={
                "left_arm": 7e4,
                "right_arm": 7e4,
                "left_hand": 3e2,
                "right_hand": 3e2,
                "torso": 7e4,
                "root_.*_joint": 7e4,
                "wheel[1-3]_joint": 7e4,
            },
            damping={
                "left_arm": 1e3,
                "right_arm": 1e3,
                "left_hand": 3e1,
                "right_hand": 3e1,
                "torso": 1e3,
                "root_.*_joint": 1e3,
                "wheel[1-3]_joint": 1e3,
            },
            max_effort={
                "left_arm": 3e6,
                "right_arm": 3e6,
                "left_hand": 3e3,
                "right_hand": 3e3,
                "torso": 3e6,
                "root_.*_joint": 3e6,
                "wheel[1-3]_joint": 3e6,
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
        """Use the simulation URDF, including an explicit ``fpath`` override."""
        return self.fpath or get_data_path("AlohaMini/alohamini2pro.urdf")

    def build_pk_serial_chain(
        self, device: torch.device = torch.device("cpu"), **kwargs: Any
    ) -> dict[str, pk.SerialChain]:
        """Build one local serial chain for each of the five control parts.

        Arm chains end at the fixed TCP links, hand chains at the moving jaws,
        and the torso chain at the vertical link. The arm chains exclude torso
        and base motion; their poses are relative to the corresponding Base.

        Args:
            device: Device for the kinematics tensors. Defaults to CPU.
            **kwargs: Reserved for compatibility with ``RobotCfg``.

        Returns:
            Serial chains keyed by control-part name, in control-joint order.
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
            for part, (root, end) in _PART_FRAMES.items()
        }


def _main() -> None:
    """Visualize both TCPs and verify their FK/IK round trips."""
    import argparse

    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import MarkerCfg, physics_cfg_for_backend
    from embodichain.utils.math import quat_error_magnitude, quat_from_matrix

    parser = argparse.ArgumentParser(description="Launch the Aloha Mini robot")
    parser.add_argument("--physics", choices=("default", "newton"), default="default")
    parser.add_argument("--device", default=None)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            device=args.device,
            num_envs=1,
            physics_cfg=physics_cfg_for_backend(args.physics),
        )
    )
    try:
        robot = sim.add_robot(cfg=AlohaMiniCfg.from_dict({}))
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
