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
    LinkPhysicsOverrideCfg,
    MassPropertiesCfg,
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


# The pinned ACD assets author CAD inertials in tonnes and tonne*m^2.
# Geometry and COM positions are already in metres. Convert mass and tensor
# together to SI; these are asset-unit corrections, not hardware measurements.
# Store source values without opening/downloading the URDF during config decode.
_SOURCE_INERTIALS = {
    "base_link": (
        0.10924916672943885,
        (
            0.01776149514025449,
            5.92945593866503e-05,
            -0.00015472877847083467,
            0.017781653971041884,
            0.0003135716580764859,
            0.007660437480320593,
        ),
    ),
    "shoulder_pitch_l_j1_link": (
        0.001022387748937657,
        (
            2.274199017700984e-06,
            4.4572217887609066e-11,
            4.256867307550405e-09,
            2.6383092580492356e-06,
            3.086859756190181e-09,
            1.227441922759216e-06,
        ),
    ),
    "shoulder_roll_l_j2_link": (
        0.0015613550419125903,
        (
            7.3728181458601535e-06,
            1.4628326041106542e-09,
            -5.616330102546891e-10,
            3.2862791094052253e-06,
            -8.381610151926439e-08,
            6.3157343728840645e-06,
        ),
    ),
    "elbow_pitch_l_j3_link": (
        0.0013059921871783956,
        (
            4.5011355549082065e-06,
            -1.0848693891764643e-08,
            -4.802661959208509e-07,
            4.851240558686273e-06,
            4.0689221271008405e-09,
            1.556160395236958e-06,
        ),
    ),
    "elbow_yaw_l_j4_link": (
        0.0011258463088203098,
        (
            3.8015495243392016e-06,
            3.466448828163349e-07,
            -9.826594874564531e-10,
            2.071810521652473e-06,
            -5.9073104754785457e-08,
            3.195734373507308e-06,
        ),
    ),
    "wrist_pitch_l_j5_link": (
        0.001503691797829414,
        (
            5.751031732044275e-06,
            -2.5904888247073935e-10,
            4.0261958325366654e-10,
            4.987934208007358e-06,
            2.064893001799057e-08,
            2.3796446815271796e-06,
        ),
    ),
    "wrist_yaw_l_j6_link": (
        0.00033757609437364743,
        (
            2.733482366338966e-07,
            -2.283352054139201e-10,
            -3.1795918936992455e-10,
            2.7408020333217485e-07,
            -1.5791932475640234e-10,
            2.434843516853184e-07,
        ),
    ),
    "wrist_roll_l_j7_link": (
        0.0005118967491441608,
        (
            1.2664986236408559e-06,
            1.614407785456914e-09,
            8.114255674948263e-10,
            7.361544652919328e-07,
            -5.947798513626531e-08,
            9.16112899252986e-07,
        ),
    ),
    "left_hand_base_link": (
        0.0005706945509411907,
        (
            6.817814849934279e-07,
            -7.553873634438599e-10,
            1.9630032314867964e-09,
            7.633192554885682e-07,
            2.972394751654703e-08,
            8.952781270801773e-07,
        ),
    ),
    "left_hand_finger_1_link": (
        6.432025198309782e-05,
        (
            3.8466244513762666e-08,
            -4.277021130990905e-09,
            2.4439175482705916e-11,
            1.0717005614605896e-08,
            4.838032509823553e-10,
            3.5771614244097386e-08,
        ),
    ),
    "left_hand_finger_2_link": (
        6.445177772551064e-05,
        (
            3.9074701022922086e-08,
            -4.329744436331003e-09,
            3.923325311590611e-11,
            1.0641228322216664e-08,
            6.284110620435163e-10,
            3.643957391478125e-08,
        ),
    ),
    "shoulder_pitch_r_j1_link": (
        0.0010388907393414969,
        (
            2.2912313829652147e-06,
            -4.0762249292438064e-10,
            -2.24933123723361e-09,
            2.662801711231245e-06,
            3.999662533996741e-09,
            1.2585631995266217e-06,
        ),
    ),
    "shoulder_roll_r_j2_link": (
        0.0015644293828819304,
        (
            7.376375347592197e-06,
            3.8011835684804914e-09,
            -1.8219275300411277e-09,
            3.2872541279977307e-06,
            -5.043663793282001e-08,
            6.319800251755692e-06,
        ),
    ),
    "elbow_pitch_r_j3_link": (
        0.0012687073893219852,
        (
            4.4305029604386495e-06,
            -1.689029198496597e-10,
            -4.946254655039799e-07,
            4.761640175470552e-06,
            -5.7792220168768085e-09,
            1.4811059931891835e-06,
        ),
    ),
    "elbow_yaw_r_j4_link": (
        0.0010931425767976867,
        (
            3.7541450952375055e-06,
            3.477543608933041e-07,
            4.1672329784509855e-09,
            2.025962030826252e-06,
            -6.822503239864959e-08,
            3.132841388935162e-06,
        ),
    ),
    "wrist_pitch_r_j5_link": (
        0.0015247527828858188,
        (
            5.909636956547121e-06,
            9.294917505819594e-10,
            -1.2483013908963513e-10,
            5.143985607270804e-06,
            3.889847432748663e-08,
            2.4076634119053065e-06,
        ),
    ),
    "wrist_yaw_r_j6_link": (
        0.00033629194399054763,
        (
            2.7377165939096053e-07,
            6.827144303567213e-10,
            4.833264015227659e-10,
            2.773838236796365e-07,
            -2.2244024340749913e-10,
            2.404273401922962e-07,
        ),
    ),
    "wrist_roll_r_j7_link": (
        0.0005264684248123574,
        (
            1.2966698102777334e-06,
            -1.0298122275393345e-09,
            2.467162868318493e-10,
            7.644983667638317e-07,
            -5.1509088697943015e-08,
            9.370760313372811e-07,
        ),
    ),
    "right_hand_base_link": (
        0.0005671668441461823,
        (
            6.739459688996715e-07,
            -3.1438277968359306e-10,
            4.711176945385528e-10,
            7.581116623344933e-07,
            2.5970346368526087e-08,
            8.875084645081255e-07,
        ),
    ),
    "right_hand_finger_1_link": (
        6.450518918135029e-05,
        (
            3.851965101412703e-08,
            -4.27190814017143e-09,
            3.255329391529175e-11,
            1.076789443151807e-08,
            4.716062172705686e-10,
            3.5780957577847674e-08,
        ),
    ),
    "right_hand_finger_2_link": (
        6.43409948173199e-05,
        (
            3.8455941445840604e-08,
            -4.268850488354619e-09,
            2.900274830102174e-11,
            1.0733351535663621e-08,
            4.968814181794604e-10,
            3.5748772232209025e-08,
        ),
    ),
}

_BARE_INERTIAL_OVERRIDES = {
    "shoulder_pitch_l_j1_link": (
        0.001022352266920573,
        (
            2.2742664587785906e-06,
            7.51998781705033e-11,
            4.3536560867291295e-09,
            2.638367186121991e-06,
            3.0150862317606652e-09,
            1.2273982752539431e-06,
        ),
    ),
    "wrist_yaw_l_j6_link": (
        0.0003375777501435227,
        (
            2.7335618126996335e-07,
            -2.3001217858011923e-10,
            -3.1796190139777424e-10,
            2.740886853659144e-07,
            -1.565700520958329e-10,
            2.434802696876589e-07,
        ),
    ),
    "wrist_yaw_r_j6_link": (
        0.0003362982463811784,
        (
            2.737853332104249e-07,
            7.014094025639766e-10,
            4.843272223616304e-10,
            2.7737547390257856e-07,
            -2.3492539002278585e-10,
            2.40418742968457e-07,
        ),
    ),
}


@configclass
class TianjiMarvinCfg(RobotCfg):
    """Configure Tianji Marvin with or without its parallel-jaw grippers.

    Both variants expose seven-joint ``left_arm`` and ``right_arm`` parts.
    The ACD source mass and inertia use tonne-based units; link overlays
    convert both to SI while preserving source COM positions.
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
        source_inertials = dict(_SOURCE_INERTIALS)
        if not self.with_gripper:
            source_inertials = {
                name: value
                for name, value in source_inertials.items()
                if "_hand_" not in name
            }
            source_inertials.update(_BARE_INERTIAL_OVERRIDES)
        self.link_attrs = {}
        for name, (mass, moments) in source_inertials.items():
            ixx, ixy, ixz, iyy, iyz, izz = moments
            self.link_attrs[name] = LinkPhysicsOverrideCfg(
                link_names_expr=[name],
                attrs=RigidBodyPhysicsCfg(
                    mass_props=MassPropertiesCfg(
                        mass=mass * 1000.0,
                        inertia=[
                            [ixx * 1000.0, ixy * 1000.0, ixz * 1000.0],
                            [ixy * 1000.0, iyy * 1000.0, iyz * 1000.0],
                            [ixz * 1000.0, iyz * 1000.0, izz * 1000.0],
                        ],
                    ),
                    collision_props=None,
                    material_props=None,
                ),
            )
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
