# Franka seven-joint arms and explicit tools

`FrankaCfg` selects a bare `fr3` (default) or original `panda` arm. Its only
selector is `arm_model`; all arm variants expose seven arm joints and a flange
TCP. End effectors are mounted through the same generic URDF component,
control-group, drive and solver-TCP overrides used by UR robots. The official
embodiments own native-hand and Robotiq installations and their calibration.

The native FR3 hand is materialized as `Franka/Panda/Fr3HandV1.urdf` from the
pinned `PandaWithHand.urdf` subtree (SHA256
`a4a2d3265dadd69c4c7d8ed9238ae44494c217db6f66f586fcc40e8754ce2315`).
The original source is unchanged; the sidecar records retained link/joint names
and both hashes. The bare arm and the source integrated model have identical
arm link/joint descriptions.

The model-owned `FrankaPandaArm` asset downloads historical `FrankaV2.zip`
(MD5 `f0675b9da98126bc3d4e18c98ef5e06c`, SHA-256
`3bf4adfc5ffe6d32a18d2cfe357fcfe787d3539312ffb1b9a4638c91df680768`).
The original URDF is retained. A separately recorded variant replaces rounded
quarter-turn RPY values with exact ±π/2 in six arm joints. The simulator and
serial-chain solver share this arm variant; the embodiment adds its tool TCP.

Robotiq uses the existing `Robotiq_V2.zip` asset (MD5
`9bcbf92946b200c903938d59323e7b2c`, SHA-256
`c37c1745e6742a77d57fe94d3f81bc7377dd657f3ceb7570c88b5bdc7733d359`).
One driver controls five mimic followers, including negative multipliers.
`robotiq85_joint_positions()` maps observed DROID aperture to this linkage;
actuator close commands remain separate from measured contact-limited positions.
The downloader is owned by `embodichain_robots.robotiq.assets.Robotiq`.
The separate `Robotiq2F85` archive is a different asset and is not a replacement
for this calibrated model; the stable asset reference, joint layout and geometry
remain unchanged after the EEF ownership migration.

Legacy CAD descriptions omit several inertials. `franka/_physics.py` owns the
identified Panda arm overlay. The explicit Robotiq embodiment supplies the tool
overlays and redirects the fixed arm-base overlay to its assembled link name:

| Source | Pinned revision / SHA-256 | Qualification |
| --- | --- | --- |
| [Franka identified inertials](https://github.com/frankarobotics/franka_ros/blob/ddd2fffd9de44b02ad15b4bbb2bfa2cec4d60d98/franka_description/robots/common/inertial.yaml) | `ddd2fffd9de44b02ad15b4bbb2bfa2cec4d60d98` / `c7d15b15e45876b44e030053bca3ca19cf471806e46031ec090b8500cb9b330b` | Identified parameters, explicitly not manufacturer certified |
| [Robotiq nominal CAD inertials](https://github.com/ros-industrial-attic/robotiq/blob/45196f6558fe8ba9d89bc8a105396c68c3e7e892/robotiq_2f_85_gripper_visualization/urdf/robotiq_arg2f_85_model_macro.xacro) | `45196f6558fe8ba9d89bc8a105396c68c3e7e892` / `bb38a58a3112017efbc273327fc65388449f4e4a54a23c3218cf35e7b89356bb` | Values originally commented in upstream xacro |

Rubber pads use an explicit 5 g simulation assumption with box inertia. Drive
gains, force limits and solver iterations are simulation settings. None of these
parameters establishes hardware contact or dynamics equivalence.

The Python implementation is Apache-2.0. This does not establish the license
or original authorship of the packaged geometry. The shared Robotiq archive's
asset-specific license and upstream revision remain unverified; see the
[owning asset provenance](../robotiq/README.md). The separately cited inertial
parameter sources do not establish the provenance of those URDFs or meshes.
Preserve the applicable upstream notices when redistributing source material.
These sources define the simulation model independently of any task or dataset.
