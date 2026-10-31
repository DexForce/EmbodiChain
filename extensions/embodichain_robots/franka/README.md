# Original Panda with Robotiq 2F-85

`FrankaPandaRobotiq85Cfg` assembles the original seven-joint Panda arm and the
Robotiq 2F-85 linkage. It is a separate catalog entry from existing Franka
presets. The flange mount and TCP are explicit configuration; the carton demo
uses an estimated coaxial 18 mm adapter and 150 mm tool offset. Preserve that
conditional calibration when interpreting public DROID input.

The model-owned `FrankaPandaArm` asset downloads historical `FrankaV2.zip`
(MD5 `f0675b9da98126bc3d4e18c98ef5e06c`, SHA-256
`3bf4adfc5ffe6d32a18d2cfe357fcfe787d3539312ffb1b9a4638c91df680768`).
The original URDF is retained. A separately recorded variant replaces rounded
quarter-turn RPY values with exact ±π/2 in six arm joints. The simulator and
serial-chain solver share this variant and the configured mount/TCP.

Robotiq uses the existing `Robotiq_V2.zip` asset (MD5
`9bcbf92946b200c903938d59323e7b2c`, SHA-256
`c37c1745e6742a77d57fe94d3f81bc7377dd657f3ceb7570c88b5bdc7733d359`).
One driver controls five mimic followers, including negative multipliers.
`robotiq85_joint_positions()` maps observed DROID aperture to this linkage;
actuator close commands remain separate from measured contact-limited positions.

Legacy CAD descriptions omit several inertials. `_panda_robotiq_physics.py`
overlays identified Panda parameters and nominal Robotiq CAD parameters:

| Source | Pinned revision / SHA-256 | Qualification |
| --- | --- | --- |
| [Franka identified inertials](https://github.com/frankarobotics/franka_ros/blob/ddd2fffd9de44b02ad15b4bbb2bfa2cec4d60d98/franka_description/robots/common/inertial.yaml) | `ddd2fffd9de44b02ad15b4bbb2bfa2cec4d60d98` / `c7d15b15e45876b44e030053bca3ca19cf471806e46031ec090b8500cb9b330b` | Identified parameters, explicitly not manufacturer certified |
| [Robotiq nominal CAD inertials](https://github.com/ros-industrial-attic/robotiq/blob/45196f6558fe8ba9d89bc8a105396c68c3e7e892/robotiq_2f_85_gripper_visualization/urdf/robotiq_arg2f_85_model_macro.xacro) | `45196f6558fe8ba9d89bc8a105396c68c3e7e892` / `bb38a58a3112017efbc273327fc65388449f4e4a54a23c3218cf35e7b89356bb` | Values originally commented in upstream xacro |

Rubber pads use an explicit 5 g simulation assumption with box inertia. Drive
gains, force limits and solver iterations are simulation settings. None of these
parameters establishes hardware contact or dynamics equivalence.

The asset distribution declares Apache-2.0. Franka's `franka_description` is
Apache-2.0; the upstream Robotiq visualization package declares BSD and credits
Daniel Ordonez and Robotiq contributors. Preserve upstream notices when
redistributing their descriptions or meshes. The carton demo's
`assets.lock.json` records source URLs, hashes and scale conventions; visual
GLB export records each converted mesh in its read-only replay bundle.
