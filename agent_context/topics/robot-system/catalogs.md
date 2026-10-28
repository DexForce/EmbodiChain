# Robot catalogs

Catalog selection and display metadata are owned by `embodichain/robot_registry.py`,
`embodichain/cli/list_robot.py` and `extensions/embodichain_robots/_catalog.py`.
For runtime configuration, return to [robot-system](robot-system.md).

The `embodichain.robots` entry points provide lightweight catalog mappings from
IDs to `class_path: module:Class` and optional `defaults`. Catalogs/package
initializers defer robot implementation imports. `embodichain list-robot`
reads only catalogs and prints robot families, supported variants, independent
control DOFs/parts, and mimic followers without constructing configs or
downloading assets. `dof` excludes mimic followers and overlapping groups count
once; `control_parts` records raw per-part counts, with `control_part_mimics`
marking follower membership. `mimic_joints` maps URDF followers to leaders,
including passive followers outside control groups. Missing summaries stay
unknown. UR presets use `variant_of` to share a family row; DualArmRobot summaries
are derived from catalog bases explicitly declaring `supports_dual_arm`.
W1 V021/V022/V025 variants advertise dual BrainCo, DH/DH-M pairings,
and `with_default_eef=False`, omitting dexterous/gripper mixtures. V030 advertises
only the standard `v030/bare` profile; optional tool assemblies need their own
mount/TCP validation before catalog promotion. Hand release versions remain
independent of body releases. Tests compare each declared profile with config defaults.
Resolution imports the selected `RobotCfg` subclass and calls `from_dict()`
with isolated defaults/overrides. Duplicate IDs fail explicitly. Gym
`robot.class_type`, workspace `--robot`, and dual-arm `base_robot` share this
resolver. Selectors use exact registered IDs; old aliases and `Cfg`-suffixed
spellings are not accepted. UR variants have their own default parameters. After changing
entry-point metadata, refresh the editable install and restart the process.

Classification is optional display metadata: `vendor` and `morphology` are
nonempty strings or `None`; `arm_count` is a nonnegative integer or `None`,
excluding booleans. Morphology describes the authored body, independently of
arm count and simulation anchoring. Missing variant fields inherit the parent;
explicit `None` stays unknown. Classification differences prevent variant rows
from being grouped. Never infer these fields from paths or control-group names.

Only the stock `FrankaPanda` hand is an official Panda preset. The former PGI
preset is removed; examples and benchmark fixtures own their explicit Panda/PGI
assemblies, including flange FK, TCP, drives and finger contacts. Flat concrete
robot configuration module paths are retired; root class exports remain stable.

`resolve_robot_cfg(..., variant=...)` selects a declared profile's defaults
between catalog defaults and caller overrides. Omitting the variant keeps
existing construction behavior; `default_variant` remains display metadata.

## Locomotion model profiles

The locomotion catalog IDs are `UnitreeGo1`, `UnitreeGo2`, `UnitreeG1`,
`UnitreeH1_2`, `ANYmalC` and `MicroDuck`. They select the bundled simulation
assets, including the 29-joint G1 and 27-joint H1_2 models, through ordinary
`RobotCfg` subclasses. Model descriptions and catalog listing remain independent
of task imports and asset downloads. Models live under `unitree/{go1,go2,g1,h1_2}`,
`anybotics/anymal_c` and `pollen_robotics/microduck`, with lazy initializers and
model-owned `cfg.py`, `assets.py` and wheel-packaged `profile.json`. Keep profile
packaging coverage when changing the resource layout. Catalog classification
separates quadrupeds, bipedal humanoids and the MicroDuck biped; G1 and H1_2
declare two arms independently of morphology.

`locomotion_profiles.load_locomotion_robot_profile()` shares asset, standing-pose,
drive and reference-frame data without importing simulation backends. Velocity
task contracts use the same model data while action scaling, observations,
rewards and fall-contact rules stay task-owned. The presets use the generic
`Robot`, retain the `locomotion` group and `build_pk_chain=False`, and expose no
serial-chain IK solver. Task initialization does not reapply model drive defaults,
so explicit Gym robot overrides survive. Before drive rules merge, the presets
remove model defaults selected by caller exact names, regex or control parts;
unselected joints keep their defaults. Runtime direct-rule priority and caller
ambiguity validation remain unchanged. Keep source-ordered `asset.init_qpos`
separate from `robot.joint_names` / `robot.default_joint_position`. These are
simulation PD presets, without external learned actuator implementations.
