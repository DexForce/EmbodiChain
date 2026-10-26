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
