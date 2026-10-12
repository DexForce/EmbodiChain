# Robot System

## Find the owner

| Change or question | Start here |
|---|---|
| Runtime control parts, FK/IK adaptation, joint ordering | `embodichain/lab/sim/objects/robot.py` → `Robot` |
| Config decoding, construction hooks, backend alternatives | `embodichain/lab/sim/cfg/robot.py` → `RobotCfg`, `RobotPresetCfg` |
| Available robot configs and variants | `embodichain_robots/_catalog.py`; `embodichain_robots/__init__.py` owns lazy exports |
| Shared Gym/CLI/dual-arm selection | `embodichain/robot_registry.py` → `list_robot_cfgs`, `resolve_robot_cfg` |
| Robot listing CLI | `embodichain/cli/list_robot.py` → `embodichain list-robot` |
| W1 variants and serialization | `embodichain_robots/dexforce_w1/` → `cfg.py`, `specs.py`, `hand_specs.py` |
| Composed robot frames and joint-name mapping | `embodichain/lab/sim/robots/dual_arm.py` |
| Shared articulation state and drive properties | `embodichain/lab/sim/objects/articulation.py`, `embodichain/lab/sim/cfg/articulation.py` |

`Robot` extends `Articulation` with named control parts, per-part solvers and
runtime workspace access. Robot configs own asset/kinematic assembly; the
manager owns creation and backend binding. For a new config use
[add-robot](../../../.agents/skills/add-robot/SKILL.md); for reusable robot/sensor
suites use [add-embodiment-component](../../../.agents/skills/add-embodiment-component/SKILL.md).

Concrete robot definitions live in the separate `embodichain_robots` import
package, bundled in the main wheel with the core version. Generic `Robot`,
`RobotCfg`, `RobotPresetCfg`, and dual-arm composition remain in the core.
Concrete presets import from `embodichain_robots`. The core
`embodichain.lab.sim.robots` package exposes only generic dual-arm composition.

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
W1 variants advertise body releases with dual BrainCo, DH/DH-M pairings,
and `with_default_eef=False`, omitting dexterous/gripper mixtures; hand release versions remain independent
of body release versions. Tests compare each declared profile with config defaults.
Resolution imports the selected `RobotCfg` subclass and calls `from_dict()`
with isolated defaults/overrides. Duplicate IDs fail explicitly. Gym
`robot.class_type`, workspace `--robot`, and dual-arm `base_robot` share this
resolver. Selectors use exact registered IDs; old aliases and `Cfg`-suffixed
spellings are not accepted. UR variants have their own default parameters. After changing
entry-point metadata, refresh the editable install and restart the process.

## Configuration resolution

Specified configs build variant defaults before applying user overrides through
`merge_robot_cfg` in `embodichain/lab/sim/utility/cfg_utils.py`. Keep
`_build_defaults()` and the base `RobotCfg.from_dict()` separate: the merge
helper calls the base decoder, so routing it back through the subclass build/
merge sequence recurses. The skill owns the scaffold; inspect a neighboring
robot config for the current implementation pattern.

Keep the simulation asset and `build_pk_serial_chain()` source aligned. The
specified configs use `_pk_urdf_path` as the kinematic source; validate chain
DOFs against the selected control parts after changing assets or variants.

Serialization must round-trip without changing components or applying derived
transforms twice. W1 is the important exception to plain inherited
serialization: runtime hand attachments and solver TCPs include a body-revision
offset, while serialized values remain raw. Its `to_dict()` removes the offset
and `from_dict()` restores it. Body and hand versions have independent registries;
do not infer one from the other. `DexforceW1Cfg` represents a complete dual-arm
robot; inspect its decoder for rejected structural options.

`DualArmRobotCfg.build_pk_serial_chain()` resolves each solver's root/end
frames against that arm's source URDF. It translates assembled names back to
source link names, keeps chains arm-local, and rejects opposite-arm/unknown
frames. Mount transforms belong to assembly, not the serial chain.

## Joint and solver boundaries

Official defaults use `arm` / `eef` for single-arm control groups and
`left_arm`, `right_arm`, `left_eef`, `right_eef` for dual-arm groups. The EEF
group can control parallel grippers or dexterous hands; its name does not
declare capabilities. Custom keys, including `hand`, remain valid and are not
automatically aliased. URDF component names such as `left_hand` and native link
or joint names belong to the asset namespace and are independent of this rule.

Panda's default `hand` group is now `eef`; Aloha Mini, Tianji Marvin and the
official dual-UR5 embodiment use `left_eef` / `right_eef` in place of
`left_hand` / `right_hand`. Callers of those defaults must update named-part
queries, drive overrides, endpoint bindings and service keys together. Existing
trajectory-file field names are independent and should be mapped explicitly
when loading older data, rather than renamed as part of a control-group change.

`control_parts` expands joint-name patterns at initialization. For Spawn-bound
robots, part IDs resolve by name against the final batch `qpos` order, not
native source traversal order. Source-ordered `init_qpos` is remapped on reset.
Mimic IDs and parents use that same final state order; select active-only IDs
explicitly when needed. Shared mimic/actuator lowering belongs to
[simulation](../simulation-system/simulation-system.md).

Action terms resolve `control_parts` with `remove_mimic=True` and own only those
independent joint IDs. Separate arm and gripper terms may share one robot but
cannot overlap the same command type and joint. Parallel grippers expose one
policy scalar while mapping it to all independent finger commands; a future
tendon or synergy action should use its own resource namespace rather than
pretending tendon indices are joint IDs.

With control parts, solver configuration is a dictionary whose keys resolve
against part names (including patterns); it need not configure every part.
`Robot.init_solver()` fills absent solver joint names from the selected part
and synchronizes effective robot limits. Check the resolved part and joint
order before diagnosing an IK algorithm.

Robot owns arena/root frame conversion; solvers consume chain-root poses.
`RobotCfg.from_dict()` resolves configured solver types through
`embodichain.lab.sim.motion.solvers`. For batch/candidate shapes and continuous
IK support, read [IK contracts](../ik-solvers/ik-solvers.md). Pose conventions
are owned by [simulation](../simulation-system/simulation-system.md); external
library adapters own conversions.

## Physics and motion integration

Keep portable intent in an ordinary `RobotCfg`; joint-property and root/body
physics ownership is defined in [simulation](../simulation-system/simulation-system.md).
Use `RobotPresetCfg` only for a complete alternative asset/actuator definition.
`SimulationManager.add_robot()` selects one deep-copied alternative from the
active backend/solver, falling back to `default`; it never merges alternatives.
`EmbodiedEnvCfg.robot` delegates to that same selection boundary.

Motion subpackages and offline workspace exports must remain lazy so Robot
initialization does not import planners or analyzers. `RobotCfg.workspace_cfg`
selects per-part caches; `get_workspace()` loads them on first access, while
`attach_workspace()` accepts an existing cache on the robot device. Follow
[robot workspace](../robot-workspace/robot-workspace.md) for cache ownership and
[motion planning](../motion-planning/motion-planning.md) for planner integration.

## Focused validation

| Changed boundary | Existing coverage |
|---|---|
| Robot catalogs, third-party selection across consumers | `tests/test_robot_registry.py` |
| Robot CLI metadata, table formatting, and deferred imports | `tests/cli/test_list_robot.py`, `tests/test_main.py` |
| Removed legacy imports and lazy canonical robot imports | `tests/sim/robots/test_imports.py` |
| Bundled robot package discovery/staging | `tests/test_robot_package.py` |
| Config merge, W1 versions/TCP round-trips, chain DOFs | `tests/sim/objects/test_robot_cfg.py` |
| Dual-arm source/assembled frames and properties | `tests/sim/objects/test_dual_arm.py` |
| Runtime part/joint/frame behavior | `tests/sim/objects/test_robot.py` |
| Spawn binding and joint order | `tests/sim/spawn/test_create_robot_integration.py` |
| Lazy motion imports and batch conversion | `tests/sim/motion/test_motion_imports.py`, `tests/sim/motion/solvers/test_analytic_batching.py` |

For drive or mimic failures, first inspect resolved articulation properties and
backend binding rather than adding robot-specific post-bind fixes. For missing
IK, inspect configured solver coverage; a gripper control part need not have a
solver. For asset changes, validate round-trip and chain DOF/frame agreement
before using executable robot smoke programs.
