# Robot System

## Find the owner

| Change or question | Start here |
|---|---|
| Runtime control parts, FK/IK adaptation, joint ordering | `embodichain/lab/sim/objects/robot.py` → `Robot` |
| Config decoding, construction hooks, backend alternatives | `embodichain/lab/sim/cfg/robot.py` → `RobotCfg`, `RobotPresetCfg` |
| Available robot configs | `embodichain/lab/sim/robots/__init__.py` |
| W1 variants and serialization | `embodichain/lab/sim/robots/dexforce_w1/` → `cfg.py`, `specs.py`, `hand_specs.py` |
| Composed robot frames and joint-name mapping | `embodichain/lab/sim/robots/dual_arm.py` |
| Shared articulation state and drive properties | `embodichain/lab/sim/objects/articulation.py`, `embodichain/lab/sim/cfg/articulation.py` |

`Robot` extends `Articulation` with named control parts, per-part solvers and
runtime workspace access. Robot configs own asset/kinematic assembly; the
manager owns creation and backend binding. For a new config use
[add-robot](../../../.agents/skills/add-robot/SKILL.md); for reusable robot/sensor
suites use [add-embodiment-component](../../../.agents/skills/add-embodiment-component/SKILL.md).

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

CobotMagic retains the asset URDF's arm effort limits. Its arm speed caps use
the stricter of the asset and standard PiPER manual references; its coupled
gripper owns qualified simulation force and speed budgets in `cobotmagic.py`.
Its decoder removes defaults selected by caller drive rules before the shared
merge, preserving legacy whole-arm patterns, control-part overrides and
per-joint overrides without overlapping default rules.
Distinguish that generalized drive force from per-joint URDF limits, firmware
control values and product clamping-force ratings. Validate actuator changes with
`scripts/benchmark/robotics/cobotmagic_drives.py`: it compares gravity holds,
per-joint steps and smooth tracking, and `--task-config` qualifies loaded Task
Program execution with measured lift, tilt, return, idle-arm displacement,
bottle-to-gripper drift and cup contact/clearance checks.
These are simulation validations; standard PiPER references do not establish
V100 hardware ratings, and selected gains are not measured device calibration.
Qualification reads effective native speed caps in final joint order rather
than assuming the asset defaults. Generalized `qf` on Default is not a drive
torque sensor; the benchmark labels its PD-demand estimate accordingly.
