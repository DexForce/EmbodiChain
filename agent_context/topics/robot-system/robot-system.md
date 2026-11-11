# Robot System

Generic configuration/runtime belongs to `lab/sim`; concrete presets, asset
identity and calibrated defaults belong to `embodichain_robots`.

## Find the owner

| Change or question | Start here |
|---|---|
| Runtime control parts, FK/IK adaptation, joint ordering | `embodichain/lab/sim/objects/robot.py` → `Robot` |
| Pure parallel-jaw defaults, mounted/existing bindings and standalone hand configs | `embodichain/robot_registry.py` → `resolve_gripper_defaults`; `embodichain/lab/gym/utils/_gripper_composition.py`; `embodichain/lab/sim/cfg/gripper.py` |
| Explicit hand geometry capture and lazy paired grasp ownership | `embodichain/lab/sim/utility/grasp_geometry.py` |
| Config decoding, construction hooks, backend alternatives | `embodichain/lab/sim/cfg/robot.py` → `RobotCfg`, `RobotPresetCfg` |
| Available robot configs and variants | `extensions/embodichain_robots/_catalog.py`; `extensions/embodichain_robots/__init__.py` owns lazy exports |
| YAM dual-arm defaults and source-aligned FK/IK | `extensions/embodichain_robots/i2rt/yam/cfg.py` → [model and asset usage](../../../docs/source/resources/robot/i2rt/yam.md) |
| Locomotion assets, standing poses, drives and reference frames | `extensions/embodichain_robots/locomotion_profiles.py` reads each vendor/model's `profile.json`; model packages own `cfg.py` and `assets.py` |
| Shared Gym/CLI/dual-arm selection | `embodichain/robot_registry.py` → `list_robot_cfgs`, `resolve_robot_cfg` |
| Robot listing CLI | `embodichain/cli/list_robot.py` → `embodichain list-robot` |
| Generic robot preview and standalone RobotCfg files | `embodichain/lab/scripts/preview_robot.py` → `embodichain preview-robot` |
| Bare Franka arm selection and explicit tool assembly | `extensions/embodichain_robots/franka/cfg.py`, `panda.py` → [catalog contracts](catalogs.md) |
| W1 variants and serialization | `extensions/embodichain_robots/dexforce/w1/` → `cfg.py`, `specs.py`, `hand_specs.py` |
| Independent Revo1 hand or shared hand asset names | `extensions/embodichain_robots/brainco/revo1/` → `cfg.py`, `_specs.py` |
| Composed robot frames and joint-name mapping | `embodichain/lab/sim/robots/dual_arm.py` |
| Shared articulation state and drive properties | `embodichain/lab/sim/objects/articulation.py`, `embodichain/lab/sim/cfg/articulation.py` |

## Configuration resolution

The lightweight catalog is the selection boundary for Gym, preview, workspace
and dual-arm construction. It imports only the selected preset; see
[catalog contracts](catalogs.md) for IDs, variants, pure hand descriptions and
locomotion profiles. Display classification does not grant runtime capabilities.

Preset `_build_defaults()` assembles model defaults; base `RobotCfg.from_dict()`
and `merge_robot_cfg` apply overrides. Do not route the base decoder back into
the subclass build/merge sequence. Configuration construction may leave MISSING
fields; validate the completed config. Model URDFs and PK chains must agree in
joint order and selected control parts.

Serialization must preserve components and avoid applying derived transforms
twice. W1 serializes raw offsets and restores revision-owned transforms on
loading; body and hand revisions are independent. Dual-arm chains resolve their
own source URDF root/end, translate assembled names back to source names and
reject opposite-arm/unknown frames. Mount transforms belong to assembly.
Model-specific constraints are in [catalog calibration](catalogs.md#model-specific-calibration).

## Joint and solver boundaries

Control-part keys are explicit configuration, not inferred hand capabilities.
Official conventions live in AGENTS; custom keys and asset-defined link/joint
names remain valid. Pattern expansion for Spawn-bound robots resolves against
final batch qpos ordering. Source-ordered initial positions, mimic followers and
parents are remapped consistently; select independent joints explicitly.

Action terms remove mimic followers and cannot overlap another term's joint
and command type. A scalar parallel-gripper policy input still maps to the
complete independent-joint command. Solver dictionaries bind named parts and may
omit parts; `Robot.init_solver()` fills joint names and effective limits.
Typed solver configs or dictionaries with `class_type` replace defaults and
need complete root/end/TCP settings; other partial dictionaries update the
selected default. Typed inputs are copied.

Robot owns arena/root conversion and rigid inverses; solvers consume chain-root
poses. Native state acquisition retains its backend synchronization boundary.
[IK contracts](../ik-solvers/ik-solvers.md) own candidate shapes and continuous
capabilities; [simulation](../simulation-system/simulation-system.md) owns pose,
actuator and backend conventions. Effective velocity limits preserve authored
asset caps unless explicitly overridden; acceleration and fallback policies are
[motion-owned](../motion-planning/planner-details.md).

## Sampled hand geometry

`utility/grasp_geometry.py` adapts Robot state to explicit numerical GraspKit
inputs. `RobotHandGeometryCfg` selects motion/hand parts, root/pad links,
commands and sample bounds. Commands cover moving subtree joints, including
followers, in resolved order. Hand root and solver end must connect through
fixed joints; TCP stays fixed during hand actuation. An opposite-arm TCP is not
a hand reference. Calibration comes from model descriptions and explicit binding.

Named FK uses the first measured reference row without changing robot state.
Native capture may reuse the private sampler with same-observation state while
preserving its original link order and arithmetic. URDF and native sample
budgets remain caller-selected. `ordered_pad_line_world` reads the two declared
live pads without sampled FK; declaration order differs from surface TCP-X order.

GraspKit receives sampled geometry and queries without importing hardware Cfgs
or looking up models. Configured hand services use the existing
[Task Program integration](../task-programs/configuration.md).
`RobotCoordinatedGraspProvider` captures lazily on its first paired query and
stays with that Robot while model, calibration and commands are unchanged.
Single-arm calls do not trigger capture. `clear_cache()` releases query state
without recapturing; rebuilding after calibration changes remains necessary.

## Physics and motion integration

`RobotCfg.sensors` declares robot-local runtime cameras; standalone
`enable_sensor` and Gym acquisition control initialization. `URDFCfg.sensors`
assembles physical sensor assets separately. Independent cameras remain
embodiment/deployment-owned. Frame qualification and rebuild behavior belong to
[sensors](../sensor-system/sensor-system.md).

Portable intent uses ordinary `RobotCfg`. `RobotPresetCfg` supplies complete
backend/solver alternatives; manager construction selects one deep-copied
alternative, falling back to `default`, without merging alternatives. Gym uses
the same selection boundary. Runtime physics ownership stays with Spawn.

Motion imports and workspace exports remain lazy. Workspace caches load on
first access or attach explicitly on the Robot device; see
[workspace ownership](../robot-workspace/robot-workspace.md).

## Focused validation

| Boundary | Coverage |
|---|---|
| Catalog metadata, pure lookup and packaging | `tests/test_robot_registry.py`, `tests/test_robot_package.py`, `tests/cli/test_list_robot.py` |
| Config/serialization and model defaults | `tests/sim/objects/test_robot_cfg.py`, `tests/sim/robots/` |
| Composition, runtime and Spawn ordering | `tests/sim/objects/test_dual_arm.py`, `test_robot.py`, `tests/sim/spawn/test_create_robot_integration.py` |
| Hand geometry and provider lifetime | `tests/sim/utility/test_grasp_geometry.py` |
| Preview and lazy motion imports | `tests/lab/scripts/test_preview_robot.py`, `tests/sim/motion/test_motion_imports.py` |

Diagnose drive/mimic failures at descriptor lowering and binding before adding
model-specific post-bind fixes. A hand control part need not have an IK solver.
