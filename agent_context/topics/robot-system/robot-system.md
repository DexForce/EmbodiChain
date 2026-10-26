# Robot System

## Find the owner

| Change or question | Start here |
|---|---|
| Runtime control parts, FK/IK adaptation, joint ordering | `embodichain/lab/sim/objects/robot.py` → `Robot` |
| Config decoding, construction hooks, backend alternatives | `embodichain/lab/sim/cfg/robot.py` → `RobotCfg`, `RobotPresetCfg` |
| Available robot configs and variants | `extensions/embodichain_robots/_catalog.py`; `extensions/embodichain_robots/__init__.py` owns lazy exports |
| Shared Gym/CLI/dual-arm selection | `embodichain/robot_registry.py` → `list_robot_cfgs`, `resolve_robot_cfg` |
| Robot listing CLI | `embodichain/cli/list_robot.py` → `embodichain list-robot` |
| Generic robot preview and standalone RobotCfg files | `embodichain/lab/scripts/preview_robot.py` → `embodichain preview-robot` |
| W1 variants and serialization | `extensions/embodichain_robots/dexforce/w1/` → `cfg.py`, `specs.py`, `hand_specs.py` |
| Composed robot frames and joint-name mapping | `embodichain/lab/sim/robots/dual_arm.py` |
| Shared articulation state and drive properties | `embodichain/lab/sim/objects/articulation.py`, `embodichain/lab/sim/cfg/articulation.py` |

`Robot` extends `Articulation` with named control parts, per-part solvers and
runtime workspace access. Robot configs own asset/kinematic assembly and
optional mounted camera declarations; the manager owns creation and backend
binding. For a new config use
[add-robot](../../../.agents/skills/add-robot/SKILL.md); for reusable robot/sensor
suites use [add-embodiment-component](../../../.agents/skills/add-embodiment-component/SKILL.md).

Concrete robot definitions live in the separate `embodichain_robots` import
package under `extensions/embodichain_robots/`, bundled in the main wheel with
the core version. `extensions/` is only a source container; imports keep the
`embodichain_` prefix. Generic `Robot`,
`RobotCfg`, `RobotPresetCfg`, and dual-arm composition remain in the core.
Concrete presets import from `embodichain_robots` and follow vendor/model
ownership. Model packages keep `cfg.py` and their own `assets.py`; shared
Franka and UR archives belong at vendor level. Model and vendor initializers
must stay lazy so asset lookup cannot load robot configurations. Asset discovery
and backward-compatible asset IDs are owned by
[data assets](../data-assets/data-assets.md). The core
`embodichain.lab.sim.robots` package exposes only generic dual-arm composition.

Catalog listing defers robot classes and asset downloads; resolution imports
one `RobotCfg` subclass and isolates its defaults/overrides. Gym, workspace
analysis, dual-arm composition, and preview share `embodichain/robot_registry.py`.
Selectors use exact registered IDs; aliases and `Cfg` suffixes are not accepted.
An explicit `variant` applies profile defaults before caller overrides, while
omission preserves existing construction behavior. See
[catalog contracts](catalogs.md) for entry points, metadata and model profiles.

`preview-robot` accepts a catalog ID or a standalone JSON/YAML RobotCfg file.
File `class_type` optionally selects the catalog; local asset paths, including
`urdf_cfg.sensors[*].sensor_source` URDF files in lists or mappings, resolve
from the file directory. Inline sensor definitions and data IDs are preserved.
Configuration decoding and validation precede World
creation. Native and Viser lifecycle, Gizmo/slider target coordination, and
cleanup belong to [preview activation](../sim-visualization/configuration.md#robot-preview).

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

W1 V030 defaults to the standard 20-DOF bare body and version-owned FEP
configuration; earlier revisions retain their SRS and attached-hand defaults.
V030 redundancy search defaults to joint distance and limit-margin scoring;
its outstretched home pose has no defined GeoFIK swivel angle.
Its default TCPs are the `left_ee` / `right_ee` flanges. Select optional hands
explicitly and configure their tool TCPs independently. V030 preserves source
arm link names and omits uncalibrated wrist-camera attachments. Its unified
asset archive keeps component visuals and inertials aligned with the full body.

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

Aloha Mini and Marvin use strict Warp SSIK arm defaults on CPU/CUDA; Marvin
selects seed-local SRS search in both gripper variants. Aloha's torso retains
its position-only Pytorch solver. Numerical restoration requires a complete
replacement with the existing root/end/TCP and original Pytorch settings; see
the [solver guide](../../../docs/source/overview/sim/motion/solvers/ssik_solver.md#selecting-the-numerical-solver).
ARX5 uses the
source X5A six-joint `arm` and two
independent prismatic fingers in `eef`; its simulation and PK share the selected
URDF. ARX5 defaults to near-parallel Warp SSIK, and Flexiv Rizon4S defaults to
prepared Warp RR with seed-local search. Both run on CPU/CUDA without the
official SSIK dependency; the prepared CPU reference artifacts under
`embodichain_robots/_ssik_artifacts/` are explicit precision-alignment tools.
These models remain outside strict Warp geometry; see
[IK mode boundaries](../ik-solvers/ik-solvers.md) before selecting one.
Typed `SolverCfg` overrides and dictionaries with `class_type` replace
the selected part's solver; typed inputs are copied.
Provide complete root/end/TCP configuration for replacements. Partial
dictionaries without `class_type` update the selected default's attributes.

Robot owns arena/root frame conversion; solvers consume chain-root poses.
Native root poses are rigid transforms. Single and batch IK use the existing
`pose_inv` rigid inverse, preserving gradients and avoiding the additional
CUDA/CPU synchronization of a general matrix inverse. Native state retrieval
has its own backend synchronization boundary.
`RobotCfg.from_dict()` resolves configured solver types through
`embodichain.lab.sim.motion.solvers`. For batch/candidate shapes and continuous
IK support, read [IK contracts](../ik-solvers/ik-solvers.md). Pose conventions
are owned by [simulation](../simulation-system/simulation-system.md); external
library adapters own conversions.

## Physics and motion integration

`RobotCfg.sensors` owns robot-local camera calibration; `enable_sensor`
opts into initialization in standalone simulation. Gym overrides that switch
on an isolated robot configuration using its environment acquisition setting.
Concrete defaults live in `embodichain_robots` and reach Gym and online buffers
through the shared catalog resolver. `URDFCfg.sensors` remains physical sensor
asset assembly, separate from these runtime camera declarations.
`TianjiMarvinCfg` preserves source mass, inertia and COM values, including for
custom URDFs. The bundled `tianji_marvin.yaml` embodiment explicitly opts into
recorded simulation mass/tensor overrides (1000 times the pinned asset entries).
Source units are unconfirmed; this is a demo calibration assumption, not a
manufacturer-verified SI conversion. The component records the asset revision
and fingerprint and preserves source COMs. The reusable embodiment aligns the
catalog tool convention with GraspKit's opening/approach axes and calibrates
gripper drives. Reuse that component for Task Programs; the standalone catalog
retains its documented source-tool TCP convention.
The gripper-equipped Marvin preset also owns two 640×360 wrist RGB-D cameras,
`cam_left_wrist` and `cam_right_wrist`, parented to the hand-base links.
Their nominal 87°×58° optics approximate D405; mounting transforms are measured
from the bundled single-lens mesh, not calibrated on hardware, and depth is
ideal rendered geometry. The bare-arm variant declares no cameras. Explicit
`sensors` overrides replace the suite; standalone creation and Gym acquisition
continue to use the existing sensor switches. The embodiment's independent
`sensor` list does not duplicate these mounted declarations.
Both Marvin variants default to the compact ready pose with lowered elbows;
the gripper variant appends four open-finger positions. `init_qpos` follows
source order (interleaved left/right arm joints, then fingers). The shared
embodiment explicitly pins the same pose, and the official Marvin demos inherit
it; their deployment files do not duplicate older ready-pose overrides.
The manager qualifies parent links with the final robot UID without changing
the declaration. Independent cameras remain deployment/embodiment-owned.
Creation, attachment and rebuild behavior belong to
[sensors](../sensor-system/sensor-system.md).

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

`RobotCfg` preserves asset velocity limits unless a drive override is configured.
Planning reads effective speeds through `get_qvel_limits()` and existing mimic
metadata. Acceleration budgets and missing-speed fallback belong to the
[planner contracts](../motion-planning/planner-details.md).

## Focused validation

| Changed boundary | Existing coverage |
|---|---|
| Robot catalogs, third-party selection across consumers | `tests/test_robot_registry.py` |
| Robot preview inputs, initialization, and cleanup | `tests/lab/scripts/test_preview_robot.py` |
| Robot CLI metadata, table formatting, and deferred imports | `tests/cli/test_list_robot.py`, `tests/test_main.py` |
| Removed legacy imports and lazy canonical robot imports | `tests/sim/robots/test_imports.py` |
| Bundled robot package discovery/staging | `tests/test_robot_package.py` |
| Config merge, W1 versions/TCP round-trips, chain DOFs | `tests/sim/objects/test_robot_cfg.py` |
| W1 V030 component assembly, source FK agreement and configured FEP branches | `tests/sim/robots/test_dexforce_w1_v030.py` |
| Dual-arm source/assembled frames and properties | `tests/sim/objects/test_dual_arm.py` |
| Runtime part/joint/frame behavior | `tests/sim/objects/test_robot.py` |
| Spawn binding and joint order | `tests/sim/spawn/test_create_robot_integration.py` |
| Lazy motion imports and batch conversion | `tests/sim/motion/test_motion_imports.py`, `tests/sim/motion/solvers/test_analytic_batching.py` |

For drive or mimic failures, first inspect resolved articulation properties and
backend binding rather than adding robot-specific post-bind fixes. For missing
IK, inspect configured solver coverage; a gripper control part need not have a
solver. For asset changes, validate round-trip and chain DOF/frame agreement
before using executable robot smoke programs.
