# Robot System

## Entry Points

| What | Path |
|---|---|
| Robot runtime class | `embodichain/lab/sim/objects/robot.py` → `Robot` |
| RobotCfg base config | `embodichain/lab/sim/cfg/robot.py` → `RobotCfg` |
| Replace-only backend preset | `embodichain/lab/sim/cfg/robot.py` → `RobotPresetCfg` |
| Environment robot declaration | `embodichain/lab/gym/envs/embodied_env.py` → `EmbodiedEnvCfg.robot` |
| ArticulationCfg parent | `embodichain/lab/sim/cfg/articulation.py` → `ArticulationCfg` |
| Joint drive/dynamics config | `embodichain/lab/sim/cfg/articulation.py` → `JointDrivePropertiesCfg` |
| Robot registry (all robots) | `embodichain/lab/sim/robots/__init__.py` |
| Robot executable smoke entry points | Each specified robot module's ``__main__`` block |
| DexforceW1 config package | `embodichain/lab/sim/robots/dexforce_w1/` |
| CobotMagic config | `embodichain/lab/sim/robots/cobotmagic.py` |
| Add-robot tutorial | `docs/source/tutorial/add_robot.rst` |
| Add-robot quick-reference | `docs/source/guides/add_robot.rst` |

## Overview

`Robot` extends `Articulation` (which extends `BatchEntity`). It adds:
- **Control parts** — named groups of joints (e.g. `left_arm`, `right_eef`) that can be driven independently.
- **IK solvers** — per-part solver config (`solver_cfg` dict keyed by control-part name).
- **Planners** — motion planner attachment point.

A `Robot` is instantiated with a `RobotCfg` and a list of DexSim `Articulation` entities.

Robot FK/IK and end-effector pose APIs use the EmbodiChain convention:
quaternions are `xyzw`, and 7D poses are `xyz + xyzw`. Solver or planner
adapters convert only when their external library uses another order.

## RobotCfg Pattern

Inheritance chain:

```
ObjectBaseCfg          uid, init_pos, init_rot, init_local_pose
  └─ ArticulationCfg   fpath, joint_drive_props, attrs, link_attrs, root_props,
  │                     init_qpos, qpos_limits, body_scale, build_pk_chain,
  │                     asset_physics_mode
      └─ RobotCfg      control_parts, urdf_cfg, solver_cfg, joint_drive_props (position+velocity force default)
          ├─ DexforceW1Cfg   version, hand_versions, with_default_eef
          └─ CobotMagicCfg   (dual-arm defaults)
```

Key fields on `RobotCfg`:

| Field | Type | Purpose |
|---|---|---|
| `control_parts` | `Dict[str, List[str]] \| None` | Part name → joint names (supports regex like `JOINT[1-6]`) |
| `urdf_cfg` | `URDFCfg \| None` | Multi-component URDF assembly (e.g. left_arm + right_arm) |
| `solver_cfg` | `SolverCfg \| Dict[str, SolverCfg] \| None` | IK solver config; dict keys must match `control_parts` keys |
| `joint_drive_props` | `JointDrivePropertiesCfg` | Single joint-property entry point for target mode, gains, effort/velocity limits, passive friction, and armature. Robot supplies the established `drive_type="force"`; unspecified fields remain source-owned |
| `asset_physics_mode` | `AssetPhysicsMode` | Robot defaults to `overlay`; generic articulations default to `preserve` |
| `attrs` | `RigidBodyPhysicsCfg` | Grouped rigid-body physics. Flat attribute keys are rejected; COM quaternions use `xyzw`. |
| `root_props` | `ArticulationRootPropertiesCfg` | Sole root-property interface. Fixed-base/self-collision are portable; root sleep and paired solver-iteration fields are Default-only |
| variant fields | `enum \| str \| bool` | Optional subclass fields (e.g. `version`, `with_default_eef`) |
| `_pk_urdf_path` | `property \| method → str` | URDF for the FK/IK serial chain (one source, so it can't drift from sim) |

## The robot config protocol

Every robot config subclasses `RobotCfg` and overrides the construction hooks.
The default `from_dict` implementation is this 3-line template:

```python
@classmethod
def from_dict(cls, init_dict):
    cfg = cls()
    cfg._build_defaults(init_dict)
    return merge_robot_cfg(cfg, init_dict)
```

- **`_build_defaults(self, init_dict=None)`** — read variant fields from `init_dict`,
  set them on `self`, then populate `urdf_cfg`, `control_parts`, `solver_cfg`,
  `joint_drive_props` and `attrs`. (Base
  `RobotCfg._build_defaults` is a no-op.)
- **`build_pk_serial_chain(self, device=...)`** — return `{control_part: pk.SerialChain}`,
  reading the PK URDF from a single `_pk_urdf_path` source (a property for
  constant-path robots, a method when the path depends on a variant).

Serialization (`to_dict` / `to_string` / `save_to_file`) is inherited from
`RobotCfg` unless the config stores version-derived runtime transforms.
`DexforceW1Cfg` is the current exception: serialized hand transforms and solver
TCPs are raw end-effector values, while the in-memory values include the selected
W1 revision offset. Its `to_dict` removes that derived offset and `from_dict`
restores it. Every config, including this exception, must satisfy
`type(cfg).from_dict(cfg.to_dict())` without changing the selected components or
applying a derived transform twice.

### Physics backend portability

Keep backend-neutral intent in one ordinary `RobotCfg`. In particular,
`CollisionPropertiesCfg.contact_offset/rest_offset` compile directly to
Default and to Newton's `margin=rest_offset`,
`gap=contact_offset-rest_offset`. Use `DefaultCollisionPropertiesCfg` only as a
Default-native extension point; those two inherited fields are portable.
Default-only articulation sleep and solver iterations belong directly in
`ArticulationRootPropertiesCfg` under `root_props`; `sleep_threshold`,
`min_position_iters`, and `min_velocity_iters` no longer exist as flat
`ArticulationCfg` fields. EmbodiChain applies these values to the Default-native
articulation root before the first reset, while Newton ignores them. Use
`DefaultRigidBodyPropertiesCfg` under `attrs` or
`link_attrs` only when the intended target is an individual rigid body/link.
Keep portable rigid-body values and one selected backend subtype in the single
matching `RigidBodyPhysicsCfg` slot. Backend-specific whole-robot alternatives
belong in `RobotPresetCfg`; there are no coexisting per-property backend blocks.

When a backend truly needs a different asset or complete actuator/physics
definition, subclass `RobotPresetCfg` and declare complete alternatives. The
required `default` field selects the Default backend and is the Newton fallback;
optional names include `newton`, `newton_mujoco_warp`/`newton_mjwarp`, and other
`newton_<solver>` profiles. `SimulationManager.add_robot()` selects from its
existing `physics_cfg` and active Newton solver, returns a deep-copied complete
`RobotCfg`, and never merges fields across alternatives. `EmbodiedEnvCfg.robot`
accepts either form and delegates selection to that same boundary. Prefer a
single portable `RobotCfg`; use a preset only for irreducible backend
differences.

W1 robot and hand releases use separate types and registries:

- `DexforceW1Version` selects body/arm assets, kinematics, and flange calibration
  through `specs.py`.
- `DexforceW1HandVersion` selects external hand/gripper assets, joint metadata,
  and raw mounting transforms through `hand_specs.py`.
- The current default is hand V021 for every W1 robot version. Never infer a
  hand version from `DexforceW1Version`.
- `DexforceW1Cfg` always represents a complete dual-arm W1. Structural
  `include_*`, `arm_sides`, and mixed `component_versions` options are not part
  of its public protocol.

.. note::
    `merge_robot_cfg` calls the base `RobotCfg.from_dict` internally, so the
    subclass `from_dict` template must stay the 3-line form above — making
    `RobotCfg.from_dict` itself call `_build_defaults` → `merge_robot_cfg` would
    infinite-recurse.

## Control Parts

`control_parts` maps a human-readable part name to a list of joint names:

```python
control_parts = {
    "left_arm": ["LEFT_JOINT1", ..., "LEFT_JOINT6"],
    "left_eef": ["LEFT_JOINT7", "LEFT_JOINT8"],
    "right_arm": ["RIGHT_JOINT1", ..., "RIGHT_JOINT6"],
    "right_eef": ["RIGHT_JOINT7", "RIGHT_JOINT8"],
}
```

- Joint names support **regex patterns** (e.g. `"JOINT[1-6]"`) — expanded at init.
- When `control_parts` is set, `solver_cfg` **must** be a dict with matching keys.
- `Robot.get_joint_ids(name)` returns joint IDs for a part; `None` returns all joints.
- `Robot.get_link_names(name)` returns child link names for a part.
- Internal `ControlGroup` dataclass stores `joint_names`, `joint_ids`, `link_names` per part.

For Spawn-bound robots, control-part IDs are resolved by name against the
final batch `qpos` layout. Newton may use a different source-articulation
traversal order, so do not derive control-part IDs by enumerating native joint
names. `init_qpos` keeps its source-articulation order and is remapped by name
when the robot resets.

### Mimic joints across physics backends

`Articulation.mimic_ids` and `mimic_parents` use the final batch-state joint
order, just like `qpos`, `qvel`, and control-part IDs. Spawn source metadata is
normalized by joint name before these properties are exposed. Newton initial
positions are also projected onto each URDF relation
`child = multiplier * parent + offset` before the first simulation step.

Newton's MuJoCo-Warp solver lowers URDF mimic joints to native joint equality
constraints, whose default solver reference is underdamped compared with
Default's PhysX mimic. For position-driven mimic parents, the Spawn-bound
articulation keeps those native equality rows enabled and tunes their MuJoCo
`solref` from the authored parent gains. A weak follower drive (1% of the
parent gains for the W1 hand) stabilizes the equality between solver updates,
and target `set_qpos()`/`set_qvel()` writes propagate the authored
`child = multiplier * parent + offset` relation. Never copy measured follower
state or disable the native equality: either change would turn mimic into an
independent servo and lose the mechanical coupling. Other Newton solvers,
gradient mode, and Default keep their native behavior.

## Drive Properties

`JointDrivePropertiesCfg` is the single joint-property config:

| Field | Type | Default | Notes |
|---|---|---|---|
| `drive_type` | `"force" \| "acceleration" \| "none"` | `"force"` (on RobotCfg) | Original drive response; active `"acceleration"` is Default-only |
| `target_mode` | `"none" \| "position" \| "velocity" \| "position_velocity" \| "effort"` or per-joint mapping | Derived from `drive_type` | Portable actuator intent; integer values 0–4 are accepted. `force` defaults to `position_velocity` |
| `stiffness` | `float \| Dict[str, float]` | `1e4` | Per-joint via dict; keys support regex |
| `damping` | `float \| Dict[str, float]` | `1e3` | Same |
| `max_effort` | `float \| Dict[str, float]` | `None` | Max torque/force |
| `max_velocity` | `float \| Dict[str, float]` | `None` | rad/s or m/s |
| `friction` | `float \| Dict[str, float]` | `None` | Passive joint friction |
| `armature` | `float \| Dict[str, float]` | `None` | Added joint-space inertia |

When using a dict, keys are joint names or regex patterns matching joint names. Control-part names can also be used as keys (resolved via `ArticulationCfg` logic).

Target mode is backend-neutral and belongs directly on
`JointDrivePropertiesCfg`. Default emulates the target selection with its drive
mode and effective gains; Newton authors `JointTargetMode` values for
`"none"`, `"position"`, `"velocity"`, `"position_velocity"`, and
`"effort"` (integer values 0–4). `NewtonJointDrivePropertiesCfg` remains only
to round-trip older `joint_drive_props.backend: newton` dictionaries; do not use it in
new specified robots.

`drive_type` retains its original meaning. With no explicit `target_mode`,
`force` and `acceleration` select `position_velocity`, while `none` selects a
passive target. An explicit target mode overrides that target default. Active
acceleration drives are rejected on Newton because Newton has no equivalent
mass-independent response.

For solver-independent safety, `none` and `effort` clear Kp/Kd, while
`velocity` clears Kp. MuJoCo Warp consumes the target-mode enum natively. Other
Newton solvers use the gain fallback; their position-only fallback assumes the
velocity target remains zero. Direct generalized effort continues through
`Articulation.set_qf()` and can also act as feed-forward effort with an active
PD drive.

These rules are resolved to exact joint names after URDF/USD source resolution
and before Spawn finalization. Common effort/velocity/armature values are
authored on `JointDesc`; the portable target intent lowers to Default drive
mode/gains and Newton's integer target mode. The dual-arm builder preserves the
config type and mirrors regex-keyed values to the generated `left_`/`right_`
names.

`qpos_limits` accepts either joint-name/regex rules or a flattened
`(num_dofs, 2)` array. Both forms are resolved into common `JointDesc`
limits before the Default or Newton model is built; do not add a post-bind
Newton rebuild for initial limits.

## Adding a New Robot

Full guide: `docs/source/tutorial/add_robot.rst` · Quick reference: `docs/source/guides/add_robot.rst`

Minimal checklist:
1. Create a `@configclass` inheriting `RobotCfg`.
2. Override `_build_defaults(self, init_dict=None)` — read variant fields from `init_dict`, then populate `urdf_cfg`, `control_parts`, `solver_cfg`, `joint_drive_props` and `attrs`.
3. Keep `from_dict` as the 3-line template (`cls()` → `_build_defaults` → `merge_robot_cfg`) unless version-derived state requires an explicitly documented post-merge step.
4. Define `control_parts` mapping part names to joint name lists.
5. Configure `solver_cfg` (one `SolverCfg` per control part).
6. Implement `build_pk_serial_chain` reading from `_pk_urdf_path` (property for constant paths, method for variant-dependent).
7. For robots with variants, use a sub-package with `types.py` (enums + `__all__`), `cfg.py` (variant-aware `_build_defaults`), optional `params.py` / `utils.py` helpers (see `dexforce_w1/` as example).
8. Export from `embodichain/lab/sim/robots/__init__.py` and set `__all__`.
9. Add robot docs in `docs/source/resources/robot/` and update `docs/source/resources/robot/index.rst`.
10. Test — a `__main__` smoke test + the DOF drift guard + `preview-asset` CLI.

Serialization (`to_dict` / `save_to_file`) is normally inherited. A robot-specific
override requires documented raw/final semantics and regression tests for default,
custom-transform, component-version, and public-builder round-trips.

## Available Robots

| Robot | Config Class | Module | Structure | Notes |
|---|---|---|---|---|
| DexForce W1 | `DexforceW1Cfg` | `embodichain/lab/sim/robots/dexforce_w1/` | Package (`cfg.py`, `types.py`, `specs.py`, `hand_specs.py`, `params.py`, `utils.py`) | Humanoid; robot and hand versions are independently registered |
| CobotMagic | `CobotMagicCfg` | `embodichain/lab/sim/robots/cobotmagic.py` | Single file | Dual-arm; 6-DOF arms + 2-DOF grippers; portable collision envelope, Default-native root iterations, OPW solver |

## Executable smoke programs

Every specified robot module accepts ``--physics {default,newton}`` in its
``__main__`` smoke program and resolves the selection through
``physics_cfg_for_backend()``. CobotMagic, Franka, UR, and DualArm retain the
Default backend as their command-line default; DexforceW1 retains Newton as its
default. These entry points exercise the same ordinary ``RobotCfg`` definitions
on either backend rather than maintaining backend-specific demo configs.

## Common Failure Modes

- **`solver_cfg` keys don't match `control_parts` keys** — solver init silently uses wrong part or errors at IK time.
- **Regex joint names not expanded** — if robot is not properly initialized, regex patterns like `JOINT[1-6]` remain unexpanded. Always construct via `from_dict()` or let `Robot.__init__` handle expansion.
- **No drive config on generic `ArticulationCfg`** — its `joint_drive_props=None` keeps source drives. Use `RobotCfg` for the standard position+velocity force-drive defaults, or provide an explicit sparse drive overlay.
- **Missing `urdf_cfg` for multi-component robots** — single-file robots use `fpath`; multi-component robots (e.g. dual-arm) require `urdf_cfg` with component transforms.
- **Mimic joints not excluded** — `get_joint_ids(remove_mimic=False)` includes mimic joints by default. Pass `remove_mimic=True` for active-only joints.
- **`init_qpos` shape mismatch** — must match active DOFs. A wrong-length array causes initialization errors.
- **`all` instead of `__all__`** — lowercase `all` does not work with `from module import *`; use `__all__`.
- **`solver_cfg` set in multiple places** — set it once in `_build_defaults` only; setting it elsewhere (e.g. a build helper) gets overwritten and is dead code.
- **PK URDF drifts from the sim URDF** — route `build_pk_serial_chain` through `_pk_urdf_path` and keep the DOF drift-guard test so silent drift is caught.
- **Reimplementing `from_dict` without a serialization protocol** — keep the 3-line template by default. If derived version state requires post-merge processing, document the raw/final values and test round-trips. (Making the base `RobotCfg.from_dict` call `merge_robot_cfg` would infinite-recurse, since `merge_robot_cfg` calls `RobotCfg.from_dict`.)
