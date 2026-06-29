# Robot System

## Entry Points

| What | Path |
|---|---|
| Robot runtime class | `embodichain/lab/sim/objects/robot.py` → `Robot` |
| RobotCfg base config | `embodichain/lab/sim/cfg.py` → `RobotCfg` (line ~1455) |
| ArticulationCfg parent | `embodichain/lab/sim/cfg.py` → `ArticulationCfg` (line ~1345) |
| JointDrivePropertiesCfg | `embodichain/lab/sim/cfg.py` → `JointDrivePropertiesCfg` (line ~654) |
| Robot registry (all robots) | `embodichain/lab/sim/robots/__init__.py` |
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

## RobotCfg Pattern

Inheritance chain:

```
ObjectBaseCfg          uid, init_pos, init_rot, init_local_pose
  └─ ArticulationCfg   fpath, drive_pros, attrs, link_attrs, fix_base,
  │                     disable_self_collision, init_qpos, body_scale,
  │                     build_pk_chain, use_usd_properties
      └─ RobotCfg      control_parts, urdf_cfg, solver_cfg, drive_pros (override default to "force")
          ├─ DexforceW1Cfg   version, arm_kind, with_default_eef
          └─ CobotMagicCfg   (dual-arm defaults)
```

Key fields on `RobotCfg`:

| Field | Type | Purpose |
|---|---|---|
| `control_parts` | `Dict[str, List[str]] \| None` | Part name → joint names (supports regex like `JOINT[1-6]`) |
| `urdf_cfg` | `URDFCfg \| None` | Multi-component URDF assembly (e.g. left_arm + right_arm) |
| `solver_cfg` | `SolverCfg \| Dict[str, SolverCfg] \| None` | IK solver config; dict keys must match `control_parts` keys |
| `drive_pros` | `JointDrivePropertiesCfg` | Default drive type is `"force"` (overrides Articulation's `"none"`) |
| `attrs` | `RigidBodyAttributesCfg` | Rigid-body physics attributes (mass, friction, damping, ...) |
| variant fields | `enum \| str \| bool` | Optional subclass fields (e.g. `version`, `arm_kind`, `with_default_eef`) |
| `_pk_urdf_path` | `property \| method → str` | URDF for the FK/IK serial chain (one source, so it can't drift from sim) |

## The robot config protocol

Every robot config subclasses `RobotCfg` and overrides two hooks. `from_dict` is a
3-line template — do not reimplement it:

```python
@classmethod
def from_dict(cls, init_dict):
    cfg = cls()
    cfg._build_defaults(init_dict)
    return merge_robot_cfg(cfg, init_dict)
```

- **`_build_defaults(self, init_dict=None)`** — read variant fields from `init_dict`,
  set them on `self`, then populate `urdf_cfg`, `control_parts`, `solver_cfg`,
  `drive_pros` and `attrs`. (Base `RobotCfg._build_defaults` is a no-op.)
- **`build_pk_serial_chain(self, device=...)`** — return `{control_part: pk.SerialChain}`,
  reading the PK URDF from a single `_pk_urdf_path` source (a property for
  constant-path robots, a method when the path depends on a variant).

Serialization (`to_dict` / `to_string` / `save_to_file`) is **inherited** from
`RobotCfg` and round-trips: `RobotCfg.from_dict(cfg.to_dict())` reproduces the cfg.

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

## Drive Properties

`JointDrivePropertiesCfg` controls the physics drive for joints:

| Field | Type | Default | Notes |
|---|---|---|---|
| `drive_type` | `"force" \| "acceleration" \| "none"` | `"force"` (on RobotCfg) | `"none"` means no applied force |
| `stiffness` | `float \| Dict[str, float]` | `1e4` | Per-joint via dict; keys support regex |
| `damping` | `float \| Dict[str, float]` | `1e3` | Same |
| `max_effort` | `float \| Dict[str, float]` | `1e10` | Max torque/force |
| `max_velocity` | `float \| Dict[str, float]` | `1e10` | rad/s or m/s |
| `friction` | `float \| Dict[str, float]` | `0.0` | Joint friction |

When using a dict, keys are joint names or regex patterns matching joint names. Control-part names can also be used as keys (resolved via `ArticulationCfg` logic).

## Adding a New Robot

Full guide: `docs/source/tutorial/add_robot.rst` · Quick reference: `docs/source/guides/add_robot.rst`

Minimal checklist:
1. Create a `@configclass` inheriting `RobotCfg`.
2. Override `_build_defaults(self, init_dict=None)` — read variant fields from `init_dict`, then populate `urdf_cfg`, `control_parts`, `solver_cfg`, `drive_pros` and `attrs`.
3. Keep `from_dict` as the 3-line template (`cls()` → `_build_defaults` → `merge_robot_cfg`); do not reimplement.
4. Define `control_parts` mapping part names to joint name lists.
5. Configure `solver_cfg` (one `SolverCfg` per control part).
6. Implement `build_pk_serial_chain` reading from `_pk_urdf_path` (property for constant paths, method for variant-dependent).
7. For robots with variants, use a sub-package with `types.py` (enums + `__all__`), `cfg.py` (variant-aware `_build_defaults`), optional `params.py` / `utils.py` helpers (see `dexforce_w1/` as example).
8. Export from `embodichain/lab/sim/robots/__init__.py` and set `__all__`.
9. Add robot docs in `docs/source/resources/robot/` and update `docs/source/resources/robot/index.rst`.
10. Test — a `__main__` smoke test + the DOF drift guard + `preview-asset` CLI.

Serialization (`to_dict` / `save_to_file`) is inherited — no need to implement it.

## Available Robots

| Robot | Config Class | Module | Structure | Notes |
|---|---|---|---|---|
| DexForce W1 | `DexforceW1Cfg` | `embodichain/lab/sim/robots/dexforce_w1/` | Package (`cfg.py`, `types.py`, `params.py`, `utils.py`) | Humanoid; versions: V021; arm kinds: ANTHROPOMORPHIC, INDUSTRIAL; component types: chassis, torso, eyes, head, left/right arm/hand |
| CobotMagic | `CobotMagicCfg` | `embodichain/lab/sim/robots/cobotmagic.py` | Single file | Dual-arm; 6-DOF arms + 2-DOF grippers; uses OPW solver |

## Common Failure Modes

- **`solver_cfg` keys don't match `control_parts` keys** — solver init silently uses wrong part or errors at IK time.
- **Regex joint names not expanded** — if robot is not properly initialized, regex patterns like `JOINT[1-6]` remain unexpanded. Always construct via `from_dict()` or let `Robot.__init__` handle expansion.
- **`drive_type="none"` inherited from ArticulationCfg** — if you inherit `ArticulationCfg` directly instead of `RobotCfg`, the default drive type is `"none"` (no forces applied). Override to `"force"`.
- **Missing `urdf_cfg` for multi-component robots** — single-file robots use `fpath`; multi-component robots (e.g. dual-arm) require `urdf_cfg` with component transforms.
- **Mimic joints not excluded** — `get_joint_ids(remove_mimic=False)` includes mimic joints by default. Pass `remove_mimic=True` for active-only joints.
- **`init_qpos` shape mismatch** — must be `(num_joints,)`. A wrong-length array causes silent truncation or index errors at sim start.
- **`all` instead of `__all__`** — lowercase `all` does not work with `from module import *`; use `__all__`.
- **`solver_cfg` set in multiple places** — set it once in `_build_defaults` only; setting it elsewhere (e.g. a build helper) gets overwritten and is dead code.
- **PK URDF drifts from the sim URDF** — route `build_pk_serial_chain` through `_pk_urdf_path` and keep the DOF drift-guard test so silent drift is caught.
- **Reimplementing `from_dict`** — keep the 3-line template; put construction logic in `_build_defaults`. (Making the base `RobotCfg.from_dict` call `merge_robot_cfg` would infinite-recurse, since `merge_robot_cfg` calls `RobotCfg.from_dict`.)
