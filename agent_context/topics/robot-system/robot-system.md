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

All robot configs support `from_dict(init_dict)` class method for dict-based construction.

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

Full guide: `docs/source/tutorial/add_robot.rst`

Minimal checklist:
1. Create a `@configclass` inheriting `RobotCfg`.
2. Define `urdf_cfg` with URDF component paths and transforms.
3. Define `control_parts` mapping part names to joint name lists.
4. Set `drive_pros` with appropriate stiffness/damping per joint or part.
5. Configure `solver_cfg` (one `SolverCfg` per control part).
6. For complex robots with multiple variants, use a sub-package with `types.py`, `params.py`, `utils.py`, `cfg.py` (see `dexforce_w1/` as example).
7. Export from `embodichain/lab/sim/robots/__init__.py`.
8. Add robot docs in `docs/source/resources/robot/` and update `docs/source/resources/robot/index.rst`.

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
