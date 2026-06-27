# Dual-Arm Composition

`DualArmRobotCfg` composes a two-arm (bimanual) robot from **any** single-arm
robot config that follows the existing `"arm"` convention — one `"arm"` URDF
component, one `control_parts["arm"]` entry, and one `solver_cfg["arm"]` entry.
Today this covers the UR family; future single-arm robots (e.g. Franka) work
for free after a one-line registry entry.

The two arms are mounted on a shared synthetic `base_link` via the existing
`URDFCfg` multi-component assembly. The left/right `control_parts`, per-arm
`solver_cfg`, and mirrored `drive_pros` are derived automatically by the
`build_dual_arm_cfg` engine — no per-robot dual-arm class is needed.

## Key Features

- **Generic engine** — `build_dual_arm_cfg(base_cfg, mounts)` derives a dual-arm
  cfg from any single-arm `RobotCfg`. No edits to the base robot's class.
- **Dict/YAML constructible** — `DualArmRobotCfg.from_dict({...})` resolves
  `base_robot` to the single-arm class via a small registry and runs the engine.
- **Preset mounts** — `side_by_side` (default) and `facing_inward`, plus a
  per-arm override escape hatch for custom layouts.
- **Per-arm IK** — one solver per arm, with link names prefixed (or kept
  arm-local) depending on whether the base solver adopts the assembled URDF.
- **Composite `dual_arm` part** — optionally a single control part addressing
  both arms' joints at once.
- **Round-trippable** — `DualArmRobotCfg.from_dict(cfg.to_dict())` reproduces
  the cfg.

## Usage

```python
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.robots import DualArmRobotCfg

sim = SimulationManager(SimulationManagerCfg(headless=True, num_envs=4))

# One-liner: two UR5 arms, side by side, 0.6 m apart.
cfg = DualArmRobotCfg.from_dict(
    {"base_robot": "ur5", "mount": {"preset": "side_by_side", "separation": 0.6}}
)
robot = sim.add_robot(cfg=cfg)
```

### Mounts

| Preset          | Layout                                                        |
|-----------------|---------------------------------------------------------------|
| `side_by_side`  | Left at `+separation/2` in Y, right at `-separation/2`, same orientation. |
| `facing_inward` | Same ±Y separation, yawed ±90° so the arms face each other.  |

A per-arm override replaces either side; both must be given together:

```python
cfg = DualArmRobotCfg.from_dict({
    "base_robot": "ur5",
    "mount": {
        "preset": "side_by_side",
        "separation": 0.6,
        "left":  {"xyz": [0.0, 0.35, 0.0], "rpy": [0, 0, 0]},
        "right": {"xyz": [0.1, -0.35, 0.0], "rpy": [0, 0, 0]},
    },
})
```

### Programmatic use

```python
from embodichain.lab.sim.robots import URRobotCfg, build_dual_arm_cfg, resolve_mounts

base = URRobotCfg.from_dict({"robot_type": "ur5"})
mounts = resolve_mounts({"preset": "facing_inward", "separation": 0.6})
cfg = build_dual_arm_cfg(base, mounts, dual_part=False)
```

## Configuration Parameters

| Parameter     | Description                                                              |
|---------------|--------------------------------------------------------------------------|
| `base_robot`  | Registry key (`"ur3"` … `"ur10e"`) or `{"type": ..., "init": {...}}` for extra base init params. |
| `mount`       | `{preset, separation, left?, right?}` consumed by `resolve_mounts`.      |
| `arm_part`    | Name of the base robot's manipulator control part (default `"arm"`).     |
| `dual_part`   | Emit a `dual_arm` composite control part (default `True`).               |

## Derived Control Parts

For a UR5 base, `DualArmRobotCfg` produces:

| Part         | Joints                                |
|--------------|---------------------------------------|
| `left_arm`   | `LEFT_JOINT1` … `LEFT_JOINT6`         |
| `right_arm`  | `RIGHT_JOINT1` … `RIGHT_JOINT6`       |
| `dual_arm`   | the 12 joints above, concatenated      |

Each arm has its own `URSolverCfg` (left/right), operating arm-local on the
single-arm URDF so FK/IK stay consistent per arm.

.. note::
    The assembled URDF's joint names (`LEFT_JOINT1` …) and link names
    (`left_base_link`, `left_ee_link`) are produced by `URDFAssemblyManager`
    prefixing + case normalization. The engine predicts these names and a test
    asserts they match the assembled URDF, so adding a new base robot that
    follows the `"arm"` convention needs no manual name bookkeeping.

## Adding a New Base Robot

A single-arm robot becomes dual-arm-able by:

1. Writing its single-arm cfg following the `"arm"` convention (one `"arm"`
   URDF component + `control_parts["arm"]` + `solver_cfg["arm"]`), exactly as
   `URRobotCfg` does.
2. Registering it in `_BASE_ROBOT_REGISTRY` in `embodichain/lab/sim/robots/dual_arm.py`:

   ```python
   "franka": (FrankaRobotCfg, {}),
   ```

No dual-arm class, no mixin, no boilerplate.
