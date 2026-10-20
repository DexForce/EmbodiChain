# embodichain_robots

Official robot configurations bundled with the `embodichain` distribution.
This import package shares the repository, version, dependencies, and wheel
with the core and `embodichain_tasks`; it has no independent release metadata.

```python
from embodichain_robots import FrankaPandaCfg

cfg = FrankaPandaCfg.from_dict({"robot_type": "panda"})
robot = sim.add_robot(cfg=cfg)
```

The package owns concrete robot defaults, hardware revisions, calibrated
transforms, and robot-specific kinematic-chain construction. Generic runtime
objects, configuration types, solvers, URDF assembly, and dual-arm composition
remain in `embodichain.lab.sim`. Assets continue to use the core download/cache
registry. Task deployments, sensor suites, and Task Program skill profiles keep
their existing owners.

## Tianji Marvin defaults

Both Marvin variants start in the shared compact ready pose: lowered elbows
and forward-folded forearms. The gripper variant also starts with open jaws.
`init_qpos` uses source articulation order (L1, R1, …, L7, R7, then four fingers
when present). Explicit caller overrides remain supported. The bundled Marvin
embodiment pins this same pose, and its three demos inherit it.

`TianjiMarvinCfg` preserves the authored URDF mass, inertia and COM values.
It does not infer the source units or apply automatic mass/inertia scaling,
including when `fpath` points to a custom asset. The bundled Task Program
embodiment explicitly opts into simulation inertial overrides using **1000 ×
the pinned asset's raw mass and tensor entries**. This is an engineering
assumption used for the qualified demos; the manufacturer has not confirmed
the source units, and this is not a certified tonnes-to-SI conversion.
The asset revision and URDF SHA-256 are recorded in the embodiment; regression
tests bind its recorded values to that snapshot and preserve source COMs.

| Parameter group | Provenance and meaning |
| --- | --- |
| Ready-pose joint positions | Designed and simulation-validated pose; arm values in radians and finger values in metres, not a factory home pose |
| Fixed base and 32/8 solver iterations | Simulation anchoring and numerical solver settings, not hardware controller parameters |
| Gripper stiffness 500 and damping 50 | Simulation tuning in N/m and N*s/m for the prismatic joints |
| Gripper effort 40 and velocity 0.1 | Per-joint simulated force/velocity limits in N and m/s, not rated gripping force or hardware speed |
| Finger friction 2.0 | Simulation tuning, not measured material friction |
| Task TCP offset 0.1125 m and rotation | Model-derived simulation reference frame aligned with GraspKit, not a measured hardware TCP |
| Finger commands 0 and -0.05 m | Joint bounds from the pinned URDF, not independently verified manufacturer specifications |
| Grasp evidence threshold 0.004 m | Tuned joint-displacement predicate, not measured contact force |

The source model is the DexForce-hosted
[TianjiMarvin asset snapshot](https://huggingface.co/datasets/DexForceAI/embodichain_data/blob/48317bd8733182f36d306cc2a7a60ae5e47ed2c9/robot_assets/TianjiMarvin.zip).
Asset traceability and simulation qualification do not establish manufacturer
certification or hardware calibration. Select the bundled embodiment for the
qualified demo settings; use source-preserving robot defaults or explicit
caller overlays for other assets.

The gripper-equipped `TianjiMarvinCfg` declares `cam_left_wrist` and
`cam_right_wrist` on the corresponding `hand_base_link`. Each returns 640×360
RGB and aligned ideal depth in metres. Nominal intrinsics
`(337.20964, 324.72860, 320, 180)` approximate the 87°×58° field of view in the
[RealSense D405 specifications](https://www.realsenseai.com/product-family/d405-series/).
Near/far render clipping is 0.005/3.0 m. Images update at the host environment's
observation cadence; the preset does not impose the hardware's frame rate.

The bundled hand mesh has a single-lens housing. The parent-relative position
`(0.000669, 0.057374, 0.061580)` m lies about 1 mm ahead of its lens face;
the optical-frame `xyzw` rotation `(0, -0.104528, 0.994522, 0)` follows its 12°
mounting inclination and places the gripper toward the lower image. Mounted
`CameraCfg` extrinsics use OpenGL (-Z forward, +Y up), so the stored quaternion
is the equivalent `(0, 0.994522, 0.104528, 0)`. These are mesh-based
simulation estimates, not measured D405 extrinsics or an interchangeable
hardware calibration. Depth comes from rendered geometry, without stereo
matching, sensor noise or hardware range invalidation. D405's nominal
0.07–0.50 m ideal working range is useful for preview coloring, not clipping RGB
or altering the raw simulated depth.

Standalone simulation opts into creation with `enable_sensor=True`. Gym uses
its environment-level `enable_sensor` switch and enables the suite by default;
`--disable-sensor` skips acquisition. An explicit `sensors` list replaces the
defaults, including `sensors=[]`. The bare-arm variant (`with_gripper=False`)
has no default cameras because it lacks the hand-base mounting links.

## Shared robot resolution

Gym's `robot.class_type`, `analyze-workspace --robot`, and the dual-arm
`base_robot` selector use `embodichain.robot_registry`:

```python
from embodichain.robot_registry import list_robot_cfgs, resolve_robot_cfg

cfg = resolve_robot_cfg("URRobot", {"robot_type": "ur5"})
cfg = resolve_robot_cfg("ur5")  # UR variant preset with its own defaults
```

The official lightweight catalog is `_catalog.py`. It contains import strings
and optional default fields; listing robots does not import their configuration
classes. Configuration classes and top-level exports load on first use.

List installed robot configurations and UR variants:

```bash
embodichain list-robot
# Equivalent module entry point:
python -m embodichain list-robot
```

The table groups registered models into robot families. URRobot lists its six
series in one entry. DualArmRobot lists compatible base variants and their own
DOFs and control parts: dual UR has 12 independent control DOFs, while dual Panda
has 16 plus two mimic followers. A `*` marks the default variant.

DexforceW1 declares body releases `v021`, `v022`, and `v025`. For each body,
the catalog covers dual BrainCo and all four pairings of DH_PGC/DH_PGC_M,
plus a bare configuration (`with_default_eef=False`). Their independent
control DOFs are 32 for dual BrainCo, 22 for dual DH grippers, and 20 without
end effectors. Mixed dexterous-hand/two-finger-gripper profiles are omitted
from the advertised catalog. The display condenses equivalent body
releases while retaining the hand pairing and per-profile DOF/mimic information.

Hand revisions are independent of body releases: all three registered hand
brands currently use hand revision `v021`, including with body `v022` or `v025`.
The compact labels DH_PGC and DH_PGC_M correspond to `DH_PGC_GRIPPER` and
`DH_PGC_GRIPPER_M`; the DH types can be paired independently through
`hand_types`. The default is body `v021` with two `BRAINCO_HAND` hands.

Control DOF counts unique independent joints in the default control parts,
excluding mimic followers. Overlapping groups such as `dual_arm` and
`full_body` count once. A part annotated `eef (1+1m)` has one independent
joint and one mimic. The Mimic Joints column names all default URDF followers,
including passive followers outside the control groups, such as the W1 hand
PIP joints. Thus Control DOF describes the configured control surface rather
than all movable joints in an asset.

Listing reads only catalog metadata and does not construct robot configurations
or download assets. Missing metadata is shown as `-`; `None` means a known
absence of mimics. Variant IDs such as `ur5` remain accepted by the shared
resolver even though their table rows are grouped under URRobot.

An installed provider can advertise its own catalog:

```toml
[project.entry-points."embodichain.robots"]
acme = "acme_robots.catalog:ROBOTS"
```

```python
# acme_robots/catalog.py: keep this module and the package initializer lightweight.
ROBOTS = {
    "acme.arm": {
        "class_path": "acme_robots.arm:ArmCfg",
        "defaults": {"variant": "standard"},
        "dof": 6,
        "control_parts": {"arm": 6},
        "mimic_joints": {},
        "control_part_mimics": {},
        "supports_dual_arm": True,
    },
}
```

The selected class must subclass `RobotCfg` and implement its normal
`from_dict()` contract. Caller fields override catalog defaults. Robot IDs must
be unique across installed catalogs; collisions raise an error. Use a vendor
prefix for third-party IDs. Catalogs are cached for the process lifetime, so
restart after installing or changing a provider's registration metadata.

The same ID is accepted by all three consumers:

```yaml
robot:
  class_type: acme.arm
```

```python
from embodichain.lab.sim.robots.dual_arm import DualArmRobotCfg

cfg = DualArmRobotCfg.from_dict({"base_robot": "acme.arm"})
```

Dual-arm composition requires the existing single-arm URDF/control-part/solver
contract. Per-use overrides remain supported through
`base_robot: {type: acme.arm, init: {...}}`.
Only bases explicitly declaring `supports_dual_arm` are advertised in the CLI's
DualArmRobot variants; the runtime still validates the selected configuration.

## Imports and installation

Import concrete robot configurations from `embodichain_robots`. Generic dual-arm
helpers remain in `embodichain.lab.sim.robots.dual_arm`. The former concrete
modules and exports under `embodichain.lab.sim.robots` have been removed.

Robot selectors use exact catalog IDs such as `FrankaPanda`, `CobotMagic`,
`URRobot`, and `ur5`. Old aliases such as `franka_panda`, `cobotmagic`, and
`ur`, as well as unregistered `Cfg`-suffixed spellings, are no longer accepted.

Install the main distribution normally:

```bash
pip install embodichain
# From a development checkout, refresh its bundled package/entry-point metadata:
pip install -e .
```
