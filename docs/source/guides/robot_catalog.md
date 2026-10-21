(robot-catalog)=
# Robot Catalog

Use the robot catalog to discover installed configurations before choosing a
robot for a Gym task, workspace analysis, or dual-arm composition. The main
`embodichain` distribution includes the official `embodichain_robots` package.

```bash
embodichain list-robot
python -m embodichain list-robot
embodichain list-robot --help
```

Listing reads catalog metadata without importing robot implementations,
starting simulation, or downloading assets. See {ref}`cli-list-robots` for the
table columns.

## Read the Control Summary

`Control DOF` counts unique independent joints in the default control parts.
The count excludes mimic followers and counts joints shared by groups such as
`dual_arm` and `full_body` once. It describes the configured control groups;
other movable joints in the asset can be outside those groups.

`Control Parts` uses independent-plus-mimic notation. For example,
`eef (1+1m)` describes a two-joint gripper with one independent joint and one
follower. Official presets use `eef` for a single arm and `left_eef` /
`right_eef` for a dual arm. Custom control-part names remain valid.

`Mimic Joints` lists follower names from the default URDF, including passive
followers outside the control parts. W1's BrainCo PIP joints are an example:
they appear in this column while each hand's six selected control joints are
independent. The displayed follower count therefore need not equal the sum
of the `m` counts in the control parts.

A `*` marks the default variant. `-` means that a provider has not supplied a
summary, and `None` in the mimic column means a known absence of followers.
When a provider omits per-part mimic membership, part counts are marked
`total` instead of being presented as independent counts.

The official catalog currently groups configurations into these families:

| Robot ID | Independent control DOF | Variant coverage |
|---|---|---|
| `AlohaMini` | 15 | Complete Aloha Mini 2 Pro default control groups |
| `CobotMagic` | 14 | Default dual six-joint arms and two independent gripper commands |
| `DexforceW1` | 20, 22, or 32 | Three body releases and six hand profiles per release |
| `DualArmRobot` | 12 or 16 | Six UR bases or a pair of Franka Panda arms with grippers |
| `FrankaPanda` | 8 | Seven arm joints and one independent gripper command |
| `TianjiMarvin` | 16 | Default dual seven-joint arms with grippers |
| `URRobot` | 6 | `ur3`, `ur3e`, `ur5`, `ur5e`, `ur10`, and `ur10e` |

These summaries describe catalog defaults. Configuration overrides can change
the control groups or attached end effectors. See {doc}`/resources/robot/index`
for the individual robot configurations and their additional options.

## Select a Variant

Use the `Robot ID` to resolve a configuration and supply its variant fields as
overrides. UR model IDs are also registered individually, even though their
display rows are grouped under `URRobot`:

```python
from embodichain.robot_registry import list_robot_cfgs, resolve_robot_cfg

robot_ids = list_robot_cfgs()
ur5_cfg = resolve_robot_cfg("URRobot", {"robot_type": "ur5"})
ur5_preset_cfg = resolve_robot_cfg("ur5")
```

`list_robot_cfgs()` returns all registered IDs, including the six UR presets.
`resolve_robot_cfg()` imports and constructs the selected configuration, so
asset resolution and robot dependencies can be required at that point.
The same registered IDs are accepted by Gym's `robot.class_type`, workspace
analysis's `--robot`, and dual-arm composition's `base_robot`.

### Dual-arm Bases

The default base is `ur10`. Every listed UR base contributes six independent
arm joints per side, giving 12 control DOFs. A `FrankaPanda` base contributes
seven arm joints and one independent gripper command per side, giving 16
control DOFs plus two mimic followers. The overlapping `dual_arm` group is
counted once.

```python
panda_pair_cfg = resolve_robot_cfg(
    "DualArmRobot", {"base_robot": "FrankaPanda"}
)
```

See {doc}`/resources/robot/dual_arm` for mounts and the base-robot contract.

### W1 Body and Hand Profiles

W1 advertises body releases `v021`, `v022`, and `v025`. Each release has six
hand profiles, for 18 combinations. Equivalent body releases share a display
row while retaining their version labels and hand-specific DOFs.

| Hand profile | Left hand | Right hand | Control DOF |
|---|---|---|---|
| `BrainCo` | `BRAINCO_HAND` | `BRAINCO_HAND` | 32 |
| `DH_PGC` | `DH_PGC_GRIPPER` | `DH_PGC_GRIPPER` | 22 |
| `DH_PGC_M` | `DH_PGC_GRIPPER_M` | `DH_PGC_GRIPPER_M` | 22 |
| `L=DH_PGC,R=DH_PGC_M` | `DH_PGC_GRIPPER` | `DH_PGC_GRIPPER_M` | 22 |
| `L=DH_PGC_M,R=DH_PGC` | `DH_PGC_GRIPPER_M` | `DH_PGC_GRIPPER` | 22 |
| `bare` | No default end effector | No default end effector | 20 |

The default is `v021/BrainCo`. Dexterous-hand/two-finger-gripper mixtures are
omitted from the advertised profiles. Body and hand release versions are
independent: all registered hand brands currently use hand revision `v021`,
including with body `v022` or `v025`.

Choose a displayed profile through `version`, `hand_types`, and
`with_default_eef`; labels such as `v025/DH_PGC` describe a profile rather than
an additional registered robot ID:

```python
w1_gripper_cfg = resolve_robot_cfg(
    "DexforceW1",
    {
        "version": "v025",
        "hand_types": {
            "left": "DH_PGC_GRIPPER",
            "right": "DH_PGC_GRIPPER",
        },
        "hand_versions": {"left": "v021", "right": "v021"},
    },
)
w1_bare_cfg = resolve_robot_cfg(
    "DexforceW1", {"version": "v025", "with_default_eef": False}
)
```

See {doc}`/resources/robot/dexforce_w1` for hardware releases and calibration.

## Register a Third-party Catalog

An installed package exposes a lightweight mapping through an
`embodichain.robots` entry point:

```toml
[project.entry-points."embodichain.robots"]
acme = "acme_robots.catalog:ROBOTS"
```

```python
# acme_robots/catalog.py
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

Keep the catalog module and package initializer free of robot implementation
imports. `class_path` is required, and the selected class must implement the
`RobotCfg.from_dict()` contract. Defaults and display metadata are optional.
In catalog metadata, `control_parts` maps names to raw joint counts;
`RobotCfg.control_parts` itself maps names to joint patterns.
`mimic_joints` maps followers to leaders, and `control_part_mimics` lists each
part's followers. Set `supports_dual_arm=True` only for a compatible single-arm
base that satisfies the assembly contract.

Robot IDs must be unique across installed catalogs. Use a vendor prefix for
third-party IDs; duplicate IDs produce a CLI error. Discovery is cached for
the process lifetime, so refresh the package's installation metadata and
restart the process after changing an entry point. See {doc}`add_robot` for
the configuration implementation checklist.

## Import and Selector Migration

Import concrete presets from `embodichain_robots`. Generic dual-arm APIs remain
in the core:

```python
from embodichain_robots import FrankaPandaCfg, URRobotCfg
from embodichain.lab.sim.robots import DualArmRobotCfg, build_dual_arm_cfg
from embodichain.lab.sim.robots.dual_arm import resolve_mounts
```

Former concrete modules and exports under `embodichain.lab.sim.robots` have
been removed. Select robots by exact registered IDs such as `FrankaPanda`,
`CobotMagic`, `URRobot`, and `ur5`. Historical aliases such as `franka`,
`franka_panda`, `cobotmagic`, and `ur`, and unregistered `Cfg`-suffixed IDs are
no longer accepted. This does not change custom robot UIDs or asset joint names.

See {doc}`/api_reference/robot_catalogs` for the resolver API and canonical
configuration imports.
