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

## Shared robot resolution

Gym's `robot.class_type`, `analyze-workspace --robot`, and the dual-arm
`base_robot` selector use `embodichain.robot_registry`:

```python
from embodichain.robot_registry import list_robot_cfgs, resolve_robot_cfg

cfg = resolve_robot_cfg("URRobot", {"robot_type": "ur5"})
cfg = resolve_robot_cfg("ur5")  # compatibility alias with variant defaults
```

The official lightweight catalog is `_catalog.py`. It contains import strings
and optional default fields; listing robots does not import their configuration
classes. Configuration classes and top-level exports load on first use.

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

## Compatibility and installation

Existing YAML selectors, CLI aliases, and imports under
`embodichain.lab.sim.robots` remain supported. Legacy leaf modules alias their
canonical modules, preserving class/enum identity and module-level monkeypatches.
Use `embodichain_robots` for new concrete-robot imports; generic dual-arm helpers
remain in the core.

Install the main distribution normally:

```bash
pip install embodichain
# From a development checkout, refresh its bundled package/entry-point metadata:
pip install -e .
```
