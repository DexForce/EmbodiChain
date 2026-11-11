# TianjiMarvinGripper

## Standalone hand configuration

`TianjiMarvinGripperCfg` is a `ParallelJawGripperCfg` registered as `TianjiMarvinGripper`.
It exposes the native finger joints in `eef`, including source mimic followers,
and declares no arm IK solver or serial FK chain. Simulation drive values are
presets rather than manufacturer actuator specifications.

```python
from embodichain_robots import TianjiMarvinGripperCfg

cfg = TianjiMarvinGripperCfg.from_dict({'variant': 'left'})
```

The same model-owned pure data supplies mounted or existing embodiment bindings.
See [parallel-jaw binding and TCP resolution](../parallel_jaw_grippers.md) for
command ownership, static metadata access, source verification and migration.
