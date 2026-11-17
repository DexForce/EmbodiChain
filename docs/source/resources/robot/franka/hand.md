# FrankaPandaHand

## Standalone hand configuration

`FrankaPandaHandCfg` is a `ParallelJawGripperCfg` registered as `FrankaPandaHand`.
It exposes the native finger joints in `eef`, including source mimic followers,
and declares no arm IK solver or serial FK chain. Simulation drive values are
presets rather than manufacturer actuator specifications.

```python
from embodichain_robots import FrankaPandaHandCfg

cfg = FrankaPandaHandCfg.from_dict({})
```

The same model-owned pure data supplies mounted or existing embodiment bindings.
See [parallel-jaw binding and TCP resolution](../parallel_jaw_grippers.md) for
command ownership, static metadata access, source verification and migration.
